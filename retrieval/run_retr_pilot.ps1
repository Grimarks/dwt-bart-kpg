# English summary: train + decode + score ONE retrieval condition and seed (g5, r5, st5, g5t, st5t). Hyper-parameters: facebook/bart-base, lr 6e-5 polynomial, warmup 100,
# batch 6 x grad-accum 16, 15 epochs, fp16; greedy decoding (--num_beams 1). Expects the folder layout in README.md
# (scripts in <root>/PC1_PC2_FINAL_PACKAGE/scripts/kpg_training, this file in <root>/output/retrieval_pilot).
#
# run_retr_pilot.ps1 - retrieval-keyphrase pilot, ONE cell (one condition, one seed).
# Same model, scripts and hyper-parameters as run_kp20k.ps1; only the data folder differs:
#   g5 = keyphrases of the 5 nearest KP20k-train documents (MiniLM, near-duplicates removed)
#   r5 = keyphrases of 5 random KP20k-train documents (control, same input length)
# Input: "<title> <kp ; kp ; ...> . <eos> <abstract>", tgt identical to ori; scored with evaluate_nopeos.py + ORI src.
#
#   PC3:  .\run_retr_pilot.ps1 -Cond g5
#   PC4:  .\run_retr_pilot.ps1 -Cond r5 -SkipEval         (PC4 cannot score; hypotheses are scored on PC3)
#   Test first (about 5-8 min, writes only to a smoke_ folder):  .\run_retr_pilot.ps1 -Cond g5 -Smoke
# Safe to run again after a power cut: continues from the last checkpoint (STATUS file decides what is done).
param(
  [Parameter(Mandatory = $true)][ValidateSet("g5", "r5", "st5", "g5t", "st5t")][string]$Cond,
  [int]$Seed = 1234,
  [switch]$SkipEval,
  [switch]$Smoke,
  [switch]$SkipGpuCheck
)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_DISABLED       = "true"
$env:PYTHONIOENCODING     = "utf-8"
$env:PYTHONUTF8           = "1"
$env:USE_TF               = "0"
$PILOT = $PSScriptRoot                      # ...\output\retrieval_pilot
$OUT   = Split-Path $PILOT -Parent          # ...\output
$ROOT  = Split-Path $OUT -Parent            # <research root>
$PKG   = Join-Path $ROOT "PC1_PC2_FINAL_PACKAGE"
$KP    = Join-Path $PKG "dataset\kp20k_v3"
$SCR   = Join-Path $PKG "scripts\kpg_training"
$WRAP  = Join-Path $OUT "resume_wrapper.py"
$utf8  = New-Object System.Text.UTF8Encoding($false)
$status = Join-Path $OUT ("STATUS_retr_pilot_{0}.txt" -f $Cond)
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $status }
function Has($pattern) { (Test-Path $status) -and (Select-String -Path $status -Pattern $pattern -Quiet) }

$name     = "kp20k_retr_$Cond"
$tag0     = "$name seed$Seed"
$testSets = @("kp20k", "inspec", "krapivin400", "nus", "semeval")
$KrapIds  = Join-Path $KP "ids\krapivin400_test_ids.txt"
$K397Excl = @("1007763", "1009027", "1024484")
$K397Ref  = Join-Path $OUT "kp20k_krapivin397\krapivin397_test_ori_v2.json"
function RetrFile($d, $split) { Join-Path $KP ("retr_{0}\{1}_{2}_retr_{0}_v2.json" -f $Cond, $d, $split) }
function OriTest($d)          { Join-Path $KP ("ori\{0}_test_ori_v2.json" -f $d) }

# ---- the exact training arguments of run_kp20k.ps1 (only files, output dir and length differ)
function TrainArgs($runDir, [string[]]$extra) {
  $a = @("--output_dir", $runDir, "--model_name_or_path", "facebook/bart-base", "--do_train", "--do_eval",
    "--train_file", (RetrFile "kp20k" "train"), "--validation_file", (RetrFile "kp20k" "valid"),
    "--src_column", "src", "--tgt_column", "tgt", "--per_device_train_batch_size", "6", "--per_device_eval_batch_size", "12",
    "--gradient_accumulation_steps", "16", "--learning_rate", "6e-5", "--lr_scheduler_type", "polynomial", "--warmup_steps", "100",
    "--overwrite_output_dir", "--predict_with_generate", "--seed", "$Seed", "--fp16", "true", "--half_precision_backend", "auto")
  return $a + $extra
}
$FullRun = @("--num_train_epochs", "15", "--logging_strategy", "steps", "--logging_steps", "500",
  "--eval_strategy", "steps", "--eval_steps", "5000", "--save_strategy", "steps", "--save_steps", "1000", "--save_total_limit", "2")

function LastCkpt($dir) {
  Get-ChildItem $dir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
}

# ---- preflight
$need = @((Join-Path $SCR "run_finetune_kpgen_seq2seq_hf.py"), (Join-Path $SCR "run_decode_kpgen_seq2seq_hf.py"), $WRAP)
if (-not $SkipEval) { $need += (Join-Path $OUT "evaluate_nopeos.py") }
$need += @((RetrFile "kp20k" "train"), (RetrFile "kp20k" "valid"))
foreach ($d in $testSets) { $need += (RetrFile $d "test"); $need += (OriTest $d) }
foreach ($f in $need) { if (-not (Test-Path $f)) { Write-Host "ABORT: file not found: $f"; exit 1 } }
if (-not $SkipGpuCheck) {
  $py = Get-Process python -ErrorAction SilentlyContinue
  if ($py) { Write-Host "ABORT: a python process is already running (another job?). Nothing was started."; exit 1 }
  $mem = $null
  try { $mem = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()) } catch { }
  if ($null -ne $mem -and $mem -gt 3000) { Write-Host "ABORT: GPU already uses $mem MiB. Nothing was started."; exit 1 }
}
Set-Location $SCR

# ================= smoke test: train 20 steps, RESUME through resume_wrapper.py to 30, decode, eval =================
if ($Smoke) {
  $S = Join-Path $PILOT ("smoke_{0}_{1}" -f $Cond, (Get-Date -Format "HHmmss"))
  New-Item -ItemType Directory -Force -Path $S | Out-Null
  $small = @("--max_train_samples", "3000", "--max_eval_samples", "48", "--logging_strategy", "steps", "--logging_steps", "5",
    "--eval_strategy", "steps", "--eval_steps", "10", "--save_strategy", "steps", "--save_steps", "10", "--save_total_limit", "2")
  Write-Host "SMOKE 1/4: train 20 steps ..."
  python run_finetune_kpgen_seq2seq_hf.py @(TrainArgs $S ($small + @("--max_steps", "20"))) *> "$S\console.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in training (exit $LASTEXITCODE):"; Get-Content "$S\console.log" -Tail 25; exit 1 }
  $ck = LastCkpt $S
  if (-not $ck -or $ck.Name -ne "checkpoint-20") { Write-Host "SMOKE FAILED: expected checkpoint-20, found $($ck.Name)"; exit 1 }

  Write-Host "SMOKE 2/4: resume from checkpoint-20 through resume_wrapper.py to step 30 ..."
  python "$WRAP" run_finetune_kpgen_seq2seq_hf.py @(TrainArgs $S ($small + @("--max_steps", "30", "--resume_from_checkpoint", $ck.FullName))) *> "$S\console_resume.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in resume (exit $LASTEXITCODE):"; Get-Content "$S\console_resume.log" -Tail 25; exit 1 }
  $ck = LastCkpt $S
  $gs = $null
  if ($ck -and (Test-Path "$($ck.FullName)\trainer_state.json")) { $gs = (Get-Content "$($ck.FullName)\trainer_state.json" -Raw | ConvertFrom-Json).global_step }
  if ($gs -ne 30) { Write-Host "SMOKE FAILED: after resume global_step = $gs (expected 30), checkpoint $($ck.Name)"; exit 1 }
  Write-Host "   resume OK: $($ck.Name), global_step 30"

  Write-Host "SMOKE 3/4: decode inspec (500 docs) ..."
  $hyp = Join-Path $S "smoke_inspec_hypotheses.txt"
  python run_decode_kpgen_seq2seq_hf.py --model_name_or_path $ck.FullName --tokenizer_name $ck.FullName --test_file (RetrFile "inspec" "test") --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir $S --output_file_name $hyp *> "$S\decode.console.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in decode (exit $LASTEXITCODE):"; Get-Content "$S\decode.console.log" -Tail 25; exit 1 }
  $nh = [System.IO.File]::ReadAllLines($hyp).Length
  $nr = [System.IO.File]::ReadAllLines((RetrFile "inspec" "test")).Length
  if ($nh -ne $nr) { Write-Host "SMOKE PROBLEM: hypotheses $nh lines vs test $nr lines"; exit 1 }
  Write-Host "   decode OK: $nh lines"

  if (-not $SkipEval) {
    Write-Host "SMOKE 4/4: evaluate with ORI src ..."
    python "$OUT\evaluate_nopeos.py" --src_file (OriTest "inspec") --pred_file "$hyp" --file_prefix unused --tgt_dir "$S" --log_file "smoke_inspec" --k_list 5 10 15 M *> "$S\eval.console.log"
    if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in eval (exit $LASTEXITCODE):"; Get-Content "$S\eval.console.log" -Tail 25; exit 1 }
    Write-Host "   eval OK"
  } else { Write-Host "SMOKE 4/4: eval skipped (-SkipEval)" }
  Select-String -Path "$S\console.log", "$S\console_resume.log" -Pattern "'loss'|train_loss|eval_loss" | Select-Object -Last 4 | ForEach-Object { $_.Line.Trim() }
  Write-Host "SMOKE OK (the smoke folder can be deleted)"
  exit 0
}

# ================= real run =================
if (Has ("DONE cell  " + $tag0 + "$")) { Write-Host "$tag0 is already DONE (see $status)"; exit 0 }
Log "PILOT START $tag0 skip_eval=$SkipEval"
$RUN = $null
$m = Select-String -Path $status -Pattern ("STARTED train  " + $tag0 + " -> (.+)$") -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($m) { $RUN = $m.Matches[0].Groups[1].Value.Trim(); if (-not (Test-Path $RUN)) { $RUN = $null } }

if (-not (Has ("DONE train  " + $tag0 + "( |$)"))) {
  $resume = $null
  if (-not $RUN) {
    $RUN = Join-Path $OUT ("{0}_{1}_pilot_bart-base_lr6e-5_polynomial_seed{2}_pc{3}" -f (Get-Date -Format "yyyyMMdd-HHmm"), $name, $Seed, $(if ($SkipEval) { "4" } else { "3" }))
    New-Item -ItemType Directory -Force -Path $RUN | Out-Null
    Log "STARTED train  $tag0 -> $RUN"
  } else {
    $ck = LastCkpt $RUN
    if ($ck) { $resume = $ck.FullName; Log "RESUME train  $tag0 from $resume" } else { Log "RESTART train  $tag0 (no checkpoint yet) in $RUN" }
  }
  $t0 = Get-Date
  if ($resume) {
    python "$WRAP" run_finetune_kpgen_seq2seq_hf.py @(TrainArgs $RUN ($FullRun + @("--resume_from_checkpoint", $resume))) *>> "$RUN\console_resume.log"
  } else {
    python run_finetune_kpgen_seq2seq_hf.py @(TrainArgs $RUN $FullRun) *>> "$RUN\console.log"
  }
  if ($LASTEXITCODE -ne 0) { Log "FAILED train  $tag0 (exit $LASTEXITCODE) - STOPPING"; exit 1 }
  $wall = ((Get-Date) - $t0).TotalSeconds
  $rt = $null
  if (Test-Path "$RUN\train_results.json") { $rt = (Get-Content "$RUN\train_results.json" -Raw | ConvertFrom-Json).train_runtime }
  $gh = if ($rt) { [math]::Round($rt / 3600, 3) } else { [math]::Round($wall / 3600, 3) }
  $json = "{`n  ""cell"": ""$tag0"",`n  ""pilot"": ""retrieval_keyphrases"",`n  ""train_runtime_s"": $(if ($rt) { $rt } else { 'null' }),`n  ""train_wallclock_s"": $([math]::Round($wall,1)),`n  ""train_gpu_hours"": $gh`n}`n"
  [System.IO.File]::WriteAllText("$RUN\gpu_hours.json", $json, $utf8)
  Log "DONE train  $tag0 (gpu_hours=$gh)"
}
if (-not $RUN) { Log "ABORT: DONE train found but run folder missing"; exit 1 }

$ckptDir = LastCkpt $RUN
if (-not $ckptDir) { Log "MISSING checkpoint in $RUN - skip decode  $tag0"; exit 1 }
$ckpt = $ckptDir.FullName
Log "DECODE uses $ckpt"

$allOk = $true
foreach ($d in $testSets) {
  if ((Has ("DONE decode\+eval  " + $tag0 + " " + $d + "$")) -or (Has ("DONE decode  " + $tag0 + " " + $d + "$"))) { continue }
  $test = RetrFile $d "test"
  $hyp = Join-Path $RUN ("{0}_seed{1}_{2}_hypotheses.txt" -f $name, $Seed, $d)
  Log "STARTED decode  $tag0 $d"
  $t0 = Get-Date
  python run_decode_kpgen_seq2seq_hf.py --model_name_or_path "$ckpt" --tokenizer_name "$ckpt" --test_file "$test" --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir "$RUN" --output_file_name "$hyp" *> "$RUN\decode_$d.console.log"
  if ($LASTEXITCODE -ne 0) { Log "FAILED decode  $tag0 $d - continuing"; $allOk = $false; continue }
  $dsec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
  [System.IO.File]::AppendAllText("$RUN\decode_times.tsv", "$d`t$dsec`n", $utf8)
  $nh = [System.IO.File]::ReadAllLines($hyp).Length
  $nr = [System.IO.File]::ReadAllLines($test).Length
  if ($nh -ne $nr) { Log "FAILED decode  $tag0 $d - line count $nh vs $nr"; $allOk = $false; continue }
  if ($SkipEval) { Log "DONE decode  $tag0 $d"; continue }
  python "$OUT\evaluate_nopeos.py" --src_file (OriTest $d) --pred_file "$hyp" --file_prefix unused --tgt_dir "$RUN" --log_file ("{0}_seed{1}_{2}" -f $name, $Seed, $d) --k_list 5 10 15 M *> "$RUN\eval_$d.console.log"
  if ($LASTEXITCODE -ne 0) { Log "FAILED eval  $tag0 $d - continuing"; $allOk = $false; continue }
  Log "DONE decode+eval  $tag0 $d"
}

# Krapivin 397: the krapivin400 hypotheses without the 3 ids of the BERTopic corpus (same as run_kp20k.ps1)
if ((-not (Has ("DONE eval  " + $tag0 + " krapivin397$"))) -and (-not (Has ("DONE subset  " + $tag0 + " krapivin397$")))) {
  $hyp400 = Join-Path $RUN ("{0}_seed{1}_krapivin400_hypotheses.txt" -f $name, $Seed)
  $hyp397 = Join-Path $RUN ("{0}_seed{1}_krapivin397_hypotheses.txt" -f $name, $Seed)
  if ((Test-Path $hyp400) -and (Test-Path "$OUT\kp20k_checks.py")) {
    python "$OUT\kp20k_checks.py" subset --ids "$KrapIds" --exclude $K397Excl --in "$hyp400" --out "$hyp397" *> "$RUN\subset_krapivin397.console.log"
    if ($LASTEXITCODE -ne 0) { Log "FAILED subset  $tag0 krapivin397 - continuing"; $allOk = $false }
    elseif ($SkipEval -or -not (Test-Path $K397Ref)) { Log "DONE subset  $tag0 krapivin397" }
    else {
      python "$OUT\evaluate_nopeos.py" --src_file "$K397Ref" --pred_file "$hyp397" --file_prefix unused --tgt_dir "$RUN" --log_file ("{0}_seed{1}_krapivin397" -f $name, $Seed) --k_list 5 10 15 M *> "$RUN\eval_krapivin397.console.log"
      if ($LASTEXITCODE -ne 0) { Log "FAILED eval  $tag0 krapivin397 - continuing"; $allOk = $false } else { Log "DONE eval  $tag0 krapivin397" }
    }
  } else { Log "SKIP krapivin397 (krapivin400 hypotheses or kp20k_checks.py missing)"; $allOk = $false }
}

# % truncated inputs (the added keyphrases make inputs longer than ori), CPU only, once
$tr = Join-Path $PILOT ("trunc_retr_{0}.json" -f $Cond)
if ((-not (Test-Path $tr)) -and (Test-Path "$OUT\kp20k_checks.py")) {
  $targs = @("--file", "train", (RetrFile "kp20k" "train"), "--file", "valid", (RetrFile "kp20k" "valid"))
  foreach ($d in $testSets) { $targs += @("--file", "test_$d", (RetrFile $d "test")) }
  python "$OUT\kp20k_checks.py" trunc @targs --tokenizer "$ckpt" --out "$tr" *> (Join-Path $PILOT ("trunc_retr_{0}.log" -f $Cond))
  if ($LASTEXITCODE -ne 0) { Log "WARNING trunc stats failed" } else { Log "DONE trunc stats -> $tr" }
}

if ($allOk) { Log "DONE cell  $tag0" } else { Log "CELL INCOMPLETE  $tag0 (see FAILED lines; run this script again to retry the failed parts)" }
Write-Host "finished. See $status"
