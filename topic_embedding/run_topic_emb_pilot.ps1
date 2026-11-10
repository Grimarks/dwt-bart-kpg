# English summary: train + decode + score one topic-embedding pilot cell (topic_emb or shuffled_emb, seed 1234).
#
# run_topic_emb_pilot.ps1 - topic-embedding pilot, ONE cell (one condition, one seed). Same hyper-parameters as run_kp20k.ps1.
#   PC3:  .\run_topic_emb_pilot.ps1 -Cond topic_emb
#   PC4:  .\run_topic_emb_pilot.ps1 -Cond shuffled_emb -SkipEval      (PC4 cannot score: evaluate.py is missing there)
#   Test (about 5 min, writes only to a smoke_ folder):  .\run_topic_emb_pilot.ps1 -Cond topic_emb -Smoke
# Safe to run again after a power cut or a crash: it continues from the last checkpoint (STATUS file decides what is done).
param(
  [Parameter(Mandatory = $true)][ValidateSet("topic_emb", "shuffled_emb")][string]$Cond,
  [int]$Seed = 1234,
  [double]$EmbLR = 0.001,
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
$PILOT = $PSScriptRoot
$OUT   = Split-Path $PILOT -Parent
$ROOT  = Split-Path $OUT -Parent
$PKG   = Join-Path $ROOT "PC1_PC2_FINAL_PACKAGE"
$KP    = Join-Path $PKG "dataset\kp20k_v3"
$KPE   = Join-Path $PKG "dataset\kp20k_v3_emb"
$utf8  = New-Object System.Text.UTF8Encoding($false)
$status = Join-Path $OUT ("STATUS_topic_emb_pilot_{0}.txt" -f $Cond)
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $status }
function Has($pattern) { (Test-Path $status) -and (Select-String -Path $status -Pattern $pattern -Quiet) }

$name     = "kp20k_$Cond"
$tag0     = "$name seed$Seed"
$testSets = @("kp20k", "inspec", "krapivin400", "nus", "semeval")
$KrapIds  = Join-Path $KP "ids\krapivin400_test_ids.txt"
$K397Excl = @("1007763", "1009027", "1024484")
$K397Ref  = Join-Path $OUT "kp20k_krapivin397\krapivin397_test_ori_v2.json"
function EmbFile($split, $d) { Join-Path $KPE ("{0}\{1}_{2}_{0}.json" -f $Cond, $d, $split) }
function OriTest($d)         { Join-Path $KP ("ori\{0}_test_ori_v2.json" -f $d) }

# ---- preflight
$need = @("topic_bart.py", "run_finetune_topic_emb.py", "run_decode_topic_emb.py", "resume_wrapper_topic.py")
foreach ($f in $need) { if (-not (Test-Path (Join-Path $PILOT $f))) { Write-Host "ABORT: $f not found in $PILOT"; exit 1 } }
$nt = Join-Path $KPE "num_topics.txt"
if (-not (Test-Path $nt)) { Write-Host "ABORT: $nt not found - run make_topic_emb_data.py first"; exit 1 }
$numTopics = [int]((Get-Content $nt -Raw).Trim())
$files = @((EmbFile "train" "kp20k"), (EmbFile "valid" "kp20k"))
foreach ($d in $testSets) { $files += (EmbFile "test" $d); $files += (OriTest $d) }
foreach ($f in $files) { if (-not (Test-Path $f)) { Write-Host "ABORT: file not found: $f"; exit 1 } }
if (-not $SkipGpuCheck) {
  $mem = $null
  try { $mem = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()) } catch { }
  if ($null -ne $mem -and $mem -gt 3000) { Write-Host "ABORT: GPU already uses $mem MiB (another job is running?). Nothing was started. Use -SkipGpuCheck only if you are sure."; exit 1 }
}
$env:TOPIC_NUM_TOPICS = "$numTopics"
$env:TOPIC_EMB_LR     = ([string]$EmbLR)
$env:TOPIC_INIT_STD   = "0.0"
Set-Location $PILOT
$py = @("resume_wrapper_topic.py", "run_finetune_topic_emb.py")

# ================= smoke test =================
if ($Smoke) {
  $S = Join-Path $PILOT ("smoke_{0}_{1}" -f $Cond, (Get-Date -Format "HHmmss"))
  New-Item -ItemType Directory -Force -Path $S | Out-Null
  Write-Host "SMOKE: 30 training steps on real bart-base + real data, then decode inspec ..."
  $a = @("--output_dir", $S, "--model_name_or_path", "facebook/bart-base", "--do_train", "--do_eval",
    "--train_file", (EmbFile "train" "kp20k"), "--validation_file", (EmbFile "valid" "kp20k"),
    "--src_column", "src", "--tgt_column", "tgt", "--per_device_train_batch_size", "6", "--per_device_eval_batch_size", "12",
    "--gradient_accumulation_steps", "16", "--max_steps", "30", "--max_train_samples", "3000", "--max_eval_samples", "48",
    "--learning_rate", "6e-5", "--lr_scheduler_type", "polynomial", "--warmup_steps", "100",
    "--logging_strategy", "steps", "--logging_steps", "5", "--eval_strategy", "steps", "--eval_steps", "15",
    "--save_strategy", "steps", "--save_steps", "15", "--save_total_limit", "1", "--overwrite_output_dir",
    "--predict_with_generate", "--seed", "$Seed", "--fp16", "true", "--half_precision_backend", "auto")
  python @py @a *> "$S\console.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in training (exit $LASTEXITCODE). Last lines of $S\console.log:"; Get-Content "$S\console.log" -Tail 25; exit 1 }
  $ck = Get-ChildItem $S -Directory -Filter "checkpoint-*" | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
  if (-not $ck) { Write-Host "SMOKE FAILED: no checkpoint saved"; exit 1 }
  $hyp = Join-Path $S "smoke_inspec_hypotheses.txt"
  python run_decode_topic_emb.py --model_name_or_path $ck.FullName --tokenizer_name $ck.FullName --test_file (EmbFile "test" "inspec") --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir $S --output_file_name $hyp *> "$S\decode.console.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED in decode (exit $LASTEXITCODE). Last lines of $S\decode.console.log:"; Get-Content "$S\decode.console.log" -Tail 25; exit 1 }
  $nh = [System.IO.File]::ReadAllLines($hyp).Length
  $nr = [System.IO.File]::ReadAllLines((EmbFile "test" "inspec")).Length
  Write-Host ("SMOKE: hypotheses {0} lines, test file {1} lines" -f $nh, $nr)
  Select-String -Path "$S\console.log" -Pattern "train_runtime|train_loss|eval_loss" | Select-Object -First 6 | ForEach-Object { $_.Line.Trim() }
  if ($nh -eq $nr) { Write-Host "SMOKE OK (the smoke folder can be deleted)" } else { Write-Host "SMOKE PROBLEM: line counts differ"; exit 1 }
  exit 0
}

# ================= real run =================
if (Has ("DONE cell  " + $tag0 + "$")) { Write-Host "$tag0 is already DONE (see $status)"; exit 0 }
Log "PILOT START $tag0 num_topics=$numTopics emb_lr=$EmbLR skip_eval=$SkipEval"
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
    $ck = Get-ChildItem $RUN -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
    if ($ck) { $resume = $ck.FullName; Log "RESUME train  $tag0 from $resume" } else { Log "RESTART train  $tag0 (no checkpoint yet) in $RUN" }
  }
  $t0 = Get-Date
  $a = @("--output_dir", $RUN, "--model_name_or_path", "facebook/bart-base", "--do_train", "--do_eval",
    "--train_file", (EmbFile "train" "kp20k"), "--validation_file", (EmbFile "valid" "kp20k"),
    "--src_column", "src", "--tgt_column", "tgt", "--per_device_train_batch_size", "6", "--per_device_eval_batch_size", "12",
    "--gradient_accumulation_steps", "16", "--num_train_epochs", "15", "--learning_rate", "6e-5",
    "--lr_scheduler_type", "polynomial", "--warmup_steps", "100",
    "--logging_strategy", "steps", "--logging_steps", "500", "--eval_strategy", "steps", "--eval_steps", "5000",
    "--save_strategy", "steps", "--save_steps", "1000", "--save_total_limit", "2", "--overwrite_output_dir",
    "--predict_with_generate", "--seed", "$Seed", "--fp16", "true", "--half_precision_backend", "auto")
  if ($resume) { $a += @("--resume_from_checkpoint", $resume) }
  python @py @a *>> "$RUN\console.log"
  if ($LASTEXITCODE -ne 0) { Log "FAILED train  $tag0 (exit $LASTEXITCODE) - STOPPING"; exit 1 }
  $wall = ((Get-Date) - $t0).TotalSeconds
  $rt = $null
  if (Test-Path "$RUN\train_results.json") { $rt = (Get-Content "$RUN\train_results.json" -Raw | ConvertFrom-Json).train_runtime }
  $gh = if ($rt) { [math]::Round($rt / 3600, 3) } else { [math]::Round($wall / 3600, 3) }
  $json = "{`n  ""cell"": ""$tag0"",`n  ""pilot"": ""topic_embedding"",`n  ""emb_lr"": $EmbLR,`n  ""train_runtime_s"": $(if ($rt) { $rt } else { 'null' }),`n  ""train_wallclock_s"": $([math]::Round($wall,1)),`n  ""train_gpu_hours"": $gh`n}`n"
  [System.IO.File]::WriteAllText("$RUN\gpu_hours.json", $json, $utf8)
  Log "DONE train  $tag0 (gpu_hours=$gh)"
}
if (-not $RUN) { Log "ABORT: DONE train found but run folder missing"; exit 1 }

$ckptDir = Get-ChildItem $RUN -Directory -Filter "checkpoint-*" | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
if (-not $ckptDir) { Log "MISSING checkpoint in $RUN - skip decode  $tag0"; exit 1 }
$ckpt = $ckptDir.FullName
Log "DECODE uses $ckpt"

$allOk = $true
foreach ($d in $testSets) {
  if ((Has ("DONE decode\+eval  " + $tag0 + " " + $d + "$")) -or (Has ("DONE decode  " + $tag0 + " " + $d + "$"))) { continue }
  $test = EmbFile "test" $d
  $hyp = Join-Path $RUN ("{0}_seed{1}_{2}_hypotheses.txt" -f $name, $Seed, $d)
  Log "STARTED decode  $tag0 $d"
  $t0 = Get-Date
  python run_decode_topic_emb.py --model_name_or_path "$ckpt" --tokenizer_name "$ckpt" --test_file "$test" --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir "$RUN" --output_file_name "$hyp" *> "$RUN\decode_$d.console.log"
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
if ($allOk) { Log "DONE cell  $tag0" } else { Log "CELL INCOMPLETE  $tag0 (see FAILED lines; run this script again to retry the failed parts)" }
Write-Host "finished. See $status"
