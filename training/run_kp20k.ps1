# English summary: KP20k chain for BART (ori), topic-label (topic) and shuffled-topic (shuffled), seeds 1234/42/2024:
# train (run_finetune_kpgen_seq2seq_hf.py), greedy decode on KP20k/Inspec/Krapivin/NUS/SemEval, score with
# evaluate_nopeos.py against the ori test file, Krapivin-397 subset, paired bootstrap. Usage: .\run_kp20k.ps1 -Root <root>
#
# KP20k chain (IJIES revision): 3 seeds x {ori, topic, shuffled}.
# Data: kpg_conditions_topicv3_v2 (zip md5 ae3a01c9fb23b6e769cbd84022203deb) in $KP, see $KP\README.md.
# Preflight (CPU only) runs first, then the chain starts.
# Resumable: cells with "DONE cell" in STATUS_kp20k.txt are skipped; a finished train is reused and only
# missing decodes/evals are redone. Stops on train FAILED.
# Krapivin is scored twice from the same hypotheses: krapivin400 (main) and krapivin397 (ids 1007763, 1009027,
# 1024484 removed: they are in the BERTopic corpus).
# Launch (do NOT run while testing):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','<root>\output\run_kp20k.ps1' -RedirectStandardOutput <root>\output\kp20k_launcher.log -RedirectStandardError <root>\output\kp20k_launcher.err
param([string]$Root = $env:DWTBART_ROOT)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_DISABLED       = "true"
$env:PYTHONIOENCODING     = "utf-8"
$env:PYTHONUTF8           = "1"
$env:USE_TF               = "0"
$OUT = "$Root\output"
$PKG = "$Root\PC1_PC2_FINAL_PACKAGE"
$DS  = "$PKG\dataset"
$KP  = "$DS\kp20k_v3"
$status     = "$OUT\STATUS_kp20k.txt"
$seedStatus = "$OUT\STATUS_seeds.txt"
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $status }
function Has($pattern) { (Test-Path $status) -and (Select-String -Path $status -Pattern $pattern -Quiet) }

# ---- data layout (kp20k_v3\README.md):
#   ori\kp20k_train_ori_v2.json, topic\kp20k_train_topic_before_topicv3_v2.json, shuffled\..._shuffled_before_topicv3_v2.json
#   tgt = "present;...;<peos>;absent;..." (identical in all conditions) -> scored with evaluate_nopeos.py + ori src
$conds    = @("ori", "topic", "shuffled")
$seeds    = @(1234, 42, 2024)
$testSets = @("kp20k", "inspec", "krapivin400", "nus", "semeval")
function Sfx($c)          { if ($c -eq "ori") { "ori_v2" } else { "${c}_before_topicv3_v2" } }
function TrainFile($c)    { "$KP\$c\kp20k_train_$(Sfx $c).json" }
function ValidFile($c)    { "$KP\$c\kp20k_valid_$(Sfx $c).json" }
function TestFile($c, $d) { "$KP\$c\${d}_test_$(Sfx $c).json" }
$KrapIds  = "$KP\ids\krapivin400_test_ids.txt"
$K397Excl = @("1007763", "1009027", "1024484")
$K397Ref  = "$OUT\kp20k_krapivin397\krapivin397_test_ori_v2.json"

# ---- preflight (CPU only, before waiting so a data problem shows up now, not after the seed chain)
foreach ($c in $conds) {
  foreach ($f in @((TrainFile $c), (ValidFile $c))) { if (-not (Test-Path $f)) { Log "ABORT: file tidak ada: $f"; exit 1 } }
  foreach ($d in $testSets) { $f = TestFile $c $d; if (-not (Test-Path $f)) { Log "ABORT: file tidak ada: $f"; exit 1 } }
}
python "$OUT\kp20k_checks.py" align --root "$KP" --pattern "{root}\{cond}\{ds}_test_{sfx}.json" --datasets $testSets *> "$OUT\kp20k_align_check.log"
if ($LASTEXITCODE -ne 0) { Log "ABORT: test sets not aligned with ori (see kp20k_align_check.log)"; exit 1 }
Log "CHECK align OK (tgt identical to ori for all test sets)"
New-Item -ItemType Directory -Force -Path (Split-Path $K397Ref) | Out-Null
python "$OUT\kp20k_checks.py" subset --ids "$KrapIds" --exclude $K397Excl --in (TestFile "ori" "krapivin400") --out "$K397Ref" *> "$OUT\kp20k_krapivin397\subset_ref.log"
if ($LASTEXITCODE -ne 0) { Log "ABORT: krapivin397 reference not built (see kp20k_krapivin397\subset_ref.log)"; exit 1 }
Log "CHECK krapivin397 reference OK -> $K397Ref"


Set-Location "$PKG\scripts\kpg_training"
Log "KP20K CHAIN START seeds=$($seeds -join ',') conds=$($conds -join ',')"

foreach ($seed in $seeds) {
  foreach ($cond in $conds) {
    $name = "kp20k_$cond"
    $tag0 = "$name seed$seed"
    if (Has ("DONE cell  " + $tag0 + "$")) { continue }

    # -- train (reuse a finished run on resume)
    $RUN = $null
    if (Has ("DONE train  " + $tag0 + "$")) {
      $m = Select-String -Path $status -Pattern ("STARTED train  " + $tag0 + " -> (.+)$") | Select-Object -Last 1
      if ($m) { $RUN = $m.Matches[0].Groups[1].Value.Trim() }
      if ($RUN -and (Test-Path $RUN)) { Log "RESUME  $tag0 using $RUN" } else { $RUN = $null }
    }
    if (-not $RUN) {
      $train = TrainFile $cond; $valid = ValidFile $cond
      $RUN = Join-Path $OUT ("{0}_{1}_v3_bart-base_lr6e-5_polynomial_seed{2}_pc3" -f (Get-Date -Format "yyyyMMdd-HHmm"), $name, $seed)
      New-Item -ItemType Directory -Force -Path $RUN | Out-Null
      Log "STARTED train  $tag0 -> $RUN"
      $t0 = Get-Date
      python run_finetune_kpgen_seq2seq_hf.py --output_dir "$RUN" --model_name_or_path facebook/bart-base --do_train --do_eval --train_file "$train" --validation_file "$valid" --src_column src --tgt_column tgt --per_device_train_batch_size 6 --per_device_eval_batch_size 12 --gradient_accumulation_steps 16 --num_train_epochs 15 --learning_rate 6e-5 --lr_scheduler_type polynomial --warmup_steps 100 --logging_strategy steps --logging_steps 500 --eval_strategy steps --eval_steps 5000 --save_strategy steps --save_steps 1000 --save_total_limit 2 --overwrite_output_dir --predict_with_generate --seed $seed --fp16 true --half_precision_backend auto *> "$RUN\console.log"
      if ($LASTEXITCODE -ne 0) { Log "FAILED train  $tag0 (exit $LASTEXITCODE) - STOPPING"; exit 1 }
      $wall = ((Get-Date) - $t0).TotalSeconds
      $rt = $null
      if (Test-Path "$RUN\train_results.json") { $rt = (Get-Content "$RUN\train_results.json" -Raw | ConvertFrom-Json).train_runtime }
      $gh = if ($rt) { [math]::Round($rt / 3600, 3) } else { [math]::Round($wall / 3600, 3) }
      $json = "{`n  ""cell"": ""$tag0"",`n  ""gpu"": ""1x RTX 4070 12GB (PC3)"",`n  ""train_runtime_s"": $(if ($rt) { $rt } else { 'null' }),`n  ""train_wallclock_s"": $([math]::Round($wall,1)),`n  ""train_gpu_hours"": $gh`n}`n"
      [System.IO.File]::WriteAllText("$RUN\gpu_hours.json", $json, $utf8)
      Log "DONE train  $tag0 (gpu_hours=$gh)"
    }

    $ckptDir = Get-ChildItem $RUN -Directory -Filter "checkpoint-*" | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
    if (-not $ckptDir) { Log "MISSING checkpoint in $RUN - skip decode  $tag0"; continue }
    $ckpt = $ckptDir.FullName

    # -- greedy decode on 5 test sets (src of the same condition); evaluate with the ORI src for every condition,
    #    through evaluate_nopeos.py (evaluate.py unchanged, separator <peos> removed from targets and predictions)
    $allOk = $true
    foreach ($d in $testSets) {
      if (Has ("DONE decode\+eval  " + $tag0 + " " + $d + "$")) { continue }
      $test = TestFile $cond $d
      $testOri = TestFile "ori" $d
      $hyp = Join-Path $RUN ("{0}_seed{1}_{2}_hypotheses.txt" -f $name, $seed, $d)
      Log "STARTED decode  $tag0 $d"
      $t0 = Get-Date
      python run_decode_kpgen_seq2seq_hf.py --model_name_or_path "$ckpt" --tokenizer_name "$ckpt" --test_file "$test" --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir "$RUN" --output_file_name "$hyp" *> "$RUN\decode_$d.console.log"
      if ($LASTEXITCODE -ne 0) { Log "FAILED decode  $tag0 $d - continuing"; $allOk = $false; continue }
      $dsec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
      [System.IO.File]::AppendAllText("$RUN\decode_times.tsv", "$d`t$dsec`n", $utf8)
      python "$OUT\evaluate_nopeos.py" --src_file "$testOri" --pred_file "$hyp" --file_prefix unused --tgt_dir "$RUN" --log_file ("{0}_seed{1}_{2}" -f $name, $seed, $d) --k_list 5 10 15 M *> "$RUN\eval_$d.console.log"
      if ($LASTEXITCODE -ne 0) { Log "FAILED eval  $tag0 $d - continuing"; $allOk = $false; continue }
      Log "DONE decode+eval  $tag0 $d"
    }

    # -- Krapivin 397: krapivin400 hypotheses without the 3 BERTopic-corpus ids, scored against the 397 ori reference
    if (-not (Has ("DONE eval  " + $tag0 + " krapivin397$"))) {
      if (Has ("DONE decode\+eval  " + $tag0 + " krapivin400$")) {
        $hyp400 = Join-Path $RUN ("{0}_seed{1}_krapivin400_hypotheses.txt" -f $name, $seed)
        $hyp397 = Join-Path $RUN ("{0}_seed{1}_krapivin397_hypotheses.txt" -f $name, $seed)
        python "$OUT\kp20k_checks.py" subset --ids "$KrapIds" --exclude $K397Excl --in "$hyp400" --out "$hyp397" *> "$RUN\subset_krapivin397.console.log"
        if ($LASTEXITCODE -ne 0) { Log "FAILED subset  $tag0 krapivin397 - continuing"; $allOk = $false }
        else {
          python "$OUT\evaluate_nopeos.py" --src_file "$K397Ref" --pred_file "$hyp397" --file_prefix unused --tgt_dir "$RUN" --log_file ("{0}_seed{1}_krapivin397" -f $name, $seed) --k_list 5 10 15 M *> "$RUN\eval_krapivin397.console.log"
          if ($LASTEXITCODE -ne 0) { Log "FAILED eval  $tag0 krapivin397 - continuing"; $allOk = $false } else { Log "DONE eval  $tag0 krapivin397" }
        }
      } else { $allOk = $false }
    }

    python "$OUT\kp20k_checks.py" targets --out_root "$OUT" --seed $seed *> "$RUN\targets_check.log"
    if ($LASTEXITCODE -ne 0) { Log "WARNING CHECK targets MISMATCH seed$seed (see $RUN\targets_check.log)" } else { Log "CHECK targets OK seed$seed after $name" }

    # -- % truncated inputs, once per condition (CPU only)
    $tr = "$OUT\kp20k_trunc_$cond.json"
    if (-not (Test-Path $tr)) {
      $targs = @("--file", "train", (TrainFile $cond), "--file", "valid", (ValidFile $cond))
      foreach ($d in $testSets) { $targs += @("--file", "test_$d", (TestFile $cond $d)) }
      python "$OUT\kp20k_checks.py" trunc @targs --tokenizer "$ckpt" --out "$tr" *> "$OUT\kp20k_trunc_$cond.log"
      if ($LASTEXITCODE -ne 0) { Log "WARNING trunc stats failed $cond" } else { Log "DONE trunc stats $cond" }
    }

    if ($allOk) { Log "DONE cell  $tag0" }
  }

  # -- paired bootstrap on everything finished so far (CPU only); krapivin400 and krapivin397 reported separately
  python "$OUT\paired_bootstrap.py" --discover_kp20k "$OUT" --kp20k_root "$KP" --test_pattern "{root}\{cond}\{ds}_test_ori_v2.json" --datasets kp20k inspec krapivin400 krapivin397 nus semeval --ref "krapivin397=$K397Ref" --out_dir "$OUT\kp20k_stats" --prefix kp20k *> "$OUT\kp20k_stats_seed$seed.log"
  if ($LASTEXITCODE -ne 0) { Log "WARNING bootstrap failed after seed$seed" } else { Log "DONE bootstrap after seed$seed -> $OUT\kp20k_stats" }
}
Log "ALL KP20K COMPLETE"

