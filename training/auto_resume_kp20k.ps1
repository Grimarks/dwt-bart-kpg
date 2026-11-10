# English summary: resume an interrupted run_kp20k.ps1 training from its last valid checkpoint-* folder.
#
param([string]$Root, [string]$StatusName, [string]$LauncherName, [int]$Delay = 0)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_DISABLED       = "true"
$env:PYTHONIOENCODING     = "utf-8"
$env:PYTHONUTF8           = "1"
$env:USE_TF               = "0"
$OUT = "$Root\output"
$PKG = "$Root\PC1_PC2_FINAL_PACKAGE"
$KP  = "$PKG\dataset\kp20k_v3"
$status = "$OUT\$StatusName"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $status }
function Sfx($c) { if ($c -eq "ori") { "ori_v2" } else { "${c}_before_topicv3_v2" } }
if ($Delay -gt 0) { Start-Sleep -Seconds $Delay }
Set-Location "$PKG\scripts\kpg_training"

if (Get-Process python -ErrorAction SilentlyContinue) { Log "AUTO-RESUME: python sudah jalan, tidak melakukan apa-apa"; exit 0 }

$lines = @(Get-Content $status)
$started = @($lines | Where-Object { $_ -match 'STARTED train  kp20k_\w+ seed\d+ -> .+$' })
if ($started.Count -gt 0) {
  $m = [regex]::Match($started[-1], 'STARTED train  (kp20k_(\w+) seed(\d+)) -> (.+)$')
  $tag0 = $m.Groups[1].Value; $cond = $m.Groups[2].Value; $seed = [int]$m.Groups[3].Value; $RUN = $m.Groups[4].Value.Trim()
  # baris "DONE train  <tag0>" harus berakhir persis di tag0 (itu yang dicari launcher)
  $doneExact = @($lines | Where-Object { $_ -match ('DONE train  ' + [regex]::Escape($tag0) + '$') })
  if ($doneExact.Count -eq 0) {
    if (-not (Test-Path $RUN)) { Log "ABORT auto-resume: folder run tidak ada: $RUN"; exit 1 }
    $final = "$RUN\checkpoint-80340"
    if (-not (Test-Path $final)) {
      $ck = $null
      $cands = Get-ChildItem $RUN -Directory -Filter "checkpoint-*" | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } -Descending
      foreach ($c in $cands) {
        $n = [int]($c.Name -replace 'checkpoint-', ''); $ok = $false
        try {
          $ts = Get-Content "$($c.FullName)\trainer_state.json" -Raw | ConvertFrom-Json
          $ok = ($ts.global_step -eq $n) -and ((Get-Item "$($c.FullName)\optimizer.pt").Length -gt 500MB) -and ((Get-Item "$($c.FullName)\model.safetensors").Length -gt 400MB) -and (Test-Path "$($c.FullName)\scheduler.pt") -and (Test-Path "$($c.FullName)\rng_state.pth")
        } catch { $ok = $false }
        if ($ok) { $ck = $c.FullName; break }
      }
      if (-not $ck) { Log "ABORT auto-resume: tidak ada checkpoint valid di $RUN"; exit 1 }
      if (-not (Select-String -Path run_finetune_kpgen_seq2seq_hf.py -Pattern "train\(resume_from_checkpoint" -Quiet)) { Log "ABORT auto-resume: skrip training tidak meneruskan resume_from_checkpoint"; exit 1 }
      $train = "$KP\$cond\kp20k_train_$(Sfx $cond).json"
      $valid = "$KP\$cond\kp20k_valid_$(Sfx $cond).json"
      Log "AUTO-RESUME train  $tag0 dari $(Split-Path $ck -Leaf)"
      python "$OUT\resume_wrapper.py" run_finetune_kpgen_seq2seq_hf.py --output_dir "$RUN" --model_name_or_path facebook/bart-base --do_train --do_eval --train_file "$train" --validation_file "$valid" --src_column src --tgt_column tgt --per_device_train_batch_size 6 --per_device_eval_batch_size 12 --gradient_accumulation_steps 16 --num_train_epochs 15 --learning_rate 6e-5 --lr_scheduler_type polynomial --warmup_steps 100 --logging_strategy steps --logging_steps 500 --eval_strategy steps --eval_steps 5000 --save_strategy steps --save_steps 1000 --save_total_limit 2 --predict_with_generate --seed $seed --fp16 true --half_precision_backend auto --resume_from_checkpoint "$ck" *>> "$RUN\console_resume.log"
      if ($LASTEXITCODE -ne 0) { Log "FAILED auto-resume train  $tag0 (exit $LASTEXITCODE) - STOPPING"; exit 1 }
      if (-not (Test-Path $final)) { Log "MISSING checkpoint-80340 setelah resume - STOPPING"; exit 1 }
    }
    Log "DONE train  $tag0"
  }
}
Log "AUTO-RESUME: lanjut ke $LauncherName"
& "$OUT\$LauncherName"
