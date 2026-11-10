# English summary: build the st5t (DWT-BART) input files on the CPU with build_title_ctx_v2.py.
#
param([string]$Root = $env:DWTBART_ROOT)
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
Set-Location "$Root\output\retrieval_pilot"
python build_title_ctx_v2.py --fmt kpt --tag st5t --mode same_topic --base st5 *> "$Root\output\retrieval_pilot\build_st5t.log"
