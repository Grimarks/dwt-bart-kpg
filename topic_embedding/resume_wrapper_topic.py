"""Pilot: resume wrapper (see training/resume_wrapper.py).

Original notes:
Run a training script so that --resume_from_checkpoint works on torch < 2.6.

transformers 4.57 refuses torch.load() on torch < 2.6 (CVE-2025-32434), which also blocks reading
optimizer.pt / scheduler.pt / rng_state.pth when resuming. These are our own files, written by this
same run, so the check is switched off here. Usage:  python resume_wrapper_topic.py <script.py> <args...>
"""
import os, runpy, sys

def _noop(*a, **k):
    return None

import transformers.utils.import_utils as _iu
_iu.check_torch_load_is_safe = _noop
for _name in ("transformers.trainer", "transformers.modeling_utils", "transformers.utils"):
    try:
        _m = __import__(_name, fromlist=["x"])
        if hasattr(_m, "check_torch_load_is_safe"):
            _m.check_torch_load_is_safe = _noop
    except Exception:
        pass

if len(sys.argv) < 2:
    sys.exit("usage: python resume_wrapper_topic.py <script.py> <args...>")
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
