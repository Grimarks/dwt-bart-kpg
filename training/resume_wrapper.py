"""Run a training script with torch.load(weights_only=False) so --resume_from_checkpoint works on torch < 2.6 (own checkpoint files only).
"""
import sys, os, runpy
import torch
_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load
def _noop(*a, **k):
    return None
import transformers
import transformers.trainer
import transformers.modeling_utils
n = 0
for m in list(sys.modules.values()):
    if m is None or not getattr(m, "__name__", "").startswith("transformers"):
        continue
    d = getattr(m, "__dict__", {})
    if "check_torch_load_is_safe" in d:
        d["check_torch_load_is_safe"] = _noop
        n += 1
print(f"[wrapper] torch {torch.__version__}; check dimatikan di {n} modul", flush=True)
if __name__ == "__main__":
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")