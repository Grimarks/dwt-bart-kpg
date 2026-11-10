"""Write the KP20k training targets (tgt), one per line, to the file given as the first argument.
"""
import glob, json, sys
import os
# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
src = glob.glob(os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE", "dataset", "kp20k_v3", "ori", "kp20k_train_*.json"))
if len(src) != 1:
    sys.exit("expected 1 train file, got %s" % src)
n = 0
with open(src[0], encoding="utf-8-sig") as f, open(sys.argv[1], "w", encoding="utf-8", newline="\n") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        g.write(json.loads(line)["tgt"].replace("\n", " ") + "\n")
        n += 1
print(src[0], "rows", n)
