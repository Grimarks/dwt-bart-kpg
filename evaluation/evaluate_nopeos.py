"""Scoring used in the paper: DeepKPG evaluate.py (unchanged) with the <peos> separator removed; present/absent decided on the ori source.

Original notes:
Wrapper around evaluate.py (unchanged) that removes the present/absent separator <peos>
from targets and predictions before scoring.

<peos> is a separator emitted by the data/model ("present kps <peos> absent kps"), not a keyphrase.
evaluate.py splits on ';' and therefore scored it as an (absent) keyphrase in every document.
Rule: after the same ';' split evaluate.py does, every item containing <peos> is split on it and
only its non-empty pieces are kept (a pure "<peos>" item disappears; a rare glued item such as
"covid-<peos>" keeps "covid-"). Items without <peos> are untouched.

Same CLI as evaluate.py (--src_file/--pred_file branch):
  python evaluate_nopeos.py --src_file ORI_TEST.json --pred_file HYP.txt --file_prefix P --tgt_dir DIR
                            --log_file NAME --k_list 5 10 15 M
"""
import argparse
import importlib.util
import json
import os
import sys

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
DEFAULT_EVALUATE_PY = os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE", "scripts", "kpg_training", "evaluate.py")
PEOS = "<peos>"


def load_evaluate(path=DEFAULT_EVALUATE_PY):
    # private module name so it cannot collide with the HF `evaluate` package
    spec = importlib.util.spec_from_file_location("kpg_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_eval_inputs(src_file, pred_file, kp_sep=";"):
    """Identical to evaluate.py __main__ (--src_file branch)."""
    hypotheses, references, sources = [], [], []
    with open(src_file, encoding="utf-8") as f1, open(pred_file, encoding="utf-8") as f2:
        for source, pred in zip(f1, f2):
            ex = json.loads(source.strip())
            sources.append(ex["src"].lower())
            references.append(ex["tgt"].lower().split(" {} ".format(kp_sep)))
            preds = pred.split(" {} ".format(kp_sep))
            preds = [p.replace("[ digit ]", "[digit]") for p in preds]
            hypotheses.append(preds)
    return hypotheses, references, sources


def strip_peos(items, stats=None, key=None):
    """items: list as evaluate.py builds it (joined with ';' and re-split in evaluate.main)."""
    flat = ";".join(items).strip().split(";")
    out = []
    for x in flat:
        if PEOS in x:
            pieces = [p for p in x.split(PEOS) if p.strip()]
            if stats is not None:
                stats[key + "_peos_items_removed"] += 1
                if pieces:
                    stats[key + "_glued_items_kept_piece"] += 1
            out.extend(pieces)
        else:
            out.append(x)
    return out


def strip_all(hypotheses, references):
    stats = {"pred_peos_items_removed": 0, "pred_glued_items_kept_piece": 0,
             "ref_peos_items_removed": 0, "ref_glued_items_kept_piece": 0}
    hyps = [strip_peos(h, stats, "pred") for h in hypotheses]
    refs = [strip_peos(r, stats, "ref") for r in references]
    stats["pred_peos_remaining"] = sum(PEOS in x for h in hyps for x in h)
    stats["ref_peos_remaining"] = sum(PEOS in x for r in refs for x in r)
    stats["docs_with_empty_reference"] = sum(1 for r in refs if not any(x.strip() for x in r))
    return hyps, refs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_file", required=True, help="ALWAYS the ori test file (present/absent split)")
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--file_prefix", required=True)
    ap.add_argument("--tgt_dir", required=True)
    ap.add_argument("--log_file", required=True)
    ap.add_argument("--k_list", nargs="+", default=[5, "M"])
    ap.add_argument("--evaluate_py", default=DEFAULT_EVALUATE_PY)
    args = ap.parse_args()

    E = load_evaluate(args.evaluate_py)
    E.args = args  # evaluate.main() reads the module-global `args.file_prefix`
    hypotheses, references, sources = read_eval_inputs(args.src_file, args.pred_file, E.KP_SEP)
    hyps, refs, stats = strip_all(hypotheses, references)
    print("PEOS_STRIP " + json.dumps(stats), flush=True)
    if stats["pred_peos_remaining"] or stats["ref_peos_remaining"]:
        sys.exit("ABORT: <peos> still present after stripping")
    E.run_eval((hyps, refs, sources), args.tgt_dir, args.log_file, args.k_list)


if __name__ == "__main__":
    main()
