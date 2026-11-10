"""Reproduce the human-evaluation means of the paper from human_evaluation_ratings_anonymised.csv.

1,792 ratings = 112 articles x 2 evaluators x 2 models (BART, DWT-BART) x 4 aspects, scores 1-5.
Evaluators appear only as codes F<form>-E<1|2>. The model column is already decoded (the A/B key
itself is not published). Usage:  python analyse_ratings.py [path/to/ratings.csv]
"""
import csv
import os
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "human_evaluation_ratings_anonymised.csv")
with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
articles = {(r["form"], r["article"]) for r in rows}
print(f"{len(rows)} ratings, {len(articles)} articles, {len({r['evaluator'] for r in rows})} evaluators")

ASPECTS = ["Faithfulness", "Duplication", "Semantic similarity", "Usefulness"]
MODELS = ["BART", "DWT-BART"]
by = defaultdict(list)
for r in rows:
    by[(r["aspect"], r["model"])].append(int(r["score"]))
print(f"{'aspect':22s}" + "".join(f"{m:>12s}" for m in MODELS) + "   n per model")
for a in ASPECTS:
    print(f"{a:22s}" + "".join(f"{sum(by[(a, m)]) / len(by[(a, m)]):12.2f}" for m in MODELS)
          + f"   {len(by[(a, MODELS[0])])}")

by_ds = defaultdict(list)
for r in rows:
    by_ds[(r["dataset"], r["aspect"], r["model"])].append(int(r["score"]))
print("\nper dataset (BART / DWT-BART):")
for ds in sorted({r["dataset"] for r in rows}):
    cells = []
    for a in ASPECTS:
        b, d = by_ds[(ds, a, "BART")], by_ds[(ds, a, "DWT-BART")]
        cells.append(f"{a[:12]} {sum(b) / len(b):.2f}/{sum(d) / len(d):.2f}")
    print(f"  {ds:12s} " + " | ".join(cells))
