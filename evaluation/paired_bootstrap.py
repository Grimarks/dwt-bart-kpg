"""Paired bootstrap (10,000 resamples) of macro P/R/F1 @5/@M, per seed and pooled over seeds; also writes mean +- sd tables. Spec files for the paper: evaluation/specs/.

Original notes:
Paired bootstrap significance test for keyphrase generation runs.

Per-document P/R/F1 (@5, @M; present/absent/all) are computed with the functions of
evaluate.py, reproducing its main() loop exactly (same parsing, stemming, filtering,
present/absent split).  The corpus-level statistic is the one evaluate.py reports as
"Macro": P = mean_doc(P), R = mean_doc(R), F1 = 2PR/(P+R).  Every bootstrap resample
recomputes that statistic, so the point estimate equals the number in results_log.

References and the source used for present/absent classification ALWAYS come from the
reference file of the dataset (the ori test file), for every condition.  The separator <peos>
is removed from targets and predictions with evaluate_nopeos.strip_peos (--keep_peos = old behaviour).

Usage
  KP20k (auto-discover run folders written by run_kp20k.ps1):
    python paired_bootstrap.py --discover_kp20k <root>/output \\
        --kp20k_root <root>/PC1_PC2_FINAL_PACKAGE/dataset/kp20k_v3 \\
        --out_dir <root>/output/kp20k_stats --prefix kp20k
  Any runs (explicit spec):
    python paired_bootstrap.py --spec spec.json --pairs st5t:ori --out_dir DIR --prefix retr
    spec.json = {"references": {"<dataset>": "<ori test json>"},
                 "systems": [{"condition": "ori", "seed": 1234, "dataset": "<dataset>",
                              "hyp": "<hypotheses.txt>"}, ...]}

Outputs in --out_dir
  <prefix>_results_summary.csv   paired bootstrap: diff, 95% CI, p-value (per seed + seed mean)
  <prefix>_per_seed_scores.csv   point estimates per condition/seed/dataset
  <prefix>_mean_std.csv / .md    mean +- std across seeds per condition
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_nopeos as NP  # noqa: E402  (<peos> removal shared with run_kp20k.ps1 evaluation)

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
DEFAULT_EVALUATE_PY = os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE", "scripts", "kpg_training", "evaluate.py")
TAGS = ["present", "absent", "all"]
METRICS = ["P", "R", "F1"]
KP20K_DATASETS = ["kp20k", "inspec", "krapivin", "nus", "semeval"]
KP20K_CONDITIONS = ["ori", "topic", "shuffled"]
KP20K_DIR_RE = re.compile(r"^\d{8}-\d{4}_kp20k_(ori|topic|shuffled)_v3_.*_seed(\d+)_pc[34]$")


def load_evaluate(path):
    # evaluate.py is loaded under a private name so it cannot collide with the HF `evaluate` package
    spec = importlib.util.spec_from_file_location("kpg_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_references(path, keep_peos):
    """Same parsing as evaluate.py __main__ (--src_file branch), <peos> stripped via evaluate_nopeos."""
    sources, references = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            sources.append(ex["src"].lower())
            references.append(ex["tgt"].lower().split(" ; "))
    if not keep_peos:
        references = [NP.strip_peos(r) for r in references]
    return sources, references


def read_hypotheses(path, keep_peos):
    hyps = []
    with open(path, encoding="utf-8") as f:
        for pred in f:
            preds = pred.split(" ; ")
            hyps.append([p.replace("[ digit ]", "[digit]") for p in preds])
    if not keep_peos:
        hyps = [NP.strip_peos(h) for h in hyps]
    return hyps


class DocScorer:
    """Per-document scores, replicating evaluate.main() line by line."""

    def __init__(self, E, k_list):
        self.E = E
        self.k_list = E.process_input_ks(k_list)
        self._stem_cache = {}

    def _stem_words(self, words):
        # identical to E.stem_word_list, memoised (PorterStemmer is deterministic)
        cache = self._stem_cache
        out = []
        for w in words:
            s = cache.get(w)
            if s is None:
                s = self.E.stemmer.stem(w.strip().lower())
                cache[w] = s
            out.append(s)
        return out

    def prepare_sources(self, sources, references):
        E = self.E
        prepared = []
        for src_l, ref in zip(sources, references):
            trg_l = ";".join(ref)
            src_l = src_l.strip()
            if E.TITLE_SEP in src_l:
                [title, context] = src_l.strip().split(E.TITLE_SEP)
            else:
                title, context = "", src_l
            src_token_list = title.strip().split(" ") + context.strip().split(" ")
            stemmed_src = self._stem_words(src_token_list)
            trg_token_2dlist = [t.strip().split(" ") for t in trg_l.strip().split(";")]
            stemmed_trg = [self._stem_words(t) for t in trg_token_2dlist]
            unique_trg, _ = E.find_unique_target(stemmed_trg)
            present_trg, absent_trg, _ = E.separate_present_absent_by_source(stemmed_src, unique_trg, False)
            prepared.append((stemmed_src, {"all": unique_trg, "present": present_trg, "absent": absent_trg}))
        return prepared

    def score(self, prepared, hyps):
        """Returns dict (tag, k, 'P'|'R') -> np.array[n_docs] and target counts per tag."""
        E = self.E
        n = len(prepared)
        out = {(t, k, m): np.zeros(n) for t in TAGS for k in self.k_list for m in ("P", "R")}
        n_trg = {t: np.zeros(n, dtype=np.int64) for t in TAGS}
        for i, ((stemmed_src, trgs), hyp) in enumerate(zip(prepared, hyps)):
            pred_l = ";".join(hyp)
            pred_str_list = pred_l.strip().split(";")[:200]
            pred_token_2dlist = [p.strip().split(" ") for p in pred_str_list]
            stemmed_pred = [self._stem_words(p) for p in pred_token_2dlist]
            filtered_pred, _, _ = E.filter_prediction(E.INVALIDATE_UNK, E.DISABLE_EXTRA_ONE_WORD_FILTER, stemmed_pred)
            present_pred, absent_pred, _ = E.separate_present_absent_by_source(stemmed_src, filtered_pred, False)
            preds = {"all": filtered_pred, "present": present_pred, "absent": absent_pred}
            for t in TAGS:
                trg, pred = trgs[t], preds[t]
                is_match = E.compute_match_result(trg, pred, type="exact", dimension=1)
                p_ks, r_ks, _, _, _ = E.compute_classification_metrics_at_ks(
                    is_match, len(pred), len(trg), k_list=self.k_list, meng_rui_precision=False)
                for k, p, r in zip(self.k_list, p_ks, r_ks):
                    out[(t, k, "P")][i] = p
                    out[(t, k, "R")][i] = r
                n_trg[t][i] = len(trg)
        return out, n_trg


def f1(p, r):
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def file_sig(path):
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def discover_kp20k(out_root, kp20k_root, test_pattern, datasets):
    systems = []
    best = {}
    for d in sorted(os.listdir(out_root)):
        m = KP20K_DIR_RE.match(d)
        if not m:
            continue
        cond, seed = m.group(1), int(m.group(2))
        for ds in datasets:
            hyp = os.path.join(out_root, d, f"kp20k_{cond}_seed{seed}_{ds}_hypotheses.txt")
            if os.path.exists(hyp):
                best[(cond, seed, ds)] = hyp  # later (newer) folder wins: names sort by timestamp
    for (cond, seed, ds), hyp in sorted(best.items()):
        systems.append({"condition": cond, "seed": seed, "dataset": ds, "hyp": hyp})
    refs = {ds: test_pattern.format(root=kp20k_root, cond="ori", ds=ds) for ds in datasets}
    return {"references": refs, "systems": systems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="JSON spec with references and systems")
    ap.add_argument("--discover_kp20k", metavar="OUT_ROOT", help="scan run folders written by run_kp20k.ps1")
    ap.add_argument("--kp20k_root", default=os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE", "dataset", "kp20k_v3"))
    ap.add_argument("--test_pattern", default=os.path.join("{root}", "{cond}", "test_{ds}.json"))
    ap.add_argument("--datasets", nargs="+", default=KP20K_DATASETS)
    ap.add_argument("--ref", nargs="+", default=[], metavar="DS=PATH",
                    help="--discover_kp20k only: reference file for DS instead of --test_pattern (e.g. krapivin397)")
    ap.add_argument("--pairs", nargs="+", default=["topic:ori", "shuffled:ori", "topic:shuffled"],
                    help="A:B pairs; diff = A - B")
    ap.add_argument("--k_list", nargs="+", default=["5", "M"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--rng_seed", type=int, default=12345)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--evaluate_py", default=DEFAULT_EVALUATE_PY)
    ap.add_argument("--keep_peos", action="store_true",
                    help="score <peos> as a keyphrase (old, buggy behaviour; only to reproduce old numbers)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="kp20k")
    args = ap.parse_args()

    if bool(args.spec) == bool(args.discover_kp20k):
        ap.error("give exactly one of --spec / --discover_kp20k")
    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = discover_kp20k(args.discover_kp20k, args.kp20k_root, args.test_pattern, args.datasets)
        for r in args.ref:
            ds, path = r.split("=", 1)
            spec["references"][ds] = path
    if not spec["systems"]:
        sys.exit("no systems found")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "perdoc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    E = load_evaluate(args.evaluate_py)
    scorer = DocScorer(E, args.k_list)
    k_list = scorer.k_list
    pairs = [tuple(p.split(":")) for p in args.pairs]

    by_ds = defaultdict(list)
    for s in spec["systems"]:
        by_ds[s["dataset"]].append(s)

    summary_rows, seed_rows = [], []
    for ds in sorted(by_ds):
        ref_path = spec["references"][ds]
        t0 = time.time()
        sources, references = read_references(ref_path, args.keep_peos)
        n = len(sources)
        prepared = scorer.prepare_sources(sources, references)
        print(f"[{ds}] {n} docs, references from {ref_path} ({time.time() - t0:.0f}s)", flush=True)

        # ---- per-document scores per system (cached on disk)
        scores = {}   # (cond, seed) -> dict (tag,k,P/R) -> array
        trg_counts = {}
        for s in by_ds[ds]:
            key = (s["condition"], int(s["seed"]))
            sig = hashlib.md5((file_sig(s["hyp"]) + "|" + file_sig(ref_path) + "|" + ",".join(map(str, k_list))
                               + "|" + file_sig(args.evaluate_py) + "|" + file_sig(NP.__file__)
                               + f"|keep_peos={args.keep_peos}").encode()).hexdigest()[:16]
            cpath = os.path.join(cache_dir, f"{ds}_{key[0]}_seed{key[1]}_{sig}.npz")
            hyps = None
            if os.path.exists(cpath):
                z = np.load(cpath)
                sc = {(t, k, m): z[f"{t}|{k}|{m}"] for t in TAGS for k in k_list for m in ("P", "R")}
                nt = {t: z[f"ntrg|{t}"] for t in TAGS}
            else:
                hyps = read_hypotheses(s["hyp"], args.keep_peos)
                if len(hyps) != n:
                    sys.exit(f"ABORT {s['hyp']}: {len(hyps)} hypothesis lines vs {n} reference docs")
                t1 = time.time()
                sc, nt = scorer.score(prepared, hyps)
                np.savez_compressed(cpath, **{f"{t}|{k}|{m}": v for (t, k, m), v in sc.items()},
                                    **{f"ntrg|{t}": v for t, v in nt.items()})
                print(f"  scored {key[0]} seed{key[1]} ({time.time() - t1:.0f}s)", flush=True)
            scores[key] = sc
            trg_counts[key] = {t: int(v.sum()) for t, v in nt.items()}
            for t in TAGS:
                for k in k_list:
                    p, r = sc[(t, k, "P")].mean(), sc[(t, k, "R")].mean()
                    for m, v in (("P", p), ("R", r), ("F1", float(f1(p, r)))):
                        seed_rows.append({"dataset": ds, "condition": key[0], "seed": key[1], "tag": t,
                                          "k": k, "metric": m, "score": float(v), "n_docs": n,
                                          "unique_targets": trg_counts[key][t]})
        uniq = {tuple(sorted(v.items())) for v in trg_counts.values()}
        print(f"  #unique targets (present/absent/all) identical across systems: {len(uniq) == 1} "
              f"{trg_counts[next(iter(trg_counts))]}", flush=True)
        if len(uniq) != 1:
            sys.exit(f"ABORT {ds}: unique target counts differ across systems: {trg_counts}")

        # ---- bootstrap: one shared resample matrix per dataset => paired across systems, seeds, metrics
        keys = sorted(scores)
        cols = [(key, t, k, m) for key in keys for t in TAGS for k in k_list for m in ("P", "R")]
        col_idx = {c: i for i, c in enumerate(cols)}
        X = np.stack([scores[c[0]][c[1:]] for c in cols], axis=1)  # n x C
        rng = np.random.default_rng(args.rng_seed)
        means = np.empty((args.n_boot, X.shape[1]))
        pvec = np.full(n, 1.0 / n)
        t1 = time.time()
        for b0 in range(0, args.n_boot, args.chunk):
            b1 = min(args.n_boot, b0 + args.chunk)
            W = rng.multinomial(n, pvec, size=b1 - b0).astype(np.float64)  # resample counts per doc
            means[b0:b1] = (W @ X) / n
        full = X.mean(axis=0)
        print(f"  bootstrap {args.n_boot} x {n} docs x {len(keys)} systems ({time.time() - t1:.0f}s)", flush=True)

        def stat(cond, seeds, t, k, m, use_full):
            """Statistic per resample (or point estimate), averaged over the given seeds."""
            vals = []
            for sd in seeds:
                src = full if use_full else means
                p = src[..., col_idx[((cond, sd), t, k, "P")]]
                r = src[..., col_idx[((cond, sd), t, k, "R")]]
                vals.append(p if m == "P" else r if m == "R" else f1(p, r))
            return np.mean(vals, axis=0)

        seeds_of = defaultdict(set)
        for c, sd in keys:
            seeds_of[c].add(sd)
        for a, b in pairs:
            common = sorted(seeds_of[a] & seeds_of[b])
            if not common:
                print(f"  skip {a} vs {b}: no common seeds", flush=True)
                continue
            scopes = [(str(sd), [sd]) for sd in common]
            if len(common) > 1:
                scopes.append(("mean(" + ",".join(map(str, common)) + ")", common))
            for scope, seeds in scopes:
                for t in TAGS:
                    for k in k_list:
                        for m in METRICS:
                            sa, sb = float(stat(a, seeds, t, k, m, True)), float(stat(b, seeds, t, k, m, True))
                            d_obs = sa - sb
                            d = stat(a, seeds, t, k, m, False) - stat(b, seeds, t, k, m, False)
                            lo, hi = np.percentile(d, [2.5, 97.5])
                            # two-sided p: bootstrap distribution shifted to the null (Efron & Tibshirani 1993, 16.4)
                            p_val = (1 + np.sum(np.abs(d - d_obs) >= abs(d_obs) - 1e-12)) / (args.n_boot + 1)
                            summary_rows.append({
                                "dataset": ds, "seed": scope, "system_A": a, "system_B": b, "tag": t,
                                "metric": f"{m}@{k}", "score_A": sa, "score_B": sb, "diff_A_minus_B": d_obs,
                                "ci95_lo": float(lo), "ci95_hi": float(hi), "p_value": float(p_val),
                                "significant_0.05": bool(p_val < 0.05), "n_docs": n, "n_boot": args.n_boot})

    # ---- write outputs
    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)

    if summary_rows:
        write_csv(os.path.join(args.out_dir, f"{args.prefix}_results_summary.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, f"{args.prefix}_per_seed_scores.csv"), seed_rows)

    agg = defaultdict(list)
    for r in seed_rows:
        agg[(r["dataset"], r["condition"], r["tag"], r["k"], r["metric"])].append((r["seed"], r["score"]))
    ms_rows = []
    for (ds, cond, t, k, m), v in sorted(agg.items(), key=lambda kv: tuple(map(str, kv[0]))):
        vals = np.array([s for _, s in v])
        ms_rows.append({"dataset": ds, "condition": cond, "tag": t, "metric": f"{m}@{k}",
                        "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                        "n_seeds": len(vals), "seeds": ",".join(str(s) for s, _ in sorted(v))})
    write_csv(os.path.join(args.out_dir, f"{args.prefix}_mean_std.csv"), ms_rows)

    # markdown: one table per metric family, rows = dataset x condition, cols = tag x k (x100)
    ms = {(r["dataset"], r["condition"], r["tag"], r["metric"]): r for r in ms_rows}
    conds = [c for c in KP20K_CONDITIONS if any(r["condition"] == c for r in ms_rows)] + \
            sorted({r["condition"] for r in ms_rows} - set(KP20K_CONDITIONS))
    lines = [f"# {args.prefix}: mean ± std across seeds (×100, evaluate.py macro)", ""]
    for m in METRICS:
        hdr = [f"{t} {m}@{k}" for t in TAGS for k in k_list]
        lines += [f"## {m}", "", "| dataset | condition | n_seeds | " + " | ".join(hdr) + " |",
                  "|---|---|---|" + "---|" * len(hdr)]
        for ds in sorted({r["dataset"] for r in ms_rows}):
            for c in conds:
                cells, ns = [], ""
                for t in TAGS:
                    for k in k_list:
                        r = ms.get((ds, c, t, f"{m}@{k}"))
                        if r is None:
                            cells.append("–")
                            continue
                        ns = r["n_seeds"]
                        cells.append(f"{100 * r['mean']:.2f}" + ("" if np.isnan(r["std"]) else f" ± {100 * r['std']:.2f}"))
                if ns != "":
                    lines.append(f"| {ds} | {c} | {ns} | " + " | ".join(cells) + " |")
        lines.append("")
    md = os.path.join(args.out_dir, f"{args.prefix}_mean_std.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", md)


if __name__ == "__main__":
    main()
