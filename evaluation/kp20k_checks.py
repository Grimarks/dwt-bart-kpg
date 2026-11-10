r"""Consistency checks (aligned targets, truncation, unique targets) and the Krapivin-397 subset builder.

Original notes:
CPU-only checks used by run_kp20k.ps1.

Data layout (kpg_conditions_topicv3_v2, see kp20k_v3\README.md):
  {root}\ori\{ds}_test_ori_v2.json, {root}\topic\{ds}_test_topic_before_topicv3_v2.json,
  {root}\shuffled\{ds}_test_shuffled_before_topicv3_v2.json, ids in {root}\ids\{ds}_test_ids.txt

  align   : every condition has the same #docs and identical `tgt`, line by line, as ori
            (hypotheses from condition X are scored against the ori test file, so lines must align)
  trunc   : % of inputs longer than max_source_length BART tokens (same call as training:
            tokenizer(src, truncation=True) -> truncated iff untruncated length > max_len)
  targets : #unique targets (all/present/absent) in results_log_* must be identical across
            conditions for the same seed and dataset
  subset  : drop the lines whose id (ids file, same order) is in --exclude; works on a .json test
            file and on a hypotheses .txt alike. Fails unless #lines == #ids and every excluded id is found.
Exit code 1 on any mismatch.
"""
import argparse
import glob
import json
import os
import re
import sys

KP20K_DATASETS = ["kp20k", "inspec", "krapivin400", "nus", "semeval"]


def suffix(cond):
    return "ori_v2" if cond == "ori" else f"{cond}_before_topicv3_v2"


def read_jsonl(path, field):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)[field] for l in f if l.strip()]


def cmd_align(a):
    ok = True
    for ds in a.datasets:
        ref = read_jsonl(a.pattern.format(root=a.root, cond="ori", ds=ds, sfx=suffix("ori")), "tgt")
        for cond in a.conditions:
            if cond == "ori":
                continue
            other = read_jsonl(a.pattern.format(root=a.root, cond=cond, ds=ds, sfx=suffix(cond)), "tgt")
            same = len(other) == len(ref) and all(x == y for x, y in zip(ref, other))
            print(f"align {ds:11s} {cond:9s} n={len(other)} (ori n={len(ref)}) identical_tgt={same}")
            ok &= same
    return ok


def cmd_trunc(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    res = {}
    for name, path in a.files:
        srcs = read_jsonl(path, "src")
        n_trunc, total_tok = 0, 0
        for i in range(0, len(srcs), 2000):
            lens = [len(x) for x in tok(srcs[i:i + 2000], truncation=False)["input_ids"]]
            n_trunc += sum(l > a.max_len for l in lens)
            total_tok += sum(lens)
        res[name] = {"file": path, "n": len(srcs), "n_truncated": n_trunc,
                     "pct_truncated": 100.0 * n_trunc / max(1, len(srcs)),
                     "mean_tokens": total_tok / max(1, len(srcs))}
        print(f"trunc {name}: {n_trunc}/{len(srcs)} = {res[name]['pct_truncated']:.2f}% > {a.max_len} tokens")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return True


def parse_unique_targets(path):
    out, tag = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^=+(all|present|absent)=+", line)
            if m:
                tag = m.group(1)
            m = re.match(r"^#unique targets: (\d+)", line)
            if m and tag:
                out[tag] = int(m.group(1))
    return out


def cmd_targets(a):
    ok = True
    for ds in a.datasets:
        found = {}
        for p in sorted(glob.glob(os.path.join(a.out_root, "*_kp20k_*_v3_*", f"results_log_kp20k_*_seed{a.seed}_{ds}.txt"))):
            cond = re.search(r"results_log_kp20k_(\w+?)_seed", os.path.basename(p)).group(1)
            found[cond] = parse_unique_targets(p)  # folders sort by timestamp: the newest run wins
        if len(found) < 2:
            print(f"targets {ds} seed{a.seed}: {found} (fewer than 2 conditions, nothing to compare)")
            continue
        same = len({tuple(sorted(v.items())) for v in found.values()}) == 1
        print(f"targets {ds} seed{a.seed}: identical={same} {found}")
        ok &= same
    return ok


def cmd_subset(a):
    with open(a.ids, encoding="utf-8") as f:
        ids = [l.strip() for l in f if l.strip()]
    with open(a.inp, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    excl = set(a.exclude)
    missing = excl - set(ids)
    if len(lines) != len(ids) or missing:
        print(f"subset ABORT {a.inp}: lines={len(lines)} ids={len(ids)} excluded_ids_not_found={sorted(missing)}")
        return False
    kept = [l for i, l in zip(ids, lines) if i not in excl]
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("".join(l + "\n" for l in kept))
    os.replace(tmp, a.out)
    print(f"subset {a.inp} -> {a.out}: {len(lines)} -> {len(kept)} (dropped {sorted(excl)})")
    return len(kept) == len(lines) - len(excl)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("align")
    p.add_argument("--datasets", nargs="+", default=KP20K_DATASETS)
    p.add_argument("--root", required=True)
    p.add_argument("--pattern", default=os.path.join("{root}", "{cond}", "{ds}_test_{sfx}.json"),
                   help="placeholders {root} {cond} {ds} {sfx}; sfx = ori_v2 | <cond>_before_topicv3_v2")
    p.add_argument("--conditions", nargs="+", default=["ori", "topic", "shuffled"])
    p = sub.add_parser("targets")
    p.add_argument("--datasets", nargs="+", default=KP20K_DATASETS + ["krapivin397"])
    p.add_argument("--out_root", required=True)
    p.add_argument("--seed", required=True)
    p = sub.add_parser("trunc")
    p.add_argument("--file", dest="files", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    p.add_argument("--tokenizer", default="facebook/bart-base")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--out", required=True)
    p = sub.add_parser("subset")
    p.add_argument("--ids", required=True)
    p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    ok = {"align": cmd_align, "trunc": cmd_trunc, "targets": cmd_targets, "subset": cmd_subset}[a.cmd](a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
