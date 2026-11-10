# make_topic_emb_data.py - build the data for the topic-embedding model from kp20k_v3 (no files in kp20k_v3 are changed).
# For every split it writes JSON lines {"src": <ORI src, no topic words>, "tgt": <same tgt>, "topic_id": <int>}.
#   topic_emb/    : topic_id = topik_asli   (the real topic of the document)
#   shuffled_emb/ : topic_id = topik_acak   (the random other topic, same one as in the shuffled text condition)
# Checks: row order of ids/mapping/json, topic_str <-> id is one-to-one, shuffled text uses the topic of topik_acak.
# Memory-lean: one split at a time (the KP20k train files are large).
"""Pilot: build {src, tgt, topic_id} files for topic_emb (true topic) and shuffled_emb (shuffled topic).
"""
import argparse, csv, json, os, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--kp", required=True, help=r"...\dataset\kp20k_v3")
ap.add_argument("--out", required=True, help=r"output root, e.g. ...\dataset\kp20k_v3_emb")
args = ap.parse_args()

TEST = ["kp20k", "inspec", "krapivin400", "nus", "semeval"]
SPLITS = [("kp20k", "train"), ("kp20k", "valid")] + [(d, "test") for d in TEST]


def read_jsonl(p):
    with open(p, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def topic_str_of(topic_src, ori_src):
    pre = topic_src.split(" . <eos>", 1)[0]
    title = ori_src.split(" <eos>", 1)[0]
    assert pre.startswith(title), "topic src does not start with ori title"
    return sys.intern(pre[len(title):].strip())


# ---- pass 1: mapping csv + ids for every split (small) -> global topic index
maps = {}
all_ids = set()
for ds, sp in SPLITS:
    with open(f"{args.kp}/shuffled_mapping/{ds}_{sp}_shuffled_mapping.csv", encoding="utf-8-sig", newline="") as f:
        mp = list(csv.DictReader(f))
    ids_path = f"{args.kp}/ids/{ds}_{sp}_ids.txt"
    ids = [l.strip() for l in open(ids_path, encoding="utf-8-sig") if l.strip()] if os.path.exists(ids_path) else None
    if ids is not None:
        assert len(ids) == len(mp) and all(str(mp[i]["id"]).strip() == ids[i] for i in range(len(mp))), f"id order mismatch {ds} {sp}"
    asli = [int(float(r["topik_asli"])) for r in mp]
    acak = [int(float(r["topik_acak"])) for r in mp]
    maps[(ds, sp)] = (asli, acak)
    all_ids.update(asli); all_ids.update(acak)
uniq = sorted(all_ids)
cmap = {int(k): i for i, k in enumerate(uniq)}
K = len(uniq)
print("topic ids (original):", uniq[:5], "...", uniq[-3:], " K =", K)

# ---- pass 2: one split at a time: check, write both output files, keep only the (small) topic strings
os.makedirs(f"{args.out}/topic_emb", exist_ok=True)
os.makedirs(f"{args.out}/shuffled_emb", exist_ok=True)
real_str, shuf_str = {}, {}
for ds, sp in SPLITS:
    ori = read_jsonl(f"{args.kp}/ori/{ds}_{sp}_ori_v2.json")
    top = read_jsonl(f"{args.kp}/topic/{ds}_{sp}_topic_before_topicv3_v2.json")
    shf = read_jsonl(f"{args.kp}/shuffled/{ds}_{sp}_shuffled_before_topicv3_v2.json")
    asli, acak = maps[(ds, sp)]
    n = len(ori)
    assert len(top) == n and len(shf) == n and len(asli) == n, (ds, sp, n, len(top), len(shf), len(asli))
    rs, ss = [], []
    for i in range(n):
        assert ori[i]["tgt"] == top[i]["tgt"] == shf[i]["tgt"], f"tgt differs {ds} {sp} row {i}"
        rs.append(topic_str_of(top[i]["src"], ori[i]["src"]))
        ss.append(topic_str_of(shf[i]["src"], ori[i]["src"]))
    real_str[(ds, sp)] = rs; shuf_str[(ds, sp)] = ss
    for cond, tid in (("topic_emb", asli), ("shuffled_emb", acak)):
        with open(f"{args.out}/{cond}/{ds}_{sp}_{cond}.json", "w", encoding="utf-8", newline="\n") as f:
            for i in range(n):
                f.write(json.dumps({"src": ori[i]["src"], "tgt": ori[i]["tgt"], "topic_id": cmap[tid[i]]}, ensure_ascii=False) + "\n")
    print(f"  {ds} {sp}: {n} rows written", flush=True)
    del ori, top, shf

# ---- checks that need all splits: id <-> topic string one-to-one, shuffled text == string of topik_acak
id2str = {}
for key, (asli, acak) in maps.items():
    for a, s in zip(asli, real_str[key]):
        if a in id2str:
            assert id2str[a] == s, f"topic id {a} has two different strings ({key})"
        else:
            id2str[a] = s
assert len(set(id2str.values())) == len(id2str), "two topic ids share the same topic string"
bad = tot = 0
for key, (asli, acak) in maps.items():
    for b, s in zip(acak, shuf_str[key]):
        tot += 1
        if b not in id2str or id2str[b] != s:
            bad += 1
assert bad == 0, f"shuffled text does not match topik_acak in {bad}/{tot} rows"
print(f"CHECK OK: {len(id2str)} topics (id <-> string one-to-one), shuffled text matches topik_acak in {tot}/{tot} rows")
same = sum(1 for (asli, acak) in maps.values() for a, b in zip(asli, acak) if a == b)
print("rows where shuffled topic == real topic:", same, "(expected 0)")
a, b = maps[("kp20k", "train")]; N = len(a)
c_real = Counter(a); c_shuf = Counter(b)
tv = 0.5 * sum(abs(c_real[k] / N - c_shuf[k] / N) for k in set(c_real) | set(c_shuf))
print(f"train: total variation distance between real and shuffled topic distribution = {tv:.4f} (small = shuffled keeps the topic frequencies)")
json.dump({"num_topics": K, "id_map": {str(k): v for k, v in cmap.items()}}, open(f"{args.out}/topic_id_map.json", "w"))
open(f"{args.out}/num_topics.txt", "w").write(str(K))
print("WROTE", args.out, " num_topics =", K)
