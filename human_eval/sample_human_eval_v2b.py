# -*- coding: utf-8 -*-
r"""Human-evaluation sampling: BART (ori) vs DWT-BART (st5t), seed 1234; 112 articles (64 KP20k + 12 each
of Inspec, Krapivin-397, NUS, SemEval), 7 forms x 16 articles, 2 evaluators per form, random.Random(2026),
A/B order random per article. Writes forms.json, ab_key.csv (the A/B key; NOT published) and a report.
Ratings: human_evaluation_ratings_anonymised.csv; means: analyse_ratings.py.

Original notes:
sample_human_eval_v2b.py (PC3, CPU) - pemilihan artikel human evaluation BARU: BART saja (ori) vs BERTopic baru + judul (st5t),
seed 1234 kedua model. Desain: evaluator seperti evaluasi sebelumnya (14 orang, 7 form x 16 artikel, 2 evaluator per form),
TIDAK semuanya KP20k:
  v2b: artikel SAMA dengan v2 (64 KP20k + 12 Inspec + 12 Krapivin + 12 NUS + 12 SemEval), tetapi SETIAP form
  dicampur: 9-10 KP20k + 6-7 lainnya, minimal 1 dari tiap dataset, urutan diacak di dalam form, A/B diacak ulang.
  Alasan: setiap pasangan evaluator menilai semua dataset, sehingga dataset tidak tercampur dengan kelompok evaluator.
Aturan (dilaporkan di artikel):
  - present/absent ditentukan dari teks sumber (judul + abstrak) dengan aturan evaluate.py (stem Porter, urutan kata), bukan dari <peos>
  - kerangka sampel: kedua model punya >=1 absent, himpunan absent kedua model berbeda, sumber >= 50 token
  - artikel uji yang punya "kembaran" di data latih dibuang: KP20k = tetangga global #1 cosine >= 0,90;
    set lain = baris di output\overlap_testsets\pasangan_<set>.csv dengan cosine >= 0,90, judul sama, atau abstrak sama
  - pengambilan acak random.Random(2026); urutan A/B acak per artikel
Keluaran: forms.json (untuk Apps Script), ab_key.csv (kunci A/B, tidak dipublikasikan), laporan_sampling.txt
"""
import argparse
import csv
import json
import os
import random
import re

import numpy as np

try:
    from nltk.stem.porter import PorterStemmer
    _ps = PorterStemmer()
    stem = _ps.stem
except ImportError:
    stem = lambda w: w

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="research output folder (contains retrieval_pilot/, pilot_stats/spec.json, overlap_testsets/)")
ap.add_argument("--kp", required=True, help="kp20k_v3 data folder (contains ids/krapivin400_test_ids.txt)")
ap.add_argument("--ori_dir", required=True)
ap.add_argument("--st5t_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--min_tokens", type=int, default=50)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
RP = os.path.join(a.out, "retrieval_pilot")
KP = a.kp
PLAN_KP = ("kp20k", 4, 16)
MIX = ["inspec", "krapivin397", "nus", "semeval"]
MIX_FORMS, MIX_PER = 3, 4


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def stem_seq(text):
    return [stem(w) for w in text.lower().split()]


def contains(src_st, kp_st):
    n = len(kp_st)
    if n == 0:
        return True
    for i in range(len(src_st) - n + 1):
        if src_st[i:i + n] == kp_st:
            return True
    return False


def split_by_src(line, src_st):
    kps = [x.strip() for x in line.split(";") if x.strip() and x.strip() != "<peos>"]
    kps = list(dict.fromkeys(kps))
    pres = [k for k in kps if contains(src_st, stem_seq(k))]
    absn = [k for k in kps if k not in pres]
    return pres, absn


def detok(s):
    s = s.replace("<digit>", "#")
    s = re.sub(r"\s+-\s+", "-", s)
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return s.strip()


with open(os.path.join(a.out, "pilot_stats", "spec.json"), encoding="utf-8") as f:
    refs = json.load(f)["references"]


def excluded(ds, n):
    bad = set()
    if ds == "kp20k":
        sim = np.load(os.path.join(RP, "sim_test_global.npy")).astype(np.float32)
        bad = set(np.where(sim[:, 0] >= 0.90)[0].tolist())
        return bad
    name = "krapivin400" if ds == "krapivin397" else ds
    rows400 = set()
    with open(os.path.join(a.out, "overlap_testsets", "pasangan_%s.csv" % name), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["cosine"]) >= 0.90 or r.get("sama_judul") == "1" or r.get("sama_abstrak") == "1" or r.get("sama_full") == "1":
                rows400.add(int(r["uji_baris"]))
    if ds != "krapivin397":
        return rows400
    ids = [l.strip() for l in read_lines(os.path.join(KP, "ids", "krapivin400_test_ids.txt")) if l.strip()]
    drop = {"1007763", "1009027", "1024484"}
    keep = [i for i, x in enumerate(ids) if x not in drop]          # baris 400 yang masuk 397, urut
    return {k for k, i400 in enumerate(keep) if i400 in rows400}


def frame_for(ds):
    test = [json.loads(l) for l in read_lines(refs[ds]) if l.strip()]
    ho = read_lines(os.path.join(a.ori_dir, "kp20k_ori_seed1234_%s_hypotheses.txt" % ds))[:len(test)]
    hs = read_lines(os.path.join(a.st5t_dir, "kp20k_retr_st5t_seed1234_%s_hypotheses.txt" % ds))[:len(test)]
    assert len(test) == len(ho) == len(hs), (ds, len(test), len(ho), len(hs))
    bad = excluded(ds, len(test))
    rows, n_abs, n_diff = [], 0, 0
    for i, (d, lo, ls) in enumerate(zip(test, ho, hs)):
        src_st = stem_seq(d["src"].replace("<eos>", " "))
        po, ao = split_by_src(lo, src_st)
        ps, as_ = split_by_src(ls, src_st)
        if ao and as_:
            n_abs += 1
            if set(ao) != set(as_):
                n_diff += 1
                if i not in bad and len(d["src"].split()) >= a.min_tokens:
                    rows.append(dict(ds=ds, i=i, d=d, po=po, ao=ao, ps=ps, as_=as_))
    rep = "%-12s uji %5d | kedua model >=1 absent %5d | absent beda %5d | kembaran dibuang %4d | kerangka %5d" % (
        ds, len(test), n_abs, n_diff, len(bad), len(rows))
    return rows, rep


rng = random.Random(a.seed)
reports, picked = [], {}
for ds, nf, per in [PLAN_KP]:
    rows, rep = frame_for(ds); reports.append(rep)
    picked[ds] = rng.sample(rows, nf * per)
for ds in MIX:
    rows, rep = frame_for(ds); reports.append(rep)
    picked[ds] = rng.sample(rows, MIX_FORMS * MIX_PER)

NF = 7
forms_rows = [[] for _ in range(NF)]
k = 0
for ds in MIX:                                   # 48 artikel non-KP20k dibagi bergiliran -> tiap form 6-7, tiap dataset >= 1
    for r in picked[ds]:
        forms_rows[k % NF].append(r); k += 1
kp = list(picked["kp20k"])                      # 64 KP20k mengisi sisa sampai 16 per form
for f in range(NF):
    need = 16 - len(forms_rows[f])
    forms_rows[f] += kp[:need]; kp = kp[need:]
assert not kp and all(len(x) == 16 for x in forms_rows)
for f in range(NF):
    rng.shuffle(forms_rows[f])
comp = []
for f in range(NF):
    c = {}
    for r in forms_rows[f]:
        c[r["ds"]] = c.get(r["ds"], 0) + 1
    assert all(c.get(ds, 0) >= 1 for ds in MIX), (f, c)
    comp.append("form %d: " % (f + 1) + ", ".join("%s %d" % (ds, c.get(ds, 0)) for ds in ["kp20k"] + MIX))

forms, key = [], []
for fi, rows in enumerate(forms_rows):
    items = []
    for k, r in enumerate(rows):
        title, _, abstract = r["d"]["src"].partition(" <eos> ")
        a_is_ori = rng.random() < 0.5
        A = (r["po"], r["ao"]) if a_is_ori else (r["ps"], r["as_"])
        B = (r["ps"], r["as_"]) if a_is_ori else (r["po"], r["ao"])
        items.append(dict(no=k + 1, title=detok(title), abstract=detok(abstract),
                          A_present=[detok(x) for x in A[0]], A_absent=[detok(x) for x in A[1]],
                          B_present=[detok(x) for x in B[0]], B_absent=[detok(x) for x in B[1]]))
        key.append(dict(form=fi + 1, no=k + 1, dataset=r["ds"], test_row=r["i"],
                        A_is="ori" if a_is_ori else "st5t", B_is="st5t" if a_is_ori else "ori"))
    forms.append(dict(form=fi + 1, items=items))

with open(os.path.join(a.out_dir, "forms.json"), "w", encoding="utf-8") as f:
    json.dump(forms, f, ensure_ascii=False, indent=1)
with open(os.path.join(a.out_dir, "ab_key.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(key[0].keys())); w.writeheader(); w.writerows(key)
lines = ["Human evaluation v2: BART saja (ori) vs BERTopic baru + judul (st5t), seed 1234; seed sampling %d" % a.seed] + reports + [
    "v2b: setiap form campuran (artikel sama dengan v2, pembagian dan A/B baru):"] + comp + [
    "A=ori: %d | A=st5t: %d" % (sum(k["A_is"] == "ori" for k in key), sum(k["A_is"] == "st5t" for k in key)),
    "absent per himpunan (rata-rata) ori %.2f | st5t %.2f" % (
        np.mean([len(it["A_absent"] if kk["A_is"] == "ori" else it["B_absent"]) for f_, ff in zip(range(len(forms)), forms) for it, kk in zip(ff["items"], [x for x in key if x["form"] == ff["form"]])]),
        np.mean([len(it["B_absent"] if kk["A_is"] == "ori" else it["A_absent"]) for f_, ff in zip(range(len(forms)), forms) for it, kk in zip(ff["items"], [x for x in key if x["form"] == ff["form"]])]))]
with open(os.path.join(a.out_dir, "laporan_sampling.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("selesai ->", a.out_dir)
