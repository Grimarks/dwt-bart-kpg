# -*- coding: utf-8 -*-
"""Overlap check KP20k-train vs Inspec/Krapivin/NUS/SemEval test documents: (A) identical word-index sequence
of title+abstract / abstract / title, (B) MiniLM cosine to the most similar training document. Writes
output/overlap_testsets/pasangan_<set>.csv, which human_eval/sample_human_eval_v2b.py uses to drop test
articles with a near-twin in the training data. Without a cache folder the test-set embeddings are
computed on the CPU with the same embed_text() as the retrieval step.

Original notes:
cek_overlap_testsets.py  (cegah leakage KP20k train vs Inspec/Krapivin/NUS/SemEval)

Dua cara, keduanya dilaporkan:
  A. Cara deret kata (kata + urutan sama persis):
     - semua kata unik di KP20k train diberi nomor indeks (kamus);
     - tiap artikel (train dan uji) diubah menjadi deret nomor indeks; kata uji yang tidak ada
       di kamus diberi -1 (otomatis tidak mungkin sama);
     - dua artikel dianggap SAMA hanya jika deret indeksnya sama persis (kata dan urutan sama).
     Dicek untuk judul+abstrak, abstrak saja, dan judul saja.
  B. Cara embedding: embedding MiniLM abstrak (judul+abstrak, sama seperti retrieval)
     tiap artikel uji dibandingkan dengan SEMUA artikel KP20k train (cosine), diambil yang
     paling mirip. Dilaporkan jumlah artikel uji dengan cosine >= 0,95 / 0,90 / 0,80.
Keluaran: output\\overlap_testsets\\ (LAPORAN_overlap.md, pasangan_<set>.csv = 3 tetangga
terdekat tiap artikel uji + judul keduanya untuk dicek manual).
CPU saja (tidak mengganggu GPU). Embedding set uji dipakai ulang dari folder cache (opsional) bila ada.
"""
import csv
import hashlib
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_pilot_prep as P  # noqa: E402

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
BASE = os.environ.get("DWTBART_ROOT", ".")
OUT = os.path.join(BASE, r"output\overlap_testsets")
EMB_CACHE = os.environ.get("DWTBART_EMB_CACHE", os.path.join(OUT, "emb_cache"))  # optional cache of emb_ref_<set>.npy
SETS = ["inspec", "krapivin400", "nus", "semeval"]
THR = [0.95, 0.90, 0.80]


def key(ids):
    return hashlib.md5(np.asarray(ids, dtype=np.int32).tobytes()).digest()


def to_ids(text, vocab):
    return [vocab.get(w, -1) for w in text.split()]


def parts(src):
    t, a = src.split("<eos>", 1)
    return t.strip(), a.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    train = P.read_jsonl(P.find_file("ori", "train"))
    P.log("KP20k train: %d artikel" % len(train))
    vocab = {}
    for r in train:
        for w in r["src"].split():
            if w not in vocab:
                vocab[w] = len(vocab)
    P.log("kamus kata unik KP20k train: %d" % len(vocab))
    idx = {"full": {}, "abstract": {}, "title": {}}
    seqs = []
    for i, r in enumerate(train):
        t, a = parts(r["src"])
        it, ia = to_ids(t, vocab), to_ids(a, vocab)
        seqs.append((it, ia))
        for name, ids in (("full", it + [-2] + ia), ("abstract", ia), ("title", it)):
            if len(ids) >= 3:
                idx[name].setdefault(key(ids), []).append(i)
    T = np.load(os.path.join(P.OUT, "emb_train.npy")).astype(np.float32)
    if T.shape[0] != len(train):
        sys.exit("GAGAL: emb_train.npy %d baris != train %d" % (T.shape[0], len(train)))
    titles_train = [parts(r["src"])[0] for r in train]
    tok = model = None
    summary = []
    for name in SETS:
        rows = P.read_jsonl(os.path.join(P.DATA, "ori", "%s_test_ori_v2.json" % name))
        texts = [P.embed_text(r["src"]) for r in rows]
        cache = os.path.join(EMB_CACHE, "emb_ref_%s.npy" % name)
        if os.path.exists(cache) and np.load(cache).shape[0] == len(rows):
            Q = np.load(cache).astype(np.float32)
            src_emb = "dipakai ulang dari Tahap B"
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if model is None:
                tok = AutoTokenizer.from_pretrained(P.MODEL)
                model = AutoModel.from_pretrained(P.MODEL).eval()
            Q = P.encode(texts, tok, model, torch, torch.device("cpu")).astype(np.float32)
            src_emb = "dihitung (CPU)"
        sims = Q @ T.T
        part = np.argpartition(-sims, 3, axis=1)[:, :3]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        top = np.take_along_axis(part, order, axis=1)
        exact = {"full": 0, "abstract": 0, "title": 0}
        thr_n = {t: 0 for t in THR}
        with open(os.path.join(OUT, "pasangan_%s.csv" % name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uji_baris", "rank", "train_baris", "cosine", "sama_full", "sama_abstrak", "sama_judul",
                        "judul_uji", "judul_train"])
            for qi, r in enumerate(rows):
                t, a = parts(r["src"])
                it, ia = to_ids(t, vocab), to_ids(a, vocab)
                hit = {}
                for nm, ids in (("full", it + [-2] + ia), ("abstract", ia), ("title", it)):
                    cand = idx[nm].get(key(ids), []) if len(ids) >= 3 else []
                    real = [j for j in cand if (seqs[j][0] + [-2] + seqs[j][1] if nm == "full"
                                                else seqs[j][1] if nm == "abstract" else seqs[j][0]) == ids]
                    hit[nm] = set(real)
                    exact[nm] += bool(real)
                best = float(sims[qi, top[qi, 0]])
                for th in THR:
                    thr_n[th] += best >= th
                extra = sorted((hit["full"] | hit["abstract"] | hit["title"]) - set(int(x) for x in top[qi]))
                for j in extra:  # artikel train yang sama persis tapi tidak masuk 3 terdekat
                    w.writerow([qi, 0, j, "%.4f" % sims[qi, j], int(j in hit["full"]), int(j in hit["abstract"]),
                                int(j in hit["title"]), t[:150], titles_train[j][:150]])
                for rank, j in enumerate(top[qi]):
                    w.writerow([qi, rank + 1, int(j), "%.4f" % sims[qi, j],
                                int(j in hit["full"]), int(j in hit["abstract"]), int(j in hit["title"]),
                                t[:150], titles_train[j][:150]])
        n = len(rows)
        summary.append((name, n, exact, thr_n, float(np.median(sims.max(axis=1))), src_emb))
        P.log("%s: %d artikel | sama persis full %d, abstrak %d, judul %d | cos>=0.95 %d, >=0.90 %d, >=0.80 %d"
              % (name, n, exact["full"], exact["abstract"], exact["title"], thr_n[0.95], thr_n[0.90], thr_n[0.80]))
    L = ["# Cek kemiripan KP20k train vs 4 dataset uji", "",
         "Dibuat %s (%.1f menit). KP20k train %d artikel, kamus %d kata unik." % (
             time.strftime("%Y-%m-%d %H:%M"), (time.time() - t0) / 60, len(train), len(vocab)), "",
         "A = cara deret kata: sama jika deret indeks kata (kata + urutan) sama persis.",
         "B = cosine embedding MiniLM judul+abstrak, artikel train paling mirip.", "",
         "| dataset | artikel | A: sama judul+abstrak | A: sama abstrak | A: sama judul | B: cos>=0,95 | B: cos>=0,90 | B: cos>=0,80 | B: median cos maks |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, n, ex, th, med, _ in summary:
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d | %.3f |" % (
            name, n, ex["full"], ex["abstract"], ex["title"], th[0.95], th[0.90], th[0.80], med))
    L += ["", "Rincian 3 artikel train terdekat per artikel uji: pasangan_<dataset>.csv (cek manual yang cosine tinggi)."]
    with open(os.path.join(OUT, "LAPORAN_overlap.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    P.log("\n".join(L))
    P.log("DONE cek_overlap ->", OUT)


if __name__ == "__main__":
    main()
