# -*- coding: utf-8 -*-
"""Retrieval step: all-MiniLM-L6-v2 embeddings of every KP20k document (title + abstract) and the 20
nearest KP20k-train neighbours per document in three modes: global, same_topic (same BERTopic topic) and
random (control, np.random.default_rng(2026)). Self and near-duplicates (cosine >= 0.95 or identical title)
are removed. Writes emb_*.npy, nbr_<split>_<mode>.npy, sim_<split>_<mode>.npy, topic_id_*.npy (Kaggle retrieval/).

Original notes:
retrieval_pilot_prep.py  (Langkah 0 pilot retrieval keyphrase, KP20k)

Tujuan: sebelum memakai GPU ~20 jam untuk melatih model, cek dulu apakah
keyphrase milik dokumen-dokumen train yang mirip (tetangga) memang memuat
absent keyphrase dokumen uji. Kalau tidak, model tidak punya bahan untuk ditiru.

Yang dikerjakan skrip ini (TIDAK melatih model):
  1. Embedding semua dokumen KP20k (train/valid/test) dengan all-MiniLM-L6-v2,
     model embedding yang sama dengan BERTopic.
  2. Untuk tiap dokumen, cari 20 tetangga terdekat DI DALAM KP20k train:
       - global      : semua dokumen train
       - same_topic  : hanya dokumen train dengan topik BERTopic yang sama
       - random      : 20 dokumen train acak (kontrol)
     Self (dokumen itu sendiri) dan near-duplicate (cosine >= 0.95 atau judul
     identik) dibuang supaya tidak ada kebocoran.
  3. Hitung cakupan oracle: berapa persen gold absent keyphrase yang muncul
     di gabungan keyphrase tetangga, dibandingkan dengan kontrol acak, daftar
     keyphrase absent paling populer (teks generik), dan string topik.

Mode:
  python retrieval_pilot_prep.py --check   cek file, format, topik, GPU (cepat, tanpa GPU)
  python retrieval_pilot_prep.py --run     jalankan semuanya (~30-60 menit)

Aman dijalankan ulang setelah mati lampu: hasil yang sudah ada (.npy) dipakai lagi.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from functools import lru_cache

os.environ.setdefault("USE_TF", "0")  # cegah crash Keras saat import transformers

import numpy as np

# ===== Pengaturan =====
# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
DATA = os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE", "dataset", "kp20k_v3")
OUT = os.path.join(os.environ.get("DWTBART_ROOT", "."), "output", "retrieval_pilot")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLITS = ["train", "valid", "test"]
K_SAVE = 20                  # jumlah tetangga yang disimpan per dokumen
K_LIST = [1, 3, 5, 10, 20]   # k yang dilaporkan di cakupan
DUP_COS = 0.95               # cosine >= ini dianggap near-duplicate -> dibuang
SEED = 2026
CHUNK = 1024                 # jumlah query per perkalian matriks di GPU
MODES = ["global", "same_topic", "random"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- membaca data ----------
def find_file(cond, split):
    """Cari tepat satu file kp20k_<split>_*.json(l) di folder kondisi."""
    folder = os.path.join(DATA, cond)
    hits = []
    for ext in ("json", "jsonl"):
        hits += glob.glob(os.path.join(folder, f"kp20k_{split}_*.{ext}"))
    if len(hits) != 1:
        log(f"GAGAL menemukan tepat 1 file untuk {cond}/{split}: {hits}")
        if os.path.isdir(folder):
            log("Isi folder:", sorted(os.listdir(folder)))
        else:
            log("Folder tidak ada:", folder)
        sys.exit(1)
    return hits[0]


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_tgt(tgt):
    """'p1;p2;<peos>;a1;a2' -> (present, absent). Pembagi present/absent = <peos>."""
    parts = [p.strip() for p in tgt.split(";")]
    if "<peos>" in parts:
        i = parts.index("<peos>")
        pres, absn = parts[:i], parts[i + 1:]
    else:
        pres, absn = parts, []
    return [p for p in pres if p], [p for p in absn if p]


def title_of(src):
    return src.split("<eos>", 1)[0].strip()


def topic_string(topic_src, ori_title):
    """Input topic = 'judul + topik + . <eos> abstrak'. Ambil bagian topiknya."""
    head = topic_src.split("<eos>", 1)[0].strip()
    if not head.startswith(ori_title):
        return None
    s = head[len(ori_title):].strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def embed_text(src):
    return src.replace("<eos>", ".").replace("<digit>", "0")


# ---------- stemming (aturan cocok sama gaya evaluate.py: Porter per kata) ----------
try:
    from nltk.stem.porter import PorterStemmer
    _ps = PorterStemmer()
    HAVE_NLTK = True
except ImportError:
    _ps = None
    HAVE_NLTK = False


@lru_cache(maxsize=None)
def stem_phrase(p):
    if _ps is None:
        return p
    return " ".join(_ps.stem(w) for w in p.split())


# ---------- GPU ----------
def gpu_apps():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return f"(nvidia-smi gagal: {e})"


def load_split(split):
    """Baca ori + topic untuk satu split, kembalikan dict yang sudah diproses."""
    ori = read_jsonl(find_file("ori", split))
    top = read_jsonl(find_file("topic", split))
    if len(ori) != len(top):
        log(f"GAGAL: jumlah baris ori ({len(ori)}) != topic ({len(top)}) untuk {split}")
        sys.exit(1)
    titles, texts, gold_abs, all_kps, topics = [], [], [], [], []
    n_fail = 0
    for o, t in zip(ori, top):
        ti = title_of(o["src"])
        titles.append(ti)
        texts.append(embed_text(o["src"]))
        pres, absn = parse_tgt(o["tgt"])
        gold_abs.append({stem_phrase(a) for a in absn})
        all_kps.append({stem_phrase(k) for k in pres + absn})
        ts = topic_string(t["src"], ti)
        if ts is None:
            n_fail += 1
        topics.append(ts)
    return dict(titles=titles, texts=texts, gold_abs=gold_abs,
                all_kps=all_kps, topics=topics, n_topic_fail=n_fail)


# ---------- mode --check ----------
def do_check():
    log("Python", sys.version.split()[0])
    for mod in ("torch", "transformers", "nltk"):
        try:
            m = __import__(mod)
            log(f"{mod} {getattr(m, '__version__', '?')} OK")
        except Exception as e:
            log(f"{mod} TIDAK ADA: {e}")
    try:
        import torch
        log("CUDA tersedia:", torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception:
        pass
    log("Proses di GPU sekarang:", gpu_apps() or "(kosong)")

    for split in SPLITS:
        fo, ft = find_file("ori", split), find_file("topic", split)
        log(f"[{split}] ori  = {fo}")
        log(f"[{split}] topic= {ft}")
        o3, t3 = read_jsonl(fo, 3), read_jsonl(ft, 3)
        for o, t in zip(o3, t3):
            ti = title_of(o["src"])
            pres, absn = parse_tgt(o["tgt"])
            log("   judul :", ti[:90])
            log("   topik :", (topic_string(t["src"], ti) or "GAGAL EKSTRAK")[:120])
            log("   present:", pres[:4], "| absent:", absn[:4])

    for split in ("valid", "test"):
        d = load_split(split)
        uniq = Counter(t for t in d["topics"] if t is not None)
        n_abs = sum(1 for g in d["gold_abs"] if g)
        log(f"[{split}] {len(d['titles'])} dok, ekstrak topik gagal {d['n_topic_fail']}, "
            f"topik unik {len(uniq)}, dok dengan >=1 gold absent {n_abs}")
    log("CEK SELESAI. Jika topik unik sekitar 65 dan gagal 0, jalankan --run.")


# ---------- embedding ----------
def encode(texts, tok, model, torch, device, bs=256):
    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])[::-1]  # panjang dulu: padding efisien
    nb = (len(texts) + bs - 1) // bs
    for b, s in enumerate(range(0, len(texts), bs)):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=256, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)          # mean pooling
        e = torch.nn.functional.normalize(e.float(), dim=-1)  # cosine = dot product
        out[idx] = e.cpu().numpy().astype(np.float16)
        if b % 200 == 0:
            log(f"   embed batch {b}/{nb}")
    return out


# ---------- tetangga ----------
def knn(Q, T, torch, device, q_is_train, same_topic, q_top, t_top, q_titles, t_titles):
    nq = Q.shape[0]
    Tt = torch.from_numpy(T).to(device)
    tt = torch.from_numpy(t_top).to(device) if same_topic else None
    nbr = np.full((nq, K_SAVE), -1, dtype=np.int32)
    sim = np.zeros((nq, K_SAVE), dtype=np.float16)
    n_dup_docs = 0
    extra = 20  # ambil lebih banyak untuk menggantikan yang dibuang
    for s in range(0, nq, CHUNK):
        q = torch.from_numpy(Q[s:s + CHUNK]).to(device)
        n = q.shape[0]
        sc = q @ Tt.T
        if q_is_train:  # buang dokumen itu sendiri
            r = torch.arange(n, device=device)
            sc[r, s + r] = float("-inf")
        if same_topic:
            qt = torch.from_numpy(q_top[s:s + n]).to(device)
            sc.masked_fill_(tt[None, :] != qt[:, None], float("-inf"))
        v, ix = sc.topk(min(K_SAVE + extra, sc.shape[1]), dim=1)
        v = v.float().cpu().numpy()
        ix = ix.cpu().numpy()
        for r in range(n):
            qi = s + r
            kept, dup = 0, False
            for c in range(ix.shape[1]):
                val, j = v[r, c], ix[r, c]
                if not np.isfinite(val):
                    break
                if val >= DUP_COS or (q_titles[qi] and q_titles[qi] == t_titles[j]):
                    dup = True
                    continue
                nbr[qi, kept], sim[qi, kept] = j, val
                kept += 1
                if kept == K_SAVE:
                    break
            n_dup_docs += dup
        if (s // CHUNK) % 50 == 0:
            log(f"   knn {s}/{nq}")
        del sc
    return nbr, sim, n_dup_docs


def random_nbr(nq, n_train, q_is_train, rng):
    r = rng.integers(0, n_train, size=(nq, K_SAVE)).astype(np.int32)
    if q_is_train:
        self_idx = np.arange(nq)[:, None]
        r = np.where(r == self_idx, (r + 1) % n_train, r)
    return r


# ---------- cakupan ----------
def coverage_rows(split, d, nbr_by_mode, train_kps, popular):
    rows = []
    docs = [i for i, g in enumerate(d["gold_abs"]) if g]
    for mode, nbr in nbr_by_mode.items():
        for k in K_LIST:
            rec = hit = size = words = prec = 0.0
            for i in docs:
                cand = set()
                for j in nbr[i, :k]:
                    if j >= 0:
                        cand |= train_kps[j]
                h = len(d["gold_abs"][i] & cand)
                rec += h / len(d["gold_abs"][i])
                hit += h > 0
                size += len(cand)
                words += sum(len(c.split()) for c in cand)
                prec += h / len(cand) if cand else 0
            n = len(docs)
            rows.append(dict(split=split, mode=mode, k=k, n_docs=n,
                             recall=rec / n, doc_hit=hit / n, cand_size=size / n,
                             cand_words=words / n, precision=prec / n))
        if mode == "global":  # kontrol teks generik dengan ukuran sama
            for k in K_LIST:
                m = int(round(next(r["cand_size"] for r in rows
                                   if r["mode"] == "global" and r["k"] == k)))
                cand = set(popular[:m])
                rec = hit = 0.0
                for i in docs:
                    h = len(d["gold_abs"][i] & cand)
                    rec += h / len(d["gold_abs"][i])
                    hit += h > 0
                rows.append(dict(split=split, mode="popular", k=k, n_docs=len(docs),
                                 recall=rec / len(docs), doc_hit=hit / len(docs),
                                 cand_size=m, cand_words=float("nan"),
                                 precision=float("nan")))
    rec = hit = 0.0
    for i in docs:  # string topik: absent keyphrase muncul utuh di string topik
        ts = d["topics"][i]
        tss = " " + stem_phrase(ts) + " " if ts else ""
        h = sum(1 for g in d["gold_abs"][i] if f" {g} " in tss)
        rec += h / len(d["gold_abs"][i])
        hit += h > 0
    rows.append(dict(split=split, mode="topic_string", k=0, n_docs=len(docs),
                     recall=rec / len(docs), doc_hit=hit / len(docs),
                     cand_size=float("nan"), cand_words=float("nan"),
                     precision=float("nan")))
    return rows


# ---------- mode --run ----------
def do_run(force):
    apps = gpu_apps()
    if apps and not apps.startswith("(") and not force:
        log("GPU sedang dipakai proses lain, BERHENTI supaya job itu tidak terganggu:")
        log(apps)
        log("Jalankan lagi setelah GPU kosong (atau --force jika yakin aman).")
        sys.exit(2)
    if not HAVE_NLTK:
        log("PERINGATAN: nltk tidak ada, pencocokan tanpa stemming.")
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)

    data = {}
    for split in SPLITS:
        log(f"Baca {split} ...")
        data[split] = load_split(split)
        log(f"   {len(data[split]['titles'])} dok, ekstrak topik gagal {data[split]['n_topic_fail']}")

    # id topik: string topik yang sama -> id yang sama; gagal ekstrak -> -1
    names = sorted({t for s in SPLITS for t in data[s]["topics"] if t is not None})
    tid = {t: i for i, t in enumerate(names)}
    with open(os.path.join(OUT, "topics.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=1)
    log(f"Topik unik di semua split: {len(names)}")
    top_ids = {s: np.array([tid.get(t, -1) if t is not None else -1
                            for t in data[s]["topics"]], dtype=np.int32) for s in SPLITS}
    for s in SPLITS:
        np.save(os.path.join(OUT, f"topic_id_{s}.npy"), top_ids[s])

    # 1. embedding (dipakai ulang kalau sudah ada)
    emb = {}
    tok = model = None
    for s in SPLITS:
        p = os.path.join(OUT, f"emb_{s}.npy")
        if os.path.exists(p):
            e = np.load(p)
            if e.shape[0] == len(data[s]["texts"]):
                emb[s] = e
                log(f"Embedding {s} dipakai ulang {e.shape}")
                continue
        if model is None:
            log("Muat model", MODEL)
            tok = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModel.from_pretrained(MODEL).to(device).eval()
        log(f"Embedding {s} ({len(data[s]['texts'])} dok) ...")
        emb[s] = encode(data[s]["texts"], tok, model, torch, device)
        np.save(p, emb[s])
    del model
    torch.cuda.empty_cache()

    # 2. tetangga
    n_train = len(data["train"]["titles"])
    rng = np.random.default_rng(SEED)
    stats = []
    nbrs = {}
    for s in SPLITS:
        for mode in MODES:
            p = os.path.join(OUT, f"nbr_{s}_{mode}.npy")
            if os.path.exists(p):
                nbrs[(s, mode)] = np.load(p)
                log(f"Tetangga {s}/{mode} dipakai ulang")
                continue
            log(f"Tetangga {s}/{mode} ...")
            if mode == "random":
                nb = random_nbr(len(data[s]["titles"]), n_train, s == "train", rng)
                nd = 0
            else:
                nb, sm, nd = knn(emb[s], emb["train"], torch, device,
                                 q_is_train=(s == "train"),
                                 same_topic=(mode == "same_topic"),
                                 q_top=top_ids[s], t_top=top_ids["train"],
                                 q_titles=data[s]["titles"],
                                 t_titles=data["train"]["titles"])
                np.save(os.path.join(OUT, f"sim_{s}_{mode}.npy"), sm)
            np.save(p, nb)
            nbrs[(s, mode)] = nb
            stats.append(dict(split=s, mode=mode, docs_with_near_dup_removed=nd,
                              docs_with_less_than_k=int((nb[:, -1] < 0).sum())))
            log(f"   dok dengan near-duplicate dibuang: {nd}")

    # 3. cakupan oracle (valid untuk memilih k, test untuk dilaporkan)
    log("Hitung cakupan ...")
    train_kps = data["train"]["all_kps"]
    pop = Counter(a for g in data["train"]["gold_abs"] for a in g)
    popular = [a for a, _ in pop.most_common()]
    rows = []
    for s in ("valid", "test"):
        rows += coverage_rows(s, data[s], {m: nbrs[(s, m)] for m in MODES},
                              train_kps, popular)

    with open(os.path.join(OUT, "coverage.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "knn_stats.csv"), "w", newline="", encoding="utf-8") as f:
        if stats:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)

    lines = ["# Cakupan oracle absent keyphrase (KP20k)", "",
             "recall = rata-rata per dokumen dari (gold absent yang ada di kandidat / jumlah gold absent)",
             "doc_hit = persen dokumen dengan >=1 gold absent di kandidat", ""]
    for s in ("valid", "test"):
        lines += [f"## {s}", "",
                  "| mode | k | recall | doc_hit | kandidat | kata | presisi |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["split"] == s:
                lines.append(f"| {r['mode']} | {r['k']} | {r['recall']*100:.2f}% | "
                             f"{r['doc_hit']*100:.2f}% | {r['cand_size']:.1f} | "
                             f"{r['cand_words']:.1f} | {r['precision']*100:.2f}% |")
        lines.append("")
    lines += ["## Near-duplicate yang dibuang", "",
              "| split | mode | dok terdampak | dok < 20 tetangga |", "|---|---|---|---|"]
    for st in stats:
        lines.append(f"| {st['split']} | {st['mode']} | {st['docs_with_near_dup_removed']} | "
                     f"{st['docs_with_less_than_k']} |")
    with open(os.path.join(OUT, "coverage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("SELESAI. Lihat", os.path.join(OUT, "coverage.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    do_check() if a.check else do_run(a.force)
