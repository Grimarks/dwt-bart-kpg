# -*- coding: utf-8 -*-
"""Build the retrieval inputs "title kp ; kp ; ... . <eos> abstract" from the k=5 nearest neighbours'
author keyphrases (stem-deduplicated, max 30): --mode global --tag g5, --mode same_topic --tag st5,
--mode random --tag r5. Also computes neighbours for Inspec/Krapivin/NUS/SemEval with the same rules.

Original notes:
build_retrieval_data.py  (Langkah 1 pilot retrieval keyphrase)

Merakit data latih/uji untuk satu kondisi pilot, dengan format yang sama seperti
kondisi topic:
    src = "<judul> <keyphrase tetangga> . <eos> <abstrak>"
    tgt = sama persis dengan ori

Keyphrase tetangga = keyphrase (present + absent) dari k dokumen KP20k train
terdekat, urut dari tetangga paling mirip, duplikat (per stem) dibuang,
maksimal MAX_KP frasa, dipisah " ; ".

Sumber tetangga:
  - KP20k train/valid/test : file nbr_<split>_<mode>.npy hasil retrieval_pilot_prep.py
  - inspec, krapivin400, nus, semeval : dihitung di sini dengan embedding dan
    aturan buang near-duplicate yang sama (fungsi diambil dari retrieval_pilot_prep.py)

Contoh:
  python build_retrieval_data.py --mode global --k 5 --tag g5
  python build_retrieval_data.py --mode random --k 5 --tag r5      (kontrol)

Keluaran: <DATA>/retr_<tag>/<set>_<split>_retr_<tag>_v2.json  + build_report.txt
Folder yang sudah ada tidak ditimpa (pakai --overwrite bila memang perlu).
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np

import retrieval_pilot_prep as P  # harus satu folder dengan skrip ini

OTHER_SETS = ["inspec", "krapivin400", "nus", "semeval"]
MAX_KP = 30


def split_src(src):
    title, abstract = src.split("<eos>", 1)
    return title.strip(), abstract.strip()


def kp_string(nbr_row, k, train_raw_kps):
    """Gabungkan keyphrase k tetangga: urut tetangga, buang duplikat per stem."""
    seen, out = set(), []
    for j in nbr_row[:k]:
        if j < 0:
            continue
        for kp in train_raw_kps[j]:
            st = P.stem_phrase(kp)
            if st in seen:
                continue
            seen.add(st)
            out.append(kp)
            if len(out) >= MAX_KP:
                return out
    return out


def norm_ws(text):
    """Samakan spasi, termasuk di sekitar <eos>, untuk gerbang pemeriksaan."""
    return " ".join(text.replace("<eos>", " <eos> ").split())


def new_src(title, kps, abstract):
    head = " ".join(x for x in (title, " ; ".join(kps)) if x)
    return f"{head} . <eos> {abstract}".strip()


def other_set_neighbors(name, mode, rows, topic_rows, emb_train, top_train, titles_train,
                        tid, torch, device, tok, model):
    titles = [P.title_of(r["src"]) for r in rows]
    if mode == "random":
        rng = np.random.default_rng([P.SEED, OTHER_SETS.index(name)])
        return P.random_nbr(len(rows), emb_train.shape[0], False, rng), 0, 0
    q_top = np.array([tid.get(P.topic_string(t["src"], ti) or "", -1)
                      for t, ti in zip(topic_rows, titles)], dtype=np.int32)
    n_unknown = int((q_top < 0).sum())
    emb = P.encode([P.embed_text(r["src"]) for r in rows], tok, model, torch, device)
    nb, _, nd = P.knn(emb, emb_train, torch, device, q_is_train=False,
                      same_topic=(mode == "same_topic"), q_top=q_top, t_top=top_train,
                      q_titles=titles, t_titles=titles_train)
    return nb, nd, n_unknown


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=P.MODES)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.k <= P.K_SAVE:
        sys.exit(f"--k harus 1..{P.K_SAVE}")

    out_dir = os.path.join(P.DATA, f"retr_{a.tag}")
    if os.path.exists(out_dir) and not a.overwrite:
        sys.exit(f"Folder sudah ada, tidak ditimpa: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    report = [f"mode={a.mode} k={a.k} tag={a.tag} MAX_KP={MAX_KP} DUP_COS={P.DUP_COS}"]

    def R(msg):
        P.log(msg)
        report.append(msg)

    # keyphrase train mentah (bukan stem) untuk ditulis ke input
    P.log("Baca KP20k train ...")
    train_rows = P.read_jsonl(P.find_file("ori", "train"))
    train_raw_kps = []
    for r in train_rows:
        pres, absn = P.parse_tgt(r["tgt"])
        train_raw_kps.append(pres + absn)
    titles_train = [P.title_of(r["src"]) for r in train_rows]

    jobs = [("kp20k", s) for s in P.SPLITS] + [(n, "test") for n in OTHER_SETS]

    torch = tok = model = device = emb_train = top_train = tid = None
    if a.mode != "random":
        import torch as _t
        from transformers import AutoModel, AutoTokenizer
        torch = _t
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(P.MODEL)
        model = AutoModel.from_pretrained(P.MODEL).to(device).eval()
        emb_train = np.load(os.path.join(P.OUT, "emb_train.npy"))
        top_train = np.load(os.path.join(P.OUT, "topic_id_train.npy"))
        with open(os.path.join(P.OUT, "topics.json"), encoding="utf-8") as f:
            tid = {t: i for i, t in enumerate(json.load(f))}
    else:
        emb_train = np.zeros((len(train_rows), 1), dtype=np.float16)  # hanya untuk ukuran

    for name, split in jobs:
        ori_path = os.path.join(P.DATA, "ori", f"{name}_{split}_ori_v2.json")
        rows = P.read_jsonl(ori_path)
        if name == "kp20k":
            nb = np.load(os.path.join(P.OUT, f"nbr_{split}_{a.mode}.npy"))
            nd = n_unknown = "lihat knn_stats.csv"
            if nb.shape[0] != len(rows):
                sys.exit(f"GAGAL: jumlah tetangga {nb.shape[0]} != baris {len(rows)} ({name} {split})")
        else:
            topic_rows = P.read_jsonl(os.path.join(
                P.DATA, "topic", f"{name}_{split}_topic_before_topicv3_v2.json"))
            if len(topic_rows) != len(rows):
                sys.exit(f"GAGAL: ori/topic beda jumlah baris ({name})")
            nb, nd, n_unknown = other_set_neighbors(
                name, a.mode, rows, topic_rows, emb_train, top_train, titles_train,
                tid, torch, device, tok, model)

        out_rows, n_kp, n_words, n_empty, n_oracle, n_abs_docs = [], 0, 0, 0, 0, 0
        n_irreg, irreg_example = 0, None
        for i, r in enumerate(rows):
            title, abstract = split_src(r["src"])
            # gerbang: src ori harus bisa dibangun ulang (spasi dinormalkan)
            rebuilt = f"{title} <eos> {abstract}"
            if rebuilt != r["src"].strip():
                if norm_ws(rebuilt) != norm_ws(r["src"]):
                    sys.exit(f"GAGAL: format src tidak terduga di {name} {split} baris {i}")
                n_irreg += 1
                if irreg_example is None:
                    irreg_example = (i, repr(r["src"][:160]))
            kps = kp_string(nb[i], a.k, train_raw_kps)
            n_kp += len(kps)
            n_words += sum(len(x.split()) for x in kps)
            n_empty += not kps
            _, absn = P.parse_tgt(r["tgt"])
            if absn:
                n_abs_docs += 1
                st = {P.stem_phrase(x) for x in kps}
                n_oracle += any(P.stem_phrase(x) in st for x in absn)
            new = dict(r)
            new["src"] = new_src(title, kps, abstract)
            out_rows.append(new)

        out_path = os.path.join(out_dir, f"{name}_{split}_retr_{a.tag}_v2.json")
        write_jsonl(out_path, out_rows)
        # gerbang: baca ulang, cek jumlah baris dan tgt identik
        back = P.read_jsonl(out_path)
        assert len(back) == len(rows), "jumlah baris beda"
        assert all(b["tgt"] == r["tgt"] for b, r in zip(back, rows)), "tgt berubah"
        n = len(rows)
        R(f"{name}_{split}: {n} dok | frasa/dok {n_kp/n:.1f} | kata/dok {n_words/n:.1f} | "
          f"tanpa frasa {n_empty} | dok absent dengan >=1 gold absent di input "
          f"{n_oracle}/{n_abs_docs} ({100*n_oracle/max(n_abs_docs,1):.1f}%) | "
          f"near-dup dibuang {nd} | topik tak dikenal {n_unknown} | md5 {md5(out_path)}")
        R(f"   contoh src: {out_rows[0]['src'][:300]}")
        if n_irreg:
            R(f"   spasi tidak rapi di sekitar <eos>: {n_irreg} baris, contoh baris "
              f"{irreg_example[0]}: {irreg_example[1]}")

    with open(os.path.join(out_dir, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    P.log("SELESAI", out_dir)


if __name__ == "__main__":
    main()
