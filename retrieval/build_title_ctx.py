# -*- coding: utf-8 -*-
"""Build g5t (DWT-style context with titles, global neighbours): "title t1 : k ; k | t2 : ... . <eos> abstract",
same neighbours and keyphrases as g5. Used for g5t:  python build_title_ctx.py --fmt kpt --tag g5t

Original notes:
build_title_ctx.py  (konteks retrieval = keyphrase saja vs keyphrase + judul)

Membuat data kondisi baru dengan tetangga yang SAMA PERSIS dengan g5 (5 tetangga global
MiniLM, nbr_<split>_global.npy) dan keyphrase yang SAMA PERSIS (urutan, buang duplikat per
stem, maks 30). Bedanya hanya isi konteks:
  g5   (sudah ada) : judul  kp1 ; kp2 ; kp3 ; ... . <eos> abstrak
  g5t  (--fmt kpt) : judul  judul_tetangga1 : kp1 ; kp2 | judul_tetangga2 : kp3 ; ... . <eos> abstrak
Jadi selisih g5t - g5 = efek menambah judul tetangga.

Contoh:
  python build_title_ctx.py --fmt kpt --tag g5t
Keluaran: <DATA>/retr_<tag>/<set>_<split>_retr_<tag>_v2.json + build_report.txt (format sama dengan g5).
Taruh di folder yang sama dengan build_retrieval_data.py dan retrieval_pilot_prep.py.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np

import retrieval_pilot_prep as P
import build_retrieval_data as B

K = 5


def grouped(nbr_row, train_raw_kps):
    """Sama dengan B.kp_string (k=5, maks 30, buang duplikat per stem) tetapi tetap tahu
    keyphrase berasal dari tetangga mana. Kembalikan [(j, [kp,...]), ...] urut tetangga."""
    seen, total, out = set(), 0, []
    for j in nbr_row[:K]:
        if j < 0:
            continue
        mine = []
        for kp in train_raw_kps[j]:
            if total >= B.MAX_KP:
                break
            st = P.stem_phrase(kp)
            if st in seen:
                continue
            seen.add(st)
            mine.append(kp)
            total += 1
        out.append((int(j), mine))
    return out


def context(groups, titles_train, fmt):
    if fmt == "t":
        return " | ".join(titles_train[j] for j, _ in groups)
    parts = []
    for j, kps in groups:
        if kps:
            parts.append("%s : %s" % (titles_train[j], " ; ".join(kps)))
        else:
            parts.append(titles_train[j])
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True, choices=["kpt", "t"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(P.DATA, "retr_%s" % a.tag)
    if os.path.exists(out_dir) and not a.overwrite:
        sys.exit("Folder sudah ada, tidak ditimpa: " + out_dir)
    g5_dir = os.path.join(P.DATA, "retr_g5")
    if not os.path.isdir(g5_dir):
        sys.exit("GAGAL: folder retr_g5 tidak ada (dipakai sebagai gerbang pembanding): " + g5_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = ["fmt=%s tag=%s k=%d MAX_KP=%d (tetangga dan keyphrase sama dengan g5)" % (a.fmt, a.tag, K, B.MAX_KP)]

    def R(msg):
        P.log(msg)
        report.append(msg)

    train_rows = P.read_jsonl(P.find_file("ori", "train"))
    train_raw_kps = []
    for r in train_rows:
        pres, absn = P.parse_tgt(r["tgt"])
        train_raw_kps.append(pres + absn)
    titles_train = [P.title_of(r["src"]) for r in train_rows]

    import torch
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(P.MODEL)
    model = AutoModel.from_pretrained(P.MODEL).to(device).eval()
    emb_train = np.load(os.path.join(P.OUT, "emb_train.npy"))
    top_train = np.load(os.path.join(P.OUT, "topic_id_train.npy"))
    with open(os.path.join(P.OUT, "topics.json"), encoding="utf-8") as f:
        tid = {t: i for i, t in enumerate(json.load(f))}

    jobs = [("kp20k", s) for s in P.SPLITS] + [(n, "test") for n in B.OTHER_SETS]
    for name, split in jobs:
        rows = P.read_jsonl(os.path.join(P.DATA, "ori", "%s_%s_ori_v2.json" % (name, split)))
        if name == "kp20k":
            nb = np.load(os.path.join(P.OUT, "nbr_%s_global.npy" % split))
        else:
            topic_rows = P.read_jsonl(os.path.join(
                P.DATA, "topic", "%s_%s_topic_before_topicv3_v2.json" % (name, split)))
            nb, _, _ = B.other_set_neighbors(name, "global", rows, topic_rows, emb_train, top_train,
                                             titles_train, tid, torch, device, tok, model)
        if nb.shape[0] != len(rows):
            sys.exit("GAGAL: jumlah tetangga != baris (%s %s)" % (name, split))
        g5_rows = P.read_jsonl(os.path.join(g5_dir, "%s_%s_retr_g5_v2.json" % (name, split)))
        out_rows, n_same, n_words = [], 0, 0
        for i, r in enumerate(rows):
            title, abstract = B.split_src(r["src"])
            groups = grouped(nb[i], train_raw_kps)
            # gerbang: keyphrase yang dipakai harus sama dengan data g5
            flat = [kp for _, kps in groups for kp in kps]
            if B.new_src(title, flat, abstract) == g5_rows[i]["src"]:
                n_same += 1
            ctx = context(groups, titles_train, a.fmt)
            n_words += len(ctx.split())
            new = dict(r)
            new["src"] = B.new_src(title, [ctx] if ctx else [], abstract)
            out_rows.append(new)
        out_path = os.path.join(out_dir, "%s_%s_retr_%s_v2.json" % (name, split, a.tag))
        B.write_jsonl(out_path, out_rows)
        back = P.read_jsonl(out_path)
        assert len(back) == len(rows) and all(b["tgt"] == r["tgt"] for b, r in zip(back, rows))
        n = len(rows)
        R("%s_%s: %d dok | kata konteks/dok %.1f | keyphrase sama dengan g5: %d/%d (%.2f%%) | md5 %s"
          % (name, split, n, n_words / n, n_same, n, 100.0 * n_same / n, B.md5(out_path)))
        R("   contoh src: " + out_rows[0]["src"][:400])
        if n_same < n * 0.99:
            R("   PERINGATAN: keyphrase berbeda dari g5 di >1% baris - JANGAN dipakai")

    with open(os.path.join(out_dir, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    P.log("SELESAI", out_dir)


if __name__ == "__main__":
    main()
