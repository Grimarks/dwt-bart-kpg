# -*- coding: utf-8 -*-
"""Recreate nbr_<split>_random.npy (control neighbours) bit-identically without embeddings or a GPU.

Original notes:
make_random_nbr.py  - membuat ulang tetangga ACAK (kontrol) persis seperti
retrieval_pilot_prep.py, tanpa embedding dan tanpa GPU.

retrieval_pilot_prep.py memakai np.random.default_rng(2026) dan memanggil
random_nbr() berurutan untuk train, valid, test. Skrip ini mengulang urutan
yang sama, jadi hasilnya identik bit-per-bit di mesin mana pun.

  - Jika nbr_<split>_random.npy SUDAH ada (PC3): dibandingkan, cetak IDENTIK / BEDA.
  - Jika BELUM ada (PC4): ditulis.
"""
import os

import numpy as np

import retrieval_pilot_prep as P


def count_rows(path):
    n = 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    os.makedirs(P.OUT, exist_ok=True)
    sizes = {s: count_rows(P.find_file("ori", s)) for s in P.SPLITS}
    P.log("jumlah dokumen:", sizes)
    rng = np.random.default_rng(P.SEED)
    for s in P.SPLITS:  # urutan sama dengan retrieval_pilot_prep.py
        nb = P.random_nbr(sizes[s], sizes["train"], s == "train", rng)
        p = os.path.join(P.OUT, f"nbr_{s}_random.npy")
        if os.path.exists(p):
            old = np.load(p)
            same = old.shape == nb.shape and bool((old == nb).all())
            P.log(f"{s}: file sudah ada -> {'IDENTIK' if same else 'BEDA'}")
        else:
            np.save(p, nb)
            P.log(f"{s}: ditulis {nb.shape}")


if __name__ == "__main__":
    main()
