# -*- coding: utf-8 -*-
r"""Analysis numbers of the paper: input truncation, same-topic search space, topic agreement of global
neighbours, retrieval time (global vs within-topic cosine top-5), GPU hours, and the leakage-sensitivity
subsets of KP20k by similarity of the nearest training document (--mode leak_prep -> spec for
paired_bootstrap.py; results in Kaggle stats/ijies_v7/).

Original notes:
ijies_angka_v7.py (PC3, CPU) - angka untuk naskah IJIES.
Hanya MEMBACA data/hipotesis yang sudah ada; menulis HANYA ke --res (default output\ijies_v7). Training GPU tidak disentuh.

--mode facts  (~5-15 menit):
  A. inventaris: spec_retr9.json (sistem KP20k), file .npy di retrieval_pilot (bentuk, dtype), file topik KP20k
  B. pemotongan input: berapa input ori / st5 / st5t / g5 / g5t yang > max_source_length token BART (tokenizer dari folder run)
  C. ruang pencarian: rata-rata bagian data train KP20k yang dicari bila dibatasi satu topik (test & valid)
  D. tetangga global (g5) yang satu topik dengan dokumen sumber (kalau file indeks tetangga global ketemu)
  E. waktu pencarian: cosine top-5 global vs di dalam topik (numpy, 2.000 kueri)
  F. GPU hours dari trainer_state.json semua run KP20k di PC3; versi pustaka; file arXiv di disk
--mode leak_prep : subset uji KP20k menurut kemiripan dokumen train teratas (sim_test_global.npy kolom 0)
                   (+ subset "semua 5 tetangga global satu topik" vs "ada yang lintas topik" bila D berhasil)
                   -> spec_leak.json untuk paired_bootstrap.py
--mode leak_report: laporan dari leak_mean_std.csv + leak_results_summary.csv
"""
import argparse, csv, glob, json, os, re, sys, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["facts", "leak_prep", "leak_report"], required=True)
ap.add_argument("--out", default=os.path.join(os.environ.get("DWTBART_ROOT", "."), "output"))
ap.add_argument("--pkg", default=os.path.join(os.environ.get("DWTBART_ROOT", "."), "PC1_PC2_FINAL_PACKAGE"))
ap.add_argument("--res", default=os.path.join(os.environ.get("DWTBART_ROOT", "."), "output", "ijies_v7"))
a = ap.parse_args()
os.makedirs(a.res, exist_ok=True)
RP = os.path.join(a.out, "retrieval_pilot")
KP = os.path.join(a.pkg, "dataset", "kp20k_v3")
TA = os.path.join(a.pkg, "dataset", "topic_assignments")
COND = ["ori", "st5", "st5t", "g5", "g5t"]
LOG = []


def say(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    LOG.append(s)


def save(name):
    with open(os.path.join(a.res, name), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return f.readlines()


def find_spec():
    best = []
    for root, dirs, files in os.walk(a.out):
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-") and d != "perdoc_cache" and not d.startswith("xfer_")]
        for fn in files:
            if re.match(r"^spec_retr\d*\.json$", fn):
                best.append(os.path.join(root, fn))
    nine = [p for p in best if os.path.basename(p) == "spec_retr9.json"]
    return (nine or sorted(best, key=os.path.getmtime))[-1] if best else None


def kp20k_systems():
    sp = find_spec()
    spec = json.load(open(sp, encoding="utf-8"))
    sysm = defaultdict(dict)
    for s in spec["systems"]:
        if s["dataset"] == "kp20k" and s["condition"] in COND:
            sysm[s["condition"]][int(s["seed"])] = s["hyp"]
    return sp, spec, sysm


def topic_file(split):
    cands = [p for p in glob.glob(os.path.join(TA, "kp20k*")) if p.lower().endswith(".csv")]
    key = {"train": "train", "valid": "val", "test": "test"}[split]
    c = [p for p in cands if key in os.path.basename(p).lower()]
    v3 = [p for p in c if "v3" in os.path.basename(p).lower()]
    c = v3 or c
    return sorted(c)[0] if c else None


def load_topics(split):
    p = topic_file(split)
    if not p:
        return None, None
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        col = "topic" if "topic" in r.fieldnames else [c for c in r.fieldnames if "topic" in c.lower()][0]
        t = [int(float(row[col])) for row in r]
    return p, t


def find_global_nbr(n_test):
    import numpy as np
    for p in sorted(glob.glob(os.path.join(RP, "*.npy"))):
        b = os.path.basename(p).lower()
        if "test" in b and "global" in b and "sim" not in b:
            m = np.load(p, mmap_mode="r")
            if m.ndim == 2 and m.shape[0] == n_test and m.dtype.kind in "iu":
                return p
    return None


# ------------------------------------------------------------------ facts
if a.mode == "facts":
    import numpy as np
    t0 = time.time()
    say("=== A. inventaris ===")
    sp, spec, sysm = kp20k_systems()
    say("spec:", sp)
    for c in COND:
        say("  %-5s seeds %s" % (c, sorted(sysm.get(c, {}))))
    ref = spec["references"]["kp20k"]
    n_test = len(read_lines(ref))
    say("referensi kp20k:", ref, n_test, "baris")
    say("file .npy di retrieval_pilot:")
    for p in sorted(glob.glob(os.path.join(RP, "*.npy"))):
        try:
            m = np.load(p, mmap_mode="r")
            say("  %-40s %-18s %s  %.0f MB" % (os.path.basename(p), str(m.shape), m.dtype, os.path.getsize(p) / 1e6))
        except Exception as e:
            say("  %-40s gagal dibaca: %s" % (os.path.basename(p), e))
    say("file topik KP20k:", [os.path.basename(p) for p in glob.glob(os.path.join(TA, "kp20k*"))])
    for fn in ("build_st5_cpu.py", "build_retrieval_data.py", "build_title_ctx_v2.py"):
        for p in glob.glob(os.path.join(RP, fn)) + glob.glob(os.path.join(a.pkg, "**", fn), recursive=True)[:1]:
            say("--- baris berisi path/aturan di", p)
            for ln in read_lines(p):
                if re.search(r"\.npy|\.csv|topic|0\.95|max_ctx|30|title", ln) and len(ln) < 220:
                    say("   ", ln.rstrip())
            break

    say("")
    say("=== B. pemotongan input (> max_source_length token) ===")
    scr = os.path.join(a.pkg, "scripts", "kpg_training", "run_finetune_kpgen_seq2seq_hf.py")
    maxlen = 1024
    if os.path.exists(scr):
        for ln in read_lines(scr):
            if "max_source_length" in ln and ("default" in ln or "add_argument" in ln or "field(" in ln):
                say("  skrip:", ln.strip()[:200])
                mm = re.search(r"default\s*=\s*(\d+)", ln)
                if mm:
                    maxlen = int(mm.group(1))
            if re.search(r"add_tokens|additional_special_tokens", ln):
                say("  skrip (token):", ln.strip()[:200])
    say("  max_source_length dipakai untuk hitungan:", maxlen)
    from transformers import AutoTokenizer
    tok = None
    for c in ("st5t", "ori"):
        for s in sorted(sysm.get(c, {})):
            d = os.path.dirname(sysm[c][s])
            for cand in (d, os.path.dirname(d)):
                if os.path.exists(os.path.join(cand, "tokenizer_config.json")):
                    tok = AutoTokenizer.from_pretrained(cand)
                    say("  tokenizer dari:", cand, "| vocab", len(tok))
                    break
            if tok:
                break
        if tok:
            break
    if tok is None:
        tok = AutoTokenizer.from_pretrained("facebook/bart-base")
        say("  tokenizer: facebook/bart-base (folder run tanpa tokenizer)")
    trunc = {}
    for cond in COND:
        dd = os.path.join(KP, "ori" if cond == "ori" else "retr_" + cond)
        if not os.path.isdir(dd):
            say("  folder tidak ada:", dd)
            continue
        for p in sorted(glob.glob(os.path.join(dd, "*.json"))):
            b = os.path.basename(p).lower()
            if "train" in b and cond not in ("st5t", "g5t"):
                continue
            srcs = []
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        srcs.append(json.loads(ln)["src"])
            n_long, lens = 0, []
            for i in range(0, len(srcs), 2000):
                enc = tok(srcs[i:i + 2000], add_special_tokens=True, truncation=False)["input_ids"]
                for e in enc:
                    lens.append(len(e))
                    n_long += len(e) > maxlen
            lens.sort()
            say("  %-5s %-45s n=%7d  >%d: %6d (%.2f%%)  median %d  p95 %d" % (
                cond, os.path.basename(p), len(srcs), maxlen, n_long, 100.0 * n_long / max(1, len(srcs)),
                lens[len(lens) // 2], lens[int(0.95 * (len(lens) - 1))]))
            trunc[(cond, os.path.basename(p))] = (n_long, len(srcs))
    save("facts.txt")

    say("")
    say("=== C. ruang pencarian di dalam topik ===")
    ptr, ttr = load_topics("train")
    say("  topik train:", ptr, None if ttr is None else len(ttr))
    if ttr:
        N = len(ttr)
        cnt = Counter(ttr)
        say("  topik unik di train: %d | terbesar %d (%.1f%%) | terkecil %d" % (len(cnt), max(cnt.values()), 100.0 * max(cnt.values()) / N, min(cnt.values())))
        for split in ("test", "valid"):
            pp, tt = load_topics(split)
            if not tt:
                say("  topik", split, "tidak ada")
                continue
            fr = sorted(cnt[t] / N for t in tt)
            say("  %-5s %s n=%d | bagian train yang dicari: rata-rata %.2f%% (median %.2f%%, min %.3f%%, maks %.2f%%) -> %.1fx lebih sedikit dari global" % (
                split, os.path.basename(pp), len(tt), 100 * sum(fr) / len(fr), 100 * fr[len(fr) // 2], 100 * fr[0], 100 * fr[-1], len(fr) / sum(fr)))
    save("facts.txt")

    say("")
    say("=== D. tetangga global (g5) yang satu topik ===")
    gp = find_global_nbr(n_test)
    simp = os.path.join(RP, "sim_test_global.npy")
    pte, tte = load_topics("test")
    if gp and ttr and tte and os.path.exists(simp):
        idx = np.load(gp)
        sim = np.load(simp).astype(np.float32)
        say("  indeks:", gp, idx.shape, "| sim:", sim.shape, "| sim urut turun:", bool((sim[:, 0] >= sim[:, -1]).all()))
        ttr_a = np.array(ttr)
        same_cnt = []
        n_same_tot = n_tot = 0
        for i in range(idx.shape[0]):
            keep = [j for j, s in zip(idx[i], sim[i]) if s < 0.95][:5]
            same = sum(1 for j in keep if ttr_a[j] == tte[i])
            same_cnt.append(same)
            n_same_tot += same
            n_tot += len(keep)
        c5 = Counter(same_cnt)
        say("  dari %d tetangga global top-5 (setelah buang cos>=0,95): %.1f%% satu topik dengan dokumen sumber" % (n_tot, 100.0 * n_same_tot / n_tot))
        say("  dokumen uji menurut jumlah tetangga satu topik (0..5):", [c5.get(k, 0) for k in range(6)])
        say("  -> semua 5 satu topik: %d (%.1f%%); minimal 1 lintas topik: %d" % (c5.get(5, 0), 100.0 * c5.get(5, 0) / len(same_cnt), len(same_cnt) - c5.get(5, 0)))
        np.save(os.path.join(a.res, "same_topic_count_test.npy"), np.array(same_cnt, dtype=np.int8))
        say("  (catatan: aturan judul sama tidak diulang di sini, jadi angka ini perkiraan)")
    else:
        say("  dilewati: indeks tetangga global =", gp, "| topik train/test ada:", bool(ttr), bool(tte), "| sim:", os.path.exists(simp))
    save("facts.txt")

    say("")
    say("=== E. waktu pencarian (cosine, numpy CPU) ===")
    ep = os.path.join(RP, "emb_train.npy")
    if os.path.exists(ep) and ttr:
        E = np.load(ep).astype(np.float32)
        if E.shape[0] == len(ttr):
            qp = [p for p in glob.glob(os.path.join(RP, "emb_test*.npy"))]
            rng = np.random.default_rng(0)
            if qp and tte:
                Q = np.load(qp[0]).astype(np.float32)
                qi = rng.choice(Q.shape[0], 2000, replace=False)
                Qs, qt = Q[qi], [tte[i] for i in qi]
                say("  kueri: 2.000 dokumen uji dari", os.path.basename(qp[0]))
            else:
                qi = rng.choice(E.shape[0], 2000, replace=False)
                Qs, qt = E[qi], [ttr[i] for i in qi]
                say("  kueri: 2.000 dokumen train acak (emb_test tidak ada)")
            ttr_a = np.array(ttr)
            groups = {t: np.where(ttr_a == t)[0] for t in set(qt)}
            sub = {t: E[ix] for t, ix in groups.items()}
            t1 = time.perf_counter()
            for i in range(0, len(Qs), 100):
                S = Qs[i:i + 100] @ E.T
                np.argpartition(-S, 6, axis=1)[:, :6]
            tg = time.perf_counter() - t1
            t1 = time.perf_counter()
            for q, t in zip(Qs, qt):
                S = sub[t] @ q
                np.argpartition(-S, 6)[:6]
            tt_ = time.perf_counter() - t1
            say("  global (batch 100): %.2f ms per kueri | di dalam topik (satu per satu): %.2f ms per kueri | matriks %s" % (1000 * tg / len(Qs), 1000 * tt_ / len(Qs), E.shape))
        else:
            say("  emb_train baris", E.shape[0], "!= topik train", len(ttr), "-> dilewati")
    else:
        say("  dilewati (emb_train.npy atau topik train tidak ada)")
    save("facts.txt")

    say("")
    say("=== F. GPU hours run KP20k (trainer_state.json) ===")
    rows = []
    for root, dirs, files in os.walk(a.out):
        b = os.path.basename(root)
        dirs[:] = [d for d in dirs if d != "perdoc_cache" and not d.startswith("xfer_") and not d.startswith("checkpoint-")]
        if "_kp20k_" not in b or "trainer_state.json" not in files:
            continue
        try:
            st = json.load(open(os.path.join(root, "trainer_state.json"), encoding="utf-8"))
            rt = [h.get("train_runtime") for h in st.get("log_history", []) if "train_runtime" in h]
            rows.append((b, st.get("global_step"), rt[-1] / 3600 if rt else None))
        except Exception:
            rows.append((b, None, None))
    for b, gs, h in sorted(rows):
        say("  %-95s step %-6s %s" % (b[:95], gs, "%.2f h" % h if h else "-"))
    hs = [h for _, gs, h in rows if h and gs and gs >= 80000 and h > 15]
    if hs:
        say("  run lengkap tanpa resume (>15 h): %d | rata-rata %.1f h | min %.1f | maks %.1f" % (len(hs), sum(hs) / len(hs), min(hs), max(hs)))
    say("")
    say("=== G. versi pustaka PC3 ===")
    say("  python", sys.version.split()[0])
    for m in ("torch", "transformers", "datasets", "accelerate", "nltk", "numpy", "sentence_transformers", "sklearn", "bertopic", "umap", "kneed"):
        try:
            mod = __import__(m)
            say("  %-22s %s" % (m, getattr(mod, "__version__", "?")))
        except Exception:
            say("  %-22s (tidak terpasang)" % m)
    say("")
    say("=== H. file arXiv di root riset (nama, ukuran, tanggal) ===")
    for root, dirs, files in os.walk(os.path.dirname(a.out)):
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-") and d not in ("venv", "perdoc_cache", ".git")]
        for fn in files:
            if "arxiv" in fn.lower():
                p = os.path.join(root, fn)
                say("  %s  %.0f MB  %s" % (p, os.path.getsize(p) / 1e6, time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(p)))))
    say("")
    say("selesai facts dalam %.1f menit" % ((time.time() - t0) / 60))
    save("facts.txt")

# ------------------------------------------------------------------ leak_prep
elif a.mode == "leak_prep":
    import numpy as np
    sp, spec, sysm = kp20k_systems()
    ref = spec["references"]["kp20k"]
    R = read_lines(ref)
    n = len(R)
    sim = np.load(os.path.join(RP, "sim_test_global.npy")).astype(np.float32)
    assert sim.shape[0] == n, (sim.shape, n)
    top = sim[:, 0]
    subsets = {
        "kp20k_lt070": top < 0.70,
        "kp20k_070_080": (top >= 0.70) & (top < 0.80),
        "kp20k_080_090": (top >= 0.80) & (top < 0.90),
        "kp20k_ge090": top >= 0.90,
        "kp20k_clean_lt090": top < 0.90,
        "kp20k_strict_lt080": top < 0.80,
    }
    scp = os.path.join(a.res, "same_topic_count_test.npy")
    if os.path.exists(scp):
        sc = np.load(scp)
        if sc.shape[0] == n:
            subsets["kp20k_nbr_all_same_topic"] = sc == 5
            subsets["kp20k_nbr_cross_topic"] = sc < 5
    bdir = os.path.join(a.res, "subsets")
    os.makedirs(bdir, exist_ok=True)
    out = {"references": {}, "systems": []}
    say("subset uji KP20k (cosine dokumen train paling mirip, sim_test_global kolom 0):")
    for name, mask in subsets.items():
        rows = np.where(mask)[0].tolist()
        say("  %-26s %6d dok (%.1f%%)" % (name, len(rows), 100.0 * len(rows) / n))
        if len(rows) < 30:
            say("    terlalu kecil, dilewati")
            continue
        rp = os.path.join(bdir, name + "_test_ori.json")
        with open(rp, "w", encoding="utf-8", newline="") as f:
            f.writelines(R[i] for i in rows)
        with open(os.path.join(bdir, name + "_rows.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, rows)) + "\n")
        out["references"][name] = rp
        for c in ("ori", "st5t", "g5", "g5t"):
            for s, hp in sorted(sysm.get(c, {}).items()):
                H = read_lines(hp)
                assert len(H) == n, (c, s, len(H), n)
                pp = os.path.join(bdir, "%s_%s_seed%d_hypotheses.txt" % (name, c, s))
                with open(pp, "w", encoding="utf-8", newline="") as f:
                    f.writelines(H[i] for i in rows)
                out["systems"].append({"condition": c, "seed": s, "dataset": name, "hyp": pp})
    with open(os.path.join(a.res, "spec_leak.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    say("spec_leak.json:", len(out["systems"]), "sistem")
    save("leak_prep.txt")

# ------------------------------------------------------------------ leak_report
else:
    ms = list(csv.DictReader(open(os.path.join(a.res, "leak_mean_std.csv"), encoding="utf-8")))
    summ = list(csv.DictReader(open(os.path.join(a.res, "leak_results_summary.csv"), encoding="utf-8")))
    M = {(r["dataset"], r["condition"], r["tag"], r["metric"]): r for r in ms}
    sizes = {}
    for p in glob.glob(os.path.join(a.res, "subsets", "*_rows.txt")):
        sizes[os.path.basename(p)[:-9]] = len(open(p).read().split())
    say("Sensitivitas kebocoran KP20k, Macro F1@M x100 (rata-rata 3 seed); bootstrap gabungan seed, CI 95%")
    ORDER = ["kp20k_lt070", "kp20k_070_080", "kp20k_080_090", "kp20k_ge090", "kp20k_clean_lt090", "kp20k_strict_lt080",
             "kp20k_nbr_all_same_topic", "kp20k_nbr_cross_topic"]
    for ds in [d for d in ORDER if d in set(r["dataset"] for r in ms)]:
        say("")
        say("=== %s (n=%s) ===" % (ds, sizes.get(ds, "?")))
        for tag in ("absent", "present"):
            say("  %-7s " % tag + "  ".join("%s %.2f" % (c, 100 * float(M[(ds, c, tag, "F1@M")]["mean"])) for c in ("ori", "st5t", "g5", "g5t") if (ds, c, tag, "F1@M") in M))
        for r in summ:
            if r["dataset"] != ds or r["metric"] != "F1@M" or r["tag"] not in ("absent", "present") or not r["seed"].startswith("mean("):
                continue
            say("  %-5s - %-5s %-7s %+6.2f [%+6.2f; %+6.2f] p=%.4f %s" % (r["system_A"], r["system_B"], r["tag"], 100 * float(r["diff_A_minus_B"]),
                100 * float(r["ci95_lo"]), 100 * float(r["ci95_hi"]), float(r["p_value"]), "*" if r["significant_0.05"] == "True" else ""))
    # cek: rata-rata berbobot 4 bin = skor penuh
    bins = ["kp20k_lt070", "kp20k_070_080", "kp20k_080_090", "kp20k_ge090"]
    if all(b in sizes for b in bins):
        say("")
        for c in ("ori", "st5t", "g5t"):
            try:
                w = sum(sizes[b] * float(M[(b, c, "absent", "F1@M")]["mean"]) for b in bins) / sum(sizes[b] for b in bins)
                say("cek: absent F1@M %s dari 4 bin = %.2f (harus sama dengan tabel: ori 5,02 / st5t 7,26 / g5t 7,96)" % (c, 100 * w))
            except KeyError:
                pass
    save("laporan_leak.txt")
