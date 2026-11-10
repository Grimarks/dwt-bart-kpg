"""Step 0 (KP20k): convert the KP20k JSON-lines files (train.json, valid.json, test.json of the
memray/kp20k release on Hugging Face) to kp20k_{train,validation,test}.parquet with columns
id, title, abstract, keywords (';'-joined). Ids are kp20k_<split>_<row> when the file has none.
Saved output of the historical run: train 514,154, validation 19,992, test 19,987 rows.

This is cell 5 of kp20k_topic.ipynb copied VERBATIM; only the CONFIG block was added (the notebook's
cell 3 defined DATA_DIR and cek()). There is no separate KP20k preparation script beyond this cell.
"""
import os, gc
import pandas as pd

# ---- CONFIG (set DWTBART_ROOT to the research root, see README 'Folder layout') ----
ROOT = os.environ.get("DWTBART_ROOT", ".")
DATA_DIR = os.path.join(ROOT, 'kp20k_project', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
def cek(label=''):   # memory print-out in the notebook; no-op here
    pass

# ======== verbatim: kp20k_topic.ipynb, cell 5 (only the 3 input paths -> CONFIG ROOT) ========
# ════════════════════════════════════════════════════════════════════
# Cell 2 — Load KP20k dari file JSON lokal & simpan ke parquet
# ════════════════════════════════════════════════════════════════════
split_map = {
    'train'      : os.path.join(ROOT, 'kp20k_project', 'datasets', 'train.json'),
    'validation' : os.path.join(ROOT, 'kp20k_project', 'datasets', 'valid.json'),
    'test'       : os.path.join(ROOT, 'kp20k_project', 'datasets', 'test.json'),
}

for split, json_path in split_map.items():
    out_path = os.path.join(DATA_DIR, f'kp20k_{split}.parquet')
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)/1e6
        print(f'✅ {split} sudah ada ({size:.1f} MB) — skip')
        continue

    print(f'Load {split} dari {json_path}...', flush=True)
    df = pd.read_json(json_path, lines=True)
    print(f'  Kolom: {df.columns.tolist()}', flush=True)

    # Tambah kolom id kalau tidak ada
    if 'id' not in df.columns:
        df.insert(0, 'id', [f'kp20k_{split}_{i}' for i in range(len(df))])

    # Ambil kolom yang dibutuhkan
    cols = ['id', 'title', 'abstract', 'keywords']
    df = df[cols].copy()
    df['title']    = df['title'].astype(str).str.strip()
    df['abstract'] = df['abstract'].astype(str).str.strip()
    df['keywords'] = df['keywords'].apply(
        lambda k: ';'.join(k) if isinstance(k, list) else str(k)
    )

    df.to_parquet(out_path, index=False)
    size = os.path.getsize(out_path)/1e6
    print(f'  ✅ {split}: {len(df):,} baris → {out_path} ({size:.1f} MB)')
    del df; gc.collect()

cek('setelah load KP20k')
