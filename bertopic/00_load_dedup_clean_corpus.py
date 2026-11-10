"""Step 0: load the raw corpora, drop duplicate abstracts (662,405 -> 661,852 documents), clean the text
and save all_dataset.parquet (+ src_texts.txt), the historical corpus that 02_build_corpora.py reads.

This file contains notebook cells copied VERBATIM, in the order they were executed:
  bertopic-arxiv (1).ipynb  cell 7  (load CSV, arXiv TXT, PubMed TXT, arXiv JSON; drop_duplicates on abstract)
                            cell 9  (empty present/absent columns)
                            cell 13 (timer helper)
                            cell 15 (text cleaning: newlines, brackets spaced, control chars, whitespace)
                            cell 18 (drop unused columns)
                            cell 21 (save data_arxiv.pkl)
  bertopic-arxivvv.ipynb    cell 15 (load data_arxiv.pkl)
                            cell 11 (second cleaning pass; write src_texts.txt and all_dataset.parquet)
Only the CONFIG block below was added (values from cell 3 of both notebooks; paths via DWTBART_ROOT)
plus a no-op cek() helper used by the last cell. The saved outputs of cell 7 show the historical run:
CSV 136,238, arXiv TXT 50,000/6,436/6,440, PubMed TXT 50,000/6,633/6,658, JSON 400,000,
Dedup 662,405 -> 661,852. Note: the raw files may have changed since that run (e.g. the arXiv
JSON snapshot is updated by its publisher), so a rebuild can differ from the historical corpus.
"""
import os, gc, pickle
import pandas as pd

# ---- CONFIG (from cell 3; set DWTBART_ROOT to the research root, see README 'Folder layout') ----
ROOT = os.environ.get("DWTBART_ROOT", ".")
SAMPLE_MODE   = False
SAMPLE_SIZE   = 1_000
USE_DATASET   = 'all'
JSON_MAX_ROWS = 400_000
TXT_MAX_ROWS  = 50_000
BASE_INPUT = os.path.join(ROOT, 'bertopic_project_wsl_original', 'dataset', 'Dataset_Keyphrase')
JSON_PATH  = os.path.join(BASE_INPUT, 'arxiv-metadata-oai-snapshot.json')
CSV_PATH   = os.path.join(BASE_INPUT, 'arXiv Scientific Research Paper Dataset', 'arXiv_scientific dataset.csv')
ARXIV_TXT  = {s: os.path.join(BASE_INPUT, 'arxiv-dataset', f'{s}.txt') for s in ['train','val','test']}
PUBMED_TXT = {s: os.path.join(BASE_INPUT, 'pubmed-dataset', f'{s}.txt') for s in ['train','val','test']}
DATA_DIR   = os.path.join(ROOT, 'bertopic_project_wsl_original', 'output', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
def cek(label=''):   # memory print-out in the notebook; no-op here
    pass

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 7 ========
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import gc

cols   = ['title', 'abstract', 'type']
frames = []


def load_txt_split(path, split_label, sample_size=None):
    """
    Load satu file .txt dari armanc/scientific_papers.
    Baca satu baris per iterasi — aman untuk file 14GB sekalipun.
    """
    if not os.path.exists(path):
        print(f" File tidak ditemukan: {path} — skip")
        return pd.DataFrame(columns=cols)

    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if sample_size and len(lines) >= sample_size:
                break

    df = pd.DataFrame({
        'abstract': lines,
        'title'   : '',
        'type'    : split_label,
    })
    return df[cols]


# ── SUMBER 1: CSV Lokal ────────────────────────────────────────────────────────
if USE_DATASET in ('csv', 'all'):
    print("Loading CSV lokal (arXiv Scientific Research Paper Dataset)...")
    if not os.path.exists(CSV_PATH):
        print(f" File tidak ditemukan: {CSV_PATH} — skip")
    else:
        df_raw = pd.read_csv(CSV_PATH, usecols=['title', 'summary'], low_memory=False)
        df_csv = df_raw.rename(columns={'summary': 'abstract'}).dropna(subset=['title', 'abstract'])
        df_csv.reset_index(drop=True, inplace=True)

        if SAMPLE_MODE:
            df_csv = df_csv.sample(n=min(SAMPLE_SIZE, len(df_csv)), random_state=42)

        tr, tmp = train_test_split(df_csv, test_size=0.2, random_state=42)
        te, va  = train_test_split(tmp, test_size=0.5, random_state=42)
        for df, t in [(tr, 'train'), (te, 'test'), (va, 'validation')]:
            df = df.copy()
            df['type'] = t
            frames.append(df[cols])
        print(f"  CSV: {len(df_csv):,} baris")
        del df_raw, df_csv, tr, tmp, te, va; gc.collect()

# ── SUMBER 2: arxiv split (txt) ───────────────────────────────────────────────
if USE_DATASET in ('arxiv', 'all'):
    print("Loading arxiv-dataset txt...")
    sample_per_split = SAMPLE_SIZE if SAMPLE_MODE else TXT_MAX_ROWS
    for split_file, split_label in [
        (ARXIV_TXT['train'], 'train'),
        (ARXIV_TXT['val'],   'validation'),
        (ARXIV_TXT['test'],  'test'),
    ]:
        df = load_txt_split(split_file, split_label, sample_per_split)
        if not df.empty:
            frames.append(df)
            print(f"  arxiv {split_label}: {len(df):,} baris")
        del df; gc.collect()

# ── SUMBER 3: pubmed split (txt) ──────────────────────────────────────────────
if USE_DATASET in ('pubmed', 'all'):
    print("Loading pubmed-dataset txt...")
    sample_per_split = SAMPLE_SIZE if SAMPLE_MODE else TXT_MAX_ROWS
    for split_file, split_label in [
        (PUBMED_TXT['train'], 'train'),
        (PUBMED_TXT['val'],   'validation'),
        (PUBMED_TXT['test'],  'test'),
    ]:
        df = load_txt_split(split_file, split_label, sample_per_split)
        if not df.empty:
            frames.append(df)
            print(f"  pubmed {split_label}: {len(df):,} baris")
        del df; gc.collect()

# ── SUMBER 4: arxiv-metadata-oai-snapshot.json (JSONL) ───────────────────────
if USE_DATASET in ('json', 'all'):
    print("Loading arxiv-metadata-oai-snapshot.json (streaming batch)...")
    if not os.path.exists(JSON_PATH):
        print(f" File tidak ditemukan: {JSON_PATH} — skip")
    else:
        BATCH_SIZE = 50_000
        sample_n   = SAMPLE_SIZE if SAMPLE_MODE else JSON_MAX_ROWS

        batch_dir = os.path.join(DATA_DIR, 'json_batches')
        os.makedirs(batch_dir, exist_ok=True)

        chunks  = pd.read_json(JSON_PATH, lines=True, chunksize=BATCH_SIZE)
        total   = 0
        batch_i = 0

        for chunk in tqdm(chunks, desc="Streaming JSON"):
            chunk = chunk[['title', 'abstract']].dropna()
            chunk = chunk[chunk['abstract'].str.strip() != '']
            chunk.to_parquet(os.path.join(batch_dir, f'batch_{batch_i:04d}.parquet'), index=False)
            total   += len(chunk)
            batch_i += 1
            if sample_n and total >= sample_n:
                break

        df_json = pd.concat(
            [pd.read_parquet(os.path.join(batch_dir, f))
             for f in sorted(os.listdir(batch_dir))],
            ignore_index=True
        )
        if sample_n:
            df_json = df_json.iloc[:sample_n]

        shutil.rmtree(batch_dir)

        tr, tmp = train_test_split(df_json, test_size=0.2, random_state=42)
        te, va  = train_test_split(tmp,     test_size=0.5, random_state=42)
        for df, t in [(tr, 'train'), (te, 'test'), (va, 'validation')]:
            df = df.copy(); df['type'] = t
            frames.append(df[cols])
        print(f"  JSON snapshot: {len(df_json):,} baris ({batch_i} batch)")
        del df_json, tr, tmp, te, va; gc.collect()

# ── Gabungkan semua sumber ────────────────────────────────────────────────────
if not frames:
    raise ValueError("Tidak ada dataset yang berhasil dimuat. Periksa path dan USE_DATASET.")

all_dataset = pd.concat(frames, ignore_index=True)
del frames; gc.collect()

all_dataset.dropna(subset=['abstract'], inplace=True)
all_dataset = all_dataset[all_dataset['abstract'].str.strip() != '']
all_dataset.reset_index(drop=True, inplace=True)

# Deduplication
sebelum = len(all_dataset)
all_dataset = all_dataset.drop_duplicates(subset=['abstract']).reset_index(drop=True)
print(f"Dedup: {sebelum:,} → {len(all_dataset):,} baris ({sebelum - len(all_dataset):,} duplikat dihapus)")

print(f"\nTotal data: {len(all_dataset):,} baris")
print(all_dataset['type'].value_counts())

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 9 ========
# arXiv tidak punya keyphrase berlabel — kolom dikosongkan
all_dataset['present'] = ''
all_dataset['absent']  = ''

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 13 ========
import time
from datetime import datetime, timedelta

class Timer:
    def __init__(self):
        self.start_total = time.time()
        self.stages = {}
        self._current = None
        self._current_start = None

    def mulai(self, nama):
        if self._current:
            self.selesai()
        self._current = nama
        self._current_start = time.time()
        print(f"\n [{datetime.now().strftime('%H:%M:%S')}] Mulai: {nama}")

    def selesai(self):
        if not self._current:
            return
        elapsed = time.time() - self._current_start
        self.stages[self._current] = elapsed
        print(f" Selesai: {self._current} — {self._format(elapsed)}")
        self._current = None

    def ringkasan(self):
        total = time.time() - self.start_total
        print(f"\n{'='*55}")
        print(f"  RINGKASAN WAKTU")
        print(f"{'='*55}")
        for nama, durasi in self.stages.items():
            pct = durasi / total * 100
            bar = '█' * int(pct / 3)
            print(f"  {nama:<30} {self._format(durasi):>10}  {bar} {pct:.1f}%")
        print(f"{'─'*55}")
        print(f"  {'TOTAL':<30} {self._format(total):>10}")
        print(f"{'='*55}")

    @staticmethod
    def _format(detik):
        if detik < 60:
            return f"{detik:.1f} detik"
        elif detik < 3600:
            return f"{detik/60:.1f} menit"
        else:
            jam  = int(detik // 3600)
            mnt  = int((detik % 3600) // 60)
            return f"{jam}j {mnt}m"

timer = Timer()

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 15 ========
timer.mulai("Text Cleaning")

import re

_re_bracket = re.compile(r'([\(\)\[\]\{\}<>])')
_re_ctrl    = re.compile(r'[\x00-\x1F\x7F-\x9F]')
_re_space   = re.compile(r'\s+')

def clean_text(value: str) -> str:
    value = value.replace('\n', ' ')
    value = _re_bracket.sub(r' \1 ', value)
    value = _re_ctrl.sub('', value)
    value = _re_space.sub(' ', value).strip()
    return value

for col in ['title', 'abstract']:
    all_dataset[col] = all_dataset[col].astype(str).apply(clean_text)

print(f"Text cleaning selesai — {len(all_dataset):,} dokumen")
timer.selesai()

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 18 ========
# Hapus kolom tidak perlu sebelum proses
cols_to_drop = [c for c in all_dataset.columns if c not in ['title', 'abstract', 'type', 'present', 'absent']]
all_dataset.drop(columns=cols_to_drop, inplace=True)
gc.collect()

# ======== verbatim: bertopic-arxiv (1).ipynb, cell 21 ========
import pickle

data_pkl_path = os.path.join(DATA_DIR, "data_arxiv.pkl")
with open(data_pkl_path, "wb") as f:
    pickle.dump({"all_dataset": all_dataset}, f)
print(f"Data disimpan ke: {data_pkl_path}")

# ======== verbatim: bertopic-arxivvv.ipynb, cell 15 ========
import pickle
import pandas as pd
import gc

# Load all_dataset dari pkl (data asli sebelum cleaning)
data_pkl_path = os.path.join(DATA_DIR, 'data_arxiv.pkl')

# Cek apakah pkl ada
if os.path.exists(data_pkl_path):
    print("Loading all_dataset dari pkl...")
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)
    all_dataset = data['all_dataset']
    del data; gc.collect()
    print(f"  all_dataset: {len(all_dataset):,} baris")
else:
    print("❌ pkl tidak ada — perlu load ulang dari Cell 4")

# ======== verbatim: bertopic-arxivvv.ipynb, cell 11 ========
import re

_ds_path  = os.path.join(DATA_DIR, 'all_dataset.parquet')
_txt_path = os.path.join(DATA_DIR, 'src_texts.txt')

if os.path.exists(_ds_path) and os.path.exists(_txt_path):
    print('Checkpoint ada — skip cleaning.')
else:
    timer.mulai('Cleaning + simpan')

    _re_ctrl  = re.compile(r'[\x00-\x1F\x7F-\x9F]')
    _re_space = re.compile(r'\s+')

    def clean_series(s):
        s = s.astype(str).str.replace('\n', ' ', regex=False)
        s = s.str.replace(_re_ctrl, '', regex=True)
        s = s.str.replace(_re_space, ' ', regex=True).str.strip()
        return s

    CHUNK   = 20_000   # ← lebih kecil, RAM lebih aman
    n_lines = 0

    # Cleaning + tulis txt sekaligus per chunk — tidak pernah cleaning 661k sekaligus
    with open(_txt_path, 'w', encoding='utf-8', buffering=1<<20) as fout:
        for start in range(0, len(all_dataset), CHUNK):
            chunk = all_dataset.iloc[start:start+CHUNK].copy()

            chunk['title']    = clean_series(chunk['title'])
            chunk['abstract'] = clean_series(chunk['abstract'])

            lines = (chunk['title'] + ' ' + chunk['abstract']).str.strip()
            valid = [l for l in lines.tolist() if l]
            fout.write('\n'.join(valid) + '\n')
            n_lines += len(valid)

            # Update all_dataset langsung
            all_dataset.iloc[start:start+CHUNK, all_dataset.columns.get_loc('title')]    = chunk['title'].values
            all_dataset.iloc[start:start+CHUNK, all_dataset.columns.get_loc('abstract')] = chunk['abstract'].values

            del chunk, lines, valid; gc.collect()
            print(f'  Chunk {start//CHUNK + 1}/{(len(all_dataset)-1)//CHUNK + 1} selesai')

    # Simpan parquet setelah semua chunk selesai
    all_dataset.to_parquet(_ds_path, index=False)
    print(f'Checkpoint disimpan : {_ds_path}')
    print(f'src_texts.txt       : {n_lines:,} baris')
    timer.selesai()
    cek('setelah cleaning')
