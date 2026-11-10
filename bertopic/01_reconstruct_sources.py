"""Step 1b (provenance only): reconstruct the source label (CSV / TXT-arXiv / TXT-PubMed / JSON) of every
corpus document from the raw files. The output reconstructed_with_source.parquet is used by
02_build_corpora.py for source attribution; it does not change the corpus.
The PubMed TXT files here are the scientific_papers PubMed split that is part of the BERTopic corpus.
"""
import os, gc, json, re
import pandas as pd

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')
BASE = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'dataset', 'Dataset_Keyphrase')
CSV_PATH  = os.path.join(BASE, 'arXiv Scientific Research Paper Dataset', 'arXiv_scientific dataset.csv')
JSON_PATH = os.path.join(BASE, 'arxiv-metadata-oai-snapshot.json')
ARXIV_TXT  = {s: os.path.join(BASE,'arxiv-dataset', f'{s}.txt') for s in ['train','val','test']}
PUBMED_TXT = {s: os.path.join(BASE,'pubmed-dataset', f'{s}.txt') for s in ['train','val','test']}

TXT_MAX_ROWS = 50_000
JSON_MAX_ROWS = 400_000

_re_ctrl  = re.compile(r'[\x00-\x1F\x7F-\x9F]')
_re_space = re.compile(r'\s+')

def clean(s):
    if s is None: return ''
    s = str(s).replace('\n', ' ')
    s = _re_ctrl.sub('', s)
    s = _re_space.sub(' ', s).strip()
    return s

frames = []

# Source 1: CSV
print('Loading CSV (source1)...', flush=True)
df = pd.read_csv(CSV_PATH, usecols=['title','summary'], low_memory=False)
df = df.rename(columns={'summary':'abstract'}).dropna(subset=['title','abstract'])
df['source'] = 'csv_arxiv_scientific'
print(f'  CSV raw (post-dropna): {len(df):,}')
frames.append(df[['title','abstract','source']])
del df; gc.collect()

# Source 2: TXT (arxiv, pubmed) -- title always ''
def load_txt(path, limit):
    if not os.path.exists(path): return []
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l: lines.append(l)
            if limit and len(lines) >= limit: break
    return lines

for src, dct in [('txt_arxiv', ARXIV_TXT), ('txt_pubmed', PUBMED_TXT)]:
    total = 0
    for split, path in dct.items():
        lines = load_txt(path, TXT_MAX_ROWS)
        total += len(lines)
        frames.append(pd.DataFrame({'title':'', 'abstract':lines, 'source':src}))
        print(f'  {src} {split}: {len(lines):,}')
    print(f'  {src} total: {total:,}')

# Source 3: JSON streaming
print('Loading JSON (source3)...', flush=True)
BATCH = 50_000
total = 0
j_frames = []
for chunk in pd.read_json(JSON_PATH, lines=True, chunksize=BATCH):
    chunk = chunk[['title','abstract']].dropna()
    chunk = chunk[chunk['abstract'].str.strip() != '']
    j_frames.append(chunk)
    total += len(chunk)
    if total >= JSON_MAX_ROWS:
        break
df_j = pd.concat(j_frames, ignore_index=True).iloc[:JSON_MAX_ROWS].copy()
df_j['source'] = 'json_arxiv_metadata'
print(f'  JSON raw (post-dropna, capped): {len(df_j):,}')
frames.append(df_j[['title','abstract','source']])
del df_j, j_frames; gc.collect()

all_raw = pd.concat(frames, ignore_index=True)
del frames; gc.collect()
print(f'Total raw concat (pre-dedup): {len(all_raw):,}')

# cell 9 order: dropna -> filter empty -> drop_duplicates on RAW (uncleaned) abstract -- BEFORE cleaning
all_raw = all_raw.dropna(subset=['abstract'])
n_before = len(all_raw)
all_raw = all_raw[all_raw['abstract'].astype(str).str.strip() != '']
all_dedup = all_raw.drop_duplicates(subset=['abstract'], keep='first').reset_index(drop=True)
print(f'Dedup (subset=abstract RAW, keep=first): {n_before:,} -> {len(all_dedup):,}')
print(all_dedup['source'].value_counts())

# cell 11: cleaning happens AFTER dedup
all_dedup['title'] = all_dedup['title'].map(clean)
all_dedup['abstract'] = all_dedup['abstract'].map(clean)

all_dedup.to_parquet(os.path.join(SCRATCH, 'reconstructed_with_source.parquet'), index=False)
print('Saved reconstructed_with_source.parquet')
