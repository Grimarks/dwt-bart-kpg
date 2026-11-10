"""Step 13: assign a K=65 topic to every KP20k train/valid/test document and to the Inspec,
Krapivin, NUS and SemEval files (MiniLM embedding -> umap_model_v3.pkl -> kmeans_model_v3.pkl ->
final topic id); writes topic_assignments/<dataset>_<split>_topics_v3.csv.
"""
import os, gc, json, pickle, time
import numpy as np
import pandas as pd

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT_DATA = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')
OUT_ASSIGN = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'topic_assignments')
os.makedirs(OUT_ASSIGN, exist_ok=True)

KP20K = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'kp20k_project')

with open(os.path.join(OUT_DATA, 'kmeans_raw_to_final_topic_v3.json')) as f:
    RAW_TO_FINAL = {int(k): v for k, v in json.load(f).items()}

print('Loading UMAP model (this file is ~3.5GB, may take a bit)...', flush=True)
t0 = time.time()
with open(os.path.join(OUT_DATA, 'umap_model_v3.pkl'), 'rb') as f:
    umap_model = pickle.load(f)
print(f'  loaded in {time.time()-t0:.1f}s', flush=True)

with open(os.path.join(OUT_DATA, 'kmeans_model_v3.pkl'), 'rb') as f:
    kmeans_model = pickle.load(f)

print('Loading SentenceTransformer (all-MiniLM-L6-v2, MPS)...', flush=True)
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='mps')


def embed_texts(texts, batch=20_000, sub_batch=256):
    n = len(texts)
    out = np.empty((n, 384), dtype='float32')
    for start in range(0, n, batch):
        chunk = texts[start:start + batch]
        part = emb_model.encode(chunk, show_progress_bar=False, device='mps', batch_size=sub_batch)
        out[start:start + len(part)] = part.astype('float32')
        gc.collect()
    return out


def assign_topics(texts, label):
    t0 = time.time()
    n = len(texts)
    print(f'[{label}] embedding {n:,} docs...', flush=True)
    emb = embed_texts(texts)
    print(f'[{label}] embedded in {time.time()-t0:.1f}s, UMAP transform...', flush=True)
    t1 = time.time()
    reduced = umap_model.transform(emb)
    print(f'[{label}] UMAP transform in {time.time()-t1:.1f}s, KMeans predict...', flush=True)
    t2 = time.time()
    raw = kmeans_model.predict(reduced)
    topics = np.array([RAW_TO_FINAL[int(r)] for r in raw])
    print(f'[{label}] KMeans predict in {time.time()-t2:.1f}s, TOTAL {time.time()-t0:.1f}s', flush=True)
    return topics


def make_text(title, abstract):
    return (str(title) + ' ' + str(abstract)).strip()


def run_and_save(texts, ids, label, out_name):
    topics = assign_topics(texts, label)
    df = pd.DataFrame({'id': ids, 'topic': topics})
    df.to_csv(os.path.join(OUT_ASSIGN, out_name), index=False)
    dist = pd.Series(topics).value_counts().sort_index()
    print(f'[{label}] saved {out_name} -- {len(df):,} rows, {dist.shape[0]} distinct topics')
    return topics


results_summary = {}

# ── KP20k ─────────────────────────────────────────────────────────────────
for split in ['train', 'validation', 'test']:
    df = pd.read_parquet(os.path.join(KP20K, 'data', f'kp20k_{split}.parquet'))
    texts = [make_text(t, a) for t, a in zip(df['title'], df['abstract'])]
    ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    label = f'kp20k_{split}'
    topics = run_and_save(texts, ids, label, f'kp20k_{split}_topics_v3.csv')
    results_summary[label] = len(topics)
    del df, texts; gc.collect()

# ── Cross-domain ──────────────────────────────────────────────────────────
def load_jsonl_dict(path, title_key='title', abstract_key='abstract', name_key='name'):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

cross_domain_specs = [
    ('inspec', 'test', 'test.json'),
    ('inspec', 'validation', 'validation.json'),
    ('krapivin', 'test', 'test.json'),
    ('krapivin', 'validation', 'validation.json'),
    ('nus', 'test', 'test.json'),
    ('semeval', 'test', 'test.json'),
    ('semeval', 'valid', 'valid.json'),
]
for dataset, split, fname in cross_domain_specs:
    path = os.path.join(KP20K, 'datasets', dataset, fname)
    if not os.path.exists(path):
        print(f'[{dataset}_{split}] SKIP -- file not found: {path}')
        continue
    rows = load_jsonl_dict(path)
    texts = [make_text(r.get('title', ''), r.get('abstract', '')) for r in rows]
    ids = [r.get('name', i) for i, r in enumerate(rows)]
    label = f'{dataset}_{split}'
    topics = run_and_save(texts, ids, label, f'{label}_topics_v3.csv')
    results_summary[label] = len(topics)
    del rows, texts; gc.collect()

print('\n' + '=' * 60)
print('SUMMARY (docs assigned per dataset):')
for k, v in results_summary.items():
    print(f'  {k:<25} {v:,}')
print(f'  {"TOTAL":<25} {sum(results_summary.values()):,}')
print('DONE')
