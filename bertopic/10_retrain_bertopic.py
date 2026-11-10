"""Step 10: fit the K=65 BERTopic model on the 626,442-document corpus.
all-MiniLM-L6-v2 embeddings; vocabulary min_df=15 (ngram 1-2), final c-TF-IDF vectorizer = first 30,000
terms, ngram (1,1), English stop words; k chosen with the Kneedle method (kneed.KneeLocator, convex,
decreasing) over k = 10..500 step 5 on a 100k sample after UMAP; UMAP 5 dims (n_neighbors 15,
min_dist 0, cosine, random_state 42); k-means clustering (random_state 42).
Writes models/BERTopic_v3 (safetensors), topic_words_v3.json, elbow_wcss_v3.npy, elbow_k_v3.txt, ...
"""
import os, gc, json, time, pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ── Fix for a verified BERTopic-internal inconsistency ──
# BERTopic's main per-topic c-TF-IDF fit strips text to alphanumeric-only
# (bertopic._bertopic._preprocess_text) before vectorizing, but our externally
# pre-built fixed vocabulary was counted from raw, unstripped text (matching the
# original notebook's own vocab-building method, faithfully reproduced). Any vocab
# term whose surface form depends on stripped punctuation (URLs, decimals, etc.)
# or that is an ngram_range(1,2)-only bigram never reachable by the final
# ngram_range(1,1) fit ends up with global document frequency 0 in the main fit,
# giving idf=inf for that term (BERTopic already treats it as effectively absent
# from the vocabulary, i.e. zero weight). The representative-document extraction
# step (run after topics/clusters are already finalized) vectorizes raw text
# directly, so it can find a nonzero count for such a term and multiply it by
# that inf weight, crashing cosine_similarity's finite-value check.
# Fix: treat those already-inf entries as the zero weight BERTopic's own global
# fit already assigns them, instead of letting them evaluate to infinity. This
# only affects the optional "representative example docs" feature -- topic
# assignments, cluster sizes, and top-words-per-topic are computed earlier and
# are completely unaffected.
import bertopic._bertopic as _bt_mod
from sklearn.metrics.pairwise import cosine_similarity as _orig_cosine_similarity
import scipy.sparse as _sp

def _safe_cosine_similarity(X, Y=None, **kw):
    if hasattr(X, 'toarray'):
        Xa = X.toarray()
        bad = ~np.isfinite(Xa)
        if bad.any():
            Xa[bad] = 0.0
            X = _sp.csr_matrix(Xa)
    if Y is not None and hasattr(Y, 'toarray'):
        Ya = Y.toarray()
        bad = ~np.isfinite(Ya)
        if bad.any():
            Ya[bad] = 0.0
            Y = _sp.csr_matrix(Ya)
    return _orig_cosine_similarity(X, Y, **kw)

_bt_mod.cosine_similarity = _safe_cosine_similarity

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT_DATA = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')
OUT_MODEL = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'models')
OUT_OUTPUT = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'output')
for d in [OUT_DATA, OUT_MODEL, OUT_OUTPUT]:
    os.makedirs(d, exist_ok=True)

class Timer:
    def __init__(self):
        self.start_total = time.time()
        self.stages = {}
    def stage(self, name):
        return _StageCtx(self, name)
timer_stages = {}

class _StageCtx:
    def __init__(self, timer, name):
        self.timer = timer; self.name = name
    def __enter__(self):
        self.t0 = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] START: {self.name}", flush=True)
        return self
    def __exit__(self, *a):
        dt = time.time() - self.t0
        timer_stages[self.name] = dt
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE : {self.name} ({dt:.1f}s)", flush=True)

t_total0 = time.time()

# ── Load corpus ───────────────────────────────────────────────────────────
with _StageCtx(None, 'Load corpus + src_texts'):
    final = pd.read_parquet(os.path.join(OUT_DATA, 'all_dataset_v3_dedup.parquet'))
    src_texts = (final['title'].astype(str) + ' ' + final['abstract'].astype(str)).str.strip().tolist()
    assert all('\n' not in t and '\r' not in t for t in src_texts), 'embedded newline found in src_texts!'
    assert len(src_texts) == len(final), f'{len(src_texts)} vs {len(final)}'
    print(f'  docs: {len(src_texts):,}')

# ── Embedding (MPS) ──────────────────────────────────────────────────────
emb_path = os.path.join(OUT_DATA, 'embeddings_v3.npy')
EMB_DIM = 384
with _StageCtx(None, 'Embedding (all-MiniLM-L6-v2, MPS)'):
    if os.path.exists(emb_path) and os.path.getsize(emb_path) == len(src_texts) * EMB_DIM * 4:
        embeddings = np.memmap(emb_path, dtype='float32', mode='r', shape=(len(src_texts), EMB_DIM))
        print(f'  loaded existing (raw memmap): {embeddings.shape}')
    else:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='mps')
        EMBED_BATCH = 20_000
        mmap = np.memmap(emb_path, dtype='float32', mode='w+', shape=(len(src_texts), 384))
        written = 0
        for start in range(0, len(src_texts), EMBED_BATCH):
            batch = src_texts[start:start+EMBED_BATCH]
            part = emb_model.encode(batch, show_progress_bar=False, device='mps', batch_size=256)
            mmap[written:written+len(part)] = part.astype('float32')
            mmap.flush()
            written += len(part)
            print(f'    {written:,}/{len(src_texts):,}', flush=True)
            del batch, part; gc.collect()
        del mmap
        embeddings = np.memmap(emb_path, dtype='float32', mode='r', shape=(len(src_texts), EMB_DIM))
        print(f'  saved: {embeddings.shape}')

# ── Vocab build (min_df=15, ngram 1-2, batched) ──────────────────────────
vocab_path = os.path.join(OUT_DATA, 'vocab_final_v3.json')
with _StageCtx(None, 'Build vocab (min_df=15, ngram(1,2), batch=10000)'):
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f'  loaded existing vocab: {len(vocab):,}')
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        VOCAB_BATCH = 10_000
        term_counts = {}
        n_batches = (len(src_texts) - 1) // VOCAB_BATCH + 1
        for i, start in enumerate(range(0, len(src_texts), VOCAB_BATCH)):
            batch = src_texts[start:start+VOCAB_BATCH]
            _cv = CountVectorizer(ngram_range=(1, 2), stop_words='english')
            X = _cv.fit_transform(batch)
            counts = np.asarray(X.sum(axis=0)).flatten()
            for term, idx in _cv.vocabulary_.items():
                c = int(counts[idx])
                if c > 1:
                    term_counts[term] = term_counts.get(term, 0) + c
            del X, batch, _cv, counts
            if (i+1) % 10 == 0 or i == n_batches - 1:
                print(f'    batch {i+1}/{n_batches}', flush=True)
        vocab = [term for term, count in term_counts.items() if count >= 15]
        del term_counts; gc.collect()
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        print(f'  vocab (min_df=15): {len(vocab):,} terms')

# ── Elbow method ──────────────────────────────────────────────────────────
elbow_k_path = os.path.join(OUT_DATA, 'elbow_k_v3.txt')
with _StageCtx(None, 'Elbow method (sample 100k, UMAP 5d, KMeans K=10..500 step5)'):
    if os.path.exists(elbow_k_path):
        N_CLUSTERS = int(open(elbow_k_path).read().strip())
        print(f'  loaded existing elbow_k: {N_CLUSTERS}')
    else:
        from umap import UMAP
        from sklearn.cluster import KMeans as SKKMeans
        from kneed import KneeLocator

        np.random.seed(42)
        ELBOW_SAMPLE = 100_000
        idx = np.random.choice(len(embeddings), ELBOW_SAMPLE, replace=False)
        emb_sampled = np.array(embeddings[idx])
        gc.collect()

        umap_elbow = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',
                           low_memory=True, random_state=42, n_jobs=1)
        emb_for_elbow = umap_elbow.fit_transform(emb_sampled)
        del emb_sampled; gc.collect()
        print(f'  UMAP-reduced elbow sample: {emb_for_elbow.shape}')

        ELBOW_MIN_K, ELBOW_MAX_K, ELBOW_STEP = 10, 500, 5
        range_k = range(ELBOW_MIN_K, ELBOW_MAX_K + 1, ELBOW_STEP)
        wcss = []
        for k in range_k:
            km = SKKMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(emb_for_elbow)
            wcss.append(km.inertia_)
        np.save(os.path.join(OUT_DATA, 'elbow_wcss_v3.npy'), np.array(wcss))

        kneedle = KneeLocator(list(range_k), wcss, curve='convex', direction='decreasing')
        elbow_k = kneedle.elbow
        if elbow_k is None:
            print('  KneeLocator found no elbow -> fallback N_CLUSTERS=30')
            N_CLUSTERS = 30
        else:
            N_CLUSTERS = elbow_k
            print(f'  Elbow found at K={N_CLUSTERS}')
        with open(elbow_k_path, 'w') as f:
            f.write(str(N_CLUSTERS))
        del emb_for_elbow; gc.collect()

print(f'>>> N_CLUSTERS = {N_CLUSTERS}')

# ── UMAP full ─────────────────────────────────────────────────────────────
umap_path = os.path.join(OUT_DATA, 'umap_reduced_v3.npy')
with _StageCtx(None, 'UMAP full corpus reduce to 5d'):
    if os.path.exists(umap_path):
        reduced_embeddings = np.load(umap_path)
        print(f'  loaded existing: {reduced_embeddings.shape}')
    else:
        from umap import UMAP
        emb_array = np.array(embeddings)
        gc.collect()
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',
                           low_memory=True, random_state=42, n_jobs=1)
        reduced_embeddings = umap_model.fit_transform(emb_array)
        del emb_array; gc.collect()
        np.save(umap_path, reduced_embeddings)
        print(f'  saved: {reduced_embeddings.shape}')

# ── Final vectorizer (vocab[:30000], ngram(1,1)) ─────────────────────────
with _StageCtx(None, 'Final vectorizer_model (vocab truncated 30k, ngram(1,1))'):
    from sklearn.feature_extraction.text import CountVectorizer
    vocab_trunc = vocab[:30_000]
    vectorizer_model = CountVectorizer(vocabulary=vocab_trunc, ngram_range=(1, 1),
                                        stop_words='english', dtype=np.float32)
    print(f'  vectorizer vocab: {len(vocab_trunc):,} terms')

# ── Final KMeans + BERTopic fit ──────────────────────────────────────────
model_dir = os.path.join(OUT_MODEL, 'BERTopic_v3')
ckpt_pkl = os.path.join(OUT_DATA, 'checkpoint_v3.pkl')
with _StageCtx(None, f'Final KMeans (K={N_CLUSTERS}) + BERTopic.fit_transform'):
    from sklearn.cluster import KMeans
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction

    kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
    empty_umap = BaseDimensionalityReduction()
    topic_model = BERTopic(
        umap_model=empty_umap,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False,
    )
    topics, probs = topic_model.fit_transform(src_texts, reduced_embeddings)
    print(f'  topics found: {len(set(topics))}')

    os.makedirs(model_dir, exist_ok=True)
    topic_model.save(model_dir, serialization='safetensors', save_ctfidf=True,
                      save_embedding_model='sentence-transformers/all-MiniLM-L6-v2')
    with open(ckpt_pkl, 'wb') as f:
        pickle.dump({'topics': topics, 'probs': probs}, f)

# ── Save topic info / distribution ───────────────────────────────────────
with _StageCtx(None, 'Save topic info + distribution'):
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(OUT_OUTPUT, 'topic_info_v3.csv'), index=False)

    topic_words = {}
    for tid in topic_info['Topic']:
        topic_words[str(tid)] = topic_model.get_topic(tid)
    with open(os.path.join(OUT_OUTPUT, 'topic_words_v3.json'), 'w') as f:
        json.dump({k: [(w, float(s)) for w, s in v] for k, v in topic_words.items()}, f, indent=2)

    dist = pd.Series(topics).value_counts().sort_index()
    dist.to_csv(os.path.join(OUT_OUTPUT, 'topic_distribution_v3.csv'), header=['count'])

    final['topic'] = topics
    final.to_parquet(os.path.join(OUT_DATA, 'all_dataset_v3_with_topic.parquet'), index=False)

    print(topic_info.head(30).to_string())

total_dt = time.time() - t_total0
print('\n' + '='*60)
print('STAGE TIMINGS:')
for name, dt in timer_stages.items():
    print(f'  {name:<55} {dt/60:.1f} min')
print(f'  {"TOTAL":<55} {total_dt/60:.1f} min')
print('='*60)
print(f'N_CLUSTERS (final): {N_CLUSTERS}')
print(f'Topics found: {len(set(topics))}')
print('DONE')
