"""Step 11: refit UMAP (and k-means) with the same parameters and seed as step 10, assert that the
result equals the cached training output, and save umap_model_v3.pkl for topic assignment of new documents.
"""
import os, pickle, time
import numpy as np

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT_DATA = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')

emb_path = os.path.join(OUT_DATA, 'embeddings_v3.npy')
n_docs = 626442
embeddings = np.memmap(emb_path, dtype='float32', mode='r', shape=(n_docs, 384))

print('Refitting UMAP (same params/seed as training) to capture the fitted transformer...', flush=True)
t0 = time.time()
from umap import UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',
                   low_memory=True, random_state=42, n_jobs=1)
reduced = umap_model.fit_transform(np.array(embeddings))
print(f'  done in {time.time()-t0:.1f}s, shape {reduced.shape}', flush=True)

cached_reduced = np.load(os.path.join(OUT_DATA, 'umap_reduced_v3.npy'))
match = np.allclose(reduced, cached_reduced, atol=1e-5)
max_diff = np.abs(reduced - cached_reduced).max()
print(f'Determinism check vs cached training UMAP output: allclose={match}, max_abs_diff={max_diff:.2e}')
assert match, 'UMAP refit does NOT match cached training output -- not safe to use for inference!'

N_CLUSTERS = int(open(os.path.join(OUT_DATA, 'elbow_k_v3.txt')).read().strip())
print(f'Refitting KMeans (K={N_CLUSTERS}, same seed) to capture the fitted clusterer...', flush=True)
t0 = time.time()
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
kmeans_model.fit(reduced)
print(f'  done in {time.time()-t0:.1f}s', flush=True)

with open(os.path.join(OUT_DATA, 'checkpoint_v3.pkl'), 'rb') as f:
    ckpt = pickle.load(f)
cached_topics = np.array(ckpt['topics'])
refit_topics = kmeans_model.predict(reduced)
n_mismatch = (cached_topics != refit_topics).sum()
print(f'Determinism check vs cached training topics: mismatches = {n_mismatch:,} / {len(cached_topics):,}')
assert n_mismatch == 0, 'KMeans refit does NOT reproduce identical training-time topic assignments!'

with open(os.path.join(OUT_DATA, 'umap_model_v3.pkl'), 'wb') as f:
    pickle.dump(umap_model, f)
with open(os.path.join(OUT_DATA, 'kmeans_model_v3.pkl'), 'wb') as f:
    pickle.dump(kmeans_model, f)
print('Saved umap_model_v3.pkl and kmeans_model_v3.pkl -- verified deterministic, safe for inference.')
