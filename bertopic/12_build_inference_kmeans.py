"""Step 12: refit k-means on the cached UMAP output, check ARI/NMI against the training topics and
save kmeans_model_v3.pkl plus the raw-cluster -> final-topic mapping (Hungarian matching).
"""
import os, pickle, json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT_DATA = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')

reduced = np.load(os.path.join(OUT_DATA, 'umap_reduced_v3.npy'))
with open(os.path.join(OUT_DATA, 'checkpoint_v3.pkl'), 'rb') as f:
    ckpt = pickle.load(f)
cached_topics = np.array(ckpt['topics'])
K = int(cached_topics.max() + 1)

print('Refitting KMeans (K=%d, random_state=42) on cached training UMAP output...' % K, flush=True)
kmeans_model = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans_model.fit(reduced)
raw_pred = kmeans_model.predict(reduced)

print('ARI vs cached training topics:', adjusted_rand_score(cached_topics, raw_pred))
print('NMI vs cached training topics:', normalized_mutual_info_score(cached_topics, raw_pred))

# Build raw-KMeans-index -> final-BERTopic-topic-ID mapping via Hungarian matching
K2 = int(raw_pred.max() + 1)
cm = np.zeros((K, K2), dtype=np.int64)
for a, b in zip(cached_topics, raw_pred):
    cm[a, b] += 1
row_ind, col_ind = linear_sum_assignment(-cm)
raw_to_final = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
assert len(raw_to_final) == K, 'mapping is not a full bijection over K clusters'
assert len(set(raw_to_final.values())) == K, 'mapping is not injective'

mapped_pred = np.array([raw_to_final[p] for p in raw_pred])
n_mismatch = (mapped_pred != cached_topics).sum()
print(f'After applying label mapping: mismatches = {n_mismatch:,} / {len(cached_topics):,} '
      f'({100*n_mismatch/len(cached_topics):.4f}%)')

assert n_mismatch / len(cached_topics) < 0.001, 'mismatch rate too high after label mapping -- investigate before saving'

with open(os.path.join(OUT_DATA, 'kmeans_model_v3.pkl'), 'wb') as f:
    pickle.dump(kmeans_model, f)
with open(os.path.join(OUT_DATA, 'kmeans_raw_to_final_topic_v3.json'), 'w') as f:
    json.dump(raw_to_final, f, indent=2)

print('Saved kmeans_model_v3.pkl (raw KMeans, predict() gives 0..%d cluster indices)' % (K-1))
print('Saved kmeans_raw_to_final_topic_v3.json (raw cluster index -> final BERTopic topic ID)')
print('For inference: topic_id = raw_to_final[str(kmeans_model.predict(umap_model.transform(embed(text)))[0])]')
