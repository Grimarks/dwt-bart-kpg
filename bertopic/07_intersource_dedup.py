"""Step 7: cross-source near-duplicate removal (MinHash-LSH 0.30 + RapidFuzz >= 85, connected
components, one keeper per group): 650,838 -> 626,442 documents; writes all_dataset_v3_dedup.parquet,
the final BERTopic corpus.
"""
import os, re, unicodedata, time
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')
DATA_DIR = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')

NUM_PERM = 64
SHINGLE_K = 5
LSH_THRESHOLD = 0.30
FUZZ_THRESHOLD = 85.0

_re_keep = re.compile(r'[^a-z0-9 ]')
_re_space = re.compile(r'\s+')

def norm(s):
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = _re_keep.sub(' ', s)
    return _re_space.sub(' ', s).strip()

SOURCE_PRIORITY = {'source1_csv': 0, 'source2_txt': 1, 'source3_json': 2, 'unresolved': 3}

def shingles(text, k=SHINGLE_K):
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {' '.join(words[i:i+k]) for i in range(len(words)-k+1)}

def make_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    for sh in shingles(text):
        m.update(sh.encode('utf8'))
    return m

def build_minhashes(texts, nproc=12):
    with Pool(nproc) as p:
        return p.map(make_minhash, texts, chunksize=2000)

def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'all_dataset_v2.parquet'))
    print(f'Loaded corpus: {len(df):,}')
    df['fulltext_norm'] = (df['title'].map(norm) + ' ' + df['abstract'].map(norm)).str.strip()
    df['prio'] = df['source'].map(SOURCE_PRIORITY).fillna(4)
    df['orig_idx'] = np.arange(len(df))

    # ── Stage 1: exact match dedup (normalized fulltext) ──────────────────
    print('\n=== EXACT DEDUP ===')
    groups = df.groupby('fulltext_norm').indices  # dict: text -> array of positions
    exact_dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f'exact-duplicate groups: {len(exact_dup_groups):,}')

    keep_mask = np.ones(len(df), dtype=bool)
    exact_pair_counter = {}
    exact_removed = 0
    for text, positions in exact_dup_groups.items():
        positions = sorted(positions, key=lambda i: (df['prio'].iloc[i], i))
        keeper = positions[0]
        losers = positions[1:]
        for l in losers:
            keep_mask[l] = False
        exact_removed += len(losers)
        srcs = tuple(sorted(set(df['source'].iloc[positions])))
        exact_pair_counter[srcs] = exact_pair_counter.get(srcs, 0) + 1

    print(f'exact duplicates removed: {exact_removed:,}')
    print('exact-dup groups by source-combo:')
    for combo, cnt in sorted(exact_pair_counter.items(), key=lambda x: -x[1]):
        print(f'  {combo}: {cnt:,} groups')

    df_after_exact = df[keep_mask].reset_index(drop=True)
    print(f'remaining after exact dedup: {len(df_after_exact):,}')

    # ── Stage 2: near-dup via MinHash LSH self-join + RapidFuzz verify ────
    print('\n=== NEAR-DUP (self-join, fuzz>=85) ===')
    texts = df_after_exact['fulltext_norm'].tolist()
    t0 = time.time()
    minhashes = build_minhashes(texts)
    print(f'minhash build: {time.time()-t0:.1f}s')

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)
    print(f'LSH indexed: {len(minhashes):,}')

    edges_i, edges_j = [], []
    t0 = time.time()
    seen_pairs = set()
    n_candidates = 0
    for i, m in enumerate(minhashes):
        cands = lsh.query(m)
        for c in cands:
            j = int(c)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            n_candidates += 1
            score = fuzz.token_sort_ratio(texts[i], texts[j])
            if score >= FUZZ_THRESHOLD:
                edges_i.append(i)
                edges_j.append(j)
        if i % 100000 == 0:
            print(f'  ... {i:,}/{len(minhashes):,}  candidates so far {n_candidates:,}  edges {len(edges_i):,}', flush=True)

    print(f'query+verify done: {time.time()-t0:.1f}s, candidates examined {n_candidates:,}, edges {len(edges_i):,}')

    n = len(df_after_exact)
    if edges_i:
        data = np.ones(len(edges_i), dtype=int)
        adj = coo_matrix((data, (edges_i, edges_j)), shape=(n, n))
        n_components, labels = connected_components(adj, directed=False)
    else:
        n_components, labels = n, np.arange(n)

    df_after_exact['component'] = labels
    comp_sizes = df_after_exact.groupby('component').size()
    dup_components = comp_sizes[comp_sizes > 1]
    print(f'near-dup connected components (size>1): {len(dup_components):,}, total docs involved: {dup_components.sum():,}')

    near_keep_mask = np.ones(n, dtype=bool)
    near_pair_counter = {}
    for comp_id in dup_components.index:
        idxs = df_after_exact.index[df_after_exact['component'] == comp_id].tolist()
        idxs_sorted = sorted(idxs, key=lambda i: (df_after_exact['prio'].iloc[i], df_after_exact['orig_idx'].iloc[i]))
        keeper = idxs_sorted[0]
        for l in idxs_sorted[1:]:
            near_keep_mask[l] = False
        srcs = tuple(sorted(set(df_after_exact['source'].iloc[idxs])))
        near_pair_counter[srcs] = near_pair_counter.get(srcs, 0) + 1

    near_removed = (~near_keep_mask).sum()
    print(f'near-dup duplicates removed: {near_removed:,}')
    print('near-dup groups by source-combo:')
    for combo, cnt in sorted(near_pair_counter.items(), key=lambda x: -x[1]):
        print(f'  {combo}: {cnt:,} groups')

    final = df_after_exact[near_keep_mask].reset_index(drop=True)
    print(f'\nFINAL corpus after exact+near inter-source dedup: {len(final):,}')
    print(final['source'].value_counts())

    out_cols = ['title', 'abstract', 'type', 'source', 'present', 'absent']
    final[out_cols].to_parquet(os.path.join(DATA_DIR, 'all_dataset_v3_dedup.parquet'), index=False)
    print('saved all_dataset_v3_dedup.parquet')

    print('\n=== SUMMARY ===')
    print(f'Start (post-KP20k-leakage-removal): {len(df):,}')
    print(f'Exact inter-source dup removed: {exact_removed:,}')
    print(f'Near-dup (>=85%) inter-source removed: {near_removed:,}')
    print(f'Total removed (inter-source): {exact_removed + near_removed:,}')
    print(f'FINAL corpus (KP20k-leak-free + inter-source-dedup-free): {len(final):,}')

if __name__ == '__main__':
    main()
