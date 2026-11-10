"""Step 4: near-duplicate leakage check against KP20k for corpus documents without an exact hit:
MinHash (64 permutations, 5-word shingles), LSH threshold 0.30, candidates verified with
RapidFuzz token_sort_ratio >= 85; writes bertopic_neardup_result.parquet.
"""
import os, time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')
NUM_PERM = 64
SHINGLE_K = 5
LSH_THRESHOLD = 0.30
FUZZ_THRESHOLD = 85.0

def shingles(text, k=SHINGLE_K):
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {' '.join(words[i:i+k]) for i in range(len(words)-k+1)}

def make_minhash_bytes(text):
    m = MinHash(num_perm=NUM_PERM)
    for sh in shingles(text):
        m.update(sh.encode('utf8'))
    return m

def build_minhashes(texts, nproc=12):
    with Pool(nproc) as p:
        minhashes = p.map(make_minhash_bytes, texts, chunksize=2000)
    return minhashes

def main():
    kp = pd.read_parquet(os.path.join(SCRATCH, 'kp20k_norm.parquet'))
    bt = pd.read_parquet(os.path.join(SCRATCH, 'bertopic_corpus_exactflag.parquet'))

    bt_todo = bt[bt['exact_match_kind'].isna()].copy()
    print(f'KP20k: {len(kp):,}  BERTopic non-exact remaining: {len(bt_todo):,}', flush=True)

    print('Building KP20k minhashes...', flush=True)
    t0 = time.time()
    kp_mh = build_minhashes(kp['fulltext_norm'].tolist())
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    print('Building LSH index...', flush=True)
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for i, m in enumerate(kp_mh):
        lsh.insert(f'kp_{i}', m)
    print(f'  indexed {len(kp_mh):,} KP20k docs', flush=True)

    print('Building BERTopic (non-exact) minhashes...', flush=True)
    t0 = time.time()
    bt_texts = bt_todo['fulltext_norm'].tolist()
    bt_mh = build_minhashes(bt_texts)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    print('Querying LSH for candidates + verifying with rapidfuzz...', flush=True)
    kp_fulltexts = kp['fulltext_norm'].tolist()
    kp_splits = kp['split'].tolist()
    kp_ids = kp['id'].tolist()

    results_idx = []
    results_bestscore = []
    results_bestsplit = []
    t0 = time.time()
    n_candidates_total = 0
    for row_i, (m, text) in enumerate(zip(bt_mh, bt_texts)):
        cand_keys = lsh.query(m)
        if not cand_keys:
            continue
        n_candidates_total += len(cand_keys)
        best_score = -1.0
        best_split = None
        for k in cand_keys:
            j = int(k.split('_', 1)[1])
            score = fuzz.token_sort_ratio(text, kp_fulltexts[j])
            if score > best_score:
                best_score = score
                best_split = kp_splits[j]
        if best_score >= FUZZ_THRESHOLD:
            results_idx.append(row_i)
            results_bestscore.append(best_score)
            results_bestsplit.append(best_split)
        if row_i % 100000 == 0:
            print(f'  ... {row_i:,}/{len(bt_mh):,} processed, {time.time()-t0:.1f}s elapsed, '
                  f'candidates so far {n_candidates_total:,}, near-dup found {len(results_idx):,}', flush=True)

    print(f'Done querying in {time.time()-t0:.1f}s. Total candidate pairs examined: {n_candidates_total:,}', flush=True)
    print(f'Near-dup (fuzz>={FUZZ_THRESHOLD}) found: {len(results_idx):,}', flush=True)

    bt_todo = bt_todo.reset_index(drop=True)
    near_dup_flag = pd.Series(False, index=bt_todo.index)
    near_dup_score = pd.Series(np.nan, index=bt_todo.index)
    near_dup_split = pd.Series('', index=bt_todo.index)
    near_dup_flag.iloc[results_idx] = True
    near_dup_score.iloc[results_idx] = results_bestscore
    near_dup_split.iloc[results_idx] = results_bestsplit

    bt_todo['near_dup_flag'] = near_dup_flag
    bt_todo['near_dup_score'] = near_dup_score
    bt_todo['near_dup_kp_split'] = near_dup_split

    bt_todo.to_parquet(os.path.join(SCRATCH, 'bertopic_neardup_result.parquet'), index=False)
    print('saved bertopic_neardup_result.parquet')

    print()
    print('Near-dup by source:')
    print(bt_todo[bt_todo['near_dup_flag']].groupby('source').size())

if __name__ == '__main__':
    main()
