"""Step 3: exact-match leakage check of every corpus document against KP20k train/valid/test
(fulltext, then abstract, then title, on normalised text); writes bertopic_corpus_exactflag.parquet.
"""
import os
import pandas as pd

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')

kp = pd.read_parquet(os.path.join(SCRATCH, 'kp20k_norm.parquet'))
bt = pd.read_parquet(os.path.join(SCRATCH, 'bertopic_corpus_norm.parquet'))

print(f'KP20k: {len(kp):,}  BERTopic corpus: {len(bt):,}')

kp_full_map = {}
for ft, split, id_ in zip(kp['fulltext_norm'], kp['split'], kp['id']):
    kp_full_map.setdefault(ft, []).append((split, id_))
kp_title_map = {}
for t, split, id_ in zip(kp['title_norm'], kp['split'], kp['id']):
    if t:
        kp_title_map.setdefault(t, []).append((split, id_))
kp_abs_map = {}
for a, split, id_ in zip(kp['abstract_norm'], kp['split'], kp['id']):
    if a:
        kp_abs_map.setdefault(a, []).append((split, id_))

print(f'KP20k unique fulltext: {len(kp_full_map):,}  unique title: {len(kp_title_map):,}  unique abstract: {len(kp_abs_map):,}')

def match_kind(ft, t, a):
    if ft in kp_full_map:
        return 'EXACT_FULLTEXT', kp_full_map[ft]
    if a and a in kp_abs_map:
        return 'EXACT_ABSTRACT', kp_abs_map[a]
    if t and t in kp_title_map:
        return 'EXACT_TITLE', kp_title_map[t]
    return None, None

kinds = []
kp_hits_split = []
for ft, t, a in zip(bt['fulltext_norm'], bt['title_norm'], bt['abstract_norm']):
    kind, hits = match_kind(ft, t, a)
    kinds.append(kind)
    kp_hits_split.append(';'.join(sorted(set(h[0] for h in hits))) if hits else '')

bt['exact_match_kind'] = kinds
bt['exact_match_kp_splits'] = kp_hits_split

print()
print('Exact match kind counts:')
print(bt['exact_match_kind'].value_counts(dropna=False))
print()
print('Exact matches by source:')
print(bt[bt['exact_match_kind'].notna()].groupby('source').size())
print()
print('Exact matches by source x kp_split (any split touched):')
exact = bt[bt['exact_match_kind'].notna()].copy()
for split in ['train','valid','test']:
    exact[f'hit_{split}'] = exact['exact_match_kp_splits'].str.contains(split)
print(exact.groupby('source')[['hit_train','hit_valid','hit_test']].sum())

bt.to_parquet(os.path.join(SCRATCH, 'bertopic_corpus_exactflag.parquet'), index=False)
print('saved bertopic_corpus_exactflag.parquet')
