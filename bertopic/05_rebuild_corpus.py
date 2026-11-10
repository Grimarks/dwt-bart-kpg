"""Step 5: remove the KP20k-overlapping documents found in steps 3-4
(661,852 -> 650,838 documents); writes all_dataset_v2.parquet and src_texts_v2.txt.
"""
import os
import re
import pandas as pd

_re_ctrl = re.compile(r'[\x00-\x1F\x7F-\x9F]')
_re_space = re.compile(r'\s+')

def clean_line(s):
    s = str(s).replace('\n', ' ').replace('\r', ' ')
    s = _re_ctrl.sub('', s)
    s = _re_space.sub(' ', s).strip()
    return s

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')
OUT_DIR = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

norm = pd.read_parquet(os.path.join(SCRATCH, 'bertopic_corpus_norm.parquet'))
exact = pd.read_parquet(os.path.join(SCRATCH, 'bertopic_corpus_exactflag.parquet'))
near = pd.read_parquet(os.path.join(SCRATCH, 'bertopic_neardup_result.parquet'))

assert len(norm) == len(exact) == 661852, f'length mismatch: norm={len(norm)} exact={len(exact)}'
# exact was derived directly from norm with no row filtering/reordering -> positional alignment is safe
assert (norm['fulltext_norm'].values == exact['fulltext_norm'].values).all(), 'norm/exact row order mismatch!'

exact_flag = exact['exact_match_kind'].notna().values
print(f'exact_flag sum: {exact_flag.sum():,} / {len(exact_flag):,}')

# near dataframe = the subset of `exact` rows where exact_match_kind is NaN, in original relative order, index reset.
# Verify this by checking length and a content spot-check.
non_exact_count = (~exact_flag).sum()
assert len(near) == non_exact_count, f'near len {len(near)} != non_exact_count {non_exact_count}'

non_exact_positions = norm.index[~exact_flag]
# spot-check alignment: fulltext at non_exact_positions must equal near's fulltext in same relative order
assert (norm.loc[non_exact_positions, 'fulltext_norm'].values == near['fulltext_norm'].values).all(), \
    'near-dup row alignment mismatch!'

near_dup_flag_full = pd.Series(False, index=norm.index)
near_dup_flag_full.loc[non_exact_positions] = near['near_dup_flag'].values

remove_mask = exact_flag | near_dup_flag_full.values
print(f'Total remove_mask: {remove_mask.sum():,}')
print(f'  exact only: {exact_flag.sum():,}')
print(f'  near-dup only: {near_dup_flag_full.values.sum():,}')
print(f'  overlap (should be 0, near-dup only computed on non-exact): {(exact_flag & near_dup_flag_full.values).sum():,}')

final = norm.loc[~remove_mask, ['title', 'abstract', 'type', 'source']].copy().reset_index(drop=True)
final['title'] = final['title'].map(clean_line)
final['abstract'] = final['abstract'].map(clean_line)
final = final[final['abstract'] != ''].reset_index(drop=True)
final['present'] = ''
final['absent'] = ''

print()
print(f'FINAL CORPUS SIZE: {len(final):,}')
print(final['source'].value_counts())
print(final['type'].value_counts())

# sanity: no row should still contain an un-parsed raw JSON blob (source2 bug fix check)
blob_like = final['abstract'].str.startswith('{"article_id"') | final['abstract'].str.startswith('{ "article_id"')
print(f'rows still looking like raw JSON blob (should be 0): {blob_like.sum()}')

out_path = os.path.join(OUT_DIR, 'all_dataset_v2.parquet')
final.to_parquet(out_path, index=False)
print(f'saved: {out_path}')

# also write plain source texts file (title + abstract) matching original src_texts.txt convention, for embedding stage
src_texts_path = os.path.join(OUT_DIR, 'src_texts_v2.txt')
lines = (final['title'].astype(str) + ' ' + final['abstract'].astype(str)).str.strip()
with open(src_texts_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines.tolist()))
print(f'saved: {src_texts_path} ({len(lines):,} lines)')
