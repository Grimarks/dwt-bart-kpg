"""Step 2: normalise the KP20k splits and the historical 661,852-document corpus
(NFKC, lowercase, non [a-z0-9 ] -> space) for leakage matching; writes kp20k_norm.parquet and
bertopic_corpus_norm.parquet to SCRATCH.
"""
import os, re, json, unicodedata, gc
import pandas as pd

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
SCRATCH = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output_v2_deleak', 'intermediate_parquets')

_re_keep = re.compile(r'[^a-z0-9 ]')
_re_space = re.compile(r'\s+')

def norm(s):
    if s is None:
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = _re_keep.sub(' ', s)
    s = _re_space.sub(' ', s).strip()
    return s

def parse_abstract_text(blob):
    try:
        d = json.loads(blob)
        sents = d.get('abstract_text') or []
        joined = ' '.join(sents)
        joined = re.sub(r'<\s*/?\s*s\s*>', ' ', joined, flags=re.IGNORECASE)
        return joined.strip()
    except Exception:
        return blob  # fall back to raw blob if parse fails

def main():
    print('Loading KP20k...', flush=True)
    kp_train = pd.read_parquet(os.path.join(os.environ.get("DWTBART_ROOT", "."), 'kp20k_project', 'data', 'kp20k_train.parquet'))
    kp_valid = pd.read_parquet(os.path.join(os.environ.get("DWTBART_ROOT", "."), 'kp20k_project', 'data', 'kp20k_validation.parquet'))
    kp_test  = pd.read_parquet(os.path.join(os.environ.get("DWTBART_ROOT", "."), 'kp20k_project', 'data', 'kp20k_test.parquet'))
    kp_train['split'] = 'train'; kp_valid['split'] = 'valid'; kp_test['split'] = 'test'
    kp = pd.concat([kp_train, kp_valid, kp_test], ignore_index=True)
    print(f'  KP20k total: {len(kp):,}')

    kp['title_norm'] = kp['title'].map(norm)
    kp['abstract_norm'] = kp['abstract'].map(norm)
    kp['fulltext_norm'] = (kp['title_norm'] + ' ' + kp['abstract_norm']).str.strip()
    kp[['id','split','title','abstract','title_norm','abstract_norm','fulltext_norm']].to_parquet(
        os.path.join(SCRATCH, 'kp20k_norm.parquet'), index=False)
    print('  saved kp20k_norm.parquet')
    del kp_train, kp_valid, kp_test
    gc.collect()

    print('Loading REAL BERTopic corpus (all_dataset.parquet)...', flush=True)
    real = pd.read_parquet(os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original', 'output', 'data', 'all_dataset.parquet'))
    print(f'  real shape: {real.shape}')

    print('Loading source-attribution reference (recon)...', flush=True)
    recon = pd.read_parquet(os.path.join(SCRATCH, 'reconstructed_with_source.parquet'))
    recon_csv_titles = set(recon.loc[recon['source']=='csv_arxiv_scientific','title'].map(norm))
    recon_json_titles = set(recon.loc[recon['source']=='json_arxiv_metadata','title'].map(norm))
    recon_csv_titles.discard('')
    recon_json_titles.discard('')
    del recon; gc.collect()

    real['title_norm'] = real['title'].map(norm)
    is_empty_title = real['title'].astype(str).str.strip() == ''

    def attribute(row_title_norm, is_empty):
        if is_empty:
            return 'source2_txt'
        if row_title_norm in recon_csv_titles:
            return 'source1_csv'
        if row_title_norm in recon_json_titles:
            return 'source3_json'
        return 'unresolved'

    print('Attributing sources...', flush=True)
    real['source'] = [attribute(t, e) for t, e in zip(real['title_norm'], is_empty_title)]
    print(real['source'].value_counts())

    print('Fixing source2 abstracts (parsing raw JSON blob -> abstract_text)...', flush=True)
    mask2 = real['source'] == 'source2_txt'
    real.loc[mask2, 'abstract'] = real.loc[mask2, 'abstract'].map(parse_abstract_text)

    real['abstract_norm'] = real['abstract'].map(norm)
    real['fulltext_norm'] = (real['title_norm'] + ' ' + real['abstract_norm']).str.strip()

    real[['title','abstract','type','source','title_norm','abstract_norm','fulltext_norm']].to_parquet(
        os.path.join(SCRATCH, 'bertopic_corpus_norm.parquet'), index=False)
    print('  saved bertopic_corpus_norm.parquet')
    print('DONE')

if __name__ == '__main__':
    main()
