"""Check whether Inspec/Krapivin/NUS/SemEval-2010 test documents occur in the final BERTopic corpus
(same exact + MinHash/RapidFuzz method as steps 3-4). Result: 3 of the 400 Krapivin test documents
(ids 1007763, 1009027, 1024484), which is why Krapivin is also reported as Krapivin-397.

Original notes:
Are Inspec/Krapivin/NUS/SemEval-2010 TEST docs inside the final
BERTopic corpus (all_dataset_v3_dedup.parquet, 626,442)?

Method copied from output_v2_deleak/scripts/03_exact_match.py and 04_near_dup.py
with the test sets in the role KP20k had there:
  - norm(): NFKC, lowercase, non [a-z0-9 ] -> space, collapse spaces
  - exact: corpus doc fulltext/abstract/title (in that priority) hashed against test maps
  - near-dup (only for corpus docs with no exact hit): MinHash (64 perm, 5-word shingles),
    LSH index of the TEST docs (threshold 0.30), corpus docs query it, candidates verified
    with rapidfuzz token_sort_ratio on fulltext_norm, keep best >= 85.
Read-only: nothing is written outside revisi_ijies_20266560/task2.
"""
import os, re, json, time, unicodedata
import pandas as pd
from multiprocessing import Pool
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
ROOT = os.environ.get("DWTBART_ROOT", ".")
CORPUS = f'{ROOT}/bertopic_project_wsl_original/output_v2_deleak/data/all_dataset_v3_dedup.parquet'
OUT = f'{ROOT}/revisi_ijies_20266560/task2'
NUM_PERM, SHINGLE_K, LSH_THRESHOLD, FUZZ_THRESHOLD = 64, 5, 0.30, 85.0

_re_keep = re.compile(r'[^a-z0-9 ]')
_re_space = re.compile(r'\s+')

def norm(s):  # identical to 02_build_corpora.py
    if s is None:
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = _re_keep.sub(' ', s)
    s = _re_space.sub(' ', s).strip()
    return s

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

def main():
    tests = []
    for ds in ['inspec', 'krapivin', 'nus', 'semeval']:
        with open(f'{ROOT}/kp20k_project/datasets/{ds}/test.json') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for pos, r in enumerate(rows):
            tests.append({'dataset': ds, 'pos': pos, 'name': r['name'], 'title': r['title'], 'abstract': r['abstract']})
    te = pd.DataFrame(tests)
    te['title_norm'] = te['title'].map(norm)
    te['abstract_norm'] = te['abstract'].map(norm)
    te['fulltext_norm'] = (te['title_norm'] + ' ' + te['abstract_norm']).str.strip()
    print(te.groupby('dataset').size().to_dict(), flush=True)

    bt = pd.read_parquet(CORPUS, columns=['title', 'abstract', 'source'])
    assert len(bt) == 626442
    bt['title_norm'] = bt['title'].map(norm)
    bt['abstract_norm'] = bt['abstract'].map(norm)
    bt['fulltext_norm'] = (bt['title_norm'] + ' ' + bt['abstract_norm']).str.strip()
    print(f'corpus: {len(bt):,}', flush=True)

    full_map, title_map, abs_map = {}, {}, {}
    for i, (ft, t, a) in enumerate(zip(te['fulltext_norm'], te['title_norm'], te['abstract_norm'])):
        full_map.setdefault(ft, []).append(i)
        if t: title_map.setdefault(t, []).append(i)
        if a: abs_map.setdefault(a, []).append(i)

    exact_rows = []
    exact_mask = []
    for ci, (ft, t, a) in enumerate(zip(bt['fulltext_norm'], bt['title_norm'], bt['abstract_norm'])):
        kind, hits = None, None
        if ft in full_map: kind, hits = 'EXACT_FULLTEXT', full_map[ft]
        elif a and a in abs_map: kind, hits = 'EXACT_ABSTRACT', abs_map[a]
        elif t and t in title_map: kind, hits = 'EXACT_TITLE', title_map[t]
        exact_mask.append(kind is not None)
        if kind:
            for h in hits:
                exact_rows.append({'corpus_idx': ci, 'kind': kind, 'test_idx': h})
    exact_df = pd.DataFrame(exact_rows, columns=['corpus_idx', 'kind', 'test_idx'])
    print(f'exact corpus hits: {sum(exact_mask):,} rows', flush=True)

    # near-dup
    t0 = time.time()
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for i, txt in enumerate(te['fulltext_norm']):
        lsh.insert(f'te_{i}', make_minhash(txt))
    todo_idx = [i for i, e in enumerate(exact_mask) if not e]
    todo_txt = bt['fulltext_norm'].values
    with Pool(12) as p:
        mhs = p.map(make_minhash, [todo_txt[i] for i in todo_idx], chunksize=2000)
    print(f'minhash corpus done {time.time()-t0:.0f}s', flush=True)
    te_ft = te['fulltext_norm'].tolist()
    near_rows, n_cand = [], 0
    all_cand_scores = []
    for ci, m in zip(todo_idx, mhs):
        cands = lsh.query(m)
        if not cands:
            continue
        n_cand += len(cands)
        best, best_j = -1.0, None
        for k in cands:
            j = int(k.split('_', 1)[1])
            sc = fuzz.token_sort_ratio(todo_txt[ci], te_ft[j])
            all_cand_scores.append({'corpus_idx': ci, 'test_idx': j, 'score': sc})
            if sc > best:
                best, best_j = sc, j
        if best >= FUZZ_THRESHOLD:
            near_rows.append({'corpus_idx': ci, 'test_idx': best_j, 'score': best})
    near_df = pd.DataFrame(near_rows, columns=['corpus_idx', 'test_idx', 'score'])
    print(f'LSH candidates examined: {n_cand:,}; near-dup >=85: {len(near_df):,} corpus rows ({time.time()-t0:.0f}s)', flush=True)

    def enrich(df):
        if df.empty:
            return df
        d = df.merge(te[['dataset', 'pos', 'name', 'title']].rename(columns={'title': 'test_title'}),
                     left_on='test_idx', right_index=True)
        d['corpus_title'] = bt['title'].values[d['corpus_idx']]
        d['corpus_source'] = bt['source'].values[d['corpus_idx']]
        d['corpus_abstract_head'] = [str(x)[:200] for x in bt['abstract'].values[d['corpus_idx']]]
        d['test_abstract_head'] = [str(x)[:200] for x in te['abstract'].values[d['test_idx']]]
        return d
    exact_e, near_e = enrich(exact_df), enrich(near_df)
    exact_e.to_csv(f'{OUT}/exact_pairs.csv', index=False)
    near_e.to_csv(f'{OUT}/neardup_pairs.csv', index=False)
    cand = pd.DataFrame(all_cand_scores, columns=['corpus_idx', 'test_idx', 'score'])
    cand = enrich(cand)
    cand.to_csv(f'{OUT}/all_lsh_candidates_scored.csv', index=False)

    print('\n=== SUMMARY per test set (distinct TEST docs affected) ===')
    summ = []
    for ds, n in te.groupby('dataset').size().items():
        ex = exact_e[exact_e.dataset == ds] if not exact_e.empty else exact_e
        nd = near_e[near_e.dataset == ds] if not near_e.empty else near_e
        ex_docs = set(ex['test_idx']) if len(ex) else set()
        nd_docs = (set(nd['test_idx']) - ex_docs) if len(nd) else set()
        row = {'dataset': ds, 'n_test': int(n),
               'exact_test_docs': len(ex_docs),
               'exact_fulltext': int(ex[ex.kind == 'EXACT_FULLTEXT']['test_idx'].nunique()) if len(ex) else 0,
               'exact_abstract': int(ex[ex.kind == 'EXACT_ABSTRACT']['test_idx'].nunique()) if len(ex) else 0,
               'exact_title_only': int(ex[ex.kind == 'EXACT_TITLE']['test_idx'].nunique()) if len(ex) else 0,
               'neardup_only_test_docs': len(nd_docs),
               'exact_corpus_rows': int(ex['corpus_idx'].nunique()) if len(ex) else 0,
               'neardup_corpus_rows': int(nd['corpus_idx'].nunique()) if len(nd) else 0}
        if ds == 'krapivin':
            row['exact_test_docs_in_first400'] = len([x for x in ex_docs if te.loc[x, 'pos'] < 400])
            row['neardup_only_in_first400'] = len([x for x in nd_docs if te.loc[x, 'pos'] < 400])
        summ.append(row)
    summ = pd.DataFrame(summ)
    print(summ.to_string(index=False))
    summ.to_csv(f'{OUT}/summary.csv', index=False)
    if len(cand):
        print('\nscore distribution of ALL LSH candidates (best per corpus row):')
        print(cand.groupby('corpus_idx')['score'].max().describe().to_string())
    print('\nEXAMPLES exact:')
    print(exact_e.head(15)[['dataset', 'name', 'kind', 'corpus_source', 'test_title', 'corpus_title']].to_string() if len(exact_e) else '  none')
    print('\nEXAMPLES near-dup:')
    print(near_e.head(15)[['dataset', 'name', 'score', 'corpus_source', 'test_title', 'corpus_title']].to_string() if len(near_e) else '  none')

if __name__ == '__main__':
    main()
