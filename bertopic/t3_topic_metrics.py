"""Topic quality of the K=65 model: gensim C_V and C_NPMI (top-10 words) and topic diversity.
Run with gensim 4.4.0, e.g.  python t3_topic_metrics.py --model v3 --n 0  (full corpus).

Original notes:
C_V, C_NPMI (gensim CoherenceModel, topn=10, default windows:
c_v=110, c_npmi=10) + topic diversity (unique words in top-10 / (K*10)).

texts = corpus docs tokenised with the SAME analyzer as the CountVectorizer in
10_retrain_bertopic.py: CountVectorizer(vocabulary=vocab[:30000], ngram_range=(1,1),
stop_words='english').build_analyzer()  (lowercase, token_pattern \\b\\w\\w+\\b, English
stop list). Doc text = title + ' ' + abstract, exactly as src_texts in 10_retrain.

usage: t3_topic_metrics.py --model v3|old --n N   (N=0 -> full corpus; else random sample, seed 42)
"""
import sys, json, time, argparse, random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

import os
# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
ROOT = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'bertopic_project_wsl_original')
OUT = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'revisi_ijies_20266560', 'task3')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['v3', 'old'], required=True)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--processes', type=int, default=11)
    args = ap.parse_args()
    t0 = time.time()

    if args.model == 'v3':
        topic_words = json.load(open(f'{ROOT}/output_v2_deleak/output/topic_words_v3.json'))
        vocab = json.load(open(f'{ROOT}/output_v2_deleak/data/vocab_final_v3.json'))
        cv = CountVectorizer(vocabulary=vocab[:30_000], ngram_range=(1, 1), stop_words='english', dtype=np.float32)
        df = pd.read_parquet(f'{ROOT}/output_v2_deleak/data/all_dataset_v3_dedup.parquet', columns=['title', 'abstract'])
        N = len(df)
        idx = np.arange(N) if args.n == 0 else np.sort(np.random.default_rng(42).choice(N, args.n, replace=False))
        docs = [(str(t) + ' ' + str(a)).strip() for t, a in zip(df['title'].values[idx], df['abstract'].values[idx])]
        del df
    else:
        # Old K=25 model (output/): topic words from output/output/topic_words.json; texts = the
        # historical src_texts.txt the old model was fit on (one doc per line, 661,852 lines).
        # Old notebook (bertopic-arxivvv.ipynb cell 22) used vocabulary=vocab[:50_000] with the same
        # ngram/stop_words -> analyzer is identical (a fixed vocabulary does not alter the analyzer).
        topic_words = json.load(open(f'{ROOT}/output/output/topic_words.json'))
        cv = CountVectorizer(ngram_range=(1, 1), stop_words='english', dtype=np.float32)
        N = 661_852
        idx = np.arange(N) if args.n == 0 else np.sort(np.random.default_rng(42).choice(N, args.n, replace=False))
        want = set(idx.tolist())
        docs = []
        with open(f'{ROOT}/output/data/src_texts.txt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in want:
                    docs.append(line.rstrip('\n'))
        print(f'  lines read: {i+1:,}', flush=True)
        assert i + 1 == N, f'src_texts.txt has {i+1} lines, expected {N}'

    analyzer = cv.build_analyzer()
    texts = [[sys.intern(tok) for tok in analyzer(d)] for d in docs]
    del docs
    ntok = sum(len(t) for t in texts)
    print(f'[{args.model}] docs={len(texts):,} (corpus N={N:,}, sample={"full" if args.n == 0 else args.n}) tokens={ntok:,} '
          f'tokenise {time.time()-t0:.0f}s', flush=True)

    dictionary = Dictionary(texts)
    K = len(topic_words)
    topics = []
    missing = []
    for k in sorted(topic_words, key=int):
        ws = [w for w, s in topic_words[k][:10]]
        miss = [w for w in ws if w not in dictionary.token2id]
        if miss:
            missing.append((k, miss))
        topics.append([w for w in ws if w in dictionary.token2id])
    print(f'K={K}; topic words missing from dictionary: {missing}', flush=True)

    all_w = [w for k in sorted(topic_words, key=int) for w, s in topic_words[k][:10]]
    diversity = len(set(all_w)) / len(all_w)

    res = {'model': args.model, 'K': K, 'N_corpus': N, 'n_docs_used': len(texts),
           'sample_seed': None if args.n == 0 else 42, 'n_tokens': ntok, 'missing_words': missing,
           'topic_diversity': diversity, 'n_topwords_total': len(all_w), 'n_topwords_unique': len(set(all_w))}
    for meas in ['c_v', 'c_npmi']:
        t1 = time.time()
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=meas, topn=10,
                            processes=args.processes)
        per = cm.get_coherence_per_topic()
        res[meas] = float(np.mean(per))
        res[f'{meas}_window'] = cm.window_size
        res[f'{meas}_per_topic'] = [float(x) for x in per]
        res[f'{meas}_seconds'] = time.time() - t1
        print(f'[{args.model}] {meas} (window={cm.window_size}) = {res[meas]:.4f}  ({time.time()-t1:.0f}s)', flush=True)
    print(f'[{args.model}] topic diversity = {diversity:.4f} ({len(set(all_w))}/{len(all_w)})')
    res['total_seconds'] = time.time() - t0
    tag = 'full' if args.n == 0 else f'sample{args.n}'
    json.dump(res, open(f'{OUT}/metrics_{args.model}_{tag}.json', 'w'), indent=1)
    print(f'TOTAL {res["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
