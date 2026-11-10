"""v1 builder of the topic-label (`topic`) and shuffled-topic (`shuffled`) inputs. Its tokenisation is
SUPERSEDED by t5v2_build_conditions.py (the version used in the paper), which imports load_split,
get_topic_str, SPECS and the evaluate.py functions from this file; keep both files together.

Original notes:
Build KP20k + Inspec/Krapivin(400)/NUS/SemEval files for 3 conditions:
  ori      : src = preprocess(f'{title} <eos> {abstract}')
  topic    : src = preprocess(f'{title} {topic_str} . <eos> {abstract}')      (position 'before')
  shuffled : as topic, topic_str of a random OTHER topic (seed 2026, fresh RNG per file,
             p proportional to the v3 topic distribution of that same file, original topic excluded)

preprocess / get_topic_str / build_trg / output format {"src","tgt"} copied verbatim from
an earlier topic-file builder of this project (not part of this release).

None of these datasets has a prmu column. P (present) vs absent is
decided with evaluate.py's OWN functions (imported, not rewritten):
  tokens = text.lower().strip().split(' ')  (same as evaluate.py main())
  stem_word_list(...)  +  check_present_keyphrases(src_stemmed, kp_stemmed_2d, match_by_str=False)
where text = the ori src (title + <eos> + abstract after preprocess), keyphrase = build_trg's
kp_clean form. present -> tag 'P', absent -> tag 'U'; then build_trg(keyphrases, tags) unchanged.
Empty keyphrases (from ';;' or trailing ';' in the raw keywords string) are dropped and counted.
"""
import os, re, sys, json, importlib.util
import numpy as np
import pandas as pd

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
ROOT = os.environ.get("DWTBART_ROOT", ".")
V3 = f'{ROOT}/bertopic_project_wsl_original/output_v2_deleak'
ASSIGN_DIR = f'{V3}/topic_assignments'
OUT = f'{ROOT}/revisi_ijies_20266560/task5/kpg_conditions_topicv3'
EVAL_PY = f'{ROOT}/PC1_PC2_FINAL_PACKAGE/scripts/kpg_training/evaluate.py'

spec = importlib.util.spec_from_file_location('kpg_evaluate', EVAL_PY)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
stem_word_list, check_present_keyphrases = ev.stem_word_list, ev.check_present_keyphrases

with open(f'{V3}/output/mistral_labels_v3.json') as f:
    mistral_labels = {int(k): v for k, v in json.load(f).items()}
with open(f'{V3}/output/topic_words_v3.json') as f:
    topic_words = {int(k): v for k, v in json.load(f).items()}

# ---- verbatim from the earlier topic-file builder ---------------------------
def get_topic_str(topic_id):
    label = mistral_labels.get(topic_id, f'Topic_{topic_id}')
    kws = ' '.join([w for w, s in topic_words.get(topic_id, [])[:10]])
    return f'{label} {kws}'.strip()


def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'\b\d+\b', '<digit>', text)
    return ' '.join(text.split())


def build_src(title, abstract, topic_str, position):
    if position == 'before':
        return preprocess(f'{title} {topic_str} . <eos> {abstract}')
    else:
        return preprocess(f'{title} <eos> {abstract} {topic_str} .')


def build_trg(keyphrases, prmu):
    present, absent = [], []
    for kp, tag in zip(keyphrases, prmu):
        kp_clean = re.sub(r'\b\d+\b', '<digit>', kp.lower().strip())
        if tag in ['P', 'R']:
            present.append(kp_clean)
        else:
            absent.append(kp_clean)
    p = ';'.join(present) if present else ''
    a = ';'.join(absent) if absent else ''
    if p and a:
        return f'{p};<peos>;{a}'
    elif p:
        return f'{p};<peos>'
    elif a:
        return f'<peos>;{a}'
    return '<peos>'
# -----------------------------------------------------------------------------

def src_ori(title, abstract):
    return preprocess(f'{title} <eos> {abstract}')


def eval_tags(ori_src, keyphrases):
    """present/absent exactly as evaluate.py decides it (tokenisation of its main())."""
    src_tokens = ori_src.lower().strip().split(' ')
    src_stem = stem_word_list(src_tokens)
    kp_tok = [re.sub(r'\b\d+\b', '<digit>', kp.lower().strip()).strip().split(' ') for kp in keyphrases]
    kp_stem = [stem_word_list(t) for t in kp_tok]
    mask = check_present_keyphrases(src_stem, kp_stem, match_by_str=False)
    return ['P' if m else 'U' for m in mask]


def load_split(dataset, split):
    if dataset == 'kp20k':
        fsplit = {'train': 'train', 'valid': 'validation', 'test': 'test'}[split]
        df = pd.read_parquet(f'{ROOT}/kp20k_project/data/kp20k_{fsplit}.parquet', columns=['id', 'title', 'abstract', 'keywords'])
        tdf = pd.read_csv(f'{ASSIGN_DIR}/kp20k_{fsplit}_topics_v3.csv')
    else:
        with open(f'{ROOT}/kp20k_project/datasets/{dataset}/test.json') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        df = pd.DataFrame({'id': [str(r['name']) for r in rows], 'title': [r['title'] for r in rows],
                           'abstract': [r['abstract'] for r in rows], 'keywords': [r['keywords'] for r in rows]})
        tdf = pd.read_csv(f'{ASSIGN_DIR}/{dataset}_test_topics_v3.csv')
        if dataset == 'krapivin':  # the manuscript's 400 = first 400 rows of the 460
            df = df.iloc[:400].reset_index(drop=True)
    tmap = dict(zip(tdf['id'].astype(str), tdf['topic'].astype(int)))
    df['id'] = df['id'].astype(str)
    df['topic'] = df['id'].map(tmap)
    return df


SPECS = [('kp20k', 'train'), ('kp20k', 'valid'), ('kp20k', 'test'),
         ('inspec', 'test'), ('krapivin', 'test'), ('nus', 'test'), ('semeval', 'test')]


def main():
    for d in ['ori', 'topic', 'shuffled', 'shuffled_mapping', 'ids']:
        os.makedirs(f'{OUT}/{d}', exist_ok=True)
    K = len(topic_words)
    topic_str = {t: get_topic_str(t) for t in range(K)}
    summary = []
    for dataset, split in SPECS:
        name = f'{dataset}{"400" if dataset == "krapivin" else ""}_{split}'
        df = load_split(dataset, split)
        n_missing = int(df['topic'].isna().sum())
        assert n_missing == 0, f'{name}: {n_missing} missing topics'
        df['topic'] = df['topic'].astype(int)

        # shuffled topic: per-file distribution, original excluded, seed 2026 fresh per file
        counts = np.bincount(df['topic'].values, minlength=K).astype(float)
        rng = np.random.default_rng(2026)
        shuf = np.empty(len(df), dtype=int)
        for i, t in enumerate(df['topic'].values):
            p = counts.copy(); p[t] = 0.0; p /= p.sum()
            shuf[i] = rng.choice(K, p=p)
        df['topic_shuf'] = shuf

        n_empty_kp = n_kp = n_present = 0
        f_ori = open(f'{OUT}/ori/{name}_ori.json', 'w', encoding='utf-8')
        f_top = open(f'{OUT}/topic/{name}_topic_before_topicv3.json', 'w', encoding='utf-8')
        f_shu = open(f'{OUT}/shuffled/{name}_shuffled_before_topicv3.json', 'w', encoding='utf-8')
        for title, abstract, kw, t, ts in zip(df['title'], df['abstract'], df['keywords'], df['topic'], df['topic_shuf']):
            title = str(title).strip(); abstract = str(abstract).strip()
            raw_kps = str(kw).split(';')
            kps = [k for k in raw_kps if k.strip() != '']
            n_empty_kp += len(raw_kps) - len(kps)
            o_src = src_ori(title, abstract)
            tags = eval_tags(o_src, kps)
            n_kp += len(kps); n_present += tags.count('P')
            tgt = build_trg(kps, tags)
            f_ori.write(json.dumps({'src': o_src, 'tgt': tgt}, ensure_ascii=False) + '\n')
            f_top.write(json.dumps({'src': build_src(title, abstract, topic_str[t], 'before'), 'tgt': tgt}, ensure_ascii=False) + '\n')
            f_shu.write(json.dumps({'src': build_src(title, abstract, topic_str[ts], 'before'), 'tgt': tgt}, ensure_ascii=False) + '\n')
        for f in (f_ori, f_top, f_shu):
            f.close()
        df[['id']].to_csv(f'{OUT}/ids/{name}_ids.txt', index=False, header=False)
        pd.DataFrame({'id': df['id'], 'topik_asli': df['topic'],
                      'label_asli': df['topic'].map(mistral_labels),
                      'topik_acak': df['topic_shuf'], 'label_acak': df['topic_shuf'].map(mistral_labels)}
                     ).to_csv(f'{OUT}/shuffled_mapping/{name}_shuffled_mapping.csv', index=False)
        row = {'file': name, 'rows': len(df), 'missing_topic': n_missing, 'distinct_topics': int((counts > 0).sum()),
               'keyphrases': n_kp, 'present_evalpy': n_present, 'present_share': round(n_present / n_kp, 4),
               'empty_kp_dropped': n_empty_kp, 'shuffled_eq_orig': int((df['topic'] == df['topic_shuf']).sum())}
        summary.append(row)
        print(row, flush=True)
    pd.DataFrame(summary).to_csv(f'{OUT}/../build_summary.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
