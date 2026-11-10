"""KP20k v2 preprocessing + condition files used in the paper (package kpg_conditions_topicv3_v2,
zip md5 ae3a01c9fb23b6e769cbd84022203deb): ori = "title <eos> abstract"; topic = "title <topic label + 10 top
words> . <eos> abstract"; shuffled = same with a random other topic (np.random.default_rng(2026), fresh per
file). Regex tokenisation \\w+|[^\\w\\s] with <eos>, <digit>, <peos> protected; pure digits -> <digit>;
targets "present;...;<peos>;absent;..." with present/absent decided by evaluate.py on the ori source.

Original notes:
v2: same as t5_build_conditions.py EXCEPT tokenisation.

preprocess v2:
  1. lowercase + strip
  2. tokenise with  <eos>|<digit>|<peos>|\\w+|[^\\w\\s]   (special tokens protected, every
     punctuation char becomes its own token)
  3. AFTER tokenisation, tokens that are pure digits (\\d+) -> <digit>
  4. join with single spaces
The SAME function is applied to title, abstract, topic_str (inside build_src) and to
every keyphrase (build_trg kp_clean). present/absent = evaluate.py stem_word_list +
check_present_keyphrases on the NEW ori src (split(' ')), keyphrase = preprocess_v2(kp).split(' ').
Everything else (get_topic_str, src templates, build_trg logic, shuffled sampling seed 2026 per
file with per-file distribution, Krapivin 400, dropping empty keyphrases) is identical to v1.
"""
import os, re, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t5_build_conditions import (load_split, get_topic_str, SPECS, topic_words, mistral_labels,
                                 stem_word_list, check_present_keyphrases)

# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT = os.path.join(os.environ.get("DWTBART_ROOT", "."), 'revisi_ijies_20266560', 'task5_v2', 'kpg_conditions_topicv3_v2')

_TOK = re.compile(r'<eos>|<digit>|<peos>|\w+|[^\w\s]')
_DIGITS = re.compile(r'\d+')


def preprocess(text):
    toks = _TOK.findall(str(text).lower().strip())
    return ' '.join('<digit>' if _DIGITS.fullmatch(t) else t for t in toks)


def build_src(title, abstract, topic_str, position):
    if position == 'before':
        return preprocess(f'{title} {topic_str} . <eos> {abstract}')
    else:
        return preprocess(f'{title} <eos> {abstract} {topic_str} .')


def build_trg(keyphrases, prmu):
    present, absent = [], []
    for kp, tag in zip(keyphrases, prmu):
        kp_clean = preprocess(kp)
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


def src_ori(title, abstract):
    return preprocess(f'{title} <eos> {abstract}')


def eval_tags(ori_src, keyphrases):
    src_stem = stem_word_list(ori_src.lower().strip().split(' '))
    kp_stem = [stem_word_list(preprocess(kp).strip().split(' ')) for kp in keyphrases]
    mask = check_present_keyphrases(src_stem, kp_stem, match_by_str=False)
    return ['P' if m else 'U' for m in mask]


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
        counts = np.bincount(df['topic'].values, minlength=K).astype(float)
        rng = np.random.default_rng(2026)
        shuf = np.empty(len(df), dtype=int)
        for i, t in enumerate(df['topic'].values):
            p = counts.copy(); p[t] = 0.0; p /= p.sum()
            shuf[i] = rng.choice(K, p=p)
        df['topic_shuf'] = shuf

        n_empty_kp = n_kp = n_present = 0
        f_ori = open(f'{OUT}/ori/{name}_ori_v2.json', 'w', encoding='utf-8')
        f_top = open(f'{OUT}/topic/{name}_topic_before_topicv3_v2.json', 'w', encoding='utf-8')
        f_shu = open(f'{OUT}/shuffled/{name}_shuffled_before_topicv3_v2.json', 'w', encoding='utf-8')
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
        pd.DataFrame({'id': df['id'], 'topik_asli': df['topic'], 'label_asli': df['topic'].map(mistral_labels),
                      'topik_acak': df['topic_shuf'], 'label_acak': df['topic_shuf'].map(mistral_labels)}
                     ).to_csv(f'{OUT}/shuffled_mapping/{name}_shuffled_mapping.csv', index=False)
        row = {'file': name, 'rows': len(df), 'missing_topic': n_missing, 'distinct_topics': int((counts > 0).sum()),
               'keyphrases': n_kp, 'present_evalpy': n_present, 'present_share': round(n_present / n_kp, 4),
               'empty_kp_dropped': n_empty_kp, 'shuffled_eq_orig': int((df['topic'] == df['topic_shuf']).sum())}
        summary.append(row)
        print(row, flush=True)
    pd.DataFrame(summary).to_csv(f'{OUT}/../build_summary_v2.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
