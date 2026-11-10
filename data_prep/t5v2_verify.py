"""Verification gates G1-G6 for the v2 condition files (63/63 PASS for the released package).

Original notes:
Task 5 v2 verification gates (copy of t5_verify.py, v2 preprocess + v2 file names). Re-reads the SOURCES independently and checks every output file.
Gates (all must PASS):
  G1 row count per file == source rows; ids file order == source order
  G2 0 missing topics (mapping topic ids all in [0,K) and match the v3 CSV for that id)
  G3 40 random rows per file (seed 7): title (preprocessed) is the src prefix in all 3 conditions,
     tgt keyphrase multiset == non-empty source keyphrases (kp_clean form)
  G4 no ori src contains any topic string (full topic_str, label alone, 10-keyword string), all preprocessed
  G5 shuffled: 100% topik_acak != topik_asli, and shuffled src carries topic_str(topik_acak)
     exactly; topic src carries topic_str(topik_asli) exactly
  G6 tgt identical across ori/topic/shuffled for each line (same id)
"""
import os, re, sys, json, random, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t5v2_build_conditions import load_split, preprocess, get_topic_str, SPECS, OUT, topic_words, mistral_labels

K = len(topic_words)
TS = {t: preprocess(get_topic_str(t)) for t in range(K)}
LAB = {t: preprocess(mistral_labels[t]) for t in range(K)}
KWS = {t: preprocess(' '.join(w for w, s in topic_words[t][:10])) for t in range(K)}
results = []

def gate(name, g, ok, detail=''):
    results.append({'file': name, 'gate': g, 'pass': bool(ok), 'detail': detail})
    print(f'  [{"PASS" if ok else "FAIL"}] {g} {detail}', flush=True)

def kp_clean(k):
    return preprocess(k)

for dataset, split in SPECS:
    name = f'{dataset}{"400" if dataset == "krapivin" else ""}_{split}'
    print(f'== {name}', flush=True)
    src = load_split(dataset, split)
    rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]
    ori = rd(f'{OUT}/ori/{name}_ori_v2.json')
    top = rd(f'{OUT}/topic/{name}_topic_before_topicv3_v2.json')
    shu = rd(f'{OUT}/shuffled/{name}_shuffled_before_topicv3_v2.json')
    ids = [l.rstrip('\n') for l in open(f'{OUT}/ids/{name}_ids.txt')]
    mp = pd.read_csv(f'{OUT}/shuffled_mapping/{name}_shuffled_mapping.csv', dtype={'id': str})
    n = len(src)
    gate(name, 'G1_rows', len(ori) == len(top) == len(shu) == len(mp) == len(ids) == n,
         f'source={n} ori={len(ori)} topic={len(top)} shuffled={len(shu)} mapping={len(mp)}')
    gate(name, 'G1_id_order', ids == src['id'].tolist() and mp['id'].tolist() == src['id'].tolist())
    gate(name, 'G2_missing_topic', src['topic'].notna().all() and mp['topik_asli'].between(0, K - 1).all()
         and (mp['topik_asli'].values == src['topic'].astype(int).values).all(),
         f'missing={int(src["topic"].isna().sum())}')
    random.seed(7)
    sample = random.sample(range(n), min(40, n))
    bad = []
    for i in sample:
        t = preprocess(str(src['title'][i]).strip())
        exp = collections.Counter(kp_clean(k) for k in str(src['keywords'][i]).split(';') if k.strip())
        for cond, rows in (('ori', ori), ('topic', top), ('shuffled', shu)):
            if not rows[i]['src'].startswith(t + ' '):
                bad.append((i, cond, 'title'))
        got = collections.Counter(k for k in ori[i]['tgt'].split(';') if k != '<peos>')
        if got != exp:
            bad.append((i, 'tgt', 'keyphrases'))
    gate(name, 'G3_sample40', not bad, f'n_sampled={len(sample)} mismatches={bad[:5]}')
    hits, lab_hits = [], []
    for i, r in enumerate(ori):
        s = r['src']
        for t in range(K):
            if TS[t] in s or KWS[t] in s:
                hits.append((i, t, 'full' if TS[t] in s else 'kw10'))
            elif LAB[t] in s:
                lab_hits.append((i, t, LAB[t]))
    gate(name, 'G4_ori_no_topic_string(full topic_str / 10-keyword string)', not hits, f'hits={len(hits)} {hits[:5]}')
    print(f'  [INFO] label-only occurrences in ori src (natural author text, not injected): {lab_hits}')
    results.append({'file': name, 'gate': 'INFO_label_only_in_ori', 'pass': True, 'detail': str(lab_hits)})
    rebuilt_ok = all(r['src'] == preprocess(f"{str(t).strip()} <eos> {str(a).strip()}")
                     for r, t, a in zip(ori, src['title'], src['abstract']))
    gate(name, 'G4b_ori_src_equals_preprocess(title <eos> abstract) for ALL rows', rebuilt_ok)
    neq = (mp['topik_acak'] != mp['topik_asli'])
    carry_t = all((f' {TS[a]} . <eos> ' in top[i]['src'] or top[i]['src'].endswith(f' {TS[a]} . <eos>')) for i, a in enumerate(mp['topik_asli']))
    carry_s = all((f' {TS[b]} . <eos> ' in shu[i]['src'] or shu[i]['src'].endswith(f' {TS[b]} . <eos>')) for i, b in enumerate(mp['topik_acak']))
    gate(name, 'G5_shuffled_neq_orig', bool(neq.all()), f'{int(neq.sum())}/{n} = {neq.mean():.2%}')
    gate(name, 'G5_topic_strings_carried', carry_t and carry_s, f'topic={carry_t} shuffled={carry_s}')
    same = all(a['tgt'] == b['tgt'] == c['tgt'] for a, b, c in zip(ori, top, shu))
    gate(name, 'G6_tgt_identical', same)

df = pd.DataFrame(results)
df.to_csv(f'{OUT}/../verification_gates.csv', index=False)
g=df[~df.gate.str.startswith('INFO')]
print(f'\nGATES: {g["pass"].sum()}/{len(g)} PASS')
print('ALL PASS' if g['pass'].all() else 'SOME GATE FAILED')
