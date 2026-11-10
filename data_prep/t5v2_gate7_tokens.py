"""Gate G7 for the v2 condition files: every token is a single tokeniser token, no raw digits (21/21 PASS).

Original notes:
Task 5 v2 gate G7: every whitespace token in every src, and in every tgt keyphrase, is a single
token of the v2 tokeniser (<eos>|<digit>|<peos>|\\w+|[^\\w\\s]) and no pure-digit token survives."""
import re, json, glob, os
# ---- CONFIG: set DWTBART_ROOT to the research root (layout: see README, 'Folder layout') ----
OUT=os.path.join(os.environ.get("DWTBART_ROOT", "."), 'revisi_ijies_20266560', 'task5_v2', 'kpg_conditions_topicv3_v2')
ok_tok = re.compile(r'<eos>|<digit>|<peos>|\w+|[^\w\s]')
allok = True
for p in sorted(glob.glob(f'{OUT}/*/*.json')):
    bad_src = bad_kp = digit = 0; ex = []
    for l in open(p, encoding='utf-8'):
        r = json.loads(l)
        for t in r['src'].split(' '):
            if not ok_tok.fullmatch(t): bad_src += 1; ex.append(t)
            elif re.fullmatch(r'\d+', t): digit += 1
        for kp in r['tgt'].split(';'):
            for t in kp.split(' '):
                if not ok_tok.fullmatch(t): bad_kp += 1; ex.append(t)
                elif re.fullmatch(r'\d+', t): digit += 1
    ok = bad_src == bad_kp == digit == 0; allok &= ok
    print(f'[{"PASS" if ok else "FAIL"}] G7 {os.path.basename(p)}: bad_src_tokens={bad_src} bad_tgt_tokens={bad_kp} raw_digit_tokens={digit} {ex[:5]}')
print('G7 ALL PASS' if allok else 'G7 FAILED')
