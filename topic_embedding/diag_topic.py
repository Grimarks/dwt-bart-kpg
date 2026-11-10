"""Pilot diagnostic: check that topic src = ori title + topic string for every split.
"""
import json, sys
kp = sys.argv[1]
SPL = [("kp20k","train"),("kp20k","valid"),("kp20k","test"),("inspec","test"),("krapivin400","test"),("nus","test"),("semeval","test")]
for ds, sp in SPL:
    fo = open(f"{kp}/ori/{ds}_{sp}_ori_v2.json", encoding="utf-8-sig")
    ft = open(f"{kp}/topic/{ds}_{sp}_topic_before_topicv3_v2.json", encoding="utf-8-sig")
    n = bad = 0; ex = []
    for a, b in zip(fo, ft):
        if not a.strip():
            continue
        o = json.loads(a)["src"]; t = json.loads(b)["src"]
        n += 1
        title = o.split(" <eos> ", 1)[0]
        pre = t.split(" . <eos> ", 1)[0]
        if not pre.startswith(title):
            bad += 1
            if len(ex) < 3:
                ex.append((n, o[:160], t[:260]))
    print(f"{ds} {sp}: rows {n}, title-prefix mismatch {bad}")
    for e in ex:
        print("   row", e[0]); print("   ORI  :", repr(e[1])); print("   TOPIC:", repr(e[2]))