# make_topic_emb_scripts.py - writes run_finetune_topic_emb.py and run_decode_topic_emb.py next to this file by applying
# small, exact text replacements to the ORIGINAL scripts (the originals are not changed). Stops if a replacement
# does not match exactly once, so a changed original can never produce a silently wrong script.
"""Pilot: generate run_finetune_topic_emb.py / run_decode_topic_emb.py from the training scripts by exact text replacements.
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--orig_dir", required=True, help=r"...\scripts\kpg_training (original scripts)")
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def patch(name_in, name_out, reps):
    raw = open(os.path.join(args.orig_dir, name_in), encoding="utf-8", newline="").read()
    # tolerate CR / CRLF / stray CR and trailing blanks in the original (they are invisible in the console)
    src = "\n".join(l.rstrip(" \t\r") for l in raw.replace("\r", "").split("\n"))
    for i, (old, new) in enumerate(reps, 1):
        c = src.count(old)
        if c != 1:
            key = old.strip().split("\n")[0].strip()
            near = [repr(l) for l in src.split("\n") if key[:25] in l][:4]
            sys.exit(f"STOP: {name_in}: replacement {i} matches {c} times (need exactly 1):\n{old}\nlines containing the start of it: {near}")
        src = src.replace(old, new)
    open(os.path.join(here, name_out), "w", encoding="utf-8", newline="\n").write(src)
    print("wrote", os.path.join(here, name_out), f"({len(reps)} replacements)")


IMPORT = ("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
          "from topic_bart import TopicBartForConditionalGeneration, TopicLRSeq2SeqTrainer\n")

patch("run_finetune_kpgen_seq2seq_hf.py", "run_finetune_topic_emb.py", [
    ("from nltk.stem.porter import PorterStemmer\n",
     "from nltk.stem.porter import PorterStemmer\n" + IMPORT),
    ("    tokenizer = _from_pretrained_with_auth(\n",
     '    config.num_topics = int(os.environ["TOPIC_NUM_TOPICS"])\n'
     '    config.topic_init_std = float(os.environ.get("TOPIC_INIT_STD", "0.0"))\n'
     "    tokenizer = _from_pretrained_with_auth(\n"),
    ("        AutoModelForSeq2SeqLM.from_pretrained,\n",
     "        TopicBartForConditionalGeneration.from_pretrained,\n"),
    ("        inputs, targets = [], []\n",
     "        inputs, targets, topics = [], [], []\n"),
    ("                targets.append(examples[tgt_column][i])\n",
     "                targets.append(examples[tgt_column][i])\n"
     '                topics.append(int(examples["topic_id"][i]))\n'),
    ('        model_inputs["labels"] = labels["input_ids"]\n',
     '        model_inputs["labels"] = labels["input_ids"]\n'
     '        model_inputs["topic_ids"] = topics\n'),
    ("    trainer = Seq2SeqTrainer(**trainer_kwargs)\n",
     "    trainer = TopicLRSeq2SeqTrainer(**trainer_kwargs)\n"),
])

patch("run_decode_kpgen_seq2seq_hf.py", "run_decode_topic_emb.py", [
    ("from transformers.utils.versions import require_version\n",
     "from transformers.utils.versions import require_version\n" + IMPORT),
    ("    model = AutoModelForSeq2SeqLM.from_pretrained(\n"
     "            args.model_name_or_path,\n"
     '            from_tf=bool(".ckpt" in args.model_name_or_path),\n'
     "            config=config,\n"
     "        )\n",
     "    model, _info = TopicBartForConditionalGeneration.from_pretrained(\n"
     "            args.model_name_or_path,\n"
     "            config=config,\n"
     "            output_loading_info=True,\n"
     "        )\n"
     '    assert not any("topic_emb" in k for k in _info["missing_keys"]), "topic_emb.weight missing in checkpoint: " + str(_info)\n'),
    ("        targets = examples[tgt_column]\n",
     "        targets = examples[tgt_column]\n"
     '        topics = [int(t) for t in examples["topic_id"]]\n'),
    ('        model_inputs["labels"] = labels["input_ids"]\n',
     '        model_inputs["labels"] = labels["input_ids"]\n'
     '        model_inputs["topic_ids"] = topics\n'),
    ('                attention_mask=batch["attention_mask"],\n                **gen_kwargs,\n',
     '                attention_mask=batch["attention_mask"],\n                topic_ids=batch["topic_ids"],\n                **gen_kwargs,\n'),
])
print("OK")