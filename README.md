# DWT-BART: BERTopic-guided retrieval for absent keyphrase generation

Code for the paper

> Novi Yusliani, Siti Nurmaini, Firdaus Firdaus, Rossi Passarella, Ahmad Sugama Farizim, Darrell Satriano,
> Muhammad Naufal Rachmatullah, Annisa Darmawahyuni. "Bridging the Context Gap: External Knowledge-Enhanced from
> BERTopic Improves Absent Keyphrase Generation using BART." International Journal of Intelligent Engineering and
> Systems, under review (2026).

```bibtex
@article{yusliani2026dwtbart,
  title   = {Bridging the Context Gap: External Knowledge-Enhanced from BERTopic Improves Absent Keyphrase Generation using BART},
  author  = {Yusliani, Novi and Nurmaini, Siti and Firdaus, Firdaus and Passarella, Rossi and Farizim, Ahmad Sugama and
             Satriano, Darrell and Rachmatullah, Muhammad Naufal and Darmawahyuni, Annisa},
  journal = {International Journal of Intelligent Engineering and Systems},
  note    = {Under review},
  year    = {2026}
}
```

Retrieval indices, test inputs, all predictions, statistics, the fitted BERTopic model and the six final
checkpoints are in the Kaggle dataset
[darrellsatriano/dwt-bart-kpg](https://www.kaggle.com/datasets/darrellsatriano/dwt-bart-kpg).
This repository contains code only.

## Method in one paragraph

A BERTopic model (65 topics) is fitted on 626,442 deduplicated scientific abstracts that do not overlap with
KP20k. Every document gets a topic. For each document, the 5 most similar KP20k *training* documents of the same
topic are retrieved (all-MiniLM-L6-v2 cosine; the document itself and near-duplicates with cosine >= 0.95 or an
identical title are removed). Their titles and author keyphrases (at most 30, stem-deduplicated) are inserted
between the title and the abstract:

```
title  t1 : k ; k | t2 : k ; ... . <eos> abstract
```

BART-base is fine-tuned on KP20k with this input (3 seeds) and decoded greedily.

## Condition codes

File and folder names use these codes:

| code | name in the paper | input |
|---|---|---|
| `ori` | BART | `title <eos> abstract` (no added text) |
| `topic` | topic-label variant | title + topic label + 10 top words of the topic, then abstract |
| `shuffled` | shuffled-topic control | as `topic`, but the topic of a random other topic (rng seed 2026) |
| `r5` | random-document control | keyphrases of 5 random training documents |
| `st5` | DWT-BART without titles | keyphrases of the 5 nearest same-topic training documents |
| `st5t` | DWT-BART | titles + keyphrases of the 5 nearest same-topic training documents |
| `g5` | unrestricted retrieval without titles (control) | as `st5`, neighbours from all training documents |
| `g5t` | unrestricted retrieval with titles (control) | as `st5t`, neighbours from all training documents |
| `topic_emb` | topic as a learned embedding (pilot, seed 1234) | `ori` text + learned topic vector added to the encoder output |
| `shuffled_emb` | shuffled topic embedding (pilot, seed 1234) | as `topic_emb` with the shuffled topic |

## Main result (KP20k test, F1@M x 100, mean +- sd over seeds 1234, 42, 2024)

Taken from `stats/retr9_stats/retr9_mean_std.csv` in the Kaggle dataset (macro scores of `evaluate.py`,
`<peos>` removed, present/absent decided on the original source).

| code | name in the paper | absent F1@M | present F1@M | all F1@M |
|---|---|---|---|---|
| ori | BART | 5.02 ± 0.05 | 37.80 ± 0.19 | 25.82 ± 0.07 |
| topic | topic-label variant | 4.99 ± 0.07 | 37.82 ± 0.12 | 25.77 ± 0.08 |
| shuffled | shuffled-topic control | 4.94 ± 0.03 | 37.77 ± 0.04 | 25.75 ± 0.01 |
| r5 | random-document control | 4.95 ± 0.09 | 37.85 ± 0.15 | 25.82 ± 0.14 |
| st5 | DWT-BART without titles | 6.74 ± 0.05 | 38.79 ± 0.17 | 26.99 ± 0.14 |
| st5t | DWT-BART | 7.26 ± 0.17 | 39.18 ± 0.06 | 27.58 ± 0.05 |
| g5 | unrestricted retrieval, no titles (control) | 7.28 ± 0.07 | 39.06 ± 0.05 | 27.33 ± 0.08 |
| g5t | unrestricted retrieval, with titles (control) | 7.96 ± 0.14 | 39.39 ± 0.23 | 27.98 ± 0.27 |

Unrestricted retrieval (`g5`, `g5t`) is a control: the neighbours are taken from all training documents, without
the same-topic restriction. As reported in the paper, it scores higher than DWT-BART on absent keyphrases (7.28 and
7.96 vs 7.26); `g5t` is also higher on present and all keyphrases, `g5` is lower on these two.

Inspec, Krapivin (400 and 397), NUS and SemEval-2010 results, P/R/F1 @5 and @M, the paired bootstrap tests and the
leakage-sensitivity analysis are in `stats/` of the Kaggle dataset.

## Pipeline

```
1 corpus + BERTopic  ->  2 topic assignment  ->  3 retrieval  ->  4 build inputs
  ->  5 train  ->  6 decode  ->  7 evaluate  ->  8 bootstrap / analysis
```

| step | where | files (the exact files used for the paper) |
|---|---|---|
| 0 KP20k to parquet | Mac | `data_prep/00_kp20k_json_to_parquet.py` |
| 1 corpus, dedup, BERTopic | Mac | `bertopic/00_load_dedup_clean_corpus.py`, `01_`-`07_`, `10_retrain_bertopic.py`, `11_`, `12_`, `t2_crossdomain_leakage.py`, `t3_topic_metrics.py`, `mistral_labeling_kernel.ipynb` (Kaggle GPU) |
| 2 topic assignment | Mac | `bertopic/13_assign_topics_downstream.py` |
| 4a ori / topic / shuffled inputs | Mac | `data_prep/t5v2_build_conditions.py` (imports `t5_build_conditions.py`), checks `t5v2_verify.py`, `t5v2_gate7_tokens.py` |
| 3 retrieval | GPU PC | `retrieval/retrieval_pilot_prep.py` (`make_random_nbr.py` recreates the random neighbours) |
| 4b retrieval inputs | GPU PC | `retrieval/build_retrieval_data.py` (g5, st5 via `build_st5_cpu.py`, r5), `build_title_ctx.py` (g5t), `build_title_ctx_v2.py` (st5t, via `run_build_st5t.ps1`) |
| 5-7 train, decode, evaluate | GPU PC | `training/run_kp20k.ps1` (ori, topic, shuffled), `retrieval/run_retr_pilot.ps1` (retrieval conditions), `training/run_finetune_kpgen_seq2seq_hf.py`, `training/run_decode_kpgen_seq2seq_hf.py`, `evaluation/evaluate_nopeos.py`, `evaluation/evaluate.py`, `evaluation/kp20k_checks.py` |
| 8 statistics | GPU PC | `evaluation/paired_bootstrap.py` with `evaluation/specs/*.json`, `analysis/ijies_angka_v7.py` |
| topic-embedding pilot | GPU PC | `topic_embedding/` |
| human evaluation | GPU PC | `retrieval/cek_overlap_testsets.py`, `human_eval/sample_human_eval_v2b.py`, `human_eval/analyse_ratings.py` |

There is no separate KP20k preparation script beyond `data_prep/00_kp20k_json_to_parquet.py` (one notebook cell)
and `data_prep/t5v2_build_conditions.py`.

`bertopic/00_load_dedup_clean_corpus.py` and `data_prep/00_kp20k_json_to_parquet.py` are notebook cells copied
verbatim (only a marked config block was added). `t5_build_conditions.py` is the v1 builder; its tokenisation was
replaced by `t5v2_build_conditions.py` (the version used for the paper, package `kpg_conditions_topicv3_v2`,
zip md5 `ae3a01c9fb23b6e769cbd84022203deb`); v1 is kept only because v2 imports functions from it.

## Folder layout

Scripts read and write paths relative to one research root, set with the environment variable `DWTBART_ROOT`
(Python) or the `-Root` parameter / `$env:DWTBART_ROOT` (PowerShell launchers). The layout below is the one the
scripts were run with; keep it, or edit the marked `CONFIG` lines.

```
<DWTBART_ROOT>/
  bertopic_project_wsl_original/
    dataset/Dataset_Keyphrase/          raw corpora (see "Data sources")
    output/data/                        all_dataset.parquet, src_texts.txt          (step 0)
    output_v2_deleak/
      intermediate_parquets/            normalised corpora, leakage flags           (steps 1-4)
      data/ models/ output/             corpus v2/v3, embeddings, UMAP, BERTopic_v3 (steps 5-12)
      topic_assignments/                <dataset>_<split>_topics_v3.csv              (step 13)
  kp20k_project/
    datasets/train.json valid.json test.json        KP20k (memray/kp20k)
    datasets/{inspec,krapivin,nus,semeval}/test.json
    data/kp20k_{train,validation,test}.parquet
  revisi_ijies_20266560/task5_v2/kpg_conditions_topicv3_v2/   output of data_prep (step 4a)
  PC1_PC2_FINAL_PACKAGE/
    dataset/kp20k_v3/                   the step-4a folder copied here: ori/ topic/ shuffled/ ids/ shuffled_mapping/,
                                        plus retr_<cond>/ written by the retrieval scripts
    scripts/kpg_training/               copy of training/*.py and evaluation/*.py here
  output/                               run folders, kp20k_stats/
    retrieval_pilot/                    copy of retrieval/* here; .npy files of step 3
    evaluate_nopeos.py paired_bootstrap.py kp20k_checks.py run_kp20k.ps1 resume_wrapper.py   (as on the lab PCs)
```

## Exact commands

BERTopic side (Python 3.10, `requirements-bertopic.txt`; `export DWTBART_ROOT=/path/to/root`):

```bash
python data_prep/00_kp20k_json_to_parquet.py
python bertopic/00_load_dedup_clean_corpus.py         # 662,405 -> 661,852 documents
python bertopic/01_reconstruct_sources.py             # source labels (provenance only)
python bertopic/02_build_corpora.py
python bertopic/03_exact_match.py
python bertopic/04_near_dup.py
python bertopic/05_rebuild_corpus.py                  # -> 650,838 (KP20k overlap removed)
python bertopic/07_intersource_dedup.py               # -> 626,442
python bertopic/10_retrain_bertopic.py                # K=65 (Kneedle), BERTopic_v3
python bertopic/11_persist_inference_models.py        # umap_model_v3.pkl
python bertopic/12_build_inference_kmeans.py          # kmeans_model_v3.pkl + mapping
python bertopic/t3_topic_metrics.py --model v3 --n 0  # C_V, C_NPMI, diversity (gensim 4.4.0)
python bertopic/t2_crossdomain_leakage.py             # test sets vs corpus (3 Krapivin documents)
#   Mistral labels: run bertopic/mistral_labeling_kernel.ipynb on a Kaggle GPU with topic_words_v3.json attached
python bertopic/13_assign_topics_downstream.py        # topic of every KP20k / test document
python data_prep/t5v2_build_conditions.py             # ori / topic / shuffled
python data_prep/t5v2_verify.py && python data_prep/t5v2_gate7_tokens.py
```

GPU side (Python 3.12, `requirements-training.txt`, Windows PowerShell; folder layout above):

```powershell
cd <root>\output\retrieval_pilot
python retrieval_pilot_prep.py --run                     # embeddings + 20 neighbours (global, same_topic, random)
python build_retrieval_data.py --mode global --k 5 --tag g5
python build_st5_cpu.py                                  # st5
python build_retrieval_data.py --mode random --k 5 --tag r5
python build_title_ctx.py --fmt kpt --tag g5t
.\run_build_st5t.ps1 -Root <root>                        # st5t (DWT-BART)
.\run_retr_pilot.ps1 -Cond st5 -Seed 1234                # train + decode + score one condition/seed
cd <root>\output
.\run_kp20k.ps1 -Root <root>                             # ori, topic, shuffled x 3 seeds
```

Fine-tuning arguments (identical for every condition and seed; verified in the `training_args.bin` of all six
released checkpoints):

```
python run_finetune_kpgen_seq2seq_hf.py --model_name_or_path facebook/bart-base --do_train --do_eval
  --train_file <cond>/kp20k_train_*.json --validation_file <cond>/kp20k_valid_*.json --src_column src --tgt_column tgt
  --per_device_train_batch_size 6 --per_device_eval_batch_size 12 --gradient_accumulation_steps 16
  --num_train_epochs 15 --learning_rate 6e-5 --lr_scheduler_type polynomial --warmup_steps 100
  --logging_strategy steps --logging_steps 500 --eval_strategy steps --eval_steps 5000
  --save_strategy steps --save_steps 1000 --save_total_limit 2 --overwrite_output_dir --predict_with_generate
  --seed <1234|42|2024> --fp16 true --half_precision_backend auto --output_dir <run>
python run_decode_kpgen_seq2seq_hf.py --model_name_or_path <ckpt> --tokenizer_name <ckpt> --test_file <test.json>
  --src_column src --tgt_column tgt --num_beams 1 --per_device_eval_batch_size 32 --output_dir <run> --output_file_name <hyp>
python evaluate_nopeos.py --src_file <ori test json> --pred_file <hyp> --file_prefix unused --tgt_dir <run>
  --log_file <name> --k_list 5 10 15 M
```

Scoring and bootstrap directly from the Kaggle dataset (run inside the dataset folder):

```bash
python <repo>/evaluation/evaluate_nopeos.py --evaluate_py <repo>/evaluation/evaluate.py \
  --src_file inputs_test/ori/kp20k_test_ori_v2.json --pred_file predictions/st5t/seed1234/kp20k_hypotheses.txt \
  --file_prefix unused --tgt_dir out --log_file st5t_seed1234_kp20k --k_list 5 10 15 M
python <repo>/evaluation/paired_bootstrap.py --spec <repo>/evaluation/specs/spec_main_8conditions.json \
  --evaluate_py <repo>/evaluation/evaluate.py --out_dir out/main --prefix main \
  --pairs st5t:ori st5t:st5 st5t:g5 st5t:g5t st5t:topic st5:topic topic:ori shuffled:ori topic:shuffled
python <repo>/evaluation/paired_bootstrap.py --spec <repo>/evaluation/specs/spec_pilot_topic_emb.json \
  --evaluate_py <repo>/evaluation/evaluate.py --out_dir out/pilot --prefix pilot \
  --pairs topic_emb:ori shuffled_emb:ori topic:ori topic_emb:shuffled_emb topic_emb:topic
python <repo>/human_eval/analyse_ratings.py
```

Check done before release: the first command gives KP20k absent F1@M = 0.072017 for st5t seed 1234, and the
bootstrap gives st5t - ori = +0.022480 [0.019624, 0.025331], p = 0.0001, identical to `stats/retr9_stats/`.

## Settings

- Model: `facebook/bart-base` (139.4M parameters), max source length 1,024 tokens, max target length 128.
- Optimisation: lr 6e-5, polynomial schedule, 100 warmup steps, batch 6 x gradient accumulation 16,
  15 epochs = 80,340 steps, fp16, AdamW (torch), weight decay 0, seeds 1234, 42, 2024.
- Decoding: greedy (`--num_beams 1`), max length 128. `generate()` also takes `no_repeat_ngram_size=3` and forced
  BOS/EOS from the checkpoint's `generation_config.json` (the BART-base defaults).
- Retrieval: all-MiniLM-L6-v2, cosine, 20 neighbours stored, k = 5 used; self, cosine >= 0.95 and identical
  titles removed; keyphrases stem-deduplicated, at most 30.
- BERTopic: all-MiniLM-L6-v2, UMAP 5 dims (n_neighbors 15, min_dist 0, cosine, seed 42), k-means k = 65 chosen with
  Kneedle (kneed.KneeLocator) over k = 10..500 step 5 on a 100k sample, c-TF-IDF vocabulary of 30,000 terms
  (min frequency 15), English stop words; labels by Mistral-7B-Instruct-v0.2 (`bertopic/MISTRAL_PROMPT_AND_SETTINGS.md`).
- Evaluation: DeepKPG `evaluate.py` through `evaluate_nopeos.py` (the `<peos>` separator is removed; present/absent
  decided on the original source), Krapivin scored on 400 documents and on the 397 not in the BERTopic corpus;
  paired bootstrap with 10,000 resamples, per seed and pooled over the three seeds.

## Hardware

Fine-tuning and decoding: three PCs, each with an NVIDIA RTX 4070 12 GB, Intel i9-12900K, 32 GB RAM, Windows 11
(run folders end in `_pc3`, `_pc4`, `_pc5`). BERTopic, deduplication and data preparation: Apple M4 Pro.
Mistral labelling: Kaggle GPU notebook.

## Code provenance and licence

The code is released under the MIT licence (`LICENSE`).
`evaluation/evaluate.py` is DeepKPG's `utils/evaluate.py`, unmodified (md5 `b0edbf2eadf502567578e98f147fc48d`,
identical to commit 99e29b6). `training/run_finetune_kpgen_seq2seq_hf.py` and `training/run_decode_kpgen_seq2seq_hf.py`
are DeepKPG's `sequence_generation/seq2seq/` scripts adapted to transformers 4.57.3 (replacement of removed or
renamed APIs: `as_target_tokenizer()`, `file_utils.is_offline_mode`, `use_auth_token`, `overwrite_output_dir`;
an `--early_stopping` flag and newline sanitising of written predictions in decoding); the DeepKPG MIT notice is
kept in these files and in `LICENSE`. `training/resume_wrapper.py` lets interrupted runs resume on torch < 2.6.

For this release only the following was changed in the scripts: hard-coded machine paths were replaced by
`DWTBART_ROOT` / `-Root` / command-line arguments, English docstrings were added on top, and comments naming people
or the internal workflow were removed. Lines that only served unrelated experiments were removed; no code path used
for this paper was changed. Remaining comments are in Indonesian. In `retrieval/run_retr_pilot.ps1`, `st5t` was
added to the allowed `-Cond` values: the released file comes from one lab PC whose copy predates the `st5t` runs;
the copies on the other two PCs already accepted it.

## Data sources and their licences

The training sets and raw corpora are not redistributed; the scripts rebuild everything from the public sources.

- KP20k, Inspec, Krapivin, NUS, SemEval-2010: the versions published by memray on Hugging Face
  (`memray/kp20k`, `memray/inspec`, `memray/krapivin`, `memray/nus`, `memray/semeval`). Their own terms apply.
- BERTopic corpus: the arXiv metadata snapshot (`arxiv-metadata-oai-snapshot.json`, the first 400,000 records with an abstract), the
  "arXiv Scientific Research Paper Dataset" CSV, and the `arxiv` and `pubmed` configurations of the Hugging Face
  `scientific_papers` dataset (at most 50,000 lines per split). Their own terms apply. The arXiv snapshot changes over time, so a
  rebuild may differ slightly from the 661,852-document historical corpus.

The Kaggle dataset contains only test-set inputs, predictions, retrieval indices (row numbers into the KP20k
training set), statistics, the fitted topic model and the checkpoints.
