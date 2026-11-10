#!/usr/bin/env python
# coding=utf-8
# Adapted from DeepKPG (https://github.com/uclanlp/DeepKPG, commit 99e29b6,
# sequence_generation/seq2seq/), MIT License, Copyright (c) 2022 Natural Language Processing @UCLA;
# DeepKPG's script is itself based on a Hugging Face transformers example (Apache-2.0).
# Changes for transformers 4.57.3 / this study are described in README.md.
# based on https://github.com/huggingface/transformers/blob/v4.16.2/examples/pytorch/summarization/run_summarization.py
"""Fine-tune facebook/bart-base on a KP20k condition (src -> tgt JSON lines). Paper settings:
lr 6e-5 polynomial, warmup 100, per-device batch 6, gradient accumulation 16, 15 epochs (80,340 steps),
fp16, max source length 1,024 tokens, seeds 1234 / 42 / 2024. Exact arguments: training/run_kp20k.ps1.

Original notes:
Fine-tuning a 🤗 Transformers model on keyphrase generation.
"""

import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk
import numpy as np
import random
import torch
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
def is_offline_mode():
    # transformers.file_utils.is_offline_mode was relocated/removed across versions;
    # reimplemented inline (same semantic: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE env var)
    # to avoid depending on an unstable internal import path.
    return os.environ.get("HF_HUB_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "0")).upper() in ("1", "ON", "YES", "TRUE")


def _from_pretrained_with_auth(load_fn, *args, use_auth_token=None, **kwargs):
    # `use_auth_token=` was renamed to `token=` in newer transformers; older
    # versions (e.g. the PC1/PC2 kpg2 env, pinned only to a >=4.15.0 floor)
    # don't accept `token=` and raise a TypeError. `from_pretrained` methods
    # take **kwargs, so their signature can't be introspected to tell which
    # name is expected -- try the new name first (what Kaggle's newer
    # transformers expects) and fall back to the old name on TypeError (what
    # PC1/PC2 expects), so this script runs unmodified on both.
    auth_val = True if use_auth_token else None
    try:
        return load_fn(*args, token=auth_val, **kwargs)
    except TypeError as e:
        if "token" not in str(e):
            raise
        return load_fn(*args, use_auth_token=auth_val, **kwargs)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.integrations import TensorBoardCallback
from nltk.stem.porter import PorterStemmer

# Inisialisasi stemmer di luar atau di dalam fungsi (lebih baik di luar loop agar efisien)
stemmer = PorterStemmer()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    src_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the source texts (for keyphrase generation)."},
    )
    tgt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the keyphrases (for keyphrase generation)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, enables early stopping: training stops once this many consecutive evaluations show no "
            "improvement in --metric_for_best_model. Requires --load_best_model_at_end true and a matching "
            "--eval_strategy/--save_strategy (same type and step interval). Disabled (fixed-epoch training) when "
            "left unset — default behavior is unchanged."
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum change in --metric_for_best_model to qualify as an improvement for early stopping. "
            "Only used when --early_stopping_patience is set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


kpgen_name_mapping = {
    "kp20k": ("src", "tgt"),
}


def load_and_preprocess_datasets(model_args, data_args, training_args, tokenizer, model):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # keyphrases (unless you specify column names for this with the `src_column` and `tgt_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = kpgen_name_mapping.get(data_args.dataset_name, None)
    if data_args.src_column is None:
        src_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        src_column = data_args.src_column
        if src_column not in column_names:
            raise ValueError(
                f"--src_column' value '{data_args.src_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.tgt_column is None:
        tgt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        tgt_column = data_args.tgt_column
        if tgt_column not in column_names:
            raise ValueError(
                f"--tgt_column' value '{data_args.tgt_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[src_column])):
            if examples[src_column][i] is not None and examples[tgt_column][i] is not None:
                inputs.append(examples[src_column][i])
                targets.append(examples[tgt_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        # tokenizer.as_target_tokenizer() was removed in newer transformers; text_target=
        # is the modern equivalent, same tokenization semantics for the target sequence.
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # train_dataset = train_dataset.select(range(data_args.max_train_samples))
            rand_data_idx = random.sample(list(range(len(train_dataset))), data_args.max_train_samples)
            train_dataset = train_dataset.select(rand_data_idx)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    return train_dataset, eval_dataset, predict_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log to both file and stderr
    fileHandler = logging.FileHandler("{}/train.log".format(training_args.output_dir))
    logger.addHandler(fileHandler)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    # overwrite_output_dir was removed from TrainingArguments in newer transformers
    # versions; getattr keeps this working on both old and new versions (defaults to
    # True — i.e. skip last-checkpoint auto-resume detection — when the field is gone).
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not getattr(training_args, "overwrite_output_dir", True):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = _from_pretrained_with_auth(
        AutoConfig.from_pretrained,
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = _from_pretrained_with_auth(
        AutoTokenizer.from_pretrained,
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    model = _from_pretrained_with_auth(
        AutoModelForSeq2SeqLM.from_pretrained,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )

    if 't5' not in model_args.model_name_or_path:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    # load and preprocess the datasets
    train_dataset, eval_dataset, predict_dataset = load_and_preprocess_datasets(model_args, data_args, training_args, tokenizer, model)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    # metric = load_metric("rouge")
    def postprocess_text_kpgen(preds, labels):
        preds = [[x.strip() for x in pred.strip().split(';')] for pred in preds]
        labels = [[x.strip() for x in label.strip().split(';')] for label in labels]
        return preds, labels

    def stem_keyphrase(phrase):
        # Melakukan stemming per kata di dalam sebuah frasa
        return " ".join([stemmer.stem(w) for w in phrase.split()])
    
    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
        
    #     # --- NEW FIX: Replace -100 in predictions before decoding ---
    #     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #     # ------------------------------------------------------------
        
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
    #     if data_args.ignore_pad_token_for_loss:
    #         # Replace -100 in the labels as we can't decode them.
    #         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     decoded_preds, decoded_labels = postprocess_text_kpgen(decoded_preds, decoded_labels)

    #     result_metrics = {}
    #     # Define the thresholds for @k
    #     k_thresholds = [5, 10, 15, 'all']

    #     for k in k_thresholds:
    #         micro_tp, micro_fp, micro_fn = 0, 0, 0
    #         macro_f1_scores = []

    #         for preds_list, labels_list in zip(decoded_preds, decoded_labels):
    #             # Slice the predictions for @k (if 'all', take the whole list)
    #             top_k_preds = preds_list[:k] if k != 'all' else preds_list
                
    #             # Convert to sets for strict exact matching with strip. 
    #             # Lowercasing is applied here as it is standard practice for keyphrase evaluation.
    #             # Terapkan lower, strip, dan STEMMING
    #             pred_set = set([stem_keyphrase(p.strip().lower()) for p in top_k_preds if p.strip()])
    #             label_set = set([stem_keyphrase(l.strip().lower()) for l in labels_list if l.strip()])

    #             # Calculate TP, FP, FN for the current sample
    #             tp = len(pred_set.intersection(label_set))
    #             fp = len(pred_set - label_set)
    #             fn = len(label_set - pred_set)

    #             # Accumulate for Micro calculation
    #             micro_tp += tp
    #             micro_fp += fp
    #             micro_fn += fn

    #             # Calculate Macro F1 for the current sample
    #             sample_p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    #             sample_r = tp / len(label_set) if len(label_set) > 0 else 0.0
    #             sample_f1 = (2 * sample_p * sample_r) / (sample_p + sample_r) if (sample_p + sample_r) > 0 else 0.0
    #             macro_f1_scores.append(sample_f1)

    #         # Overall Micro F1 calculation
    #         micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    #         micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    #         micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    #         # Overall Macro F1 calculation
    #         macro_f1 = np.mean(macro_f1_scores) if macro_f1_scores else 0.0

    #         # Format the dictionary keys
    #         k_str = str(k)
    #         result_metrics[f'micro_f1@{k_str}'] = micro_f1
    #         result_metrics[f'macro_f1@{k_str}'] = macro_f1

    #     return result_metrics

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # --- NEW FIX: Replace -100 in predictions before decoding ---
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # ------------------------------------------------------------
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text_kpgen(decoded_preds, decoded_labels)
    
        result_metrics = {}
        k_thresholds = [5, 10, 15, 'all']
    
        for k in k_thresholds:
            micro_tp, micro_fp, micro_fn = 0, 0, 0
            macro_f1_scores = []
    
            for preds_list, labels_list in zip(decoded_preds, decoded_labels):
                # Ambil top K prediksi
                top_k_preds = preds_list[:k] if k != 'all' else preds_list
                
                # Terapkan lower, strip, dan STEMMING
                pred_set = set([stem_keyphrase(p.strip().lower()) for p in top_k_preds if p.strip()])
                label_set = set([stem_keyphrase(l.strip().lower()) for l in labels_list if l.strip()])
    
                # Hitung TP, FP, FN
                tp = len(pred_set.intersection(label_set))
                fp = len(pred_set - label_set)
                fn = len(label_set - pred_set)
    
                micro_tp += tp
                micro_fp += fp
                micro_fn += fn
    
                # Macro P, R, F1 untuk sample saat ini
                # Menggunakan panjang pred_set sebagai pembagi precision (jumlah prediksi unik yg ter-generate)
                sample_p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
                sample_r = tp / len(label_set) if len(label_set) > 0 else 0.0
                sample_f1 = (2 * sample_p * sample_r) / (sample_p + sample_r) if (sample_p + sample_r) > 0 else 0.0
                
                macro_f1_scores.append(sample_f1)
    
            # Micro P, R, F1 keseluruhan
            micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
            micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
            micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
            # Macro F1 keseluruhan
            macro_f1 = np.mean(macro_f1_scores) if macro_f1_scores else 0.0
    
            k_str = str(k)
            result_metrics[f'micro_f1@{k_str}'] = micro_f1
            result_metrics[f'macro_f1@{k_str}'] = macro_f1
    
        return result_metrics
    
    # Initialize our Trainer
    callbacks = [TensorBoardCallback()]
    if data_args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=data_args.early_stopping_patience,
            early_stopping_threshold=data_args.early_stopping_threshold,
        ))

    # Seq2SeqTrainer's `tokenizer=` kwarg was renamed to `processing_class=` in newer
    # transformers; detect which one this version accepts instead of hardcoding either.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=callbacks,
    )
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    """
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    """

    # Explicit cleanup before returning: when launched under torchrun (DDP), the NCCL
    # process group can otherwise fail to tear down cleanly on interpreter exit, leaving
    # worker processes (and the subprocess.run() call waiting on them, and the Kaggle
    # session itself) hanging indefinitely even though training already finished and
    # saved successfully. No-op under plain DataParallel (distributed never initialized).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    # Forcefully terminate immediately instead of going through normal Python interpreter
    # shutdown (atexit handlers, thread joins, CUDA context teardown) — a known source of
    # post-training hangs under torchrun/DDP. Everything that matters (model, metrics,
    # checkpoints) is already flushed to disk by the time main() returns.
    os._exit(0)