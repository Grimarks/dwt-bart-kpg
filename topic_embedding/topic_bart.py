# topic_bart.py - BART with a learned topic embedding added to the ENCODER OUTPUT (between encoder and decoder).
#
# Idea: every document has a topic id (0..K-1). A learned vector e[topic_id] (size d_model) is added to every
# encoder output state before the decoder cross-attends to it. The encoder input stays the plain "title <eos>
# abstract" text (no topic words in the text). Everything else is the normal BartForConditionalGeneration.
#
# Notes
# - topic_ids must be passed to forward() and generate(); a missing topic_ids raises an error (never silent).
# - The embedding is saved in model.safetensors as topic_emb.weight and config.num_topics in config.json.
#   Load it ONLY with TopicBartForConditionalGeneration (AutoModelForSeq2SeqLM would silently drop it).
"""Pilot (seed 1234 only): BART with a learned topic embedding added to the encoder output (topic_emb / shuffled_emb).
"""
import inspect
import os

import torch
from torch import nn
from transformers import BartForConditionalGeneration, Seq2SeqTrainer
from transformers.modeling_outputs import BaseModelOutput


class TopicBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        n = int(getattr(config, "num_topics", 0) or 0)
        if n <= 0:
            raise ValueError("config.num_topics must be set to the number of topics (> 0)")
        self.topic_emb = nn.Embedding(n, config.d_model)
        nn.init.normal_(self.topic_emb.weight, mean=0.0, std=float(getattr(config, "topic_init_std", 0.0)))
        self._topic_ids = None
        self.model.encoder.register_forward_hook(self._add_topic_to_encoder_output)

    def _init_weights(self, module):
        # from_pretrained() initialises the missing topic_emb through this hook: keep our own init
        # (default 0.0 = start exactly from pretrained BART; the embedding is learned from there).
        if module is getattr(self, "topic_emb", None):
            nn.init.normal_(module.weight, mean=0.0, std=float(getattr(self.config, "topic_init_std", 0.0)))
        else:
            super()._init_weights(module)

    def _add_topic_to_encoder_output(self, module, inputs, output):
        t = self._topic_ids
        if t is None:
            raise RuntimeError("topic_ids is None: pass topic_ids to forward()/generate() (topic embedding model)")
        if isinstance(output, BaseModelOutput):
            h = output.last_hidden_state
            e = self.topic_emb(t.to(h.device)).to(h.dtype).unsqueeze(1)
            output.last_hidden_state = h + e
            return output
        h = output[0]
        e = self.topic_emb(t.to(h.device)).to(h.dtype).unsqueeze(1)
        return (h + e,) + tuple(output[1:])

    def forward(self, *args, topic_ids=None, **kwargs):
        self._topic_ids = topic_ids
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._topic_ids = None

    def generate(self, *args, topic_ids=None, **kwargs):
        self._topic_ids = topic_ids
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._topic_ids = None


# Make inspect.signature(model.forward) list topic_ids, so Trainer(remove_unused_columns=True) keeps the column.
_base_sig = inspect.signature(BartForConditionalGeneration.forward)
_params = [p for p in _base_sig.parameters.values() if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)]
# NOTE: do NOT add **kwargs here: Trainer would then think the model handles num_items_in_batch itself and
# skip dividing the loss by gradient_accumulation_steps (changes the effective loss scale vs the baseline).
_params.append(inspect.Parameter("topic_ids", inspect.Parameter.KEYWORD_ONLY, default=None))
TopicBartForConditionalGeneration.forward.__signature__ = _base_sig.replace(parameters=_params)


class TopicLRSeq2SeqTrainer(Seq2SeqTrainer):
    """Same as Seq2SeqTrainer, but the new topic embedding gets its own (larger) learning rate: it starts
    from scratch, while all pretrained weights keep the normal --learning_rate. Set with env TOPIC_EMB_LR."""

    def create_optimizer(self):
        opt = super().create_optimizer()
        if getattr(self, "_topic_group_done", False):
            return opt
        p = self.model.topic_emb.weight
        for g in opt.param_groups:
            g["params"] = [q for q in g["params"] if q is not p]
        opt.add_param_group({"params": [p], "lr": float(os.environ.get("TOPIC_EMB_LR", "1e-3")), "weight_decay": 0.0})
        self._topic_group_done = True
        return opt
