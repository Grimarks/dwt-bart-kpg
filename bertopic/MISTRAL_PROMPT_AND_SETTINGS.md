# Mistral topic labelling (K=65)

Labels in `mistral_labels_v3.json` (Kaggle dataset, `topic_model/`) were produced by the Kaggle notebook
`mistral_labeling_kernel.ipynb` (GPU kernel). Input: `topic_words_v3.json`, the 10 c-TF-IDF words per topic.

- Model: `unsloth/mistral-7b-instruct-v0.2`, loaded in 4-bit NF4 with double quantisation, compute dtype bfloat16.
- Prompt (paper, Appendix A):
  `[INST] Topic keywords: [KEYWORDS]\nGive a short label (max 5 words) for this topic. Return only the label. [/INST]`
  where `[KEYWORDS]` is `', '.join` of the 10 topic words. No representative documents are used.
- Generation: `transformers.pipeline("text-generation")`, `do_sample=False` (greedy, so `temperature=0.1` has no
  effect), `max_new_tokens=25`, `repetition_penalty=1.1`, `truncation=True`, `return_full_text=False`.
- Post-processing: first line of the output, stripped; a refusal-pattern filter would fall back to the first five
  topic words. Result: 0 of 65 topics flagged, so raw labels = final labels.
- The model does not always respect "max 5 words": median label length 6 words, maximum 12.

Provenance note: the notebook in this repository is a reconstruction of version 2 of the kernel (the original
kernel files could not be downloaded again). The transformers / bitsandbytes / accelerate versions were those of
the Kaggle image at the time and were not recorded.
