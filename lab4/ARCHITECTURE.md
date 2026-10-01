# Lab 4 — System Architecture

End-to-end data flow from Lab-3's tokenized outputs to the final embedding-approach comparison, as orchestrated by `src/main.py`.

```
                                          ┌──────────────────────────────┐
                                          │ data/input/  (Lab-3 outputs) │
                                          │     combined_dataset.csv     │
                                          │ subword_token_frequency.csv  │
                                          │   subword_vocabulary.json    │
                                          │      bpe_tokenizer.json      │
                                          └──────────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │ data_loader.py │
                                                 └────────────────┘
                                                          │
                                                          ▼
                                          ┌───────────────────────────────┐
                                          │      token_selection.py       │
                                          │   (manual pick of 20 tokens   │
                                          │ from the top 50 by frequency) │
                                          └───────────────────────────────┘
                                                          │
                                                          ▼
                                               ┌────────────────────┐
                                               │ 20 selected tokens │
                                               └────────────────────┘
                                                          │
             ┌─────────────────────────────┬──────────────┴──────────────┬─────────────────────────────┐
             │                             │                             │                             │
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ frequency_embedding.py  │   │  position_embedding.py  │   │  word2vec_embedding.py  │   │  code2vec_embedding.py  │
│   (frequency profile)   │   │   (position profile)    │   │       (skip-gram)       │   │  (structural context)   │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
             │                             │                             │                             │
  similarity_analysis.py        similarity_analysis.py        similarity_analysis.py        similarity_analysis.py
             │                             │                             │                             │
             └─────────────────────────────┴──────────────┬──────────────┴─────────────────────────────┘
                                                          │
                                                          ▼
                                     ┌────────────────────────────────────────┐
                                     │             comparison.py              │
                                     │  (reads the 4 *_similarity.csv files,  │
                                     │ summarizes highest / lowest / average) │
                                     └────────────────────────────────────────┘
                                                          │
                                                          ▼
                                                     ┌─────────┐
                                                     │ output/ │
                                                     └─────────┘
```

- `data_loader.py` loads `combined_dataset.csv`, `subword_token_frequency.csv`, and `bpe_tokenizer.json`. `subword_vocabulary.json` is pictured because it is one of Lab-3's four handoff files, but the pipeline never actually loads it (see the README's "Common issues" section).
- `token_selection.py` prints the 50 most frequent tokens and blocks on terminal input until 20 valid, distinct indices are entered.
- The four embedding modules run one after another (not in parallel) but are independent of each other — each reads the same 20 selected tokens and is immediately followed by its own `similarity_analysis.py` pass.
- `comparison.py` runs last and only after all four `*_similarity.csv` files exist, producing `comparison.csv`.
