# Lab 3 — System Architecture

End-to-end data flow from the Lab 2 dataset to the tokenization, embedding, and similarity artifacts in `output/`.

```
               ┌──────────────────────┐
               │ combined_dataset.csv │
               └──────────────────────┘
                          │
      ┌───────────────────┴───────────────────┐
      ▼                  ▼                    ▼
┌───────────┐      ┌───────────┐      ┌───────────────┐
│ Character │      │    Word   │      │ Subword / BPE │
│ Tokenizer │      │ Tokenizer │      │   Tokenizer   │
└───────────┘      └───────────┘      └───────────────┘
      ▼                  ▼                    ▼
 char vocab         word vocab          subword vocab
                                              │
                                              ▼
                      ┌────────────────────────────────────────────────┐
                      │          Embedding Analysis (Task 2)           │
                      │ random embeddings + top-50 + cosine similarity │
                      └────────────────────────────────────────────────┘
                                              │
                                              ▼
                            ┌───────────────────────────────────┐
                            │   Token Similarity (Tasks 3 & 4)  │
                            │ human selection + pair similarity │
                            └───────────────────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │   Similarity Improvement (Tasks 5 & 6)  │
                         │ character-overlap nudge + re-experiment │
                         └─────────────────────────────────────────┘
                                              │
                                              ▼
                                ┌────────────────────────────┐
                                │ output/  (JSON, CSV, .npy) │
                                └────────────────────────────┘
```

All three tokenizers read the same input and each write their own vocabulary/statistics files. Only the subword/BPE branch continues downstream — `embedding_analysis.py` builds its random `50 × 128` matrix from the BPE token frequencies, and every later stage (Tasks 3–6) reuses that same top-50 token set and embedding matrix.
