# Lab 2 — Pipeline Architecture

End-to-end data flow from Lab-1's per-repository CSV outputs to Lab-2's cleaned, merged, analysis-ready datasets.

```
                                   LAB-1 OUTPUTS PER REPOSITORY
                         (fastapi, flask, pytest, requests, scikit-learn)

                                   ┌───────────────────────────┐
                                   │     file_metrics.csv      │
                                   │  git_history_metrics.csv  │
                                   └───────────────────────────┘
                                                 │
                                                 ▼
                                     data/input/<repository>/
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │         prepare_datasets.py         │
                              │  reads 5 repos, tags each row with  │
                              │  its repository, concatenates them  │
                              └─────────────────────────────────────┘
                                                 │
                                                 ▼
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
          ┌───────────────────────────┐                   ┌──────────────────────────────┐
          │  source_code_dataset.csv  │                   │  commit_history_dataset.csv  │
          └───────────────────────────┘                   └──────────────────────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                            ┌────────────────────────────────────────┐
                            │           clean_and_merge.py           │
                            │   cleans + validates both datasets,    │
                            │  LEFT JOINs on repository + file_path  │
                            └────────────────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │  combined_dataset.csv  │
                                    └────────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │         analyze_datasets.py         │
                              │  computes statistics for all three  │
                              │  datasets and writes them as JSON   │
                              └─────────────────────────────────────┘
                                                 │
                                                 ▼
                 ┌───────────────────────────────┼───────────────────────────────┐
                 ▼                               ▼                               ▼
      ┌────────────────────┐          ┌────────────────────┐          ┌─────────────────────┐
      │    source_code_    │          │  commit_history_   │          │  combined_dataset_  │
      │  statistics.json   │          │  statistics.json   │          │   statistics.json   │
      └────────────────────┘          └────────────────────┘          └─────────────────────┘
                 │                               │                               │
                 └───────────────────────────────┼───────────────────────────────┘
                                                 │
                                                 ▼
                                              output/
```

## Stages

- **`prepare_datasets.py`** — loads each repository's `file_metrics.csv` and `git_history_metrics.csv`, tags every row with its `repository`, and concatenates all five repositories into two combined CSVs.
- **`clean_and_merge.py`** — normalizes and validates both datasets independently, then `LEFT JOIN`s them on the composite key `repository + file_path` so every current source file is kept even if it has no matching history record.
- **`analyze_datasets.py`** — computes descriptive statistics for the source-code, commit-history, and combined datasets, writing each as its own JSON file.
- **`main.py`** — orchestrates the three stages above in order; not shown separately since it adds no transformation of its own.
