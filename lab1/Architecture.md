# Lab 1 — System Architecture

End-to-end data flow from a GitHub repository URL to the generated reports.

```
                                        USER
                                          │
                                          │  GitHub repository URL
                                          ▼
                          ┌──────────────────────────────┐
                          │      mine_repository.py      │
                          │ (entry point / orchestrator) │
                          └──────────────────────────────┘
                                          │
                                          ▼
                         ┌────────────────────────────────┐
                         │      repository_setup.py       │
                         │ (validate URL, clone or reuse) │
                         └────────────────────────────────┘
                                          │
                                          ▼
                                Local Git Repository
                           (data/repo/<repository-name>/)
                                          │
                      ┌───────────────────┴───────────────────┐
                      │                                       │
                      ▼                                       ▼
             ┌─────────────────┐                 ┌────────────────────────┐
             │ file_metrics.py │                 │ git_history_metrics.py │
             └─────────────────┘                 └────────────────────────┘
                      │                                       │
                      ▼                                       ▼
              file_metrics.csv                     git_history_metrics.csv
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                             ┌────────────────────────┐
                             │  report_generator.py   │
                             │ (writes final reports) │
                             └────────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
         ┌────────────────────────────┐              ┌────────────┐
         │ repository_statistics.json │              │ summary.md │
         └────────────────────────────┘              └────────────┘
```
