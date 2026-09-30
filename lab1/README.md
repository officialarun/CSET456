# Lab 1 — Mining and Profiling a Software Repository

A revision guide to the Lab 1 implementation: what it does, how it works, and how to run it.

## 1. Objective

Treat a Git/GitHub repository as a source of software-engineering data. Given a public GitHub URL, the tool:

1. Clones (or reuses) the repository locally.
2. Analyzes the **current snapshot** — files, directories, languages, lines of code (LOC).
3. Mines the **Git history** — commits, contributors, file-change frequency, code churn.
4. Writes the results out as CSV, JSON, and a Markdown report.

This distinction — *"what is in the repo right now"* vs. *"how did the repo evolve"* — is the central idea of the lab:

| Question | Approach | Module |
|---|---|---|
| What is currently in the repository? | Source-code analysis | `file_metrics.py` |
| How has the repository evolved? | Repository mining | `git_history_metrics.py` |

## 2. Directory Structure

```
lab1/
├── src/
│   ├── mine_repository.py       # entry point / orchestrator
│   ├── repository_setup.py      # URL validation + clone/reuse
│   ├── file_metrics.py          # current-snapshot analysis
│   ├── git_history_metrics.py   # git history mining
│   └── report_generator.py      # JSON + Markdown report writer
├── data/repo/<repo-name>/       # cloned repositories (created at runtime)
├── output/<repo-name>/          # generated datasets/reports per repo
├── requirements.txt
└── README.md
```

`data/` and `output/` are created at runtime and are per-repository — each analyzed repo gets its own subfolder named after it.

## 3. How to Run

From the `lab1` directory:

```bash
python src/mine_repository.py
```

You'll be prompted:

```
Enter GitHub repository URL:
```

Enter any public GitHub URL, e.g. `https://github.com/psf/requests`. The tool needs Python 3 and Git on the `PATH`, plus internet access if the repository isn't already cloned locally.

> **Note:** `requirements.txt` lists `pydriller`, but the current implementation doesn't import it anywhere — all Git operations go through `subprocess` calls to the `git` CLI directly, and everything else uses the standard library (`pathlib`, `collections`, `csv`, `json`, `datetime`, `urllib.parse`).

### Execution stages

```
[1/4] Repository setup   → validate URL, clone or reuse local copy
[2/4] File metrics       → walk current snapshot, compute LOC/languages/types
[3/4] Git history        → scan commit log, compute churn/contributor stats
[4/4] Report generation  → write JSON + Markdown summary
```

Output lands in `output/<repository-name>/`:

```
file_metrics.csv
git_history_metrics.csv
repository_statistics.json
summary.md
```

If the repo was already cloned into `data/repo/<name>/`, the tool reuses it as-is rather than re-cloning (it does **not** `git pull` to refresh it — delete the folder or pull manually if you need up-to-date history).

## 4. Module Responsibilities

### `mine_repository.py` — orchestrator
Prompts for the URL, calls the other three modules in sequence, prints progress, and catches `KeyboardInterrupt`/exceptions so failures don't crash with a raw traceback.

### `repository_setup.py` — acquisition
- `get_repository_name(url)`: validates the host is `github.com`/`www.github.com`, extracts the repo name from the URL path (strips a trailing `.git`). Raises `ValueError` on anything else.
- `setup_repository(url)`: computes the local path `data/repo/<name>/`. If it already exists and contains a `.git` folder, reuses it. If it exists but isn't a Git repo, raises an error. Otherwise clones with:
  ```
  git clone --filter=blob:none --no-tags <url> <path>
  ```
  (`--filter=blob:none` skips downloading blob contents until needed — a partial/"treeless" clone that's much faster for large repos.)

### `file_metrics.py` — current-snapshot analysis
Walks the repository with `rglob("*")`, skipping anything under `.git`. For every file it classifies the extension against a fixed `SOURCE_EXTENSIONS` map to decide if it's a "source file":

| Extension | Language | Extension | Language |
|---|---|---|---|
| `.py` | Python | `.java` | Java |
| `.pyx`, `.pxd`, `.pxi` | Cython | `.js` | JavaScript |
| `.c` | C | `.ts` | TypeScript |
| `.h` | C/C++ | `.rs` | Rust |
| `.cpp`, `.hpp` | C++ | `.go` | Go |
| | | `.f90`, `.f` | Fortran |

For each source file it records `file_path`, `language`, `extension`, `loc`, `size_bytes` and writes them to `file_metrics.csv`. It also computes total/average LOC, the top 10 largest source files by LOC, a language breakdown, and a file-type distribution (by extension, across *all* files, not just source files).

**LOC = physical line count.** Files are opened in binary mode and lines are counted directly — comments, blank lines, and generated code are all included. It's not a logical-statement count.

### `git_history_metrics.py` — history mining
Uses two different scopes to keep the analysis cheap on large repos:

- **Full history** (`git log --all --format=%H%x09%an%x09%aI`) — lightweight, metadata-only. Used for total commit count, contributor list (by distinct author *name* — two names for the same person count as two contributors), most active contributor, and commits grouped by `YYYY-MM`.
- **Latest 400 commits** (`git log -400 --numstat --format=COMMIT%x09%H%x09%aI`) — used for anything requiring file-level diff stats: per-file change counts, total lines added/deleted, and average files-changed-per-month. `--numstat` gives additions/deletions/path per file without generating full diffs, which is what keeps this fast.

Key derived numbers:
- **Average files changed per month** = total file-change *events* in the 400-commit window ÷ number of distinct calendar months in that window. This counts events, not unique files — a file touched in 5 commits contributes 5.
- **Average added/deleted lines per commit** = totals from `--numstat` ÷ number of commits in the 400-commit window.

Writes `git_history_metrics.csv` (`file_path,change_count`, most-changed first).

### `report_generator.py` — output
Combines both modules' return dicts into `repository_statistics.json` (`{"repository": {...}, "git_history": {...}}`) and renders a human-readable `summary.md` with the inventory, language/file-type tables, largest files, git history stats, top changed files, and commits-per-month table.

## 5. Generated Files Reference

**`file_metrics.csv`** — one row per source file: `file_path,language,extension,loc,size_bytes`

**`git_history_metrics.csv`** — one row per file touched in the last 400 commits: `file_path,change_count`

**`repository_statistics.json`** — everything above in one machine-readable file, under `repository` and `git_history` keys.

**`summary.md`** — human-readable version of the same, as Markdown tables.

Sample output already exists in this repo under `output/` for `fastapi`, `flask`, `pytest`, `requests`, and `scikit-learn` — useful as reference for what real results look like.

## 6. Points to Remember for Revision

- **Two analysis dimensions**: current snapshot (source-code analysis) vs. Git log (repository mining) — this split maps directly onto `file_metrics.py` vs. `git_history_metrics.py`.
- **Why 400 commits for file-change data?** Full-history diff stats are expensive on large repos; a recent window keeps `--numstat` parsing fast while still being representative. Commit *count*, *contributors*, and *commits-per-month* still use the entire history since that only needs cheap metadata (no diffs).
- **LOC is physical lines**, not logical statements — no comment/blank-line stripping.
- **Language detection is extension-based**, not a real parser — a `.h` file is always "C/C++" regardless of its actual contents.
- **`change_count` ≠ unique files changed** — it's the number of times a file appears across the analyzed commits.
- **Existing local clones are reused, not refreshed.** Re-running against a repo you've already cloned analyzes whatever is currently checked out locally, not the latest upstream state.
- Only GitHub URLs are accepted (`get_repository_name` explicitly checks the host).

## 7. Common Issues

- **"Please enter a valid GitHub repository URL"** — the URL's host isn't `github.com`, or the path doesn't have at least a `<user>/<repo>` segment.
- **"`<path>` exists but is not a Git repository"** — something other than a clone is already sitting at `data/repo/<name>/`; remove or rename it.
- **Clone failures** — usually network access or an invalid/private repository URL; the underlying `git` stderr is surfaced in the error message.
- **Large repositories are slow on the full-history pass** (`git log --all`) since every commit is read for metadata — this is expected and is why file-level diff analysis is capped at 400 commits.
