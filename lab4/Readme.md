# Lab 4 — Embedding Approaches on Software Repository Tokens

A revision guide to the Lab 4 implementation: what it does, how it works, and how to run it.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full pipeline diagram.

## 1. Objective

Lab 3 produced a BPE tokenizer, a subword vocabulary, and per-token frequency counts from a combined dataset of file paths across 5 repositories. Lab 4 picks up from there and experiments with turning tokens into numeric vectors ("embeddings") using four different approaches, then compares how each approach behaves:

1. **Frequency embedding** — a hand-engineered vector built from how often a token occurs, globally and per repository.
2. **Position embedding** — a hand-engineered vector built from *where* a token tends to appear inside a tokenized file path.
3. **Word2Vec** — an actual trained skip-gram model (via `gensim`) over tokenized file paths.
4. **Code2Vec-inspired embedding** — a deterministic, hash-based approximation of Code2Vec's "aggregate the contexts around a token" idea, applied to file-path structure instead of AST paths.

The same 20 manually selected tokens are run through all four approaches, pairwise cosine similarity is computed for each, and `comparison.py` summarizes how the four similarity distributions differ.

> **Important dataset constraint:** `combined_dataset.csv` only contributes `repository` and `file_path` to this pipeline — there is no source code or AST data available. Every approach that needs "context" (position, Word2Vec, Code2Vec-inspired) gets it by BPE-tokenizing the file-path string (e.g. `src/utils/parser.py` → `["src", "/", "utils", "/", "parser", ".py"]`), not by reading actual code. This is a deliberate, documented limitation of the lab, not an oversight.

## 2. Directory Structure

```
lab4/
├── data/
│   └── input/                          # Lab-3 outputs (already present)
│       ├── combined_dataset.csv
│       ├── subword_token_frequency.csv
│       ├── subword_vocabulary.json
│       └── bpe_tokenizer.json
├── src/
│   ├── main.py                  # entry point / orchestrator
│   ├── data_loader.py           # loads Lab-3 artifacts
│   ├── token_selection.py       # interactive manual selection of 20 tokens
│   ├── frequency_embedding.py   # custom embedding 1
│   ├── position_embedding.py    # custom embedding 2
│   ├── word2vec_embedding.py    # trained Word2Vec embedding
│   ├── code2vec_embedding.py    # Code2Vec-inspired structural embedding
│   ├── similarity_analysis.py   # pairwise cosine similarity, used by all 4 approaches
│   ├── comparison.py            # cross-approach summary
│   └── requirements.txt
├── output/                      # created at runtime (not present until main.py runs)
├── Readme.md
└── ARCHITECTURE.md
```

## 3. How to Run

From the `lab4` directory:

```bash
pip install -r src/requirements.txt
python3 src/main.py
```

Partway through the run you'll be dropped into a prompt like:

```
TOP 50 MOST FREQUENT TOKENS
 1. '_'                       frequency = 6355.0
 2. '/'                       frequency = 5356.0
 ...
Select 20 tokens manually.
Enter their numbers separated by spaces.
Example:
2 4 5 8 11 14 17 21 25 28 31 34 36 39 41 43 45 47 49 50

Enter 20 token numbers:
```

Type **exactly 20 distinct numbers between 1 and 50**, separated by spaces, and press Enter. Anything else (wrong count, duplicates, out-of-range, non-numeric) is rejected and re-prompted.

`main.py` is otherwise non-interactive and uses paths relative to the script's own location (`Path(__file__).resolve().parent.parent`), so it can be re-run safely from anywhere. To run it unattended (e.g. for scripting or regression-testing), pipe the 20 numbers in via stdin:

```bash
echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" | python3 src/main.py
```

## 4. Module Responsibilities

### `data_loader.py`
Loads the three Lab-3 artifacts the pipeline actually uses: `combined_dataset.csv` (validated to contain `repository` and `file_path`), `subword_token_frequency.csv`, and `bpe_tokenizer.json` — the last one via `Tokenizer.from_file()`, not `json.load()`, since the file is a serialized tokenizer model. A fourth function, `load_subword_vocabulary()`, can load `subword_vocabulary.json`, but `main.py` never calls it (see Common Issues).

### `token_selection.py`
Sorts tokens by frequency (descending), takes the top 50, prints them with 1-based indices, and blocks on `input()` until the user types 20 valid, distinct indices in that range. Saves the selections (rank, token, frequency) to `output/selected_tokens.json` and returns the plain list of 20 token strings used by every later stage.

### `frequency_embedding.py` — Custom Approach 1
For each selected token: `normalized_frequency` = its global frequency divided by the **max frequency among the 20 selected tokens**; `log_frequency` = `log1p(frequency)`. It then appends one value per repository: `normalized_frequency × (that repository's share of all rows in combined_dataset.csv)`. The final vector is `[normalized_frequency, log_frequency, repo_1_component, ..., repo_N_component]`, L2-normalized.

> Watch out: the per-repository components are **not** a per-repository token frequency breakdown — `subword_token_frequency.csv` only has global counts, with no repository column. Each repository component is just the token's single global `normalized_frequency` scaled by that repository's fixed share of the dataset, so it mainly encodes repository *size*, not where the token is actually concentrated.

### `position_embedding.py` — Custom Approach 2
Tokenizes every `file_path` with the Lab-3 BPE tokenizer and, for each occurrence of a selected token, records its position normalized to `[0, 1]` (first token → 0, last → 1, a length-1 sequence → 0.5). These are aggregated per token into `mean_position`, `position_std`, `minimum_position`, `maximum_position`, and `occurrences` (saved to `position_features.csv`; a token never observed defaults to `mean=min=max=0.5`, `std=0`). Each of the four statistics is run through a standard sinusoidal positional encoding (the same `sin`/`cos` scheme used for Transformer positional encodings, dimension 128), and the four resulting vectors are combined as `mean + 0.20·std + 0.10·min + 0.10·max`, then L2-normalized.

### `word2vec_embedding.py`
Builds one training "sentence" per `file_path` (its BPE tokens, skipped if fewer than 2 tokens) and trains a `gensim` `Word2Vec` model: `vector_size=128`, `window=5`, `min_count=1`, `sg=1` (skip-gram), `epochs=30`, `workers=1`, `seed=42`. It then looks up each of the 20 selected tokens in the trained vocabulary — any token that never appeared in a 2+-token path is skipped with a warning, so `word2vec_embeddings.npy` / `word2vec_selected_tokens.csv` can have fewer than 20 rows. Saves the full model (`word2vec_model.model`, reloadable with `Word2Vec.load`), the selected-token embedding matrix, and the list of tokens actually found.

`workers=1` together with a fixed `seed` is what makes this reproducible — `gensim`'s multithreaded training can otherwise introduce run-to-run drift even with a seed set.

### `code2vec_embedding.py` — "Code2Vec-inspired", not canonical Code2Vec
For each occurrence of a selected token in a tokenized file path, it captures up to 2 neighboring tokens on each side (`CONTEXT_WINDOW = 2`) as that occurrence's structural context, and logs every occurrence to `code2vec_contexts.csv`. Each context is turned into a vector by averaging **deterministic, hash-derived vectors** for the center token and every left/right neighbor (`deterministic_vector()` seeds a NumPy RNG from the first 8 bytes of a SHA-256 hash of a role-tagged string like `"LEFT::py"`, draws a 128-dim standard normal vector, and L2-normalizes it — so the same string always maps to the same vector, with no training or lookup table involved). A token's final embedding is the mean of all of its occurrence-context vectors, L2-normalized (or a zero vector if the token has no contexts).

This reproduces Code2Vec's *shape* — token → structural contexts → context vectors → aggregated embedding — but **not its substance**: real Code2Vec learns path-context and attention weights from AST paths through a trained neural network; this implementation has no learned parameters at all, works on file-path segments (there being no AST or source code available), and aggregates with a plain mean instead of attention.

### `similarity_analysis.py`
`cosine_similarity(a, b)` = `dot(a, b) / (‖a‖·‖b‖)`, returning `0.0` if either vector is zero. `calculate_pairwise_similarity()` computes this for every unique pair among the given tokens, sorts descending, and writes `{approach}_similarity.csv` (all pairs) and `{approach}_top_pairs.csv` (top 5). This function is called once per approach, immediately after that approach's embeddings are generated — not once at the end for all four.

### `comparison.py`
Runs last. Reads whichever of the four `*_similarity.csv` files exist, and for each computes `highest_similarity`, `lowest_similarity`, and `average_similarity` over that approach's `similarity` column. Writes the four-row summary to `comparison.csv` and prints it as a table. An approach whose similarity file is missing or empty is skipped with a warning rather than failing the run.

### `main.py`
Pure orchestrator: load data → select 20 tokens → (generate embeddings → compute pairwise similarity) for each of the four approaches in turn → compare. Wraps everything in a `try/except` that prints the failing stage's exception type/message before re-raising, so a mid-pipeline failure is easy to locate.

## 5. Generated Output Files

All of the following land in `output/`, created automatically on first run:

| File | Contents |
|---|---|
| `selected_tokens.json` | The 20 manually chosen tokens, with rank and frequency |
| `frequency_embeddings.npy` | `(20, 2+R)` L2-normalized frequency-profile vectors (R = number of repositories) |
| `frequency_embedding_features.csv` | Human-readable version: token, frequency, normalized/log frequency, per-repository components |
| `frequency_embedding_similarity.csv` / `..._top_pairs.csv` | All pairwise cosine similarities / top 5, for the frequency approach |
| `position_features.csv` | Per-token mean/std/min/max position and occurrence count |
| `position_embeddings.npy` | `(20, 128)` sinusoidal-encoded position vectors |
| `position_embedding_similarity.csv` / `..._top_pairs.csv` | Same, for the position approach |
| `word2vec_model.model` | The full trained `gensim` Word2Vec model |
| `word2vec_embeddings.npy` | `(≤20, 128)` embeddings for the selected tokens found in the trained vocabulary |
| `word2vec_selected_tokens.csv` | Which selected tokens were actually found |
| `word2vec_similarity.csv` / `word2vec_top_pairs.csv` | Same, for Word2Vec |
| `code2vec_contexts.csv` | One row per occurrence of a selected token, with its left/right structural context |
| `code2vec_embeddings.npy` | `(20, 128)` Code2Vec-inspired embeddings |
| `code2vec_similarity.csv` / `code2vec_top_pairs.csv` | Same, for the Code2Vec-inspired approach |
| `comparison.csv` | One row per approach: highest / lowest / average pairwise similarity |

## 6. Points to Remember for Revision

- **Only `repository` and `file_path` are used** from `combined_dataset.csv`, even though the file also has `language`, `extension`, `loc`, `size_bytes`, etc. There is no source-code text anywhere in this lab's inputs.
- **Frequency embedding never tokenizes anything** — it only reads `subword_token_frequency.csv` (global counts) and `combined_dataset.csv`'s `repository` column. Position, Word2Vec, and Code2Vec-inspired all tokenize `file_path` with the Lab-3 BPE tokenizer.
- **Word2Vec is the only approach that actually learns anything** from data via gradient-based training; frequency, position, and Code2Vec-inspired are all deterministic/hand-engineered — the "Code2Vec-inspired" vectors come from hashing, not training.
- **Everything here is reproducible given the same 20 selected tokens.** Frequency/position are pure arithmetic; Code2Vec-inspired is hash-seeded (always the same output for the same string); Word2Vec is deterministic because `workers=1` and `seed=42` are both set. The only run-to-run variation you should see is `gensim`'s own bookkeeping (e.g. `total_train_time`, a wall-clock figure) saved inside `word2vec_model.model` — never the learned vectors themselves.
- **The 20 tokens are chosen manually, not randomly** — two runs only produce comparable output if the same 20 indices are entered both times.
- **Similarity is computed per approach, right after that approach's embeddings**, not all at once at the end — `comparison.py` is simply the last stage that reads back whatever `*_similarity.csv` files already exist.

## 7. Common Issues

- **`main.py` appears to hang** — it isn't; it's waiting on `input()` from `token_selection.py`. Either type 20 numbers interactively, or pipe them in (see §3).
- **`[ERROR] You must select exactly 20 tokens.` / duplicate / out-of-range errors** — the indices must be 20 distinct integers between 1 and 50 inclusive, referring to the printed top-50 list, not raw token frequencies.
- **A selected token is missing from `word2vec_selected_tokens.csv`** — it never occurred in any file path that tokenized to 2+ BPE tokens, so Word2Vec's corpus never saw it as part of a sentence.
- **`subword_vocabulary.json` looks unused** — it is. `data_loader.py` defines `load_subword_vocabulary()` for it, but `main.py` never calls that function; only `combined_dataset.csv`, `subword_token_frequency.csv`, and `bpe_tokenizer.json` actually feed the pipeline.
- **`src/requirements.txt` lists `scipy`**, but nothing under `src/` imports it — `numpy`, `pandas`, `gensim`, and `tokenizers` are the packages actually used.
- **`code2vec_embedding.py` defines a module-level `SEED = 42`** that is never referenced anywhere else in the file — `deterministic_vector()` derives its RNG seed purely from hashing the input string, so this constant has no effect on the output.
- **Missing input files** — `data_loader.py` raises `FileNotFoundError` immediately if any of the three files it loads isn't present under `data/input/`; this lab expects Lab 3's outputs to already be there.
- **Working directory doesn't matter** — every module resolves its paths from the script's own location (`BASE_DIR = Path(__file__).resolve().parent.parent`), not the current working directory, so `output/` always lands inside `lab4/` regardless of where you invoke `python3` from.
