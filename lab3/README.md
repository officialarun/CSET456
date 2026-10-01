# Lab 3 — Tokenization and Embedding Analysis

A revision guide to the Lab 3 implementation: what it does, how it works, and how to run it.

## 1. Objective

Take the dataset produced by Lab 2 (`combined_dataset.csv`) and explore how repository data can be represented numerically for machine learning:

1. Tokenize the same text three different ways — character-level, word-level, and subword/BPE — and compare vocabulary size vs. sequence length for each.
2. Build a small random embedding matrix for the most frequent subword tokens and measure how "close" tokens are to each other using cosine similarity.
3. Let a human pick token pairs they consider related, and compare that judgement against the random embedding's similarity score.
4. Apply a naive rule-based heuristic (shared characters → nudge embeddings together) and re-measure similarity to see the heuristic's effect.

This is explicitly an embedding-mechanics exercise, not a trained-model exercise: the embeddings are randomly initialized and only ever modified by the hand-written heuristic in step 4 — nothing here is learned from data via gradient descent.

## 2. Directory Structure

```
lab3/
├── data/
│   └── input/
│       └── combined_dataset.csv     # Lab 2 output — the input to this lab
├── src/
│   ├── character_tokenizer.py       # Task 1: character-level tokenizer
│   ├── word_tokenizer.py            # Task 1: word-level tokenizer
│   ├── subword_tokenizer.py         # Task 1: BPE subword tokenizer
│   ├── embedding_analysis.py        # Task 2: random embeddings + similarity
│   ├── token_similarity.py          # Tasks 3 & 4: human selection + pair similarity
│   ├── similarity_improvement.py    # Tasks 5 & 6: naive improvement + re-experiment
│   ├── main.py                      # orchestrator / entry point
│   └── requirements.txt
├── output/                          # generated artifacts (JSON/CSV/.npy) + submitted screenshots
└── README.md
```

`data/` and `output/` already contain the dataset and the artifacts from the submitted run — re-running the pipeline overwrites the files in `output/` by name.

## 3. How to Run

From the `lab3` directory:

```bash
pip install -r src/requirements.txt
python src/main.py
```

Every module resolves its paths from `Path(__file__).resolve().parent.parent`, so it doesn't matter which directory you invoke it from — it always reads `lab3/data/input/combined_dataset.csv` and writes to `lab3/output/`. Each stage script can also be run standalone (e.g. `python src/character_tokenizer.py`) as long as an earlier stage has already produced whatever file it depends on.

**Tasks 3 & 4 are interactive.** `token_similarity.py` prints the top-50 token table and uses `input()` to ask you to pick 5 token numbers, then one "related" token number for each — there's no way to run the full pipeline unattended unless you pipe answers into stdin.

Execution order (each stage depends on files written by the previous one):

```
Task 1 (tokenizers) → Task 2 (embeddings) → Tasks 3 & 4 (human selection) → Tasks 5 & 6 (improvement)
```

## 4. What Each Module Does

### `character_tokenizer.py` — Task 1a

Splits each text record into individual characters. Builds the vocabulary with `collections.Counter`, then ranks characters by frequency (`Counter.most_common()`) and assigns IDs starting at 1 — ID `0` is reserved as the "unknown" fallback used when converting text to ID sequences. Reports vocabulary size, total/average/min/max sequence length, and a *hypothetical* embedding matrix size (`vocabulary_size × 128`) — no actual embedding array is created here, only the would-be shape and parameter count.

### `word_tokenizer.py` — Task 1b

Splits text using a single `re.findall` regex (`re.VERBOSE`) with four alternatives tried in order: identifiers (`[A-Za-z_][A-Za-z0-9_]*`), numbers (optionally decimal), a fixed set of two-character operators (`==`, `!=`, `<=`, `>=`, `->`, `=>`, `::`, `&&`, `||`), and finally any single non-whitespace character as a catch-all. The multi-character operators must appear before the catch-all branch or they'd never match as a unit. Vocabulary building mirrors the character tokenizer (frequency-ranked, 1-indexed), but this module never converts tokens to ID sequences — only token *lists* are kept, used solely to compute sequence-length statistics.

### `subword_tokenizer.py` — Task 1c

Uses Hugging Face's `tokenizers` library to train a real BPE model from scratch: `Tokenizer(BPE(unk_token="[UNK]"))` with a `ByteLevel` pre-tokenizer (`add_prefix_space=False`) and a `BpeTrainer(vocab_size=10000, special_tokens=[...], min_frequency=2)`. BPE starts from individual bytes and greedily merges whichever adjacent pair is most frequent in the corpus, repeating until either the vocabulary cap is hit or no remaining pair occurs at least `min_frequency` times — on this dataset the corpus is small enough that training stops well short of the 10,000 cap (around 1,750 learned tokens including the 5 special tokens). The trained tokenizer is saved as `bpe_tokenizer.json` (full merge rules + vocab, reusable later), alongside a token-frequency CSV used by Task 2.

### `embedding_analysis.py` — Task 2

Reads `subword_token_frequency.csv`, takes the top 50 most frequent BPE tokens, and creates a **random** `50 × 128` embedding matrix: `np.random.seed(42)` then `np.random.normal(loc=0.0, scale=0.1, ...)`. Pairwise cosine similarity is computed for all tokens at once via matrix ops (normalize every row, then `normalized @ normalized.T`), producing the top 10 most- and least-similar pairs out of all `50×49/2 = 1225` unique pairs. Because the vectors are random, these similarity scores don't carry any semantic meaning by themselves — they're a baseline for Tasks 5/6 to improve on.

### `token_similarity.py` — Tasks 3 & 4

Loads the Task 2 outputs (`top_50_tokens.csv`, `random_embedding_matrix.npy`) and interactively asks the user to pick exactly 5 distinct tokens by number, then for each one, pick a second token they personally judge as "related" (explicitly *not* using the embedding similarity to decide — that's the point of the exercise). The only validation on the related-token choice is that its *text* differs from the token it's paired with; nothing stops reusing the same related-token number across different pairs. Task 4 then computes cosine similarity for those 5 human-chosen pairs using the *same random embeddings* from Task 2 — so it directly shows how (un)correlated human intuition is with an untrained embedding space.

### `similarity_improvement.py` — Tasks 5 & 6

A naive, hand-written heuristic — not a trained model. For every pair among the same top-50 tokens (independent of whatever pairs the user picked in Task 3):

1. Compute **character overlap** = `|set(chars(token_a)) ∩ set(chars(token_b))| / min(|chars(token_a)|, |chars(token_b)|)`.
2. If overlap ≥ `0.40`, treat the pair as "related" and nudge both embeddings toward each other:
   `new_a = (1 - 0.05) * a + 0.05 * b`, `new_b = (1 - 0.05) * b + 0.05 * a`.

Each related pair is processed once in index order; a token can be nudged multiple times if it overlaps with several others. Task 6 then recomputes cosine similarity for all 1225 pairs using both the original and the improved matrix and records the `change` per pair. The average similarity rising after the update is an expected, mechanical consequence of the nudging rule (related pairs are pulled together by construction) — it's not evidence the embedding space has become semantically "better".

### `main.py` — orchestrator

Runs the six stages in order through a shared `run_task()` wrapper that times each stage and catches `FileNotFoundError` / `ValueError` / any other `Exception` separately, printing a stage-specific error and **stopping the pipeline** on the first failure (later stages depend on earlier stages' output files). Prints a banner, a progress header per stage, and a final listing of every file written to `output/`.

## 5. Key Concepts as Implemented Here

**Byte Pair Encoding (BPE).** Starts from raw bytes/characters; at each step, finds the most frequent adjacent symbol pair in the (pre-tokenized) corpus and merges it into a single new symbol, repeating until the vocabulary budget is used up. Frequent substrings (e.g. common words, prefixes) end up as single tokens while rare text still decomposes into smaller pieces. This lab uses the `tokenizers` library's trainer directly rather than implementing merges by hand.

**Cosine similarity.** `cosine(A, B) = (A · B) / (‖A‖ ‖B‖)`, computed in two places: as a full matrix in `embedding_analysis.py` (vectorized over all 50 tokens at once) and per-pair in `token_similarity.py` / `similarity_improvement.py`. Range is -1 (opposite) to +1 (same direction); a zero vector is special-cased to return `0.0` rather than divide by zero.

**Embedding matrix size.** For a vocabulary of size `V` and embedding dimension `D = 128`, the (hypothetical, for Tasks 1a/1b, or actual, for Task 2) matrix has `V × D` parameters. All three tokenizer stats files report this even though only the subword tokenizer's top 50 tokens get a real matrix.

**Fixed random seed.** `np.random.seed(42)` before generating the embedding matrix makes Task 2 (and everything downstream that reads `random_embedding_matrix.npy`) reproducible across runs.

**Character-overlap heuristic.** A deliberately simple, non-learned similarity proxy based on shared *unique* characters between two token strings, used only to decide which pairs Task 5 nudges together.

## 6. Generated Output Files

| File | Produced by | Contents |
|---|---|---|
| `character_tokenizer_statistics.json` | Task 1a | vocab size, sequence-length stats, hypothetical embedding size |
| `character_vocabulary.json` | Task 1a | `token_to_id` map + raw frequency counts |
| `word_tokenizer_statistics.json` | Task 1b | same shape as above, for word tokens |
| `word_vocabulary.json` | Task 1b | `token_to_id` map + raw frequency counts |
| `subword_tokenizer_statistics.json` | Task 1c | BPE stats, including target vs. actual vocab size |
| `subword_vocabulary.json` | Task 1c | BPE `token → id` map (see note below on ordering) |
| `subword_token_frequency.csv` | Task 1c | every observed BPE token with its frequency, sorted descending |
| `bpe_tokenizer.json` | Task 1c | the full trained tokenizer (vocab + merge rules), reloadable via `Tokenizer.from_file` |
| `top_50_tokens.csv` | Task 2 | the 50 most frequent BPE tokens used for all later stages |
| `random_embedding_matrix.npy` | Task 2 | the `50 × 128` random embedding matrix (seed 42) |
| `embedding_statistics.json` | Task 2 | shape, seed, and the top similar/dissimilar pairs |
| `top_10_most_similar.csv` / `top_10_least_similar.csv` | Task 2 | extremes of the 1225 pairwise cosine similarities |
| `task3_selected_tokens.csv` | Task 3 | the 5 human-chosen `token_a, token_b` pairs |
| `task4_pair_similarity.csv` | Task 4 | cosine similarity of those 5 pairs under the random embeddings |
| `improved_embedding_matrix.npy` | Task 5 | embeddings after the character-overlap nudging |
| `improved_similarity_results.csv` | Task 6 | original vs. improved similarity + `change`, for all 1225 pairs |
| `similarity_improvement_statistics.json` | Task 6 | related-pair count and average similarity before/after/change |

## 7. Points to Remember for Revision

- **Three tokenizers, one trade-off:** character tokenization gives the smallest vocabulary but the longest sequences; word tokenization gives a much larger vocabulary but short sequences; BPE sits in between and additionally handles unseen/rare terms by falling back to smaller subword pieces instead of a single `[UNK]`.
- **Only the subword tokenizer gets a real embedding matrix.** The character and word tokenizer stats include an embedding-size *calculation* (`V × 128`), but no `.npy` matrix is ever created for them — Task 2 only materializes embeddings for the top-50 BPE tokens.
- **The human relationship (Task 3) and the heuristic relationship (Task 5) are independent.** Task 5's character-overlap rule runs over all 1225 pairs of the top-50 tokens regardless of which 5 pairs the user picked in Task 3 — nothing from Task 3/4 feeds into Task 5.
- **Task 5 is a hand-written rule, not a trained embedding.** It demonstrates that *a* rule can reshape an embedding space, not that the result is semantically meaningful — a related pair's similarity rising is guaranteed by the update formula itself.
- **`vocab_size=10000` is a ceiling, not a target.** BPE training stops early once no mergeable pair meets `min_frequency=2`, which is why the actual learned vocabulary (~1,750 on this dataset) is far below the configured cap.
- **Cosine similarity of a zero vector is special-cased to `0.0`** in every module that computes it, avoiding a divide-by-zero rather than raising.

## 8. Common Issues / Things to Watch For

- **`source_code` column missing.** All three tokenizers first look for a `source_code` column, then `code`, and otherwise fall back to `file_path` with a printed warning. The current `combined_dataset.csv` only has `repository, file_path, language, extension, loc, size_bytes, change_count, history_present` — so every run tokenizes **file paths**, not actual source code. This matters when interpreting the vocabulary/sequence-length numbers.
- **`subword_vocabulary.json` key order is not reproducible between runs**, even with identical input and no code changes — `tokenizer.get_vocab()` returns a dict whose iteration order depends on the underlying Rust hash map, which is not seeded for determinism. The *contents* (every token → id mapping) are identical every run; only the on-disk key order can differ. Don't treat a reordered-but-same-content diff on this one file as a real regression.
- **Tasks 3 & 4 need a live terminal (or piped stdin).** `get_integer_input()` loops on `input()` with no timeout — automating a full `main.py` run requires feeding it 10 numeric answers (5 token picks + 5 related-token picks) via stdin in the exact order the prompts expect.
- **Re-running overwrites by filename, not by directory.** `output/` is never cleared before a run — each stage just overwrites the specific files it owns, so stale files from an unrelated earlier experiment (if any) could linger alongside the new ones.
- **`requirements.txt` is accurate.** Unlike some of the other labs in this project, every package listed (`pandas`, `numpy`, `tokenizers`) is actually imported and used somewhere in `src/` — there's no unused dependency to prune here.
