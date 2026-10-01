from pathlib import Path

import numpy as np
import pandas as pd

from gensim.models import Word2Vec


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_SIZE = 128
WINDOW_SIZE = 5
MIN_COUNT = 1
EPOCHS = 30
SEED = 42


# ============================================================
# TOKENIZER
# ============================================================

def tokenize_file_path(tokenizer, file_path):
    """
    Tokenize one file path using the Lab-3 BPE tokenizer.
    """

    normalized_path = str(file_path).replace("\\", "/")
    encoded = tokenizer.encode(normalized_path)

    return [str(token) for token in encoded.tokens]


# ============================================================
# CORPUS CREATION
# ============================================================

def build_word2vec_corpus(dataset, tokenizer):
    """
    Build multiple token sequences.

    Each file path becomes one sentence.
    """

    print("\n")
    print("=" * 75)
    print("BUILDING WORD2VEC TOKEN SEQUENCES")
    print("=" * 75)

    sentences = []
    total_rows = len(dataset)

    for row_number, (_, row) in enumerate(dataset.iterrows(), start=1):

        file_path = str(row["file_path"])
        tokens = tokenize_file_path(tokenizer, file_path)

        # Word2Vec needs at least two tokens to form a context window,
        # so single-token paths can't contribute a training sentence.
        if len(tokens) >= 2:
            sentences.append(tokens)

        if row_number % 500 == 0 or row_number == total_rows:
            print(f"    Processed {row_number}/{total_rows} paths")

    if not sentences:
        raise ValueError(
            "No usable token sequences were created for Word2Vec."
        )

    print("\n[✓] Word2Vec corpus created.")
    print(f"    Number of sequences : {len(sentences)}")
    print(f"    Example sequence:")
    print(f"    {sentences[0]}")

    return sentences


# ============================================================
# TRAIN WORD2VEC
# ============================================================

def train_word2vec(
    selected_tokens,
    dataset,
    tokenizer,
    vector_size=VECTOR_SIZE
):
    """
    Train Word2Vec and generate embeddings for the manually
    selected 20 tokens.
    """

    print("\n")
    print("=" * 75)
    print("WORD2VEC EMBEDDING")
    print("=" * 75)

    # --------------------------------------------------------
    # Build corpus
    # --------------------------------------------------------

    sentences = build_word2vec_corpus(dataset, tokenizer)

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTraining Word2Vec model...")
    print(f"    Vector size : {vector_size}")
    print(f"    Window      : {WINDOW_SIZE}")
    print(f"    Epochs      : {EPOCHS}")
    print(f"    Min count   : {MIN_COUNT}")

    # workers=1 + a fixed seed keeps training deterministic (gensim's
    # multithreaded training can otherwise introduce run-to-run drift).
    # sg=1 selects skip-gram over CBOW.
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=1,
        sg=1,
        epochs=EPOCHS,
        seed=SEED
    )

    print("\n[✓] Word2Vec model trained.")
    print(f"    Learned vocabulary : {len(model.wv)} tokens")

    # --------------------------------------------------------
    # Selected-token embeddings
    # --------------------------------------------------------

    embeddings = []
    valid_tokens = []

    print("\nGenerating embeddings for selected tokens...")

    for token in selected_tokens:

        token = str(token)

        if token in model.wv:

            vector = model.wv[token]

            embeddings.append(vector)
            valid_tokens.append(token)

            print(f"    [✓] {token!r}")

        else:

            print(
                f"    [WARNING] {token!r} not present in "
                f"Word2Vec vocabulary"
            )

    if not embeddings:
        raise ValueError(
            "None of the selected tokens were found in the "
            "Word2Vec vocabulary."
        )

    embeddings = np.array(embeddings, dtype=float)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_path = OUTPUT_DIR / "word2vec_model.model"
    embedding_path = OUTPUT_DIR / "word2vec_embeddings.npy"
    token_path = OUTPUT_DIR / "word2vec_selected_tokens.csv"

    model.save(str(model_path))
    np.save(embedding_path, embeddings)

    pd.DataFrame({"token": valid_tokens}).to_csv(token_path, index=False)

    print("\n[✓] Word2Vec outputs saved.")
    print(f"    Model      : {model_path}")
    print(f"    Embeddings : {embedding_path}")
    print(f"    Tokens     : {token_path}")
    print(f"    Matrix     : {embeddings.shape}")

    return (valid_tokens, embeddings, model)
