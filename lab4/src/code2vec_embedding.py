from pathlib import Path

import hashlib

import numpy as np
import pandas as pd


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
CONTEXT_WINDOW = 2
SEED = 42

EMBEDDING_FILE = OUTPUT_DIR / "code2vec_embeddings.npy"
CONTEXT_FILE = OUTPUT_DIR / "code2vec_contexts.csv"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize_file_path(tokenizer, file_path):
    """
    Tokenize a repository file path using the Lab-3 BPE tokenizer.
    """

    normalized_path = str(file_path).replace("\\", "/")
    encoded = tokenizer.encode(normalized_path)

    return [str(token) for token in encoded.tokens]


# ============================================================
# DETERMINISTIC TOKEN VECTOR
# ============================================================

def deterministic_vector(value, dimension=VECTOR_SIZE):
    """
    Generate a deterministic vector for a structural token/path.

    The vector is generated from a hash rather than from a global
    random matrix, making the experiment reproducible.

    This is used as a fixed representation of structural context.
    """

    # Hashing the string into a per-value RNG seed means the same
    # left/right/center token always maps to the same vector, without
    # needing to store or look up an embedding table.
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    seed_bytes = digest[:8]
    seed = int.from_bytes(seed_bytes, byteorder="little")

    rng = np.random.default_rng(seed)
    vector = rng.normal(0.0, 1.0, dimension)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


# ============================================================
# STRUCTURAL CONTEXT CREATION
# ============================================================

def build_structural_contexts(selected_tokens, dataset, tokenizer):
    """
    Create structural contexts for selected tokens.

    For each occurrence of a selected token in a tokenized file path,
    nearby tokens are used to construct local path contexts.

    Example:

        ["src", "/", "utils", "/", "parser", ".py"]

    For "parser" with window=2:

        / -> parser -> .py

    A context identifier is created from the neighboring structure.
    """

    selected_set = set(str(token) for token in selected_tokens)
    contexts = []

    print("\n")
    print("=" * 75)
    print("BUILDING CODE2VEC-STYLE STRUCTURAL CONTEXTS")
    print("=" * 75)

    print(
        "\nUsing file-path structure because "
        "source-code/AST data is not present."
    )
    print(f"    Context window : {CONTEXT_WINDOW}")

    total_rows = len(dataset)

    for row_number, (_, row) in enumerate(dataset.iterrows(), start=1):

        file_path = str(row["file_path"])
        tokens = tokenize_file_path(tokenizer, file_path)

        for index, token in enumerate(tokens):

            if token not in selected_set:
                continue

            # Clip the context window to the sequence bounds so tokens
            # near the start/end of a path still get a (shorter) context.
            start = max(0, index - CONTEXT_WINDOW)
            end = min(len(tokens), index + CONTEXT_WINDOW + 1)

            left_context = tokens[start:index]
            right_context = tokens[index + 1:end]

            context = (tuple(left_context), token, tuple(right_context))

            contexts.append(
                {
                    "token": token,
                    "file_path": file_path,
                    "left_context": " ".join(left_context),
                    "right_context": " ".join(right_context),
                    "context": str(context)
                }
            )

        if row_number % 500 == 0 or row_number == total_rows:
            print(f"    Processed {row_number}/{total_rows} paths")

    contexts_df = pd.DataFrame(contexts)

    if contexts_df.empty:
        raise ValueError(
            "No structural contexts were created for the selected tokens."
        )

    contexts_df.to_csv(CONTEXT_FILE, index=False)

    print("\n[✓] Structural contexts created.")
    print(f"    Contexts : {len(contexts_df)}")
    print(f"    Saved    : {CONTEXT_FILE}")

    return contexts_df


# ============================================================
# CONTEXT VECTOR
# ============================================================

def context_vector(row):
    """
    Convert a structural context into a vector.

    The vector is composed of:

        left context vectors
        center-token vector
        right context vectors

    The components are averaged.
    """

    parts = []

    center = str(row["token"])
    parts.append(deterministic_vector(f"CENTER::{center}"))

    left = str(row["left_context"]).strip()
    right = str(row["right_context"]).strip()

    # Prefixing with "LEFT::"/"RIGHT::"/"CENTER::" keeps a token's
    # vector different depending on the role it plays in the context,
    # so the same subword seen on the left vs. the right isn't identical.
    if left:
        for token in left.split():
            parts.append(deterministic_vector(f"LEFT::{token}"))

    if right:
        for token in right.split():
            parts.append(deterministic_vector(f"RIGHT::{token}"))

    vector = np.mean(parts, axis=0)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


# ============================================================
# TOKEN EMBEDDINGS
# ============================================================

def generate_code2vec_embeddings(
    selected_tokens,
    dataset,
    tokenizer,
    vector_size=VECTOR_SIZE
):
    """
    Generate structural Code2Vec-style embeddings for the selected
    tokens.
    """

    print("\n")
    print("=" * 75)
    print("CODE2VEC-STYLE STRUCTURAL EMBEDDING")
    print("=" * 75)

    contexts_df = build_structural_contexts(selected_tokens, dataset, tokenizer)

    embeddings = []

    print("\nAggregating structural contexts into token embeddings...")

    for token in selected_tokens:

        token_contexts = contexts_df[contexts_df["token"] == str(token)]

        if token_contexts.empty:

            print(f"    [WARNING] No contexts for {token!r}")

            # Zero vector keeps token alignment.
            vector = np.zeros(vector_size, dtype=float)

        else:

            # Code2Vec's core idea, simplified here: a token's embedding
            # is the aggregate (mean) of all of its structural-context
            # vectors across every occurrence in the dataset.
            context_vectors = []

            for _, row in token_contexts.iterrows():
                context_vectors.append(context_vector(row))

            vector = np.mean(context_vectors, axis=0)

            norm = np.linalg.norm(vector)

            if norm > 0:
                vector = vector / norm

        embeddings.append(vector)

        print(f"    [✓] {token!r} ← {len(token_contexts)} contexts")

    embeddings = np.array(embeddings, dtype=float)

    np.save(EMBEDDING_FILE, embeddings)

    print("\n[✓] Code2Vec-style embeddings generated.")
    print(f"    Matrix shape : {embeddings.shape}")
    print(f"    Dimension    : {vector_size}")
    print(f"    Saved        : {EMBEDDING_FILE}")

    return embeddings
