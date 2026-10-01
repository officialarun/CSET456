from pathlib import Path

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

POSITION_FEATURES_FILE = OUTPUT_DIR / "position_features.csv"
POSITION_EMBEDDINGS_FILE = OUTPUT_DIR / "position_embeddings.npy"


# ============================================================
# TOKENIZER HELPER
# ============================================================

def tokenize_file_path(tokenizer, file_path):
    """
    Tokenize one repository-relative file path
    using the Lab-3 BPE tokenizer.
    """

    normalized_path = str(file_path).replace("\\", "/")
    encoded = tokenizer.encode(normalized_path)

    return [str(token) for token in encoded.tokens]


# ============================================================
# POSITION FEATURE EXTRACTION
# ============================================================

def build_position_features(selected_tokens, dataset, tokenizer):
    """
    Build positional features for selected tokens.

    The position is calculated within each tokenized file path.

    The resulting features describe where a token tends to occur
    in the available repository path sequences.
    """

    print("\n")
    print("=" * 75)
    print("BUILDING TOKEN POSITION / WORD-FLOW FEATURES")
    print("=" * 75)

    token_set = set(str(token) for token in selected_tokens)
    positions = {token: [] for token in selected_tokens}

    print(f"\nSelected tokens : {len(selected_tokens)}")
    print(f"Dataset records : {len(dataset)}")
    print("\nScanning tokenized file-path sequences...")

    total_rows = len(dataset)

    for row_number, (_, row) in enumerate(dataset.iterrows(), start=1):

        file_path = str(row["file_path"])
        tokens = tokenize_file_path(tokenizer, file_path)
        sequence_length = len(tokens)

        if sequence_length == 0:
            continue

        for index, token in enumerate(tokens):

            if token not in token_set:
                continue

            # Normalize position to [0, 1].
            #
            # First token  -> 0
            # Last token   -> 1
            #
            # For a one-token sequence we use 0.5.

            if sequence_length == 1:
                normalized_position = 0.5
            else:
                normalized_position = index / (sequence_length - 1)

            positions[token].append(normalized_position)

        if row_number % 500 == 0 or row_number == total_rows:
            print(f"    Processed {row_number}/{total_rows} paths")

    # ========================================================
    # AGGREGATE FEATURES
    # ========================================================

    feature_rows = []

    for token in selected_tokens:

        values = np.array(positions[token], dtype=float)

        if len(values) == 0:

            mean_position = 0.5
            position_std = 0.0
            minimum_position = 0.5
            maximum_position = 0.5

        else:

            mean_position = float(np.mean(values))
            position_std = float(np.std(values))
            minimum_position = float(np.min(values))
            maximum_position = float(np.max(values))

        feature_rows.append(
            {
                "token": str(token),
                "mean_position": mean_position,
                "position_std": position_std,
                "minimum_position": minimum_position,
                "maximum_position": maximum_position,
                "occurrences": int(len(values))
            }
        )

    features = pd.DataFrame(feature_rows)

    print("\n[✓] Position features calculated.")
    print(f"    Feature rows : {len(features)}")
    print(f"    Feature columns : {len(features.columns)}")

    features.to_csv(POSITION_FEATURES_FILE, index=False)

    print(f"\n[✓] Position features saved:")
    print(f"    {POSITION_FEATURES_FILE}")

    return features


# ============================================================
# SINUSOIDAL POSITION ENCODING
# ============================================================

def sinusoidal_encoding(position, dimension):
    """
    Generate a standard sinusoidal positional vector.

    This converts a scalar normalized position into a
    fixed-dimensional representation.
    """

    vector = np.zeros(dimension, dtype=float)

    for i in range(dimension // 2):

        denominator = 10000 ** ((2 * i) / dimension)
        angle = position / denominator

        vector[2 * i] = np.sin(angle)

        if 2 * i + 1 < dimension:
            vector[2 * i + 1] = np.cos(angle)

    return vector


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def create_position_embeddings(selected_tokens, dataset, tokenizer):
    """
    Create embeddings from token-position statistics.
    """

    print("\n")
    print("=" * 75)
    print("CUSTOM APPROACH 2 — POSITION / WORD-FLOW EMBEDDING")
    print("=" * 75)

    features = build_position_features(selected_tokens, dataset, tokenizer)

    embeddings = []

    print("\nConverting positional features into embedding vectors...")

    for _, row in features.iterrows():

        # Main positional representation.
        mean_vector = sinusoidal_encoding(
            float(row["mean_position"]), VECTOR_SIZE
        )

        # Position variability representation.
        std_vector = sinusoidal_encoding(
            min(float(row["position_std"]), 1.0), VECTOR_SIZE
        )

        # Minimum position.
        min_vector = sinusoidal_encoding(
            float(row["minimum_position"]), VECTOR_SIZE
        )

        # Maximum position.
        max_vector = sinusoidal_encoding(
            float(row["maximum_position"]), VECTOR_SIZE
        )

        # Combine the positional components. Mean position dominates;
        # spread/min/max are added as smaller corrections so two tokens
        # with the same average position but different spread still end
        # up with distinguishable vectors.
        vector = (
            mean_vector
            + 0.20 * std_vector
            + 0.10 * min_vector
            + 0.10 * max_vector
        )

        # L2 normalization.
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        embeddings.append(vector)

    embeddings = np.array(embeddings, dtype=float)

    np.save(POSITION_EMBEDDINGS_FILE, embeddings)

    print("\n[✓] Position/order embeddings generated.")
    print(f"    Shape : {embeddings.shape}")
    print(f"    Dimension : {VECTOR_SIZE}")
    print(f"    Saved : {POSITION_EMBEDDINGS_FILE}")

    return embeddings


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_position_embeddings(selected_tokens, dataset, tokenizer):
    """
    Public function used by main.py.
    """

    return create_position_embeddings(selected_tokens, dataset, tokenizer)
