from pathlib import Path
import math

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_columns(df):

    token_column = None
    frequency_column = None

    for name in ["token", "tokens", "subword", "word"]:
        if name in df.columns:
            token_column = name
            break

    for name in ["frequency", "count", "token_frequency"]:
        if name in df.columns:
            frequency_column = name
            break

    if token_column is None:
        raise ValueError("Token column not found.")

    if frequency_column is None:
        raise ValueError("Frequency column not found.")

    return token_column, frequency_column


def generate_frequency_embeddings(
    selected_tokens,
    token_frequency,
    combined_dataset
):

    print("\n")
    print("=" * 75)
    print("CUSTOM APPROACH 1")
    print("FREQUENCY / OCCURRENCE EMBEDDING")
    print("=" * 75)

    token_column, frequency_column = _find_columns(token_frequency)

    frequency_lookup = {}

    for _, row in token_frequency.iterrows():

        token = str(row[token_column])

        try:
            frequency = float(row[frequency_column])
        except (ValueError, TypeError):
            frequency = 0.0

        frequency_lookup[token] = frequency

    # Repository information
    if "repository" in combined_dataset.columns:

        repository_counts = combined_dataset.groupby("repository").size()
        repositories = repository_counts.index.tolist()

        total = repository_counts.sum()
        repository_weights = (repository_counts / total).to_dict()

    else:

        repositories = []
        repository_weights = {}

    # Used to normalize raw frequencies into [0, 1]; the trailing "+ [1]"
    # guards against a zero/empty max when no selected token is found.
    max_frequency = max(
        [frequency_lookup.get(token, 0) for token in selected_tokens] + [1]
    )

    embeddings = []
    rows = []

    print("\nGenerating frequency profiles...")

    for token in selected_tokens:

        frequency = frequency_lookup.get(token, 0)
        normalized_frequency = frequency / max_frequency
        log_frequency = math.log1p(frequency)

        # Per-repository component: how much of this token's normalized
        # frequency is "explained" by each repository's share of the
        # dataset. This is what lets the embedding distinguish a token
        # that is frequent everywhere from one concentrated in one repo.
        repository_profile = []

        for repository in repositories:
            weight = repository_weights[repository]
            repository_profile.append(normalized_frequency * weight)

        vector = np.array(
            [normalized_frequency, log_frequency, *repository_profile],
            dtype=float
        )

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        embeddings.append(vector)

        row = {
            "token": token,
            "frequency": frequency,
            "normalized_frequency": normalized_frequency,
            "log_frequency": log_frequency
        }

        for index, repository in enumerate(repositories):
            row[f"repository_{index + 1}"] = repository_profile[index]

        rows.append(row)

    embeddings = np.array(embeddings, dtype=float)
    feature_df = pd.DataFrame(rows)

    embedding_path = OUTPUT_DIR / "frequency_embeddings.npy"
    feature_path = OUTPUT_DIR / "frequency_embedding_features.csv"

    np.save(embedding_path, embeddings)
    feature_df.to_csv(feature_path, index=False)

    print("\n[✓] Frequency embeddings generated.")
    print(f"    Shape: {embeddings.shape}")
    print(f"    Saved: {embedding_path}")
    print(f"    Features: {feature_path}")

    return embeddings
