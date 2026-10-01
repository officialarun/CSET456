from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def calculate_pairwise_similarity(tokens, embeddings, approach_name):

    print("\n")
    print("=" * 75)
    print(f"PAIRWISE COSINE SIMILARITY — {approach_name}")
    print("=" * 75)

    # Every unique token pair (i, j) with i < j — one triangle of the
    # similarity matrix, since cosine similarity is symmetric.
    rows = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):

            similarity = cosine_similarity(embeddings[i], embeddings[j])

            rows.append(
                {
                    "token_1": tokens[i],
                    "token_2": tokens[j],
                    "similarity": similarity
                }
            )

    similarity_df = pd.DataFrame(rows)

    similarity_df = similarity_df.sort_values(
        "similarity", ascending=False
    ).reset_index(drop=True)

    safe_name = (
        approach_name.lower().replace(" ", "_").replace("/", "_")
    )

    output_path = OUTPUT_DIR / f"{safe_name}_similarity.csv"
    similarity_df.to_csv(output_path, index=False)

    top_5 = similarity_df.head(5)
    top_path = OUTPUT_DIR / f"{safe_name}_top_pairs.csv"
    top_5.to_csv(top_path, index=False)

    print("\nTop 5 most similar pairs:")

    for _, row in top_5.iterrows():
        print(
            f"    {row['token_1']!r} <-> {row['token_2']!r} "
            f"= {row['similarity']:.4f}"
        )

    print(f"\n[✓] Similarity matrix saved:")
    print(f"    {output_path}")
    print(f"[✓] Top pairs saved:")
    print(f"    {top_path}")

    return similarity_df
