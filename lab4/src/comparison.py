from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def compare_similarity_results():

    print("\n")
    print("=" * 75)
    print("EMBEDDING APPROACH COMPARISON")
    print("=" * 75)

    files = {
        "Frequency": "frequency_embedding_similarity.csv",
        "Position": "position_embedding_similarity.csv",
        "Word2Vec": "word2vec_similarity.csv",
        "Code2Vec": "code2vec_similarity.csv"
    }

    rows = []

    for approach, filename in files.items():

        path = OUTPUT_DIR / filename

        if not path.exists():
            print(f"[WARNING] Missing: {filename}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        # Summarize each approach's pairwise-similarity distribution down
        # to three numbers so the four approaches can be compared side
        # by side in a single table.
        rows.append(
            {
                "approach": approach,
                "highest_similarity": df["similarity"].max(),
                "lowest_similarity": df["similarity"].min(),
                "average_similarity": df["similarity"].mean()
            }
        )

    if not rows:
        print("[WARNING] No similarity files found.")
        return None

    comparison = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / "comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nComparison:")
    print(comparison.to_string(index=False))

    print(f"\n[✓] Comparison saved:")
    print(f"    {output_path}")

    return comparison
