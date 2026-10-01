from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def identify_columns(df):

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


def select_tokens(frequency_df, number_of_tokens=20):

    token_column, frequency_column = identify_columns(frequency_df)

    df = frequency_df.copy()
    df[frequency_column] = df[frequency_column].astype(float)

    df = df.sort_values(
        frequency_column, ascending=False
    ).reset_index(drop=True)

    # Candidates are capped at the 50 most frequent tokens so the manual
    # selection below only ever has to validate against that range.
    top_50 = df.head(50)

    print("\n")
    print("=" * 75)
    print("TOP 50 MOST FREQUENT TOKENS")
    print("=" * 75)

    for index, row in top_50.iterrows():

        token = str(row[token_column])
        frequency = row[frequency_column]

        print(f"{index + 1:2d}. {token!r:<25} frequency = {frequency}")

    print("\n" + "-" * 75)
    print(f"Select {number_of_tokens} tokens manually.")
    print("Enter their numbers separated by spaces.")
    print("Example:")
    print("2 4 5 8 11 14 17 21 25 28 31 34 36 39 41 43 45 47 49 50")

    # Loop until the user provides exactly `number_of_tokens` distinct,
    # in-range indices — this is manual selection, not a random sample.
    while True:

        user_input = input("\nEnter 20 token numbers: ").strip()

        try:
            indices = [int(x) for x in user_input.split()]
        except ValueError:
            print("[ERROR] Please enter numbers only.")
            continue

        if len(indices) != number_of_tokens:
            print(f"[ERROR] You must select exactly {number_of_tokens} tokens.")
            continue

        if len(set(indices)) != number_of_tokens:
            print("[ERROR] Duplicate token numbers are not allowed.")
            continue

        if any(index < 1 or index > 50 for index in indices):
            print("[ERROR] Token numbers must be between 1 and 50.")
            continue

        break

    selected = []

    print("\nSelected tokens:")

    for index in indices:

        row = top_50.iloc[index - 1]
        token = str(row[token_column])
        frequency = float(row[frequency_column])

        selected.append(
            {
                "rank": index,
                "token": token,
                "frequency": frequency
            }
        )

        print(f"  {index:2d}. {token!r} (frequency={frequency})")

    output_path = OUTPUT_DIR / "selected_tokens.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(selected, file, indent=4)

    print(f"\n[✓] Selected tokens saved:")
    print(f"    {output_path}")

    return [item["token"] for item in selected]
