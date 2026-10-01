from pathlib import Path
import json

import pandas as pd

from tokenizers import Tokenizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"


# ============================================================
# COMBINED DATASET
# ============================================================

def load_combined_dataset():

    path = INPUT_DIR / "combined_dataset.csv"

    print("\n[DATA] Loading combined dataset...")
    print(f"       {path}")

    if not path.exists():
        raise FileNotFoundError(f"combined_dataset.csv not found:\n{path}")

    df = pd.read_csv(path)

    required_columns = {"repository", "file_path"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "combined_dataset.csv is missing "
            f"required columns: {sorted(missing)}"
        )

    print("[✓] Combined dataset loaded")
    print(f"    Rows    : {len(df)}")
    print(f"    Columns : {len(df.columns)}")

    return df


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def load_token_frequency():

    path = INPUT_DIR / "subword_token_frequency.csv"

    print("\n[DATA] Loading subword token frequencies...")
    print(f"       {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"subword_token_frequency.csv not found:\n{path}"
        )

    df = pd.read_csv(path)

    print("[✓] Token frequency dataset loaded")
    print(f"    Tokens : {len(df)}")

    return df


# ============================================================
# SUBWORD VOCABULARY
# ============================================================

def load_subword_vocabulary():

    path = INPUT_DIR / "subword_vocabulary.json"

    print("\n[DATA] Loading subword vocabulary...")
    print(f"       {path}")

    if not path.exists():
        raise FileNotFoundError(f"subword_vocabulary.json not found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    print("[✓] Subword vocabulary loaded")

    return vocabulary


# ============================================================
# BPE TOKENIZER
# ============================================================

def load_bpe_tokenizer():

    path = INPUT_DIR / "bpe_tokenizer.json"

    print("\n[DATA] Loading Lab-3 BPE tokenizer...")
    print(f"       {path}")

    if not path.exists():
        raise FileNotFoundError(f"bpe_tokenizer.json not found:\n{path}")

    # The JSON file is the serialized tokenizer model, not plain data —
    # Tokenizer.from_file() reconstructs the actual tokenizer object,
    # so this must not be read with json.load().
    tokenizer = Tokenizer.from_file(str(path))

    print("[✓] BPE tokenizer reconstructed")

    return tokenizer
