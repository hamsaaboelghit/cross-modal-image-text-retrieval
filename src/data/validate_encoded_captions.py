
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT / "data" / "processed" / "flickr30k"
ENCODED_DIR = BASE_DIR / "encoded"
VOCAB_DIR = BASE_DIR / "vocabularies"

MAX_LENGTH = 34


def validate_fold(fold_number):
    print(f"\n{'=' * 50}")
    print(f"VALIDATING FOLD {fold_number}")
    print("=" * 50)

    expected_splits = {
        "train": 101700 if fold_number == 1 else 101705,
        "validation": 25430 if fold_number == 1 else 25425,
        "test": 31785,
    }

    vocab_path = VOCAB_DIR / f"fold_{fold_number}_vocab.json"
    data_path = (
        ENCODED_DIR / f"fold_{fold_number}" / "encoded_captions.csv"
    )

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    df = pd.read_csv(data_path)

    errors = []

    required_tokens = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    for token, expected_id in required_tokens.items():
        if vocab.get(token) != expected_id:
            errors.append(f"Incorrect special token: {token}")

    vocab_ids = set(vocab.values())

    if vocab_ids != set(range(len(vocab))):
        errors.append("Vocabulary IDs are not contiguous and unique")

    if len(df) != 158915:
        errors.append(f"Incorrect total captions: {len(df)}")

    if df.duplicated(["image_name", "caption_number"]).any():
        errors.append("Duplicate caption IDs found")

    for split, expected_count in expected_splits.items():
        actual_count = int((df["dataset_split"] == split).sum())

        if actual_count != expected_count:
            errors.append(
                f"{split}: expected {expected_count}, got {actual_count}"
            )

    for image_name, group in df.groupby("image_name"):
        if len(group) != 5:
            errors.append(f"{image_name}: does not have 5 captions")

        if group["dataset_split"].nunique() != 1:
            errors.append(f"{image_name}: appears in multiple splits")

    unk_count = 0
    total_words = 0

    for row_index, row in df.iterrows():
        try:
            sequence = json.loads(row["sequence"])
        except (ValueError, TypeError):
            errors.append(f"Row {row_index}: invalid JSON sequence")
            continue

        if len(sequence) != MAX_LENGTH:
            errors.append(f"Row {row_index}: incorrect sequence length")
            continue

        if sequence[0] != 2:
            errors.append(f"Row {row_index}: missing BOS")
            continue

        if any(token not in vocab_ids for token in sequence):
            errors.append(f"Row {row_index}: token ID outside vocabulary")

        try:
            eos_index = sequence.index(3)
        except ValueError:
            errors.append(f"Row {row_index}: missing EOS")
            continue

        if any(token != 0 for token in sequence[eos_index + 1:]):
            errors.append(f"Row {row_index}: invalid tokens after EOS")

        unk_count += sequence.count(1)
        total_words += sum(
            token not in (0, 2, 3) for token in sequence
        )

    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Total captions: {len(df):,}")
    print(f"UNK percentage: {100 * unk_count / max(total_words, 1):.4f}%")

    if errors:
        print(f"\nFAILED: {len(errors)} errors")

        for error in errors[:20]:
            print("-", error)

        return False

    print("\nPASSED: All checks successful!")
    return True


def main():
    results = []

    for fold in range(1, 6):
        results.append(validate_fold(fold))

    print(f"\n{'=' * 50}")
    print(f"FINAL RESULT: {sum(results)}/5 folds passed")
    print("=" * 50)


if __name__ == "__main__":
    main()
