
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "captions.csv"
)

SPLITS_PATH = (
    PROJECT_ROOT / "data" / "splits" / "flickr30k" / "image_splits.csv"
)

VOCAB_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "vocabularies"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "encoded"
)

MAX_LENGTH = 34
N_SPLITS = 5


def load_vocabulary(fold_number):
    vocab_path = VOCAB_DIR / f"fold_{fold_number}_vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as file:
        word_to_idx = json.load(file)

    return word_to_idx


def encode_caption(caption, word_to_idx):
    tokens = caption.lower()
    tokens = __import__("re").findall(r"\w+|[^\w\s]", tokens)

    tokens = ["<BOS>"] + tokens + ["<EOS>"]

    unk_id = word_to_idx["<UNK>"]

    sequence = [
        word_to_idx.get(token, unk_id)
        for token in tokens
    ]

    if len(sequence) > MAX_LENGTH:
        sequence = sequence[:MAX_LENGTH - 1]
        sequence.append(word_to_idx["<EOS>"])

    pad_id = word_to_idx["<PAD>"]

    sequence += [pad_id] * (MAX_LENGTH - len(sequence))

    return sequence


def process_fold(fold_number, captions_df, splits_df):
    print(f"\nProcessing Fold {fold_number}...")

    word_to_idx = load_vocabulary(fold_number)

    fold_splits = splits_df.copy()

    fold_splits["dataset_split"] = "train"

    validation_mask = (
        (fold_splits["split"] == "development") &
        (fold_splits["fold"] == fold_number)
    )

    test_mask = fold_splits["split"] == "test"

    fold_splits.loc[validation_mask, "dataset_split"] = "validation"
    fold_splits.loc[test_mask, "dataset_split"] = "test"

    fold_splits = fold_splits[
        ["image_name", "dataset_split"]
    ]

    encoded_df = captions_df.merge(
        fold_splits,
        on="image_name",
        how="inner",
        validate="many_to_one"
    )

    encoded_df["sequence"] = encoded_df["caption"].apply(
        lambda caption: encode_caption(caption, word_to_idx)
    )

    encoded_df["sequence"] = encoded_df["sequence"].apply(
        json.dumps
    )

    output_dir = OUTPUT_DIR / f"fold_{fold_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "encoded_captions.csv"

    encoded_df[
        ["image_name", "caption_number", "dataset_split", "sequence"]
    ].to_csv(output_path, index=False)

    print(f"Vocabulary size: {len(word_to_idx):,}")
    print(f"Total encoded captions: {len(encoded_df):,}")
    print(encoded_df["dataset_split"].value_counts())
    print(f"Saved to: {output_path}")


def main():
    captions_df = pd.read_csv(CAPTIONS_PATH)
    splits_df = pd.read_csv(SPLITS_PATH)

    for fold_number in range(1, N_SPLITS + 1):
        process_fold(fold_number, captions_df, splits_df)

    print("\nAll folds encoded successfully!")


if __name__ == "__main__":
    main()
