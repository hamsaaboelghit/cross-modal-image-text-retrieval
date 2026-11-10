
from pathlib import Path

import json
import re
from collections import Counter

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "captions.csv"
)

SPLITS_PATH = (
    PROJECT_ROOT / "data" / "splits" / "flickr30k" / "image_splits.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "vocabularies"
)


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]


# Vocabulary settings
MIN_WORD_FREQUENCY = 2
N_SPLITS = 5
MAX_LENGTH = 34

def tokenize_caption(caption):
    """
    Clean and tokenize a caption.
    """

    # Convert text to lowercase
    caption = caption.lower()

    # Extract words and punctuation separately
    tokens = re.findall(r"\w+|[^\w\s]", caption)

    return tokens



def build_vocabulary(captions):
    """
    Build a word-to-index vocabulary from training captions only.
    """

    word_counter = Counter()

    # Count word frequencies
    for caption in captions:

        tokens = tokenize_caption(caption)

        word_counter.update(tokens)

    # Start vocabulary with special tokens
    word_to_idx = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    # Add frequent words
    for word, frequency in word_counter.most_common():

        if frequency >= MIN_WORD_FREQUENCY:

            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    return word_to_idx, word_counter


def save_vocabulary(word_to_idx, output_path):
    """
    Save vocabulary as a JSON file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            word_to_idx,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Vocabulary saved to: {output_path}")
 

def encode_caption(caption, word_to_idx):
    """
    Convert a caption into a fixed-length numerical sequence.
    """

    # 1. Tokenize caption
    tokens = tokenize_caption(caption)

    # 2. Add beginning and ending tokens
    tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]

    # 3. Convert words into numerical IDs
    unk_id = word_to_idx[UNK_TOKEN]

    sequence = [
        word_to_idx.get(token, unk_id)
        for token in tokens
    ]

    # 4. Truncate if sequence is too long
    if len(sequence) > MAX_LENGTH:
        sequence = sequence[:MAX_LENGTH - 1]
        sequence.append(word_to_idx[EOS_TOKEN])

    # 5. Pad sequence if it is too short
    pad_id = word_to_idx[PAD_TOKEN]

    while len(sequence) < MAX_LENGTH:
        sequence.append(pad_id)

    return sequence


def main():

    # 1. Load captions and image splits
    captions_df = pd.read_csv(CAPTIONS_PATH)
    splits_df = pd.read_csv(SPLITS_PATH)

    # 2. Process each fold independently
    for fold_number in range(1, N_SPLITS + 1):

        print(f"\nProcessing Fold {fold_number}...")

        # 3. Select training images
        training_images = splits_df[
            (splits_df["split"] == "development") &
            (splits_df["fold"] != fold_number)
        ]["image_name"]

        # 4. Get captions belonging to training images
        training_captions = captions_df[
            captions_df["image_name"].isin(training_images)
        ]["caption"].tolist()

                # Analyze caption lengths
        caption_lengths = pd.Series([
            len(tokenize_caption(caption))
            for caption in training_captions
        ])

        print("\nCaption Length Statistics:")
        print(f"Mean: {caption_lengths.mean():.2f}")
        print(f"Median: {caption_lengths.median():.0f}")
        print(f"95th percentile: {caption_lengths.quantile(0.95):.0f}")
        print(f"99th percentile: {caption_lengths.quantile(0.99):.0f}")
        print(f"Maximum: {caption_lengths.max()}")

        # 5. Build vocabulary
        word_to_idx, word_counter = build_vocabulary(
            training_captions
        )

        # 6. Save vocabulary
        output_path = OUTPUT_DIR / f"fold_{fold_number}_vocab.json"

        save_vocabulary(word_to_idx, output_path)
        if fold_number == 1:
            test_encoding(word_to_idx)
        # 7. Print statistics
        print(f"Training images: {len(training_images):,}")
        print(f"Training captions: {len(training_captions):,}")
        print(f"Vocabulary size: {len(word_to_idx):,}")
        print(f"Unique words before filtering: {len(word_counter):,}")



def test_encoding(word_to_idx):

    print("\nTesting Caption Encoding...")

    # Test 1: Normal caption
    caption = "A man is playing football"

    sequence = encode_caption(caption, word_to_idx)

    print("\nOriginal caption:")
    print(caption)

    print("\nEncoded sequence:")
    print(sequence)

    print(f"Sequence length: {len(sequence)}")

    # Test 2: Unknown word
    unknown_caption = "A man is playing zzunknownwordxyz"

    unknown_sequence = encode_caption(
        unknown_caption,
        word_to_idx
    )

    print("\nUnknown word test:")
    print(unknown_caption)
    print(unknown_sequence)

    # Test 3: Validate sequence
    assert len(sequence) == MAX_LENGTH
    assert sequence[0] == word_to_idx[BOS_TOKEN]
    assert sequence[-1] == word_to_idx[PAD_TOKEN]

    assert len(unknown_sequence) == MAX_LENGTH
    assert word_to_idx[UNK_TOKEN] in unknown_sequence

    print("\nAll encoding tests passed!")


if __name__ == "__main__":
    main()
