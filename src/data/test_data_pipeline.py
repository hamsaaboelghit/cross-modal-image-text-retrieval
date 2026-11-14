
import json

import pandas as pd
import torch

from src.data.flickr30k_dataset import Flickr30kDataset
from src.data.flickr30k_dataset import ENCODED_DIR


N_SPLITS = 5
MAX_LENGTH = 34

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

EXPECTED_IMAGES = {
    "train": [20340, 20341, 20341, 20341, 20341],
    "validation": [5086, 5085, 5085, 5085, 5085],
    "test": [6357, 6357, 6357, 6357, 6357],
}


def validate_encoded_sequences(data, fold_number):

    print("\nChecking encoded sequences...")

    total_sequences = 0

    for row_index, sequence_json in enumerate(data["sequence"]):

        sequence = json.loads(sequence_json)

        assert isinstance(sequence, list), (
            f"Invalid sequence type at row {row_index}"
        )

        assert len(sequence) == MAX_LENGTH, (
            f"Incorrect sequence length at row {row_index}"
        )

        assert all(isinstance(token, int) for token in sequence), (
            f"Non-integer token at row {row_index}"
        )

        assert sequence[0] == BOS_ID, (
            f"Missing BOS token at row {row_index}"
        )

        assert EOS_ID in sequence, (
            f"Missing EOS token at row {row_index}"
        )

        eos_position = sequence.index(EOS_ID)

        padding_tokens = sequence[eos_position + 1:]

        assert all(token == PAD_ID for token in padding_tokens), (
            f"Invalid padding after EOS at row {row_index}"
        )

        total_sequences += 1

    print(
        f"Fold {fold_number}: "
        f"{total_sequences:,} sequences validated | PASS"
    )


def validate_fold(fold_number):

    print("\n" + "=" * 60)
    print(f"DATA PIPELINE TEST - FOLD {fold_number}")
    print("=" * 60)

    data_path = (
        ENCODED_DIR
        / f"fold_{fold_number}"
        / "encoded_captions.csv"
    )

    data = pd.read_csv(data_path)

    # 1. Validate total captions
    assert len(data) == 158915, (
        "Incorrect total number of captions"
    )

    # 2. Validate unique images
    unique_images = data["image_name"].nunique()

    assert unique_images == 31783, (
        "Incorrect number of unique images"
    )

    # 3. Validate captions per image
    captions_per_image = data.groupby("image_name").size()

    assert (captions_per_image == 5).all(), (
        "Some images do not have exactly 5 captions"
    )

    # 4. Validate split assignments
    for split in ["train", "validation", "test"]:

        split_data = data[data["dataset_split"] == split]

        expected_images = EXPECTED_IMAGES[split][fold_number - 1]

        actual_images = split_data["image_name"].nunique()

        expected_captions = expected_images * 5

        assert actual_images == expected_images, (
            f"Incorrect image count in {split}"
        )

        assert len(split_data) == expected_captions, (
            f"Incorrect caption count in {split}"
        )

        print(
            f"{split.capitalize():12} | "
            f"Images: {actual_images:,} | "
            f"Captions: {len(split_data):,} | PASS"
        )

    # 5. Validate encoded sequences
    validate_encoded_sequences(data, fold_number)

    # 6. Validate Dataset and actual sample loading
    for split in ["train", "validation", "test"]:

        dataset = Flickr30kDataset(
            fold_number=fold_number,
            split=split
        )

        expected_images = EXPECTED_IMAGES[split][fold_number - 1]

        assert len(dataset) == expected_images

        sample = dataset[0]

        assert sample["image"].shape == (3, 224, 224)

        assert sample["captions"].shape == (5, MAX_LENGTH)

        assert sample["captions"].dtype == torch.long

        assert torch.isfinite(sample["image"]).all()

        assert sample["captions"][0, 0].item() == BOS_ID

        print(
            f"{split.capitalize():12} | "
            f"Dataset loading and Tensor shapes | PASS"
        )

    print(f"\nFold {fold_number}: ALL CHECKS PASSED")


def main():

    print("=" * 60)
    print("FLICKR30K COMPLETE DATA PIPELINE INTEGRATION TEST")
    print("=" * 60)

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    for fold_number in range(1, N_SPLITS + 1):

        validate_fold(fold_number)

    print("\n" + "=" * 60)
    print("ALL 5 FOLDS PASSED!")
    print("DATA PIPELINE INTEGRATION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
