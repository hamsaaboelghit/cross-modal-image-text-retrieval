
import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.data.flickr30k_dataset import (
    Flickr30kDataset,
    ENCODED_DIR,
)

N_SPLITS = 5
MAX_LENGTH = 34

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

EXPECTED_IMAGES = {
    "train": [20340, 20341, 20341, 20341, 20341],
    "validation": [5086, 5085, 5085, 5085, 5085],
    "test": [6357, 6357, 6357, 6357, 6357],
}


def validate_encoded_sequences(data):
    for row_index, sequence_json in enumerate(data["sequence"]):
        sequence = json.loads(sequence_json)

        assert isinstance(sequence, list)
        assert len(sequence) == MAX_LENGTH
        assert all(isinstance(token, int) for token in sequence)
        assert sequence[0] == BOS_ID
        assert EOS_ID in sequence

        eos_position = sequence.index(EOS_ID)
        padding_tokens = sequence[eos_position + 1:]

        assert all(token == PAD_ID for token in padding_tokens), (
            f"Invalid padding at row {row_index}"
        )


@pytest.mark.parametrize("fold_number", range(1, N_SPLITS + 1))
def test_encoded_data_integrity(fold_number):
    data_path = (
        ENCODED_DIR
        / f"fold_{fold_number}"
        / "encoded_captions.csv"
    )

    data = pd.read_csv(data_path)

    assert len(data) == 158915
    assert data["image_name"].nunique() == 31783

    captions_per_image = data.groupby("image_name").size()
    assert (captions_per_image == 5).all()

    for split in ("train", "validation", "test"):
        split_data = data[data["dataset_split"] == split]

        expected_images = EXPECTED_IMAGES[split][fold_number - 1]

        assert split_data["image_name"].nunique() == expected_images
        assert len(split_data) == expected_images * 5

    validate_encoded_sequences(data)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_dataset_sample_and_tensorflow_pipeline(split):
    dataset = Flickr30kDataset(
        fold_number=1,
        split=split,
    )

    assert len(dataset) == EXPECTED_IMAGES[split][0]

    sample = dataset[0]

    assert sample["image"].shape == (224, 224, 3)
    assert sample["image"].dtype == np.float32
    assert np.isfinite(sample["image"]).all()
    assert sample["image"].min() >= 0
    assert sample["image"].max() <= 255

    assert sample["captions"].shape == (5, MAX_LENGTH)
    assert sample["captions"].dtype == np.int32
    assert sample["captions"][0, 0] == BOS_ID

    tf_dataset = dataset.as_dataset(
        batch_size=2,
        shuffle=False,
        prefetch=False,
    )

    batch = next(iter(tf_dataset))

    assert tuple(batch["image"].shape) == (2, 224, 224, 3)
    assert tuple(batch["captions"].shape) == (2, 5, MAX_LENGTH)
    assert tuple(batch["image_name"].shape) == (2,)

    assert batch["image"].dtype == tf.float32
    assert batch["captions"].dtype == tf.int32
    assert batch["image_name"].dtype == tf.string
