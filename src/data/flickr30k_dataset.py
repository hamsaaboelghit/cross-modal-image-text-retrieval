
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"

ENCODED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "encoded"
)

IMAGE_SIZE = 224
EXPECTED_CAPTIONS = 5
MAX_LENGTH = 34


class Flickr30kDataset:
    """
    TensorFlow-compatible Flickr30k dataset.

    Each sample contains:
        image:      float32 [224, 224, 3], RGB, range [0, 255]
        captions:   int32   [5, 34]
        image_name: string

    Images remain in [0, 255] because ImageEncoder applies
    ResNet preprocessing internally.
    """

    def __init__(
        self,
        fold_number,
        split,
        image_dir=IMAGE_DIR,
        encoded_dir=ENCODED_DIR,
        image_size=IMAGE_SIZE,
        max_length=MAX_LENGTH,
        expected_captions=EXPECTED_CAPTIONS,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'."
            )

        if fold_number not in range(1, 6):
            raise ValueError("fold_number must be between 1 and 5.")

        self.fold_number = fold_number
        self.split = split
        self.image_dir = Path(image_dir)
        self.encoded_dir = Path(encoded_dir)
        self.image_size = image_size
        self.max_length = max_length
        self.expected_captions = expected_captions

        data_path = (
            self.encoded_dir
            / f"fold_{fold_number}"
            / "encoded_captions.csv"
        )

        if not data_path.exists():
            raise FileNotFoundError(
                f"Encoded captions file not found: {data_path}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        self.data = pd.read_csv(data_path)

        required_columns = {
            "image_name",
            "dataset_split",
            "sequence",
        }

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in encoded CSV: {missing_columns}"
            )

        self.data = self.data[
            self.data["dataset_split"] == split
        ].copy()

        if self.data.empty:
            raise ValueError(
                f"No samples found for fold={fold_number}, split={split}."
            )

        if "caption_number" in self.data.columns:
            self.data = self.data.sort_values(
                ["image_name", "caption_number"]
            )

        self.data = self.data.reset_index(drop=True)

        self.image_names = self.data["image_name"].drop_duplicates().tolist()

        self.captions_by_image = {}

        for image_name, group in self.data.groupby(
            "image_name",
            sort=False
        ):
            captions = [
                self._parse_sequence(value)
                for value in group["sequence"].tolist()
            ]

            if len(captions) != self.expected_captions:
                raise ValueError(
                    f"{image_name} has {len(captions)} captions; "
                    f"expected {self.expected_captions}."
                )

            for caption in captions:
                if len(caption) != self.max_length:
                    raise ValueError(
                        f"Caption for {image_name} has length "
                        f"{len(caption)}; expected {self.max_length}."
                    )

            self.captions_by_image[image_name] = np.asarray(
                captions,
                dtype=np.int32
            )

    @staticmethod
    def _parse_sequence(value):
        """Convert a JSON-encoded caption sequence into integer IDs."""

        if isinstance(value, str):
            sequence = json.loads(value)
        else:
            sequence = value

        if not isinstance(sequence, list):
            raise ValueError("Encoded caption must be a list.")

        return [int(token) for token in sequence]

    def __len__(self):
        """Return the number of unique images."""

        return len(self.image_names)

    def _load_image(self, image_name):
        """Load and resize an image as RGB float32 in [0, 255]."""

        image_path = self.image_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )
            image_array = np.asarray(image, dtype=np.float32)

        return image_array

    def __getitem__(self, index):
        """Return one image and all its captions."""

        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range.")

        image_name = self.image_names[index]

        image = self._load_image(image_name)
        captions = self.captions_by_image[image_name]

        return {
            "image": image,
            "captions": captions,
            "image_name": image_name,
        }

    def as_dataset(
        self,
        batch_size=32,
        shuffle=False,
        drop_remainder=False,
        num_parallel_calls=tf.data.AUTOTUNE,
        prefetch=True,
        seed=42,
    ):
        """
        Create a tf.data.Dataset.

        Output batch shapes:
            image:      [B, 224, 224, 3]
            captions:   [B, 5, 34]
            image_name: [B]
        """

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        output_signature = {
            "image": tf.TensorSpec(
                shape=(self.image_size, self.image_size, 3),
                dtype=tf.float32,
            ),
            "captions": tf.TensorSpec(
                shape=(
                    self.expected_captions,
                    self.max_length
                ),
                dtype=tf.int32,
            ),
            "image_name": tf.TensorSpec(
                shape=(),
                dtype=tf.string,
            ),
        }

        def generator():
            indices = np.arange(len(self))

            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)

            for index in indices:
                sample = self[int(index)]

                yield {
                    "image": sample["image"],
                    "captions": sample["captions"],
                    "image_name": sample["image_name"].encode("utf-8"),
                }

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(self),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_remainder,
        )

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def test_dataset():
    print("Flickr30k TensorFlow Dataset Test")
    print("-" * 45)

    dataset = Flickr30kDataset(
        fold_number=1,
        split="train",
    )

    print(f"Number of images: {len(dataset):,}")

    sample = dataset[0]

    assert sample["image"].shape == (224, 224, 3)
    assert sample["image"].dtype == np.float32
    assert sample["image"].min() >= 0
    assert sample["image"].max() <= 255

    assert sample["captions"].shape == (5, 34)
    assert sample["captions"].dtype == np.int32

    print("Single image shape: PASS")
    print("Image range and dtype: PASS")
    print("Caption shape and dtype: PASS")

    tf_dataset = dataset.as_dataset(
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(tf_dataset))

    assert batch["image"].shape == (4, 224, 224, 3)
    assert batch["captions"].shape == (4, 5, 34)
    assert batch["image_name"].shape == (4,)

    print("Batch image shape:", batch["image"].shape)
    print("Batch captions shape:", batch["captions"].shape)
    print("Batch image names:", batch["image_name"].shape)

    print("\nAll Flickr30k Dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
