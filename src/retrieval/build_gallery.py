
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from app.backend.model_service import ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "captions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "retrieval_gallery_fold1"
)

OUTPUT_PATH = OUTPUT_DIR / "gallery.npz"

BATCH_SIZE = 32
FOLD_NUMBER = 1
EMBEDDING_DIM = 512
CAPTIONS_PER_IMAGE = 5


def load_development_dataset():
    """
    Combine training and validation datasets.

    The test split is intentionally excluded.
    """

    train_dataset = Flickr30kDataset(
        fold_number=FOLD_NUMBER,
        split="train",
    )

    validation_dataset = Flickr30kDataset(
        fold_number=FOLD_NUMBER,
        split="validation",
    )

    train_tf = train_dataset.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
    )

    validation_tf = validation_dataset.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
    )

    dataset = train_tf.concatenate(validation_tf)

    train_count = len(train_dataset)
    validation_count = len(validation_dataset)
    total_count = train_count + validation_count

    if total_count == 0:
        raise ValueError(
            "The development dataset contains no images."
        )

    print("Training images:", train_count)
    print("Validation images:", validation_count)
    print("Total development images:", total_count)

    return dataset, {
        "train_count": train_count,
        "validation_count": validation_count,
        "total_count": total_count,
    }


def load_original_captions():
    """Load the five original captions associated with each image."""

    if not CAPTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Captions file not found: {CAPTIONS_PATH}"
        )

    dataframe = pd.read_csv(CAPTIONS_PATH)

    required_columns = {
        "image_name",
        "caption",
        "caption_number",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Captions CSV is missing columns: {missing}"
        )

    if dataframe[list(required_columns)].isnull().any().any():
        raise ValueError(
            "Captions CSV contains missing required values."
        )

    captions_by_image = {}

    for image_name, group in dataframe.groupby(
        "image_name",
        sort=False,
    ):
        group = group.sort_values("caption_number")

        captions = (
            group["caption"]
            .astype(str)
            .tolist()
        )

        if len(captions) != CAPTIONS_PER_IMAGE:
            raise ValueError(
                f"Expected {CAPTIONS_PER_IMAGE} captions "
                f"for {image_name}, found {len(captions)}."
            )

        if len(set(group["caption_number"].tolist())) != (
            CAPTIONS_PER_IMAGE
        ):
            raise ValueError(
                f"Duplicate caption numbers found for {image_name}."
            )

        if any(not caption.strip() for caption in captions):
            raise ValueError(
                f"Empty caption found for {image_name}."
            )

        captions_by_image[str(image_name)] = captions

    return captions_by_image


def decode_image_names(names):
    """Convert TensorFlow string values into Python strings."""

    decoded_names = []

    for name in names:
        if isinstance(name, bytes):
            decoded_names.append(name.decode("utf-8"))
        else:
            decoded_names.append(str(name))

    return decoded_names


def validate_embeddings(
    image_embeddings,
    text_embeddings,
    image_count,
):
    """Validate embedding shapes, values, and normalization."""

    expected_image_shape = (
        image_count,
        EMBEDDING_DIM,
    )

    expected_text_shape = (
        image_count,
        CAPTIONS_PER_IMAGE,
        EMBEDDING_DIM,
    )

    if image_embeddings.shape != expected_image_shape:
        raise ValueError(
            f"Unexpected image embedding shape: "
            f"{image_embeddings.shape}; "
            f"expected {expected_image_shape}."
        )

    if text_embeddings.shape != expected_text_shape:
        raise ValueError(
            f"Unexpected text embedding shape: "
            f"{text_embeddings.shape}; "
            f"expected {expected_text_shape}."
        )

    if not np.isfinite(image_embeddings).all():
        raise ValueError(
            "Image embeddings contain NaN or Inf."
        )

    if not np.isfinite(text_embeddings).all():
        raise ValueError(
            "Text embeddings contain NaN or Inf."
        )

    image_norms = np.linalg.norm(
        image_embeddings,
        axis=-1,
    )

    text_norms = np.linalg.norm(
        text_embeddings,
        axis=-1,
    )

    if not np.allclose(
        image_norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Image embeddings are not L2-normalized."
        )

    if not np.allclose(
        text_norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Text embeddings are not L2-normalized."
        )

    print("Image embedding shape and values: PASS")
    print("Text embedding shape and values: PASS")
    print("L2 normalization: PASS")


def save_gallery(
    image_embeddings,
    text_embeddings,
    image_names,
    original_captions,
    dataset_counts,
    service,
):
    """Save gallery arrays and metadata in a compressed NPZ file."""

    metadata = {
        "fold": FOLD_NUMBER,
        "model_type": service.model_type,
        "image_count": len(image_names),
        "train_image_count": dataset_counts["train_count"],
        "validation_image_count": dataset_counts[
            "validation_count"
        ],
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "embedding_dim": EMBEDDING_DIM,
        "source": "train + validation",
        "test_set_used": False,
        "checkpoint": str(service.checkpoint_path),
        "format": "numpy_npz",
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_DIR / "gallery.tmp"

    print("\nSaving gallery...")

    # Writing through a file handle prevents NumPy from
    # automatically adding an extra .npz extension.
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            image_embeddings=image_embeddings.astype(
                np.float32,
                copy=False,
            ),
            text_embeddings=text_embeddings.astype(
                np.float32,
                copy=False,
            ),
            image_names=np.asarray(
                image_names,
                dtype=str,
            ),
            captions_json=np.asarray(
                json.dumps(
                    original_captions,
                    ensure_ascii=False,
                ),
                dtype=str,
            ),
            metadata_json=np.asarray(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                ),
                dtype=str,
            ),
        )

    temporary_path.replace(OUTPUT_PATH)

    print("Gallery saved successfully:", OUTPUT_PATH)

    return metadata


def build_gallery():
    print("\n" + "=" * 65)
    print("VISIONMATCH AI - GALLERY BUILDER")
    print("=" * 65)

    print("\nLoading training and validation datasets...")

    dataset, dataset_counts = load_development_dataset()

    expected_image_count = dataset_counts["total_count"]

    print("\nLoading original captions...")

    captions_by_image = load_original_captions()

    print(
        "Images with original captions:",
        len(captions_by_image),
    )

    print("\nLoading trained TensorFlow model...")

    service = ModelService().load()

    if service.model is None:
        raise RuntimeError(
            "ModelService did not load a TensorFlow model."
        )

    if service.checkpoint_path is None:
        raise RuntimeError(
            "No checkpoint path was recorded by ModelService."
        )

    print("Model type:", service.model_type)
    print("Checkpoint:", service.checkpoint_path)

    # Preallocate arrays to avoid retaining all batch arrays
    # in separate Python lists.
    image_embeddings = np.empty(
        (expected_image_count, EMBEDDING_DIM),
        dtype=np.float32,
    )

    text_embeddings = np.empty(
        (
            expected_image_count,
            CAPTIONS_PER_IMAGE,
            EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    image_names = []
    seen_names = set()
    offset = 0

    print("\nExtracting image and text embeddings...")

    for batch_index, batch in enumerate(
        dataset,
        start=1,
    ):
        images = batch["image"]
        captions = batch["captions"]

        batch_names = decode_image_names(
            batch["image_name"].numpy().tolist()
        )

        batch_size = len(batch_names)

        if batch_size == 0:
            continue

        if len(batch_names) != len(set(batch_names)):
            raise ValueError(
                f"Duplicate image names found in batch {batch_index}."
            )

        duplicates = seen_names.intersection(batch_names)

        if duplicates:
            raise ValueError(
                f"Images appeared more than once: {duplicates}"
            )

        outputs = service.model(
            images,
            captions,
            training=False,
        )

        required_outputs = {
            "image_embeddings",
            "text_embeddings",
        }

        if not isinstance(outputs, dict):
            raise TypeError(
                "The model must return a dictionary of embeddings."
            )

        missing_outputs = required_outputs - outputs.keys()

        if missing_outputs:
            raise KeyError(
                f"Model outputs are missing: {missing_outputs}"
            )

        batch_image_embeddings = outputs[
            "image_embeddings"
        ]

        batch_text_embeddings = outputs[
            "text_embeddings"
        ]

        tf.debugging.assert_all_finite(
            batch_image_embeddings,
            "Non-finite image embeddings detected.",
        )

        tf.debugging.assert_all_finite(
            batch_text_embeddings,
            "Non-finite text embeddings detected.",
        )

        tf.debugging.assert_equal(
            tf.shape(batch_image_embeddings),
            (batch_size, EMBEDDING_DIM),
            message="Unexpected image embedding batch shape.",
        )

        tf.debugging.assert_equal(
            tf.shape(batch_text_embeddings),
            (
                batch_size,
                CAPTIONS_PER_IMAGE,
                EMBEDDING_DIM,
            ),
            message="Unexpected text embedding batch shape.",
        )

        end_offset = offset + batch_size

        if end_offset > expected_image_count:
            raise ValueError(
                "More images were generated than expected."
            )

        image_embeddings[offset:end_offset] = (
            batch_image_embeddings.numpy()
        )

        text_embeddings[offset:end_offset] = (
            batch_text_embeddings.numpy()
        )

        image_names.extend(batch_names)
        seen_names.update(batch_names)

        offset = end_offset

        if batch_index % 25 == 0:
            print(
                f"Processed batches: {batch_index} "
                f"| Images: {offset}/{expected_image_count}"
            )

    print("\nValidating generated gallery...")

    if offset != expected_image_count:
        raise ValueError(
            f"Expected {expected_image_count} images, "
            f"but processed {offset}."
        )

    if len(image_names) != len(set(image_names)):
        raise ValueError(
            "Duplicate image names detected in the gallery."
        )

    original_captions = []

    for image_name in image_names:
        if image_name not in captions_by_image:
            raise ValueError(
                f"Original captions are missing for: {image_name}"
            )

        original_captions.append(
            captions_by_image[image_name]
        )

    if len(original_captions) != expected_image_count:
        raise ValueError(
            "Original captions count does not match gallery size."
        )

    validate_embeddings(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_count=expected_image_count,
    )

    metadata = save_gallery(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_names=image_names,
        original_captions=original_captions,
        dataset_counts=dataset_counts,
        service=service,
    )

    print("\n" + "=" * 65)
    print("GALLERY BUILD COMPLETED")
    print("=" * 65)

    print("Image embeddings:", image_embeddings.shape)
    print("Text embeddings:", text_embeddings.shape)
    print("Total images:", len(image_names))
    print(
        "Total captions:",
        len(image_names) * CAPTIONS_PER_IMAGE,
    )
    print("Model type:", metadata["model_type"])
    print("Source:", metadata["source"])
    print("Test set used:", metadata["test_set_used"])
    print("Saved to:", OUTPUT_PATH)

    return OUTPUT_PATH


if __name__ == "__main__":
    build_gallery()
