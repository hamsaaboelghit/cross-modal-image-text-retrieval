
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from app.backend.model_service import ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "captions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "retrieval_gallery_fold1"
OUTPUT_PATH = OUTPUT_DIR / "gallery.npz"

BATCH_SIZE = 32
FOLD_NUMBER = 1


def load_development_dataset():
    """Combine train and validation only. Never include test data."""

    train_dataset = Flickr30kDataset(FOLD_NUMBER, "train")
    validation_dataset = Flickr30kDataset(FOLD_NUMBER, "validation")

    train_tf = train_dataset.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        prefetch=True,
    )

    validation_tf = validation_dataset.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        prefetch=True,
    )

    dataset = train_tf.concatenate(validation_tf)

    print("Train images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print(
        "Total development images:",
        len(train_dataset) + len(validation_dataset),
    )

    return dataset, len(train_dataset) + len(validation_dataset)


def load_original_captions():
    if not CAPTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Captions file not found: {CAPTIONS_PATH}"
        )

    dataframe = pd.read_csv(CAPTIONS_PATH)

    required_columns = {"image_name", "caption", "caption_number"}
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Captions CSV is missing columns: {missing}"
        )

    captions_by_image = {}

    for image_name, group in dataframe.groupby(
        "image_name",
        sort=False,
    ):
        group = group.sort_values("caption_number")

        captions = group["caption"].astype(str).tolist()

        if len(captions) != 5:
            raise ValueError(
                f"Expected 5 captions for {image_name}, "
                f"found {len(captions)}."
            )

        captions_by_image[str(image_name)] = captions

    return captions_by_image


def build_gallery():
    print("\nLoading development dataset...")
    dataset, expected_image_count = load_development_dataset()

    print("\nLoading original captions...")
    captions_by_image = load_original_captions()

    print("Images with original captions:", len(captions_by_image))

    print("\nLoading TensorFlow model...")
    service = ModelService().load()

    print("Model type:", service.model_type)
    print("Checkpoint:", service.checkpoint_path)

    image_embeddings_list = []
    text_embeddings_list = []
    image_names = []

    print("\nExtracting embeddings...")

    for batch_index, batch in enumerate(dataset, start=1):
        images = batch["image"]
        captions = batch["captions"]
        batch_names = [
            name.decode("utf-8")
            for name in batch["image_name"].numpy().tolist()
        ]

        outputs = service.model(
            images,
            captions,
            training=False,
        )

        image_embeddings = outputs["image_embeddings"]
        text_embeddings = outputs["text_embeddings"]

        tf.debugging.assert_all_finite(
            image_embeddings,
            "Non-finite image embeddings detected.",
        )

        tf.debugging.assert_all_finite(
            text_embeddings,
            "Non-finite text embeddings detected.",
        )

        image_embeddings_list.append(
            image_embeddings.numpy()
        )

        text_embeddings_list.append(
            text_embeddings.numpy()
        )

        image_names.extend(batch_names)

        if batch_index % 50 == 0:
            print(
                f"Processed batches: {batch_index} "
                f"| Images: {len(image_names)}"
            )

    print("\nCombining embeddings...")

    all_image_embeddings = np.concatenate(
        image_embeddings_list,
        axis=0,
    )

    all_text_embeddings = np.concatenate(
        text_embeddings_list,
        axis=0,
    )

    print("\nValidating gallery...")

    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate image names detected.")

    if len(image_names) != expected_image_count:
        raise ValueError(
            "Unexpected number of images in the generated gallery."
        )

    original_captions = []

    for image_name in image_names:
        if image_name not in captions_by_image:
            raise ValueError(
                f"Original captions missing for: {image_name}"
            )

        original_captions.append(
            captions_by_image[image_name]
        )

    expected_images_shape = (len(image_names), 512)
    expected_texts_shape = (len(image_names), 5, 512)

    if all_image_embeddings.shape != expected_images_shape:
        raise ValueError(
            f"Unexpected image embedding shape: "
            f"{all_image_embeddings.shape}"
        )

    if all_text_embeddings.shape != expected_texts_shape:
        raise ValueError(
            f"Unexpected text embedding shape: "
            f"{all_text_embeddings.shape}"
        )

    if not np.isfinite(all_image_embeddings).all():
        raise ValueError("Non-finite image embeddings detected.")

    if not np.isfinite(all_text_embeddings).all():
        raise ValueError("Non-finite text embeddings detected.")

    metadata = {
        "fold": FOLD_NUMBER,
        "model_type": service.model_type,
        "image_count": len(image_names),
        "captions_per_image": 5,
        "embedding_dim": 512,
        "source": "train + validation",
        "test_set_used": False,
        "checkpoint": str(service.checkpoint_path),
        "format": "numpy_npz",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving gallery...")

    np.savez_compressed(
        OUTPUT_PATH,
        image_embeddings=all_image_embeddings.astype(np.float32),
        text_embeddings=all_text_embeddings.astype(np.float32),
        image_names=np.asarray(image_names, dtype=str),
        captions_json=np.asarray(
            json.dumps(original_captions, ensure_ascii=False),
            dtype=str,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata, ensure_ascii=False),
            dtype=str,
        ),
    )

    print("\n" + "=" * 55)
    print("GALLERY BUILD COMPLETED")
    print("=" * 55)
    print("Image embeddings:", all_image_embeddings.shape)
    print("Text embeddings:", all_text_embeddings.shape)
    print("Total images:", len(image_names))
    print("Total captions:", len(image_names) * 5)
    print("Model type:", service.model_type)
    print("Test set used:", False)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    build_gallery()
