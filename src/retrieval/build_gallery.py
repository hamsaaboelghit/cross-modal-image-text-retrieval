
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.flickr30k_dataset import Flickr30kDataset
from app.backend.model_service import ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "captions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "retrieval_gallery_fold1"
OUTPUT_PATH = OUTPUT_DIR / "gallery.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0


def load_development_dataset():
    train_dataset = Flickr30kDataset(1, "train")
    validation_dataset = Flickr30kDataset(1, "validation")

    dataset = ConcatDataset([
        train_dataset,
        validation_dataset,
    ])

    print("Train images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Total development images:", len(dataset))

    return dataset


def load_original_captions():
    if not CAPTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Captions file not found: {CAPTIONS_PATH}"
        )

    df = pd.read_csv(CAPTIONS_PATH)

    captions_by_image = {}

    for image_name, group in df.groupby("image_name", sort=False):
        group = group.sort_values("caption_number")

        captions_by_image[image_name] = (
            group["caption"].astype(str).tolist()
        )

    return captions_by_image


@torch.inference_mode()
def build_gallery():
    print("\nLoading development dataset...")

    dataset = load_development_dataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("\nLoading original captions...")

    captions_by_image = load_original_captions()

    print("Images with captions:", len(captions_by_image))

    print("\nLoading model...")

    service = ModelService().load()

    print("Model loaded on:", service.device)

    image_embeddings_list = []
    text_embeddings_list = []
    image_names = []

    total_batches = len(loader)

    print("\nExtracting embeddings...")

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            service.device,
            non_blocking=True,
        )

        captions = batch["captions"].to(
            service.device,
            non_blocking=True,
        )

        batch_names = list(batch["image_name"])

        batch_size = images.size(0)

        # Image embeddings: [B, 512]
        image_embeddings = service.model.encode_image(images)

        # Caption tokens: [B, 5, 34]
        # Flatten to [B * 5, 34]
        flat_captions = captions.reshape(
            batch_size * 5,
            captions.size(-1),
        )

        # Text embeddings: [B * 5, 512]
        flat_text_embeddings = service.model.encode_text(
            flat_captions
        )

        # Restore shape: [B, 5, 512]
        text_embeddings = flat_text_embeddings.reshape(
            batch_size,
            5,
            -1,
        )

        image_embeddings_list.append(
            image_embeddings.cpu()
        )

        text_embeddings_list.append(
            text_embeddings.cpu()
        )

        image_names.extend(batch_names)

        if batch_index % 50 == 0 or batch_index == total_batches:
            print(
                f"Processed batches: {batch_index}/{total_batches} "
                f"| Images: {len(image_names)}"
            )

    print("\nCombining embeddings...")

    all_image_embeddings = torch.cat(
        image_embeddings_list,
        dim=0,
    )

    all_text_embeddings = torch.cat(
        text_embeddings_list,
        dim=0,
    )

    print("\nValidating gallery metadata...")

    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate image names detected.")

    original_captions = []

    for image_name in image_names:
        if image_name not in captions_by_image:
            raise ValueError(
                f"Original captions missing for: {image_name}"
            )

        image_captions = captions_by_image[image_name]

        if len(image_captions) != 5:
            raise ValueError(
                f"Expected 5 captions for {image_name}, "
                f"found {len(image_captions)}"
            )

        original_captions.append(image_captions)

    if all_image_embeddings.shape != (len(image_names), 512):
        raise ValueError("Unexpected image embedding shape.")

    if all_text_embeddings.shape != (len(image_names), 5, 512):
        raise ValueError("Unexpected text embedding shape.")

    if not torch.isfinite(all_image_embeddings).all():
        raise ValueError("Non-finite image embeddings detected.")

    if not torch.isfinite(all_text_embeddings).all():
        raise ValueError("Non-finite text embeddings detected.")

    gallery = {
        "image_embeddings": all_image_embeddings,
        "text_embeddings": all_text_embeddings,
        "image_names": image_names,
        "captions": original_captions,
        "metadata": {
            "fold": 1,
            "model_type": "ResNet50-LSTM",
            "image_count": len(image_names),
            "captions_per_image": 5,
            "embedding_dim": 512,
            "source": "train + validation",
            "test_set_used": False,
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving gallery...")

    torch.save(gallery, OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("GALLERY BUILD COMPLETED")
    print("=" * 50)

    print("Image embeddings:", all_image_embeddings.shape)
    print("Text embeddings:", all_text_embeddings.shape)
    print("Total images:", len(image_names))
    print("Total captions:", len(image_names) * 5)
    print("Image embedding norm:",
          all_image_embeddings[0].norm().item())
    print("Text embedding norm:",
          all_text_embeddings[0, 0].norm().item())
    print("Test set used:", False)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    build_gallery()
