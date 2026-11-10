
from pathlib import Path
import json

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"

ENCODED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "encoded"
)

IMAGE_SIZE = 224


class Flickr30kDataset(Dataset):

    def __init__(self, fold_number, split, transform=None):

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be train, validation, or test"
            )

        self.fold_number = fold_number
        self.split = split

        data_path = (
            ENCODED_DIR
            / f"fold_{fold_number}"
            / "encoded_captions.csv"
        )

        self.data = pd.read_csv(data_path)

        self.data = self.data[
            self.data["dataset_split"] == split
        ].copy()

        self.data = self.data.reset_index(drop=True)

        self.transform = transform or transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.image_names = self.data["image_name"].unique()

        self.captions_by_image = (
            self.data.groupby("image_name")["sequence"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = IMAGE_DIR / image_name

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        captions = [
            json.loads(sequence)
            for sequence in self.captions_by_image[image_name]
        ]

        captions = torch.tensor(
            captions,
            dtype=torch.long
        )

        return {
            "image": image,
            "captions": captions,
            "image_name": image_name
        }



if __name__ == "__main__":

    dataset = Flickr30kDataset(
        fold_number=1,
        split="train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print("DataLoader Test")
    print("-" * 40)

    batch = next(iter(dataloader))

    print(f"Images shape: {batch['image'].shape}")
    print(f"Captions shape: {batch['captions'].shape}")
    print(f"Image names: {len(batch['image_name'])}")

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["captions"].shape == (4, 5, 34)
    assert len(batch["image_name"]) == 4

    print("\nDataLoader test passed!")
