
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, KFold


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "captions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "splits" / "flickr30k"
)

OUTPUT_PATH = OUTPUT_DIR / "image_splits.csv"


# Reproducibility
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20


def main():

    # 1. Load captions
    df = pd.read_csv(CAPTIONS_PATH)

    # 2. Extract unique image names
    image_names = df["image_name"].drop_duplicates().to_numpy()

    print(f"Total unique images: {len(image_names):,}")

    # 3. Separate the final test set
    development_images, test_images = train_test_split(
        image_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # 4. Create five folds inside development set
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_mapping = {}

    for fold_number, (_, validation_indices) in enumerate(
        kfold.split(development_images),
        start=1
    ):
        for index in validation_indices:
            image_name = development_images[index]
            fold_mapping[image_name] = fold_number

    # 5. Build split information
    records = []

    for image_name in development_images:
        records.append({
            "image_name": image_name,
            "split": "development",
            "fold": fold_mapping[image_name]
        })

    for image_name in test_images:
        records.append({
            "image_name": image_name,
            "split": "test",
            "fold": None
        })

    splits_df = pd.DataFrame(records)

    # 6. Validate the splits

    # Every image must appear exactly once
    assert splits_df["image_name"].is_unique

    # No overlap between development and test
    development_set = set(development_images)
    test_set = set(test_images)

    assert development_set.isdisjoint(test_set)

    # Every development image must have a fold
    assert splits_df[
        splits_df["split"] == "development"
    ]["fold"].notna().all()

    # Test images must not have a fold
    assert splits_df[
        splits_df["split"] == "test"
    ]["fold"].isna().all()

    # Every image must have exactly five captions
    caption_counts = df.groupby("image_name").size()

    assert (caption_counts == 5).all()

    # Total number of assigned images must match the dataset
    assert len(splits_df) == len(image_names)

    print("\nAll split validation checks passed!")


    # 7. Save split information
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(OUTPUT_PATH, index=False)

    # 8. Print summary
    print("\nDataset splitting completed!")
    print(f"Development images: {len(development_images):,}")
    print(f"Test images: {len(test_images):,}")

    print("\nFold distribution:")

    print(
        splits_df[splits_df["split"] == "development"]
        ["fold"]
        .value_counts()
        .sort_index()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

