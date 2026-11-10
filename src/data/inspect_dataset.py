
from pathlib import Path
from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"
CAPTIONS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "flickr30k" / "results_20130124.token"
)


def inspect_dataset():

    print("=" * 50)
    print("FLICKR30K DATASET INSPECTION")
    print("=" * 50)

    # Check paths
    if not IMAGE_DIR.exists():
        print("ERROR: Image directory not found!")
        return

    if not CAPTIONS_FILE.exists():
        print("ERROR: Captions file not found!")
        return

    # Collect image filenames
    image_files = {
        path.name
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }

    # Read captions
    captions_per_image = Counter()
    total_captions = 0
    malformed_lines = 0
    sample_captions = []

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                malformed_lines += 1
                continue

            caption_id, caption_text = parts

            if "#" not in caption_id:
                malformed_lines += 1
                continue

            image_name, caption_number = caption_id.rsplit("#", 1)

            captions_per_image[image_name] += 1
            total_captions += 1

            if len(sample_captions) < 3:
                sample_captions.append(
                    (image_name, caption_number, caption_text)
                )

    # Compare images and captions
    caption_image_names = set(captions_per_image.keys())

    missing_images = caption_image_names - image_files
    images_without_captions = image_files - caption_image_names

    caption_distribution = Counter(captions_per_image.values())

    # Print report
    print(f"\nTotal images: {len(image_files):,}")
    print(f"Total captions: {total_captions:,}")
    print(f"Unique images in captions: {len(caption_image_names):,}")

    print("\nCaption distribution:")
    for number, count in sorted(caption_distribution.items()):
        print(f"  {number} captions: {count:,} images")

    print("\nMatching results:")
    print(f"Images missing from folder: {len(missing_images):,}")
    print(f"Images without captions: {len(images_without_captions):,}")
    print(f"Malformed lines: {malformed_lines:,}")

    print("\nSample captions:")
    for image, number, caption in sample_captions:
        print(f"{image} #{number}: {caption}")

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset()
