
import csv
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"

CSV_FILE = (
    PROJECT_ROOT / "data" / "processed" /
    "flickr30k" / "captions.csv"
)


def validate_captions():

    print("=" * 50)
    print("PROCESSED CAPTIONS VALIDATION")
    print("=" * 50)

    captions_per_image = Counter()
    caption_numbers = {}
    empty_captions = 0
    duplicate_ids = 0
    seen_ids = set()

    with open(CSV_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            image_name = row["image_name"]
            caption_number = int(row["caption_number"])
            caption = row["caption"].strip()

            caption_id = (image_name, caption_number)

            if caption_id in seen_ids:
                duplicate_ids += 1

            seen_ids.add(caption_id)

            if not caption:
                empty_captions += 1

            captions_per_image[image_name] += 1

            caption_numbers.setdefault(image_name, []).append(
                caption_number
            )

    image_files = {
        path.name
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
    }

    images_in_csv = set(captions_per_image.keys())

    wrong_caption_count = {
        name: count
        for name, count in captions_per_image.items()
        if count != 5
    }

    wrong_caption_numbers = {
        name: numbers
        for name, numbers in caption_numbers.items()
        if sorted(numbers) != [0, 1, 2, 3, 4]
    }

    print(f"\nTotal captions: {len(seen_ids):,}")
    print(f"Unique images: {len(images_in_csv):,}")
    print(f"Images in folder: {len(image_files):,}")

    print("\nValidation results:")
    print(f"Images missing from folder: {len(images_in_csv - image_files):,}")
    print(f"Images without captions: {len(image_files - images_in_csv):,}")
    print(f"Images without exactly 5 captions: {len(wrong_caption_count):,}")
    print(f"Images with incorrect caption numbering: {len(wrong_caption_numbers):,}")
    print(f"Empty captions: {empty_captions:,}")
    print(f"Duplicate caption IDs: {duplicate_ids:,}")

    print("\nValidation completed.")


if __name__ == "__main__":
    validate_captions()
