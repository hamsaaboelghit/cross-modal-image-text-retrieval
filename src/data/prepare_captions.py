
import csv
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" /
    "flickr30k" / "results_20130124.token"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k"
)

OUTPUT_FILE = OUTPUT_DIR / "captions.csv"


def prepare_captions():

    print("=" * 50)
    print("FLICKR30K CAPTION PREPARATION")
    print("=" * 50)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Captions file not found: {INPUT_FILE}"
        )

    records = []
    seen_ids = set()

    # Read and parse captions
    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line at line {line_number}"
                )

            caption_id, caption_text = parts

            if "#" not in caption_id:
                raise ValueError(
                    f"Invalid caption ID: {caption_id}"
                )

            image_name, caption_number = caption_id.rsplit("#", 1)

            if not caption_number.isdigit():
                raise ValueError(
                    f"Invalid caption number: {caption_id}"
                )

            if caption_id in seen_ids:
                raise ValueError(
                    f"Duplicate caption ID: {caption_id}"
                )

            seen_ids.add(caption_id)

            records.append({
                "image_name": image_name,
                "caption_number": int(caption_number),
                "caption": caption_text
            })

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save structured captions
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "caption_number",
                "caption"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    # Report
    unique_images = len({
        record["image_name"] for record in records
    })

    print(f"\nTotal captions: {len(records):,}")
    print(f"Unique images: {unique_images:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nCaption preparation completed successfully!")


if __name__ == "__main__":
    prepare_captions()
