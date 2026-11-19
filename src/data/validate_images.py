"""Verify that Flickr30k image files can be decoded by Pillow."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.data.image_files import iter_image_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"


def validate_images(image_dir: str | Path = IMAGE_DIR) -> tuple[int, list[str]]:
    image_dir = Path(image_dir)
    checked = 0
    corrupt = []

    print("=" * 55)
    print("FLICKR30K IMAGE INTEGRITY CHECK")
    print("=" * 55)

    for image_name in iter_image_names(image_dir):
        checked += 1
        image_path = image_dir / image_name
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            corrupt.append(image_name)
            print(f"[CORRUPT] {image_name}: {error}")

        if checked % 1000 == 0:
            print(f"Verified {checked:,} images...", flush=True)

    print(f"\nImages checked: {checked:,}")
    print(f"Corrupt/unreadable images: {len(corrupt):,}")
    print("Image integrity: PASS" if not corrupt else "Image integrity: REVIEW REQUIRED")
    return checked, corrupt


if __name__ == "__main__":
    validate_images()
