
from pathlib import Path
from PIL import Image, ImageFile
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k-images"

IMAGE_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_images():
    image_paths = [
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"Total images found: {len(image_paths):,}")

    image_sizes = Counter()
    corrupted_images = []

    for image_path in tqdm(image_paths, desc="Inspecting images"):
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image_sizes[image.size] += 1

        except Exception:
            corrupted_images.append(image_path.name)

    print("\nImage Inspection Results")
    print("-" * 35)
    print(f"Readable images: {len(image_paths) - len(corrupted_images):,}")
    print(f"Corrupted images: {len(corrupted_images):,}")

    print("\nMost common original image sizes:")
    for size, count in image_sizes.most_common(10):
        print(f"{size}: {count:,} images")

    if corrupted_images:
        print("\nCorrupted image examples:")
        for name in corrupted_images[:10]:
            print(name)

    return image_paths, corrupted_images


def preprocess_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

    return image


def test_preprocessing(image_paths):
    print("\nTesting Image Preprocessing...")

    if not image_paths:
        raise ValueError("No images found!")

    sample_path = image_paths[0]
    processed_image = preprocess_image(sample_path)

    print(f"Sample image: {sample_path.name}")
    print(f"Processed image size: {processed_image.size}")
    print(f"Image mode: {processed_image.mode}")

    assert processed_image.size == IMAGE_SIZE
    assert processed_image.mode == "RGB"

    print("\nImage preprocessing test passed!")


def main():
    image_paths, corrupted_images = inspect_images()

    valid_images = [
        path for path in image_paths
        if path.name not in set(corrupted_images)
    ]

    test_preprocessing(valid_images)


if __name__ == "__main__":
    main()
