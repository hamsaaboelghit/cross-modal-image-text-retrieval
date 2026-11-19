"""Fast, low-overhead iteration over Flickr30k image files."""

import os
from pathlib import Path
from typing import Iterator

IMAGE_SUFFIXES = (".jpg", ".jpeg")


def iter_image_names(image_dir: str | Path) -> Iterator[str]:
    """Yield image filenames recursively using scandir."""
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    pending = [os.fspath(root)]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(IMAGE_SUFFIXES):
                    yield entry.name
