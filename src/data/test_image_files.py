from pathlib import Path

from src.data.image_files import iter_image_names


def test_iter_image_names_finds_jpeg_files_recursively(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"jpg")
    (tmp_path / "two.JPEG").write_bytes(b"jpeg")
    (tmp_path / "ignore.png").write_bytes(b"png")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "three.jpeg").write_bytes(b"jpeg")
    (nested / "ignore.txt").write_text("text", encoding="utf-8")

    assert set(iter_image_names(tmp_path)) == {
        "one.jpg", "two.JPEG", "three.jpeg"
    }


def test_iter_image_names_raises_for_missing_directory(tmp_path):
    missing = tmp_path / "missing"
    try:
        list(iter_image_names(missing))
    except FileNotFoundError as error:
        assert "Image directory not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")
