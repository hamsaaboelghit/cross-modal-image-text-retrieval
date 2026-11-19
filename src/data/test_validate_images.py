from PIL import Image

from src.data.validate_images import validate_images


def test_validate_images_reports_corrupt_files(tmp_path):
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(tmp_path / "valid.jpg")
    (tmp_path / "broken.jpg").write_bytes(b"not a real image")
    (tmp_path / "ignored.png").write_bytes(b"not scanned")

    checked, corrupt = validate_images(tmp_path)

    assert checked == 2
    assert corrupt == ["broken.jpg"]
