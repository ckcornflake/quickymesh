"""
Tests for src/image_utils.py — pad_to_square and make_review_sheet.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.image_utils import make_review_sheet, pad_to_square, pad_image_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_file(tmp_path: Path, name: str, color=(200, 100, 50),
                     size=(300, 200)) -> Path:
    """Create a small solid-colour PNG for testing."""
    img = Image.new("RGB", size, color)
    p = tmp_path / name
    img.save(str(p))
    return p


# ---------------------------------------------------------------------------
# pad_to_square
# ---------------------------------------------------------------------------


class TestPadToSquare:
    def test_small_image_is_centred_on_square(self):
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        result = pad_to_square(img, size=512)
        assert result.size == (512, 512)

    def test_background_is_white_by_default(self):
        img = Image.new("RGB", (100, 100), (0, 0, 255))
        result = pad_to_square(img, size=512)
        # Corner pixels should be white (background)
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((511, 511)) == (255, 255, 255)

    def test_square_image_unchanged_if_same_size(self):
        img = Image.new("RGB", (512, 512), (0, 255, 0))
        result = pad_to_square(img, size=512)
        assert result.size == (512, 512)
        assert result.getpixel((0, 0)) == (0, 255, 0)

    def test_oversized_image_is_downsized(self):
        img = Image.new("RGB", (2000, 1000), (100, 100, 100))
        result = pad_to_square(img, size=512)
        assert result.size == (512, 512)

    def test_custom_bg_color(self):
        img = Image.new("RGB", (100, 50), (0, 0, 0))
        result = pad_to_square(img, size=256, bg_color=(0, 0, 128))
        assert result.getpixel((0, 0)) == (0, 0, 128)

    def test_output_is_rgb(self):
        img = Image.new("RGBA", (100, 100), (10, 20, 30, 128))
        result = pad_to_square(img, size=256)
        assert result.mode == "RGB"


class TestPadImageFile:
    def test_creates_output_file(self, tmp_path):
        src = _make_image_file(tmp_path, "src.png")
        dest = tmp_path / "out" / "padded.png"
        result = pad_image_file(src, dest)
        assert result == dest
        assert dest.exists()

    def test_output_is_square(self, tmp_path):
        src = _make_image_file(tmp_path, "src.png", size=(300, 150))
        dest = tmp_path / "padded.png"
        pad_image_file(src, dest, size=512)
        img = Image.open(dest)
        assert img.size == (512, 512)


# ---------------------------------------------------------------------------
# make_review_sheet
# ---------------------------------------------------------------------------


class TestMakeReviewSheet:
    def _make_images(self, tmp_path, n: int) -> list[Path]:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        ]
        return [
            _make_image_file(tmp_path, f"img_{i}.png", color=colors[i % len(colors)])
            for i in range(n)
        ]

    def test_creates_output_file(self, tmp_path):
        images = self._make_images(tmp_path, 4)
        out = tmp_path / "review.png"
        result = make_review_sheet(images, out, thumb_size=64)
        assert result == out
        assert out.exists()

    def test_output_is_valid_image(self, tmp_path):
        images = self._make_images(tmp_path, 4)
        out = tmp_path / "review.png"
        make_review_sheet(images, out, thumb_size=64)
        img = Image.open(out)
        assert img.mode == "RGB"
        assert img.width > 0 and img.height > 0

    @pytest.mark.parametrize("n, expected_cols, expected_rows", [
        (1, 1, 1),
        (4, 2, 2),
        (5, 3, 2),
        (6, 3, 2),
    ])
    def test_grid_dimensions_match_review_sheet_dims(
        self, tmp_path, n, expected_cols, expected_rows
    ):
        images = self._make_images(tmp_path, n)
        out = tmp_path / "review.png"
        thumb = 64
        padding = 4
        make_review_sheet(images, out, thumb_size=thumb, padding=padding)
        img = Image.open(out)
        expected_w = expected_cols * thumb + (expected_cols + 1) * padding
        expected_h = expected_rows * thumb + (expected_rows + 1) * padding
        assert img.width == expected_w
        assert img.height == expected_h

    def test_creates_parent_dirs(self, tmp_path):
        images = self._make_images(tmp_path, 2)
        out = tmp_path / "deep" / "nested" / "review.png"
        make_review_sheet(images, out, thumb_size=32)
        assert out.exists()

    def test_empty_images_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            make_review_sheet([], tmp_path / "review.png")

    def test_missing_image_produces_placeholder(self, tmp_path):
        images = self._make_images(tmp_path, 2)
        images.append(tmp_path / "nonexistent.png")
        out = tmp_path / "review.png"
        # Should not raise — missing images get a grey placeholder
        make_review_sheet(images, out, thumb_size=32)
        assert out.exists()
