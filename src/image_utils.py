"""
Image utilities for the quickymesh pipeline.

  pad_to_square   — paste an image centred on a white square background
                    without rescaling (used to bring concept arts to 1024×1024)
  make_review_sheet — combine labelled thumbnails into a near-square grid PNG
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.state import review_sheet_dims


# ---------------------------------------------------------------------------
# pad_to_square
# ---------------------------------------------------------------------------


def pad_to_square(
    image: Image.Image,
    size: int = 1024,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Return a new `size × size` image with `image` centred on a solid-colour
    background.  The original image is never rescaled — only padded.

    If the image is larger than `size` in either dimension it is downsized
    while preserving aspect ratio so it fits, then centred.
    """
    img = image.convert("RGB")

    # Downscale only if necessary (preserve aspect ratio)
    if img.width > size or img.height > size:
        img.thumbnail((size, size), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), bg_color)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def pad_image_file(
    src: Path,
    dest: Path,
    size: int = 1024,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Load `src`, pad it, save to `dest`, return `dest`."""
    img = Image.open(src)
    result = pad_to_square(img, size=size, bg_color=bg_color)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(dest))
    return dest


# ---------------------------------------------------------------------------
# make_review_sheet
# ---------------------------------------------------------------------------


def make_review_sheet(
    images: list[Path],
    output_path: Path,
    thumb_size: int = 256,
    padding: int = 4,
    label_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (30, 30, 30),
) -> Path:
    """
    Combine images into a labelled review-sheet PNG.

    Layout is as close to square as possible (see `review_sheet_dims`).
    Each cell is `thumb_size × thumb_size` pixels.  Images are fitted
    inside the cell without distortion (letterboxed on the cell bg).
    Each cell gets a 1-based numeric label in its top-right corner.

    Args:
        images:       Ordered list of image paths (1-indexed in the sheet).
        output_path:  Destination file path (PNG).
        thumb_size:   Cell size in pixels (square).
        padding:      Gap between cells and around the border.
        label_color:  RGB colour for the index label text.
        bg_color:     RGB colour for the grid background.

    Returns:
        output_path
    """
    if not images:
        raise ValueError("images list must not be empty")

    tw = th = thumb_size
    cols, rows = review_sheet_dims(len(images))

    grid_w = cols * tw + (cols + 1) * padding
    grid_h = rows * th + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, tw // 16))
    except (IOError, OSError):
        font = ImageFont.load_default()

    for idx, img_path in enumerate(images):
        col = idx % cols
        row = idx // cols
        cell_x = padding + col * (tw + padding)
        cell_y = padding + row * (th + padding)

        # Load and fit image into cell without distortion
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((tw, th), Image.LANCZOS)
            # Centre the thumbnail in the cell
            ox = cell_x + (tw - img.width) // 2
            oy = cell_y + (th - img.height) // 2
        except Exception:
            img = Image.new("RGB", (tw, th), color=(80, 80, 80))
            ox, oy = cell_x, cell_y

        grid.paste(img, (ox, oy))

        # Label: 1-based index, top-right corner of the cell
        label = str(idx + 1)
        draw = ImageDraw.Draw(grid)
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        margin = 4
        lx = cell_x + tw - lw - margin
        ly = cell_y + margin
        # Drop shadow for legibility on any background
        draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0), font=font)
        draw.text((lx, ly), label, fill=label_color, font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    return output_path
