"""
image_handler.py
----------------
Extracts figure/table images from rendered page images using layout analysis
bounding boxes.  Saves them as PNG files for EPUB inclusion.

This replaces the old PyMuPDF-based embedded image extraction.  Layout
analysis gives more reliable results because it detects figures *visually*,
catching drawn graphics and charts that are not image XObjects.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


@dataclasses.dataclass
class ExtractedImage:
    """Metadata and raw bytes for a single extracted figure/table image."""
    image_id: str           # e.g. "img_0001"
    page_number: int        # 0-based
    x0: float
    y0: float
    x1: float
    y1: float
    width_px: int
    height_px: int
    ext: str                # always "png" for cropped regions
    data: bytes
    saved_path: Optional[Path] = None


def crop_figures(
    rendered_image: Image.Image,
    figure_bboxes: List[Tuple[float, float, float, float]],
    page_number: int,
    start_counter: int = 1,
) -> List[ExtractedImage]:
    """
    Crop figure/table regions from a rendered page image.

    Args:
        rendered_image: Full-page PIL Image rendered at target DPI.
        figure_bboxes: List of (x0, y0, x1, y1) pixel-coordinate bboxes
                       from layout_analyzer.
        page_number: 0-based page index.
        start_counter: Starting number for image_id generation.

    Returns:
        List of ExtractedImage with PNG data.
    """
    w, h = rendered_image.size
    results: List[ExtractedImage] = []

    for i, bbox in enumerate(figure_bboxes):
        x0 = max(0, int(bbox[0]))
        y0 = max(0, int(bbox[1]))
        x1 = min(w, int(bbox[2]))
        y1 = min(h, int(bbox[3]))

        if x1 <= x0 or y1 <= y0:
            continue

        cropped = rendered_image.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        png_data = buf.getvalue()

        img_id = f"img_{start_counter + i:04d}"
        results.append(ExtractedImage(
            image_id=img_id,
            page_number=page_number,
            x0=float(x0), y0=float(y0),
            x1=float(x1), y1=float(y1),
            width_px=x1 - x0,
            height_px=y1 - y0,
            ext="png",
            data=png_data,
        ))

    return results


def save_images(images: List[ExtractedImage], output_dir: str | Path) -> None:
    """
    Save all extracted images to output_dir.
    Sets the saved_path attribute on each image.
    Deduplicates by image_id.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    for img in images:
        if img.image_id in written:
            img.saved_path = written[img.image_id]
            continue

        filename = f"{img.image_id}.{img.ext}"
        dest = output_dir / filename
        dest.write_bytes(img.data)
        img.saved_path = dest
        written[img.image_id] = dest
