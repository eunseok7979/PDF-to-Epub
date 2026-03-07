"""
image_handler.py
----------------
Extracts embedded images from a PDF at their native resolution.
Preserves JPEG encoding where possible; falls back to PNG for other formats.
Records each image's page position so the EPUB builder can place it correctly.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import io


@dataclasses.dataclass
class ExtractedImage:
    """Metadata and raw bytes for a single extracted image."""
    image_id: str           # unique identifier, e.g. "img_0001"
    page_number: int        # 0-based page index
    # Bounding box on the page (PDF points)
    x0: float
    y0: float
    x1: float
    y1: float
    width_px: int           # native pixel width
    height_px: int          # native pixel height
    ext: str                # "jpg" or "png"
    data: bytes             # raw image bytes
    saved_path: Optional[Path] = None  # set after save_images() is called


def _image_hash(data: bytes) -> str:
    """Short hash used to deduplicate identical images."""
    return hashlib.md5(data).hexdigest()[:8]


def extract_images(pdf_path: str | Path) -> List[ExtractedImage]:
    """
    Extract all embedded raster images from the PDF.

    Strategy:
    - Use page.get_images(full=True) to enumerate image XObjects per page.
    - Use doc.extract_image(xref) to get raw bytes (JPEG preserved as-is).
    - Deduplicate by content hash to avoid saving the same image twice.
    - For each unique image, record all page positions where it appears.

    Args:
        pdf_path: Path to the input PDF.

    Returns:
        List of ExtractedImage, one entry per unique image occurrence.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    seen_hashes: Dict[str, str] = {}   # hash → image_id
    results: List[ExtractedImage] = []
    counter = 1

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]

            try:
                img_dict = doc.extract_image(xref)
            except Exception:
                continue

            raw_bytes: bytes = img_dict["image"]
            ext: str = img_dict.get("ext", "png").lower()
            width: int = img_dict.get("width", 0)
            height: int = img_dict.get("height", 0)

            # Normalise extension
            if ext in ("jpeg", "jpg"):
                ext = "jpg"
            else:
                # Convert non-JPEG to PNG for compatibility
                try:
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG", optimize=False)
                    raw_bytes = buf.getvalue()
                    width, height = pil_img.size
                except Exception:
                    pass
                ext = "png"

            img_hash = _image_hash(raw_bytes)

            if img_hash in seen_hashes:
                image_id = seen_hashes[img_hash]
            else:
                image_id = f"img_{counter:04d}"
                seen_hashes[img_hash] = image_id
                counter += 1

            # Get bounding box of this image on the page
            bbox = _get_image_bbox(page, xref)

            results.append(ExtractedImage(
                image_id=image_id,
                page_number=page_index,
                x0=bbox[0], y0=bbox[1],
                x1=bbox[2], y1=bbox[3],
                width_px=width,
                height_px=height,
                ext=ext,
                data=raw_bytes,
            ))

    doc.close()
    return results


def _get_image_bbox(page: fitz.Page, xref: int) -> Tuple[float, float, float, float]:
    """
    Return the bounding box (x0, y0, x1, y1) of the image on the page.
    Falls back to (0, 0, page.rect.width, page.rect.height) if not found.
    """
    for item in page.get_image_rects(xref):
        r = item  # fitz.Rect
        return (r.x0, r.y0, r.x1, r.y1)
    # Fallback: full page
    r = page.rect
    return (r.x0, r.y0, r.x1, r.y1)


def save_images(images: List[ExtractedImage], output_dir: str | Path) -> None:
    """
    Save all extracted images to output_dir.
    Sets the saved_path attribute on each ExtractedImage.
    Deduplicates by image_id so repeated occurrences are saved only once.

    Args:
        images: List of ExtractedImage from extract_images().
        output_dir: Directory where image files will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}  # image_id → path

    for img in images:
        if img.image_id in written:
            img.saved_path = written[img.image_id]
            continue

        filename = f"{img.image_id}.{img.ext}"
        dest = output_dir / filename
        dest.write_bytes(img.data)
        img.saved_path = dest
        written[img.image_id] = dest


def build_page_image_map(
    images: List[ExtractedImage],
) -> Dict[int, List[ExtractedImage]]:
    """
    Group images by page number for easy lookup during EPUB assembly.

    Returns:
        Dict mapping page_number → list of ExtractedImage on that page,
        sorted by vertical position (y0 ascending).
    """
    page_map: Dict[int, List[ExtractedImage]] = {}
    for img in images:
        page_map.setdefault(img.page_number, []).append(img)
    for page_imgs in page_map.values():
        page_imgs.sort(key=lambda i: i.y0)
    return page_map
