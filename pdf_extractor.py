"""
pdf_extractor.py
----------------
Opens a PDF via PyMuPDF and provides:
  - Page rendering to PIL Image (for layout analysis and OCR).
  - Clip-based native text extraction (for a specific bounding box region).
  - Font metadata extraction for structure detection (headings, etc.).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PageData:
    """Basic data for a single PDF page."""
    page_number: int        # 0-based
    width: float            # in PDF points
    height: float           # in PDF points
    is_scanned: bool        # True if no meaningful selectable text
    rendered_image: Optional[Image.Image] = None  # page rendered at target DPI


# Minimum character count to consider a page as having selectable text.
_BORN_DIGITAL_CHAR_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a PDF page to a PIL Image at the given DPI."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _page_is_scanned(page: fitz.Page) -> bool:
    """Return True if the page has no meaningful selectable text."""
    text = page.get_text("text").strip()
    return len(text) < _BORN_DIGITAL_CHAR_THRESHOLD


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def get_clip_text(page: fitz.Page, bbox_pixels: Tuple[float, float, float, float],
                  dpi: int) -> str:
    """
    Extract native (selectable) text from a rectangular region of the page.

    Args:
        page: A fitz.Page object.
        bbox_pixels: (x0, y0, x1, y1) in **pixel** coordinates from the
                     rendered image (at the given DPI).
        dpi: The DPI used when rendering the page image (needed for
             pixel-to-PDF-point conversion).

    Returns:
        Plain text found within the clipped region.
    """
    zoom = dpi / 72.0
    # Convert pixel coords back to PDF points
    clip = fitz.Rect(
        bbox_pixels[0] / zoom,
        bbox_pixels[1] / zoom,
        bbox_pixels[2] / zoom,
        bbox_pixels[3] / zoom,
    )
    return page.get_text("text", clip=clip).strip()


def get_full_page_text(page: fitz.Page) -> str:
    """Extract all selectable text from a page (no clipping)."""
    return page.get_text("text").strip()


def get_font_info_in_region(
    page: fitz.Page,
    bbox_pixels: Tuple[float, float, float, float],
    dpi: int,
) -> Tuple[float, bool]:
    """
    Return (dominant_font_size, is_bold) for the text inside a pixel-coord bbox.

    Used to classify layout regions as headings vs body text.
    Falls back to (12.0, False) when no spans are found.
    """
    zoom = dpi / 72.0
    clip = fitz.Rect(
        bbox_pixels[0] / zoom,
        bbox_pixels[1] / zoom,
        bbox_pixels[2] / zoom,
        bbox_pixels[3] / zoom,
    )

    sizes: List[float] = []
    bold_count = 0
    total_count = 0

    try:
        raw = page.get_text("dict", clip=clip)
    except Exception:
        return 12.0, False

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                sizes.append(span.get("size", 12.0))
                flags = span.get("flags", 0)
                if flags & (1 << 4):  # bold bit
                    bold_count += 1
                total_count += 1

    if not sizes:
        return 12.0, False

    # Dominant = most frequent, rounded to 1 decimal
    from collections import Counter
    rounded = [round(s, 1) for s in sizes]
    dominant = Counter(rounded).most_common(1)[0][0]
    is_bold = bold_count > total_count / 2
    return dominant, is_bold


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path, dpi: int = 300) -> List[PageData]:
    """
    Open the PDF and return basic page data with rendered images.

    Every page is rendered at *dpi* so that layout analysis and OCR engines
    can operate on the image.  For born-digital (or scanned-with-OCR-layer)
    pages, native text is extracted later via get_clip_text() on a
    per-region basis.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages: List[PageData] = []

    for idx in range(len(doc)):
        page = doc[idx]
        pages.append(PageData(
            page_number=idx,
            width=page.rect.width,
            height=page.rect.height,
            is_scanned=_page_is_scanned(page),
            rendered_image=_render_page(page, dpi),
        ))

    doc.close()
    return pages


def open_pdf(pdf_path: str | Path) -> fitz.Document:
    """Open and return a fitz.Document (caller must close it)."""
    return fitz.open(str(Path(pdf_path)))
