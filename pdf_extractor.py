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


def get_paragraph_text(
    page: fitz.Page,
    bbox_pixels: Tuple[float, float, float, float],
    dpi: int,
) -> str:
    """
    Extract text from a region with paragraph boundaries restored using line bboxes.

    Lines whose right edge ends significantly before the region's right margin,
    or that end with sentence-final punctuation, are treated as paragraph breaks.
    Remaining consecutive lines are joined with a space to form a single paragraph.

    Falls back to get_clip_text() if line-level data is unavailable.
    """
    zoom = dpi / 72.0
    clip = fitz.Rect(
        bbox_pixels[0] / zoom,
        bbox_pixels[1] / zoom,
        bbox_pixels[2] / zoom,
        bbox_pixels[3] / zoom,
    )

    try:
        raw = page.get_text("dict", clip=clip)
    except Exception:
        return page.get_text("text", clip=clip).strip()

    # Collect (line_text, x1_pixels) for every non-empty line
    line_items: List[Tuple[str, float]] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            lbbox = line.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1_px = lbbox[2] * zoom
            line_items.append((text, x1_px))

    if not line_items:
        return page.get_text("text", clip=clip).strip()

    # Estimate right margin from 95th-percentile x1 across all lines
    x1_vals = sorted(x for _, x in line_items)
    idx_95 = max(0, int(len(x1_vals) * 0.95) - 1)
    right_margin = x1_vals[idx_95]
    short_threshold = right_margin * 0.82  # line ending before 82% → paragraph break

    _SENTENCE_END = frozenset(".!?。！？")
    _KO_ENDINGS = (
        "다.", "다!", "다?", "습니다.", "습니다!", "입니다.",
        "이다.", "였다.", "됩니다.", "합니다.", "했다.",
    )

    paragraphs: List[str] = []
    current: List[str] = []

    for text, x1 in line_items:
        current.append(text)
        is_short = x1 < short_threshold
        ends_sentence = bool(text) and (
            text[-1] in _SENTENCE_END
            or any(text.endswith(e) for e in _KO_ENDINGS)
        )
        if is_short or ends_sentence:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n".join(paragraphs) if paragraphs else page.get_text("text", clip=clip).strip()


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
