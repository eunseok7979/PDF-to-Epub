"""
pdf_extractor.py
----------------
Extracts text blocks, structural metadata, and rendered page images from a PDF.
Uses PyMuPDF (fitz) as the primary parsing engine.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import io


# Minimum ratio of text characters to page area to consider a page "born-digital"
_BORN_DIGITAL_CHAR_THRESHOLD = 10  # characters per page (very conservative)


@dataclasses.dataclass
class TextBlock:
    """A single block of text with font/position metadata."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_height: float       # used to compute relative vertical position
    font_size: float
    is_bold: bool
    is_italic: bool
    block_type: str = "paragraph"  # paragraph | heading | footnote | endnote | caption


@dataclasses.dataclass
class PageData:
    """All data extracted from a single PDF page."""
    page_number: int          # 0-based
    width: float
    height: float
    is_scanned: bool          # True → no selectable text, OCR needed
    text_blocks: List[TextBlock]
    # PIL Image rendered at target_dpi — populated only when is_scanned=True
    rendered_image: Optional[Image.Image] = None


def _page_is_scanned(page: fitz.Page) -> bool:
    """Return True if the page has no meaningful selectable text."""
    text = page.get_text("text").strip()
    return len(text) < _BORN_DIGITAL_CHAR_THRESHOLD


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a PDF page to a PIL Image at the requested DPI."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """
    Extract text blocks from a born-digital page.
    Returns blocks sorted in reading order (top-to-bottom, left-to-right).
    """
    blocks = []
    page_height = page.rect.height

    # get_text("dict") gives per-span font information
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # type 0 = text, type 1 = image
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue

                flags = span.get("flags", 0)
                is_bold = bool(flags & 2**4)    # bit 4 = bold
                is_italic = bool(flags & 2**1)  # bit 1 = italic
                font_size = span.get("size", 12.0)
                bbox = span.get("bbox", (0, 0, 0, 0))

                blocks.append(TextBlock(
                    text=text,
                    x0=bbox[0], y0=bbox[1],
                    x1=bbox[2], y1=bbox[3],
                    page_height=page_height,
                    font_size=font_size,
                    is_bold=is_bold,
                    is_italic=is_italic,
                ))

    # Sort: top-to-bottom, then left-to-right
    blocks.sort(key=lambda b: (round(b.y0 / 5) * 5, b.x0))
    return blocks


def extract_pdf(pdf_path: str | Path, dpi: int = 300) -> List[PageData]:
    """
    Main entry point. Opens the PDF and returns a list of PageData objects,
    one per page.

    Args:
        pdf_path: Path to the input PDF.
        dpi: Resolution for rendering scanned pages (used by OCR engines).

    Returns:
        List of PageData, one per page, in document order.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages: List[PageData] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        rect = page.rect
        scanned = _page_is_scanned(page)

        # Always render the page so born-digital pages can also go through OCR
        # engines for reconciliation comparison.
        rendered = _render_page(page, dpi)

        if scanned:
            text_blocks = []
        else:
            text_blocks = _extract_text_blocks(page)

        pages.append(PageData(
            page_number=page_index,
            width=rect.width,
            height=rect.height,
            is_scanned=scanned,
            text_blocks=text_blocks,
            rendered_image=rendered,
        ))

    doc.close()
    return pages


def get_dominant_font_size(pages: List[PageData]) -> float:
    """
    Compute the most common font size across all born-digital pages.
    Used as the 'body' font size baseline for heading detection.
    """
    from collections import Counter
    sizes: List[float] = []
    for page in pages:
        for block in page.text_blocks:
            sizes.append(round(block.font_size, 1))
    if not sizes:
        return 12.0
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]
