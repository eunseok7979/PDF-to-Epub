"""
layout_analyzer.py
------------------
Page layout analysis using PaddleOCR PP-StructureV2.

Detects region types on each page:
  - header, footer, page_number  -> discarded (not included in EPUB)
  - figure, table               -> cropped as images for EPUB
  - text, title, reference, etc -> passed to OCR reconciliation

This module acts as the gatekeeper for the entire pipeline:
headers/footers are stripped, figures are extracted as images,
and only text regions go through the 3-way OCR comparison.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LayoutRegion:
    """A detected region on a rendered page."""
    region_type: str  # text, title, figure, table, header, footer, reference, ...
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) pixel coords
    confidence: float = 0.0
    page_number: int = 0


# Region type classification
DISCARD_TYPES = frozenset({"header", "footer", "page_number"})
FIGURE_TYPES = frozenset({"figure", "table"})
TEXT_TYPES = frozenset({
    "text", "title", "reference", "equation",
    "figure_caption", "table_caption",
})


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine = None
_engine_init_attempted = False


def _get_engine():
    """Lazily initialise the PPStructure layout analysis engine."""
    global _engine, _engine_init_attempted
    if _engine_init_attempted:
        return _engine
    _engine_init_attempted = True

    try:
        from paddleocr import PPStructure
    except ImportError:
        logger.warning(
            "paddleocr.PPStructure not available. "
            "Layout analysis disabled; entire page treated as one text region."
        )
        return None

    # Try several parameter combinations (API varies across PaddleOCR versions)
    init_attempts = [
        {"table": False, "ocr": False, "layout": True, "lang": "ko"},
        {"table": False, "ocr": False, "layout": True},
        {"table": False, "ocr": False, "lang": "ko"},
        {"table": False, "ocr": False},
        {},
    ]

    for kwargs in init_attempts:
        try:
            _engine = PPStructure(**kwargs)
            logger.info("PPStructure initialised with: %s", kwargs)
            return _engine
        except (TypeError, Exception) as exc:
            logger.debug("PPStructure(%s) failed: %s", kwargs, exc)
            continue

    logger.warning(
        "All PPStructure init attempts failed. Layout analysis disabled."
    )
    return None


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_result(result, page_number: int) -> List[LayoutRegion]:
    """Parse PPStructure output into LayoutRegion objects."""
    regions: List[LayoutRegion] = []

    if not result:
        return regions

    items = result if isinstance(result, list) else [result]

    for item in items:
        # Handle list-of-lists (some versions nest results)
        if isinstance(item, list):
            regions.extend(_parse_result(item, page_number))
            continue

        if isinstance(item, dict):
            rtype = str(item.get("type", "text")).lower().strip()
            bbox = item.get("bbox", None)
            score = item.get("score", item.get("confidence", 0.0))

            if bbox is None:
                continue

            # bbox formats: [x0,y0,x1,y1] or [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            try:
                if isinstance(bbox[0], (list, tuple)):
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                else:
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            except (IndexError, TypeError):
                continue

            regions.append(LayoutRegion(
                region_type=rtype,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                confidence=float(score) if score else 0.0,
                page_number=page_number,
            ))

        # Handle object-based results (newer PaddleOCR may return objects)
        elif hasattr(item, "type") and hasattr(item, "bbox"):
            rtype = str(getattr(item, "type", "text")).lower().strip()
            bbox = getattr(item, "bbox", None)
            score = getattr(item, "score", getattr(item, "confidence", 0.0))
            if bbox is not None:
                try:
                    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                except (IndexError, TypeError):
                    continue
                regions.append(LayoutRegion(
                    region_type=rtype,
                    bbox=(x0, y0, x1, y1),
                    confidence=float(score) if score else 0.0,
                    page_number=page_number,
                ))

    return regions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _full_page_fallback(image: Image.Image, page_number: int) -> List[LayoutRegion]:
    """Return a single text region covering the entire page."""
    w, h = image.size
    return [LayoutRegion(
        region_type="text",
        bbox=(0.0, 0.0, float(w), float(h)),
        confidence=1.0,
        page_number=page_number,
    )]


def analyze_layout(
    image: Image.Image,
    page_number: int = 0,
) -> List[LayoutRegion]:
    """
    Run layout analysis on a rendered page image.

    Returns a list of LayoutRegion sorted by vertical position.
    Falls back to a single full-page "text" region if layout analysis
    is unavailable or fails.
    """
    engine = _get_engine()
    if engine is None:
        return _full_page_fallback(image, page_number)

    img_array = np.array(image)

    try:
        # Try predict() (newer API) first, then __call__() (older API)
        try:
            result = engine.predict(img_array)
        except (AttributeError, TypeError):
            result = engine(img_array)

        regions = _parse_result(result, page_number)

        if not regions:
            return _full_page_fallback(image, page_number)

        # Sort: top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    except Exception as exc:
        logger.warning("Layout analysis failed for page %d: %s", page_number, exc)
        return _full_page_fallback(image, page_number)


def filter_regions(
    regions: List[LayoutRegion],
) -> Tuple[List[LayoutRegion], List[LayoutRegion], List[LayoutRegion]]:
    """
    Classify regions into three groups.

    Returns:
        (text_regions, figure_regions, discarded_regions)
    """
    text_regions: List[LayoutRegion] = []
    figure_regions: List[LayoutRegion] = []
    discarded: List[LayoutRegion] = []

    for r in regions:
        if r.region_type in DISCARD_TYPES:
            discarded.append(r)
        elif r.region_type in FIGURE_TYPES:
            figure_regions.append(r)
        else:
            # Default unknown types to text
            text_regions.append(r)

    return text_regions, figure_regions, discarded


def crop_region(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop a region from the page image using pixel-coordinate bbox."""
    w, h = image.size
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2]))
    y1 = min(h, int(bbox[3]))
    return image.crop((x0, y0, x1, y1))
