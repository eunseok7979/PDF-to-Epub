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
    """
    Lazily initialise the layout analysis engine.

    Priority:
    1. paddlex.create_pipeline("layout_detection")  — PaddleOCR v3+
    2. paddleocr.PPStructure                        — PaddleOCR v2 fallback
    """
    global _engine, _engine_init_attempted
    if _engine_init_attempted:
        return _engine
    _engine_init_attempted = True

    # --- Option 1: paddlex (PaddleOCR v3 / paddlepaddle 3.x) ---
    try:
        from paddlex import create_pipeline  # noqa: PLC0415
        pipeline = create_pipeline(pipeline="layout_detection")
        _engine = ("paddlex", pipeline)
        logger.info("Layout analysis: paddlex layout_detection pipeline ready")
        return _engine
    except Exception as exc:
        logger.debug("paddlex layout_detection unavailable: %s", exc)

    # --- Option 2: PPStructure (PaddleOCR v2) ---
    try:
        from paddleocr import PPStructure  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "Layout analysis disabled: neither paddlex nor paddleocr.PPStructure "
            "could be imported. Entire page treated as one text region."
        )
        return None

    init_attempts = [
        {"table": False, "ocr": False, "layout": True, "lang": "ko"},
        {"table": False, "ocr": False, "layout": True},
        {"table": False, "ocr": False},
        {},
    ]
    for kwargs in init_attempts:
        try:
            engine = PPStructure(**kwargs)
            _engine = ("ppstructure", engine)
            logger.info("Layout analysis: PPStructure initialised with %s", kwargs)
            return _engine
        except (TypeError, Exception) as exc:
            logger.debug("PPStructure(%s) failed: %s", kwargs, exc)

    logger.warning("Layout analysis disabled: all init attempts failed.")
    return None


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

def _parse_paddlex_result(result, page_number: int) -> List[LayoutRegion]:
    """
    Parse paddlex layout_detection output.

    Each item yielded by pipeline.predict() contains:
      boxes:  list of [x0, y0, x1, y1]
      labels: list of category name strings
      scores: list of confidence floats
    """
    regions: List[LayoutRegion] = []
    # result may be a generator; materialise it
    items = list(result) if not isinstance(result, list) else result

    for item in items:
        # Support both dict-like and attribute access
        def _get(key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        boxes  = _get("boxes",  [])
        labels = _get("labels", [])
        scores = _get("scores", [])

        for i, (box, label) in enumerate(zip(boxes, labels)):
            score = float(scores[i]) if i < len(scores) else 0.0
            try:
                x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            except (IndexError, TypeError, ValueError):
                continue
            regions.append(LayoutRegion(
                region_type=str(label).lower().strip(),
                bbox=(x0, y0, x1, y1),
                confidence=score,
                page_number=page_number,
            ))

    return regions


def _parse_ppstructure_result(result, page_number: int) -> List[LayoutRegion]:
    """Parse PPStructure (PaddleOCR v2) output."""
    regions: List[LayoutRegion] = []
    items = result if isinstance(result, list) else [result]

    for item in items:
        if isinstance(item, list):
            regions.extend(_parse_ppstructure_result(item, page_number))
            continue
        if not isinstance(item, dict):
            continue

        rtype = str(item.get("type", "text")).lower().strip()
        bbox  = item.get("bbox")
        score = item.get("score", item.get("confidence", 0.0))
        if bbox is None:
            continue
        try:
            if isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
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
    engine_info = _get_engine()
    if engine_info is None:
        return _full_page_fallback(image, page_number)

    engine_type, engine = engine_info
    img_array = np.array(image)

    try:
        if engine_type == "paddlex":
            # predict() is a generator — pass img_array directly
            result = engine.predict(img_array)
            regions = _parse_paddlex_result(result, page_number)
        else:
            # PPStructure: try predict() then __call__()
            try:
                result = engine.predict(img_array)
            except (AttributeError, TypeError):
                result = engine(img_array)
            regions = _parse_ppstructure_result(result, page_number)

        if not regions:
            return _full_page_fallback(image, page_number)

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
