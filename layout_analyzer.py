"""
layout_analyzer.py
------------------
Page layout analysis with Surya (primary) or PaddleX (fallback).

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
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Skip the slow model-source-availability check at startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LayoutRegion:
    """A detected region on a rendered page."""
    region_type: str  # text, title, figure, table, header, footer, reference, ...
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) pixel coords
    confidence: float = 0.0
    page_number: int = 0


# Region type classification (lowercase, covers both Surya and PaddleX labels)
DISCARD_TYPES = frozenset({
    "header", "footer", "page_number", "number",
    "pageheader", "pagefooter",
})
FIGURE_TYPES = frozenset({"figure", "table", "image", "picture"})
TEXT_TYPES = frozenset({
    "text", "title", "reference", "equation",
    "figure_caption", "table_caption",
    "caption", "sectionheader", "listitem", "footnote",
    "tableofcontents", "form", "code",
})

# Minimum confidence score to accept a detected region
MIN_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine = None
_engine_init_attempted = False


def _get_engine():
    """
    Lazily initialise the layout analysis engine.

    Priority:
    1. Surya LayoutPredictor              — most accurate
    2. paddlex layout_parsing + PP-DocLayout-M  — PaddleOCR v3+
    3. paddleocr.PPStructure              — PaddleOCR v2 fallback
    """
    global _engine, _engine_init_attempted
    if _engine_init_attempted:
        return _engine
    _engine_init_attempted = True

    # --- Option 1: Surya ---
    try:
        from surya.layout import LayoutPredictor  # noqa: PLC0415
        predictor = LayoutPredictor()
        _engine = ("surya", predictor)
        logger.info("Layout analysis: Surya LayoutPredictor ready")
        return _engine
    except Exception as exc:
        logger.debug("Surya unavailable: %s", exc)

    # --- Option 2: paddlex with PP-DocLayout-M ---
    try:
        from paddlex import create_pipeline  # noqa: PLC0415

        pipeline = None
        try:
            from paddlex.inference.pipelines import load_pipeline_config
            config = load_pipeline_config("layout_parsing")
            try:
                config["SubModules"]["LayoutDetection"]["model_name"] = "PP-DocLayout-M"
            except (KeyError, TypeError):
                pass
            pipeline = create_pipeline(config=config)
            logger.info("Layout analysis: paddlex layout_parsing + PP-DocLayout-M ready")
        except Exception as exc:
            logger.debug("load_pipeline_config failed (%s)", exc)

        if pipeline is None:
            pipeline = create_pipeline(pipeline="layout_parsing")
            logger.info("Layout analysis: paddlex layout_parsing pipeline ready (default)")

        _engine = ("paddlex", pipeline)
        return _engine

    except Exception as exc:
        logger.debug("paddlex layout_parsing unavailable: %s", exc)

    # --- Option 3: PPStructure (PaddleOCR v2) ---
    try:
        from paddleocr import PPStructure  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "Layout analysis disabled: no engine available. "
            "Entire page treated as one text region."
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

def _normalise_label(label: str) -> str:
    """Normalise Surya PascalCase labels to lowercase for unified classification."""
    return label.lower().replace("-", "").replace("_", "").strip()


def _parse_surya_result(results, page_number: int) -> List[LayoutRegion]:
    """Parse Surya LayoutPredictor output into LayoutRegion list."""
    regions: List[LayoutRegion] = []
    if not results:
        return regions

    for box in results[0].bboxes:
        label = _normalise_label(box.label)
        score = box.confidence if box.confidence is not None else 0.0
        bbox = box.bbox  # [x_min, y_min, x_max, y_max]
        regions.append(LayoutRegion(
            region_type=label,
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            confidence=float(score),
            page_number=page_number,
        ))

    return regions


def _parse_paddlex_result(result, page_number: int) -> List[LayoutRegion]:
    """
    Parse paddlex layout_parsing output.

    PaddleX v3 pipeline.predict() yields items. Each item is typically a dict
    with structure:
      item['layout_det_res']['boxes'] = [
          {'cls_id': int, 'label': str, 'score': float,
           'coordinate': [x0, y0, x1, y1]},
          ...
      ]

    Older paddlex formats may use top-level 'boxes'/'labels'/'scores' keys.
    """
    regions: List[LayoutRegion] = []
    # result may be a generator — materialise it
    items = list(result) if not isinstance(result, list) else result

    for item in items:
        if item is None:
            continue

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # --- PaddleX v3: item['layout_det_res']['boxes'] ---
        det_res = _get(item, "layout_det_res")
        if det_res is not None:
            boxes = _get(det_res, "boxes", [])
            for box_info in (boxes or []):
                if not isinstance(box_info, dict):
                    continue
                label = str(box_info.get("label", "text")).lower().strip()
                score = float(box_info.get("score", 0.0))
                coord = box_info.get("coordinate", [])
                try:
                    x0 = float(coord[0])
                    y0 = float(coord[1])
                    x1 = float(coord[2])
                    y1 = float(coord[3])
                except (IndexError, TypeError, ValueError):
                    continue
                regions.append(LayoutRegion(
                    region_type=label,
                    bbox=(x0, y0, x1, y1),
                    confidence=score,
                    page_number=page_number,
                ))
            continue

        # --- Older paddlex: top-level boxes/labels/scores ---
        boxes  = _get(item, "boxes",  [])
        labels = _get(item, "labels", [])
        scores = _get(item, "scores", [])

        for i, (box, label) in enumerate(zip(boxes or [], labels or [])):
            score = float(scores[i]) if scores and i < len(scores) else 0.0
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
        if engine_type == "surya":
            results = engine([image])
            regions = _parse_surya_result(results, page_number)
        elif engine_type == "paddlex":
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

        # Filter out low-confidence detections
        regions = [r for r in regions if r.confidence >= MIN_CONFIDENCE]

        if not regions:
            return _full_page_fallback(image, page_number)

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    except Exception as exc:
        logger.warning("Layout analysis failed for page %d: %s", page_number, exc)
        return _full_page_fallback(image, page_number)


def _is_positional_header_footer(
    region: LayoutRegion,
    page_height: float,
) -> bool:
    """
    Detect likely header/footer by position heuristic.
    A short region in the top 8% or bottom 10% of the page is likely
    a running header/footer even if the model labels it as text.
    """
    if page_height <= 0:
        return False
    _, y0, _, y1 = region.bbox
    region_height = y1 - y0
    # Must be a short region (< 3% of page height)
    if region_height > page_height * 0.03:
        return False
    # Top 8% or bottom 10%
    if y0 < page_height * 0.08 or y1 > page_height * 0.90:
        return True
    return False


def filter_regions(
    regions: List[LayoutRegion],
    page_height: float = 0.0,
) -> Tuple[List[LayoutRegion], List[LayoutRegion], List[LayoutRegion]]:
    """
    Classify regions into three groups.

    Args:
        regions: Layout regions from analyze_layout().
        page_height: Rendered page height in pixels (for positional filtering).

    Returns:
        (text_regions, figure_regions, discarded_regions)
    """
    text_regions: List[LayoutRegion] = []
    figure_regions: List[LayoutRegion] = []
    discarded: List[LayoutRegion] = []

    for r in regions:
        if r.region_type in DISCARD_TYPES:
            discarded.append(r)
        elif page_height > 0 and _is_positional_header_footer(r, page_height):
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
