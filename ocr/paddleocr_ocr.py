"""
ocr/paddleocr_ocr.py
--------------------
PaddleOCR wrapper for Korean and English.
Handles both old API (.ocr()) and new API (.predict()).
Only invoked for pure-scan pages (no embedded text layer) where a third
OCR candidate is needed for the majority vote.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# PaddleOCR language codes (chinese_cht removed for compatibility)
_PADDLE_LANGS = ["korean", "en"]

# Singleton readers keyed by language string
_readers: dict = {}


def _get_reader(lang: str):
    """Lazily initialise and cache a PaddleOCR reader for the given language."""
    if lang not in _readers:
        try:
            from paddleocr import PaddleOCR

            # Try init without show_log first (newer API), fall back with it
            init_kwargs_attempts = [
                {"use_angle_cls": True, "lang": lang},
                {"lang": lang},
            ]

            for kwargs in init_kwargs_attempts:
                try:
                    _readers[lang] = PaddleOCR(**kwargs)
                    break
                except TypeError:
                    continue
            else:
                _readers[lang] = None

        except Exception as exc:
            warnings.warn(f"PaddleOCR init failed for lang='{lang}': {exc}")
            _readers[lang] = None

    return _readers[lang]


def _run_reader(reader, img_array: np.ndarray) -> list:
    """
    Call the reader using the correct API method.
    Returns the raw result list, or empty list on failure.
    """
    # Try .predict() first (newer PaddleOCR), then .ocr() (older)
    for method_name, kwargs in [("predict", {}), ("ocr", {"cls": True})]:
        method = getattr(reader, method_name, None)
        if method is None:
            continue
        try:
            result = method(img_array, **kwargs)
            if result is not None:
                return result
        except TypeError:
            # Wrong kwargs — try without kwargs
            try:
                result = method(img_array)
                if result is not None:
                    return result
            except Exception:
                continue
        except Exception:
            continue

    return []


def _extract_lines(result) -> List[Tuple[float, str]]:
    """
    Parse PaddleOCR result into (y_centre, text) tuples.
    Handles both old and new result formats.
    """
    items: List[Tuple[float, str]] = []

    if not result:
        return items

    # Old format: list of page results, each page is list of [bbox, (text, conf)]
    if isinstance(result, list):
        for page_or_line in result:
            if page_or_line is None:
                continue

            # Could be a page result (list of lines) or a single line
            if isinstance(page_or_line, list):
                for entry in page_or_line:
                    if entry is None:
                        continue

                    if isinstance(entry, list) and len(entry) == 2:
                        bbox, text_conf = entry
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                            text, _conf = text_conf
                        elif isinstance(text_conf, str):
                            text = text_conf
                        else:
                            continue

                        try:
                            ys = [pt[1] for pt in bbox]
                            y_centre = sum(ys) / len(ys)
                        except (TypeError, IndexError):
                            y_centre = 0.0

                        if str(text).strip():
                            items.append((y_centre, str(text).strip()))

    # New format: might be a dict or object with attributes
    elif hasattr(result, "rec_texts"):
        texts = getattr(result, "rec_texts", [])
        boxes = getattr(result, "det_boxes", [])
        for i, text in enumerate(texts):
            y_centre = 0.0
            if i < len(boxes):
                try:
                    ys = [pt[1] for pt in boxes[i]]
                    y_centre = sum(ys) / len(ys)
                except (TypeError, IndexError):
                    pass
            if str(text).strip():
                items.append((y_centre, str(text).strip()))

    return items


def _ocr_with_lang(img_array: np.ndarray, lang: str) -> List[Tuple[float, str]]:
    """Run PaddleOCR for a single language. Returns (y_centre, text) tuples."""
    reader = _get_reader(lang)
    if reader is None:
        return []
    try:
        result = _run_reader(reader, img_array)
        return _extract_lines(result)
    except Exception as exc:
        warnings.warn(f"PaddleOCR run failed for lang='{lang}': {exc}")
        return []


def run(image: Image.Image, langs: Optional[List[str]] = None) -> str:
    """
    Run PaddleOCR on a PIL Image across configured languages and return
    the recognised text sorted by vertical reading order.

    Deduplicates overlapping detections by y-position proximity (10 px).

    Args:
        image: PIL Image (RGB) of a page or cropped text region.
        langs: Override the default language list.

    Returns:
        Recognised text joined by newlines. Empty string on failure.
    """
    langs = langs or _PADDLE_LANGS
    img_array = np.array(image)

    all_items: List[Tuple[float, str]] = []
    for lang in langs:
        all_items.extend(_ocr_with_lang(img_array, lang))

    if not all_items:
        return ""

    # Sort by y-centre
    all_items.sort(key=lambda x: x[0])

    # Deduplicate: same text within 10 px vertical distance
    deduped: List[Tuple[float, str]] = []
    for y, text in all_items:
        duplicate = False
        for dy, dt in deduped:
            if abs(y - dy) < 10 and text == dt:
                duplicate = True
                break
        if not duplicate:
            deduped.append((y, text))

    return "\n".join(text for _, text in deduped)
