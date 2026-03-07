"""
ocr/paddleocr_ocr.py
--------------------
PaddleOCR wrapper used as the third OCR engine for scanned pages.
Supports Korean, English, and Traditional Chinese via separate Reader instances
that are merged and sorted by vertical position.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np


# PaddleOCR language codes relevant to our use case
_PADDLE_LANGS = ["korean", "en"]

# Singleton readers keyed by language string
_readers: dict = {}


def _get_reader(lang: str):
    """Lazily initialise and cache a PaddleOCR reader for the given language."""
    if lang not in _readers:
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True handles rotated text; show_log=False suppresses verbose output
            _readers[lang] = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception as exc:
            warnings.warn(f"PaddleOCR init failed for lang='{lang}': {exc}")
            _readers[lang] = None
    return _readers[lang]


def _ocr_with_lang(
    img_array: np.ndarray, lang: str
) -> List[Tuple[float, str]]:
    """
    Run PaddleOCR for a single language and return (y_centre, text) tuples.
    Returns empty list on failure.
    """
    reader = _get_reader(lang)
    if reader is None:
        return []
    try:
        result = reader.ocr(img_array, cls=True)
        items: List[Tuple[float, str]] = []
        if not result:
            return items
        for page_result in result:
            if not page_result:
                continue
            for line in page_result:
                bbox, (text, _conf) = line
                # bbox: [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                ys = [pt[1] for pt in bbox]
                y_centre = sum(ys) / len(ys)
                if text.strip():
                    items.append((y_centre, text.strip()))
        return items
    except Exception as exc:
        warnings.warn(f"PaddleOCR run failed for lang='{lang}': {exc}")
        return []


def run(image: Image.Image, langs: Optional[List[str]] = None) -> str:
    """
    Run PaddleOCR on a PIL Image across all configured languages and return
    the recognised text sorted by vertical reading order.

    Strategy: run Korean, English, and Traditional Chinese readers separately,
    merge all detected text regions, deduplicate overlapping detections by
    y-position proximity (within 10 px), then sort top-to-bottom.

    Args:
        image: PIL Image (RGB) of a rendered PDF page.
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

    # Deduplicate: if two entries have the same text within 10 px, keep one
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
