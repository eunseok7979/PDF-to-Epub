"""
ocr/easyocr_ocr.py
------------------
EasyOCR wrapper for Korean and English.
(Chinese Traditional is incompatible with Korean in EasyOCR.)
Uses a module-level singleton Reader to avoid expensive repeated initialisation.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image
import numpy as np


# EasyOCR language codes (ko + ch_tra are incompatible)
EASYOCR_LANGS: List[str] = ["ko", "en"]

_reader = None  # singleton


def _get_reader(langs: Optional[List[str]] = None):
    """Return (and lazily initialise) the EasyOCR Reader singleton."""
    global _reader
    if _reader is None:
        import easyocr
        langs = langs or EASYOCR_LANGS
        _reader = easyocr.Reader(langs, gpu=False)
    return _reader


def run(image: Image.Image, langs: Optional[List[str]] = None) -> str:
    """
    Run EasyOCR on a PIL Image and return the recognised text.

    Text regions are sorted by their vertical centre position so that the
    output respects reading order (top-to-bottom).

    Args:
        image: PIL Image (RGB) of a page or cropped text region.
        langs: EasyOCR language list. Defaults to EASYOCR_LANGS.

    Returns:
        Recognised text joined by newlines. Empty string on failure.
    """
    try:
        reader = _get_reader(langs)
        img_array = np.array(image)

        results = reader.readtext(img_array, detail=1, paragraph=False)
        # results: list of (bbox, text, confidence)
        # bbox: [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]

        # Sort by vertical centre of bounding box
        def _vert_centre(result):
            bbox = result[0]
            ys = [pt[1] for pt in bbox]
            return sum(ys) / len(ys)

        results_sorted = sorted(results, key=_vert_centre)
        lines = [res[1] for res in results_sorted if res[1].strip()]
        return "\n".join(lines)
    except Exception as exc:
        import warnings
        warnings.warn(f"EasyOCR failed: {exc}")
        return ""
