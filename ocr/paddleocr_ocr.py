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
    Call the reader using the correct API method and return a plain list.
    Materialises generators immediately so downstream code can check truthiness
    and iterate multiple times.
    """
    for method_name, kwargs in [("predict", {}), ("ocr", {"cls": True})]:
        method = getattr(reader, method_name, None)
        if method is None:
            continue
        try:
            result = method(img_array, **kwargs)
        except TypeError:
            try:
                result = method(img_array)
            except Exception:
                continue
        except Exception:
            continue

        if result is None:
            continue

        # predict() in PaddleOCR v3 returns a generator — materialise it now
        if not isinstance(result, (list, dict)):
            try:
                result = list(result)
            except Exception:
                continue

        return result

    return []


def _extract_lines(result) -> List[Tuple[float, str]]:
    """
    Parse a materialised PaddleOCR result into (y_centre, text) tuples.

    Handles two formats:
      New (PaddleOCR v3): list of dicts with 'rec_texts' and 'dt_polys'
      Old (PaddleOCR v2): list of page results, each a list of [bbox, (text, conf)]
    """
    items: List[Tuple[float, str]] = []

    if not result:
        return items

    if not isinstance(result, list):
        return items

    for page_item in result:
        if page_item is None:
            continue

        # ------------------------------------------------------------------
        # New format (PaddleOCR v3): dict with rec_texts / dt_polys
        # ------------------------------------------------------------------
        if isinstance(page_item, dict):
            rec_texts = page_item.get("rec_texts", [])
            dt_polys  = page_item.get("dt_polys",  [])

            for i, text in enumerate(rec_texts):
                text = str(text).strip()
                if not text:
                    continue
                y_centre = 0.0
                if i < len(dt_polys):
                    try:
                        poly = list(dt_polys[i])
                        ys = [float(pt[1]) for pt in poly]
                        y_centre = sum(ys) / len(ys) if ys else 0.0
                    except (TypeError, IndexError, ValueError):
                        pass
                items.append((y_centre, text))
            continue

        # ------------------------------------------------------------------
        # Old format (PaddleOCR v2): page_item is a list of lines
        # ------------------------------------------------------------------
        if isinstance(page_item, list):
            for entry in page_item:
                if entry is None:
                    continue
                if not (isinstance(entry, list) and len(entry) == 2):
                    continue
                bbox, text_conf = entry
                if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                    text = str(text_conf[0]).strip()
                elif isinstance(text_conf, str):
                    text = text_conf.strip()
                else:
                    continue
                if not text:
                    continue
                try:
                    ys = [float(pt[1]) for pt in bbox]
                    y_centre = sum(ys) / len(ys)
                except (TypeError, IndexError, ValueError):
                    y_centre = 0.0
                items.append((y_centre, text))

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
