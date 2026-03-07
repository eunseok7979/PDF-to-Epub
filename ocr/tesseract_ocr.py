"""
ocr/tesseract_ocr.py
--------------------
Tesseract OCR wrapper for Korean, English, and Traditional Chinese.
Returns plain text extracted from a PIL Image.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image
import pytesseract


# Tesseract language string for Korean + English + Traditional Chinese
TESSERACT_LANG = "kor+eng+chi_tra"

# Tesseract page segmentation mode:
# 3 = Fully automatic page segmentation, but no OSD (default)
_PSM = 3


def run(image: Image.Image, lang: Optional[str] = None) -> str:
    """
    Run Tesseract OCR on a PIL Image and return the recognised text.

    Args:
        image: PIL Image (RGB or grayscale) of a rendered PDF page.
        lang: Tesseract language string. Defaults to TESSERACT_LANG.

    Returns:
        Recognised text as a plain string. Empty string on failure.
    """
    lang = lang or TESSERACT_LANG
    config = f"--psm {_PSM} --oem 1"  # oem 1 = LSTM engine only

    try:
        text: str = pytesseract.image_to_string(
            image,
            lang=lang,
            config=config,
        )
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found. "
            "Install it via your system package manager "
            "(e.g. 'apt install tesseract-ocr tesseract-ocr-kor tesseract-ocr-chi-tra')."
        )
    except Exception as exc:
        # Log but don't crash — reconciler treats empty string as 'no result'
        import warnings
        warnings.warn(f"Tesseract OCR failed: {exc}")
        return ""
