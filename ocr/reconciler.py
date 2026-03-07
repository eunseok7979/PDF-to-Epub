"""
ocr/reconciler.py
-----------------
Majority-vote text reconciliation for three OCR candidates.

Given three text versions of a page, the reconciler:
1. Splits each version into sentences.
2. Aligns the sentence lists pairwise using difflib SequenceMatcher.
3. Applies a majority-vote rule: if ≥ 2 of 3 candidates agree on a sentence,
   that sentence is accepted as correct.
4. When all three differ, candidate[0] (the most trusted source) is kept and
   the discrepancy is logged.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Sentence-ending punctuation shared by Korean, English, and CJK scripts
_SENT_END = re.compile(
    r'(?<=[.!?\u3002\uff01\uff1f])\s+|(?<=[.!?\u3002\uff01\uff1f])$',
    re.MULTILINE,
)

# Korean sentence-ending verbal endings (simplified heuristic)
_KO_SENT_END = re.compile(
    r'(?<=\ub2c8\ub2e4)\s|(?<=\uc2b5\ub2c8\ub2e4)\s|(?<=\uc694)\s|(?<=\uc2ed\ub2c8\ub2e4)\s',
)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences across Korean, English, and CJK scripts.

    Strategy:
    - Split on universal punctuation (.  !  ?  。 ！ ？) followed by whitespace
      or end-of-line.
    - Additionally handle Korean verbal endings that often lack explicit
      punctuation (best-effort).
    - Filter out empty fragments.
    """
    if not text.strip():
        return []

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Primary split on sentence-ending punctuation
    parts = _SENT_END.split(text)

    sentences: List[str] = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Sentence alignment
# ---------------------------------------------------------------------------

def _align_two(a: List[str], b: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align two sentence lists using SequenceMatcher.
    Returns a list of (sentence_from_a, sentence_from_b) pairs where
    None indicates a gap (insertion/deletion).
    """
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs: List[Tuple[Optional[str], Optional[str]]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for sa, sb in zip(a[i1:i2], b[j1:j2]):
                pairs.append((sa, sb))
        elif tag == "replace":
            # Pair up what we can, None-pad the shorter side
            for sa, sb in zip(a[i1:i2], b[j1:j2]):
                pairs.append((sa, sb))
            for sa in a[i1 + (j2 - j1):i2]:
                pairs.append((sa, None))
            for sb in b[j1 + (i2 - i1):j2]:
                pairs.append((None, sb))
        elif tag == "delete":
            for sa in a[i1:i2]:
                pairs.append((sa, None))
        elif tag == "insert":
            for sb in b[j1:j2]:
                pairs.append((None, sb))

    return pairs


def _sentences_match(s1: Optional[str], s2: Optional[str]) -> bool:
    """Return True if two sentences are considered equal (normalised)."""
    if s1 is None or s2 is None:
        return False
    # Normalise whitespace for comparison
    return s1.split() == s2.split()


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def reconcile(
    candidate_a: str,
    candidate_b: str,
    candidate_c: str,
    page_number: int = -1,
) -> str:
    """
    Reconcile three text candidates using majority vote at the sentence level.

    Algorithm:
    - Align (a, b) → get aligned pairs.
    - For each aligned pair, also look up the corresponding sentence in c.
    - If a == b → accept a (2/3 agree).
    - Else if a == c → accept a (2/3 agree).
    - Else if b == c → accept b (2/3 agree).
    - Else → accept a (most trusted: native text for born-digital, or
      Tesseract for scanned), log the discrepancy.

    Args:
        candidate_a: First candidate (native text or Tesseract).
        candidate_b: Second candidate (Tesseract or EasyOCR).
        candidate_c: Third candidate (EasyOCR or PaddleOCR).
        page_number: Used only for logging.

    Returns:
        Reconciled plain text with sentences joined by newlines.
    """
    sents_a = split_sentences(candidate_a)
    sents_b = split_sentences(candidate_b)
    sents_c = split_sentences(candidate_c)

    # Handle degenerate cases
    if not sents_a and not sents_b and not sents_c:
        return ""
    if not sents_a:
        sents_a = []
    if not sents_b:
        sents_b = []
    if not sents_c:
        sents_c = []

    # Align a and b
    ab_pairs = _align_two(sents_a, sents_b)

    # Consume c sequentially (rough alignment — c has no guarantee of matching length)
    c_iter = iter(sents_c)

    accepted: List[str] = []

    for sa, sb in ab_pairs:
        sc: Optional[str] = next(c_iter, None)

        if _sentences_match(sa, sb):
            # a and b agree → accept
            accepted.append(sa)  # type: ignore[arg-type]
        elif _sentences_match(sa, sc):
            # a and c agree → accept a
            accepted.append(sa)  # type: ignore[arg-type]
        elif _sentences_match(sb, sc):
            # b and c agree → accept b
            accepted.append(sb)  # type: ignore[arg-type]
        else:
            # All three differ → fall back to a (most trusted)
            chosen = sa or sb or sc
            if chosen:
                logger.warning(
                    "Page %d: All three OCR candidates differ. "
                    "Falling back to candidate A.\n  A: %r\n  B: %r\n  C: %r",
                    page_number, sa, sb, sc,
                )
                accepted.append(chosen)

    # Append any remaining c sentences not consumed above
    for sc in c_iter:
        if sc:
            accepted.append(sc)

    return "\n".join(s for s in accepted if s)


# ---------------------------------------------------------------------------
# Page-level entry point
# ---------------------------------------------------------------------------

def reconcile_page(
    page_number: int,
    is_scanned: bool,
    native_text: str,
    tesseract_text: str,
    easyocr_text: str,
    paddleocr_text: str = "",
) -> str:
    """
    Select the correct three candidates based on page type and run reconciliation.

    Born-digital or scanned-with-embedded-text (is_scanned=False):
        three-way vote: (native_text, tesseract_text, easyocr_text)

    Pure scan — no selectable text layer (is_scanned=True):
        three-way vote: (tesseract_text, easyocr_text, paddleocr_text)

    A scanned page with an embedded OCR text layer has enough characters to
    pass the born-digital threshold in pdf_extractor, so it is classified as
    is_scanned=False and handled identically to a born-digital page.
    """
    if is_scanned:
        return reconcile(
            tesseract_text,
            easyocr_text,
            paddleocr_text,
            page_number=page_number,
        )
    else:
        return reconcile(
            native_text,
            tesseract_text,
            easyocr_text,
            page_number=page_number,
        )
