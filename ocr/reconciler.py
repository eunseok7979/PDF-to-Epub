"""
ocr/reconciler.py
-----------------
Character-level 3-way text reconciliation.

Given three text versions of a region, the reconciler:
1. Aligns B and C to A using difflib.SequenceMatcher (character level).
2. Uses A as the positional anchor.
3. At each character position: majority vote among A, B, C.
4. Handles insertions (chars in B/C not in A): included only if both
   B and C agree on the insertion.

Voting rules:
  - 2+ of 3 agree  -> accepted character
  - All 3 differ    -> keep A, log the position
  - Any candidate entirely empty -> system error, logged, proceed with
    remaining candidates
"""

from __future__ import annotations

import difflib
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Character-level alignment
# ---------------------------------------------------------------------------

def _map_candidate_to_anchor(
    anchor: str, candidate: str,
) -> Tuple[Dict[int, Optional[str]], Dict[int, List[str]]]:
    """
    Align *candidate* to *anchor* character-by-character.

    Returns:
        char_map:   {anchor_index: candidate_char_or_None}
                    None means the candidate deleted that character.
        insertions: {anchor_index: [chars inserted AFTER this position]}
                    Use key -1 for chars inserted BEFORE position 0.
    """
    sm = difflib.SequenceMatcher(None, anchor, candidate, autojunk=False)
    char_map: Dict[int, Optional[str]] = {}
    insertions: Dict[int, List[str]] = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_len = i2 - i1
        c_len = j2 - j1

        if tag == "equal":
            for k in range(a_len):
                char_map[i1 + k] = candidate[j1 + k]

        elif tag == "replace":
            min_len = min(a_len, c_len)
            for k in range(min_len):
                char_map[i1 + k] = candidate[j1 + k]
            # Extra anchor chars -> deletion in candidate
            for k in range(min_len, a_len):
                char_map[i1 + k] = None
            # Extra candidate chars -> insertion
            if c_len > a_len:
                insert_after = i1 + a_len - 1 if a_len > 0 else i1 - 1
                insertions.setdefault(insert_after, []).extend(
                    list(candidate[j1 + a_len : j2])
                )

        elif tag == "delete":
            for k in range(a_len):
                char_map[i1 + k] = None

        elif tag == "insert":
            insert_after = i1 - 1
            insertions.setdefault(insert_after, []).extend(
                list(candidate[j1:j2])
            )

    return char_map, insertions


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _vote_char(
    a: str, b: Optional[str], c: Optional[str],
) -> Optional[str]:
    """
    Majority vote for one character position.

    *a* is always non-None (comes from the anchor string).
    *b* or *c* can be None (meaning that candidate deleted this position).
    Returns None only when b and c both vote for deletion (2:1 against a).
    """
    if a == b:
        return a
    if a == c:
        return a
    if b == c:
        # b and c agree (could be a different char, or both None = deletion)
        return b
    # All three differ -> fallback to anchor
    return a


# ---------------------------------------------------------------------------
# Three-candidate reconciliation
# ---------------------------------------------------------------------------

def _reconcile_three(
    text_a: str, text_b: str, text_c: str,
    page_number: int = -1,
    region_label: str = "",
) -> str:
    """
    Full character-level 3-way reconciliation with A as anchor.
    """
    b_map, b_ins = _map_candidate_to_anchor(text_a, text_b)
    c_map, c_ins = _map_candidate_to_anchor(text_a, text_c)

    result: List[str] = []

    # Pre-insertions (before position 0)
    b_pre = b_ins.get(-1, [])
    c_pre = c_ins.get(-1, [])
    if b_pre and b_pre == c_pre:
        result.extend(b_pre)

    for i in range(len(text_a)):
        a_ch = text_a[i]
        b_ch = b_map.get(i)   # None = B deleted this
        c_ch = c_map.get(i)   # None = C deleted this

        chosen = _vote_char(a_ch, b_ch, c_ch)

        if chosen is not None:
            result.append(chosen)
        # else: B and C both voted for deletion -> character removed

        # Post-insertions after position i
        b_after = b_ins.get(i, [])
        c_after = c_ins.get(i, [])
        if b_after and b_after == c_after:
            result.extend(b_after)

    return "".join(result)


# ---------------------------------------------------------------------------
# Two-candidate reconciliation (fallback when one engine failed)
# ---------------------------------------------------------------------------

def _reconcile_two(text_a: str, text_b: str) -> str:
    """
    Simple pairwise reconciliation: where they agree keep the char,
    where they differ keep A (the more trusted candidate).
    """
    sm = difflib.SequenceMatcher(None, text_a, text_b, autojunk=False)
    result: List[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            result.append(text_a[i1:i2])
        elif tag == "replace":
            result.append(text_a[i1:i2])
        elif tag == "delete":
            result.append(text_a[i1:i2])
        elif tag == "insert":
            pass  # not in A -> skip

    return "".join(result)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def reconcile_text(
    text_a: str,
    text_b: str,
    text_c: str,
    page_number: int = -1,
    region_label: str = "",
) -> str:
    """
    Reconcile up to three text candidates for a single text region.

    Handles degenerate cases where one or more candidates are empty
    (system error / engine crash) by falling back to fewer candidates.

    Args:
        text_a: Primary candidate (native text, or Tesseract for pure scans).
        text_b: Secondary candidate (Tesseract, or EasyOCR for pure scans).
        text_c: Tertiary candidate (EasyOCR, or PaddleOCR for pure scans).
        page_number: For logging.
        region_label: For logging (e.g. "region 3").

    Returns:
        Reconciled text string.
    """
    has_a = bool(text_a)
    has_b = bool(text_b)
    has_c = bool(text_c)
    available = sum([has_a, has_b, has_c])

    if available == 0:
        return ""

    if available < 3:
        logger.error(
            "Page %d %s: only %d/3 OCR candidates available (system error). "
            "A=%d chars, B=%d chars, C=%d chars",
            page_number, region_label, available,
            len(text_a), len(text_b), len(text_c),
        )

    if available == 1:
        return text_a or text_b or text_c

    if available == 2:
        pair = [t for t in [text_a, text_b, text_c] if t]
        return _reconcile_two(pair[0], pair[1])

    # All three present -> full character-level reconciliation
    return _reconcile_three(text_a, text_b, text_c, page_number, region_label)


def reconcile_page(
    page_number: int,
    is_scanned: bool,
    native_text: str,
    tesseract_text: str,
    easyocr_text: str,
    paddleocr_text: str = "",
) -> str:
    """
    Select the correct candidates based on page type and reconcile.

    Born-digital / scanned-with-embedded-text (is_scanned=False):
        A=native_text, B=tesseract_text, C=easyocr_text

    Pure scan (is_scanned=True):
        A=tesseract_text, B=easyocr_text, C=paddleocr_text
    """
    if is_scanned:
        return reconcile_text(
            tesseract_text, easyocr_text, paddleocr_text,
            page_number=page_number,
        )
    else:
        return reconcile_text(
            native_text, tesseract_text, easyocr_text,
            page_number=page_number,
        )
