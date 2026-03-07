"""
structure_parser.py
-------------------
Analyses the raw text blocks (from pdf_extractor.py) and the reconciled
plain text (from ocr/reconciler.py) to produce a structured document model.

Detects:
  - Headings  (by font size / bold heuristics)
  - Footnotes (small font at the bottom of a page, prefixed by a note marker)
  - Endnotes  (numbered list near the end of the document under a keyword heading)
  - Body paragraphs
  - Image placeholders (by page/position)

Output: a flat list of Block objects in reading order.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Tuple

from pdf_extractor import PageData, TextBlock, get_dominant_font_size


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------

BLOCK_TYPES = {"heading", "paragraph", "footnote", "endnote", "image", "caption"}


@dataclasses.dataclass
class Block:
    block_type: str          # one of BLOCK_TYPES
    text: str                # plain text content (empty for image blocks)
    level: int = 0           # heading level 1-6; 0 for non-headings
    note_id: Optional[str] = None   # e.g. "fn-1", "en-3"
    image_id: Optional[str] = None  # for image blocks
    page_number: int = 0


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

# A text block is considered a heading if its font size is at least this much
# larger than the dominant body font size.
_HEADING_SIZE_RATIO = 1.25

# A text block is a footnote candidate if it appears in the bottom N% of the page
_FOOTNOTE_BOTTOM_FRACTION = 0.20

# Footnote / endnote marker patterns (Korean circle numbers, Arabic, Roman, asterisk, etc.)
_NOTE_MARKER_RE = re.compile(
    r'^(?:'
    r'[\u2460-\u2473\u3251-\u32BF]'   # circled numbers ①–⑳ etc.
    r'|\*+'                              # asterisks
    r'|\d{1,3}[.\)]\s'                  # "1. " or "1) "
    r'|[ivxlcdmIVXLCDM]{1,6}[.\)]\s'   # Roman numerals
    r')',
    re.UNICODE,
)

# Endnote section heading keywords (Korean and English)
_ENDNOTE_HEADING_RE = re.compile(
    r'^(notes?|endnotes?|미주|후주|각주|참고\s*문헌|주석)\s*$',
    re.IGNORECASE | re.UNICODE,
)

# Reference marker inline in body text (superscript-style markers)
_INLINE_MARKER_RE = re.compile(
    r'(?:'
    r'[\u2460-\u2473\u3251-\u32BF]'    # circled numbers
    r'|\[(\d{1,3})\]'                    # [1]
    r'|(?<!\d)(\d{1,3})(?=[,\s\)\]])'   # bare superscript-style number
    r')',
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_heading(block: TextBlock, body_font_size: float) -> Tuple[bool, int]:
    """
    Return (is_heading, level).
    Level 1 = largest headings, level 3 = smallest detected headings.
    """
    ratio = block.font_size / body_font_size if body_font_size else 1.0
    if ratio >= _HEADING_SIZE_RATIO or (block.is_bold and ratio >= 1.05):
        if ratio >= 2.0:
            return True, 1
        elif ratio >= 1.5:
            return True, 2
        else:
            return True, 3
    return False, 0


def _is_footnote_candidate(block: TextBlock) -> bool:
    """
    Return True if this block looks like a footnote:
    - Appears in the bottom portion of the page
    - Starts with a note marker
    """
    relative_y = block.y0 / block.page_height if block.page_height else 0.5
    in_bottom = relative_y >= (1.0 - _FOOTNOTE_BOTTOM_FRACTION)
    has_marker = bool(_NOTE_MARKER_RE.match(block.text.strip()))
    return in_bottom and has_marker


def _extract_note_id(text: str, prefix: str) -> Tuple[str, str]:
    """
    Extract the marker from the start of a note's text.
    Returns (note_id, text_without_marker).
    """
    m = _NOTE_MARKER_RE.match(text.strip())
    if m:
        marker = text[:m.end()].strip().rstrip(".")
        rest = text[m.end():].strip()
        note_id = f"{prefix}-{marker}"
        return note_id, rest
    return f"{prefix}-?", text


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_structure(
    pages: List[PageData],
    reconciled_texts: Dict[int, str],
) -> List[Block]:
    """
    Build a flat list of Block objects from page data and reconciled text.

    For born-digital pages the TextBlock metadata (font size, position) drives
    heading/footnote detection. For scanned pages we apply lightweight regex
    heuristics to the reconciled plain text.

    Args:
        pages: Output of pdf_extractor.extract_pdf().
        reconciled_texts: Dict mapping page_number → reconciled plain text
                          (output of reconciler.reconcile_page()).

    Returns:
        Ordered list of Block objects representing the full document structure.
    """
    body_font_size = get_dominant_font_size(pages)
    all_blocks: List[Block] = []

    footnote_counter = 1
    endnote_counter = 1
    in_endnote_section = False

    for page in pages:
        page_blocks: List[Block] = []

        if page.is_scanned:
            # ------------------------------------------------------------------
            # Scanned page: use reconciled text, apply text-only heuristics
            # ------------------------------------------------------------------
            text = reconciled_texts.get(page.page_number, "")
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            for line in lines:
                # Check for endnote section heading
                if _ENDNOTE_HEADING_RE.match(line):
                    in_endnote_section = True
                    page_blocks.append(Block(
                        block_type="heading",
                        text=line,
                        level=2,
                        page_number=page.page_number,
                    ))
                    continue

                if in_endnote_section and _NOTE_MARKER_RE.match(line):
                    note_id, note_text = _extract_note_id(line, "en")
                    page_blocks.append(Block(
                        block_type="endnote",
                        text=note_text,
                        note_id=f"en-{endnote_counter}",
                        page_number=page.page_number,
                    ))
                    endnote_counter += 1
                else:
                    page_blocks.append(Block(
                        block_type="paragraph",
                        text=line,
                        page_number=page.page_number,
                    ))

        else:
            # ------------------------------------------------------------------
            # Born-digital page: use TextBlock metadata for rich detection
            # ------------------------------------------------------------------
            footnotes_this_page: List[Block] = []

            for tb in page.text_blocks:
                text = tb.text.strip()
                if not text:
                    continue

                # Check for endnote section heading
                if _ENDNOTE_HEADING_RE.match(text):
                    in_endnote_section = True
                    page_blocks.append(Block(
                        block_type="heading",
                        text=text,
                        level=2,
                        page_number=page.page_number,
                    ))
                    continue

                is_h, level = _is_heading(tb, body_font_size)

                if is_h:
                    page_blocks.append(Block(
                        block_type="heading",
                        text=text,
                        level=level,
                        page_number=page.page_number,
                    ))
                elif _is_footnote_candidate(tb):
                    note_id_str, note_text = _extract_note_id(text, "fn")
                    real_id = f"fn-{footnote_counter}"
                    footnotes_this_page.append(Block(
                        block_type="footnote",
                        text=note_text,
                        note_id=real_id,
                        page_number=page.page_number,
                    ))
                    footnote_counter += 1
                elif in_endnote_section and _NOTE_MARKER_RE.match(text):
                    real_id = f"en-{endnote_counter}"
                    _, note_text = _extract_note_id(text, "en")
                    page_blocks.append(Block(
                        block_type="endnote",
                        text=note_text,
                        note_id=real_id,
                        page_number=page.page_number,
                    ))
                    endnote_counter += 1
                else:
                    page_blocks.append(Block(
                        block_type="paragraph",
                        text=text,
                        page_number=page.page_number,
                    ))

            # Footnotes go after body paragraphs of this page (before next page)
            page_blocks.extend(footnotes_this_page)

        all_blocks.extend(page_blocks)

    return all_blocks


def inject_image_placeholders(
    blocks: List[Block],
    page_image_map: Dict[int, list],  # page_number → List[ExtractedImage]
) -> List[Block]:
    """
    Insert image placeholder Blocks into the block list at the correct page
    positions.

    Images are inserted after the last paragraph block on the same page,
    before the first block of the next page.

    Args:
        blocks: Output of parse_structure().
        page_image_map: Output of image_handler.build_page_image_map().

    Returns:
        New block list with image placeholders inserted.
    """
    if not page_image_map:
        return blocks

    result: List[Block] = []
    pages_with_images = set(page_image_map.keys())
    last_seen_page = -1
    images_inserted: set = set()

    for block in blocks:
        # Before appending this block, check if we've moved to a new page
        # and there are images from the previous page not yet inserted
        if block.page_number != last_seen_page:
            if last_seen_page in pages_with_images and last_seen_page not in images_inserted:
                for img in page_image_map[last_seen_page]:
                    result.append(Block(
                        block_type="image",
                        text="",
                        image_id=img.image_id,
                        page_number=last_seen_page,
                    ))
                images_inserted.add(last_seen_page)
            last_seen_page = block.page_number

        result.append(block)

    # Handle images on the last page
    if last_seen_page in pages_with_images and last_seen_page not in images_inserted:
        for img in page_image_map[last_seen_page]:
            result.append(Block(
                block_type="image",
                text="",
                image_id=img.image_id,
                page_number=last_seen_page,
            ))

    return result
