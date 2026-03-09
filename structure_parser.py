"""
structure_parser.py
-------------------
Builds a document structure from layout analysis regions and reconciled text.

Input:  per-page lists of (LayoutRegion, reconciled_text) and figure regions.
Output: a flat list of Block objects in reading order.

Heading detection uses two sources:
  1. PP-StructureV2 "title" region type (primary).
  2. Font-size heuristic from PyMuPDF metadata (secondary, for born-digital pages).

Footnote/endnote detection uses regex heuristics on the reconciled text,
scoped to "reference"-type regions or text at the bottom of the page.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------

BLOCK_TYPES = {"heading", "paragraph", "footnote", "endnote", "image", "caption"}


@dataclasses.dataclass
class Block:
    block_type: str           # one of BLOCK_TYPES
    text: str                 # plain text (empty for image blocks)
    level: int = 0            # heading level 1-6; 0 for non-headings
    note_id: Optional[str] = None   # e.g. "fn-1", "en-3"
    image_id: Optional[str] = None  # for image blocks
    page_number: int = 0


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_HEADING_SIZE_RATIO = 1.25

# Footnote / endnote marker patterns
_NOTE_MARKER_RE = re.compile(
    r'^(?:'
    r'[\u2460-\u2473\u3251-\u32BF]'   # circled numbers
    r'|\*+'
    r'|\d{1,3}[.\)]\s'
    r'|[ivxlcdmIVXLCDM]{1,6}[.\)]\s'
    r')',
    re.UNICODE,
)

# Endnote section heading keywords
_ENDNOTE_HEADING_RE = re.compile(
    r'^(notes?|endnotes?|\uBBF8\uC8FC|\uD6C4\uC8FC|\uAC01\uC8FC'
    r'|\uCC38\uACE0\s*\uBB38\uD5CC|\uC8FC\uC11D)\s*$',
    re.IGNORECASE | re.UNICODE,
)


_SENTENCE_END = frozenset(".!?。！？")
_KO_ENDINGS = (
    "다.", "다!", "다?", "습니다.", "습니다!", "입니다.",
    "이다.", "였다.", "됩니다.", "합니다.", "했다.",
)


def _restore_paragraphs(text: str) -> str:
    """
    Merge OCR output lines into paragraphs using text-level heuristics.

    Used for scanned pages where bbox coordinates are not available.
    Lines shorter than 70% of the median line length, or ending with
    sentence-final punctuation, are treated as paragraph breaks.
    Adjacent lines that don't trigger a break are joined with a space.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return text.strip()

    lengths = sorted(len(l) for l in lines)
    median_len = lengths[len(lengths) // 2]
    short_threshold = median_len * 0.70

    paragraphs: List[str] = []
    current: List[str] = []

    for i, line in enumerate(lines):
        current.append(line)
        is_short = len(line) < short_threshold and len(lines) > 3
        ends_sentence = bool(line) and (
            line[-1] in _SENTENCE_END
            or any(line.endswith(e) for e in _KO_ENDINGS)
        )
        next_indented = (
            i + 1 < len(lines)
            and len(lines[i + 1]) > 0
            and lines[i + 1][0] in (" ", "\t", "\u3000", "\u00a0")
        )
        if is_short or ends_sentence or next_indented:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n".join(paragraphs)


def _extract_note_id(text: str, prefix: str) -> Tuple[str, str]:
    """Extract marker from the start of a note. Returns (note_id, remaining text)."""
    m = _NOTE_MARKER_RE.match(text.strip())
    if m:
        marker = text[:m.end()].strip().rstrip(".")
        rest = text[m.end():].strip()
        return f"{prefix}-{marker}", rest
    return f"{prefix}-?", text


# ---------------------------------------------------------------------------
# Region-based structure building
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RegionBlock:
    """Intermediate representation: one layout region with its reconciled text."""
    region_type: str   # from layout_analyzer: text, title, reference, figure, ...
    text: str          # reconciled text content
    page_number: int
    font_size: float = 0.0   # dominant font size from PyMuPDF (0 = unknown)
    is_bold: bool = False


def build_blocks(
    region_blocks: List[RegionBlock],
    body_font_size: float = 12.0,
) -> List[Block]:
    """
    Convert a list of RegionBlock (one per layout region, in reading order)
    into a structured list of Block objects.

    Heading detection:
      - PP-StructureV2 "title" regions -> heading (level from font size ratio).
      - Font-size heuristic for "text" regions with large/bold font.

    Footnote/endnote detection:
      - "reference" regions -> treated as footnote/endnote content.
      - Endnote section heading keyword -> switches to endnote mode.

    Image placeholders:
      - "figure"/"table" regions -> image Block with image_id set.
    """
    blocks: List[Block] = []
    footnote_counter = 1
    endnote_counter = 1
    in_endnote_section = False

    for rb in region_blocks:
        text = rb.text.strip()
        if not text and rb.region_type not in ("figure", "table"):
            continue

        # --- Figure / table -> image placeholder ---
        if rb.region_type in ("figure", "table"):
            blocks.append(Block(
                block_type="image",
                text="",
                image_id=rb.text or None,   # image_id set by caller
                page_number=rb.page_number,
            ))
            continue

        # --- Caption ---
        if rb.region_type in ("figure_caption", "table_caption"):
            blocks.append(Block(
                block_type="caption",
                text=text,
                page_number=rb.page_number,
            ))
            continue

        # --- Check for endnote section heading ---
        if _ENDNOTE_HEADING_RE.match(text):
            in_endnote_section = True
            blocks.append(Block(
                block_type="heading",
                text=text,
                level=2,
                page_number=rb.page_number,
            ))
            continue

        # --- Reference region -> footnote/endnote ---
        if rb.region_type == "reference":
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for line in lines:
                if _NOTE_MARKER_RE.match(line):
                    note_id, note_text = _extract_note_id(line, "en")
                    blocks.append(Block(
                        block_type="endnote",
                        text=note_text,
                        note_id=f"en-{endnote_counter}",
                        page_number=rb.page_number,
                    ))
                    endnote_counter += 1
                else:
                    blocks.append(Block(
                        block_type="paragraph",
                        text=line,
                        page_number=rb.page_number,
                    ))
            continue

        # --- Title region -> heading ---
        if rb.region_type == "title":
            level = _font_to_heading_level(rb.font_size, body_font_size)
            blocks.append(Block(
                block_type="heading",
                text=text,
                level=level,
                page_number=rb.page_number,
            ))
            continue

        # --- Endnote section body ---
        if in_endnote_section:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for line in lines:
                if _NOTE_MARKER_RE.match(line):
                    _, note_text = _extract_note_id(line, "en")
                    blocks.append(Block(
                        block_type="endnote",
                        text=note_text,
                        note_id=f"en-{endnote_counter}",
                        page_number=rb.page_number,
                    ))
                    endnote_counter += 1
                else:
                    blocks.append(Block(
                        block_type="paragraph",
                        text=line,
                        page_number=rb.page_number,
                    ))
            continue

        # --- Text region: check font-size heuristic for headings ---
        if rb.font_size > 0 and body_font_size > 0:
            ratio = rb.font_size / body_font_size
            if ratio >= _HEADING_SIZE_RATIO or (rb.is_bold and ratio >= 1.05):
                level = _font_to_heading_level(rb.font_size, body_font_size)
                blocks.append(Block(
                    block_type="heading",
                    text=text,
                    level=level,
                    page_number=rb.page_number,
                ))
                continue

        # --- Footnote detection or paragraph restoration ---
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        has_footnotes = any(_NOTE_MARKER_RE.match(l) for l in lines)

        if has_footnotes:
            # Process line-by-line to capture footnote markers
            for line in lines:
                if _NOTE_MARKER_RE.match(line):
                    _, note_text = _extract_note_id(line, "fn")
                    blocks.append(Block(
                        block_type="footnote",
                        text=note_text,
                        note_id=f"fn-{footnote_counter}",
                        page_number=rb.page_number,
                    ))
                    footnote_counter += 1
                else:
                    blocks.append(Block(
                        block_type="paragraph",
                        text=line,
                        page_number=rb.page_number,
                    ))
        else:
            # Restore paragraph structure from line-level OCR output
            restored = _restore_paragraphs(text)
            for para in restored.splitlines():
                para = para.strip()
                if para:
                    blocks.append(Block(
                        block_type="paragraph",
                        text=para,
                        page_number=rb.page_number,
                    ))

    return blocks


def _font_to_heading_level(font_size: float, body_font_size: float) -> int:
    """Map a font size ratio to a heading level (1-3)."""
    if body_font_size <= 0:
        return 2
    ratio = font_size / body_font_size
    if ratio >= 2.0:
        return 1
    elif ratio >= 1.5:
        return 2
    else:
        return 3
