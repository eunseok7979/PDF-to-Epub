"""
notes_linker.py
---------------
Transforms footnote and endnote Block objects into EPUB3-compliant HTML.

Footnotes → inline <aside epub:type="footnote"> elements placed immediately
            after the paragraph that references them. Reading systems that
            support EPUB3 will render these as tap-to-reveal popups.

Endnotes  → <aside epub:type="footnote"> elements grouped at the end of the
            document under their section heading, with bidirectional anchor
            links back to the reference in the body text.

The linker works in two passes:
  Pass 1: Scan all blocks to build a map of note_id → Block.
  Pass 2: Walk body paragraphs, detect inline reference markers, replace them
          with <a epub:type="noteref"> anchors, and attach the corresponding
          <aside> element.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Tuple

from structure_parser import Block


# ---------------------------------------------------------------------------
# Reference marker patterns (same as structure_parser but used for substitution)
# ---------------------------------------------------------------------------

# Matches inline note markers inside body text
_INLINE_REF_RE = re.compile(
    r'('
    r'[\u2460-\u2473\u3251-\u32BF]'       # ①–⑳ circled numbers
    r'|\[\d{1,3}\]'                          # [1]
    r'|\(\d{1,3}\)'                          # (1)
    r'|\d{1,3}(?=[,\s\)\]]|\Z)'             # bare superscript-like number
    r')',
    re.UNICODE,
)

# Matches the marker at the START of a note's own text (to extract the key)
_NOTE_START_RE = re.compile(
    r'^(?P<marker>'
    r'[\u2460-\u2473\u3251-\u32BF]'
    r'|\*+'
    r'|\d{1,3}[.\)]\s?'
    r'|[ivxlcdmIVXLCDM]{1,6}[.\)]\s?'
    r')',
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _noteref_anchor(note_id: str, display: str) -> str:
    """<a epub:type="noteref"> for use inside body paragraphs."""
    return (
        f'<a epub:type="noteref" href="#{note_id}" id="{note_id}-ref">'
        f'{_escape(display)}</a>'
    )


def _footnote_aside(note_id: str, text: str) -> str:
    """EPUB3 inline footnote as <aside>."""
    back_link = f'<a href="#{note_id}-ref" title="Back to text">↩</a>'
    return (
        f'<aside epub:type="footnote" id="{note_id}">\n'
        f'  <p>{_escape(text)} {back_link}</p>\n'
        f'</aside>'
    )


def _endnote_aside(note_id: str, text: str) -> str:
    """EPUB3 endnote as <aside> (at end of document)."""
    back_link = f'<a href="#{note_id}-ref" title="Back to text">↩</a>'
    return (
        f'<aside epub:type="footnote" id="{note_id}">\n'
        f'  <p>{_escape(text)} {back_link}</p>\n'
        f'</aside>'
    )


# ---------------------------------------------------------------------------
# Linked HTML block
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LinkedBlock:
    """A Block that has been converted to HTML with note links resolved."""
    html: str
    block_type: str        # mirrors Block.block_type
    note_id: Optional[str] = None
    level: int = 0


# ---------------------------------------------------------------------------
# Pass 1: build note map
# ---------------------------------------------------------------------------

def _build_note_map(blocks: List[Block]) -> Dict[str, Block]:
    """Return {note_id: Block} for all footnote and endnote blocks."""
    return {
        b.note_id: b
        for b in blocks
        if b.block_type in ("footnote", "endnote") and b.note_id
    }


# ---------------------------------------------------------------------------
# Pass 2: resolve references
# ---------------------------------------------------------------------------

def _substitute_refs_in_paragraph(
    text: str,
    note_map: Dict[str, Block],
    resolved_notes: Dict[str, bool],
) -> Tuple[str, List[str]]:
    """
    Scan a paragraph's text for inline note markers, replace each with an
    <a epub:type="noteref"> anchor, and record which note_ids were referenced.

    Returns:
        (html_paragraph_text, list_of_referenced_note_ids)
    """
    referenced: List[str] = []
    result = _escape(text)

    # We iterate over note_map entries sorted by their numeric part so that
    # larger numbers don't accidentally match inside smaller ones.
    note_ids_sorted = sorted(
        note_map.keys(),
        key=lambda nid: int(re.search(r'\d+', nid).group()) if re.search(r'\d+', nid) else 0,
        reverse=True,
    )

    for note_id in note_ids_sorted:
        # Derive the expected display marker from the note_id ("fn-3" → "3")
        num_match = re.search(r'\d+', note_id)
        if not num_match:
            continue
        num = num_match.group()

        # Build patterns that could appear in the escaped text
        patterns = [
            re.escape(f"[{num}]"),
            re.escape(f"({num})"),
        ]
        # Also match the circled Unicode digit if num <= 20
        try:
            n = int(num)
            if 1 <= n <= 20:
                patterns.append(re.escape(chr(0x2460 + n - 1)))
        except ValueError:
            pass

        for pat in patterns:
            compiled = re.compile(pat)
            if compiled.search(result):
                anchor = _noteref_anchor(note_id, _unescape_marker(pat))
                result = compiled.sub(anchor, result, count=1)
                referenced.append(note_id)
                resolved_notes[note_id] = True
                break

    return result, referenced


def _unescape_marker(escaped_pat: str) -> str:
    """Convert an re.escape'd pattern back to a display string."""
    return re.sub(r'\\(.)', r'\1', escaped_pat)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def link_notes(blocks: List[Block]) -> List[LinkedBlock]:
    """
    Convert a list of Block objects to LinkedBlock HTML fragments with
    all footnote/endnote references resolved.

    Footnotes: the <aside> is inserted immediately after the paragraph
               that contains the reference marker.
    Endnotes:  the <aside> is placed in-position within the endnote section
               (they already appear at the end of the document as Block objects).

    Args:
        blocks: Ordered list of Block from structure_parser (with image placeholders).

    Returns:
        Ordered list of LinkedBlock ready for epub_builder to consume.
    """
    note_map = _build_note_map(blocks)
    resolved: Dict[str, bool] = {}  # note_id → True once placed as <aside>
    linked: List[LinkedBlock] = []

    for block in blocks:
        btype = block.block_type

        if btype == "paragraph":
            para_html, referenced_ids = _substitute_refs_in_paragraph(
                block.text, note_map, resolved
            )
            linked.append(LinkedBlock(
                html=f"<p>{para_html}</p>",
                block_type="paragraph",
            ))
            # Inject footnote asides right after this paragraph
            for note_id in referenced_ids:
                note_block = note_map.get(note_id)
                if note_block and note_block.block_type == "footnote":
                    linked.append(LinkedBlock(
                        html=_footnote_aside(note_id, note_block.text),
                        block_type="footnote",
                        note_id=note_id,
                    ))

        elif btype == "heading":
            tag = f"h{max(1, min(block.level, 6))}"
            heading_id = re.sub(r'\W+', '-', block.text.strip().lower())[:60]
            linked.append(LinkedBlock(
                html=f'<{tag} id="{_escape(heading_id)}">{_escape(block.text)}</{tag}>',
                block_type="heading",
                level=block.level,
            ))

        elif btype == "footnote":
            # Footnotes are injected after their referencing paragraph above.
            # If a footnote was never referenced in the body (orphan), place it here.
            if block.note_id and block.note_id not in resolved:
                linked.append(LinkedBlock(
                    html=_footnote_aside(block.note_id, block.text),
                    block_type="footnote",
                    note_id=block.note_id,
                ))

        elif btype == "endnote":
            # Endnotes placed in their section at the end of the document
            if block.note_id:
                linked.append(LinkedBlock(
                    html=_endnote_aside(block.note_id, block.text),
                    block_type="endnote",
                    note_id=block.note_id,
                ))

        elif btype == "image":
            # Image blocks are passed through unchanged; epub_builder handles them
            linked.append(LinkedBlock(
                html=f'__IMAGE__{block.image_id}__',
                block_type="image",
            ))

        else:
            # Fallback
            linked.append(LinkedBlock(
                html=f"<p>{_escape(block.text)}</p>",
                block_type=btype,
            ))

    return linked
