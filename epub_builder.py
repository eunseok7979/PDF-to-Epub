"""
epub_builder.py
---------------
Assembles the final EPUB3 file from LinkedBlock HTML fragments and extracted
images.

Output structure:
  - Single XHTML content file (the entire book)
  - Embedded images as EPUB media items
  - nav.xhtml generated from heading blocks
  - Correct dc:language metadata (ko, en, zh-Hant)
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ebooklib import epub

from notes_linker import LinkedBlock
from image_handler import ExtractedImage


# ---------------------------------------------------------------------------
# EPUB3 namespace declaration needed for epub:type attributes
# ---------------------------------------------------------------------------

_XHTML_NS = 'xmlns="http://www.w3.org/1999/xhtml"'
_EPUB_NS = 'xmlns:epub="http://www.idpf.org/2007/ops"'

_CONTENT_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html {xhtml_ns} {epub_ns} xml:lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>{title}</title>
  <link rel="stylesheet" type="text/css" href="../styles/book.css"/>
</head>
<body>
{body}
</body>
</html>
"""

_CSS = """\
body {
  font-family: serif;
  font-size: 1em;
  line-height: 1.6;
  margin: 1em 2em;
  color: #111;
}
h1, h2, h3, h4, h5, h6 {
  font-weight: bold;
  margin-top: 1.4em;
  margin-bottom: 0.4em;
}
p {
  margin: 0.5em 0;
  text-indent: 1em;
}
aside[epub|type="footnote"] {
  font-size: 0.85em;
  border-left: 2px solid #aaa;
  padding-left: 0.8em;
  margin: 0.5em 0 0.5em 1em;
  color: #444;
}
a[epub|type="noteref"] {
  vertical-align: super;
  font-size: 0.75em;
  text-decoration: none;
  color: #0055aa;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1em auto;
}
"""

_NAV_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html {xhtml_ns} {epub_ns} xml:lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>Table of Contents</title>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
{toc_items}
    </ol>
  </nav>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heading_id(text: str) -> str:
    """Generate a safe HTML id from heading text."""
    slug = re.sub(r'\W+', '-', text.strip().lower())
    return slug[:60].strip('-') or "heading"


def _build_toc_items(linked_blocks: List[LinkedBlock]) -> str:
    """Generate <li> entries for the nav.xhtml TOC from heading blocks."""
    items: List[str] = []
    for lb in linked_blocks:
        if lb.block_type != "heading":
            continue
        # Extract text content from the heading HTML tag
        m = re.search(r'<h\d[^>]*>(.*?)</h\d>', lb.html, re.DOTALL)
        if not m:
            continue
        raw_text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        hid = _heading_id(raw_text)
        indent = "  " * max(0, lb.level - 1)
        items.append(
            f'{indent}      <li><a href="content.xhtml#{hid}">{raw_text}</a></li>'
        )
    return "\n".join(items)


def _resolve_image_html(
    image_id: str,
    image_map: Dict[str, ExtractedImage],
) -> str:
    """Return an <img> tag for the given image_id, or empty string if not found."""
    img = image_map.get(image_id)
    if img is None:
        return ""
    src = f"../images/{img.image_id}.{img.ext}"
    return (
        f'<figure>\n'
        f'  <img src="{src}" alt="{img.image_id}" '
        f'width="{img.width_px}" height="{img.height_px}"/>\n'
        f'</figure>'
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_epub(
    linked_blocks: List[LinkedBlock],
    images: List[ExtractedImage],
    output_path: str | Path,
    title: str = "Converted Book",
    author: str = "",
    language: str = "ko",
) -> None:
    """
    Assemble and write an EPUB3 file.

    Args:
        linked_blocks: Output of notes_linker.link_notes().
        images: Output of image_handler.extract_images() (with saved_path set).
        output_path: Where to write the .epub file.
        title: Book title for metadata.
        author: Book author for metadata.
        language: Primary language code (default "ko").
    """
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # 1. Build image lookup map  {image_id → ExtractedImage}
    # ------------------------------------------------------------------
    image_map: Dict[str, ExtractedImage] = {img.image_id: img for img in images}

    # ------------------------------------------------------------------
    # 2. Build the body HTML
    # ------------------------------------------------------------------
    body_parts: List[str] = []

    for lb in linked_blocks:
        if lb.block_type == "image":
            # lb.html is "__IMAGE__img_0001__"
            m = re.match(r'__IMAGE__(.+)__', lb.html)
            if m:
                img_html = _resolve_image_html(m.group(1), image_map)
                if img_html:
                    body_parts.append(img_html)
        else:
            body_parts.append(lb.html)

    body_html = "\n".join(body_parts)

    content_html = _CONTENT_TEMPLATE.format(
        xhtml_ns=_XHTML_NS,
        epub_ns=_EPUB_NS,
        title=title,
        body=body_html,
    )

    # ------------------------------------------------------------------
    # 3. Build nav.xhtml
    # ------------------------------------------------------------------
    toc_items = _build_toc_items(linked_blocks)
    nav_html = _NAV_TEMPLATE.format(
        xhtml_ns=_XHTML_NS,
        epub_ns=_EPUB_NS,
        toc_items=toc_items,
    )

    # ------------------------------------------------------------------
    # 4. Assemble the EPUB book object
    # ------------------------------------------------------------------
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language(language)
    if author:
        book.add_author(author)
    # Additional language declarations
    book.add_metadata("DC", "language", "en")
    book.add_metadata("DC", "language", "zh-Hant")

    # CSS stylesheet
    css_item = epub.EpubItem(
        uid="style",
        file_name="styles/book.css",
        media_type="text/css",
        content=_CSS.encode("utf-8"),
    )
    book.add_item(css_item)

    # Main content XHTML
    content_item = epub.EpubHtml(
        uid="content",
        file_name="content.xhtml",
        media_type="application/xhtml+xml",
    )
    content_item.set_content(content_html.encode("utf-8"))
    content_item.add_item(css_item)
    book.add_item(content_item)

    # nav.xhtml
    nav_item = epub.EpubNav(
        uid="nav",
        file_name="nav.xhtml",
    )
    nav_item.set_content(nav_html.encode("utf-8"))
    book.add_item(nav_item)

    # Images — use .data directly (always present); deduplicate by image_id
    added_image_ids: set = set()
    for img in images:
        if img.image_id in added_image_ids:
            continue
        added_image_ids.add(img.image_id)
        media_type = "image/jpeg" if img.ext == "jpg" else "image/png"
        img_item = epub.EpubItem(
            uid=uid,
            file_name=f"images/{img.image_id}.{img.ext}",
            media_type=media_type,
            content=img.data,
        )
        book.add_item(img_item)

    # ------------------------------------------------------------------
    # 5. Spine and TOC
    # ------------------------------------------------------------------
    book.spine = ["nav", content_item]
    book.toc = [epub.Link("content.xhtml", title, "content")]

    book.add_item(epub.EpubNcx())

    # ------------------------------------------------------------------
    # 6. Write the file
    # ------------------------------------------------------------------
    epub.write_epub(str(output_path), book, {})
    print(f"EPUB written to: {output_path}")
