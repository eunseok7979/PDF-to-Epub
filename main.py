"""
main.py
-------
Command-line entry point for the PDF → EPUB converter.

Usage:
    python main.py input.pdf output.epub [options]

Options:
    --title TEXT        Book title  (default: PDF filename stem)
    --author TEXT       Author name (default: empty)
    --dpi INT           Render DPI for scanned pages (default: 300)
    --lang TEXT         Tesseract language string (default: kor+eng+chi_tra)
    --ocr-log FILE      Write OCR discrepancy log to FILE (default: stderr)
    --images-dir DIR    Temp directory for extracted images (default: <output>.images/)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict

from tqdm import tqdm

import pdf_extractor
import image_handler
import structure_parser
import notes_linker
import epub_builder
from ocr import tesseract_ocr, easyocr_ocr, paddleocr_ocr, reconciler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str | None) -> None:
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# OCR pipeline for a single page
# ---------------------------------------------------------------------------

def _run_ocr_for_page(
    page: pdf_extractor.PageData,
    tesseract_lang: str,
) -> Dict[str, str]:
    """
    Run the appropriate OCR engines for the given page and return a dict of
    raw text outputs keyed by engine name.

    Born-digital or scanned-with-embedded-text: native + Tesseract + EasyOCR (3-way)
    Pure scan (no text layer):                  Tesseract + EasyOCR + PaddleOCR (3-way)
    """
    img = page.rendered_image  # always set (pdf_extractor renders all pages)
    results: Dict[str, str] = {
        "tesseract": tesseract_ocr.run(img, lang=tesseract_lang),
        "easyocr": easyocr_ocr.run(img),
    }
    if page.is_scanned:
        results["paddleocr"] = paddleocr_ocr.run(img)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert(
    input_pdf: Path,
    output_epub: Path,
    title: str,
    author: str,
    dpi: int,
    tesseract_lang: str,
    images_dir: Path,
) -> None:
    print(f"[1/6] Extracting PDF: {input_pdf}")
    pages = pdf_extractor.extract_pdf(input_pdf, dpi=dpi)
    total_pages = len(pages)
    scanned_count = sum(1 for p in pages if p.is_scanned)
    digital_count = total_pages - scanned_count
    print(f"      {total_pages} pages  ({digital_count} born-digital, {scanned_count} scanned)")

    print(f"[2/6] Extracting images …")
    images = image_handler.extract_images(input_pdf)
    print(f"      {len(images)} image occurrence(s) found")
    image_handler.save_images(images, images_dir)
    page_image_map = image_handler.build_page_image_map(images)

    print(f"[3/6] Running OCR & reconciliation …")
    reconciled_texts: Dict[int, str] = {}

    for page in tqdm(pages, desc="Pages", unit="page"):
        # Build native text from born-digital text blocks
        native_text = "\n".join(
            tb.text for tb in page.text_blocks if tb.text.strip()
        )

        ocr_results = _run_ocr_for_page(page, tesseract_lang)

        reconciled = reconciler.reconcile_page(
            page_number=page.page_number,
            is_scanned=page.is_scanned,
            native_text=native_text,
            tesseract_text=ocr_results.get("tesseract", ""),
            easyocr_text=ocr_results.get("easyocr", ""),
            paddleocr_text=ocr_results.get("paddleocr", ""),
        )
        reconciled_texts[page.page_number] = reconciled

    print(f"[4/6] Parsing document structure …")
    blocks = structure_parser.parse_structure(pages, reconciled_texts)
    blocks = structure_parser.inject_image_placeholders(blocks, page_image_map)
    headings = [b for b in blocks if b.block_type == "heading"]
    footnotes = [b for b in blocks if b.block_type == "footnote"]
    endnotes = [b for b in blocks if b.block_type == "endnote"]
    print(
        f"      {len(headings)} headings, "
        f"{len(footnotes)} footnotes, "
        f"{len(endnotes)} endnotes"
    )

    print(f"[5/6] Linking footnotes & endnotes …")
    linked_blocks = notes_linker.link_notes(blocks)

    print(f"[6/6] Building EPUB …")
    epub_builder.build_epub(
        linked_blocks=linked_blocks,
        images=images,
        output_path=output_epub,
        title=title,
        author=author,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-to-epub",
        description="Convert a PDF file to EPUB3 with OCR reconciliation and linked notes.",
    )
    parser.add_argument("input", metavar="INPUT.pdf", help="Input PDF file")
    parser.add_argument("output", metavar="OUTPUT.epub", help="Output EPUB file")
    parser.add_argument("--title", default="", help="Book title (default: PDF filename)")
    parser.add_argument("--author", default="", help="Author name")
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Render resolution for scanned pages (default: 300)"
    )
    parser.add_argument(
        "--lang", default="kor+eng+chi_tra",
        help="Tesseract language string (default: kor+eng+chi_tra)"
    )
    parser.add_argument(
        "--ocr-log", default=None, metavar="FILE",
        help="Write OCR discrepancy warnings to FILE"
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help="Directory for extracted images (default: <output>.images/)"
    )

    args = parser.parse_args()

    _setup_logging(args.ocr_log)

    input_pdf = Path(args.input)
    if not input_pdf.exists():
        print(f"Error: input file not found: {input_pdf}", file=sys.stderr)
        sys.exit(1)

    output_epub = Path(args.output)
    title = args.title or input_pdf.stem.replace("_", " ").replace("-", " ").title()

    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = output_epub.parent / (output_epub.stem + ".images")

    convert(
        input_pdf=input_pdf,
        output_epub=output_epub,
        title=title,
        author=args.author,
        dpi=args.dpi,
        tesseract_lang=args.lang,
        images_dir=images_dir,
    )


if __name__ == "__main__":
    main()
