"""
main.py
-------
Command-line entry point for the PDF-to-EPUB converter.

Two-stage pipeline:
  Stage 1: PaddleOCR PP-StructureV2 layout analysis per page.
           - header / footer / page_number  -> discarded
           - figure / table                 -> cropped as images
           - text / title / reference       -> passed to Stage 2

  Stage 2: Per-region OCR reconciliation (character-level 3-way vote).
           - Pages with selectable text: native + Tesseract + EasyOCR
           - Pure-scan pages:            Tesseract + EasyOCR + PaddleOCR

Usage:
    python main.py input.pdf output.epub [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import pdf_extractor
import layout_analyzer
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
    page_range: tuple | None = None,
) -> None:

    # ==================================================================
    # Step 1: Render all pages
    # ==================================================================
    print(f"[1/6] Extracting PDF: {input_pdf}")
    pages = pdf_extractor.extract_pdf(input_pdf, dpi=dpi)
    total_pages = len(pages)
    scanned_count = sum(1 for p in pages if p.is_scanned)
    digital_count = total_pages - scanned_count
    print(f"      {total_pages} pages  "
          f"({digital_count} born-digital/embedded-OCR, {scanned_count} pure-scan)")

    # Filter to requested page range
    if page_range:
        start, end = page_range
        pages = [p for p in pages if start <= p.page_number < end]
        print(f"      Processing pages {start + 1}-{end} ({len(pages)} page(s))")

    # Keep a fitz.Document open for clip-based text extraction
    doc = pdf_extractor.open_pdf(input_pdf)

    # ==================================================================
    # Step 2: Layout analysis + figure extraction + OCR reconciliation
    # ==================================================================
    print(f"[2/6] Layout analysis + OCR reconciliation ...")

    all_region_blocks: List[structure_parser.RegionBlock] = []
    all_images: List[image_handler.ExtractedImage] = []
    image_counter = 1

    # Collect all font sizes for body-font baseline
    all_font_sizes: List[float] = []

    for page in tqdm(pages, desc="Pages", unit="page"):
        rendered = page.rendered_image
        fitz_page = doc[page.page_number]

        # ---- Stage 1: layout analysis ----
        regions = layout_analyzer.analyze_layout(rendered, page.page_number)
        page_h = rendered.height if rendered else 0
        text_regions, figure_regions, discarded = layout_analyzer.filter_regions(
            regions, page_height=page_h
        )

        # ---- Extract figures ----
        if figure_regions:
            fig_bboxes = [r.bbox for r in figure_regions]
            cropped = image_handler.crop_figures(
                rendered, fig_bboxes, page.page_number,
                start_counter=image_counter,
            )
            all_images.extend(cropped)

            # Insert image placeholder RegionBlocks at the correct position
            for i, fr in enumerate(figure_regions):
                img_id = f"img_{image_counter + i:04d}"
                all_region_blocks.append(structure_parser.RegionBlock(
                    region_type=fr.region_type,
                    text=img_id,  # image_id stored in .text for image blocks
                    page_number=page.page_number,
                ))
            image_counter += len(cropped)

        # ---- Stage 2: per-region OCR reconciliation for text regions ----
        for region in text_regions:
            bbox = region.bbox

            # Native text via PyMuPDF clip with paragraph restoration
            # (born-digital pages: uses line bboxes; scanned pages: fallback)
            native_text = pdf_extractor.get_paragraph_text(fitz_page, bbox, dpi)

            # Crop the region image for OCR engines
            region_image = layout_analyzer.crop_region(rendered, bbox)

            # Run OCR engines
            tess_text = tesseract_ocr.run(region_image, lang=tesseract_lang)
            easy_text = easyocr_ocr.run(region_image)

            # Determine which candidates to use
            if page.is_scanned:
                # Pure scan: no native text -> Tesseract + EasyOCR + PaddleOCR
                paddle_text = paddleocr_ocr.run(region_image)
                reconciled = reconciler.reconcile_text(
                    tess_text, easy_text, paddle_text,
                    page_number=page.page_number,
                    region_label=f"region({int(bbox[0])},{int(bbox[1])})",
                )
            else:
                # Born-digital / embedded OCR: native + Tesseract + EasyOCR
                reconciled = reconciler.reconcile_text(
                    native_text, tess_text, easy_text,
                    page_number=page.page_number,
                    region_label=f"region({int(bbox[0])},{int(bbox[1])})",
                )

            # Get font metadata for heading detection
            font_size, is_bold = pdf_extractor.get_font_info_in_region(
                fitz_page, bbox, dpi
            )
            all_font_sizes.append(font_size)

            all_region_blocks.append(structure_parser.RegionBlock(
                region_type=region.region_type,
                text=reconciled,
                page_number=page.page_number,
                font_size=font_size,
                is_bold=is_bold,
            ))

    doc.close()

    # Save extracted figure images
    if all_images:
        image_handler.save_images(all_images, images_dir)
    print(f"      {len(all_images)} figure(s)/table(s) extracted")

    # ==================================================================
    # Step 3: Build document structure
    # ==================================================================
    print(f"[3/6] Parsing document structure ...")

    # Compute dominant body font size
    if all_font_sizes:
        from collections import Counter
        rounded = [round(s, 1) for s in all_font_sizes if s > 0]
        body_font_size = Counter(rounded).most_common(1)[0][0] if rounded else 12.0
    else:
        body_font_size = 12.0

    blocks = structure_parser.build_blocks(all_region_blocks, body_font_size)

    headings = [b for b in blocks if b.block_type == "heading"]
    footnotes = [b for b in blocks if b.block_type == "footnote"]
    endnotes = [b for b in blocks if b.block_type == "endnote"]
    print(f"      {len(headings)} headings, "
          f"{len(footnotes)} footnotes, "
          f"{len(endnotes)} endnotes")

    # ==================================================================
    # Step 4: Link footnotes & endnotes
    # ==================================================================
    print(f"[4/6] Linking footnotes & endnotes ...")
    linked_blocks = notes_linker.link_notes(blocks)

    # ==================================================================
    # Step 5: Build EPUB
    # ==================================================================
    print(f"[5/6] Building EPUB ...")
    epub_builder.build_epub(
        linked_blocks=linked_blocks,
        images=all_images,
        output_path=output_epub,
        title=title,
        author=author,
    )

    print(f"[6/6] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-to-epub",
        description="Convert a PDF to EPUB3 with layout analysis, "
                    "OCR reconciliation, and linked notes.",
    )
    parser.add_argument("input", metavar="INPUT.pdf", help="Input PDF file")
    parser.add_argument("output", metavar="OUTPUT.epub", help="Output EPUB file")
    parser.add_argument("--title", default="", help="Book title (default: PDF filename)")
    parser.add_argument("--author", default="", help="Author name")
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Render resolution for all pages (default: 300)"
    )
    parser.add_argument(
        "--lang", default="kor+eng",
        help="Tesseract language string (default: kor+eng)"
    )
    parser.add_argument(
        "--pages", default=None, metavar="RANGE",
        help="Page range to process, e.g. '50' or '10-20' (1-based, default: all)"
    )
    parser.add_argument(
        "--ocr-log", default=None, metavar="FILE",
        help="Write OCR discrepancy warnings to FILE"
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help="Directory for extracted figure images (default: <output>.images/)"
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

    # Parse --pages
    page_range = None
    if args.pages:
        if "-" in args.pages:
            start, end = args.pages.split("-", 1)
            page_range = (int(start) - 1, int(end))  # convert to 0-based start
        else:
            p = int(args.pages) - 1  # convert to 0-based
            page_range = (p, p + 1)

    convert(
        input_pdf=input_pdf,
        output_epub=output_epub,
        title=title,
        author=args.author,
        dpi=args.dpi,
        tesseract_lang=args.lang,
        images_dir=images_dir,
        page_range=page_range,
    )


if __name__ == "__main__":
    main()
