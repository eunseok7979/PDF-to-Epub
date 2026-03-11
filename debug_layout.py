"""
debug_layout.py
---------------
Diagnostic script: render one page and dump layout analysis output.

Supports two engines:
  --engine paddlex  (default) PP-DocLayout-M via PaddleX
  --engine surya    Surya layout detection

Usage:
    python debug_layout.py test.pdf 50
    python debug_layout.py test.pdf 50 --engine surya
"""

from __future__ import annotations

import argparse
import sys
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# Classification sets (used for both engines after label normalisation)
DISCARD_LABELS = frozenset({
    "header", "footer", "page_number", "number",
    "pageheader", "pagefooter",
})
FIGURE_LABELS = frozenset({
    "figure", "table", "image", "picture",
})
MIN_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Engine: PaddleX (PP-DocLayout-M)
# ---------------------------------------------------------------------------

def _run_paddlex(img, img_array, page_num):
    """Run PaddleX layout analysis. Returns list of (label, score, [x0,y0,x1,y1])."""
    import numpy as np
    from paddlex import create_pipeline

    pipeline = None
    try:
        from paddlex.inference.pipelines import load_pipeline_config
        config = load_pipeline_config("layout_parsing")
        try:
            config["SubModules"]["LayoutDetection"]["model_name"] = "PP-DocLayout-M"
        except (KeyError, TypeError):
            pass
        pipeline = create_pipeline(config=config)
        print("    Pipeline: layout_parsing + PP-DocLayout-M")
    except Exception as e:
        print(f"    [warn] config path failed: {e}")

    if pipeline is None:
        pipeline = create_pipeline(pipeline="layout_parsing")
        print("    Pipeline: layout_parsing (default model)")

    result = pipeline.predict(img_array)
    items = list(result)

    all_boxes = []
    for item in items:
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        det_res = _get(item, "layout_det_res")
        if det_res is not None:
            boxes = _get(det_res, "boxes", [])
            for box in (boxes or []):
                if not isinstance(box, dict):
                    continue
                label = str(box.get("label", "?"))
                score = float(box.get("score", 0))
                coord = box.get("coordinate", [])
                try:
                    c = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
                except (IndexError, TypeError, ValueError):
                    continue
                all_boxes.append((label, score, c))
        else:
            boxes = _get(item, "boxes", [])
            labels = _get(item, "labels", [])
            scores = _get(item, "scores", [])
            for j, (box, label) in enumerate(zip(boxes or [], labels or [])):
                score = float(scores[j]) if scores and j < len(scores) else 0
                try:
                    c = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                except (IndexError, TypeError, ValueError):
                    continue
                all_boxes.append((str(label), score, c))

    return all_boxes


# ---------------------------------------------------------------------------
# Engine: Surya
# ---------------------------------------------------------------------------

def _run_surya(img, img_array, page_num):
    """Run Surya layout detection. Returns list of (label, score, [x0,y0,x1,y1])."""
    from surya.layout import LayoutPredictor

    print("    Loading Surya layout model ...")
    try:
        # Surya >= 0.17: requires FoundationPredictor
        from surya.foundation import FoundationPredictor
        predictor = LayoutPredictor(FoundationPredictor())
    except (ImportError, TypeError):
        # Surya 0.16.x: LayoutPredictor takes no args or checkpoint
        predictor = LayoutPredictor()
    print("    Running layout detection ...")
    results = predictor([img])

    all_boxes = []
    for box in results[0].bboxes:
        label = box.label
        score = box.confidence if box.confidence is not None else 0.0
        bbox = box.bbox  # [x_min, y_min, x_max, y_max]
        all_boxes.append((label, score, bbox))

    return all_boxes


# ---------------------------------------------------------------------------
# Classification + visualisation (shared)
# ---------------------------------------------------------------------------

def _classify(label):
    """Return tag: FIGURE, TEXT, DISCARD, or SKIP."""
    key = label.lower().replace("-", "").replace("_", "").strip()
    if key in DISCARD_LABELS:
        return "DISCARD"
    elif key in FIGURE_LABELS:
        return "FIGURE"
    else:
        return "TEXT"


def _print_results(all_boxes):
    """Print detected boxes and classification summary."""
    print(f"\n[4] Detected regions:")
    text_count = figure_count = discard_count = skipped_count = 0

    for label, score, coord in all_boxes:
        if score < MIN_CONFIDENCE:
            tag = "SKIP"
            skipped_count += 1
        else:
            tag = _classify(label)
            if tag == "DISCARD":
                discard_count += 1
            elif tag == "FIGURE":
                figure_count += 1
            else:
                text_count += 1

        print(f"    [{tag:7s}]  {label!r:25s} score={score:.3f}  "
              f"bbox=[{coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f}, {coord[3]:.0f}]")

    print(f"\n    Summary: {text_count} text  |  {figure_count} figure  |  "
          f"{discard_count} discard  |  {skipped_count} skipped")


def _save_debug_image(img, all_boxes, page_num, engine_name):
    """Draw annotated bboxes and save as PNG."""
    from PIL import ImageDraw

    COLORS = {
        "FIGURE": (255, 0, 0),      # red
        "TEXT": (0, 0, 255),         # blue
        "DISCARD": (160, 160, 160),  # grey
        "SKIP": (200, 200, 0),       # yellow
    }

    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)

    for label, score, coord in all_boxes:
        tag = "SKIP" if score < MIN_CONFIDENCE else _classify(label)
        color = COLORS[tag]

        x0, y0, x1, y1 = coord[0], coord[1], coord[2], coord[3]

        # Draw bbox (3px thick)
        for offset in range(3):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                           outline=color)

        caption = f"{label} ({score:.2f}) [{tag}]"
        draw.text((x0 + 4, y0 + 4), caption, fill=color)

    out_name = f"debug_page{page_num + 1}_{engine_name}.png"
    debug_img.save(out_name)
    print(f"\n[5] Saved: {out_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug layout analysis on a single PDF page."
    )
    parser.add_argument("pdf_path", help="Input PDF file")
    parser.add_argument("page", type=int, help="Page number (1-based)")
    parser.add_argument(
        "--engine", choices=["paddlex", "surya"], default="paddlex",
        help="Layout engine to use (default: paddlex)"
    )
    args = parser.parse_args()

    page_num = args.page - 1  # convert to 0-based

    # ------------------------------------------------------------------
    # [1] Render page
    # ------------------------------------------------------------------
    print(f"[1] Rendering page {args.page} of {args.pdf_path} ...")
    import fitz
    doc = fitz.open(args.pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 dpi
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image
    import numpy as np
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)
    print(f"    Image size: {img.width}x{img.height}")
    doc.close()

    # ------------------------------------------------------------------
    # [2-3] Run engine
    # ------------------------------------------------------------------
    print(f"\n[2] Running {args.engine} layout analysis ...")

    if args.engine == "paddlex":
        all_boxes = _run_paddlex(img, img_array, page_num)
    else:
        all_boxes = _run_surya(img, img_array, page_num)

    print(f"    Detected {len(all_boxes)} region(s)")

    # ------------------------------------------------------------------
    # [4] Print results
    # ------------------------------------------------------------------
    _print_results(all_boxes)

    # ------------------------------------------------------------------
    # [5] Save debug image
    # ------------------------------------------------------------------
    _save_debug_image(img, all_boxes, page_num, args.engine)

    print("\n[Done]")


if __name__ == "__main__":
    main()
