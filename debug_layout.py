"""
debug_layout.py
---------------
Diagnostic script: render one page and dump raw PP-DocLayout-M output.

Uses the SAME pipeline initialisation as layout_analyzer.py (production):
  1. load_pipeline_config("layout_parsing") + PP-DocLayout-M override
  2. Fallback: create_pipeline(pipeline="layout_parsing") with default model

Usage:
    python debug_layout.py test.pdf 50

Prints every detected box with label, score, and coordinates,
then shows how filter_regions() would classify each one.
"""

from __future__ import annotations

import sys
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# Mirror of layout_analyzer.DISCARD_TYPES / FIGURE_TYPES / TEXT_TYPES
DISCARD_TYPES = frozenset({"header", "footer", "page_number", "number"})
FIGURE_TYPES  = frozenset({"figure", "table", "image"})
MIN_CONFIDENCE = 0.5


def _init_pipeline():
    """
    Initialise layout_parsing pipeline exactly as layout_analyzer.py does.
    Returns (pipeline, description_str).
    """
    from paddlex import create_pipeline

    # --- Option 1: layout_parsing + PP-DocLayout-M (same as production) ---
    try:
        from paddlex.inference.pipelines import load_pipeline_config
        config = load_pipeline_config("layout_parsing")
        try:
            config["SubModules"]["LayoutDetection"]["model_name"] = "PP-DocLayout-M"
        except (KeyError, TypeError) as e:
            print(f"    [warn] Could not set PP-DocLayout-M in config: {e}")
        pipeline = create_pipeline(config=config)
        return pipeline, "layout_parsing + PP-DocLayout-M (production path 1)"
    except Exception as e:
        print(f"    [warn] load_pipeline_config path failed: {e}")

    # --- Option 2: layout_parsing default model (production fallback) ---
    try:
        pipeline = create_pipeline(pipeline="layout_parsing")
        return pipeline, "layout_parsing default model (production path 2)"
    except Exception as e:
        print(f"    [error] layout_parsing also failed: {e}")
        return None, "FAILED"


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_layout.py <pdf_path> <page_number (1-based)>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) - 1  # convert to 0-based

    # ------------------------------------------------------------------
    # [1] Render page
    # ------------------------------------------------------------------
    print(f"[1] Rendering page {page_num + 1} of {pdf_path} ...")
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 dpi — same as production
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image
    import numpy as np
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)
    print(f"    Image size: {img.width}x{img.height}")
    doc.close()

    # ------------------------------------------------------------------
    # [2] Initialise pipeline (production logic)
    # ------------------------------------------------------------------
    print(f"\n[2] Initialising pipeline (production logic) ...")
    pipeline, desc = _init_pipeline()
    if pipeline is None:
        print("Pipeline initialisation failed. Cannot continue.")
        sys.exit(1)
    print(f"    Pipeline ready: {desc}")

    # ------------------------------------------------------------------
    # [3] Run predict
    # ------------------------------------------------------------------
    print(f"\n[3] Running pipeline.predict() ...")
    try:
        result = pipeline.predict(img_array)
        items = list(result)
        print(f"    Got {len(items)} item(s)")
    except Exception as e:
        print(f"    predict() FAILED: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [4] Dump raw structure
    # ------------------------------------------------------------------
    print(f"\n[4] Raw result structure:")
    all_boxes = []  # collect (label, score, coord) for summary

    for i, item in enumerate(items):
        print(f"\n  --- item[{i}] ---")
        print(f"  type: {type(item)}")

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        det_res = _get(item, "layout_det_res")
        if det_res is not None:
            boxes = _get(det_res, "boxes", [])
            print(f"  layout_det_res -> boxes count: {len(boxes or [])}")
            for j, box in enumerate(boxes or []):
                label = box.get("label", "?") if isinstance(box, dict) else getattr(box, "label", "?")
                score = box.get("score", 0)   if isinstance(box, dict) else getattr(box, "score", 0)
                coord = box.get("coordinate", []) if isinstance(box, dict) else getattr(box, "coordinate", [])
                print(f"    box[{j}]: label={label!r:25s} score={score:.3f}  coord={coord}")
                all_boxes.append((str(label), float(score), coord))
        else:
            # Older flat format
            boxes  = _get(item, "boxes",  [])
            labels = _get(item, "labels", [])
            scores = _get(item, "scores", [])
            print(f"  (older format) boxes={len(boxes or [])}, labels={len(labels or [])}")
            for j, (box, label) in enumerate(zip(boxes or [], labels or [])):
                score = scores[j] if scores and j < len(scores) else 0
                print(f"    box[{j}]: label={label!r:25s} score={score:.3f}  coord={box}")
                all_boxes.append((str(label), float(score), box))

    # ------------------------------------------------------------------
    # [5] filter_regions() simulation
    # ------------------------------------------------------------------
    print(f"\n[5] filter_regions() classification (MIN_CONFIDENCE={MIN_CONFIDENCE}):")
    text_count = figure_count = discard_count = skipped_count = 0
    for label, score, coord in all_boxes:
        label_lower = label.lower().strip()
        if score < MIN_CONFIDENCE:
            tag = "SKIP   "
            skipped_count += 1
        elif label_lower in DISCARD_TYPES:
            tag = "DISCARD"
            discard_count += 1
        elif label_lower in FIGURE_TYPES:
            tag = "FIGURE "
            figure_count += 1
        else:
            tag = "TEXT   "
            text_count += 1
        print(f"    [{tag}]  {label!r:25s} score={score:.3f}")

    print(f"\n    Summary: {text_count} text  |  {figure_count} figure  |  {discard_count} discard  |  {skipped_count} skipped")
    if text_count + figure_count + discard_count == 0:
        print("    WARNING: no boxes detected — production will use full-page fallback")

    print("\n[Done]")


if __name__ == "__main__":
    main()
