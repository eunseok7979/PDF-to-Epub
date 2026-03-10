"""
debug_layout.py
---------------
Diagnostic script: render one page and dump raw PP-DocLayout-M output.

Usage:
    python debug_layout.py test.pdf 50

Prints every detected box with label, score, and coordinates.
"""

from __future__ import annotations

import sys
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_layout.py <pdf_path> <page_number (1-based)>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) - 1  # convert to 0-based

    # --- Render page ---
    print(f"[1] Rendering page {page_num + 1} of {pdf_path} ...")
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 dpi
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image
    import numpy as np
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)
    print(f"    Image size: {img.width}x{img.height}")

    # --- Run layout analysis ---
    print(f"[2] Running paddlex layout_parsing pipeline ...")
    try:
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline="layout_parsing")
        result = pipeline.predict(img_array)
        items = list(result)
        print(f"    Got {len(items)} item(s) from predict()")
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)

    # --- Dump raw structure ---
    print(f"\n[3] Raw result structure:")
    for i, item in enumerate(items):
        print(f"\n  --- item[{i}] ---")
        print(f"  type: {type(item)}")

        if isinstance(item, dict):
            print(f"  keys: {list(item.keys())}")

            # PaddleX v3: layout_det_res
            det_res = item.get("layout_det_res")
            if det_res is not None:
                print(f"  layout_det_res keys: {list(det_res.keys()) if isinstance(det_res, dict) else type(det_res)}")
                boxes = det_res.get("boxes", []) if isinstance(det_res, dict) else getattr(det_res, "boxes", [])
                print(f"  boxes count: {len(boxes or [])}")
                for j, box in enumerate(boxes or []):
                    label = box.get("label", "?") if isinstance(box, dict) else getattr(box, "label", "?")
                    score = box.get("score", 0) if isinstance(box, dict) else getattr(box, "score", 0)
                    coord = box.get("coordinate", []) if isinstance(box, dict) else getattr(box, "coordinate", [])
                    print(f"    box[{j}]: label={label!r:20s} score={score:.3f}  coord={coord}")
            else:
                # Older format
                boxes  = item.get("boxes",  [])
                labels = item.get("labels", [])
                scores = item.get("scores", [])
                print(f"  (older format) boxes={len(boxes or [])}, labels={len(labels or [])}, scores={len(scores or [])}")
                for j, (box, label) in enumerate(zip(boxes or [], labels or [])):
                    score = scores[j] if scores and j < len(scores) else 0
                    print(f"    box[{j}]: label={label!r:20s} score={score:.3f}  coord={box}")
        else:
            # Object with attributes
            attrs = [a for a in dir(item) if not a.startswith("_")]
            print(f"  attrs: {attrs}")
            det_res = getattr(item, "layout_det_res", None)
            if det_res is not None:
                boxes = getattr(det_res, "boxes", [])
                print(f"  boxes count: {len(boxes or [])}")
                for j, box in enumerate(boxes or []):
                    label = getattr(box, "label", box.get("label", "?") if isinstance(box, dict) else "?")
                    score = getattr(box, "score", box.get("score", 0) if isinstance(box, dict) else 0)
                    coord = getattr(box, "coordinate", box.get("coordinate", []) if isinstance(box, dict) else [])
                    print(f"    box[{j}]: label={label!r:20s} score={score:.3f}  coord={coord}")

    # --- Summary of unique labels ---
    print(f"\n[4] Unique labels found:")
    labels_found = set()
    for item in items:
        det_res = item.get("layout_det_res") if isinstance(item, dict) else getattr(item, "layout_det_res", None)
        if det_res is not None:
            boxes = (det_res.get("boxes", []) if isinstance(det_res, dict) else getattr(det_res, "boxes", []))
            for box in (boxes or []):
                label = box.get("label", "?") if isinstance(box, dict) else getattr(box, "label", "?")
                score = box.get("score", 0) if isinstance(box, dict) else getattr(box, "score", 0)
                labels_found.add((str(label), round(float(score), 3)))
        else:
            for label in (item.get("labels", []) if isinstance(item, dict) else []):
                labels_found.add((str(label), -1))

    for label, score in sorted(labels_found):
        print(f"    {label!r:30s} (score example: {score})")

    doc.close()
    print("\n[Done]")


if __name__ == "__main__":
    main()
