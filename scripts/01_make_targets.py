#!/usr/bin/env python3
"""Step 1 - Generate printable targets into targets/:

  * checkerboard.png   - for camera calibration (print at 100% scale, tape FLAT to cardboard)
  * aruco_0..3.png     - the four board-corner markers (ids from config.yaml)
  * aruco_sheet.png    - all four on one page

After printing, MEASURE a checkerboard square and put the real size in config.yaml.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.common import load_config, resolve  # noqa: E402

DPI = 300
MM_PER_INCH = 25.4


def mm(px_mm):
    return int(round(px_mm * DPI / MM_PER_INCH))


def checkerboard(cols_inner, rows_inner, square_mm):
    sq = mm(square_mm)
    cols, rows = cols_inner + 1, rows_inner + 1
    margin = sq
    img = np.full((rows * sq + 2 * margin, cols * sq + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    return img


def aruco_marker(dict_name, marker_id, side_mm=40):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    side = mm(side_mm)
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(d, marker_id, side)
    else:
        m = cv2.aruco.drawMarker(d, marker_id, side)
    pad = mm(5)                                   # white quiet zone is required for detection
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(m, f"id {marker_id}", (pad, pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return m


def main():
    cfg = load_config()
    out = resolve("targets")
    out.mkdir(exist_ok=True)
    c = cfg["calibration"]
    cb = checkerboard(c["board_cols"], c["board_rows"], c["square_size_mm"])
    cv2.imwrite(str(out / "checkerboard.png"), cb)
    print(f"checkerboard.png: {c['board_cols']+1}x{c['board_rows']+1} squares of {c['square_size_mm']} mm @ {DPI} dpi")

    b = cfg["board"]
    markers = []
    for name, mid in b["marker_ids"].items():
        m = aruco_marker(b["aruco_dict"], mid)
        cv2.imwrite(str(out / f"aruco_{mid}.png"), m)
        markers.append(m)
        print(f"aruco_{mid}.png -> place at {name}")
    gap = np.full((markers[0].shape[0], mm(15)), 255, np.uint8)
    sheet = np.hstack([x for m in markers for x in (m, gap)][:-1])
    cv2.imwrite(str(out / "aruco_sheet.png"), sheet)
    print(f"Saved to {out}. Print at 100% / 'actual size' - no 'fit to page'.")


if __name__ == "__main__":
    main()
