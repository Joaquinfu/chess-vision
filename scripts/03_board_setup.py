#!/usr/bin/env python3
"""Step 3 - Find the board and warp it to a clean top-down 8x8 grid.

Two ways to locate the board:
  * ArUco (default): 4 markers taped just outside the board corners, ids per config.yaml.
  * Manual (press m): click the 4 playing-area corners in order TL, TR, BR, BL.

Rotate with r until a8 is top-left / white is at the bottom. Press s to save the
homography to calib/board.yaml - later scripts use it so the camera must NOT move after.

Keys: m = manual click mode   r = rotate 90deg   s = save   q = quit
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import (ArucoFinder, board_corners_from_markers, draw_grid,  # noqa: E402
                               homography, save_board, warp)
from chessvision.common import (Undistorter, load_calibration, load_config,  # noqa: E402
                                open_camera, put_text, resolve)

clicks: list = []


def on_mouse(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))


def main():
    cfg = load_config()
    b = cfg["board"]
    size, rot = b["warp_size"], b["rotate_k"]
    cap = open_camera(cfg)
    K, dist = load_calibration(resolve(cfg["calibration"]["output"]))
    if K is None:
        print("WARNING: no camera calibration found - running on raw (distorted) frames.")
    finder = ArucoFinder(b["aruco_dict"])
    undist = None
    manual = False
    corners = None
    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undist is None:
            undist = Undistorter(K, dist, (frame.shape[1], frame.shape[0]))
        frame = undist(frame)
        view = frame.copy()

        if manual:
            for p in clicks:
                cv2.circle(view, p, 6, (0, 0, 255), -1)
            if len(clicks) == 4:
                corners = np.array(clicks, np.float32)
            put_text(view, f"MANUAL: click TL,TR,BR,BL ({len(clicks)}/4)  [m] back to ArUco", (10, 30), 0.6)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mk, ids = finder.detect(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(view, mk, ids)
            found = board_corners_from_markers(mk, ids, b["marker_ids"])
            if found is not None:
                corners = found
            n = 0 if ids is None else len(ids)
            put_text(view, f"ArUco markers seen: {n}/4   [m]anual [r]otate [s]ave [q]uit", (10, 30), 0.6)

        if corners is not None:
            cv2.polylines(view, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            H = homography(corners, size)
            top = draw_grid(warp(frame, H, size, rot))
            cv2.imshow("warped (a8 should be top-left)", top)

        cv2.imshow("camera", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("m"):
            manual = not manual
            clicks.clear()
            corners = None
        elif k == ord("r"):
            rot = (rot + 1) % 4
        elif k == ord("s") and corners is not None:
            out = resolve(b["output"])
            save_board(out, corners, homography(corners, size), size, rot)
            cv2.imwrite(str(resolve("data/board_reference.png")), warp(frame, homography(corners, size), size, rot))
            print(f"saved board geometry -> {out} (rotate_k={rot})")
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
