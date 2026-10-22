#!/usr/bin/env python3
"""Step 2 - Camera calibration (intrinsics K + lens distortion).

Hold the printed checkerboard in front of the camera and press SPACE to capture a view
whenever the corners are drawn in colour. Vary the views: tilt it, move it to the
corners and edges of the frame, near and far. ~15-25 views.

Keys: SPACE = capture   c = calibrate & save   u = toggle undistort preview   q = quit

Good result: reprojection error < 0.5 px (< 1.0 px is usable for a webcam).
Re-run this any time you change camera, resolution, or focus.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.common import (Undistorter, load_config, open_camera, put_text,  # noqa: E402
                                resolve, save_calibration)


def main():
    cfg = load_config()
    c = cfg["calibration"]
    pattern = (c["board_cols"], c["board_rows"])
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * c["square_size_mm"]

    view_dir = resolve("data/calib_views")
    view_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(cfg)
    obj_pts, img_pts, size = [], [], None
    undist, show_undist = None, False
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        view = undist(frame) if (show_undist and undist) else frame.copy()
        if found and not show_undist:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        put_text(view, f"views: {len(img_pts)}/{c['min_views']}  [SPACE] capture [c]alibrate "
                       f"[u]ndistort [q]uit", (10, 30), 0.6)
        if show_undist:
            put_text(view, "UNDISTORTED preview - straight edges should look straight", (10, 60), 0.6, (0, 200, 255))
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(1) & 0xFF

        if k == ord(" ") and found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
            cv2.imwrite(str(view_dir / f"view_{len(img_pts):02d}.png"), frame)
            print(f"captured view {len(img_pts)}")
        elif k == ord("c"):
            if len(img_pts) < c["min_views"]:
                print(f"need at least {c['min_views']} views (have {len(img_pts)})")
                continue
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
            # per-view error helps you spot (and delete) a bad capture
            errs = []
            for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
                proj, _ = cv2.projectPoints(o, r, t, K, dist)
                errs.append(cv2.norm(i, proj, cv2.NORM_L2) / np.sqrt(len(proj)))
            out = resolve(c["output"])
            save_calibration(out, K, dist, size, rms)
            undist = Undistorter(K, dist, size)
            print(f"\nRMS reprojection error: {rms:.3f} px  (worst view: #{int(np.argmax(errs)) + 1} "
                  f"at {max(errs):.3f} px)")
            print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
            print(f"saved -> {out}\n")
        elif k == ord("u"):
            show_undist = not show_undist
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
