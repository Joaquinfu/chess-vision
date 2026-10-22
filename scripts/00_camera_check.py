#!/usr/bin/env python3
"""Step 0 - Is the webcam working? Shows live feed with resolution + FPS.

Keys:  s = save snapshot to data/snapshots/   q / ESC = quit
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.common import load_config, open_camera, put_text, resolve  # noqa: E402


def main():
    cfg = load_config()
    cap = open_camera(cfg)
    out_dir = resolve("data/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed - is another program using the camera?")
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        h, w = frame.shape[:2]
        view = frame.copy()
        put_text(view, f"{w}x{h}  {fps:4.1f} fps   [s]ave  [q]uit", (10, 30))
        cv2.imshow("camera check", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            p = out_dir / f"snap_{int(now)}.png"
            cv2.imwrite(str(p), frame)
            print("saved", p)
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
