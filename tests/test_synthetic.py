#!/usr/bin/env python3
"""Camera-free sanity test: renders a fake board (tilted, with ArUco corner markers and
disc 'pieces'), then runs board finding -> warp -> move inference on several moves.

  python tests/test_synthetic.py        # prints PASS/FAIL per case, writes tests/out/*.png
"""
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import ArucoFinder, board_corners_from_markers, homography, warp  # noqa: E402
from chessvision.moves import infer_move, square_diffs  # noqa: E402

SQ, MARGIN, MK = 60, 70, 50             # px: square size, border, marker size
IDS = {"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3}
OUT = Path(__file__).parent / "out"


def render(board: chess.Board) -> np.ndarray:
    side = 8 * SQ + 2 * MARGIN
    img = np.full((side, side, 3), 200, np.uint8)
    for r in range(8):
        for c in range(8):
            col = (181, 217, 240) if (r + c) % 2 == 0 else (99, 136, 181)
            img[MARGIN + r * SQ:MARGIN + (r + 1) * SQ, MARGIN + c * SQ:MARGIN + (c + 1) * SQ] = col
            p = board.piece_at(chess.square(c, 7 - r))
            if p:
                centre = (MARGIN + c * SQ + SQ // 2, MARGIN + r * SQ + SQ // 2)
                fill = (245, 245, 245) if p.color else (30, 30, 30)
                radius = {1: 12, 2: 16, 3: 16, 4: 17, 5: 20, 6: 22}[p.piece_type]
                cv2.circle(img, centre, radius, fill, -1)
                cv2.circle(img, centre, radius, (0, 0, 255), 2)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gen = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker
    lo, hi = MARGIN - MK - 5, MARGIN + 8 * SQ + 5
    for (x, y), mid in zip([(lo, lo), (hi, lo), (hi, hi), (lo, hi)], IDS.values()):
        m = cv2.cvtColor(gen(d, mid, MK), cv2.COLOR_GRAY2BGR)
        img[y:y + MK, x:x + MK] = m
    # pad with white quiet zone, then apply a camera-like perspective tilt
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 30], [w - 20, 0], [w, h], [0, h - 40]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w + 40, h + 40),
                               borderValue=(255, 255, 255))


def locate(img):
    corners, ids = ArucoFinder().detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    pts = board_corners_from_markers(corners, ids, IDS)
    assert pts is not None, f"markers not found (ids={None if ids is None else ids.ravel()})"
    return homography(pts, 800)


def run():
    OUT.mkdir(exist_ok=True)
    games = {
        "opening e4": ([], "e2e4"),
        "capture exd5": (["e2e4", "d7d5"], "e4d5"),
        "kingside castle": (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"], "e1g1"),
        "en passant": (["e2e4", "a7a6", "e4e5", "d7d5"], "e5d6"),
    }
    ok_all = True
    for name, (pre, target) in games.items():
        b = chess.Board()
        for u in pre:
            b.push_uci(u)
        before = render(b)
        after_b = b.copy()
        after_b.push_uci(target)
        after = render(after_b)
        H = locate(before)
        wb, wa = warp(before, H, 800), warp(after, H, 800)
        best, score, _ = infer_move(b, square_diffs(wb, wa), 18)
        passed = best is not None and best.uci() == target
        ok_all &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:16s} expected {target}  got {best.uci() if best else None}")
        if name == "opening e4":
            cv2.imwrite(str(OUT / "camera_view.png"), before)
            cv2.imwrite(str(OUT / "warped.png"), wb)
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(run())
