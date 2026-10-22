#!/usr/bin/env python3
"""Step 4 - Collect labelled training data from YOUR camera, board, pieces and lighting.

Set up a position on the real board, tell the script what it is (FEN), and press SPACE.
It saves the full warped board plus all 64 square crops, already sorted into class
folders - ready for a PyTorch ImageFolder classifier later:

  data/raw/squares/<class>/<session>_<n>_<square>.png     class = empty | wP wN ... bK
  data/raw/boards/<session>_<n>.png  +  boards.csv (image, fen)

Usage:
  python scripts/04_capture_dataset.py                         # starting position
  python scripts/04_capture_dataset.py --fen empty             # empty board
  python scripts/04_capture_dataset.py --fen "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

Tips: capture each position under 2-3 lighting conditions; nudge pieces off-centre a bit;
shuffle pieces into random legal-looking positions so every class appears on light AND
dark squares. A few hundred board captures beats a perfect model architecture.

Keys: SPACE = save sample   q = quit
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import chess
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import iter_squares, load_board, warp  # noqa: E402
from chessvision.common import (Undistorter, load_calibration, load_config,  # noqa: E402
                                open_camera, put_text, resolve)


def labels_from_fen(fen: str) -> dict[str, str]:
    if fen == "empty":
        board = chess.Board(None)
    else:
        board = chess.Board(chess.STARTING_FEN if fen == "start" else fen)
    out = {}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        out[chess.square_name(sq)] = "empty" if p is None else ("w" if p.color else "b") + p.symbol().upper()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default="start", help='"start", "empty", or a FEN string')
    args = ap.parse_args()

    cfg = load_config()
    labels = labels_from_fen(args.fen)
    _, H, size, rot = load_board(resolve(cfg["board"]["output"]))
    K, dist = load_calibration(resolve(cfg["calibration"]["output"]))
    root = resolve(cfg["dataset"]["root"])
    (root / "boards").mkdir(parents=True, exist_ok=True)
    csv_path = root / "boards.csv"
    session = time.strftime("%Y%m%d_%H%M%S")

    cap = open_camera(cfg)
    undist, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undist is None:
            undist = Undistorter(K, dist, (frame.shape[1], frame.shape[0]))
        top = warp(undist(frame), H, size, rot)
        view = top.copy()
        put_text(view, f"saved: {n}   [SPACE] save  [q]uit", (10, 30), 0.6)
        cv2.imshow("dataset capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            n += 1
            stem = f"{session}_{n:03d}"
            cv2.imwrite(str(root / "boards" / f"{stem}.png"), top)
            for name, _, _, patch in iter_squares(top):
                d = root / "squares" / labels[name]
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stem}_{name}.png"), patch)
            new = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["image", "fen"])
                w.writerow([f"boards/{stem}.png", args.fen])
            print(f"saved sample {n} (64 squares)")
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
