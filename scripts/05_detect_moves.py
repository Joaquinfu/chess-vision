#!/usr/bin/env python3
"""Step 5 - Follow a real game from the camera (no ML yet).

Start with the pieces in the starting position. Make moves on the real board by hand.
When the image settles (your hand is out of view), the script compares the board to the
last settled image, finds which squares changed, and asks python-chess which LEGAL move
explains that change best. The game is saved as PGN in data/games/.

Keys: r = re-take reference (if lighting changed)   z = undo last move   q = quit

Offline test with two still images (no camera needed):
  python scripts/05_detect_moves.py --before a.png --after b.png [--fen <FEN before>]
"""
import argparse
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import draw_grid, load_board, warp  # noqa: E402
from chessvision.common import (Undistorter, load_calibration, load_config,  # noqa: E402
                                open_camera, put_text, resolve)
from chessvision.moves import infer_move, square_diffs  # noqa: E402


def report(board, best, score, ranked):
    alt = ", ".join(f"{board.san(m)}({s:.0f})" for s, m in ranked[1:3])
    print(f"  -> {board.san(best)}  score {score:.0f}   next best: {alt}")


def offline(args, det):
    board = chess.Board(args.fen) if args.fen else chess.Board()
    before, after = cv2.imread(args.before), cv2.imread(args.after)
    diffs = square_diffs(before, after, det["square_crop"])
    best, score, ranked = infer_move(board, diffs, det["change_threshold"])
    print("changed squares:", sorted(s for s, v in diffs.items() if v > det["change_threshold"]))
    if best is None:
        print("no move detected")
    else:
        report(board, best, score, ranked)


def save_pgn(board):
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "chess-vision test"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    out = resolve("data/games")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"game_{time.strftime('%Y%m%d_%H%M%S')}.pgn"
    p.write_text(str(game) + "\n")
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--fen")
    args = ap.parse_args()
    cfg = load_config()
    det = cfg["detection"]
    if args.before and args.after:
        return offline(args, det)

    _, H, size, rot = load_board(resolve(cfg["board"]["output"]))
    K, dist = load_calibration(resolve(cfg["calibration"]["output"]))
    cap = open_camera(cfg)
    board = chess.Board(args.fen) if args.fen else chess.Board()
    undist, reference, prev, stable, highlight = None, None, None, 0, {}
    stable_thresh = det["change_threshold"] * 0.5      # "nothing is moving" level

    print("Ready. White to move. Keep your hand out of view between moves.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undist is None:
            undist = Undistorter(K, dist, (frame.shape[1], frame.shape[0]))
        top = warp(undist(frame), H, size, rot)

        # stability: compare against the previous frame
        if prev is not None:
            motion = max(square_diffs(prev, top, det["square_crop"]).values())
            stable = stable + 1 if motion < stable_thresh else 0
        prev = top

        if reference is None and stable >= det["settle_frames"]:
            reference = top.copy()
            print("reference captured")
        elif reference is not None and stable == det["settle_frames"]:
            diffs = square_diffs(reference, top, det["square_crop"])
            best, score, ranked = infer_move(board, diffs, det["change_threshold"])
            if best is not None:
                print(f"move {board.fullmove_number}{'.' if board.turn else '...'}")
                report(board, best, score, ranked)
                highlight = {chess.square_name(best.from_square): (0, 0, 255),
                             chess.square_name(best.to_square): (0, 255, 0)}
                board.push(best)
                reference = top.copy()
                if board.is_game_over():
                    print("game over:", board.result())

        view = draw_grid(top, labels=False, highlight=highlight)
        state = "STABLE" if stable >= det["settle_frames"] else "moving..."
        put_text(view, f"{'White' if board.turn else 'Black'} to move  {state}  moves:{len(board.move_stack)}",
                 (10, 30), 0.6)
        cv2.imshow("move detection", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("r"):
            reference = None
        elif k == ord("z") and board.move_stack:
            print("undo", board.pop())
            reference, highlight = None, {}
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    if board.move_stack:
        save_pgn(board)


if __name__ == "__main__":
    main()
