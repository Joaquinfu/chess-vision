"""Move inference: which legal move best explains the per-square image change?

We don't need to recognise pieces to follow a game. The vision side measures how much
each square changed between "before" and "after"; python-chess knows which squares each
legal move touches. The legal move whose touched squares changed the most (and whose
untouched squares changed the least) wins. Captures, castling and en passant all work
because they're just different sets of touched squares.
"""
from __future__ import annotations

import cv2
import numpy as np
import chess

from .board import iter_squares


def square_diffs(before, after, crop: float = 0.6) -> dict[str, float]:
    """Mean absolute grayscale difference per square, after light blurring."""
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.GaussianBlur(g, (5, 5), 0)

    b, a = prep(before), prep(after)
    d = cv2.absdiff(b, a)
    return {name: float(p.mean()) for name, _, _, p in iter_squares(d, crop)}


def touched_squares(board: chess.Board, move: chess.Move) -> set[str]:
    sq = {chess.square_name(move.from_square), chess.square_name(move.to_square)}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from, rook_to = (7, 5) if kingside else (0, 3)
        sq |= {chess.square_name(chess.square(rook_from, rank)), chess.square_name(chess.square(rook_to, rank))}
    if board.is_en_passant(move):
        cap = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        sq.add(chess.square_name(cap))
    return sq


def infer_move(board: chess.Board, diffs: dict[str, float], threshold: float):
    """Return (best_move, score, ranked_candidates). best_move is None if nothing changed."""
    changed = {s for s, v in diffs.items() if v > threshold}
    if not changed:
        return None, 0.0, []
    ranked = []
    for mv in board.legal_moves:
        t = touched_squares(board, mv)
        hit = sum(diffs[s] for s in t) / len(t)                     # touched squares should change
        miss = sum(diffs[s] for s in changed - t)                  # changes this move can't explain
        quiet = sum(diffs[s] < threshold for s in t)               # touched squares that didn't change
        score = hit - miss - 10.0 * quiet
        ranked.append((score, mv))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]
    return best, best_score, ranked[:5]
