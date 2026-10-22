"""Board localisation (ArUco or manual corners), perspective warp, and square slicing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

FILES = "abcdefgh"
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


# ---------- ArUco (works across OpenCV 4.5 - 5.x) ----------

class ArucoFinder:
    def __init__(self, dict_name: str = "DICT_4X4_50"):
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        if hasattr(aruco, "ArucoDetector"):               # OpenCV >= 4.7
            self._det = aruco.ArucoDetector(self.dictionary, aruco.DetectorParameters())
            self._legacy = None
        else:                                              # older JetPack OpenCV
            self._det = None
            self._legacy = aruco.DetectorParameters_create()

    def detect(self, gray):
        if self._det is not None:
            corners, ids, _ = self._det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self._legacy)
        return corners, ids


def board_corners_from_markers(corners, ids, marker_ids: dict) -> np.ndarray | None:
    """Return the 4 playing-area corners (TL, TR, BR, BL) in image pixels.

    Each marker sits just outside one corner of the board. For each marker we take
    its own corner that points toward the centre of all four markers (its 'inner'
    corner) - this works no matter how each marker is rotated on the table.
    """
    if ids is None:
        return None
    found = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}
    want = [marker_ids[name] for name in CORNER_ORDER]
    if not all(m in found for m in want):
        return None
    centre = np.mean([found[m].mean(axis=0) for m in want], axis=0)
    pts = []
    for m in want:
        quad = found[m]
        pts.append(quad[np.argmin(np.linalg.norm(quad - centre, axis=1))])
    return np.array(pts, dtype=np.float32)


# ---------- homography / warp ----------

def homography(img_corners: np.ndarray, size: int) -> np.ndarray:
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(img_corners.astype(np.float32), dst)


def warp(frame, H, size: int, rotate_k: int = 0):
    top = cv2.warpPerspective(frame, H, (size, size))
    return np.ascontiguousarray(np.rot90(top, k=rotate_k)) if rotate_k % 4 else top


def save_board(path: Path, img_corners, H, size, rotate_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"image_corners_TL_TR_BR_BL": np.asarray(img_corners).tolist(),
                        "homography": np.asarray(H).tolist(),
                        "warp_size": int(size), "rotate_k": int(rotate_k)}, f, sort_keys=False)


def load_board(path: Path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return np.array(d["image_corners_TL_TR_BR_BL"], np.float32), np.array(d["homography"]), d["warp_size"], d["rotate_k"]


# ---------- squares ----------

def square_name(row: int, col: int) -> str:
    """Warped image is oriented with a8 top-left (white at the bottom)."""
    return f"{FILES[col]}{8 - row}"


def iter_squares(warped, crop: float = 1.0):
    """Yield (name, row, col, patch). crop<1 takes the centre fraction of each square."""
    n = warped.shape[0] // 8
    m = int(n * (1 - crop) / 2)
    for r in range(8):
        for c in range(8):
            y0, x0 = r * n, c * n
            yield square_name(r, c), r, c, warped[y0 + m:y0 + n - m, x0 + m:x0 + n - m]


def draw_grid(warped, labels: bool = True, highlight: dict | None = None):
    out = warped.copy()
    n = out.shape[0] // 8
    for i in range(9):
        cv2.line(out, (i * n, 0), (i * n, 8 * n), (0, 255, 0), 1)
        cv2.line(out, (0, i * n), (8 * n, i * n), (0, 255, 0), 1)
    for name, r, c, _ in iter_squares(out):
        if highlight and name in highlight:
            cv2.rectangle(out, (c * n + 2, r * n + 2), (c * n + n - 2, r * n + n - 2), highlight[name], 3)
        if labels:
            cv2.putText(out, name, (c * n + 4, r * n + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out
