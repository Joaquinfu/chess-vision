"""Shared helpers: config loading, camera opening, calibration I/O."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Paths in config are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def open_camera(cfg: dict) -> cv2.VideoCapture:
    src = cfg["camera"]["source"]
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    if isinstance(src, str) and "!" in src:          # GStreamer pipeline
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    elif isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    # MJPG lets most USB webcams hit 720p/1080p at 30 fps over USB 2.0
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {src!r}. Try `v4l2-ctl --list-devices`.")
    return cap


# ---------- calibration (camera intrinsics) ----------

def save_calibration(path: Path, K, dist, image_size, rms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": np.asarray(K).tolist(),
        "dist_coeffs": np.asarray(dist).ravel().tolist(),
        "reprojection_error_px": float(rms),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_calibration(path: Path):
    """Returns (K, dist) or (None, None) if the camera hasn't been calibrated yet."""
    if not Path(path).exists():
        return None, None
    with open(path) as f:
        d = yaml.safe_load(f)
    return np.array(d["camera_matrix"], dtype=np.float64), np.array(d["dist_coeffs"], dtype=np.float64)


class Undistorter:
    """Precomputes the undistort maps once so per-frame cost is a single remap."""

    def __init__(self, K, dist, size):
        self.enabled = K is not None
        if self.enabled:
            w, h = size
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

    def __call__(self, frame):
        if not self.enabled:
            return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


def put_text(img, text, org, scale=0.7, color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
