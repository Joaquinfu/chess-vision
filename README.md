# chess-vision

Machine-vision system for a chess-playing robotic manipulator.
CU Denver EURēCA! — *Machine Vision for Robotic Manipulators* (Dr. Jae-Do Park's group).

**Goal:** a fixed camera watches a physical chessboard and reports the board state and each
move — first so two Hiwonder 6-DOF arms can play each other, later so a human can play the robot.
The vision system must also tell each arm *where* (in its own coordinates) every square is.

```
camera ─► undistort ─► find board ─► warp top-down ─► 64 squares ─► what changed? ─► legal move ─► robot
 (02)        (02)          (03)           (03)            (03)          (05)          (05)       (later)
```

## Hardware (phase 1)
| Item | Notes |
|---|---|
| NVIDIA Jetson Orin Nano | runs everything below |
| USB webcam | temporarily hung from a desk lamp, pointing straight down |
| Chess set | high contrast pieces/board help a lot |
| Printed targets | checkerboard + 4 ArUco markers (`scripts/01_make_targets.py`) |
| 2× Hiwonder arms | not needed until phase 3 |

**Camera mount rule:** once `03_board_setup.py` is saved, the camera and board must not move.
Tape the board down. If anything moves, just re-run step 3 (takes seconds).

## Setup (Jetson)
```bash
git clone <this repo> && cd chess-vision
bash setup_jetson.sh
source .venv/bin/activate
python3 tests/test_synthetic.py      # camera-free self-test, should print ALL PASS
```

## Run order
| Step | Command | Done when |
|---|---|---|
| 0 | `python3 scripts/00_camera_check.py` | live feed, ~30 fps |
| 1 | `python3 scripts/01_make_targets.py` → print `targets/` at 100% scale | checkerboard square measured, `config.yaml` updated |
| 2 | `python3 scripts/02_calibrate.py` | `calib/camera.yaml` saved, reprojection error < 0.5–1.0 px |
| 3 | `python3 scripts/03_board_setup.py` | warped view shows a clean grid, a8 top-left → `calib/board.yaml` |
| 4 | `python3 scripts/04_capture_dataset.py --fen start` | labelled square crops in `data/raw/squares/<class>/` |
| 5 | `python3 scripts/05_detect_moves.py` | play 20 moves by hand, check the printed moves/PGN |

ArUco placement: tape marker 0 at the top-left corner of the playing area (as the camera sees
it), 1 top-right, 2 bottom-right, 3 bottom-left, each just *outside* the board with one corner
touching the board corner. No markers yet? Press `m` in step 3 and click the 4 corners.

## How move detection works (no ML needed)
The system always knows the position before a move. After a move it measures how much each
square changed and asks python-chess which **legal** move touches exactly those squares.
Captures, castling and en passant fall out for free. Limitation: promotions assume a queen —
that's one of the things piece classification (phase 2) fixes.

## Roadmap
- **Phase 1 – classical pipeline (now):** camera, calibration, board warp, move following, accuracy log.
- **Phase 2 – learning:** train a per-square classifier (PyTorch) or YOLO on data from step 4,
  on the RTX 5070 PC; deploy to the Jetson (TensorRT). Gives full board state from one image.
- **Phase 3 – robot frame:** square centres → each arm's base frame (x, y, z); hand-eye calibration
  if a camera rides on an arm.
- **Phase 4 – interface:** publish board state + moves to the arm controllers (ROS 2 topic or serial/JSON);
  human-vs-robot mode.

## Repo layout
```
chessvision/   common.py (config, camera, calibration I/O)  board.py (ArUco, warp, squares)  moves.py (move inference)
scripts/       00–05 numbered steps above
tests/         test_synthetic.py – runs the whole pipeline on rendered images
calib/         camera.yaml, board.yaml (generated, committed)
data/          raw images & datasets (git-ignored)
docs/          NOTEBOOK.md – lab notebook template (mirror in Notion)
```
