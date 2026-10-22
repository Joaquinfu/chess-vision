#!/usr/bin/env bash
# One-time setup on the Jetson Orin Nano (JetPack 6.x / Ubuntu 22.04).
# Usage:  bash setup_jetson.sh
set -e

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip v4l-utils git

# venv that can still see JetPack's system packages (CUDA-enabled OpenCV/torch later)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pyyaml chess

# Need cv2 with the aruco module. JetPack's OpenCV (4.8+) normally has it.
if ! python3 -c "import cv2; cv2.aruco.getPredefinedDictionary" 2>/dev/null; then
  echo "System OpenCV lacks aruco -> installing opencv-contrib-python"
  pip install opencv-contrib-python
fi

python3 -c "import cv2, chess; print('OpenCV', cv2.__version__, '| python-chess', chess.__version__)"
echo
echo "Cameras found:"
v4l2-ctl --list-devices || true
echo
echo "Next:  source .venv/bin/activate && python3 scripts/00_camera_check.py"
