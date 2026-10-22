# Lab Notebook — Chess Vision (EURēCA!)

> Copy this structure into Notion. One page per session; keep code + data in GitHub and
> link to commits from here.

## 0. Project charter
- **Supervisor:** Dr. Jae-Do Park · **PhD student:** Tanvir Ahmad
- **Goal:** vision system that tracks a physical chess game and gives the robot arms
  board state, moves, and square coordinates. Two Hiwonder arms play each other first;
  human vs. robot later.
- **My scope:** camera setup & calibration · board detection · piece/occupancy detection ·
  move detection · camera→robot coordinate transform · interface to motion control.
- **Platform:** Jetson Orin Nano + USB webcam (training on RTX 5070 PC).
- **Out of scope for now:** arm kinematics, 5→7 DOF upgrade.

## 1. Hardware inventory
| Item | Model / spec | Notes |
|---|---|---|
| SBC | Jetson Orin Nano | JetPack version: |
| Camera | | resolution / fps / FOV: |
| Chess set | | board square size (mm): |
| Arms (×2) | Hiwonder 6-DOF | count servos: arm joints vs. gripper? servo model / bus: |
| Mount | desk lamp (temporary) | height above board (cm): |

## 2. Open questions for Park / Tanvir
- [ ] Final camera placement: fixed overhead, or one per arm?
- [ ] Where do the arms sit relative to the board? (needed for camera→robot transform)
- [ ] How are the arms controlled (Hiwonder controller board / serial / ROS 2)?
- [ ] Does "5 to 7 DOF" mean new mechanics, new drives, or kinematics software?
- [ ] Any existing lab code / datasets to reuse?
- [ ] Weekly hours & meeting cadence

## 3. Session log (template — duplicate per session)
### YYYY-MM-DD — <title>
- **Goal today:**
- **What I did:**
- **Results / numbers:** (reprojection error, move accuracy x/20, fps …)
- **Problems & fixes:**
- **Commit:** <link>
- **Next:**

## 4. Results tracker
| Date | Test | Setup (lighting, height) | Result |
|---|---|---|---|
| | Calibration | | RMS = __ px |
| | Move detection, 20 moves | | __/20 correct |

## 5. References
- Wölflein & Arandjelović 2021, *Determining Chess Game State From an Image* — github.com/georgw777/chesscog
- gm-robot-arm (6-DOF chess robot) — github.com/Charbel199/gm-robot-arm
- LSS Chess Robot (RobotShop tutorial) — github.com/EDGE-tronics/Chess-Robot
- Roboflow chess-board blog (YOLO corners + pieces)
- OpenCV calibration & ArUco tutorials (docs.opencv.org)
- Lynch & Park, *Modern Robotics* (Ch. 3: transforms) · Corke, *Robotics, Vision and Control*
