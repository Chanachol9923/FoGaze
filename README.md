# FoGaze — Gaze-Driven Object Focus & Blink-to-Grasp

**Look at a real-world object, and FoGaze knows what it is, how far away it is,
and where it sits relative to other things. Blink three times, and a robot arm
picks it up.**

FoGaze combines a face camera (for gaze estimation), a scene + depth source
(for object detection and 3D position), and a triple-blink gesture that fires a
ROS 2 / MoveIt pickup of whatever you're looking at.

```
Face camera  ──► EyeTrax gaze model ──► gaze (x, y)
                                            │
Scene + depth ──► YOLO detection      ──► bounding boxes + 3D depth
                                            │
                          which bbox contains the gaze point?
                                            │
                                     focused object  ──► spoken description
                                            │
                       triple-blink ──► ROS 2 pickup ──► MoveIt arm grasps it
```

---

## Features

- **Gaze tracking** — face camera + MediaPipe landmarks → EyeTrax Ridge model →
  scene-frame gaze point. Optional **face-lock** zoom for sharper landmarks.
- **Object detection** — Ultralytics YOLO (closed-vocab `yolov8n`/`yolov8s` or
  open-vocab YOLO-World). The bounding box under your gaze is the *focused* object.
- **Depth & 3D** — pluggable depth source: **stereo** (two USB cams, SGBM),
  **PrimeSense** (OpenNI2), or **sim** (synthetic scene, no hardware).
- **Spoken feedback** — speaks the focused object's name, spatial relationship
  (in / on / under / next to / near / behind) and distance, via the Piper neural
  voice (falls back to espeak-ng).
- **Blink → grasp** — a triple blink fires a robot-arm pickup of the focused
  object. Graspability is checked (reachable distance, gripper width) and shown
  by box colour. Drives a **MoveIt Franka Panda** arm or a lightweight **mock arm**.
- **ImGui UI** — live status, depth toggle/sliders, stereo tuning, face + eye
  previews, all tunable while running.
- **Calibration** — 9-point gaze calibration with outlier rejection and accuracy
  grading, chessboard lens calibration, and a one-key drift re-center.

---

## Requirements

- Linux (developed on Ubuntu 22.04, GNOME + XWayland)
- Python 3.10+
- A face-facing USB camera, plus a scene/depth source:
  - two USB cameras (stereo), **or** a PrimeSense / OpenNI2 depth camera, **or**
  - nothing — run in `--sim` mode
- [EyeTrax](https://github.com/ck-zhang/EyeTrax) (installed separately)
- For the arm pipeline: **ROS 2 Humble** + MoveIt 2 (Franka Panda) — optional

---

## Installation

```bash
# 1. Install EyeTrax from source
git clone https://github.com/ck-zhang/EyeTrax /tmp/EyeTrax
pip install -e /tmp/EyeTrax

# 2. Install FoGaze dependencies
pip install -r requirements.txt

# 3. (Optional) Download the natural Piper voice for clearer speech.
#    Without it, FoGaze falls back to the robotic espeak-ng voice.
python3 -m piper.download_voices en_US-amy-medium \
    --download-dir ~/.cache/fogaze3/piper

# 4. Run
python3 main.py
```

On first run, FoGaze scans the cameras and lets you pick the face camera and
depth source from the start screen.

---

## Running

```bash
# Standalone, pick depth source on the start screen
python3 main.py

# Simulation mode — no hardware needed
python3 main.py --sim

# Explicit stereo depth
python3 main.py --depth-source stereo --stereo-left 2 --stereo-right 4

# PrimeSense (OpenNI2) depth
python3 main.py --depth-source primesense

# Publish detections + blink pickups to ROS 2
python3 main.py --ros
```

### Blink → grasp (full robot-arm pipeline)

`fogaze_arm.sh` launches the arm pipeline **and** the app together, on the same
ROS domain, in one command. Triple-blink a **green** (graspable) object to pick it.

```bash
./fogaze_arm.sh                 # Tier 2: MoveIt Panda + RViz + app
./fogaze_arm.sh mock            # Tier 1: lightweight mock-arm marker + app
./fogaze_arm.sh --sim           # Tier 2 in sim mode (no depth cam)
./fogaze_arm.sh mock --sim      # Tier 1 in sim mode
# Any extra flags are forwarded to main.py, e.g.:
./fogaze_arm.sh moveit --stereo-left 2 --stereo-right 4

# Watch the pickup state:
ros2 topic echo /fogaze/pickup_status
```

`fogaze.sh` runs just the app + RViz visualization (no arm).

---

## Controls

| Key | Action |
|-----|--------|
| `ENTER` | Capture current calibration point |
| `BACKSPACE` | Undo last calibration point |
| `ESC` | Finish calibration / quit |
| `c` | Re-calibrate gaze while running |
| `g` | One-point drift re-center |
| `f` | Toggle fullscreen |

Most settings (depth toggle/offsets, stereo params, smoothing) are tunable live
in the left ImGui panel.

---

## Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--face-camera` | auto | Face/gaze camera index |
| `--scene-camera` | auto | Scene/object camera index |
| `--depth-source` | `stereo` | `stereo` \| `primesense` \| `sim` |
| `--stereo-left` / `--stereo-right` | auto | Stereo camera indices |
| `--sim` | — | Force simulation mode (no hardware) |
| `--model` | `ridge` | EyeTrax gaze model |
| `--filter` | `one_euro` | `one_euro` \| `kalman` \| `kalman_ema` \| `kde` \| `none` |
| `--ema-alpha` | `0.5` | EMA smoothing (0=off, 1=max) |
| `--blink-ratio` | `0.6` | Blink sensitivity (lower = less twitchy) |
| `--blink-count` | `3` | Blinks for the pickup gesture (min 3) |
| `--face-lock` / `--no-face-lock` | on | Zoom/track face for sharper landmarks |
| `--yolo-model` | `models/yolov8n.pt` | YOLO weights path |
| `--confidence` | `0.35` | YOLO confidence threshold |
| `--iou` | `0.45` | YOLO NMS IoU |
| `--detection-interval` | `3` | Run YOLO every N frames |
| `--imgsz` | `320` | YOLO inference size |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:0` |
| `--ros` | — | Publish detections + pickups to ROS 2 |
| `--headless` | — | Run without a display window |
| `--reset-model` | — | Delete saved gaze model and exit |

---

## Calibration

1. **Face detection** — look at the face camera for ~2 s (countdown arc).
2. **9-point grid** — a target appears on the scene feed; look at the real-world
   area it covers and press `ENTER`. The model trains with outlier rejection and
   reports a GOOD / OK / POOR accuracy grade.
3. **Drift re-center** (`g`) — look at the centre dot to recapture a single-point
   offset without a full recalibration.
4. **Lens calibration** (optional) — chessboard-based undistortion for the face
   camera. Generate a printable board with `python3 modules/make_chessboard.py`.

The trained model is saved to `~/.cache/fogaze3/eyetrax_model.pkl`.

---

## Project structure

```
FoGaze/
├── main.py                      # Entry point: calibration + main loop + UI + ROS
├── modules/
│   ├── eyetrax_features.py      # 486-D gaze feature extraction
│   ├── calibrator_sklearn.py    # sklearn gaze calibrators
│   ├── camera_calibrator.py     # Chessboard lens calibration
│   ├── make_chessboard.py       # Generate a printable calibration board
│   ├── face_lock.py             # Face-region zoom tracking
│   ├── object_detector.py       # YOLO wrapper (closed- & open-vocab)
│   ├── depth_estimator.py       # PrimeSense (OpenNI2) depth
│   ├── stereo_depth_estimator.py# Stereo SGBM depth
│   ├── stereo_calibrator.py     # Stereo camera calibration
│   ├── mock_depth_estimator.py  # Simulated depth (sim mode)
│   ├── mock_scene.py            # Procedural test scene
│   ├── grasp.py                 # Grasp geometry + graspability rules
│   ├── filters.py               # Gaze smoothing (One-Euro / Kalman / EMA / KDE)
│   ├── gui_overlay.py           # GLFW + Dear ImGui + OpenGL textures
│   └── ui.py                    # Dark-theme UI components
├── ros2_ws/                     # ROS 2 packages (fogaze_manip, fogaze_rviz)
├── fogaze_arm.sh                # Launch arm pipeline + app together
├── fogaze.sh                    # Launch app + RViz
└── requirements.txt
```

### ROS 2 topics

| Topic | Type | Direction | Content |
|-------|------|-----------|---------|
| `/detections/markers` | `MarkerArray` | published | 3D object markers + labels |
| `fogaze/objects` | `String` (JSON) | published | Per-frame object list (pose, size, graspable) |
| `fogaze/pickup` | `String` (JSON) | published on blink | Focused-object pickup request |
| `fogaze/pick_pose` | `PoseStamped` | planner → arm | Validated grasp pose |
| `fogaze/pickup_status` | `String` | arm → app | Pick progress / result |

The ROS 2 packages live in `ros2_ws/src/` (`fogaze_manip` for planning/execution,
`fogaze_rviz` for visualization).

---

## Notes

- On **Wayland**, FoGaze forces `QT_QPA_PLATFORM=xcb`.
- RViz launched from a VS Code *snap* terminal needs the env de-snapped; both
  `fogaze*.sh` scripts handle this and the `LD_PRELOAD` libpthread fix automatically.
- The PrimeSense USB device is exclusive — only one process can open it, which is
  why `main.py` publishes ROS topics directly rather than via a separate node.
