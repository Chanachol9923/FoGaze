# FoGaze — Two-Camera Gaze + Object Tracking

Determine **which real-world object or area** a user is looking at, using two USB cameras and a gaze-estimation ML model.

## Architecture

```
Face camera  ──►  EyeTrax GazeEstimator  ──►  Gaze (x, y) in scene frame
                                                     │
Scene camera ──►  YOLO object detection   ──►  Bounding boxes
                                                     │
                                          Check: which bbox contains gaze point?
                                                     │
                                              Focused object
```

- **Face camera** — captures the user's face; MediaPipe FaceLandmarker extracts eye/head features
- **Scene camera** — captures the real-world view; YOLOv8 detects objects
- **Gaze prediction** — trained Ridge regression model maps face features → scene-frame coordinates
- **Object focus** — the object whose bounding box contains the predicted gaze point is considered "focused"

## Features

- **Grid calibration** — fixed 3×3 grid of circled targets overlaid on the scene camera feed; user looks at each real-world point and presses ENTER to capture (BACKSPACE to undo, ESC to finish)
- **Dark-theme UI** — animated gaze cursor with trail, YOLO bounding boxes with focus highlighting, top bar with FPS, HUD info panel, face PIP display
- **Gaze smoothing** — Kalman filter + EMA (or KDE / raw) for stable cursor movement
- **Two-camera setup** — independent face and scene cameras via `--face-camera` / `--scene-camera` flags
- **Re-calibration** — press `c` at any time to recalibrate without restarting

## Requirements

- Linux (developed on Ubuntu 22.04 / Gnome + XWayland)
- 2 USB cameras (one pointed at face, one pointed at scene)
- Python 3.10+
- [EyeTrax](https://github.com/ck-zhang/EyeTrax) package (must be installed separately)

## Installation

```bash
# 1. Install EyeTrax from source
git clone https://github.com/ck-zhang/EyeTrax /tmp/EyeTrax
pip install -e /tmp/EyeTrax

# 2. Install FoGaze dependencies
pip install -r requirements.txt

# 3. Run
python3 main.py
```

On **first run**, the program will scan cameras and prompt you to select face and scene camera indices.

## Usage

```bash
python3 main.py \
    --face-camera 0 \
    --scene-camera 2 \
    --model ridge \
    --filter kalman_ema
```

### Controls

| Key | Action |
|-----|--------|
| `ENTER` | Capture current calibration point |
| `BACKSPACE` | Undo last calibration point |
| `ESC` | Finish calibration / Quit |
| `c` | Re-calibrate gaze (while running) |
| `f` | Toggle fullscreen |

### Calibration

1. **Face detection** — look at the face camera for 2 seconds (countdown arc)
2. **Grid capture** — a circle appears on the scene camera feed at each grid position; **look at the real-world area the circle covers** and press ENTER
3. Repeat for all 9 grid points
4. Press ESC to finish and train the model
5. Press SPACEBAR to enter the main program

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--face-camera` | auto-detect | Face camera index |
| `--scene-camera` | auto-detect | Scene camera index |
| `--model` | `ridge` | EyeTrax ML model |
| `--filter` | `kalman_ema` | Smoothing filter (`kalman`, `kalman_ema`, `kde`, `none`) |
| `--ema-alpha` | `0.8` | EMA smoothing factor |
| `--kde-confidence` | `0.5` | KDE confidence threshold |
| `--headless` | — | Run without display window |
| `--reset-model` | — | Delete saved model and exit |
| `--detection-interval` | `3` | YOLO detection interval (frames) |

## Project structure

```
FoGaze/
├── main.py                   # Entry point — calibration + main loop
├── modules/
│   ├── __init__.py
│   ├── ui.py                 # Dark-theme UI components (TopBar, GazeCursor, PIPDisplay, HUDInfo)
│   └── object_detector.py    # YOLO object detection wrapper
├── requirements.txt
└── README.md
```

## Notes

- On **Wayland**, the program forces `QT_QPA_PLATFORM=xcb` because the Qt Wayland plugin is unavailable
- Calibrated model is saved to `~/.cache/fogaze3/eyetrax_model.pkl`
- Scene camera resolution determines the coordinate space for calibration and prediction
