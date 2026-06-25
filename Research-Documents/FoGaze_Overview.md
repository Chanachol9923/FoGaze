# FoGaze — Full Project Overview for AI Assistants

## TL;DR
Dual-camera (face webcam + PrimeSense) gaze tracking + YOLO object detection + triple-blink robot pickup trigger + Rviz 3D visualization. Built with Python, OpenCV, MediaPipe, EyeTrax, Ultralytics YOLO, OpenNI2, Dear ImGui, ROS 2 Humble.

---

## Architecture (One File)
**Everything runs in a single `main.py` (2027 lines)** — no separate ROS nodes for perception. PrimeSense USB can only be opened by one process, so main.py handles:
- Face camera read + EyeTrax gaze estimation
- PrimeSense color+depth read (OpenNI2)
- YOLO object detection
- ROS publishing (markers, pickup triggers, object list)
- TF broadcasting (`map`→`fogaze_base`)
- ImGui UI (GLFW+OpenGL)

ROS 2 packages under `ros2_ws/` are downstream consumers of main.py's published topics — they handle arm planning/execution.

---

## Hardware
| Device | Role | Resolution | FPS | Interface |
|--------|------|-----------|-----|-----------|
| USB webcam | Face tracking + gaze | 640×480 MJPG | 30 | OpenCV |
| PrimeSense (PS1080) | Scene color + depth | 640×480 RGB888 + 16-bit mm | 30 | OpenNI2 (libopenni2-0) |

---

## Data Flow (Runtime)
```
Face Cam ──cv2.flip→ undistort → face_lock crop
  └→ GazeEstimator.extract_features() → predict() → smoother → (gx, gy)
       └→ FaceLock updates from MediaPipe landmarks (zoom-to-face)

PrimeSense
  ├── Color stream (RGB888) → cv2.COLOR_RGB2BGR → cv2.flip(,1)
  │     └→ YOLO detect → bboxes
  │     └→ _ros_publish_markers() → /detections/markers (MarkerArray, frame=map)
  │     └→ scene_publisher (via fogaze/objects JSON)
  └── Depth stream (16-bit mm) → cv2.flip(,1) → apply X/Y offset
        └→ depth_at_bbox() for each detection → cm
        └→ colormap for preview

Gaze (gx,gy) + Detections:
  focused = detection whose bbox contains (gx, gy)
  nearest_3d = _nearest_by_depth(focused, ...)  # real 3D Euclidean
  2d_fallback = _get_relation(focused, ...)      # In/On/Under/Next To/Near/Behind

TripleBlinkDetector:
  3 blinks within 1.5s → TTS + ROS pickup /fogaze/pickup (JSON)
    - If focused object → class, pose (3D), size, graspable flag
    - Graspability: distance 0.2-0.85m, width < 8cm (Franka Panda)
```

---

## UI (Dear ImGui + GLFW)
- **MainMenu** (268px left panel): Status card (FPS/Tracker/Cal/Zone/Focus/Relation/Depth/Gaze/Camera) + buttons (Re-center, Re-calibrate, Calibrate Cameras) + Depth toggle + Depth settings sliders (Off X/Y, Min/Max mm, Colormap) + Stereo settings + Face preview + Eye L/R side-by-side preview + Depth colormap preview
- **Scene view**: Right of panel, letterboxed. Overlays: dim, depth colormap, YOLO bboxes (color by graspability), 9-zone grid (22/56/22 col, 22/73/5 row), gaze cursor (red trail+glow), focus indicator line, relationship text
- **Calibration panels**: Bottom-left instructions, progress bars, pulsing target, green flash+beep on capture
- Key events via `gui.was_key_pressed()` chain

---

## Camera Calibration
### Gaze calibration (9-point grid)
1. Guide screen → ENTER
2. Face detection → 2s arc countdown
3. 9 targets (col:22/56/22, row:22/73/5):
   - Press ENTER → 250 fast frames with head rotation + lean
   - BACKSPACE undo, ESC cancel
4. Train Ridge regressor → outlier rejection (drop worst 10%) → refit
5. Validation: 5 auto-test points → grade GOOD/OK/POOR
6. Model saved to `~/.cache/fogaze3/eyetrax_model.pkl`

### Camera lens calibration (chessboard)
- Face camera only (PrimeSense has no chessboard cal)
- 9×6 internal corners, ≥10 samples → `~/.cache/fogaze3/calib/cam{N}.json`
- Auto-default `k1=-0.30, k2=0.10` if no chessboard cal

### Drift re-center (G key)
- 1-point: look at center dot → 0.6s dwell + 1.2s collect → median offset applied

---

## PrimeSense Depth (`modules/depth_estimator.py`)
- OpenNI2: `openni2.initialize()` → `Device.open_file(uri)` → depth + color streams
- Async worker thread: read frames → apply X/Y offset (sliders) → store with lock
- Color: RGB888 → BGR → flip(,1)
- Depth: 16-bit mm → flip(,1) → offset via warpAffine
- `depth_at_bbox()`: clip min/max mm → mean → return cm (divide by 10)
- `colormap()`: normalize → apply colormap (15 OpenCV options)
- `close()`: stop streams → close device → `openni2.unload()`

## Stereo Depth Fallback (`modules/stereo_depth_estimator.py`)
- Two USB cameras → StereoSGBM → `depth = focal_px * baseline / disparity`
- Live-tunable: focal_px, baseline_cm, num_disp, block_size

## Object Detector (`modules/object_detector.py`)
- Ultralytics YOLO, configurable confidence/imgsz/detection_interval
- Frame-skip cache: every N frames runs inference, returns cached otherwise

## Face Lock (`modules/face_lock.py`)
- Zoom face region for sharper MediaPipe landmarks
- EMA-smoothed ROI (α=0.6), generous padding (0.7)
- Miss counter (max 8) → release lock
- Wrapped inside `gaze_estimator.extract_features()` — transparent to callers

## Triple Blink (`TripleBlinkDetector`)
- Window: 1.5s, threshold: 3 blinks
- Edge-triggered on blink→open transition
- Uses `last_focus` snapshot (blink suppresses gaze) — speaks object name even during blink
- Zone phrase + relation phrase + depth → TTS (pyttsx3)

---

## ROS 2 Integration
### Published by main.py (built-in, no separate node)
| Topic | Type | Frame | Rate | Content |
|-------|------|-------|------|---------|
| `/detections/markers` | MarkerArray | `map` | ~30fps | 3D cubes + TEXT_VIEW_FACING labels per YOLO det |
| `fogaze/objects` | String (JSON) | — | ~30fps | Per-frame list: class, pose, size, graspable, reason |
| `fogaze/pickup` | String (JSON) | — | on blink | Target: class, confidence, pose, size, graspable, reason |

### TF broadcast
- `map` → `fogaze_base` (identity, 10Hz) — so rviz2 accepts `map` as fixed frame

### Downstream consumers (ROS 2 packages)
```
pickup_planner (fogaze_manip):
  sub: fogaze/pickup → re-validate graspability → TF camera→arm base
  pub: fogaze/pick_pose (PoseStamped) + fogaze/pickup_status (String)

scene_publisher (fogaze_manip):
  sub: fogaze/objects → every object → cylinder CollisionObject in arm frame
  pub: collision_object (moveit_msgs/CollisionObject) @ 4Hz
  Pauses during pick execution (subs fogaze/pickup_status)

mock_arm_executor (fogaze_manip):
  sub: fogaze/pick_pose → animates gripper marker through approach→descend→grasp→lift
  pub: fogaze/arm_marker + fogaze/pickup_status

moveit_pick_executor (fogaze_manip):
  sub: fogaze/pick_pose → MoveIt2: approach→descend→grasp→lift (Panda)
  Uses pymoveit2
```

### Rviz config
- Fixed frame: `map`
- Grid display (default)
- MarkerArray display (topic `/detections/markers`)
- Default orbit view: distance 1.5, focal [0,0,0.5]

### Launch (`fogaze.sh`)
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 main.py &  sleep 1
LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 rviz2 -d <config> &
```
`LD_PRELOAD` fixes snap rviz2 crash (`__libc_pthread_init` missing in snap core20's glibc 2.31 vs. system glibc 2.35).

---

## Modules File Map
| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 2027 | Entry point: cal + main loop + ROS + UI |
| `gui_overlay.py` | 411 | GLFW + ImGui + OpenGL textures |
| `ui.py` | 319 | Dark theme components |
| `depth_estimator.py` | 183 | PrimeSense OpenNI2 |
| `stereo_depth_estimator.py` | 184 | Stereo SGBM fallback |
| `mock_depth_estimator.py` | 101 | Simulated depth |
| `mock_scene.py` | 242 | Procedural test scene |
| `object_detector.py` | 93 | YOLO wrapper |
| `camera_calibrator.py` | 128 | Chessboard lens cal |
| `face_lock.py` | 72 | Face zoom tracking |
| `grasp.py` | 110 | Geometry + grasp rules |
| `gaze_tracker.py` | 714 | Legacy MediaPipe gaze (not actively used) |
| `calibrator.py` | 356 | Legacy 7D Gaussian cal |
| `calibrator_sklearn.py` | 428 | Legacy sklearn cal |
| `eyetrax_features.py` | 157 | 486D feature extraction |
| `filters.py` | 325 | Smoothing filters |
| `fogaze.sh` | 31 | Launch script |

---

## Key Known Issues
1. **rviz2 "Frame [map] does not exist"** — Fixed frame `map` must be in TF tree. Currently publishing `map`→`fogaze_base` identity at 10Hz. If still broken, try `tf2_ros.StaticTransformBroadcaster` or change rviz fixed frame.
2. **rviz2 snap crash** — Requires `LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0`
3. **PrimeSense USB exclusive** — Only one process can open the device. Main.py publishes directly; no separate ROS node possible.
4. **3D marker positions** — Hardcoded intrinsics (FX=FY=532, CX/CY from frame). May need actual PrimeSense calibration for accurate poses.
5. **Wayland/XWayland** — Rviz view controls may not work properly under Wayland. Force `QT_QPA_PLATFORM=xcb`.

---

## How to Run
```bash
# Full pipeline with Rviz
bash fogaze.sh

# Standalone (no ROS, no Rviz)
python3 main.py --depth-source primesense

# Simulation mode (no hardware)
python3 main.py --depth-source sim

# Stereo mode
python3 main.py --depth-source stereo --stereo-left 2 --stereo-right 4

# With ROS (YOLO markers + pickup triggers published)
python3 main.py --ros --depth-source primesense
```
