#!/usr/bin/env python3
"""
FoGaze — Two-camera gaze + object tracking.

Architecture:
  Face camera  → EyeTrax GazeEstimator (gaze prediction)
  Scene camera → YOLO object detection

Goal: determine which object the user is looking at.
"""

import os
# Force XCB Qt backend before any cv2 import touches Qt
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eyetrax.gaze import GazeEstimator
from eyetrax.filters import (
    KalmanEMASmoother,
    KDESmoother,
    NoSmoother,
    make_kalman,
)
from eyetrax.utils.screen import get_screen_size

from modules.object_detector import ObjectDetector
from modules.depth_estimator import DepthEstimator
from modules.ui import draw_text_stroke
from modules.ui import Theme, TopBar, GazeCursor, HUDInfo


class DoubleBlinkDetector:
    def __init__(self, window=1.2):
        self.window = window
        self._times = []
        self._was = False

    def update(self, blinking, now=None):
        if now is None:
            now = time.time()
        triggered = False
        if not blinking and self._was:
            self._times.append(now)
            cutoff = now - self.window
            self._times = [t for t in self._times if t >= cutoff]
            if len(self._times) >= 2:
                self._times = []
                triggered = True
        self._was = blinking
        return triggered


class SpeechOutput:
    def __init__(self, cooldown=2.0):
        self._engine = None
        self._last = 0.0
        self._cooldown = cooldown

    def _init(self):
        if self._engine is not None:
            return
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 150)

    def speak(self, text):
        now = time.time()
        if now - self._last < self._cooldown:
            return
        self._last = now
        self._init()
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        self._engine.say(text)
        self._engine.runAndWait()



ZONE_PHRASES = [
    "At My Upper Left Hand",
    "At My Upper Side",
    "At My Upper Right Hand",
    "At My Left Hand",
    "In Front of me",
    "At My Right Hand",
    "At My Lower Left Hand",
    "At My Lower Side",
    "At My Lower Right Hand",
]

THING_THRESHOLD = 0.2  # below this → say "That Thing" instead of class

def _zone_for(gx, gy, w, h):
    c1, c2 = int(w * 0.22), int(w * 0.78)
    r1, r2 = int(h * 0.22), int(h * 0.95)
    col = 0 if gx < c1 else (1 if gx < c2 else 2)
    row = 0 if gy < r1 else (1 if gy < r2 else 2)
    return row * 3 + col, (c1, c2, r1, r2)


def _get_relation(a, b, w, h):
    x1a, y1a, x2a, y2a = a["bbox"]
    x1b, y1b, x2b, y2b = b["bbox"]
    xi1 = max(x1a, x1b); yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b); yi2 = min(y2a, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    aa = (x2a - x1a) * (y2a - y1a)
    ab = (x2b - x1b) * (y2b - y1b)
    union = aa + ab - inter
    iou = inter / union if union > 0 else 0
    if iou > 0.15:
        return "In"
    cx_a, cy_a = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx_b, cy_b = (x1b + x2b) / 2, (y1b + y2b) / 2
    m = min(w, h) * 0.04
    if abs(y2a - y1b) < m and x1b <= cx_a <= x2b:
        return "On"
    if abs(y1a - y2b) < m and x1b <= cx_a <= x2b:
        return "Under"
    if y1a <= y2b and y2a >= y1b:
        if abs(x2a - x1b) < m * 2 or abs(x2b - x1a) < m * 2:
            return "Next To"
    dist = ((cx_a - cx_b)**2 + (cy_a - cy_b)**2)**0.5
    th = ((x2a - x1a + x2b - x1b) / 4 + (y2a - y1a + y2b - y1b) / 4) * 0.7
    if dist < th:
        return "Near"
    if cy_a > cy_b + m and abs(cx_a - cx_b) < (x2a - x1a + x2b - x1b) / 2:
        return "Behind"
    return None

_CJK_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
_cjk_fonts = {}

def _cjk_font(size):
    if size not in _cjk_fonts:
        _cjk_fonts[size] = ImageFont.truetype(_CJK_FONT_PATH, size)
    return _cjk_fonts[size]


DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/fogaze3/eyetrax_model.pkl")


def _is_trained(m):
    return (m is not None and hasattr(m, 'scaler')
            and hasattr(m.scaler, 'mean_'))


def _scan_cameras(max_cam=10):
    avail = []
    for i in range(max_cam):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                avail.append(i)
            cap.release()
    return avail


def _pick_camera(avail, label):
    if not avail:
        print(f"No {label} found.")
        sys.exit(1)
    if len(avail) == 1:
        print(f"  {label}: camera {avail[0]}")
        return avail[0]
    while True:
        try:
            c = int(input(f"Select {label} {avail}: "))
            if c in avail:
                return c
        except (ValueError, EOFError):
            pass


def _wait_for_spacebar(sw, sh, message="Press SPACEBAR to start"):
    cv2.destroyAllWindows()
    win = "Ready"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        canvas = np.full((sh, sw, 3), Theme.BG_PRIMARY, dtype=np.uint8)
        size, _ = cv2.getTextSize(message, font, 1.5, 3)
        tx = (sw - size[0]) // 2
        ty = (sh + size[1]) // 2
        cv2.putText(canvas, message, (tx, ty), font, 1.5, Theme.ACCENT_GREEN, 3)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACEBAR
            break
        if key == 27:  # ESC
            break
    cv2.destroyWindow(win)


def _calibrate_two_cam(gaze_estimator, cap_face, cap_scene,
                       capture_frames=250, grid_cols=3, grid_rows=3):
    """Grid calibration: fixed circles on scene camera feed.

    User looks at the real-world area each circle covers.
    Auto-captures features during a short countdown per point.
    ESC to cancel.
    """
    ret, tmp = cap_scene.read()
    if not ret:
        print("[FoGaze] Cannot read scene camera for calibration.")
        return False

    cv2.destroyAllWindows()
    h_s, w_s = tmp.shape[:2]
    win = "Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Grid of target points (10% margin)
    xs = np.linspace(int(w_s * 0.1), int(w_s * 0.9), grid_cols, dtype=int)
    ys = np.linspace(int(h_s * 0.1), int(h_s * 0.9), grid_rows, dtype=int)
    targets = [(int(x), int(y)) for y in ys for x in xs]

    collected = []  # list of [scene_x, scene_y, features]

    # ── Step-by-step instructions (EN + JP, black bg) ─────────────────
    guide_items = [
        ("===== Calibration Guide =====", 24, (0, 255, 255)),
        ("", 0, None),
        ("Step 1  -  Face detection", 22, (0, 230, 0)),
        ("  Position your face in front of the face camera", 18, (220, 220, 220)),
        ("  顔カメラの前に座って顔を映してください", 18, (160, 160, 160)),
        ("", 0, None),
        ("Step 2  -  Look at the target & press ENTER", 22, (0, 230, 0)),
        ("  Look at the circled point, then press ENTER to capture", 18, (220, 220, 220)),
        ("  丸いターゲットを見て、ENTERを押して撮影", 18, (160, 160, 160)),
        ("", 0, None),
        ("Step 3  -  Rotate your head during capture", 22, (0, 230, 0)),
        ("  Keep your eyes on the target, slowly rotate your head", 18, (220, 220, 220)),
        ("  ターゲットを見たままゆっくり頭を動かしてください", 18, (160, 160, 160)),
        ("", 0, None),
        ("Step 4  -  Repeat for all 9 points", 22, (0, 230, 0)),
        ("  ENTER=capture  BACKSPACE=undo last point  ESC=finish", 18, (220, 220, 220)),
        ("  ENTER=撮影  BACKSPACE=戻る  ESC=終了", 18, (160, 160, 160)),
        ("", 0, None),
        ("Press ENTER to start", 26, (0, 255, 255)),
    ]
    while True:
        ret_s, frame_s = cap_scene.read()
        if not ret_s:
            continue
        canvas = np.zeros((h_s, w_s, 3), dtype=np.uint8)
        y = 50
        for txt, size, color in guide_items:
            if not txt:
                y += 12
                continue
            if any(ord(c) > 127 for c in txt):
                pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((80, y), txt, font=_cjk_font(size), fill=color)
                canvas[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(canvas, txt, (80, y), font, size / 18,
                            color, 2)
            y += size + 8
        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        if k == 27:
            cv2.destroyWindow(win)
            return False

    # ── Wait for face once ────────────────────────────────────────────
    fd_start = None
    while True:
        ret_s, frame_s = cap_scene.read()
        ret_f, frame_f = cap_face.read()
        if not ret_s or not ret_f:
            continue
        frame_face = cv2.flip(frame_f, 1)
        f, blink = gaze_estimator.extract_features(frame_face)
        face = f is not None and not blink
        canvas = frame_s.copy()
        now = time.time()
        if face:
            if fd_start is None:
                fd_start = now
            if now - fd_start >= 2.0:
                break
            t = (now - fd_start) / 2.0
            ang = int(360 * (1 - t * t * (3 - 2 * t)))
            cv2.ellipse(canvas, (w_s // 2, h_s // 2), (50, 50),
                        0, -90, -90 + ang, Theme.ACCENT_GREEN, -1)
        else:
            fd_start = None
            txt = "Face not detected"
            size, _ = cv2.getTextSize(txt, font, 2, 3)
            tx = (w_s - size[0]) // 2
            ty = (h_s + size[1]) // 2
            cv2.putText(canvas, txt, (tx, ty), font, 2, Theme.ACCENT_RED, 3)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            cv2.destroyWindow(win)
            return False

    # ── Grid calibration loop (ENTER to capture each point) ──────────
    for idx, (tx, ty) in enumerate(targets):
        while True:
            ret_s, frame_s = cap_scene.read()
            ret_f, frame_f = cap_face.read()
            if not ret_s or not ret_f:
                continue
            canvas = frame_s.copy()

            # Target circle at this grid point
            cv2.circle(canvas, (tx, ty), 30, Theme.ACCENT_CYAN, -1)
            cv2.circle(canvas, (tx, ty), 36, (255, 255, 255), 2)
            cv2.line(canvas, (tx - 20, ty), (tx + 20, ty), (255, 255, 255), 1)
            cv2.line(canvas, (tx, ty - 20), (tx, ty + 20), (255, 255, 255), 1)

            # Previously captured points
            for pt in collected:
                px, py = pt[0], pt[1]
                cv2.circle(canvas, (px, py), 8, Theme.ACCENT_GREEN, -1)
                cv2.circle(canvas, (px, py), 12, (255, 255, 255), 1)

            # Info overlay
            n_captured = len(collected)
            lines = [
                f"Point {idx + 1} / {len(targets)}   |   Captured: {n_captured}",
                "ENTER: capture this point  |  BACKSPACE: undo  |  ESC: done",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(canvas, txt, (50, 50 + i * 35),
                            font, 0.8, (255, 255, 255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER — capture
                ret_f, frame_f = cap_face.read()
                if not ret_f:
                    continue
                for _ in range(capture_frames):
                    ret_f2, frame_f2 = cap_face.read()
                    if not ret_f2:
                        continue
                    ft, blink = gaze_estimator.extract_features(
                        cv2.flip(frame_f2, 1))
                    if ft is not None and not blink:
                        collected.append([tx, ty, ft])
                # Green flash + beep × 2
                for _ in range(2):
                    ret_s2, fb = cap_scene.read()
                    if ret_s2:
                        fb2 = fb.copy()
                        cv2.rectangle(fb2, (0, 0), (w_s, h_s),
                                      (0, 230, 0), -1)
                        cv2.putText(fb2, "Done!", (w_s // 2 - 80, h_s // 2),
                                    font, 2, (255, 255, 255), 3)
                        cv2.imshow(win, fb2)
                        cv2.waitKey(1)
                    # Beep via multiple methods
                    print('\a', end='', flush=True)
                    os.system('echo -ne "\\a" > /dev/tty 2>/dev/null &')
                    cv2.waitKey(200)
                break  # move to next point

            if key == 8:  # BACKSPACE — undo last point
                if collected:
                    removed = collected.pop()
                    print(f"[FoGaze] Removed point ({removed[0]}, {removed[1]})")

            if key == 27:  # ESC — cancel
                cv2.destroyWindow(win)
                return False

    cv2.destroyWindow(win)

    # ── Train ──────────────────────────────────────────────────────────
    if len(collected) < 3:
        print(f"[FoGaze] Too few samples ({len(collected)}), cannot calibrate.")
        return False

    feats = np.array([c[2] for c in collected])  # (N, 348)
    targs = np.array([[c[0], c[1]] for c in collected])  # (N, 2)
    print(f"[FoGaze] Training on {len(feats)} samples...")
    gaze_estimator.train(feats, targs)
    print("[FoGaze] Calibration complete.")
    return True


def _calibrate_depth(depth_estimator, cap_scene, sw, sh):
    """Interactive depth calibration.

    User places an object/hand at known distances; records DA2V norm values
    and fits a linear model: cm = slope * norm + intercept.
    """
    win = "Depth Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    distances = [30, 50, 100, 150, 200]
    samples = []  # (distance_cm, norm)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Instruction screen
    guide = [
        ("===== Depth Calibration =====", (0, 255, 255)),
        ("", None),
        ("Place your hand or an object at each distance shown.", (220, 220, 220)),
        ("手や物体を表示された距離に置いてください。", (160, 160, 160)),
        ("", None),
        ("Press ENTER to capture each distance", (0, 230, 0)),
        ("ENTERで各距離を撮影", (0, 230, 0)),
        ("", None),
        ("ESC to cancel", (220, 100, 100)),
        ("ESC=キャンセル", (160, 80, 80)),
    ]
    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        y = 50
        for txt, color in guide:
            if not txt:
                y += 12
                continue
            if any(ord(c) > 127 for c in txt):
                from PIL import Image, ImageDraw
                pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((80, y), txt, font=_cjk_font(22))
                canvas[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(canvas, txt, (80, y), font, 1.0, color, 2)
            y += 34
        cv2.putText(canvas, "Press ENTER to begin",
                    (80, y + 20), font, 1.2, (0, 255, 255), 2)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        if k == 27:
            cv2.destroyWindow(win)
            return False

    # Capture for each distance
    for i, dist in enumerate(distances):
        while True:
            ret, frame = cap_scene.read()
            if not ret:
                continue
            canvas = frame.copy()
            cv2.putText(canvas, f"Distance: {dist} cm   ({i + 1}/{len(distances)})",
                        (50, 50), font, 1.2, (0, 255, 255), 3)
            cv2.putText(canvas, "Place object/hand at this distance, then press ENTER",
                        (50, 100), font, 0.8, (220, 220, 220), 2)
            cv2.putText(canvas, f"Press ENTER to capture  |  ESC to cancel",
                        (50, 140), font, 0.7, (160, 160, 160), 2)

            # Center ROI indicator
            cx, cy = sw // 2, sh // 2
            r = 30
            cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r),
                          (0, 255, 0), 2)
            cv2.putText(canvas, "CENTER",
                        (cx - 40, cy - 40), font, 0.6, (0, 255, 0), 2)

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == 13:
                ret, frame = cap_scene.read()
                if not ret:
                    continue
                # Run depth synchronously
                depth = depth_estimator.estimate_sync(frame)
                if depth is None:
                    continue
                center_roi = depth[cy - r:cy + r, cx - r:cx + r]
                if center_roi.size == 0:
                    continue
                scene_min = float(depth.min())
                scene_max = float(depth.max())
                if scene_max == scene_min:
                    continue
                avg_depth = float(center_roi.mean())
                norm = (avg_depth - scene_min) / (scene_max - scene_min)
                norm = np.clip(norm, 0, 1)
                samples.append((dist, norm))
                print(f"[CalibrateDepth] {dist}cm -> norm={norm:.4f}")

                # Short flash
                fb = frame.copy()
                cv2.rectangle(fb, (0, 0), (sw, sh), (0, 180, 0), -1)
                cv2.putText(fb, f"{dist}cm captured!", (sw // 2 - 120, sh // 2),
                            font, 1.5, (255, 255, 255), 3)
                cv2.imshow(win, fb)
                cv2.waitKey(400)
                break

            if k == 27:
                cv2.destroyWindow(win)
                return False

    cv2.destroyWindow(win)

    if len(samples) < 3:
        print("[CalibrateDepth] Too few samples.")
        return False

    # Fit linear regression: dist = slope * norm + intercept
    norms = np.array([s[1] for s in samples])
    cm = np.array([s[0] for s in samples])
    A = np.vstack([norms, np.ones(len(norms))]).T
    slope, intercept = np.linalg.lstsq(A, cm, rcond=None)[0]

    print(f"[CalibrateDepth] Fit: cm = {slope:.3f} * norm + {intercept:.1f}")
    depth_estimator.save_cal(slope, intercept)

    for dist, norm in samples:
        pred = slope * norm + intercept
        print(f"  {dist}cm -> norm={norm:.4f} -> pred={pred:.1f}cm (err={pred - dist:.1f})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="FoGaze — two-camera gaze + object tracking"
    )

    parser.add_argument("--face-camera", type=int, default=None,
                        help="Camera index for face/gaze tracking")
    parser.add_argument("--scene-camera", type=int, default=None,
                        help="Camera index for scene/object detection")
    parser.add_argument("--model", default="ridge",
                        help="EyeTrax ML model")
    parser.add_argument("--model-file", default=None,
                        help="Path to trained gaze model")
    parser.add_argument("--reset-model", action="store_true",
                        help="Delete saved gaze model and exit")
    parser.add_argument("--filter",
                        choices=["kalman", "kalman_ema", "kde", "none"],
                        default="kalman_ema",
                        help="Gaze smoothing filter")
    parser.add_argument("--ema-alpha", type=float, default=0.8,
                        help="EMA smoothing (0=off, 1=max smooth)")
    parser.add_argument("--kde-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-model",
                        default=str(Path(__file__).resolve().parent
                                    / "models" / "yolov8n.pt"),
                        help="YOLO weights path")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="YOLO confidence threshold")
    parser.add_argument("--detection-interval", type=int, default=3,
                        help="YOLO inference every N frames")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="YOLO inference size")
    parser.add_argument("--reset-depth-cal", action="store_true",
                        help="Delete depth calibration and exit")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")

    args = parser.parse_args()

    # ── Camera selection ──────────────────────────────────────────────
    avail = _scan_cameras()
    face_cam = (args.face_camera
                if args.face_camera is not None
                else _pick_camera(avail, "face camera"))
    scene_cam = (args.scene_camera
                 if args.scene_camera is not None
                 else _pick_camera(
                     [c for c in avail if c != face_cam] or avail,
                     "scene camera"))
    print(f"[FoGaze] Face cam={face_cam}  Scene cam={scene_cam}")

    # ── Model file ────────────────────────────────────────────────────
    model_path = args.model_file or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if args.reset_model:
        if os.path.isfile(model_path):
            os.remove(model_path)
            print(f"[FoGaze] Deleted {model_path}")
        else:
            print("[FoGaze] No model file found.")
        return

    if args.reset_depth_cal:
        de = DepthEstimator(device="cuda")
        de.reset_cal()
        return

    # ── Screen size ───────────────────────────────────────────────────
    sw, sh = get_screen_size()
    print(f"[FoGaze] Screen: {sw}x{sh}")

    # ── GazeEstimator ─────────────────────────────────────────────────
    print(f"[FoGaze] Creating GazeEstimator (model={args.model}) ...")
    gaze_estimator = GazeEstimator(model_name=args.model)

    # Load or calibrate
    if os.path.isfile(model_path):
        try:
            gaze_estimator.load_model(model_path)
            print(f"[FoGaze] Loaded model from {model_path}")
        except Exception as e:
            print(f"[FoGaze] Failed to load model ({e}), will calibrate.")

    if not _is_trained(gaze_estimator.model):
        print("[FoGaze] No valid model — starting calibration")
        print("[FoGaze] Look at the circled area on scene camera. Then rotate head slowly at center.")

        # Custom two-camera calibration: scene cam shows targets, face cam captures features
        cap_tmp_face = cv2.VideoCapture(face_cam)
        cap_tmp_scene = cv2.VideoCapture(scene_cam)
        if not cap_tmp_face.isOpened() or not cap_tmp_scene.isOpened():
            raise RuntimeError("Cannot open cameras for calibration")
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_tmp_scene.set(cv2.CAP_PROP_FRAME_WIDTH, sw)
        cap_tmp_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)

        ok = _calibrate_two_cam(
            gaze_estimator, cap_tmp_face, cap_tmp_scene,
        )
        cap_tmp_face.release()
        cap_tmp_scene.release()

        if not ok:
            print("[FoGaze] Calibration failed or was cancelled.")
            return

        try:
            gaze_estimator.save_model(model_path)
            print(f"[FoGaze] Model saved to {model_path}")
        except Exception as e:
            print(f"[FoGaze] Warning: could not save model ({e})")

        _wait_for_spacebar(sw, sh, "Calibration done \u2014 press SPACEBAR to start")

    # ── Filter (create before opening cameras) ────────────────────────────
    if args.filter == "kalman_ema":
        kf = make_kalman(process_var=1.0, measurement_var=5.0)
        smoother = KalmanEMASmoother(kf, ema_alpha=args.ema_alpha)
    elif args.filter == "kalman":
        kf = make_kalman(process_var=1.0, measurement_var=5.0)
        smoother = KalmanEMASmoother(kf, ema_alpha=0.0)
    elif args.filter == "kde":
        smoother = KDESmoother(sw, sh, confidence=args.kde_confidence)
    else:
        smoother = NoSmoother()

    # ── Open cameras ──────────────────────────────────────────────────────
    cap_face = cv2.VideoCapture(face_cam)
    cap_scene = cv2.VideoCapture(scene_cam)

    if not cap_face.isOpened():
        raise RuntimeError(f"Cannot open face camera {face_cam}")
    if not cap_scene.isOpened():
        raise RuntimeError(f"Cannot open scene camera {scene_cam}")

    cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_scene.set(cv2.CAP_PROP_FRAME_WIDTH, sw)
    cap_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)

    for _ in range(10):
        cap_face.read()
        cap_scene.read()

    # ── Object detector (scene camera) ────────────────────────────────
    detector = ObjectDetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        imgsz=args.imgsz,
    )
    detector.set_detection_interval(args.detection_interval)

    # ── Depth estimator ───────────────────────────────────────────────
    depth_estimator = DepthEstimator(device="cuda")
    show_depth = False

    # ── UI components ─────────────────────────────────────────────────
    topbar = TopBar()
    cursor = GazeCursor()
    hud = HUDInfo()

    # ── State ─────────────────────────────────────────────────────────
    gaze_active = False
    focused = None
    cal_notified = False

    blink_detector = DoubleBlinkDetector()
    speech = SpeechOutput()
    last_valid_gx = sw // 2
    last_valid_gy = sh // 2

    fps_n = 0
    fps_t0 = time.perf_counter()
    fps_val = 0

    depth_map = None

    help_t = time.time()
    win_name = "FoGaze — Gaze + Object Tracking"

    # Clear any leftover windows (tune, calibration, etc.)
    cv2.destroyAllWindows()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    face_win = "Face Camera"
    cv2.namedWindow(face_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(face_win, 320, 240)
    cv2.moveWindow(face_win, 50, 50)

    try:
        while True:
            # ── Read cameras ──────────────────────────────────────────
            ret_face, frame_face = cap_face.read()
            ret_scene, frame_scene = cap_scene.read()
            if not ret_face or not ret_scene:
                break

            frame_face = cv2.flip(frame_face, 1)
            h_scene, w_scene = frame_scene.shape[:2]

            # FPS
            fps_n += 1
            if fps_n >= 15:
                now = time.perf_counter()
                dt = now - fps_t0
                fps_val = int(fps_n / dt) if dt > 0 else 0
                fps_n, fps_t0 = 0, now
            topbar.set_fps(fps_val)

            # ── Gaze estimation (face camera) ─────────────────────────
            features, blink_detected = gaze_estimator.extract_features(
                frame_face
            )

            gx, gy = last_valid_gx, last_valid_gy
            gaze_active = False

            if features is not None and not blink_detected:
                try:
                    pred = gaze_estimator.predict(np.array([features]))[0]
                    gx, gy = float(pred[0]), float(pred[1])
                    gaze_active = True
                except Exception:
                    pass

            if gaze_active:
                gx, gy = smoother.step(gx, gy)
                last_valid_gx, last_valid_gy = int(gx), int(gy)

            gx_scene = int(gx)
            gy_scene = int(gy)

            cursor.update(gx_scene, gy_scene, gaze_active)

            # ── Object detection (scene camera) ───────────────────────
            detections = detector.detect(frame_scene)

            # ── Depth estimation (async, non-blocking) ──────────────────
            depth_map = depth_estimator.estimate(frame_scene)
            focused_depth = None
            focused_depth_cm = None

            focused = None
            if gaze_active:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    if x1 <= gx_scene <= x2 and y1 <= gy_scene <= y2:
                        focused = det
                        break
            if focused:
                x1, y1, x2, y2 = focused["bbox"]
                focused_depth = depth_estimator.depth_at_bbox(x1, y1, x2, y2)
                if focused_depth is not None:
                    focused_depth_cm = depth_estimator.depth_to_distance_cm(focused_depth)

            # ── Relationship detection (all pairs) ────────────────────
            relations = {}  # (id_a, id_b) → rel
            for i, da in enumerate(detections):
                for j, db in enumerate(detections):
                    if i >= j:
                        continue
                    rel = _get_relation(da, db, w_scene, h_scene)
                    if rel:
                        relations[(i, j)] = rel
            focused_rel = None
            if focused is not None:
                fi = detections.index(focused)
                for (i, j), rel in relations.items():
                    if i == fi:
                        focused_rel = (rel, detections[j])
                        break
                    if j == fi:
                        focused_rel = (rel, detections[i])
                        break

            # ── Double-blink → speech (zone + relation) ───────────────
            if blink_detector.update(blink_detected, time.time()):
                zi, _ = _zone_for(gx_scene, gy_scene, w_scene, h_scene)
                phrase = ZONE_PHRASES[zi]
                if focused:
                    name = (focused['class_name'] if
                            focused['confidence'] >= THING_THRESHOLD
                            else "That Thing")
                    if focused_rel:
                        speech.speak(
                            f"{name} {focused_rel[0]} "
                            f"{focused_rel[1]['class_name']} {phrase}")
                    else:
                        speech.speak(f"{name} {phrase}")
                else:
                    speech.speak(phrase)

            # ── Face camera display (separate window) ─────────────────
            face_display = frame_face.copy()
            if (features is not None
                    and hasattr(gaze_estimator, '_face_landmarker')):
                try:
                    import mediapipe as mp
                    rgb = np.ascontiguousarray(
                        cv2.cvtColor(face_display, cv2.COLOR_BGR2RGB)
                    )
                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb
                    )
                    ts_ms = int(time.time() * 1000)
                    result = gaze_estimator._face_landmarker.detect_for_video(
                        mp_img, ts_ms
                    )
                    if result and result.face_landmarks:
                        lm = result.face_landmarks[0]
                        h_f, w_f = face_display.shape[:2]
                        xs = [p.x for p in lm]
                        ys = [p.y for p in lm]
                        cx = (min(xs) + max(xs)) * 0.5 * w_f
                        cy = (min(ys) + max(ys)) * 0.5 * h_f
                        hhw = (max(xs) - min(xs)) * 0.7 * w_f
                        hhh = (max(ys) - min(ys)) * 0.7 * h_f
                        fx1 = int(max(0, cx - hhw))
                        fx2 = int(min(w_f, cx + hhw))
                        fy1 = int(max(0, cy - hhh))
                        fy2 = int(min(h_f, cy + hhh))
                        if fx2 > fx1 and fy2 > fy1:
                            cv2.rectangle(face_display, (fx1, fy1),
                                          (fx2, fy2), Theme.ACCENT_GREEN, 2)
                except Exception:
                    pass
            cv2.imshow(face_win, face_display)

            # ── Canvas (scene feed + overlays) ────────────────────────
            canvas = frame_scene.copy()

            # Dim overlay for better UI contrast
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w_scene, h_scene),
                          Theme.BG_PRIMARY, -1)
            cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

            # YOLO bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                is_focused = (focused is not None and focused is det)
                color = Theme.ACCENT_GREEN if is_focused else Theme.ACCENT_CYAN
                thickness = 3 if is_focused else 2
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                draw_text_stroke(
                    canvas,
                    f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, y1 - 8), scale=0.5, color=color,
                )

            # 3×3 zone grid (center zone larger)
            zi, (c1, c2, r1, r2) = _zone_for(
                gx_scene, gy_scene, w_scene, h_scene)
            cv2.line(canvas, (c1, 0), (c1, h_scene), (60, 60, 80), 1)
            cv2.line(canvas, (c2, 0), (c2, h_scene), (60, 60, 80), 1)
            cv2.line(canvas, (0, r1), (w_scene, r1), (60, 60, 80), 1)
            cv2.line(canvas, (0, r2), (w_scene, r2), (60, 60, 80), 1)
            draw_text_stroke(canvas, ZONE_PHRASES[zi],
                             (12, h_scene - 60), scale=0.5,
                             color=Theme.ACCENT_CYAN, thickness=1)
            # Relationship text + depth
            if focused and focused_rel:
                fname = (focused['class_name'] if
                         focused['confidence'] >= THING_THRESHOLD
                         else "That Thing")
                dist_txt = (f"(~{int(focused_depth_cm)}cm)"
                            if focused_depth_cm is not None else "")
                draw_text_stroke(
                    canvas,
                    f"{fname} {focused_rel[0]} {focused_rel[1]['class_name']} {dist_txt}",
                    (12, h_scene - 36), scale=0.5,
                    color=Theme.ACCENT_GREEN, thickness=1)

            # Focus indicator
            if focused:
                cx_f = (focused["bbox"][0] + focused["bbox"][2]) // 2
                cy_f = (focused["bbox"][1] + focused["bbox"][3]) // 2
                cv2.line(canvas, (gx_scene, gy_scene), (cx_f, cy_f),
                         Theme.ACCENT_GREEN, 1, cv2.LINE_AA)

            # Gaze cursor
            cursor.draw(canvas)

            # Top bar
            topbar.draw(canvas, w_scene)

            # Calibration notification
            if _is_trained(gaze_estimator.model) and not cal_notified:
                topbar.toast("Calibration ready", Theme.ACCENT_GREEN)
                cal_notified = True

            # HUD
            cal_status = "CAL" if _is_trained(gaze_estimator.model) else "UNCAL"
            zi, _ = _zone_for(gx_scene, gy_scene, w_scene, h_scene)
            fname = (focused['class_name'] if focused['confidence'] >= THING_THRESHOLD
                     else "That Thing") if focused else "--"
            rel_txt = (f"{fname} {focused_rel[0]} "
                       f"{focused_rel[1]['class_name']}"
                       ) if focused_rel else "--"
            depth_txt = (f"{int(focused_depth_cm)}cm"
                         if focused_depth_cm is not None else "--")
            hud.draw(canvas, [
                (f"Face cam:{face_cam}  Scene cam:{scene_cam}",
                 Theme.ACCENT_CYAN),
                (f"EyeTrax {args.model.upper()} | {args.filter.upper()} | {cal_status}",
                 Theme.ACCENT_GREEN if cal_status == "CAL"
                 else Theme.ACCENT_ORANGE),
                (f"Zone: {ZONE_PHRASES[zi]}",
                 Theme.ACCENT_CYAN),
                (f"Relation: {rel_txt}",
                 Theme.ACCENT_GREEN if focused_rel else Theme.TEXT_DIM),
                (f"Focus: {fname}",
                 Theme.ACCENT_GREEN if focused else Theme.TEXT_DIM),
                (f"Depth: {depth_txt}",
                 Theme.ACCENT_CYAN if focused_depth_cm is not None else Theme.TEXT_DIM),
                (f"Gaze: ({gx_scene}, {gy_scene})", Theme.TEXT_DIM),
            ])

            # Help hint
            if time.time() - help_t < 5:
                draw_text_stroke(
                    canvas,
                    "ESC: quit  |  c: re-calibrate  |  f: fullscreen  |  d: depth",
                    (w_scene // 2 - 250, h_scene - 40),
                    scale=0.45, color=Theme.TEXT_DIM,
                )

            # ── Depth overlay (small heatmap at top-right) ────────────
            if show_depth and depth_map is not None:
                dh = int(h_scene * 0.2)
                dw = int(dh * 4 / 3)
                depth_color = depth_estimator.colormap(depth_map)
                depth_thumb = cv2.resize(depth_color, (dw, dh))
                x_off = w_scene - dw - 10
                y_off = 66
                canvas[y_off:y_off+dh, x_off:x_off+dw] = depth_thumb
                cv2.rectangle(canvas, (x_off, y_off),
                              (x_off + dw, y_off + dh),
                              (255, 255, 255), 1)
                draw_text_stroke(canvas, "DEPTH",
                                 (x_off + 4, y_off + 14),
                                 scale=0.4, color=Theme.ACCENT_CYAN)

            # ── Display ───────────────────────────────────────────────
            if not args.headless:
                cv2.imshow(win_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[FoGaze] User quit.")
                break

            elif key == ord('c'):
                print("[FoGaze] Re-calibrating gaze...")
                cv2.destroyAllWindows()
                cap_face.release()
                cap_scene.release()

                cap_re_face = cv2.VideoCapture(face_cam)
                cap_re_scene = cv2.VideoCapture(scene_cam)
                if not cap_re_face.isOpened() or not cap_re_scene.isOpened():
                    print("[FoGaze] Failed to open camera for re-calibration.")
                    break
                cap_re_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_re_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap_re_scene.set(cv2.CAP_PROP_FRAME_WIDTH, sw)
                cap_re_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)

                try:
                    ok = _calibrate_two_cam(
                        gaze_estimator, cap_re_face, cap_re_scene,
                    )
                    if ok:
                        gaze_estimator.save_model(model_path)
                        print(f"[FoGaze] Model saved to {model_path}")
                        _wait_for_spacebar(sw, sh,
                                           "Re-calibration done \u2014 press SPACEBAR to resume")
                        cursor = GazeCursor()
                        cal_notified = False
                        topbar.toast("Re-calibrated!", Theme.ACCENT_GREEN)
                    else:
                        topbar.toast("Calibration cancelled", Theme.ACCENT_ORANGE)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[FoGaze] Calibration failed: {e}")
                    topbar.toast("Calibration failed", Theme.ACCENT_RED)

                cap_re_face.release()
                cap_re_scene.release()

                # Re-open main cameras
                cap_face = cv2.VideoCapture(face_cam)
                cap_scene = cv2.VideoCapture(scene_cam)
                if not cap_face.isOpened() or not cap_scene.isOpened():
                    print("[FoGaze] Failed to re-open camera.")
                    break
                cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap_scene.set(cv2.CAP_PROP_FRAME_WIDTH, sw)
                cap_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.namedWindow(face_win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(face_win, 320, 240)
                cv2.moveWindow(face_win, 50, 50)

            elif key == ord('f'):
                fs = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
                new_fs = not (fs == cv2.WINDOW_FULLSCREEN)
                cv2.setWindowProperty(
                    win_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if new_fs else cv2.WINDOW_NORMAL,
                )

            elif key == ord('d'):
                show_depth = not show_depth
                topbar.toast(
                    f"Depth overlay {'ON' if show_depth else 'OFF'}",
                    Theme.ACCENT_GREEN if show_depth else Theme.ACCENT_ORANGE,
                )

            elif key == ord('C'):
                topbar.toast("Depth calibration...", Theme.ACCENT_CYAN)
                cv2.destroyWindow(win_name)
                cv2.destroyWindow(face_win)
                ok = _calibrate_depth(depth_estimator, cap_scene, sw, sh)
                if ok:
                    topbar.toast("Depth calibrated!", Theme.ACCENT_GREEN)
                else:
                    topbar.toast("Depth cal. cancelled", Theme.ACCENT_ORANGE)
                # Re-create windows
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.namedWindow(face_win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(face_win, 320, 240)
                cv2.moveWindow(face_win, 50, 50)

    except KeyboardInterrupt:
        print("\n[FoGaze] Interrupted.")
    finally:
        cap_face.release()
        cap_scene.release()
        cv2.destroyAllWindows()
        gaze_estimator.close()
        print("[FoGaze] Shutdown.")


if __name__ == "__main__":
    main()
