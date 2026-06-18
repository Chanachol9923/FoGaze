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
import glfw
import imgui
import numpy as np

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
from modules.gui_overlay import GUIOverlay
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


def _draw_main_menu(gui, fps_val=0, show_depth=False,
                     zone_txt="--", focus_txt="--", depth_txt="--",
                     cal_mode=None, cal_step=None, cal_progress=None,
                     face_tex_id=None, depth_tex_id=None):
    """Left-side MainMenu panel. Returns action string or None."""
    flags = (imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR |
             imgui.WINDOW_NO_SCROLLBAR)
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(250, gui.height)
    imgui.begin("##MainMenu", None, flags)

    imgui.text_colored("FoGaze", 0.3, 0.8, 1.0, 1.0)
    imgui.separator()

    action = None
    if cal_mode is None:
        imgui.text(f"FPS: {fps_val}")
        imgui.separator()
        if imgui.button("Re-calibrate Gaze", -1, 36):
            action = 'gaze_cal'
        if imgui.button("Calibrate Depth", -1, 36):
            action = 'depth_cal'
        s = show_depth
        _, s = imgui.checkbox("Depth Overlay", s)
        if s != show_depth:
            action = ('toggle_depth', s)
        imgui.separator()
        if imgui.button("Quit", -1, 36):
            action = 'quit'
        imgui.separator()
        imgui.text(f"Zone: {zone_txt}")
        imgui.text(f"Focus: {focus_txt}")
        imgui.text(f"Depth: {depth_txt}")
    else:
        imgui.text_colored("Calibrating", 0.2, 1.0, 0.5, 1.0)
        if cal_step:
            imgui.text(f"Step: {cal_step}")
        if cal_progress:
            imgui.text(cal_progress)
        imgui.separator()
        imgui.text_colored("ESC = Cancel", 0.8, 0.3, 0.3, 1.0)

    # Face camera preview at bottom of panel
    pw, ph = 230, 172  # 4:3 fits 250-wide panel
    if face_tex_id is not None:
        imgui.separator()
        imgui.image(face_tex_id, pw, ph)

    # Depth colormap preview below face (only when toggled on)
    if show_depth and depth_tex_id is not None:
        imgui.separator()
        imgui.image(depth_tex_id, pw, ph)

    imgui.end()
    return action


def _draw_instructions(lines, gui_height):
    """Bottom-left instruction panel."""
    n = len(lines)
    h = min(250, max(60, n * 22 + 30))
    flags = (imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR |
             imgui.WINDOW_NO_SCROLLBAR)
    imgui.set_next_window_position(0, gui_height - h)
    imgui.set_next_window_size(250, h)
    imgui.begin("##Instructions", None, flags)
    for line in lines:
        imgui.text_wrapped(line)
    imgui.end()


def _calibrate_two_cam(gaze_estimator, cap_face, cap_scene, gui,
                       capture_frames=250, grid_cols=3, grid_rows=3):
    """Grid calibration rendered through GUIOverlay + ImGui panels."""
    ret, tmp = cap_scene.read()
    if not ret:
        print("[FoGaze] Cannot read scene camera for calibration.")
        return False

    h_s, w_s = tmp.shape[:2]
    xs = np.linspace(int(w_s * 0.1), int(w_s * 0.9), grid_cols, dtype=int)
    ys = np.linspace(int(h_s * 0.1), int(h_s * 0.9), grid_rows, dtype=int)
    targets = [(int(x), int(y)) for y in ys for x in xs]
    collected = []

    guide_items = [
        "===== Calibration Guide =====",
        "",
        "Step 1 - Face detection",
        "  Position your face in front of the face camera",
        "  ɡāカメラの前に座って顔を映してください",
        "",
        "Step 2 - Look at the target & press ENTER",
        "  Look at the circled point, then press ENTER to capture",
        "  丸いターゲットを見て、ENTERを押して撮影",
        "",
        "Step 3 - Rotate your head during capture",
        "  Keep eyes on target, slowly rotate your head",
        "  ターゲットを見たままゆっくり頭を動かしてください",
        "",
        "Step 4 - Repeat for all 9 points",
        "  ENTER=capture  BACKSPACE=undo  ESC=finish",
        "  ENTER=撮影  BACKSPACE=戻る  ESC=終了",
        "",
        "Press ENTER to begin",
    ]

    def _render_frame(canvas, face_disp, cal_step, cal_progress, instructions):
        gui.update_scene_texture(canvas)
        if face_disp is not None:
            gui.update_face_texture(face_disp)
        gui.begin_frame()
        _draw_main_menu(gui, cal_mode='calibrating',
                        cal_step=cal_step, cal_progress=cal_progress,
                        face_tex_id=gui.face_texture_id,
                        depth_tex_id=gui.depth_texture_id)
        _draw_instructions(instructions, gui.height)
        gui.render()

    # ── Phase 1: Guide screen ─────────────────────────────────────────
    while True:
        ret_s, frame_s = cap_scene.read()
        if not ret_s:
            continue
        canvas = np.zeros((h_s, w_s, 3), dtype=np.uint8)
        for i, txt in enumerate(guide_items):
            cv2.putText(canvas, txt, (80, 50 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        _render_frame(canvas, None, "Guide", "", guide_items)
        if gui.was_key_pressed(glfw.KEY_ENTER):
            break
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            return False

    # ── Phase 2: Wait for face ────────────────────────────────────────
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
            cv2.putText(canvas, "Face not detected",
                        (w_s // 2 - 150, h_s // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, Theme.ACCENT_RED, 3)

        _render_frame(canvas, frame_face, "Face Detection",
                      "Keep face in front of camera",
                      ["Keep your face centered in the face camera",
                       "顔をカメラの中央に保ってください",
                       "",
                       "Hold still for 2 seconds",
                       "2秒間そのままの位置を保ってください"])
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            return False

    # ── Phase 3: Grid capture ─────────────────────────────────────────
    for idx, (tx, ty) in enumerate(targets):
        while True:
            ret_s, frame_s = cap_scene.read()
            ret_f, frame_f = cap_face.read()
            if not ret_s or not ret_f:
                continue
            frame_face = cv2.flip(frame_f, 1)
            canvas = frame_s.copy()

            # Target at this grid point
            cv2.circle(canvas, (tx, ty), 30, Theme.ACCENT_CYAN, -1)
            cv2.circle(canvas, (tx, ty), 36, (255, 255, 255), 2)
            cv2.line(canvas, (tx - 20, ty), (tx + 20, ty), (255, 255, 255), 1)
            cv2.line(canvas, (tx, ty - 20), (tx, ty + 20), (255, 255, 255), 1)

            for pt in collected:
                px, py = pt[0], pt[1]
                cv2.circle(canvas, (px, py), 8, Theme.ACCENT_GREEN, -1)
                cv2.circle(canvas, (px, py), 12, (255, 255, 255), 1)

            n_captured = len(collected)
            instructions = [
                f"Point {idx + 1} / {len(targets)} | Captured: {n_captured}",
                "",
                "ENTER = capture this point    BACKSPACE = undo",
                "ENTER = 撮影    BACKSPACE = 戻る",
            ]

            _render_frame(canvas, frame_face, f"Point {idx+1}/{len(targets)}",
                          f"Captured: {n_captured}", instructions)
            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                return False

            if gui.was_key_pressed(glfw.KEY_ENTER):
                for _ in range(capture_frames):
                    ret_f2, frame_f2 = cap_face.read()
                    if not ret_f2:
                        continue
                    ft, blink = gaze_estimator.extract_features(
                        cv2.flip(frame_f2, 1))
                    if ft is not None and not blink:
                        collected.append([tx, ty, ft])
                # Green flash + beep ×2
                for _ in range(2):
                    ret_s2, fb = cap_scene.read()
                    if ret_s2:
                        fb2 = fb.copy()
                        cv2.rectangle(fb2, (0, 0), (w_s, h_s), (0, 230, 0), -1)
                        cv2.putText(fb2, "Done!", (w_s // 2 - 80, h_s // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        gui.update_scene_texture(fb2)
                        gui.begin_frame()
                        _draw_main_menu(gui, cal_mode='calibrating',
                                        cal_step="Capturing...",
                                        cal_progress="",
                                        face_tex_id=gui.face_texture_id,
                                        depth_tex_id=gui.depth_texture_id)
                        gui.render()
                    print('\a', end='', flush=True)
                    os.system('echo -ne "\\a" > /dev/tty 2>/dev/null &')
                    time.sleep(0.2)
                break

            if gui.was_key_pressed(glfw.KEY_BACKSPACE):
                if collected:
                    removed = collected.pop()
                    print(f"[FoGaze] Removed point ({removed[0]}, {removed[1]})")

    # ── Train ──────────────────────────────────────────────────────────
    if len(collected) < 3:
        print(f"[FoGaze] Too few samples ({len(collected)}), cannot calibrate.")
        return False

    feats = np.array([c[2] for c in collected])
    targs = np.array([[c[0], c[1]] for c in collected])
    print(f"[FoGaze] Training on {len(feats)} samples...")
    gaze_estimator.train(feats, targs)
    print("[FoGaze] Calibration complete.")
    return True


def _calibrate_depth(depth_estimator, cap_scene, gui, sw, sh,
                     cap_face=None):
    """Interactive depth calibration rendered through GUIOverlay + ImGui.
    Uses a single 1m reference point: scale = 1.0 / depth_m.

    distances = [100]
    samples = []  # (distance_cm, scale)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    guide = [
        "===== Depth Calibration =====",
        "",
        "Place your hand or object 1 meter (100cm) away.",
        "手や物体を1メートル(100cm)の距離に置いてください。",
        "",
        "Make sure the object is centered in the crosshair.",
        "十字線の中央に物体が来るようにしてください。",
        "",
        "Press ENTER to capture",
        "ENTERで撮影",
        "",
        "ESC to cancel",
        "ESC=キャンセル",
        "",
        "Press ENTER to begin",
    ]

    def _render_frame(canvas, face_disp, cal_step, cal_progress, instructions):
        gui.update_scene_texture(canvas)
        if face_disp is not None:
            gui.update_face_texture(face_disp)
        gui.begin_frame()
        _draw_main_menu(gui, cal_mode='calibrating',
                        cal_step=cal_step, cal_progress=cal_progress,
                        face_tex_id=gui.face_texture_id,
                        depth_tex_id=gui.depth_texture_id)
        _draw_instructions(instructions, gui.height)
        gui.render()

    # ── Phase 1: Guide screen ─────────────────────────────────────────
    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        for i, txt in enumerate(guide):
            cv2.putText(canvas, txt, (80, 50 + i * 34),
                        font, 0.9, (200, 200, 200), 2)
        _render_frame(canvas, None, "Guide", "", guide)
        if gui.was_key_pressed(glfw.KEY_ENTER):
            break
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            return False

    # ── Phase 2: Capture each distance ────────────────────────────────
    for i, dist in enumerate(distances):
        while True:
            ret, frame = cap_scene.read()
            if not ret:
                continue
            fh, fw = frame.shape[:2]
            depth_map = depth_estimator.estimate(frame)

            canvas = frame.copy()
            cv2.putText(canvas, "Depth Calibration  |  1 meter",
                        (50, 50), font, 1.2, (0, 255, 255), 3)
            cv2.putText(canvas, "Place object/hand 1m away, centered in crosshair, then press ENTER",
                        (50, 100), font, 0.8, (220, 220, 220), 2)

            # Crosshair
            cx, cy = fw // 2, fh // 2
            r = 25
            sample_r = 50
            cv2.line(canvas, (0, cy), (fw, cy), (0, 255, 0), 1)
            cv2.line(canvas, (cx, 0), (cx, fh), (0, 255, 0), 1)
            gap, L = 15, 30
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                x0, y0 = cx + dx * gap, cy + dy * gap
                cv2.line(canvas, (x0, y0), (x0 + dx * L, y0), (0, 255, 0), 2)
                cv2.line(canvas, (x0, y0), (x0, y0 + dy * L), (0, 255, 0), 2)
            cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r),
                          (0, 255, 0), 1)
            cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1)

            # Depth PIP
            live_m = None
            if depth_map is not None:
                depth_color = depth_estimator.colormap(depth_map)
                pip_h, pip_w = fh // 4, fw // 4
                depth_small = cv2.resize(depth_color, (pip_w, pip_h))
                x_off, y_off = fw - pip_w - 10, fh - pip_h - 10
                canvas[y_off:y_off + pip_h, x_off:x_off + pip_w] = depth_small
                freshness = depth_estimator.depth_freshness
                label = "DEPTH (old)" if freshness > 2.0 else "DEPTH"
                cv2.putText(canvas, label, (x_off, y_off - 5),
                            font, 0.5, (0, 0, 255) if freshness > 2.0 else (0, 255, 0), 1)
                center_roi = depth_map[cy - sample_r:cy + sample_r, cx - sample_r:cx + sample_r]
                if center_roi.size > 0:
                    avg = float(np.median(center_roi))
                    live_m = avg
                    cv2.putText(canvas, f"{avg:.2f}m",
                                (x_off, y_off + pip_h + 20),
                                font, 0.6, (0, 255, 255), 2)

            face_disp = None
            if cap_face is not None:
                ret_f, ff = cap_face.read()
                if ret_f:
                    face_disp = cv2.flip(ff, 1)

            instructions = [
                "Place hand/object 1 meter away",
                "手や物体を1mの距離に置いてください",
                "",
                "Center it in the crosshair",
                "十字線の中央に合わせてください",
                "",
                "ENTER = capture    ESC = cancel",
            ]

            _render_frame(canvas, face_disp, "Depth Calibration",
                          "", instructions)

            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                return False

            if gui.was_key_pressed(glfw.KEY_ENTER):
                ret, frame = cap_scene.read()
                if not ret:
                    continue
                depth_map = depth_estimator.estimate(frame)
                if depth_map is None:
                    print("[CalibrateDepth] Depth not ready, waiting...")
                    for _ in range(60):
                        time.sleep(0.04)
                        depth_map = depth_estimator.estimate(frame)
                        if depth_map is not None:
                            break
                if depth_map is None:
                    continue
                center_roi = depth_map[cy - sample_r:cy + sample_r, cx - sample_r:cx + sample_r]
                if center_roi.size == 0:
                    continue
                avg_depth = float(np.median(center_roi))
                scale = 1.0 / avg_depth
                samples.append((dist, scale))
                print(f"[CalibrateDepth] 1m -> {avg_depth:.3f}m, scale={scale:.4f}")

                # Green flash
                fb = frame.copy()
                cv2.rectangle(fb, (0, 0), (fw, fh), (0, 180, 0), -1)
                cv2.putText(fb, f"{dist}cm captured!",
                            ((fw - 250) // 2, (fh + 30) // 2),
                            font, 1.5, (255, 255, 255), 3)
                gui.update_scene_texture(fb)
                gui.begin_frame()
                _draw_main_menu(gui, cal_mode='calibrating',
                                cal_step="Saving...", cal_progress="",
                                face_tex_id=gui.face_texture_id,
                                depth_tex_id=gui.depth_texture_id)
                gui.render()
                time.sleep(0.4)
                break

    if len(samples) < 1:
        print("[CalibrateDepth] No sample captured.")
        return False

    # Single-point calibration: scale = 1.0 / depth_m
    scale = samples[0][1]
    depth_estimator.save_cal(scale)
    print(f"[CalibrateDepth] 1m -> scale={scale:.4f}")
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

    # ── Screen size ───────────────────────────────────────────────────
    sw, sh = get_screen_size()
    print(f"[FoGaze] Screen: {sw}x{sh}")

    # ── Camera selection via GUI ──────────────────────────────────────
    avail = _scan_cameras()
    cv2.destroyAllWindows()
    gui = GUIOverlay(sw, sh)

    # Camera selection dialog
    face_cam_idx = 0
    scene_cam_idx = 1 if len(avail) > 1 else 0
    cam_selected = False
    while not cam_selected:
        gui.begin_frame()
        imgui.set_next_window_size(500, 300, imgui.ONCE)
        imgui.set_next_window_position(gui.width//2 - 250, gui.height//2 - 150,
                                        imgui.ONCE)
        imgui.begin("Camera Selection", None,
                    imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
        imgui.text("Select Face Camera and Scene Camera")
        imgui.separator()
        imgui.text("")
        camera_names = [f"Camera {i}" for i in avail]
        _, face_cam_idx = imgui.combo("Face Camera", face_cam_idx, camera_names)
        _, scene_cam_idx = imgui.combo("Scene Camera", scene_cam_idx, camera_names)
        imgui.text("")
        imgui.separator()
        imgui.text(f"Selected: Face=Camera {avail[face_cam_idx]}, "
                   f"Scene=Camera {avail[scene_cam_idx]}")
        if imgui.button("Start", 200, 50):
            if face_cam_idx != scene_cam_idx or len(avail) == 1:
                cam_selected = True
            else:
                # Force different cameras if possible
                if len(avail) > 1:
                    # Swap scene to next available
                    scene_cam_idx = (face_cam_idx + 1) % len(avail)
        imgui.end()
        gui.render()
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            gui.close()
            print("[FoGaze] User cancelled camera selection.")
            return

    face_cam = avail[face_cam_idx]
    scene_cam = avail[scene_cam_idx]
    print(f"[FoGaze] Face cam={face_cam}  Scene cam={scene_cam}")

    # gui stays open for calibration + main loop
    model_path = args.model_file or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if args.reset_model:
        gui.close()
        if os.path.isfile(model_path):
            os.remove(model_path)
            print(f"[FoGaze] Deleted {model_path}")
        else:
            print("[FoGaze] No model file found.")
        return

    if args.reset_depth_cal:
        gui.close()
        de = DepthEstimator(device="cuda", depth_size=384)
        de.reset_cal()
        return

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

        cap_tmp_face = cv2.VideoCapture(face_cam)
        cap_tmp_scene = cv2.VideoCapture(scene_cam)
        if not cap_tmp_face.isOpened() or not cap_tmp_scene.isOpened():
            raise RuntimeError("Cannot open cameras for calibration")
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_tmp_scene.set(cv2.CAP_PROP_FRAME_WIDTH, sw)
        cap_tmp_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)

        ok = _calibrate_two_cam(
            gaze_estimator, cap_tmp_face, cap_tmp_scene, gui,
        )
        cap_tmp_face.release()
        cap_tmp_scene.release()

        if not ok:
            print("[FoGaze] Calibration failed or was cancelled.")
            gui.close()
            return

        try:
            gaze_estimator.save_model(model_path)
            print(f"[FoGaze] Model saved to {model_path}")
        except Exception as e:
            print(f"[FoGaze] Warning: could not save model ({e})")

        # Done screen via gui
        while True:
            canvas = np.full((sh, sw, 3), Theme.BG_PRIMARY, dtype=np.uint8)
            cv2.putText(canvas, "Calibration done - press ENTER to start",
                        (sw // 2 - 300, sh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, Theme.ACCENT_GREEN, 3)
            gui.update_scene_texture(canvas)
            gui.begin_frame()
            _draw_main_menu(gui, cal_mode='calibrating',
                            cal_step="Done", cal_progress="Press ENTER to start",
                            face_tex_id=gui.face_texture_id,
                            depth_tex_id=gui.depth_texture_id)
            _draw_instructions(["Calibration complete!",
                                "キャリブレーション完了！",
                                "",
                                "Press ENTER to continue",
                                "ENTERで続行"],
                               gui.height)
            gui.render()
            if gui.was_key_pressed(glfw.KEY_ENTER):
                break
            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                gui.close()
                return

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
    depth_estimator = DepthEstimator(device="cuda", depth_size=384,
                                     min_interval=1.5)
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

    # Clear any leftover windows
    cv2.destroyAllWindows()

    # gui already open from camera selection
    _trigger_quit = False
    _trigger_recal = False
    _trigger_depth_cal = False

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
            gui.update_face_texture(face_display)

            # ── Canvas (scene feed + overlays) ────────────────────────
            canvas = frame_scene.copy()

            # Dim overlay for better UI contrast
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w_scene, h_scene),
                          Theme.BG_PRIMARY, -1)
            cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

            # YOLO bounding boxes with depth labels
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                is_focused = (focused is not None and focused is det)
                color = Theme.ACCENT_GREEN if is_focused else Theme.ACCENT_CYAN
                thickness = 3 if is_focused else 2
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                # Depth distance for this detection
                det_depth = depth_estimator.depth_at_bbox(x1, y1, x2, y2)
                det_depth_cm = depth_estimator.depth_to_distance_cm(det_depth) if det_depth is not None else None
                det_label = f"{det['class_name']} {det['confidence']:.2f}"
                if det_depth_cm is not None:
                    det_label += f"  {det_depth_cm/100:.1f}m"
                draw_text_stroke(
                    canvas, det_label,
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

            # ── Depth for MainMenu preview ────────────────────────────
            if depth_map is not None:
                depth_color = depth_estimator.colormap(depth_map)
                depth_thumb = cv2.resize(depth_color, (230, 172))
                gui.update_depth_texture(depth_thumb)

            # ── Display via GUIOverlay ──────────────────────────────────
            if not args.headless:
                gui.update_scene_texture(canvas)
                gui.update_face_texture(face_display)
                gui.begin_frame()

                # ── MainMenu (left panel) ────────────────────────────────
                action = _draw_main_menu(
                    gui, fps_val, show_depth,
                    ZONE_PHRASES[zi], fname, depth_txt,
                    face_tex_id=gui.face_texture_id,
                    depth_tex_id=gui.depth_texture_id,
                )
                if action == 'quit':
                    _trigger_quit = True
                elif action == 'gaze_cal':
                    _trigger_recal = True
                elif action == 'depth_cal':
                    _trigger_depth_cal = True
                elif isinstance(action, tuple) and action[0] == 'toggle_depth':
                    show_depth = action[1]

                gui.render()

            # ── Key events (GLFW keys also processed via GUI) ──────────
            if gui.was_key_pressed(glfw.KEY_ESCAPE) or _trigger_quit:
                print("[FoGaze] User quit.")
                break

            if gui.was_key_pressed(glfw.KEY_C) or _trigger_recal:
                print("[FoGaze] Re-calibrating gaze...")
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
                        gaze_estimator, cap_re_face, cap_re_scene, gui,
                    )
                    if ok:
                        gaze_estimator.save_model(model_path)
                        print(f"[FoGaze] Model saved to {model_path}")
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

            elif gui.was_key_pressed(glfw.KEY_D):
                show_depth = not show_depth
                topbar.toast(
                    f"Depth overlay {'ON' if show_depth else 'OFF'}",
                    Theme.ACCENT_GREEN if show_depth else Theme.ACCENT_ORANGE,
                )

            elif gui.was_key_pressed(glfw.KEY_P) or _trigger_depth_cal:
                topbar.toast("Depth calibration...", Theme.ACCENT_CYAN)
                ok = _calibrate_depth(depth_estimator, cap_scene, gui, sw, sh,
                                      cap_face=cap_face)
                if ok:
                    topbar.toast("Depth calibrated!", Theme.ACCENT_GREEN)
                else:
                    topbar.toast("Depth cal. cancelled", Theme.ACCENT_ORANGE)

            _trigger_quit = False
            _trigger_recal = False
            _trigger_depth_cal = False

    except KeyboardInterrupt:
        print("\n[FoGaze] Interrupted.")
    finally:
        cap_face.release()
        cap_scene.release()
        if 'gui' in dir():
            gui.close()
        else:
            cv2.destroyAllWindows()
        gaze_estimator.close()
        print("[FoGaze] Shutdown.")


if __name__ == "__main__":
    main()
