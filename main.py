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
from modules.ui import Theme, TopBar, GazeCursor
from modules.camera_calibrator import CameraCalibrator
from modules.grasp import (
    CameraIntrinsics, bbox_size_m, classify_graspable, project_bbox_to_3d,
)

# ── Optional ROS publishing (for Rviz) ───────────────────────────────
_ros_node = None
_ros_pub_markers = None
_ros_pub_pickup = None
_ros_pub_objects = None
_ros_spin_thread = None


def _ros_init():
    global _ros_node, _ros_pub_markers, _ros_pub_pickup, _ros_pub_objects
    global _ros_spin_thread
    try:
        import rclpy
        from visualization_msgs.msg import MarkerArray
        from std_msgs.msg import String
        import tf2_ros
        from geometry_msgs.msg import TransformStamped

        rclpy.init()
        _ros_node = rclpy.create_node("fogaze_ros_bridge")
        _ros_pub_markers = _ros_node.create_publisher(MarkerArray, "detections/markers", 1)
        # Manipulation link (consumed by fogaze_manip/pickup_planner).
        _ros_pub_pickup = _ros_node.create_publisher(String, "fogaze/pickup", 1)
        _ros_pub_objects = _ros_node.create_publisher(String, "fogaze/objects", 1)

        # Periodically publish map frame to /tf so rviz2 accepts it as fixed frame
        tf_broadcaster = tf2_ros.TransformBroadcaster(_ros_node)
        identity_tf = TransformStamped()
        identity_tf.header.stamp = _ros_node.get_clock().now().to_msg()
        identity_tf.header.frame_id = "map"
        identity_tf.child_frame_id = "fogaze_base"  # different child so TF accepts it
        identity_tf.transform.translation.x = 0.0
        identity_tf.transform.translation.y = 0.0
        identity_tf.transform.translation.z = 0.0
        identity_tf.transform.rotation.w = 1.0

        # Spin in background and republish TF periodically
        import threading
        def _spin_loop():
            while rclpy.ok():
                identity_tf.header.stamp = _ros_node.get_clock().now().to_msg()
                tf_broadcaster.sendTransform(identity_tf)
                rclpy.spin_once(_ros_node, timeout_sec=0.1)
        _ros_spin_thread = threading.Thread(target=_spin_loop, daemon=True)
        _ros_spin_thread.start()

        print("[FoGaze] ROS bridge ready — publishing /detections/markers + TF map→fogaze_base")
    except Exception as e:
        print(f"[FoGaze] ROS not available: {e}")


def _ros_publish_markers(frame, detections, depth_est):
    global _ros_node, _ros_pub_markers
    if _ros_pub_markers is None:
        return
    try:
        from visualization_msgs.msg import Marker, MarkerArray
        from std_msgs.msg import ColorRGBA
        from builtin_interfaces.msg import Time
        import numpy as np

        now = _ros_node.get_clock().now().to_msg() if hasattr(_ros_node, 'get_clock') else Time()
        h, w = frame.shape[:2]
        FX, FY = 532.0, 532.0
        CX, CY = w / 2.0, h / 2.0

        arr = MarkerArray()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            d = depth_est.depth_at_bbox(x1, y1, x2, y2)
            z_m = d / 100.0 if d is not None else 1.0
            cx_d, cy_d = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x3d = (cx_d - CX) * z_m / FX
            y3d = (cy_d - CY) * z_m / FY

            box = Marker()
            box.header.stamp = now
            box.header.frame_id = "map"
            box.ns = "detections"
            box.id = i * 2
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = x3d
            box.pose.position.y = y3d
            box.pose.position.z = z_m
            w3d = abs(x2 - x1) * z_m / FX
            h3d = abs(y2 - y1) * z_m / FY
            box.scale.x = max(w3d, 0.02)
            box.scale.y = max(h3d, 0.02)
            box.scale.z = max(z_m * 0.1, 0.02)
            box.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.35)
            box.lifetime.sec = 1
            arr.markers.append(box)

            label = Marker()
            label.header.stamp = now
            label.header.frame_id = "map"
            label.ns = "labels"
            label.id = i * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x3d
            label.pose.position.y = y3d
            label.pose.position.z = z_m + 0.05
            label.scale.z = 0.06
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"{det['class_name']} {det['confidence']:.2f}"
            label.lifetime.sec = 1
            arr.markers.append(label)

        _ros_pub_markers.publish(arr)
    except Exception as e:
        print(f"[FoGaze] ROS publish error: {e}")


def _ros_publish_objects(objects, frame_id):
    """Publish the per-frame object list (with graspability) as JSON."""
    if _ros_pub_objects is None:
        return
    try:
        import json
        from std_msgs.msg import String
        _ros_pub_objects.publish(String(
            data=json.dumps({"frame_id": frame_id, "objects": objects})))
    except Exception as e:
        print(f"[FoGaze] ROS objects publish error: {e}")


def _ros_publish_pickup(target, frame_id):
    """Publish a single pick request (triggered by triple-blink) as JSON."""
    if _ros_pub_pickup is None:
        return
    try:
        import json
        from std_msgs.msg import String
        _ros_pub_pickup.publish(String(
            data=json.dumps({"frame_id": frame_id, "target": target})))
        print(f"[Pickup] -> ROS fogaze/pickup: {target.get('class')} "
              f"graspable={target.get('graspable')} ({target.get('reason')})")
    except Exception as e:
        print(f"[FoGaze] ROS pickup publish error: {e}")


class TripleBlinkDetector:
    def __init__(self, window=1.5):
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
            if len(self._times) >= 3:
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


PANEL_W = 268  # left panel width (px)


def _stat_row(label, value, value_rgb=(0.90, 0.91, 0.94)):
    """One aligned 'Label    value' row inside the status card."""
    imgui.text_colored(label, 0.50, 0.53, 0.60, 1.0)
    imgui.same_line(96)
    imgui.text_colored(str(value), *value_rgb, 1.0)


def _draw_main_menu(gui, fps_val=0, show_depth=False,
                     zone_txt="--", focus_txt="--", depth_txt="--",
                     tracker_txt="--", cal_status="--", rel_txt="--",
                     gaze_txt="--", scene_txt="--",
                     cal_mode=None, cal_step=None, cal_progress=None,
                     face_tex_id=None, depth_tex_id=None,
                     eye_tex_id=None):
    """Left-side MainMenu panel. Returns action string or None."""
    from modules.depth_estimator import OPTS, COLORMAPS
    flags = (imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR)
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(PANEL_W, gui.height)
    imgui.begin("##MainMenu", None, flags)

    # ── Header ──────────────────────────────────────────────────────────
    imgui.text_colored("FoGaze", 0.20, 0.78, 1.0, 1.0)
    imgui.same_line()
    imgui.text_colored("v3", 0.50, 0.53, 0.60, 1.0)
    imgui.spacing()

    action = None
    if cal_mode is None:
        # ── Live status card ────────────────────────────────────────────
        fps_rgb = ((0.45, 0.85, 0.45) if fps_val >= 20 else
                   (0.95, 0.75, 0.25) if fps_val >= 12 else
                   (0.95, 0.40, 0.40))
        cal_rgb = ((0.39, 0.78, 0.0) if cal_status == "CAL"
                   else (1.0, 0.65, 0.0))
        imgui.begin_child("##status", PANEL_W - 28, 196, border=True)
        _stat_row("FPS", fps_val, fps_rgb)
        _stat_row("Tracker", tracker_txt or "--")
        _stat_row("Cal", cal_status or "--", cal_rgb)
        _stat_row("Zone", zone_txt or "--")
        _stat_row("Focus", focus_txt or "--", (0.20, 0.78, 1.0))
        _stat_row("Relation", rel_txt or "--", (0.39, 0.78, 0.0))
        _stat_row("Depth", depth_txt or "--", (0.20, 0.78, 1.0))
        _stat_row("Gaze", gaze_txt or "--")
        _stat_row("Camera", scene_txt or "--")
        imgui.end_child()
        imgui.spacing()

        # ── Primary actions ─────────────────────────────────────────────
        if imgui.button("Re-calibrate Gaze  (C)", -1, 38):
            action = 'gaze_cal'
        if imgui.button("Calibrate Cameras  (V)", -1, 38):
            action = 'cam_cal'

        s = show_depth
        _, s = imgui.checkbox("Depth overlay  (D)", s)
        if s != show_depth:
            action = ('toggle_depth', s)

        imgui.spacing()

        # ── Depth settings (collapsed by default → clean view) ──────────
        if imgui.collapsing_header("Depth settings")[0]:
            changed, v = imgui.slider_int("Off X", OPTS["off_x"], -50, 50)
            if changed:
                OPTS["off_x"] = v
            changed, v = imgui.slider_int("Off Y", OPTS["off_y"], -50, 50)
            if changed:
                OPTS["off_y"] = v
            changed, v = imgui.slider_int("Min mm", OPTS["min_dist_mm"], 100, 1000)
            if changed:
                OPTS["min_dist_mm"] = v
            changed, v = imgui.slider_int("Max mm", OPTS["max_dist_mm"], 500, 8000)
            if changed:
                OPTS["max_dist_mm"] = v
            cmap_names = [f"cmap {i}" for i in range(len(COLORMAPS))]
            _, idx = imgui.combo("Colormap", OPTS["cmap_idx"], cmap_names)
            OPTS["cmap_idx"] = idx

        # ── Help / shortcuts ────────────────────────────────────────────
        if imgui.collapsing_header("Shortcuts")[0]:
            imgui.text_colored("C", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Re-calibrate gaze")
            imgui.text_colored("V", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Calibrate cameras")
            imgui.text_colored("D", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Toggle depth")
            imgui.text_colored("F", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Fullscreen")
            imgui.text_colored("ESC", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Quit")
    else:
        imgui.begin_child("##status", PANEL_W - 28, 90, border=True)
        imgui.text_colored("Calibrating...", 0.20, 0.78, 1.0, 1.0)
        if cal_step:
            imgui.text(f"Step: {cal_step}")
        if cal_progress:
            imgui.text(cal_progress)
        imgui.end_child()
        imgui.spacing()
        imgui.text_colored("ESC = Cancel", 0.95, 0.40, 0.40, 1.0)

    # ── Camera previews ─────────────────────────────────────────────────
    pw = PANEL_W - 28
    ph = int(pw * 3 / 4)
    if face_tex_id is not None:
        imgui.spacing()
        imgui.text_colored("FACE TRACK", 0.50, 0.53, 0.60, 1.0)
        imgui.image(face_tex_id, pw, ph)

    if eye_tex_id is not None:
        imgui.text_colored("EYE TRACK", 0.50, 0.53, 0.60, 1.0)
        imgui.image(eye_tex_id, pw, 60)

    if show_depth and depth_tex_id is not None:
        imgui.text_colored("DEPTH MAP", 0.50, 0.53, 0.60, 1.0)
        imgui.image(depth_tex_id, pw, ph)

    # ── Quit pinned to bottom, danger-styled ────────────────────────────
    if cal_mode is None:
        btn_h = 36
        avail_y = imgui.get_content_region_available()[1]
        if avail_y > btn_h + 8:
            imgui.dummy(1, avail_y - btn_h - 4)
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.32, 0.12, 0.14, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.70, 0.20, 0.22, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.90, 0.25, 0.27, 1.0)
        if imgui.button("Quit", -1, btn_h):
            action = 'quit'
        imgui.pop_style_color(3)

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
    imgui.set_next_window_size(PANEL_W, h)
    imgui.begin("##Instructions", None, flags)
    for line in lines:
        imgui.text_wrapped(line)
    imgui.end()


def _calibrate_two_cam(gaze_estimator, cap_face, depth_estimator, gui,
                       capture_frames=250, grid_cols=3, grid_rows=3):
    """Grid calibration rendered through GUIOverlay + ImGui panels.
    Scene feed comes from PrimeSense via depth_estimator.
    """
    scene, _ = depth_estimator.get_frame()
    if scene is None:
        print("[FoGaze] Cannot read PrimeSense for calibration.")
        return False

    h_s, w_s = scene.shape[:2]
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
        scene, _ = depth_estimator.get_frame()
        if scene is None:
            continue
        frame_s = scene
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
        scene, _ = depth_estimator.get_frame()
        if scene is None:
            continue
        frame_s = scene
        ret_f, frame_f = cap_face.read()
        if not ret_f:
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
            scene, _ = depth_estimator.get_frame()
            if scene is None:
                continue
            frame_s = scene
            ret_f, frame_f = cap_face.read()
            if not ret_f:
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
                    fb_scene, _ = depth_estimator.get_frame()
                    if fb_scene is not None:
                        fb2 = fb_scene.copy()
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


def _calibrate_cameras(calib_face: CameraCalibrator,
                       cap_face, depth_estimator, gui):
    """Calibrate face camera lens distortion using chessboard pattern."""
    import cv2 as _cv2
    font = _cv2.FONT_HERSHEY_SIMPLEX

    for cam_name, cap, calib in [
        ("FACE CAMERA", cap_face, calib_face),
    ]:
        frames = []
        print(f"[CalibrateCameras] === {cam_name} ===")
        for _ in range(30):
            cap.read()
        guide = [
            f"Calibrating {cam_name}",
            "",
            "Show a printed chessboard (9x6 internal corners)",
            "",
            "Move it at different angles and positions",
            "",
            f"Samples: 0/10  (need >= 10)",
            "",
            "ENTER = capture    ESC = skip",
        ]
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            ret_cb, corners = _cv2.findChessboardCorners(
                gray, CameraCalibrator.CHESSBOARD, None
            )
            display = frame.copy()
            if ret_cb:
                _cv2.drawChessboardCorners(display,
                                           CameraCalibrator.CHESSBOARD,
                                           corners, ret_cb)
            h, w = frame.shape[:2]
            _cv2.putText(display, f"Samples: {len(frames)}/10",
                         (30, h - 30), font, 0.8,
                         (0, 255, 0) if len(frames) >= 10 else (0, 255, 255), 2)

            guide[6] = f"Samples: {len(frames)}/10  (need >= 10)"

            # Also show PrimeSense scene
            scene, _ = depth_estimator.get_frame()
            if scene is not None:
                gui.update_scene_texture(scene)

            gui.update_face_texture(display)
            gui.begin_frame()
            flags = (imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE |
                     imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR |
                     imgui.WINDOW_NO_SCROLLBAR)
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(PANEL_W, gui.height)
            imgui.begin("##CalibCam", None, flags)
            imgui.text_colored("Camera Calibration", 0.20, 0.78, 1.0, 1.0)
            imgui.separator()
            for line in guide:
                imgui.text_wrapped(line)
            imgui.separator()
            imgui.text_colored("ENTER = capture    ESC = skip",
                               0.8, 0.3, 0.3, 1.0)
            imgui.end()
            gui.render()

            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                break
            if gui.was_key_pressed(glfw.KEY_ENTER):
                if ret_cb and corners is not None:
                    frames.append(frame)
                    print(f"[CalibrateCameras] {cam_name}: sample {len(frames)}")
                    if len(frames) >= 10:
                        break

        if len(frames) >= 10:
            h, w = frames[0].shape[:2]
            ok = calib.calibrate(frames, w, h)
            if ok:
                print(f"[CalibrateCameras] {cam_name}: calibration successful")
                ret, fb = cap.read()
                if ret:
                    _cv2.putText(fb, f"{cam_name} calibrated!", (50, 50),
                                font, 1.2, (0, 255, 0), 3)
                    gui.update_face_texture(fb)
            else:
                print(f"[CalibrateCameras] {cam_name}: calibration failed")
        else:
            print(f"[CalibrateCameras] {cam_name}: skipped")

    # Re-init face undistort maps
    ret, ff = cap_face.read()
    if ret:
        hf, wf = ff.shape[:2]
        calib_face._ensure_maps(wf, hf)

    print("[CalibrateCameras] Done.")
    return True


# Face mesh landmark groups for visualization
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
             159, 160, 161, 246]
_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
              386, 387, 388, 466]
_LEFT_IRIS = 468
_RIGHT_IRIS = 473
_LEFT_EYE_CORNERS = [33, 133]
_RIGHT_EYE_CORNERS = [362, 263]


def _align_face_and_eyes(face_bgr, landmarks, h_f, w_f):
    """Draw face mesh on full frame + extract eye crops with iris markers.

    Returns: (face_display, eye_display) or (None, None)
      face_display: full BGR frame with face/eye/iris overlay lines
      eye_display:  BGR image with L/R eyes side-by-side + iris points
    """
    display = face_bgr.copy()
    xs = [lm.x * w_f for lm in landmarks]
    ys = [lm.y * h_f for lm in landmarks]

    # ── Draw face oval ──
    for i in range(len(_FACE_OVAL) - 1):
        a, b = _FACE_OVAL[i], _FACE_OVAL[i + 1]
        p1 = (int(landmarks[a].x * w_f), int(landmarks[a].y * h_f))
        p2 = (int(landmarks[b].x * w_f), int(landmarks[b].y * h_f))
        cv2.line(display, p1, p2, (100, 200, 100), 1)

    # ── Eye contours (yellow) ──
    for pts_list in [_LEFT_EYE, _RIGHT_EYE]:
        for i in range(len(pts_list) - 1):
            a, b = pts_list[i], pts_list[i + 1]
            p1 = (int(landmarks[a].x * w_f), int(landmarks[a].y * h_f))
            p2 = (int(landmarks[b].x * w_f), int(landmarks[b].y * h_f))
            cv2.line(display, p1, p2, (0, 255, 255), 2)
        a, b = pts_list[-1], pts_list[0]
        p1 = (int(landmarks[a].x * w_f), int(landmarks[a].y * h_f))
        p2 = (int(landmarks[b].x * w_f), int(landmarks[b].y * h_f))
        cv2.line(display, p1, p2, (0, 255, 255), 2)

    # ── Iris centers (cyan + crosshair) ──
    for iris_idx in [_LEFT_IRIS, _RIGHT_IRIS]:
        ix = int(landmarks[iris_idx].x * w_f)
        iy = int(landmarks[iris_idx].y * h_f)
        cv2.circle(display, (ix, iy), 3, (255, 255, 0), -1)
        cv2.line(display, (ix - 5, iy), (ix + 5, iy), (255, 255, 0), 1)
        cv2.line(display, (ix, iy - 5), (ix, iy + 5), (255, 255, 0), 1)

    # ── Extract L/R eye regions ──
    def _get_eye_roi(corner_a, corner_b, iris_idx):
        ax, ay = landmarks[corner_a].x * w_f, landmarks[corner_a].y * h_f
        bx, by = landmarks[corner_b].x * w_f, landmarks[corner_b].y * h_f
        cx = (ax + bx) * 0.5
        cy = (ay + by) * 0.5
        eye_r = int(max(abs(bx - ax), abs(by - ay)) * 1.3)
        x1 = int(max(0, cx - eye_r))
        y1 = int(max(0, cy - eye_r))
        x2 = int(min(w_f, cx + eye_r))
        y2 = int(min(h_f, cy + eye_r))
        crop = display[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        # Redraw iris on crop with local coords
        lx = int(landmarks[iris_idx].x * w_f - x1)
        ly = int(landmarks[iris_idx].y * h_f - y1)
        cv2.circle(crop, (lx, ly), 2, (255, 255, 0), -1)
        cv2.line(crop, (lx - 4, ly), (lx + 4, ly), (255, 255, 0), 1)
        cv2.line(crop, (lx, ly - 4), (lx, ly + 4), (255, 255, 0), 1)
        return crop

    l_eye = _get_eye_roi(33, 133, 468)
    r_eye = _get_eye_roi(362, 263, 473)

    if l_eye is None or r_eye is None:
        return display, None

    lh, lw = l_eye.shape[:2]
    rh, rw = r_eye.shape[:2]
    eye_h = max(lh, rh)
    if lh < eye_h:
        pad = np.zeros((eye_h - lh, lw, 3), dtype=np.uint8)
        l_eye = np.vstack([l_eye, pad])
    if rh < eye_h:
        pad = np.zeros((eye_h - rh, rw, 3), dtype=np.uint8)
        r_eye = np.vstack([r_eye, pad])

    eye_display = np.hstack([l_eye, r_eye])
    return display, eye_display


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
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--ros", action="store_true",
                        help="Publish detections + triple-blink pickups to "
                             "ROS 2 (fogaze/objects, fogaze/pickup)")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame",
                        help="ROS frame_id for published object poses")

    args = parser.parse_args()

    # ── Screen size ───────────────────────────────────────────────────
    sw, sh = get_screen_size()
    print(f"[FoGaze] Screen: {sw}x{sh}")

    # ── Camera selection via GUI ──────────────────────────────────────
    avail = _scan_cameras()
    cv2.destroyAllWindows()
    gui = GUIOverlay(sw, sh)
    # Reserve the left strip for the side panel so the scene view is
    # rendered to the right of it instead of being covered by it.
    gui.margin_left = PANEL_W

    # Only need face camera; PrimeSense provides scene color + depth
    face_cam_idx = 0
    cam_selected = False
    while not cam_selected:
        gui.begin_frame()
        imgui.set_next_window_size(400, 220, imgui.ONCE)
        imgui.set_next_window_position(gui.width//2 - 200, gui.height//2 - 110,
                                        imgui.ONCE)
        imgui.begin("Camera Selection", None,
                    imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
        imgui.text_colored("FoGaze", 0.20, 0.78, 1.0, 1.0)
        imgui.text("Select the face camera.")
        imgui.text_colored("PrimeSense provides the scene view.",
                           0.50, 0.53, 0.60, 1.0)
        imgui.spacing()
        camera_names = [f"Camera {i}" for i in avail]
        _, face_cam_idx = imgui.combo("Face Camera", face_cam_idx, camera_names)
        imgui.spacing()
        imgui.text_colored(f"-> Camera {avail[face_cam_idx]} + PrimeSense",
                           0.45, 0.85, 0.45, 1.0)
        imgui.spacing()
        if imgui.button("Start", -1, 46):
            cam_selected = True
        imgui.end()
        gui.render()
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            gui.close()
            print("[FoGaze] User cancelled camera selection.")
            return

    face_cam = avail[face_cam_idx]
    print(f"[FoGaze] Face cam={face_cam}  Scene=PrimeSense (built-in)")

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
        if not cap_tmp_face.isOpened():
            raise RuntimeError("Cannot open face camera for calibration")
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_tmp_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok = _calibrate_two_cam(
            gaze_estimator, cap_tmp_face, depth_estimator, gui,
        )
        cap_tmp_face.release()

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

    if not cap_face.isOpened():
        raise RuntimeError(f"Cannot open face camera {face_cam}")

    cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(10):
        cap_face.read()

    # ── PrimeSense depth camera (also provides scene color) ──────────
    depth_estimator = DepthEstimator()
    # Warm up
    for _ in range(30):
        _, _ = depth_estimator.get_frame()

    # ── Camera calibrator (face camera lens distortion) ──────────────
    calib_face = CameraCalibrator(face_cam)

    # ── Object detector (scene camera via PrimeSense color) ──────────
    detector = ObjectDetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        imgsz=args.imgsz,
    )
    detector.set_detection_interval(args.detection_interval)

    # ── PrimeSense depth estimator (also provides scene color stream) ──
    show_depth = False

    # ── UI components ─────────────────────────────────────────────────
    topbar = TopBar()  # kept for toast()/set_fps() calls; bar no longer drawn
    cursor = GazeCursor()

    # ── State ─────────────────────────────────────────────────────────
    gaze_active = False
    focused = None
    cal_notified = False

    blink_detector = TripleBlinkDetector()
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
    _trigger_cam_cal = False

    # Optional ROS bridge for Rviz
    _ros_init()

    try:
        while True:
            # ── Read cameras ──────────────────────────────────────────
            ret_face, frame_face = cap_face.read()
            scene_frame, depth_map = depth_estimator.get_frame()
            ret_scene = scene_frame is not None
            frame_scene = scene_frame if ret_scene else np.zeros((sh, sw, 3), dtype=np.uint8)
            if not ret_face:
                break

            frame_face = cv2.flip(frame_face, 1)

            # Undistort face feed (auto-defaults if no chessboard cal)
            frame_face = calib_face.undistort(frame_face)

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

            # ── Publish YOLO markers to ROS (for Rviz) ─────────────
            _ros_publish_markers(frame_scene, detections, depth_estimator)

            # ── Depth (already from PrimeSense get_frame) ─────────────
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

            # ── Triple-blink → speech + pickup signal ────────────────
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
                    print(f"[Pickup] TRIGGER: {focused['class_name']} at "
                          f"depth={focused_depth_cm:.0f}cm" if focused_depth_cm
                          else f"[Pickup] TRIGGER: {focused['class_name']}")
                    # ── Dispatch the pick request to the manipulation stack ──
                    if args.ros:
                        intr = CameraIntrinsics(cx=w_scene / 2.0,
                                                cy=h_scene / 2.0)
                        if focused_depth_cm is not None:
                            z_m = focused_depth_cm / 100.0
                            size_wh = bbox_size_m(focused["bbox"], z_m, intr)
                            pose = project_bbox_to_3d(focused["bbox"], z_m, intr)
                            graspable, reason = classify_graspable(z_m, size_wh)
                        else:
                            z_m = size_wh = pose = None
                            graspable, reason = False, "no-depth"
                        _ros_publish_pickup({
                            "class": focused["class_name"],
                            "confidence": focused["confidence"],
                            "pose": list(pose) if pose else None,
                            "size": list(size_wh) if size_wh else None,
                            "graspable": graspable, "reason": reason,
                        }, args.camera_frame)
                else:
                    speech.speak(phrase)

            # ── Face camera display (separate window) ─────────────────
            face_display = frame_face.copy()
            eye_display = None
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
                        aligned_face, eye_img = _align_face_and_eyes(
                            face_display, lm, h_f, w_f
                        )
                        if aligned_face is not None:
                            face_display = aligned_face
                        if eye_img is not None:
                            eye_display = eye_img
                except Exception:
                    pass
            gui.update_face_texture(face_display)
            if eye_display is not None:
                gui.update_eye_texture(eye_display)

            # ── Canvas (scene feed + overlays) ────────────────────────
            canvas = frame_scene.copy()

            # Dim overlay for better UI contrast
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w_scene, h_scene),
                          Theme.BG_PRIMARY, -1)
            cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

            # Depth colormap overlay (when toggled)
            if show_depth and depth_map is not None:
                depth_color = depth_estimator.colormap(depth_map)
                depth_resized = cv2.resize(depth_color, (w_scene, h_scene))
                cv2.addWeighted(depth_resized, 0.5, canvas, 0.5, 0, canvas)

            # YOLO bounding boxes with depth labels + graspability
            grasp_intr = CameraIntrinsics(cx=w_scene / 2.0, cy=h_scene / 2.0)
            ros_objects = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                is_focused = (focused is not None and focused is det)
                # Depth distance for this detection
                det_depth = depth_estimator.depth_at_bbox(x1, y1, x2, y2)
                det_depth_cm = depth_estimator.depth_to_distance_cm(det_depth) if det_depth is not None else None

                # ── Graspability (by distance + real-world size) ──────────
                if det_depth_cm is not None:
                    z_m = det_depth_cm / 100.0
                    size_wh = bbox_size_m(det["bbox"], z_m, grasp_intr)
                    pose = project_bbox_to_3d(det["bbox"], z_m, grasp_intr)
                    graspable, reason = classify_graspable(z_m, size_wh)
                else:
                    z_m = size_wh = pose = None
                    graspable, reason = False, "no-depth"
                det["_grasp"] = {
                    "class": det["class_name"], "confidence": det["confidence"],
                    "pose": list(pose) if pose else None,
                    "size": list(size_wh) if size_wh else None,
                    "graspable": graspable, "reason": reason,
                }
                ros_objects.append(det["_grasp"])

                # Colour encodes graspability; focus adds thickness.
                if det_depth_cm is None:
                    color = Theme.ACCENT_CYAN          # unknown depth
                elif graspable:
                    color = Theme.ACCENT_GREEN         # can pick up
                else:
                    color = Theme.ACCENT_RED           # cannot pick up
                thickness = 3 if is_focused else 2
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

                det_label = f"{det['class_name']} {det['confidence']:.2f}"
                if det_depth_cm is not None:
                    det_label += f"  {det_depth_cm/100:.1f}m"
                det_label += "  [grab]" if graspable else f"  [{reason}]"
                # Clamp label inside the canvas so it never spills off-screen
                (lbl_w, lbl_h), lbl_base = cv2.getTextSize(
                    det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                lbl_x = min(max(x1, 0), w_scene - lbl_w)
                lbl_y = y1 - 8
                if lbl_y - lbl_h < 0:  # off the top → draw below the box's top edge
                    lbl_y = y1 + lbl_h + 8
                lbl_y = min(lbl_y, h_scene - lbl_base)
                draw_text_stroke(
                    canvas, det_label,
                    (lbl_x, lbl_y), scale=0.5, color=color,
                )

            if args.ros:
                _ros_publish_objects(ros_objects, args.camera_frame)

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

            # Calibration notification
            if _is_trained(gaze_estimator.model) and not cal_notified:
                cal_notified = True

            # HUD values (rendered in the left panel, not on the scene canvas)
            cal_status = "CAL" if _is_trained(gaze_estimator.model) else "UNCAL"
            zi, _ = _zone_for(gx_scene, gy_scene, w_scene, h_scene)
            fname = (focused['class_name'] if focused['confidence'] >= THING_THRESHOLD
                     else "That Thing") if focused else "--"
            rel_txt = (f"{fname} {focused_rel[0]} "
                       f"{focused_rel[1]['class_name']}"
                       ) if focused_rel else "--"
            depth_txt = (f"{int(focused_depth_cm)}cm"
                         if focused_depth_cm is not None else "--")
            tracker_txt = f"{args.model.upper()} | {args.filter.upper()}"
            gaze_txt = f"({gx_scene}, {gy_scene})"
            scene_txt = f"Scene:PrimeSense  Face:{face_cam}"

            # Help hint
            if time.time() - help_t < 5:
                draw_text_stroke(
                    canvas,
                    "ESC: quit  |  c: re-calibrate  |  v: cal cameras  |  d: depth",
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
                    tracker_txt=tracker_txt, cal_status=cal_status,
                    rel_txt=rel_txt, gaze_txt=gaze_txt, scene_txt=scene_txt,
                    face_tex_id=gui.face_texture_id,
                    depth_tex_id=gui.depth_texture_id,
                    eye_tex_id=gui.eye_texture_id,
                )
                if action == 'quit':
                    _trigger_quit = True
                elif action == 'gaze_cal':
                    _trigger_recal = True
                elif action == 'cam_cal':
                    _trigger_cam_cal = True
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

                cap_re_face = cv2.VideoCapture(face_cam)
                if not cap_re_face.isOpened():
                    print("[FoGaze] Failed to open camera for re-calibration.")
                    break
                cap_re_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_re_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                try:
                    ok = _calibrate_two_cam(
                        gaze_estimator, cap_re_face, depth_estimator, gui,
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

                # Re-open main face camera
                cap_face = cv2.VideoCapture(face_cam)
                if not cap_face.isOpened():
                    print("[FoGaze] Failed to re-open camera.")
                    break
                cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            elif gui.was_key_pressed(glfw.KEY_D):
                show_depth = not show_depth
                topbar.toast(
                    f"Depth overlay {'ON' if show_depth else 'OFF'}",
                    Theme.ACCENT_GREEN if show_depth else Theme.ACCENT_ORANGE,
                )

            elif gui.was_key_pressed(glfw.KEY_V) or _trigger_cam_cal:
                topbar.toast("Camera calibration...", Theme.ACCENT_CYAN)
                _calibrate_cameras(calib_face,
                                   cap_face, depth_estimator, gui)
                topbar.toast("Camera cal. done", Theme.ACCENT_GREEN)

            _trigger_quit = False
            _trigger_recal = False
            _trigger_cam_cal = False

    except KeyboardInterrupt:
        print("\n[FoGaze] Interrupted.")
    finally:
        cap_face.release()
        depth_estimator.close()
        if 'gui' in dir():
            gui.close()
        else:
            cv2.destroyAllWindows()
        gaze_estimator.close()
        global _ros_node
        if _ros_node is not None:
            try:
                import rclpy
                _ros_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
        print("[FoGaze] Shutdown.")


if __name__ == "__main__":
    main()
