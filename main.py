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
import shutil
import subprocess
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
from modules.filters import OneEuroSmoother

from modules.object_detector import ObjectDetector, available_models
from modules.depth_estimator import DepthEstimator, OPTS
from modules.gui_overlay import GUIOverlay
from modules.ui import draw_text_stroke
from modules.ui import Theme, TopBar, GazeCursor
from modules.camera_calibrator import CameraCalibrator
from modules.face_lock import FaceLock
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


class BlinkGestureDetector:
    """Counts a quick burst of blinks and reports its size once it settles.

    A double-blink (communicate) is a prefix of a triple-blink (robot
    command), so we can't fire on the Nth blink — we wait for the burst to
    end.  The count is emitted when blinking pauses for ``settle`` seconds
    or the ``window`` since the first blink elapses, whichever comes first.
    ``update`` returns the burst size (>= ``min_count``) on the frame it
    settles, otherwise ``0``.
    """

    def __init__(self, window=1.6, settle=0.5, min_count=2):
        self.window = window
        self.settle = settle
        self.min_count = max(2, int(min_count))
        self._times = []
        self._was = False

    def update(self, blinking, now=None):
        if now is None:
            now = time.time()
        if not blinking and self._was:          # a blink just ended
            self._times.append(now)
        self._was = blinking
        if not self._times:
            return 0
        settled = (not blinking) and (now - self._times[-1] >= self.settle)
        expired = now - self._times[0] >= self.window
        if settled or expired:
            count = len(self._times)
            self._times = []
            return count if count >= self.min_count else 0
        return 0


class RobustGazeGate:
    """Rejects blink-induced gaze spikes before they reach the smoother.

    A blink corrupts the eye landmarks for a few frames *before* the openness
    drops far enough to be flagged as a blink, and again as the eye reopens —
    so the raw prediction briefly flings far away and snaps back.  Two guards
    suppress that without hurting genuine eye movement:

      • **Jump confirmation** — a sudden large jump from the held point is held
        as 'pending' for one frame and only accepted if the next prediction
        lands near it.  A real saccade persists (accepted, 1-frame lag); a
        blink spike snaps back (discarded).
      • **Post-blink refractory** — after a blink clears, predictions are
        ignored for a short window so the still-reopening eye can't fling the
        cursor.

    ``accept`` returns the point to use, or ``None`` to hold the last cursor.
    """

    def __init__(self, refractory=0.18, jump_px=120.0, confirm_px=80.0):
        self._refractory = refractory
        self._jump_px = jump_px
        self._confirm_px = confirm_px
        self._blink_end_t = -1e9
        self._was_blink = False
        self._last = None       # last accepted point
        self._pending = None    # large-jump sample awaiting confirmation

    def accept(self, gx, gy, blink, now):
        if blink:                       # never trust a blink / no-face frame
            self._was_blink = True
            self._pending = None
            return None
        if self._was_blink:             # first clean frame after a blink
            self._was_blink = False
            self._blink_end_t = now
        if now - self._blink_end_t < self._refractory:
            return None                 # eye still reopening → hold
        if self._last is None:
            self._last = (gx, gy)
            return self._last
        jump = ((gx - self._last[0]) ** 2 + (gy - self._last[1]) ** 2) ** 0.5
        if jump <= self._jump_px:       # normal small movement → accept
            self._pending = None
            self._last = (gx, gy)
            return self._last
        # Large jump: only believe it if the previous frame jumped here too.
        if self._pending is not None:
            d = ((gx - self._pending[0]) ** 2
                 + (gy - self._pending[1]) ** 2) ** 0.5
            if d <= self._confirm_px:   # confirmed saccade
                self._last = (gx, gy)
                self._pending = None
                return self._last
        self._pending = (gx, gy)        # hold one frame, await confirmation
        return None


PIPER_MODEL = os.path.expanduser(
    "~/.cache/fogaze3/piper/en_US-amy-medium.onnx")


class SpeechOutput:
    """Speaks short phrases off the main thread.

    Prefers Piper neural TTS (a natural, human-sounding voice) when its model
    is present; otherwise falls back to a tuned espeak-ng via pyttsx3.  A
    cooldown plus a 'busy' guard keep utterances from overlapping or queueing.
    """

    def __init__(self, cooldown=2.0, piper_model=PIPER_MODEL):
        self._last = 0.0
        self._cooldown = cooldown
        self._piper_model = piper_model
        self._backend = None      # None until first use → 'piper' | 'espeak'
        self._piper = None
        self._player = None       # external WAV player for piper output
        self._engine = None       # pyttsx3 engine (espeak fallback)
        self._lock = threading.Lock()
        self._speaking = False

    def _init(self):
        if self._backend is not None:
            return
        # ── Preferred: Piper neural voice ──────────────────────────────
        if os.path.exists(self._piper_model):
            try:
                from piper import PiperVoice
                self._piper = PiperVoice.load(self._piper_model)
                for p in ("aplay", "paplay", "pw-play"):
                    if shutil.which(p):
                        self._player = p
                        break
                if self._player:
                    self._backend = "piper"
                    print(f"[Speech] Piper voice ready (via {self._player})")
                    return
            except Exception as e:
                print(f"[Speech] Piper unavailable ({e}); using espeak")
        # ── Fallback: tuned espeak-ng via pyttsx3 ──────────────────────
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 165)     # slower → clearer
        self._engine.setProperty('volume', 1.0)
        try:                                       # prefer the clearer en-US
            for v in self._engine.getProperty('voices'):
                tag = f"{v.id} {v.name}".lower()
                if 'en-us' in tag or 'english (america)' in tag:
                    self._engine.setProperty('voice', v.id)
                    break
        except Exception:
            pass
        self._backend = "espeak"
        print("[Speech] espeak voice ready (Piper model not found)")

    def speak(self, text):
        now = time.time()
        if now - self._last < self._cooldown:
            return
        with self._lock:           # drop if a previous phrase is still playing
            if self._speaking:
                return
            self._speaking = True
        self._last = now
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        try:
            self._init()
            if self._backend == "piper":
                self._speak_piper(text)
            else:
                self._engine.say(text)
                self._engine.runAndWait()
        except Exception as e:
            print(f"[Speech] error: {e}")
        finally:
            with self._lock:
                self._speaking = False

    def _speak_piper(self, text):
        import wave
        import tempfile
        from piper.config import SynthesisConfig
        cfg = SynthesisConfig(length_scale=1.06, volume=1.0)  # 1.06 → unhurried
        path = tempfile.mktemp(suffix=".wav")
        try:
            with wave.open(path, "wb") as wf:
                self._piper.synthesize_wav(text, wf, syn_config=cfg)
            cmd = (["aplay", "-q", path] if self._player == "aplay"
                   else [self._player, path])
            subprocess.run(cmd, capture_output=True)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass



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

THING_THRESHOLD = 0.2  # below this → hedge the label ("thing that might be a …")


def _hedged_label(det):
    """Object label for speech/UI, hedged when YOLO isn't confident.

    A confident detection reads as its class ("bottle"); a low-confidence one
    becomes "thing that might be a bottle" (with the right a/an article) so the
    system never claims certainty it doesn't have.
    """
    cls = det.get("class_name", "object")
    if det.get("confidence", 0.0) >= THING_THRESHOLD:
        return cls
    article = "an" if cls[:1].lower() in "aeiou" else "a"
    return f"thing that might be {article} {cls}"


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


def _nearest_by_depth(focused, detections, depth_est, intr):
    """Find the object physically closest to *focused* in 3D and describe it.

    Uses the depth map (not just 2D bbox overlap): each object's bbox centre
    is back-projected to a 3D point with its depth, then the nearest other
    object is picked by real Euclidean distance.  Returns
    ``(other_det, dist_m, direction)`` where *direction* is a spoken phrase
    ("left of" / "right of" / "above" / "below" / "in front of" / "behind"),
    or ``None`` when depth is unavailable or there is no other object.
    """
    fz = depth_est.depth_at_bbox(*focused["bbox"])
    if fz is None:
        return None
    fp = project_bbox_to_3d(focused["bbox"], fz / 100.0, intr)
    best = None
    for ob in detections:
        if ob is focused:
            continue
        z = depth_est.depth_at_bbox(*ob["bbox"])
        if z is None:
            continue
        p = project_bbox_to_3d(ob["bbox"], z / 100.0, intr)
        d = ((fp[0] - p[0]) ** 2 + (fp[1] - p[1]) ** 2
             + (fp[2] - p[2]) ** 2) ** 0.5
        if best is None or d < best[1]:
            best = (ob, d, p)
    if best is None:
        return None
    ob, d, p = best
    # Camera optical frame: x right, y down, z forward.  Left/right is the
    # grid's job; here we describe the inter-object relation from depth +
    # geometry as next to / on top of / under / in front of / behind.
    dx, dy, dz = p[0] - fp[0], p[1] - fp[1], p[2] - fp[2]
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if ay >= ax and ay >= az:
        direction = "on top of" if dy > 0 else "under"
    elif az >= ax:
        direction = "in front of" if dz > 0 else "behind"
    else:
        direction = "next to"
    return ob, d, direction


# ── Natural-language speech design ──────────────────────────────────────
# Spoken (natural) versions of the 3×3 ZONE_PHRASES, in the same order.
ZONE_NATURAL = [
    "up on my left",
    "up above",
    "up on my right",
    "on my left-hand side",
    "right in front of me",
    "on my right-hand side",
    "down on my left",
    "down below",
    "down on my right",
]

# Turn the terse relation codes from _get_relation / _nearest_by_depth into
# natural prepositions that flow inside a spoken sentence.
_RELATION_WORDS = {
    "In": "inside", "On": "on top of", "Under": "under",
    "Next To": "next to", "Near": "next to", "Behind": "behind",
    "on top of": "on top of", "under": "under", "next to": "next to",
    "in front of": "in front of",
}

# Large objects a gripper can't grab but a person interacts with: speak the
# intended *action* instead of "I want that fridge".  Keyed by YOLO class.
_LARGE_OBJECT_ACTIONS = {
    "refrigerator": "Please open the refrigerator",
    "oven": "Please open the oven",
    "microwave": "Please open the microwave",
    "tv": "Please turn on the television",
    "laptop": "Please open the laptop",
    "door": "Please open the door",
    "sink": "Please turn on the tap",
    "toilet": "I need to use the toilet",
    "chair": "I would like to sit on the chair",
    "couch": "I would like to sit on the couch",
    "bed": "I would like to lie down",
}


def _distance_phrase(cm):
    """Coarse, human distance band for speech ('nearby' / 'far away')."""
    if cm is None:
        return None
    if cm <= 80:
        return "very close to me"
    if cm <= 150:
        return "nearby"
    if cm <= 300:
        return "a little way off"
    return "far away"


def _communicate_sentence(name, zone_i, depth_cm, rel):
    """Natural sentence for the double-blink 'tell people what I see' mode.

    Order follows the spec: object name, distance band, body-relative zone,
    then its relation to a neighbouring object.  Uses the demonstrative
    'that' so it reads as pointing the object out, not just describing it.
    e.g. "That bottle, nearby on my right-hand side, next to the cup."
    """
    # "That bottle, ..." when confident; "Thing that might be a bottle, ..."
    # when hedged (avoid the clumsy "That thing that might be ...").
    if name.startswith("thing that might be"):
        opener = name[0].upper() + name[1:]
    else:
        opener = f"That {name}"
    rest = []
    dist = _distance_phrase(depth_cm)
    if dist:
        rest.append(dist)
    rest.append(ZONE_NATURAL[zone_i])
    sentence = opener + ", " + " ".join(rest)
    if rel:
        relw = _RELATION_WORDS.get(rel[0], rel[0])
        sentence += f", {relw} the {rel[1]['class_name']}"
    return sentence + "."


def _robot_command(name, graspable, reason, depth_cm):
    """Speech + intent for the triple-blink 'do something' mode.

    Returns ``(text, do_pick)``.  ``do_pick`` is True only when the arm can
    actually reach and grasp the object; otherwise we either voice an action
    for big fixed objects or ask a nearby person to hand it over.
    """
    if graspable:
        return f"Grabbing the {name}.", True
    # Too big for the gripper but a person interacts with it → voice an action.
    if name in _LARGE_OBJECT_ACTIONS:
        return _LARGE_OBJECT_ACTIONS[name] + ".", False
    # Out of reach / too close / too wide → ask someone to bring it over.
    deixis = "this" if (depth_cm is not None and depth_cm <= 120) else "that"
    return f"I want {deixis} {name}, please.", False


def _draw_dashed_line(canvas, p0, p1, color, dash=8, gap=6, thickness=1):
    """Draw a dashed line from p0 to p1 (visualises the gaze→snap pull)."""
    x0, y0 = p0
    x1, y1 = p1
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length < 1:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    step = dash + gap
    d = 0.0
    while d < length:
        sx = int(x0 + ux * d)
        sy = int(y0 + uy * d)
        e = min(d + dash, length)
        ex = int(x0 + ux * e)
        ey = int(y0 + uy * e)
        cv2.line(canvas, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        d += step


def _snap_point(gx, gy, detections, strength):
    """Magnetically pull the gaze point toward the nearest object.

    Returns ``(bx, by, target_det)`` — the snapped ('blue') point and the
    object it locked onto (``None`` if nothing is in range).  ``strength``
    in [0, 1] is the live-tunable magnetism: 0 leaves the point on the raw
    gaze, 1 pins it to the object's centre.  An object whose bbox already
    contains the gaze always wins; otherwise the nearest centre within a
    capture radius that grows with strength is used.
    """
    if strength <= 0.0 or not detections:
        return gx, gy, None
    # Prefer an object directly under the gaze.
    inside = None
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if x1 <= gx <= x2 and y1 <= gy <= y2:
            inside = det
            break
    target = inside
    if target is None:
        # Otherwise grab the nearest centre within the capture radius.
        capture_r = 40.0 + strength * 220.0
        best_d = capture_r
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            if d < best_d:
                best_d, target = d, det
    if target is None:
        return gx, gy, None
    x1, y1, x2, y2 = target["bbox"]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bx = gx + (cx - gx) * strength
    by = gy + (cy - gy) * strength
    return bx, by, target


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


def _v4l2_set(index, **ctrls):
    """Best-effort set of V4L2 controls on /dev/video{index} via v4l2-ctl.

    Returns True if applied.  Silently no-ops when v4l2-ctl is missing or the
    device path doesn't exist (e.g. non-Linux), so the camera still works.
    """
    if not ctrls or shutil.which("v4l2-ctl") is None:
        return False
    dev = f"/dev/video{index}"
    if not os.path.exists(dev):
        return False
    spec = ",".join(f"{k}={v}" for k, v in ctrls.items())
    try:
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", spec],
                       capture_output=True, timeout=2)
        return True
    except Exception:
        return False


def _open_face_cam(index, exposure=-1):
    """Open the face webcam at 640x480 / 30 fps (MJPG), bright AND smooth.

    Low FPS is a camera problem, not a CV one.  UVC webcams default to
    auto-exposure with *dynamic framerate* enabled, so in indoor light the
    driver lengthens exposure and drops the sensor to ~7.5 fps — capping the
    whole pipeline.  The fix is to keep auto-exposure (so brightness stays
    correct) but disable the dynamic-framerate priority bit, which holds a
    constant 30 fps.  That bit isn't exposed via OpenCV, so we set it with
    v4l2-ctl (``exposure_dynamic_framerate=0``).

    ``exposure``: -1 (default) = auto-exposure + fixed 30 fps (recommended).
    A value >= 0 forces *manual* exposure to that V4L2 value (darker, only
    if you specifically want a fixed exposure).
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if exposure is not None and exposure >= 0:
        # Manual exposure: fixed framerate, brightness fixed by the value.
        _v4l2_set(index, auto_exposure=1, exposure_time_absolute=int(exposure))
    else:
        # Auto-exposure (Aperture Priority) but pinned to a constant 30 fps.
        _v4l2_set(index, auto_exposure=3, exposure_dynamic_framerate=0)
    return cap


PANEL_W = 268  # left panel width (px)
MODELS_DIR = Path(__file__).resolve().parent / "models"


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
                     eye_tex_id=None, stereo=False, cal_frac=None,
                     detector=None):
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
        if imgui.button("Re-center Gaze  (G)", -1, 38):
            action = 'recenter'
        if imgui.button("Re-calibrate Gaze  (C)", -1, 38):
            action = 'gaze_cal'
        if imgui.button("Calibrate Cameras  (V)", -1, 38):
            action = 'cam_cal'

        s = show_depth
        _, s = imgui.checkbox("Depth overlay  (D)", s)
        if s != show_depth:
            action = ('toggle_depth', s)

        imgui.spacing()

        # ── Gaze smoothing: tame red-cursor jitter (One-Euro filter) ──
        imgui.text_colored("Gaze smoothing", 0.20, 0.78, 1.0, 1.0)
        changed, v = imgui.slider_float(
            "##gazesmooth", OPTS["gaze_smooth"], 0.0, 1.0, "%.2f")
        if changed:
            OPTS["gaze_smooth"] = v

        imgui.spacing()

        # ── Blue snap point: how hard the cursor is pulled onto objects ──
        imgui.text_colored("Snap strength", 0.20, 0.78, 1.0, 1.0)
        changed, v = imgui.slider_float(
            "##snap", OPTS["snap_strength"], 0.0, 1.0, "%.2f")
        if changed:
            OPTS["snap_strength"] = v

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

        # ── Object detection (model + device + sensitivity) ─────────────
        if detector is not None and imgui.collapsing_header("Detection")[0]:
            dev_on_gpu = str(detector.device).startswith("cuda")
            dev_rgb = (0.39, 0.78, 0.0) if dev_on_gpu else (1.0, 0.65, 0.0)
            imgui.text_colored(f"Device: {detector.device}", *dev_rgb, 1.0)

            models = available_models(MODELS_DIR)
            labels = [m[0] for m in models]
            cur = next((i for i, m in enumerate(models)
                        if Path(m[1]).stem == detector.model_name), 0)
            changed, idx = imgui.combo("Model", cur, labels)
            if changed and idx != cur:
                action = ('set_model', models[idx][1])
            if models[cur][2] and not dev_on_gpu:
                imgui.text_colored("open-vocab is slow on CPU", 0.95, 0.40, 0.40, 1.0)

            changed, v = imgui.slider_float(
                "Confidence", float(detector.confidence), 0.05, 0.9, "%.2f")
            if changed:
                detector.confidence = v

            # ── Box stability (anti-flicker / anti-jitter) ──────────────
            imgui.spacing()
            _, on = imgui.checkbox("Stabilize boxes", detector.stabilize)
            detector.stabilize = on
            stab = getattr(detector, "_stabilizer", None)
            if on and stab is not None:
                # Lower smoothing = steadier box (more lag); higher = snappier.
                changed, v = imgui.slider_float(
                    "Smoothing", float(stab.smooth), 0.1, 1.0, "%.2f")
                if changed:
                    stab.smooth = v
                changed, v = imgui.slider_int("Keep frames", int(stab.max_age), 0, 10)
                if changed:
                    stab.max_age = v
                changed, v = imgui.slider_int("Confirm frames", int(stab.min_hits), 1, 5)
                if changed:
                    stab.min_hits = v
                imgui.text_colored("lower Smoothing = steadier", 0.50, 0.53, 0.60, 1.0)

        # ── Stereo tuning (only when the depth source is the stereo pair) ──
        if stereo:
            from modules.stereo_depth_estimator import STEREO_OPTS
            if imgui.collapsing_header("Stereo settings")[0]:
                imgui.text_colored("Tune until distances look right",
                                   0.50, 0.53, 0.60, 1.0)
                changed, v = imgui.slider_int(
                    "Focal px", int(STEREO_OPTS["focal_px"]), 200, 1200)
                if changed:
                    STEREO_OPTS["focal_px"] = v
                changed, v = imgui.slider_float(
                    "Baseline cm", float(STEREO_OPTS["baseline_cm"]), 1.0, 30.0)
                if changed:
                    STEREO_OPTS["baseline_cm"] = v
                changed, v = imgui.slider_int(
                    "Disparities", int(STEREO_OPTS["num_disp"]), 16, 256)
                if changed:
                    STEREO_OPTS["num_disp"] = v
                changed, v = imgui.slider_int(
                    "Block size", int(STEREO_OPTS["block_size"]), 5, 21)
                if changed:
                    STEREO_OPTS["block_size"] = v

        # ── Help / shortcuts ────────────────────────────────────────────
        if imgui.collapsing_header("Shortcuts")[0]:
            imgui.text_colored("G", 0.20, 0.78, 1.0, 1.0)
            imgui.same_line(40); imgui.text("Re-center gaze (drift fix)")
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
        card_h = 118 if cal_frac is not None else 92
        imgui.begin_child("##status", PANEL_W - 28, card_h, border=True)
        imgui.text_colored("CALIBRATING", 0.20, 0.78, 1.0, 1.0)
        imgui.spacing()
        if cal_step:
            imgui.text_wrapped(str(cal_step))
        if cal_frac is not None:
            imgui.spacing()
            imgui.progress_bar(max(0.0, min(1.0, cal_frac)), (PANEL_W - 44, 18))
        if cal_progress:
            imgui.text_colored(str(cal_progress), 0.50, 0.53, 0.60, 1.0)
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


def _draw_cal_overlay(canvas, idx, total, collected, tx, ty, w_s, h_s,
                      capturing=False):
    """On-canvas calibration wizard: progress bar, captured dots, big target."""
    # Top progress bar + "Point N / total"
    bx, by, bw = 40, 22, w_s - 80
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 14), (60, 60, 80), -1)
    fill = int(bw * (idx / max(1, total)))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + 14),
                  Theme.ACCENT_GREEN, -1)
    draw_text_stroke(canvas, f"Point {idx + 1} / {total}", (bx, by + 40),
                     scale=0.7, color=Theme.ACCENT_CYAN, thickness=2)

    # Already-captured points (faint green checks)
    for pt in collected:
        cv2.circle(canvas, (pt[0], pt[1]), 6, Theme.ACCENT_GREEN, -1)
        cv2.circle(canvas, (pt[0], pt[1]), 10, (255, 255, 255), 1)

    # Pulsing target (green while capturing, cyan while waiting)
    pulse = abs((time.time() * 2) % 2 - 1)  # 0..1 triangle wave
    r = int(28 + 7 * pulse)
    col = Theme.ACCENT_GREEN if capturing else Theme.ACCENT_CYAN
    cv2.circle(canvas, (tx, ty), r, col, -1)
    cv2.circle(canvas, (tx, ty), r + 6, (255, 255, 255), 2)
    cv2.line(canvas, (tx - 18, ty), (tx + 18, ty), (255, 255, 255), 1)
    cv2.line(canvas, (tx, ty - 18), (tx, ty + 18), (255, 255, 255), 1)

    # Big instruction near the target
    msg = ("Capturing... keep eyes on the real spot, rotate head" if capturing
           else "Look at this spot in front of you, press ENTER")
    (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    mx = min(max(tx - mw // 2, 12), w_s - mw - 12)
    my = ty - r - 16 if ty - r - 16 - mh > 70 else ty + r + 34
    draw_text_stroke(canvas, msg, (mx, my), scale=0.7,
                     color=(255, 255, 255), thickness=2)


def _calibrate_two_cam(gaze_estimator, cap_face, depth_estimator, gui,
                       capture_frames=90, grid_cols=3, grid_rows=3,
                       calib_face=None):
    """Grid calibration rendered through GUIOverlay + ImGui panels.
    Scene feed comes from PrimeSense via depth_estimator.

    ``calib_face`` (CameraCalibrator) is applied to every face frame here so
    training sees the SAME undistorted image the main loop feeds at runtime —
    otherwise the model is trained and used on differently-warped faces.
    """
    def _prep(frame):
        f = cv2.flip(frame, 1)
        return calib_face.undistort(f) if calib_face is not None else f

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
        "You are mapping your gaze onto the real space in",
        "front of you - exactly like during normal use.",
        "The dots mark points OUT IN THAT SPACE, not the screen.",
        "",
        "1)  Look at each point in front of you, then press ENTER",
        "2)  Keep your eyes ON that spot while it captures",
        "3)  Slowly rotate AND lean your head a little",
        "      (this is what makes tracking robust)",
        "",
        "A quick accuracy check follows the 9 points.",
        "",
        "ENTER = start        ESC = cancel",
    ]

    def _render_frame(canvas, face_disp, cal_step, cal_progress, instructions,
                      cal_frac=None):
        gui.update_scene_texture(canvas)
        if face_disp is not None:
            gui.update_face_texture(face_disp)
        gui.begin_frame()
        _draw_main_menu(gui, cal_mode='calibrating',
                        cal_step=cal_step, cal_progress=cal_progress,
                        face_tex_id=gui.face_texture_id,
                        depth_tex_id=gui.depth_texture_id, cal_frac=cal_frac)
        _draw_instructions(instructions, gui.height)
        gui.render()

    # ── Phase 1: Guide screen ─────────────────────────────────────────
    while True:
        scene, _ = depth_estimator.get_frame()
        if scene is None:
            continue
        frame_s = scene
        canvas = np.zeros((h_s, w_s, 3), dtype=np.uint8)
        draw_text_stroke(canvas, "Gaze Calibration", (70, 70),
                         scale=1.1, color=Theme.ACCENT_CYAN, thickness=3)
        draw_text_stroke(canvas, "9 points", (72, 104),
                         scale=0.6, color=(180, 180, 180), thickness=1)
        for i, txt in enumerate(guide_items):
            cv2.putText(canvas, txt, (74, 150 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (210, 210, 210), 2,
                        cv2.LINE_AA)
        _render_frame(canvas, None, "Ready", "Press ENTER to start",
                      ["Read the steps, then press ENTER to begin."])
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
        frame_face = _prep(frame_f)
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
            frame_face = _prep(frame_f)
            canvas = frame_s.copy()

            _draw_cal_overlay(canvas, idx, len(targets), collected,
                              tx, ty, w_s, h_s, capturing=False)

            instructions = [
                "Look at the marked spot in the space ahead of you,",
                "then press ENTER - not the screen, the real point.",
                "",
                "ENTER = capture   BACKSPACE = undo   ESC = finish",
            ]
            _render_frame(canvas, frame_face,
                          "Look at the point in front of you, press ENTER",
                          f"{len(collected)} samples",
                          instructions, cal_frac=idx / len(targets))
            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                return False

            if gui.was_key_pressed(glfw.KEY_ENTER):
                for _ci in range(capture_frames):
                    ret_f2, frame_f2 = cap_face.read()
                    if not ret_f2:
                        continue
                    face_f2 = _prep(frame_f2)
                    ft, blink = gaze_estimator.extract_features(face_f2)
                    if ft is not None and not blink:
                        collected.append([tx, ty, ft])
                    # Re-upload the LIVE face frame every few iters so the
                    # preview keeps moving (don't freeze the whole burst).
                    if _ci % 3 == 0:
                        cap_canvas = frame_s.copy()
                        _draw_cal_overlay(cap_canvas, idx, len(targets),
                                          collected, tx, ty, w_s, h_s,
                                          capturing=True)
                        gui.update_scene_texture(cap_canvas)
                        gui.update_face_texture(face_f2)
                        gui.begin_frame()
                        _draw_main_menu(
                            gui, cal_mode='calibrating',
                            cal_step="Capturing - eyes on the real spot, rotate & lean",
                            cal_progress=f"{_ci + 1}/{capture_frames} frames",
                            face_tex_id=gui.face_texture_id,
                            depth_tex_id=gui.depth_texture_id,
                            cal_frac=(_ci + 1) / capture_frames)
                        gui.render()
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
    # Outlier rejection: drop the worst-fitting ~10% of samples (micro-
    # saccades / half-blinks) and refit — steadier model, no visible cost.
    if len(feats) >= 20:
        try:
            pred = np.asarray(gaze_estimator.predict(feats))
            err = np.linalg.norm(pred - targs, axis=1)
            keep = err <= np.quantile(err, 0.90)
            if 10 <= int(keep.sum()) < len(feats):
                gaze_estimator.train(feats[keep], targs[keep])
                print(f"[FoGaze] Refit after dropping "
                      f"{len(feats) - int(keep.sum())} outliers "
                      f"(median err {np.median(err):.0f}px).")
        except Exception as e:
            print(f"[FoGaze] Outlier refit skipped: {e}")
    print("[FoGaze] Calibration complete.")

    # ── Accuracy check; offer a redo if it came out poor ──────────────
    verdict = _validate_calibration(gaze_estimator, cap_face, depth_estimator,
                                    gui, calib_face=calib_face)
    if verdict == 'redo':
        return _calibrate_two_cam(
            gaze_estimator, cap_face, depth_estimator, gui,
            capture_frames=capture_frames, grid_cols=grid_cols,
            grid_rows=grid_rows, calib_face=calib_face)
    return True


def _recenter_gaze(gaze_estimator, cap_face, depth_estimator, gui,
                   calib_face=None):
    """Quick 1-point drift correction during a session.

    User looks at a centre dot; we measure where the model *thinks* they look
    and return the offset (true_centre - predicted) to add to every later
    prediction.  Returns ``(off_x, off_y)`` or ``None`` if cancelled/failed.
    """
    def _prep(frame):
        f = cv2.flip(frame, 1)
        return calib_face.undistort(f) if calib_face is not None else f

    scene, _ = depth_estimator.get_frame()
    if scene is None:
        return None
    h_s, w_s = scene.shape[:2]
    cx, cy = w_s // 2, h_s // 2
    preds = []
    t0 = time.time()
    DWELL, COLLECT = 0.6, 1.2
    while time.time() - t0 < DWELL + COLLECT:
        scn, _ = depth_estimator.get_frame()
        ret_f, frame_f = cap_face.read()
        if scn is None or not ret_f:
            continue
        frame_face = _prep(frame_f)
        feats, blink = gaze_estimator.extract_features(frame_face)
        elapsed = time.time() - t0
        collecting = elapsed >= DWELL
        if collecting and feats is not None and not blink:
            try:
                p = gaze_estimator.predict(np.array([feats]))[0]
                preds.append((float(p[0]), float(p[1])))
            except Exception:
                pass
        canvas = scn.copy()
        col = (0, 230, 0) if collecting else (40, 200, 255)
        frac = min(1.0, elapsed / (DWELL + COLLECT))
        cv2.circle(canvas, (cx, cy), 28, col, -1)
        cv2.circle(canvas, (cx, cy), 36, (255, 255, 255), 2)
        cv2.ellipse(canvas, (cx, cy), (46, 46), -90, 0, int(360 * frac), col, 3)
        gui.update_scene_texture(canvas)
        if frame_face is not None:
            gui.update_face_texture(frame_face)
        gui.begin_frame()
        _draw_main_menu(gui, cal_mode='calibrating', cal_step="Re-centering",
                        cal_progress="Look at the center point",
                        face_tex_id=gui.face_texture_id,
                        depth_tex_id=gui.depth_texture_id)
        _draw_instructions(["Drift re-center",
                            "Look at the center spot in front of you",
                            "", "ESC = cancel"], gui.height)
        gui.render()
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            return None
    if len(preds) < 5:
        return None
    mx = float(np.median([p[0] for p in preds]))
    my = float(np.median([p[1] for p in preds]))
    ox = max(-0.4 * w_s, min(0.4 * w_s, cx - mx))
    oy = max(-0.4 * h_s, min(0.4 * h_s, cy - my))
    print(f"[FoGaze] Re-center offset = ({ox:.0f}, {oy:.0f})px")
    return (ox, oy)


def _validate_calibration(gaze_estimator, cap_face, depth_estimator, gui,
                          calib_face=None):
    """Show a few test dots, measure gaze error, and let the user accept/redo.

    Returns 'ok' (accept) or 'redo'.  Each point is captured on ENTER (same
    as calibration): look at the spot, press ENTER, then ~1s of predictions
    whose median gives the error.  Mean error is graded GOOD / OK / POOR.
    """
    def _prep(frame):
        f = cv2.flip(frame, 1)
        return calib_face.undistort(f) if calib_face is not None else f

    scene, _ = depth_estimator.get_frame()
    if scene is None:
        return 'ok'
    h_s, w_s = scene.shape[:2]
    test_pts = [(0.5, 0.5), (0.22, 0.22), (0.78, 0.22),
                (0.22, 0.78), (0.78, 0.78)]
    targets = [(int(w_s * fx), int(h_s * fy)) for fx, fy in test_pts]
    COLLECT = 1.1
    errors = []

    for ti, (tx, ty) in enumerate(targets):
        # ── Wait for the user to fixate the real spot and press ENTER ──
        armed = False
        while not armed:
            scn, _ = depth_estimator.get_frame()
            ret_f, frame_f = cap_face.read()
            if scn is None or not ret_f:
                continue
            frame_face = _prep(frame_f)
            canvas = scn.copy()
            cv2.circle(canvas, (tx, ty), 26, (40, 200, 255), -1)
            cv2.circle(canvas, (tx, ty), 34, (255, 255, 255), 2)
            cv2.line(canvas, (tx - 18, ty), (tx + 18, ty), (255, 255, 255), 1)
            cv2.line(canvas, (tx, ty - 18), (tx, ty + 18), (255, 255, 255), 1)
            gui.update_scene_texture(canvas)
            gui.update_face_texture(frame_face)
            gui.begin_frame()
            _draw_main_menu(gui, cal_mode='calibrating',
                            cal_step=f"Accuracy check {ti + 1}/{len(targets)}",
                            cal_progress="Look at the point, press ENTER",
                            face_tex_id=gui.face_texture_id,
                            depth_tex_id=gui.depth_texture_id)
            _draw_instructions([
                "Accuracy check",
                "Look at the marked spot in the space ahead,",
                "then press ENTER - just like calibration.",
                "", "ENTER = check point   ESC = skip"], gui.height)
            gui.render()
            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                return 'ok'
            if gui.was_key_pressed(glfw.KEY_ENTER):
                armed = True

        # ── Collect ~1s of predictions while the user holds their gaze ──
        preds = []
        t0 = time.time()
        while time.time() - t0 < COLLECT:
            scn, _ = depth_estimator.get_frame()
            ret_f, frame_f = cap_face.read()
            if scn is None or not ret_f:
                continue
            frame_face = _prep(frame_f)
            feats, blink = gaze_estimator.extract_features(frame_face)
            if feats is not None and not blink:
                try:
                    p = gaze_estimator.predict(np.array([feats]))[0]
                    preds.append((float(p[0]), float(p[1])))
                except Exception:
                    pass

            canvas = scn.copy()
            frac = min(1.0, (time.time() - t0) / COLLECT)
            cv2.circle(canvas, (tx, ty), 26, (0, 230, 0), -1)
            cv2.circle(canvas, (tx, ty), 34, (255, 255, 255), 2)
            cv2.ellipse(canvas, (tx, ty), (44, 44), -90, 0,
                        int(360 * frac), (0, 230, 0), 3)
            if preds:
                px = int(np.median([p[0] for p in preds]))
                py = int(np.median([p[1] for p in preds]))
                cv2.circle(canvas, (px, py), 9, (0, 180, 255), 2)
                cv2.line(canvas, (px, py), (tx, ty), (0, 180, 255), 1)
            gui.update_scene_texture(canvas)
            if frame_face is not None:
                gui.update_face_texture(frame_face)
            gui.begin_frame()
            _draw_main_menu(gui, cal_mode='calibrating',
                            cal_step=f"Accuracy check {ti + 1}/{len(targets)}",
                            cal_progress="Hold your gaze...",
                            face_tex_id=gui.face_texture_id,
                            depth_tex_id=gui.depth_texture_id)
            _draw_instructions(["Accuracy check",
                                "Keep looking at the spot in front of you...",
                                "", "ESC = skip check"], gui.height)
            gui.render()
            if gui.was_key_pressed(glfw.KEY_ESCAPE):
                return 'ok'

        if preds:
            mx = float(np.median([p[0] for p in preds]))
            my = float(np.median([p[1] for p in preds]))
            errors.append(((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5)

    if not errors:
        return 'ok'
    mean_err = float(np.mean(errors))
    ref = min(w_s, h_s)
    if mean_err < 0.06 * ref:
        grade, gc = "GOOD", (0.39, 0.78, 0.0)
    elif mean_err < 0.11 * ref:
        grade, gc = "OK", (0.95, 0.75, 0.25)
    else:
        grade, gc = "POOR", (0.95, 0.40, 0.40)

    # ── Result screen (big, clear, two buttons) ───────────────────────
    while True:
        scn, _ = depth_estimator.get_frame()
        canvas = (scn.copy() if scn is not None
                  else np.zeros((h_s, w_s, 3), np.uint8))
        gui.update_scene_texture(canvas)
        gui.begin_frame()
        _draw_main_menu(gui, cal_mode='calibrating', cal_step="Result",
                        cal_progress="", face_tex_id=gui.face_texture_id,
                        depth_tex_id=gui.depth_texture_id)
        imgui.set_next_window_size(380, 230)
        imgui.set_next_window_position(gui.width // 2 - 190 + PANEL_W // 2,
                                       gui.height // 2 - 115)
        imgui.begin("Calibration Result", None,
                    imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE |
                    imgui.WINDOW_NO_MOVE)
        imgui.text("Calibration accuracy")
        imgui.spacing()
        imgui.text_colored(grade, *gc, 1.0)
        imgui.same_line()
        imgui.text(f"   avg error {mean_err:.0f}px")
        imgui.spacing()
        imgui.text_colored(
            "If POOR: redo and keep eyes on the real spot while",
            0.50, 0.53, 0.60, 1.0)
        imgui.text_colored("slowly rotating + leaning your head.",
                           0.50, 0.53, 0.60, 1.0)
        imgui.spacing()
        accept = imgui.button("Accept  (Enter)", 175, 44)
        imgui.same_line()
        redo = imgui.button("Redo  (R)", 175, 44)
        imgui.end()
        gui.render()
        if accept or gui.was_key_pressed(glfw.KEY_ENTER):
            return 'ok'
        if redo or gui.was_key_pressed(glfw.KEY_R):
            return 'redo'
        if gui.was_key_pressed(glfw.KEY_ESCAPE):
            return 'ok'


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


# Landmarks the gaze model ACTUALLY consumes (imported from eyetrax so the
# overlay can never drift from the real feature set). extract_features() feeds
# LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES (head-pose normalised),
# and normalises around anchors 33/263 (eye corners) and 10 (top of head).
from eyetrax.constants import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MUTUAL_INDICES,
)

_LEFT_IRIS_RING = [468, 469, 470, 471, 472]
_RIGHT_IRIS_RING = [473, 474, 475, 476, 477]
_IRIS_RING = _LEFT_IRIS_RING + _RIGHT_IRIS_RING
_ANCHOR_CORNERS = (33, 263)   # inter-eye x-axis anchors
_ANCHOR_TOP = 10              # up (y-axis) anchor
# Eye/brow feature points minus the iris (iris drawn separately, highlighted).
_EYE_FEATURE_PTS = [i for i in (LEFT_EYE_INDICES + RIGHT_EYE_INDICES)
                    if i not in _IRIS_RING]
_LEFT_IRIS = 468
_RIGHT_IRIS = 473


def _align_face_and_eyes(face_bgr, landmarks, h_f, w_f):
    """Overlay the REAL landmarks the gaze model uses + extract eye crops.

    Every point drawn here is a landmark actually fed to / used by
    extract_features():
      green  = eye/brow feature points (LEFT/RIGHT_EYE_INDICES)
      cyan   = iris ring (468-472 / 473-477) — drives gaze direction
      orange = head-pose reference points (MUTUAL_INDICES)
      magenta= normalisation frame: anchors 33/263 (x) + 10 (y) and its axes

    Returns: (face_display, eye_display) or (display, None).
    """
    display = face_bgr.copy()

    def _P(idx):
        return (int(landmarks[idx].x * w_f), int(landmarks[idx].y * h_f))

    # Eye/brow feature points (green) — the bulk of the feature vector.
    for idx in _EYE_FEATURE_PTS:
        cv2.circle(display, _P(idx), 1, (90, 230, 90), -1)

    # Head-pose reference points (orange).
    for idx in MUTUAL_INDICES:
        cv2.circle(display, _P(idx), 2, (40, 170, 255), -1)

    # Iris ring (cyan) — gaze direction.
    for idx in _IRIS_RING:
        cv2.circle(display, _P(idx), 2, (255, 255, 0), -1)

    # Normalisation frame (magenta): inter-eye x-axis + up y-axis from anchors.
    la, ra = _P(_ANCHOR_CORNERS[0]), _P(_ANCHOR_CORNERS[1])
    top = _P(_ANCHOR_TOP)
    eye_center = ((la[0] + ra[0]) // 2, (la[1] + ra[1]) // 2)
    cv2.line(display, la, ra, (255, 0, 255), 1)
    cv2.line(display, eye_center, top, (255, 0, 255), 1)
    for p in (la, ra, top):
        cv2.circle(display, p, 3, (255, 0, 255), -1)

    # ── Extract L/R eye regions (crop already carries the real overlay) ──
    def _get_eye_roi(corner_a, corner_b):
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
        return crop

    l_eye = _get_eye_roi(33, 133)
    r_eye = _get_eye_roi(362, 263)

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
        description="FoGaze — gaze-driven object focus with blink-to-grasp "
                    "robot pickup (gaze + YOLO + depth + ROS 2/MoveIt)"
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
                        choices=["one_euro", "kalman", "kalman_ema",
                                 "kde", "none"],
                        default="one_euro",
                        help="Gaze smoothing filter (one_euro = adaptive, "
                             "best for jitter; tune live in the left panel)")
    parser.add_argument("--ema-alpha", type=float, default=0.5,
                        help="EMA smoothing (0=off, 1=max smooth). Lower = "
                             "snappier/less drift; higher = smoother/laggier.")
    parser.add_argument("--kde-confidence", type=float, default=0.5)
    parser.add_argument("--blink-ratio", type=float, default=0.6,
                        help="Blink sensitivity: gaze is suppressed when "
                             "eye-openness < ratio*average. Lower = less "
                             "twitchy (default 0.6; eyetrax default 0.8).")
    parser.add_argument("--blink-count", type=int, default=3,
                        help="Blinks within ~1.6s for the robot-command "
                             "gesture (default 3, min 3). A double-blink (2) "
                             "always triggers the spoken description.")
    parser.add_argument("--face-lock", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Zoom/track the face region for sharper landmarks "
                             "(default on; --no-face-lock to disable).")
    parser.add_argument("--yolo-model",
                        default=str(Path(__file__).resolve().parent
                                    / "models" / "yolov8n.pt"),
                        help="YOLO weights path")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="YOLO confidence threshold (lower = sees more)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="YOLO NMS IoU (higher = keeps more overlapping boxes)")
    parser.add_argument("--detection-interval", type=int, default=3,
                        help="YOLO inference every N frames")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="YOLO inference size (smaller = faster)")
    parser.add_argument("--device", default="auto",
                        help="YOLO device: auto | cpu | cuda | cuda:0")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--ros", action="store_true",
                        help="Publish detections + triple-blink pickups to "
                             "ROS 2 (fogaze/objects, fogaze/pickup)")
    parser.add_argument("--sim", action="store_true",
                        help="Simulation mode: use a fixed synthetic scene + "
                             "depth instead of the PrimeSense (no hardware)")
    parser.add_argument("--depth-source",
                        choices=["stereo", "primesense", "sim"],
                        default="stereo",
                        help="Scene+depth source (default stereo). Overridden "
                             "by the picker on the start screen; --sim forces "
                             "sim.")
    parser.add_argument("--stereo-left", type=int, default=None,
                        help="Left camera index for stereo depth")
    parser.add_argument("--stereo-right", type=int, default=None,
                        help="Right camera index for stereo depth")
    parser.add_argument("--face-exposure", type=int, default=-1,
                        help="-1 (default) = auto-exposure pinned to 30fps "
                             "(bright + smooth). A value >=0 forces a fixed "
                             "manual exposure (darker; only for special cases).")
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

    # ── Depth-source picker (default Stereo) ──────────────────────────
    DEPTH_SOURCES = ["Stereo (2 cams)", "PrimeSense", "Simulation"]
    src_idx = 2 if args.sim else {"stereo": 0, "primesense": 1, "sim": 2}.get(
        args.depth_source, 0)
    face_cam_idx = 0
    # Sensible default stereo pair = first two cams that aren't the face cam.
    others = [i for i in range(len(avail)) if i != face_cam_idx]
    stereoL_idx = (avail.index(args.stereo_left)
                   if args.stereo_left in avail else (others[0] if others else 0))
    stereoR_idx = (avail.index(args.stereo_right)
                   if args.stereo_right in avail
                   else (others[1] if len(others) > 1 else 0))
    face_lock_on = bool(args.face_lock)
    cam_selected = False
    while not cam_selected:
        gui.begin_frame()
        imgui.set_next_window_size(420, 320, imgui.ONCE)
        imgui.set_next_window_position(gui.width//2 - 210, gui.height//2 - 160,
                                        imgui.ONCE)
        imgui.begin("Camera Selection", None,
                    imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
        imgui.text_colored("FoGaze", 0.20, 0.78, 1.0, 1.0)
        imgui.text("Select the face camera + depth source.")
        imgui.spacing()
        camera_names = [f"Camera {i}" for i in avail]
        _, face_cam_idx = imgui.combo("Face Camera", face_cam_idx, camera_names)
        _, src_idx = imgui.combo("Depth Source", src_idx, DEPTH_SOURCES)
        if src_idx == 0:  # stereo → pick the two scene cameras
            _, stereoL_idx = imgui.combo("Left Cam", stereoL_idx, camera_names)
            _, stereoR_idx = imgui.combo("Right Cam", stereoR_idx, camera_names)
        _, face_lock_on = imgui.checkbox(
            "Face lock (zoom to face = sharper)", face_lock_on)
        imgui.spacing()
        imgui.text_colored(
            f"-> Face {avail[face_cam_idx]} + {DEPTH_SOURCES[src_idx]}"
            + ("  + lock" if face_lock_on else ""),
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
    sim_mode = (src_idx == 2)
    stereo_mode = (src_idx == 0)
    stereo_left = avail[stereoL_idx]
    stereo_right = avail[stereoR_idx]
    print(f"[FoGaze] Face cam={face_cam}  Scene={DEPTH_SOURCES[src_idx]}"
          + (f"  (L={stereo_left} R={stereo_right})" if stereo_mode else ""))

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
    gaze_estimator = GazeEstimator(model_name=args.model,
                                   blink_threshold_ratio=args.blink_ratio)

    # Face-camera undistortion — created here (before calibration) so the
    # SAME correction is applied during calibration and at runtime.
    calib_face = CameraCalibrator(face_cam)

    # ── Landmark capture + face-lock, wrapped at the gaze_estimator level
    # so EVERY caller (calibration, validation, runtime) gets the same face
    # crop transparently — no per-call-site changes, and train/infer stay
    # consistent.  _lm_cap also feeds the face/eye preview without a 2nd
    # MediaPipe pass. ───────────────────────────────────────────────────
    _lm_cap = {"result": None}
    face_lock = FaceLock(enabled=face_lock_on)
    if hasattr(gaze_estimator, "_face_landmarker"):
        _fl = gaze_estimator._face_landmarker
        _orig_detect = _fl.detect_for_video

        def _detect_capture(mp_image, ts_ms, _orig=_orig_detect, _cap=_lm_cap):
            r = _orig(mp_image, ts_ms)
            _cap["result"] = r
            return r

        _fl.detect_for_video = _detect_capture

    _orig_extract = gaze_estimator.extract_features

    def _extract_locked(frame, _orig=_orig_extract, _cap=_lm_cap,
                        _lock=face_lock):
        crop, _origin = _lock.crop(frame)
        feats, blink = _orig(crop)
        res = _cap["result"]
        if res is not None and res.face_landmarks:
            ch, cw = crop.shape[:2]
            _lock.update_from_landmarks(res.face_landmarks[0], cw, ch)
        else:
            _lock.miss()
        return feats, blink

    gaze_estimator.extract_features = _extract_locked

    # Load or calibrate
    if os.path.isfile(model_path):
        try:
            gaze_estimator.load_model(model_path)
            print(f"[FoGaze] Loaded model from {model_path}")
        except Exception as e:
            print(f"[FoGaze] Failed to load model ({e}), will calibrate.")

    if not _is_trained(gaze_estimator.model):
        print("[FoGaze] No valid model — starting calibration")

        cap_tmp_face = _open_face_cam(face_cam, args.face_exposure)
        if not cap_tmp_face.isOpened():
            raise RuntimeError("Cannot open face camera for calibration")

        ok = _calibrate_two_cam(
            gaze_estimator, cap_tmp_face, depth_estimator, gui,
            calib_face=calib_face,
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
    if args.filter == "one_euro":
        # Low beta + low dcutoff so jitter (which looks like fast motion) does
        # not defeat the smoothing; min_cutoff is re-set live from the slider.
        smoother = OneEuroSmoother(min_cutoff=0.45, beta=0.002, dcutoff=0.7)
    elif args.filter == "kalman_ema":
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
    cap_face = _open_face_cam(face_cam, args.face_exposure)

    if not cap_face.isOpened():
        raise RuntimeError(f"Cannot open face camera {face_cam}")

    for _ in range(10):
        cap_face.read()

    # ── Depth camera (also provides scene color) ─────────────────────
    if sim_mode:
        from modules.mock_depth_estimator import MockDepthEstimator
        depth_estimator = MockDepthEstimator()
    elif stereo_mode:
        from modules.stereo_depth_estimator import StereoDepthEstimator
        depth_estimator = StereoDepthEstimator(stereo_left, stereo_right)
    else:
        depth_estimator = DepthEstimator()
    # Warm up
    for _ in range(30):
        _, _ = depth_estimator.get_frame()

    # (calib_face already created above, before calibration)

    # ── Object detector (scene camera via PrimeSense color) ──────────
    detector = ObjectDetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        imgsz=args.imgsz,
        iou=args.iou,
        device=args.device,
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

    blink_detector = BlinkGestureDetector(min_count=2)
    command_blinks = max(3, args.blink_count)  # 3+ blinks = robot command
    speech = SpeechOutput()
    last_valid_gx = sw // 2
    last_valid_gy = sh // 2
    last_gaze_t = 0.0  # wall-clock of the last live gaze (for blink hold)
    # Reject blink-induced spikes before smoothing (thresholds scale with res).
    gaze_gate = RobustGazeGate(jump_px=0.22 * min(sw, sh),
                               confirm_px=0.14 * min(sw, sh))
    last_focus = None  # snapshot of the last focused object (for blink trigger)
    blue_pt = None     # smoothed magnetic snap point (EMA state)
    gaze_off_x = 0.0   # drift-recenter offset added to predictions (G key)
    gaze_off_y = 0.0

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
    _trigger_recenter = False

    # Optional ROS bridge for Rviz
    _ros_init()
    # (_lm_cap landmark capture + face_lock wrapper already installed above,
    # before calibration, so the framing is identical at calibrate + runtime.)

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
            now_t = time.time()

            if features is None or blink_detected:
                gaze_gate.accept(0.0, 0.0, True, now_t)   # flag blink/no-face
            else:
                try:
                    pred = gaze_estimator.predict(np.array([features]))[0]
                    raw_x = float(pred[0]) + gaze_off_x
                    raw_y = float(pred[1]) + gaze_off_y
                    acc = gaze_gate.accept(raw_x, raw_y, False, now_t)
                    if acc is not None:
                        gx, gy = acc
                        gaze_active = True
                except Exception:
                    pass

            if gaze_active:
                # Live-tune One-Euro: higher slider → lower cutoff → steadier.
                if isinstance(smoother, OneEuroSmoother):
                    sm = float(OPTS.get("gaze_smooth", 0.7))
                    smoother.set_params(min_cutoff=max(0.12, 1.5 * (1.0 - sm)))
                gx, gy = smoother.step(gx, gy)
                last_valid_gx, last_valid_gy = int(gx), int(gy)
                last_gaze_t = now_t

            gx_scene = int(gx)
            gy_scene = int(gy)

            # Hold the cursor at the last good spot through brief blinks /
            # dropouts (<0.6s) so a normal blink doesn't flicker it off.
            # Only after a sustained loss does it actually go inactive.
            gaze_held = gaze_active or (time.time() - last_gaze_t) < 0.6
            cursor.update(gx_scene, gy_scene, gaze_held)

            # ── Object detection (scene camera) ───────────────────────
            if sim_mode:
                detections = depth_estimator.mock_detections(frame_scene)
            else:
                detections = detector.detect(frame_scene)

            # ── Publish YOLO markers to ROS (for Rviz) ─────────────
            _ros_publish_markers(frame_scene, detections, depth_estimator)

            # ── Depth (already from PrimeSense get_frame) ─────────────
            focused_depth = None
            focused_depth_cm = None

            # ── Blue magnetic snap point ─────────────────────────────
            # The red cursor is the raw gaze; the blue point is sucked
            # toward the nearest object so focus / speech / actions lock on
            # precisely.  Strength is tuned live from the left panel.
            snap_strength = float(OPTS.get("snap_strength", 0.6))
            bx, by, focused = _snap_point(
                gx_scene, gy_scene,
                detections if gaze_held else [], snap_strength)
            # Smooth the blue point so the lock doesn't jitter frame-to-frame.
            if blue_pt is None:
                blue_pt = (bx, by)
            else:
                a = 0.55
                blue_pt = (a * blue_pt[0] + (1 - a) * bx,
                           a * blue_pt[1] + (1 - a) * by)
            bx_scene, by_scene = int(blue_pt[0]), int(blue_pt[1])

            if focused:
                x1, y1, x2, y2 = focused["bbox"]
                focused_depth = depth_estimator.depth_at_bbox(x1, y1, x2, y2)
                if focused_depth is not None:
                    focused_depth_cm = depth_estimator.depth_to_distance_cm(focused_depth)

            # ── Relationship of the focused object to its nearest 3D
            # neighbour (by depth distance, with a 2D-overlap fallback) ──
            focused_rel = None  # (direction_phrase, other_det, dist_m|None)
            if focused is not None:
                rel_intr = CameraIntrinsics(cx=w_scene / 2.0, cy=h_scene / 2.0)
                near = _nearest_by_depth(focused, detections,
                                         depth_estimator, rel_intr)
                if near is not None:
                    ob, dist_m, direction = near
                    focused_rel = (direction, ob, dist_m)
                else:
                    for db in detections:
                        if db is focused:
                            continue
                        rel = _get_relation(focused, db, w_scene, h_scene)
                        if rel:
                            focused_rel = (rel, db, None)
                            break

            # Remember the focus context briefly: a blink suppresses gaze, so
            # at trigger time `focused` may be gone — keep what the user was
            # looking at just before, so its NAME still gets spoken/picked.
            if focused is not None:
                last_focus = {"det": focused, "rel": focused_rel,
                              "depth_cm": focused_depth_cm, "t": time.time()}

            # ── Blink gesture: 2 = describe aloud, 3+ = command the robot ──
            blink_count = blink_detector.update(blink_detected, time.time())
            if blink_count:
                zi, _ = _zone_for(bx_scene, by_scene, w_scene, h_scene)
                # Fall back to the just-before-blink focus if gaze is currently
                # suppressed by the blink itself (so the name still gets used).
                trig = focused
                trig_rel = focused_rel
                trig_depth_cm = focused_depth_cm
                if trig is None and last_focus is not None and \
                        time.time() - last_focus["t"] < 2.0:
                    trig = last_focus["det"]
                    trig_rel = last_focus["rel"]
                    trig_depth_cm = last_focus["depth_cm"]

                is_command = blink_count >= command_blinks
                name = None
                zi_obj = zi  # direction phrase keys off the OBJECT's grid cell
                if trig is not None:
                    name = _hedged_label(trig)
                    ox = (trig['bbox'][0] + trig['bbox'][2]) / 2.0
                    oy = (trig['bbox'][1] + trig['bbox'][3]) / 2.0
                    zi_obj, _ = _zone_for(ox, oy, w_scene, h_scene)

                if trig is None:
                    # Nothing focused — at least voice where the user is looking.
                    speech.speak(f"Something {ZONE_NATURAL[zi]}.")
                elif not is_command:
                    # ── Double-blink: describe the object for people nearby. ──
                    sentence = _communicate_sentence(
                        name, zi_obj, trig_depth_cm, trig_rel)
                    speech.speak(sentence)
                    print(f"[Describe] {sentence!r}")
                else:
                    # ── Triple-blink: act on the object (grab / fetch / use). ──
                    intr = CameraIntrinsics(cx=w_scene / 2.0, cy=h_scene / 2.0)
                    if trig_depth_cm is not None:
                        z_m = trig_depth_cm / 100.0
                        size_wh = bbox_size_m(trig["bbox"], z_m, intr)
                        pose = project_bbox_to_3d(trig["bbox"], z_m, intr)
                        graspable, reason = classify_graspable(z_m, size_wh)
                    else:
                        z_m = size_wh = pose = None
                        graspable, reason = False, "no-depth"
                    text, do_pick = _robot_command(
                        name, graspable, reason, trig_depth_cm)
                    speech.speak(text)
                    print(f"[Command] {text!r} grab={do_pick} ({reason})")
                    # Hand the request to the manipulation stack; the executor
                    # gates on `graspable` and ignores out-of-reach targets.
                    if args.ros:
                        _ros_publish_pickup({
                            "class": trig["class_name"],
                            "confidence": trig["confidence"],
                            "pose": list(pose) if pose else None,
                            "size": list(size_wh) if size_wh else None,
                            "graspable": graspable, "reason": reason,
                        }, args.camera_frame)

            # ── Face camera display (separate window) ─────────────────
            # Shows the locked crop the model actually sees (follows + zooms
            # the face); landmarks captured above are in this crop's space.
            _lock_base = (face_lock.last_crop if face_lock.last_crop is not None
                          else frame_face)
            face_display = _lock_base.copy()
            eye_display = None
            _lm_result = _lm_cap["result"]
            if (features is not None and _lm_result is not None
                    and _lm_result.face_landmarks):
                try:
                    lm = _lm_result.face_landmarks[0]
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

            # 3×3 zone grid (center zone larger) — keyed to the blue snap point
            zi, (c1, c2, r1, r2) = _zone_for(
                bx_scene, by_scene, w_scene, h_scene)
            cv2.line(canvas, (c1, 0), (c1, h_scene), (60, 60, 80), 1)
            cv2.line(canvas, (c2, 0), (c2, h_scene), (60, 60, 80), 1)
            cv2.line(canvas, (0, r1), (w_scene, r1), (60, 60, 80), 1)
            cv2.line(canvas, (0, r2), (w_scene, r2), (60, 60, 80), 1)
            draw_text_stroke(canvas, ZONE_PHRASES[zi],
                             (12, h_scene - 60), scale=0.5,
                             color=Theme.ACCENT_CYAN, thickness=1)
            # Relationship text + depth
            if focused and focused_rel:
                fname = _hedged_label(focused)
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
                cv2.line(canvas, (bx_scene, by_scene), (cx_f, cy_f),
                         Theme.ACCENT_GREEN, 1, cv2.LINE_AA)

            # Gaze cursor (red) + dashed pull line to the blue snap point
            cursor.draw(canvas)
            _draw_dashed_line(canvas, (gx_scene, gy_scene),
                              (bx_scene, by_scene), Theme.ACCENT_BLUE)
            # Blue magnetic snap point: glow brighter when locked onto an object
            _r = 7 if focused is not None else 5
            cv2.circle(canvas, (bx_scene, by_scene), _r + 4,
                       Theme.ACCENT_BLUE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_scene, by_scene), _r,
                       Theme.ACCENT_BLUE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_scene, by_scene), _r,
                       (255, 255, 255), 1, cv2.LINE_AA)

            # Calibration notification
            if _is_trained(gaze_estimator.model) and not cal_notified:
                cal_notified = True

            # HUD values (rendered in the left panel, not on the scene canvas)
            cal_status = "CAL" if _is_trained(gaze_estimator.model) else "UNCAL"
            zi, _ = _zone_for(bx_scene, by_scene, w_scene, h_scene)
            fname = _hedged_label(focused) if focused else "--"
            rel_txt = (f"{fname} {focused_rel[0]} "
                       f"{focused_rel[1]['class_name']}"
                       ) if focused_rel else "--"
            depth_txt = (f"{int(focused_depth_cm)}cm"
                         if focused_depth_cm is not None else "--")
            tracker_txt = f"{args.model.upper()} | {args.filter.upper()}"
            gaze_txt = f"({gx_scene}, {gy_scene})"
            _scene_src = ("SIM" if sim_mode else
                          "Stereo" if stereo_mode else "PrimeSense")
            scene_txt = f"Scene:{_scene_src}  Face:{face_cam}"

            # Help hint
            if time.time() - help_t < 5:
                draw_text_stroke(
                    canvas,
                    "ESC: quit  |  g: re-center  |  c: re-calibrate  |  "
                    "v: cal cameras  |  d: depth",
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
                    stereo=stereo_mode,
                    detector=detector,
                )
                if action == 'quit':
                    _trigger_quit = True
                elif action == 'gaze_cal':
                    _trigger_recal = True
                elif action == 'cam_cal':
                    _trigger_cam_cal = True
                elif action == 'recenter':
                    _trigger_recenter = True
                elif isinstance(action, tuple) and action[0] == 'toggle_depth':
                    show_depth = action[1]
                elif isinstance(action, tuple) and action[0] == 'set_model':
                    topbar.toast(f"Loading {Path(action[1]).stem}...")
                    try:
                        detector.swap_model(action[1])
                        topbar.toast(f"Model: {detector.model_name} "
                                     f"({detector.device})")
                    except Exception as e:
                        print(f"[FoGaze] Model swap failed: {e}")
                        topbar.toast("Model load failed")

                gui.render()

            # ── Key events (GLFW keys also processed via GUI) ──────────
            if gui.was_key_pressed(glfw.KEY_ESCAPE) or _trigger_quit:
                print("[FoGaze] User quit.")
                break

            if gui.was_key_pressed(glfw.KEY_C) or _trigger_recal:
                print("[FoGaze] Re-calibrating gaze...")
                cap_face.release()

                cap_re_face = _open_face_cam(face_cam, args.face_exposure)
                if not cap_re_face.isOpened():
                    print("[FoGaze] Failed to open camera for re-calibration.")
                    break

                try:
                    ok = _calibrate_two_cam(
                        gaze_estimator, cap_re_face, depth_estimator, gui,
                        calib_face=calib_face,
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
                cap_face = _open_face_cam(face_cam, args.face_exposure)
                if not cap_face.isOpened():
                    print("[FoGaze] Failed to re-open camera.")
                    break

            elif gui.was_key_pressed(glfw.KEY_D):
                show_depth = not show_depth
                topbar.toast(
                    f"Depth overlay {'ON' if show_depth else 'OFF'}",
                    Theme.ACCENT_GREEN if show_depth else Theme.ACCENT_ORANGE,
                )

            elif gui.was_key_pressed(glfw.KEY_G) or _trigger_recenter:
                off = _recenter_gaze(gaze_estimator, cap_face, depth_estimator,
                                     gui, calib_face=calib_face)
                if off is not None:
                    gaze_off_x, gaze_off_y = off
                    topbar.toast("Gaze re-centered", Theme.ACCENT_GREEN)
                else:
                    topbar.toast("Re-center cancelled", Theme.ACCENT_ORANGE)

            elif gui.was_key_pressed(glfw.KEY_V) or _trigger_cam_cal:
                topbar.toast("Camera calibration...", Theme.ACCENT_CYAN)
                _calibrate_cameras(calib_face,
                                   cap_face, depth_estimator, gui)
                topbar.toast("Camera cal. done", Theme.ACCENT_GREEN)

            _trigger_quit = False
            _trigger_recal = False
            _trigger_cam_cal = False
            _trigger_recenter = False

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
