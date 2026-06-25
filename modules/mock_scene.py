"""Procedural mock scene for running FoGaze without a depth camera.

Renders a fixed 640x480 POV tabletop image (water bottle, glass, spoon,
fruit, chair) together with a matching depth map (millimetres) and an
object manifest.  Used by :class:`modules.mock_depth_estimator.MockDepthEstimator`
so the whole gaze -> YOLO -> blink -> grasp pipeline can be demoed with no
PrimeSense attached.

Run standalone to (re)generate the cached assets::

    python3 -m modules.mock_scene

Geometry note: the app projects bboxes with fx=fy=532, cx=320, cy=240 and
classifies an object graspable when 0.20 m <= z <= 0.85 m and the smaller
real-world side <= 0.08 m (see modules/grasp.py).  The distances/boxes below
are chosen so the four tabletop items come out graspable and the chair is
correctly rejected as out-of-reach.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


W, H = 640, 480

# Project-root /assets where the generated frame is cached.
_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
SCENE_PNG = os.path.join(_ASSETS_DIR, "mock_scene.png")
DEPTH_NPY = os.path.join(_ASSETS_DIR, "mock_scene_depth.npy")
OBJECTS_JSON = os.path.join(_ASSETS_DIR, "mock_scene_objects.json")


# ── Object manifest ─────────────────────────────────────────────────────────
# class_name uses COCO labels so the rest of the app treats them like real
# YOLO detections.  bbox = [x1, y1, x2, y2]; dist_cm = fixed distance.
OBJECTS = [
    {"class_name": "chair",  "bbox": [380, 25, 520, 150], "dist_cm": 120,
     "confidence": 0.81},
    {"class_name": "bottle", "bbox": [150, 150, 212, 355], "dist_cm": 55,
     "confidence": 0.88},
    {"class_name": "cup",    "bbox": [285, 235, 350, 330], "dist_cm": 50,
     "confidence": 0.84},
    {"class_name": "spoon",  "bbox": [385, 315, 505, 352], "dist_cm": 45,
     "confidence": 0.79},
    {"class_name": "apple",  "bbox": [250, 360, 322, 430], "dist_cm": 60,
     "confidence": 0.86},
]

_TABLE_TOP_Y = 150  # horizon: wall above, table below


def _vgradient(img, y0, y1, c_top, c_bot):
    """Fill rows [y0, y1) with a vertical colour gradient (BGR tuples)."""
    n = max(1, y1 - y0)
    for i in range(n):
        t = i / n
        col = [int(c_top[k] + (c_bot[k] - c_top[k]) * t) for k in range(3)]
        img[y0 + i, :] = col


def _soft_shadow(img, cx, cy, rx, ry, strength=0.45):
    """Darken an elliptical contact shadow under an object."""
    overlay = img.copy()
    cv2.ellipse(overlay, (int(cx), int(cy)), (int(rx), int(ry)),
                0, 0, 360, (20, 18, 16), -1)
    cv2.addWeighted(overlay, strength, img, 1 - strength, 0, img)


def _draw_chair(img, box):
    x1, y1, x2, y2 = box
    wood = (70, 95, 130)
    dark = (45, 62, 88)
    # Backrest posts
    cv2.rectangle(img, (x1 + 6, y1), (x1 + 22, y2), wood, -1)
    cv2.rectangle(img, (x2 - 22, y1), (x2 - 6, y2), wood, -1)
    cv2.rectangle(img, (x1 + 6, y1), (x1 + 22, y2), dark, 2)
    cv2.rectangle(img, (x2 - 22, y1), (x2 - 6, y2), dark, 2)
    # Horizontal slats
    for fy in (y1 + 14, y1 + 46, y1 + 78):
        cv2.rectangle(img, (x1 + 6, fy), (x2 - 6, fy + 16), wood, -1)
        cv2.rectangle(img, (x1 + 6, fy), (x2 - 6, fy + 16), dark, 2)


def _draw_bottle(img, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    _soft_shadow(img, cx, y2 - 4, (x2 - x1), 12)
    body_top = y1 + 46
    # Cap
    cv2.rectangle(img, (cx - 9, y1), (cx + 9, y1 + 18), (40, 60, 95), -1)
    # Neck
    cv2.rectangle(img, (cx - 7, y1 + 16), (cx + 7, body_top), (180, 215, 230), -1)
    # Body (translucent blue)
    cv2.rectangle(img, (x1, body_top), (x2, y2), (210, 175, 120), -1)
    cv2.rectangle(img, (x1, body_top), (x2, y2), (150, 120, 80), 2)
    # Water level + highlight
    wl = body_top + int((y2 - body_top) * 0.30)
    cv2.rectangle(img, (x1 + 3, wl), (x2 - 3, y2 - 3), (225, 200, 150), -1)
    cv2.line(img, (x1 + 10, body_top + 6), (x1 + 10, y2 - 8), (245, 235, 215), 3)
    # Label band
    ly = body_top + int((y2 - body_top) * 0.45)
    cv2.rectangle(img, (x1, ly), (x2, ly + 34), (250, 250, 250), -1)
    cv2.putText(img, "WATER", (x1 + 4, ly + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 90, 40), 1, cv2.LINE_AA)


def _draw_cup(img, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    rx = (x2 - x1) // 2
    _soft_shadow(img, cx, y2 - 2, rx + 6, 9)
    # Glass body (slightly tapered)
    pts = np.array([[x1 + 4, y1 + 8], [x2 - 4, y1 + 8],
                    [x2 - 9, y2], [x1 + 9, y2]], np.int32)
    cv2.fillPoly(img, [pts], (225, 220, 205))
    # Water inside
    wl = y1 + int((y2 - y1) * 0.40)
    wpts = np.array([[x1 + 7, wl], [x2 - 7, wl],
                     [x2 - 10, y2 - 3], [x1 + 10, y2 - 3]], np.int32)
    cv2.fillPoly(img, [wpts], (235, 200, 150))
    # Rim ellipse + highlight
    cv2.ellipse(img, (cx, y1 + 8), (rx - 4, 7), 0, 0, 360, (245, 245, 235), 2)
    cv2.line(img, (x1 + 12, y1 + 16), (x1 + 12, y2 - 8), (250, 248, 240), 2)
    cv2.polylines(img, [pts], True, (170, 165, 150), 1)


def _draw_spoon(img, box):
    x1, y1, x2, y2 = box
    cy = (y1 + y2) // 2
    _soft_shadow(img, (x1 + x2) // 2, y2 - 2, (x2 - x1) // 2, 7, 0.35)
    metal = (200, 200, 205)
    edge = (140, 140, 150)
    # Bowl (left end)
    cv2.ellipse(img, (x1 + 22, cy), (20, 14), 0, 0, 360, metal, -1)
    cv2.ellipse(img, (x1 + 22, cy), (20, 14), 0, 0, 360, edge, 1)
    cv2.ellipse(img, (x1 + 18, cy - 3), (10, 6), 0, 0, 360, (235, 235, 240), -1)
    # Handle
    cv2.line(img, (x1 + 40, cy), (x2 - 2, cy - 4), metal, 7)
    cv2.line(img, (x1 + 40, cy), (x2 - 2, cy - 4), edge, 1)


def _draw_apple(img, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
    _soft_shadow(img, cx, y2 - 4, rx, 9)
    cv2.ellipse(img, (cx, cy), (rx - 2, ry - 2), 0, 0, 360, (40, 50, 200), -1)
    cv2.ellipse(img, (cx, cy), (rx - 2, ry - 2), 0, 0, 360, (30, 35, 150), 2)
    # Shine
    cv2.ellipse(img, (cx - rx // 3, cy - ry // 3), (rx // 4, ry // 5),
                0, 0, 360, (180, 190, 255), -1)
    # Stem + leaf
    cv2.line(img, (cx, y1 + 6), (cx, y1 - 6), (40, 60, 90), 3)
    cv2.ellipse(img, (cx + 9, y1 - 4), (9, 5), -30, 0, 360, (60, 150, 70), -1)


_DRAW = {
    "chair": _draw_chair,
    "bottle": _draw_bottle,
    "cup": _draw_cup,
    "spoon": _draw_spoon,
    "apple": _draw_apple,
}


def render_scene():
    """Render the mock scene.

    Returns ``(bgr, depth_mm, objects)`` where ``bgr`` is a uint8 BGR image,
    ``depth_mm`` is a float32 (H, W) depth map in millimetres, and
    ``objects`` is the manifest list (copies, with int bboxes).
    """
    img = np.zeros((H, W, 3), np.uint8)

    # Wall (top) then wood table (bottom), both with gentle gradients.
    _vgradient(img, 0, _TABLE_TOP_Y, (120, 120, 128), (150, 152, 160))
    _vgradient(img, _TABLE_TOP_Y, H, (60, 95, 140), (95, 140, 195))
    # Table front edge highlight
    cv2.line(img, (0, _TABLE_TOP_Y), (W, _TABLE_TOP_Y), (70, 105, 150), 3)

    # Depth map: wall flat-far, table plane near the bottom of the frame.
    depth = np.full((H, W), 1600.0, np.float32)
    ys = np.arange(_TABLE_TOP_Y, H, dtype=np.float32)
    table_mm = 1400.0 - (ys - _TABLE_TOP_Y) * (1400.0 - 400.0) / (H - _TABLE_TOP_Y)
    depth[_TABLE_TOP_Y:H, :] = table_mm[:, None]

    # Draw + stamp depth (chair first so tabletop items sit in front).
    for obj in OBJECTS:
        _DRAW[obj["class_name"]](img, obj["bbox"])
        x1, y1, x2, y2 = obj["bbox"]
        depth[y1:y2, x1:x2] = float(obj["dist_cm"]) * 10.0

    # A little camera noise so it reads as a real frame.
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    objects = [
        {"class_name": o["class_name"], "bbox": [int(v) for v in o["bbox"]],
         "dist_cm": int(o["dist_cm"]), "confidence": float(o["confidence"])}
        for o in OBJECTS
    ]
    return img, depth, objects


def generate_and_cache(force=False):
    """Render once and write the PNG / depth / manifest into ``assets/``.

    Returns ``(bgr, depth_mm, objects)``.  Reuses the cache unless *force*.
    """
    os.makedirs(_ASSETS_DIR, exist_ok=True)
    if (not force and os.path.isfile(SCENE_PNG) and os.path.isfile(DEPTH_NPY)
            and os.path.isfile(OBJECTS_JSON)):
        bgr = cv2.imread(SCENE_PNG, cv2.IMREAD_COLOR)
        depth = np.load(DEPTH_NPY)
        with open(OBJECTS_JSON) as f:
            objects = json.load(f)
        return bgr, depth, objects

    bgr, depth, objects = render_scene()
    cv2.imwrite(SCENE_PNG, bgr)
    np.save(DEPTH_NPY, depth)
    with open(OBJECTS_JSON, "w") as f:
        json.dump(objects, f, indent=2)
    return bgr, depth, objects


if __name__ == "__main__":
    _bgr, _depth, _objs = generate_and_cache(force=True)
    print(f"[mock_scene] wrote {SCENE_PNG} ({_bgr.shape})")
    print(f"[mock_scene] wrote {DEPTH_NPY} "
          f"(min={_depth.min():.0f}mm max={_depth.max():.0f}mm)")
    print(f"[mock_scene] {len(_objs)} objects:")
    for o in _objs:
        print(f"   {o['class_name']:7s} bbox={o['bbox']} "
              f"dist={o['dist_cm']}cm")
