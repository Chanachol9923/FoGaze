"""Drop-in replacement for :class:`DepthEstimator` with no hardware.

Feeds a fixed, procedurally generated tabletop frame (see
:mod:`modules.mock_scene`) as the scene colour stream and a matching fixed
depth map, so FoGaze runs end-to-end without a PrimeSense attached.  The
public API mirrors ``DepthEstimator`` exactly (``get_frame``,
``depth_at_bbox``, ``colormap``, ``estimate``/``estimate_sync``,
``depth_to_distance_cm``, ``save_params``, ``reset_cal``, ``close``,
``depth_freshness``) so ``main.py`` can swap it in without other changes.

Extra: ``mock_detections()`` returns the scene's known objects as YOLO-style
detection dicts, so the blink->grasp demo is deterministic even though the
synthetic image is not a real photo.
"""

from __future__ import annotations

import numpy as np

from modules import mock_scene
from modules.depth_estimator import OPTS, COLORMAPS

import cv2


class MockDepthEstimator:
    """Simulated depth+colour source backed by a cached synthetic scene."""

    def __init__(self):
        bgr, depth_mm, objects = mock_scene.generate_and_cache()
        self._color = np.ascontiguousarray(bgr)
        self._depth = np.ascontiguousarray(depth_mm.astype(np.float32))
        self._objects = objects
        print(f"[MockDepthEstimator] Simulated scene "
              f"{self._color.shape[1]}x{self._color.shape[0]} "
              f"with {len(objects)} objects (no hardware).")

    # ── Frame access (mirrors DepthEstimator) ──────────────────────────
    def get_frame(self):
        return self._color, self._depth

    def estimate(self, image_bgr=None):
        return self._depth

    def estimate_sync(self, image_bgr=None):
        return self._depth

    def save_params(self):
        pass

    def reset_cal(self):
        pass

    @property
    def depth_freshness(self) -> float:
        return 0.0

    def depth_at_bbox(self, x1: int, y1: int, x2: int, y2: int):
        roi = self._depth[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        valid = roi[(roi >= OPTS["min_dist_mm"]) & (roi <= OPTS["max_dist_mm"])]
        if valid.size == 0:
            return None
        return float(valid.mean()) / 10.0

    def depth_to_distance_cm(self, depth_value, scene_min=None, scene_max=None):
        return depth_value

    def colormap(self, depth_map=None):
        d = self._depth if depth_map is None else depth_map
        hw = (self._depth.shape[0], self._depth.shape[1])
        if d is None:
            return np.zeros((*hw, 3), dtype=np.uint8)
        clipped = np.where(
            (d >= OPTS["min_dist_mm"]) & (d <= OPTS["max_dist_mm"]), d, 0
        ).astype(np.uint16)
        if clipped.max() == 0:
            return np.zeros((*hw, 3), dtype=np.uint8)
        norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cmap = COLORMAPS[min(OPTS["cmap_idx"], len(COLORMAPS) - 1)]
        return cv2.applyColorMap(norm, cmap)

    # ── Simulated detections (sim-mode replacement for YOLO) ───────────
    def mock_detections(self, frame=None):
        """Return the scene's known objects as YOLO-style detection dicts."""
        src = self._color if frame is None else frame
        dets = []
        for o in self._objects:
            x1, y1, x2, y2 = o["bbox"]
            crop = src[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else np.array([])
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": o["class_name"],
                "confidence": float(o["confidence"]),
                "crop": crop,
            })
        return dets

    def close(self):
        print("[MockDepthEstimator] Closed.")
