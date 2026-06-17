"""Depth estimation using Depth-Anything-V2 (transformers/HuggingFace).

Runs inference in a background thread so the main loop is never blocked.
"""

from __future__ import annotations

import threading
import time as _time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


DEPTH_CAL_PATH = Path.home() / ".cache" / "fogaze3" / "depth_cal.npz"


class DepthEstimator:
    MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device="cuda"):
        # Check CUDA compatibility (GPU compute capability)
        cuda_ok = False
        if device == "cuda" and torch.cuda.is_available():
            try:
                cc = torch.cuda.get_device_capability()
                cuda_ok = cc >= (7, 0)
                if not cuda_ok:
                    warnings.warn(
                        f"GPU CC {cc[0]}.{cc[1]} too low, falling back to CPU"
                    )
            except Exception:
                cuda_ok = False

        self.device = torch.device(
            "cuda" if (device == "cuda" and cuda_ok) else "cpu"
        )
        print(f"[DepthEstimator] Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.MODEL_NAME)
            .to(self.device)
            .eval()
        )

        self._lock = threading.Lock()
        self._pending_frame = None
        self._last_depth = None
        self._last_depth_time = 0.0
        self._h, self._w = None, None

        # Calibration model: cm = slope * norm + intercept
        self.cal_model = None
        self._load_cal()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _load_cal(self):
        if DEPTH_CAL_PATH.exists():
            try:
                data = np.load(DEPTH_CAL_PATH)
                slope = float(data["slope"])
                intercept = float(data["intercept"])
                self.cal_model = {"slope": slope, "intercept": intercept}
                print(
                    f"[DepthEstimator] Loaded calibration: cm = {slope:.3f} * norm + {intercept:.1f}"
                )
            except Exception as e:
                print(f"[DepthEstimator] Failed to load depth calibration: {e}")

    def save_cal(self, slope: float, intercept: float):
        self.cal_model = {"slope": slope, "intercept": intercept}
        DEPTH_CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(DEPTH_CAL_PATH, slope=slope, intercept=intercept)
        print(f"[DepthEstimator] Saved calibration: cm = {slope:.3f} * norm + {intercept:.1f}")

    def reset_cal(self):
        self.cal_model = None
        if DEPTH_CAL_PATH.exists():
            DEPTH_CAL_PATH.unlink()
        print("[DepthEstimator] Calibration reset.")

    def _worker_loop(self):
        while True:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
            if frame is not None:
                depth = self._run_inference(frame)
                with self._lock:
                    self._last_depth = depth
                    self._last_depth_time = _time.time()
                    self._h, self._w = depth.shape[:2]
            else:
                _time.sleep(0.005)

    @torch.no_grad()
    def _run_inference(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return depth.cpu().numpy().astype(np.float32)

    def estimate_sync(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run depth estimation synchronously (blocks until done). Used during calibration."""
        depth = self._run_inference(image_bgr)
        with self._lock:
            self._last_depth = depth
            self._last_depth_time = _time.time()
            self._h, self._w = depth.shape[:2]
        return depth

    @property
    def depth_freshness(self) -> float:
        """Seconds since the last depth map was computed. Inf if never."""
        with self._lock:
            if self._last_depth_time == 0:
                return float('inf')
            return _time.time() - self._last_depth_time

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Submit frame for async inference. Returns the latest completed depth map
        (may be from a previous frame — non-blocking)."""
        self._h, self._w = image_bgr.shape[:2]
        with self._lock:
            self._pending_frame = image_bgr
            return self._last_depth

    def depth_at_bbox(self, x1: int, y1: int, x2: int, y2: int) -> float | None:
        with self._lock:
            d = self._last_depth
        if d is None:
            return None
        roi = d[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return float(roi.mean())

    def depth_to_distance_cm(
        self, depth_value: float, scene_min: float | None = None,
        scene_max: float | None = None,
    ) -> float:
        """Convert raw DA2V depth to approximate cm.

        DA2V: higher value = farther away.
        If calibrated: cm = slope * norm + intercept.
        Otherwise falls back to heuristic (20-200cm linear)."""
        if scene_min is None or scene_max is None:
            with self._lock:
                d = self._last_depth
            if d is None:
                return 0
            scene_min = float(d.min())
            scene_max = float(d.max())
        if scene_max == scene_min:
            return 100
        norm = (depth_value - scene_min) / (scene_max - scene_min)
        norm = np.clip(norm, 0, 1)
        if self.cal_model is not None:
            return self.cal_model["slope"] * norm + self.cal_model["intercept"]
        return 200 * norm + 20

    def colormap(self, depth_map: np.ndarray | None = None) -> np.ndarray:
        """Return a color-mapped depth image (overlay)."""
        if depth_map is not None:
            d = depth_map
        else:
            with self._lock:
                d = self._last_depth
        with self._lock:
            h, w = self._h, self._w
        if d is None:
            return np.zeros((h or 480, w or 640, 3), dtype=np.uint8)
        norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        norm = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
