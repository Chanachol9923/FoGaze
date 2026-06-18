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
    MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"

    def __init__(self, device="cuda", depth_size: int = 384,
                 min_interval: float = 1.5):
        self._depth_size = depth_size
        self._min_interval = max(0.0, min_interval)
        self._last_submit_time = 0.0
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
        self.processor.size = {"height": self._depth_size, "width": self._depth_size}
        print(f"[DepthEstimator] Processor input size: {self._depth_size}x{self._depth_size}")
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

        self.cal_scale = None
        self._load_cal()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _load_cal(self):
        if DEPTH_CAL_PATH.exists():
            try:
                data = np.load(DEPTH_CAL_PATH)
                self.cal_scale = float(data["scale"])
                print(f"[DepthEstimator] Loaded calibration scale: {self.cal_scale:.4f}")
            except Exception as e:
                print(f"[DepthEstimator] Failed to load depth calibration: {e}")

    def save_cal(self, scale: float):
        self.cal_scale = scale
        DEPTH_CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(DEPTH_CAL_PATH, scale=scale)
        print(f"[DepthEstimator] Saved calibration scale: {scale:.4f}")

    def reset_cal(self):
        self.cal_scale = None
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
        with self._lock:
            if self._last_depth_time == 0:
                return float('inf')
            return _time.time() - self._last_depth_time

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray | None:
        self._h, self._w = image_bgr.shape[:2]
        now = _time.time()
        if now - self._last_submit_time >= self._min_interval:
            with self._lock:
                self._pending_frame = image_bgr
            self._last_submit_time = now
        with self._lock:
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
        """Convert raw DA2V metric depth (meters) to cm.

        Metric model output is already in meters.
        If calibrated, applies a fine-tuning scale factor.
        """
        cm = depth_value * 100.0
        if self.cal_scale is not None:
            cm *= self.cal_scale
        return cm

    def colormap(self, depth_map: np.ndarray | None = None) -> np.ndarray:
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
