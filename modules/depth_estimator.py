from __future__ import annotations

import threading
import time as _time

import cv2
import numpy as np
from openni import openni2


COLORMAPS = [
    cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET,
    cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
    cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT,
    cv2.COLORMAP_PARULA, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO,
]


OPTS = {
    "min_dist_mm": 200,
    "max_dist_mm": 3500,
    "off_x": 0,
    "off_y": 0,
    "cmap_idx": 2,
}


class DepthEstimator:
    def __init__(self):
        openni2.initialize()
        uris = openni2.Device.enumerate_uris()
        if not uris:
            raise RuntimeError("No PrimeSense device found")
        self._dev = openni2.Device.open_file(uris[0])
        info = self._dev.get_device_info()
        print(f"[DepthEstimator] Device: {info.name} / {info.vendor}")

        # Depth stream 640x480 @30fps 1mm precision
        self._depth_stream = self._dev.create_depth_stream()
        depth_modes = self._depth_stream.get_sensor_info().videoModes
        self._depth_mode = None
        for m in depth_modes:
            if (m.resolutionX == 640 and m.resolutionY == 480
                    and str(m.pixelFormat) == "OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM"):
                self._depth_mode = m
                break
        if self._depth_mode is None:
            self._depth_mode = depth_modes[4]
        self._depth_stream.set_video_mode(self._depth_mode)
        self._depth_stream.start()
        print(f"[DepthEstimator] Depth: {self._depth_mode.resolutionX}x{self._depth_mode.resolutionY} "
              f"@{self._depth_mode.fps}fps")

        # Color stream 640x480 @30fps RGB
        self._color_stream = self._dev.create_color_stream()
        color_modes = self._color_stream.get_sensor_info().videoModes
        self._color_mode = None
        for m in color_modes:
            if (m.resolutionX == 640 and m.resolutionY == 480
                    and str(m.pixelFormat) == "OniPixelFormat.ONI_PIXEL_FORMAT_RGB888"):
                self._color_mode = m
                break
        if self._color_mode is None:
            self._color_mode = color_modes[4]
        self._color_stream.set_video_mode(self._color_mode)
        self._color_stream.start()
        print(f"[DepthEstimator] Color: {self._color_mode.resolutionX}x{self._color_mode.resolutionY} "
              f"@{self._color_mode.fps}fps")

        self._lock = threading.Lock()
        self._last_depth = None
        self._last_color = None
        self._latest_depth = None
        self._latest_color = None

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        while True:
            try:
                df = self._depth_stream.read_frame()
                cf = self._color_stream.read_frame()

                depth_buf = df.get_buffer_as_uint16()
                depth = np.frombuffer(depth_buf, dtype=np.uint16).reshape(
                    df.height, df.width).astype(np.float32)

                color_buf = cf.get_buffer_as_uint8()
                color = np.frombuffer(color_buf, dtype=np.uint8).reshape(
                    cf.height, cf.width, 3)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                color = cv2.flip(color, 1)
                depth = cv2.flip(depth, 1)

                off_x = OPTS["off_x"]
                off_y = OPTS["off_y"]
                if off_x != 0 or off_y != 0:
                    M = np.float32([[1, 0, off_x], [0, 1, off_y]])
                    depth = cv2.warpAffine(
                        depth, M, (df.width, df.height),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                with self._lock:
                    self._last_depth = depth
                    self._last_color = color
            except Exception as e:
                print(f"[DepthEstimator] Stream read error: {e}")
                _time.sleep(0.01)

    def get_frame(self):
        with self._lock:
            return self._last_color, self._last_depth

    def save_params(self):
        pass

    def reset_cal(self):
        pass

    @property
    def depth_freshness(self) -> float:
        return 0.0

    def estimate(self, image_bgr: np.ndarray | None = None) -> np.ndarray | None:
        with self._lock:
            return self._last_depth

    def estimate_sync(self, image_bgr: np.ndarray | None = None) -> np.ndarray | None:
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
        valid = roi[(roi >= OPTS["min_dist_mm"]) & (roi <= OPTS["max_dist_mm"])]
        if valid.size == 0:
            return None
        return float(valid.mean()) / 10.0

    def depth_to_distance_cm(
        self, depth_value: float, scene_min: float | None = None,
        scene_max: float | None = None,
    ) -> float:
        return depth_value

    def colormap(self, depth_map: np.ndarray | None = None) -> np.ndarray:
        with self._lock:
            d = self._last_depth if depth_map is None else depth_map
            hw = (self._last_depth.shape[0], self._last_depth.shape[1]) if self._last_depth is not None else (480, 640)
        if d is None:
            return np.zeros((*hw, 3), dtype=np.uint8)
        clipped = np.where(
            (d >= OPTS["min_dist_mm"]) & (d <= OPTS["max_dist_mm"]),
            d, 0
        ).astype(np.uint16)
        if clipped.max() == 0:
            return np.zeros((*hw, 3), dtype=np.uint8)
        norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cmap = COLORMAPS[min(OPTS["cmap_idx"], len(COLORMAPS) - 1)]
        return cv2.applyColorMap(norm, cmap)

    def close(self):
        try:
            self._depth_stream.stop()
        except Exception:
            pass
        try:
            self._color_stream.stop()
        except Exception:
            pass
        try:
            self._dev.close()
        except Exception:
            pass
        openni2.unload()
        print("[DepthEstimator] Closed.")
