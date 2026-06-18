from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


CALIB_DIR = Path.home() / ".cache" / "fogaze3" / "calib"


class CameraCalibrator:
    CHESSBOARD = (9, 6)
    SQUARE_MM = 25.0

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CALIB_DIR / f"cam{camera_id}.json"

        self.mtx: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.map1: np.ndarray | None = None
        self.map2: np.ndarray | None = None
        self._map_w = self._map_h = 0

        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self.mtx = np.array(d["mtx"], dtype=np.float64)
            self.dist = np.array(d["dist"], dtype=np.float64)
            print(f"[CameraCalibrator] Loaded cal for camera {self.camera_id}")
        except Exception as e:
            print(f"[CameraCalibrator] Load failed: {e}")

    def save(self):
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "mtx": self.mtx.tolist(),
            "dist": self.dist.tolist(),
        }, indent=2))
        print(f"[CameraCalibrator] Saved cal for camera {self.camera_id}")

    def reset(self):
        self.mtx = self.dist = self.map1 = self.map2 = None
        if self._path.exists():
            self._path.unlink()

    @property
    def ready(self) -> bool:
        return self.mtx is not None and self.dist is not None

    def _ensure_maps(self, w: int, h: int):
        if self.mtx is None or self.dist is None:
            self._init_default(w, h)
        if w == self._map_w and h == self._map_h and self.map1 is not None:
            return
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.mtx, (w, h), cv2.CV_32FC1
        )
        self._map_w, self._map_h = w, h

    def _init_default(self, w: int, h: int):
        """Internal: set reasonable default for a typical ~75° FOV webcam."""
        f = max(w, h) * 1.0
        self.mtx = np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist = np.array([-0.30, 0.10, 0, 0], dtype=np.float64)
        self._map_w, self._map_h = w, h
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.mtx, (w, h), cv2.CV_32FC1
        )
        print(f"[CameraCalibrator] Default undistort for {w}x{h} — k1={self.dist[0]:.2f} k2={self.dist[1]:.2f}")

    def enable_default(self, w: int, h: int):
        """Public: re-initialize with default parameters (overrides chessboard cal)."""
        self._init_default(w, h)
        self.save()

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._ensure_maps(w, h)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def find_chessboard(self, gray: np.ndarray) -> np.ndarray | None:
        ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD, None)
        if not ret:
            return None
        return cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

    def calibrate(self, frames: list[np.ndarray], w: int, h: int) -> bool:
        objp = np.zeros((self.CHESSBOARD[0] * self.CHESSBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.CHESSBOARD[0], 0:self.CHESSBOARD[1]].T.reshape(-1, 2)
        objp *= self.SQUARE_MM

        objpoints, imgpoints = [], []
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = self.find_chessboard(gray)
            if corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)
                print(f"[CameraCalibrator] Sample {i+1}/{len(frames)}: OK")
            else:
                print(f"[CameraCalibrator] Sample {i+1}/{len(frames)}: FAIL")

        if len(objpoints) < 5:
            print(f"[CameraCalibrator] Need >=5 good samples, got {len(objpoints)}")
            return False

        ret, self.mtx, self.dist, *_ = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
        self._ensure_maps(w, h)
        self.save()
        print(f"[CameraCalibrator] RMS error: {ret:.4f}")
        return True
