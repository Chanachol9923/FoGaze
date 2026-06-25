"""Depth from two separate USB cameras (uncalibrated, UI-tunable).

Drop-in replacement for ``DepthEstimator``: opens a left + right camera,
runs StereoSGBM block matching, and converts disparity to a depth map with
``depth = focal_px * baseline / disparity``.  There is no calibration file —
``focal_px`` and ``baseline`` are exposed in ``STEREO_OPTS`` so they can be
tuned live from the FoGaze side panel until the reported distances look right
(this directly scales graspability).  Same public API as ``DepthEstimator``
(``get_frame``, ``depth_at_bbox``, ``colormap``, ``estimate``/``_sync``,
``depth_to_distance_cm``, ``save_params``, ``reset_cal``, ``close``,
``depth_freshness``).
"""

from __future__ import annotations

import threading
import time as _time

import cv2
import numpy as np

# Shared clip range + colormap list live with the PrimeSense estimator so the
# existing "Depth settings" sliders (min/max mm, colormap) keep working.
try:
    from modules.depth_estimator import OPTS, COLORMAPS
except ModuleNotFoundError:
    # Allow running this file directly (``python3 stereo_depth_estimator.py``)
    # from inside ``modules/`` by putting the project root on sys.path.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.depth_estimator import OPTS, COLORMAPS


# Live-tunable stereo parameters (UI sliders in main.py's side panel).
STEREO_OPTS = {
    "focal_px": 500,     # pinhole focal length in pixels (per-camera)
    "baseline_cm": 6.0,  # distance between the two camera centres (cm)
    "num_disp": 64,      # disparity search range (rounded to a multiple of 16)
    "block_size": 7,     # SGBM matched block size (odd, >=5)
}


class StereoDepthEstimator:
    def __init__(self, left_index: int, right_index: int, width=640, height=480):
        self._w, self._h = width, height
        self._capL = self._open(left_index)
        self._capR = self._open(right_index)
        if not self._capL.isOpened() or not self._capR.isOpened():
            raise RuntimeError(
                f"Cannot open stereo pair (L={left_index}, R={right_index})")
        print(f"[StereoDepthEstimator] Stereo pair L={left_index} R={right_index} "
              f"@ {width}x{height}")

        # If this pair was stereo-calibrated, load the rectification so the
        # left/right rows line up (clean depth) and pull the true focal +
        # baseline so reported distances are metric, not hand-tuned guesses.
        self._cal = None
        try:
            from modules.stereo_calibrator import StereoCalibrator
            cal = StereoCalibrator(left_index, right_index)
            if cal.ready:
                self._cal = cal
                STEREO_OPTS["focal_px"] = cal.focal_px
                STEREO_OPTS["baseline_cm"] = cal.baseline_mm / 10.0
                print("[StereoDepthEstimator] Using stereo calibration "
                      "(rectified + metric depth).")
            else:
                print("[StereoDepthEstimator] No stereo calibration found — "
                      "raw frames, depth is approximate. Run "
                      "modules/stereo_calibrator.py to calibrate.")
        except Exception as e:
            print(f"[StereoDepthEstimator] Calibration load skipped: {e}")

        self._matcher = None
        self._matcher_key = None

        self._lock = threading.Lock()
        self._last_color = None
        self._last_depth = None
        self._last_t = 0.0

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _open(self, index):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _build_matcher(self):
        num = max(16, (int(STEREO_OPTS["num_disp"]) // 16) * 16)
        block = int(STEREO_OPTS["block_size"]) | 1  # force odd
        block = max(5, block)
        key = (num, block)
        if key != self._matcher_key:
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num,
                blockSize=block,
                P1=8 * 3 * block * block,
                P2=32 * 3 * block * block,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
            )
            self._matcher_key = key
        return self._matcher

    def _worker_loop(self):
        while self._running:
            try:
                okL, fL = self._capL.read()
                okR, fR = self._capR.read()
                if not okL or not okR or fL is None or fR is None:
                    _time.sleep(0.01)
                    continue

                # The two USB cameras may ignore the requested resolution and
                # hand back different sizes; SGBM requires identical L/R sizes.
                if fR.shape[:2] != fL.shape[:2]:
                    fR = cv2.resize(fR, (fL.shape[1], fL.shape[0]))

                # Rectify so matching points share a row (only if calibrated).
                if self._cal is not None and self._cal.ready:
                    fL, fR = self._cal.rectify(fL, fR)

                grayL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
                matcher = self._build_matcher()
                disp = matcher.compute(grayL, grayR).astype(np.float32) / 16.0

                # depth(mm) = focal_px * baseline(mm) / disparity(px)
                focal = float(STEREO_OPTS["focal_px"])
                baseline_mm = float(STEREO_OPTS["baseline_cm"]) * 10.0
                depth = np.zeros_like(disp, dtype=np.float32)
                valid = disp > 0.5
                depth[valid] = focal * baseline_mm / disp[valid]
                # Clamp wild values so colormap/mean stay sane.
                depth[depth > 1e5] = 0.0

                with self._lock:
                    self._last_color = fL
                    self._last_depth = depth
                    self._last_t = _time.time()
            except Exception as e:
                print(f"[StereoDepthEstimator] loop error: {e}")
                _time.sleep(0.02)

    # ── Public API (mirrors DepthEstimator) ────────────────────────────
    def get_frame(self):
        with self._lock:
            return self._last_color, self._last_depth

    def estimate(self, image_bgr=None):
        with self._lock:
            return self._last_depth

    def estimate_sync(self, image_bgr=None):
        with self._lock:
            return self._last_depth

    def save_params(self):
        pass

    def reset_cal(self):
        pass

    @property
    def depth_freshness(self) -> float:
        return _time.time() - self._last_t

    def depth_at_bbox(self, x1: int, y1: int, x2: int, y2: int):
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
        return float(np.median(valid)) / 10.0

    def depth_to_distance_cm(self, depth_value, scene_min=None, scene_max=None):
        return depth_value

    def colormap(self, depth_map=None):
        with self._lock:
            d = self._last_depth if depth_map is None else depth_map
            hw = ((self._last_depth.shape[0], self._last_depth.shape[1])
                  if self._last_depth is not None else (self._h, self._w))
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

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        for cap in (self._capL, self._capR):
            try:
                cap.release()
            except Exception:
                pass
        print("[StereoDepthEstimator] Closed.")


# ── Standalone self-test ────────────────────────────────────────────────
# Run from anywhere:
#   python3 modules/stereo_depth_estimator.py            # pick cameras interactively
#   python3 modules/stereo_depth_estimator.py -l 0 -r 2  # specify indices directly
# Shows the left camera and the live depth colormap side by side; press 'q'
# to quit.  Useful for tuning STEREO_OPTS without launching the full app.
def _probe_cameras(max_index: int = 10):
    """Return a list of camera indices that open and deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        cap.release()
    return found


def _ask_index(prompt: str, available: list[int]) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) in available:
            return int(raw)
        print(f"  Please enter one of: {available}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Live stereo depth preview")
    ap.add_argument("-l", "--left", type=int, default=None, help="left camera index")
    ap.add_argument("-r", "--right", type=int, default=None, help="right camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    left, right = args.left, args.right
    if left is None or right is None:
        print("Scanning for cameras...")
        cams = _probe_cameras()
        if len(cams) < 2:
            raise SystemExit(
                f"Need at least 2 cameras, found {len(cams)}: "
                + ", ".join(f"#{i} ({w}x{h})" for i, w, h in cams))
        print("Available cameras:")
        for i, w, h in cams:
            print(f"  [{i}]  {w}x{h}")
        indices = [i for i, _, _ in cams]
        if left is None:
            left = _ask_index("Left camera index:  ", indices)
        if right is None:
            right = _ask_index("Right camera index: ", indices)

    est = StereoDepthEstimator(left, right, args.width, args.height)
    print("[self-test] Press 'q' in the window to quit.")
    try:
        while True:
            color, depth = est.get_frame()
            if color is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            cmap = est.colormap()
            if cmap.shape[:2] != color.shape[:2]:
                cmap = cv2.resize(cmap, (color.shape[1], color.shape[0]))
            cm = est.depth_at_bbox(
                color.shape[1] // 2 - 20, color.shape[0] // 2 - 20,
                color.shape[1] // 2 + 20, color.shape[0] // 2 + 20)
            label = f"center: {cm:.1f} cm" if cm is not None else "center: --"
            cv2.putText(color, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("stereo: left | depth", np.hstack([color, cmap]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        est.close()
        cv2.destroyAllWindows()
