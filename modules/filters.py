"""Smoothing / filtering strategies for gaze output.

All smoothers expose ``step(x, y) -> (x, y)``.
"""

from __future__ import annotations

import time
from collections import deque
from abc import ABC, abstractmethod

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Base
# ═══════════════════════════════════════════════════════════════════════

class BaseSmoother(ABC):
    @abstractmethod
    def step(self, x: float, y: float) -> tuple[float, float]:
        ...

    def reset(self):
        pass

    def tune(self, *args, **kwargs):
        pass


# ═══════════════════════════════════════════════════════════════════════
#  No-op
# ═══════════════════════════════════════════════════════════════════════

class NoSmoother(BaseSmoother):
    def step(self, x, y):
        return x, y


# ═══════════════════════════════════════════════════════════════════════
#  One Euro Filter  (from gaze_tracker.py)
# ═══════════════════════════════════════════════════════════════════════

class _OneEuro:
    __slots__ = ('min_cutoff', 'beta', 'dcutoff',
                 '_x_prev', '_dx_prev', '_t_prev')

    def __init__(self, min_cutoff=1.0, beta=0.007, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self):
        self._x_prev = self._dx_prev = self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def __call__(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = t
            return float(x)
        dt = max(t - self._t_prev, 1e-6)
        dx = (float(x) - self._x_prev) / dt
        ad = self._alpha(self.dcutoff, dt)
        dx_smooth = ad * dx + (1.0 - ad) * self._dx_prev
        fc = self.min_cutoff + self.beta * abs(dx_smooth)
        ap = self._alpha(fc, dt)
        x_smooth = ap * float(x) + (1.0 - ap) * self._x_prev
        self._x_prev = x_smooth
        self._dx_prev = dx_smooth
        self._t_prev = t
        return x_smooth


class OneEuroSmoother(BaseSmoother):
    def __init__(self, min_cutoff=1.0, beta=0.007, dcutoff=1.0):
        self._fx = _OneEuro(min_cutoff, beta, dcutoff)
        self._fy = _OneEuro(min_cutoff, beta, dcutoff)

    def step(self, x, y):
        return self._fx(x), self._fy(y)

    def reset(self):
        self._fx.reset()
        self._fy.reset()


# ═══════════════════════════════════════════════════════════════════════
#  Kalman (4-state CV)
# ═══════════════════════════════════════════════════════════════════════

def make_kalman(process_noise: float = 50.0, measure_noise: float = 0.2):
    """Create a 4-state constant-velocity Kalman filter.

    State: [x, y, vx, vy]   Measure: [x, y]
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
    return kf


class KalmanSmoother(BaseSmoother):
    def __init__(self, kalman: cv2.KalmanFilter | None = None):
        self._kf = kalman or make_kalman()
        self._initialised = False

    def step(self, x, y):
        if not self._initialised:
            self._kf.statePost = np.array([x, y, 0, 0], dtype=np.float32)
            self._initialised = True
            return x, y
        pred = self._kf.predict()
        meas = np.array([x, y], dtype=np.float32)
        corr = self._kf.correct(meas)
        return float(corr[0]), float(corr[1])

    def reset(self):
        self._initialised = False
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

    def tune(self, gaze_estimator, cap, sw, sh):
        """Auto-tune measurement noise from calibration-like targets."""
        import cv2
        from modules.calibrator_sklearn import _draw_countdown_arc

        targets = [
            (sw // 4, sh // 4),
            (sw // 2, sh // 2),
            (sw * 3 // 4, sh * 3 // 4),
        ]
        variances = []
        for tx, ty in targets:
            # Show target for 2 seconds, collect predictions
            start = time.time()
            pts = []
            win_name = "Tuning Kalman — look at the cross"
            cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            while time.time() - start < 2.0:
                ret, frame = cap.read()
                if not ret:
                    continue
                canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
                _draw_countdown_arc(canvas, (tx, ty), 40,
                                    (time.time() - start) / 2.0, (0, 255, 255))
                cv2.imshow(win_name, canvas)
                cv2.waitKey(1)

                feats, blink = gaze_estimator.extract_features(frame)
                if feats is not None and not blink:
                    pred = gaze_estimator.predict(np.array([feats]))[0]
                    pts.append(pred)

            cv2.destroyWindow(win_name)
            if len(pts) >= 5:
                var = np.var(pts, axis=0).mean()
                variances.append(var)

        if variances:
            m_noise = np.mean(variances)
            m_noise = max(m_noise, 0.01)
            self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * m_noise
            print(f"[Kalman] Tuned measurementNoiseCov = {m_noise:.2f}")


class KalmanEMASmoother(BaseSmoother):
    """Kalman + configurable EMA cascade."""

    def __init__(self, kalman: cv2.KalmanFilter | None = None, ema_alpha=0.25):
        self._inner = KalmanSmoother(kalman)
        self._alpha = np.clip(ema_alpha, 0, 1)
        self._ex = None
        self._ey = None

    def step(self, x, y):
        kx, ky = self._inner.step(x, y)
        if self._ex is None:
            self._ex, self._ey = kx, ky
        else:
            self._ex = self._alpha * kx + (1 - self._alpha) * self._ex
            self._ey = self._alpha * ky + (1 - self._alpha) * self._ey
        return self._ex, self._ey

    def reset(self):
        self._inner.reset()
        self._ex = self._ey = None

    def tune(self, gaze_estimator, cap, sw, sh):
        self._inner.tune(gaze_estimator, cap, sw, sh)


# ═══════════════════════════════════════════════════════════════════════
#  KDE
# ═══════════════════════════════════════════════════════════════════════

class KDESmoother(BaseSmoother):
    """Gaussian KDE over recent gaze history, returns confidence-contour mode."""

    def __init__(self, sw: int, sh: int, window_s: float = 0.5,
                 confidence: float = 0.5, grid_w: int = 320, grid_h: int = 200):
        from scipy.stats import gaussian_kde
        self._kde = gaussian_kde
        self._window = window_s
        self._confidence = confidence
        self._grid_w = grid_w
        self._grid_h = grid_h
        self._sw = sw
        self._sh = sh
        self._buf = deque(maxlen=1024)  # (ts, x, y)
        self.debug = {}

    def step(self, x, y):
        now = time.time()
        self._buf.append((now, x, y))

        # Expire old
        while self._buf and now - self._buf[0][0] > self._window:
            self._buf.popleft()

        if len(self._buf) < 3:
            return x, y

        pts = np.array([(p[1], p[2]) for p in self._buf])
        try:
            kde = self._kde(pts.T)
            xs = np.linspace(0, self._sw, self._grid_w)
            ys = np.linspace(0, self._sh, self._grid_h)
            X, Y = np.meshgrid(xs, ys)
            grid = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(grid).reshape(self._grid_h, self._grid_w)
            Z /= Z.max() + 1e-12

            # Find confidence contour
            from skimage import measure
            try:
                contours = measure.find_contours(Z.T, self._confidence)
            except Exception:
                contours = []

            best = None
            best_area = 0
            self.debug["contours"] = []
            for c in contours:
                if len(c) < 3:
                    continue
                px = c[:, 0] / self._grid_w * self._sw
                py = c[:, 1] / self._grid_h * self._sh
                area = cv2.contourArea(np.column_stack([px, py]).astype(np.float32))
                if area > best_area:
                    best_area = area
                    best = np.column_stack([px, py])
                    self.debug["contours"] = [best.astype(np.int32)]

            if best is not None and best_area > 10:
                cx = float(np.mean(best[:, 0]))
                cy = float(np.mean(best[:, 1]))
                return cx, cy
        except Exception:
            pass

        return x, y

    def reset(self):
        self._buf.clear()
        self.debug = {}


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════

SMOOTHER_KWARGS: dict[str, tuple] = {
    "none": (),
    "oef": ("min_cutoff", "beta", "dcutoff"),
    "kalman": ("process_noise", "measure_noise"),
    "kalman_ema": ("process_noise", "measure_noise", "ema_alpha"),
    "kde": ("window_s", "confidence", "grid_w", "grid_h"),
}


def make_smoother(name: str, sw: int = 1920, sh: int = 1080,
                  **kwargs) -> BaseSmoother:
    """Factory: return a smoother by short name."""
    name = name.lower().replace("-", "_")
    if name == "none":
        return NoSmoother()
    elif name == "oef":
        kw = {k: kwargs[k] for k in SMOOTHER_KWARGS["oef"] if k in kwargs}
        return OneEuroSmoother(**kw)
    elif name == "kalman":
        kw = {k: kwargs[k] for k in SMOOTHER_KWARGS["kalman"] if k in kwargs}
        kf = make_kalman(**kw)
        return KalmanSmoother(kf)
    elif name == "kalman_ema":
        kw = {k: kwargs[k] for k in SMOOTHER_KWARGS["kalman_ema"] if k in kwargs}
        kf_kw = {k: kw[k] for k in ("process_noise", "measure_noise") if k in kw}
        kf = make_kalman(**kf_kw)
        return KalmanEMASmoother(kf, ema_alpha=kw.get("ema_alpha", 0.25))
    elif name == "kde":
        kw = {k: kwargs[k] for k in SMOOTHER_KWARGS["kde"] if k in kwargs}
        return KDESmoother(sw, sh, **kw)
    else:
        raise ValueError(f"Unknown smoother '{name}'.")
