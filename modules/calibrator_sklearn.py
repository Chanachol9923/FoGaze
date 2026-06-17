"""Sklearn-based calibrator — pulse-and-capture, StandardScaler, Ridge/SVR/MLP.

Replicates EyeTrax's calibration approach:
  • wait_for_face_and_countdown()  — 2s arc, restarts if face lost
  • _pulse_and_capture()           — 1s pulse + 1s capture per point
  • StandardScaler + sklearn model  — Ridge (default), LinearSVR, ElasticNet, MLP
  • Save/load via pickle (.pkl)
  • Fullscreen calibration UI
"""

from __future__ import annotations

import os
import time
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor


# ═══════════════════════════════════════════════════════════════════════
#  Helpers  (shared across calibration functions)
# ═══════════════════════════════════════════════════════════════════════

def compute_grid_points(order, sw, sh, margin_ratio=0.10):
    """Convert (row,col) indices → absolute pixel coordinates."""
    if not order:
        return []
    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)
    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my
    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r
    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def compute_grid(rows, cols, sw, sh, margin_ratio=0.10, order="default"):
    """Generate grid points with serpentine or row-major ordering."""
    if order == "serpentine":
        indices = []
        for r in range(rows):
            col_r = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            indices.extend((r, c) for c in col_r)
    else:
        indices = [(r, c) for r in range(rows) for c in range(cols)]
    return compute_grid_points(indices, sw, sh, margin_ratio)


def _draw_countdown_arc(canvas, center, radius, t, color, thickness=4):
    """Draw a smoothstepped arc (0→360°)."""
    t = np.clip(t, 0, 1)
    eased = t * t * (3 - 2 * t)
    ang = int(360 * (1 - eased))
    cv2.ellipse(canvas, center, (radius, radius), 0, -90, -90 + ang,
                color, thickness, cv2.LINE_AA)


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration types & point generators
# ═══════════════════════════════════════════════════════════════════════

CALIBRATION_REGISTRY = {
    "5p": lambda sw, sh: compute_grid_points([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
                                              sw, sh, margin_ratio=0.08),
    "9p": lambda sw, sh: compute_grid_points(
        [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)],
        sw, sh, margin_ratio=0.10),
}


# ═══════════════════════════════════════════════════════════════════════
#  Model wrappers  (EyTrax-compatible interface)
# ═══════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    def __init__(self):
        self.scaler = StandardScaler()

    @abstractmethod
    def _native_train(self, X, y): ...
    @abstractmethod
    def _native_predict(self, X): ...

    def train(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self._native_train(Xs, y)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self._native_predict(Xs)

    def save(self, path):
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


class RidgeModel(BaseModel):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = Ridge(alpha=alpha)

    def _native_train(self, X, y):
        self.model.fit(X, y)

    def _native_predict(self, X):
        return self.model.predict(X)


class SVRModel(BaseModel):
    def __init__(self, C=5.0, epsilon=5.0):
        super().__init__()
        self.model_x = LinearSVR(C=C, epsilon=epsilon, max_iter=5000)
        self.model_y = LinearSVR(C=C, epsilon=epsilon, max_iter=5000)

    def _native_train(self, X, y):
        self.model_x.fit(X, y[:, 0])
        self.model_y.fit(X, y[:, 1])

    def _native_predict(self, X):
        px = self.model_x.predict(X)
        py = self.model_y.predict(X)
        return np.column_stack([px, py])


class ElasticNetModel(BaseModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

    def _native_train(self, X, y):
        # ElasticNet is single-output; fit x and y separately
        self.model_x = ElasticNet(alpha=self.model.alpha,
                                  l1_ratio=self.model.l1_ratio, max_iter=5000)
        self.model_y = ElasticNet(alpha=self.model.alpha,
                                  l1_ratio=self.model.l1_ratio, max_iter=5000)
        self.model_x.fit(X, y[:, 0])
        self.model_y.fit(X, y[:, 1])

    def _native_predict(self, X):
        px = self.model_x.predict(X)
        py = self.model_y.predict(X)
        return np.column_stack([px, py])


class MLPModel(BaseModel):
    def __init__(self, hidden=(64, 32), max_iter=500):
        super().__init__()
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                                  random_state=0)

    def _native_train(self, X, y):
        self.model.fit(X, y)

    def _native_predict(self, X):
        return self.model.predict(X)


MODEL_REGISTRY = {
    "ridge": lambda **kw: RidgeModel(**kw),
    "svr": lambda **kw: SVRModel(**kw),
    "elastic_net": lambda **kw: ElasticNetModel(**kw),
    "mlp": lambda **kw: MLPModel(**kw),
}


def create_model(name: str, **kwargs) -> BaseModel:
    name = name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  Calibrator  (holds state, runs calibration, provides predict)
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CALIB_PATH = os.path.join(os.path.dirname(__file__),
                                  "..", "calibration_sklearn.pkl")


class SklearnCalibrator:
    """Full calibration pipeline — pulse-and-capture + sklearn regression.

    Usage::

        cal = SklearnCalibrator()
        cal.run(cap, feat_extractor, calib_type="9p")
        # or
        cal.load()

        x, y = cal.predict(486D_feature_vector)
    """

    def __init__(self, model_name="ridge", model_kwargs=None,
                 calib_path=None, ear_history_len=50):
        self.model = create_model(model_name, **(model_kwargs or {}))
        self._calib_path = calib_path or DEFAULT_CALIB_PATH
        self._sw = 1920
        self._sh = 1080
        self._ear_history = []
        self._ear_hist_len = ear_history_len

    # ── Persistence ──────────────────────────────────────────────────

    def is_calibrated(self):
        return hasattr(self.model.model, "coef_") or \
               hasattr(self.model.model, "coefs_") or \
               getattr(getattr(self.model, 'model_x', None), 'coef_', None) is not None

    def save(self, path=None):
        path = path or self._calib_path
        self.model.save(path)
        return True

    def load(self, path=None):
        path = path or self._calib_path
        if not os.path.isfile(path):
            return False
        try:
            self.model = BaseModel.load(path)
            return True
        except Exception:
            return False

    def delete_save(self, path=None):
        path = path or self._calib_path
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False

    # ── Screen size ──────────────────────────────────────────────────

    def set_screen_size(self, sw, sh):
        self._sw, self._sh = sw, sh

    # ── Calibration runner (blocking) ────────────────────────────────

    def run(self, cap, feat_extractor, calib_type="9p", win_name="Calibration"):
        """Run full calibration.  Returns True on success."""
        sw, sh = self._sw, self._sh

        # Fullscreen
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        if not self._wait_for_face(cap, feat_extractor, sw, sh, win_name):
            cv2.destroyWindow(win_name)
            return False

        if calib_type not in CALIBRATION_REGISTRY:
            cv2.destroyWindow(win_name)
            raise ValueError(f"Unknown calibration '{calib_type}'. "
                             f"Choose from {list(CALIBRATION_REGISTRY)}")

        pts = CALIBRATION_REGISTRY[calib_type](sw, sh)
        feats, targs = self._pulse_and_capture(cap, feat_extractor, pts,
                                                sw, sh, win_name)
        cv2.destroyWindow(win_name)

        if feats is None or len(feats) < 3:
            print("[Calibrator] Not enough samples collected.")
            return False

        X = np.array(feats, dtype=np.float64)
        y = np.array(targs, dtype=np.float64)
        self.model.train(X, y)
        self.save()
        print(f"[Calibrator] Trained on {len(X)} samples. Saved to {self._calib_path}")
        return True

    # ── Face countdown ───────────────────────────────────────────────

    def _wait_for_face(self, cap, feat_extractor, sw, sh, win_name,
                       dur=2.0) -> bool:
        """Wait for a non-blinking face, show countdown arc.  Returns False on ESC."""
        fd_start = None
        countdown = False
        self._ear_history.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            feats, blink = feat_extractor.process(frame)
            self._ear_history.append(blink)
            if len(self._ear_history) > self._ear_hist_len:
                self._ear_history.pop(0)

            # Face is present + not consistently blinking
            last_n = list(self._ear_history)[-10:] if len(self._ear_history) >= 10 else self._ear_history
            face = feats is not None and (not any(last_n) or True)

            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            now = time.time()

            if face:
                if not countdown:
                    fd_start = now
                    countdown = True
                elapsed = now - fd_start
                if elapsed >= dur:
                    return True
                _draw_countdown_arc(canvas, (sw // 2, sh // 2), 50,
                                    elapsed / dur, (0, 255, 255), 6)
                cv2.putText(canvas, "Look at the center",
                            (sw // 2 - 120, sh // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            else:
                countdown = False
                fd_start = None
                txt = "No face detected"
                fs, thick = 2, 3
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0]
                tx = (sw - size[0]) // 2
                ty = (sh + size[1]) // 2
                cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, (0, 0, 255), thick, cv2.LINE_AA)

            cv2.imshow(win_name, canvas)
            if cv2.waitKey(1) == 27:
                return False

    # ── Pulse-and-capture ────────────────────────────────────────────

    def _pulse_and_capture(self, cap, feat_extractor, pts, sw, sh, win_name,
                           pulse_d=1.0, cap_d=1.0):
        """Pulse → Capture for each point. Returns (features, targets) or None."""
        feats, targs = [], []

        for i, (x, y) in enumerate(pts):
            label = f"Point {i + 1} / {len(pts)}"

            # ── Pulse phase ──
            ps = time.time()
            while True:
                e = time.time() - ps
                if e > pulse_d:
                    break
                ret, frame = cap.read()
                if not ret:
                    continue

                canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
                radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
                cv2.circle(canvas, (x, y), radius, (0, 255, 255), -1)
                cv2.putText(canvas, label, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
                cv2.putText(canvas, "Look at the circle",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (150, 150, 150), 1)
                self._draw_progress_bar(canvas, i, len(pts), sw, sh)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) == 27:
                    return None

            # ── Capture phase ──
            cs = time.time()
            while True:
                e = time.time() - cs
                if e > cap_d:
                    break
                ret, frame = cap.read()
                if not ret:
                    continue

                canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
                cv2.circle(canvas, (x, y), 20, (0, 255, 255), -1)
                _draw_countdown_arc(canvas, (x, y), 40, e / cap_d,
                                    (255, 255, 255), 4)
                cv2.putText(canvas, label, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
                self._draw_progress_bar(canvas, i, len(pts), sw, sh)
                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) == 27:
                    return None

                ft, blink = feat_extractor.process(frame)
                if ft is not None and not blink:
                    feats.append(ft)
                    targs.append([x, y])

        return feats, targs

    @staticmethod
    def _draw_progress_bar(canvas, step, total, sw, sh):
        bar_w = int(sw * 0.6)
        bar_h = 8
        x1 = (sw - bar_w) // 2
        y1 = sh - 40
        filled = bar_w * (step + 1) // total if total > 0 else 0
        cv2.rectangle(canvas, (x1, y1), (x1 + bar_w, y1 + bar_h),
                      (80, 80, 80), -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + filled, y1 + bar_h),
                      (0, 255, 255), -1)
        pct = f"{int(100 * (step + 1) / total)}%"
        cv2.putText(canvas, pct, (x1 + bar_w + 10, y1 + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, feature_vector):
        """Map 486D feature vector → (screen_x, screen_y)."""
        X = np.array([feature_vector], dtype=np.float64)
        pred = self.model.predict(X)[0]
        return float(np.clip(pred[0], 0, self._sw - 1)), \
               float(np.clip(pred[1], 0, self._sh - 1))
