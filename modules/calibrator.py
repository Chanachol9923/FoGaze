import os
import cv2
import numpy as np


def draw_text_stroke(frame, text, pos, scale=0.6, color=(0, 255, 0),
                     thickness=2, outline_thickness=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    ot = outline_thickness or thickness + 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), ot,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness,
                lineType=cv2.LINE_AA)


# ── 4×4 grid = 16 points ────────────────────────────────────────────────

GRID_POINTS = []
for row in range(4):
    for col in range(4):
        GRID_POINTS.append((0.08 + col * 0.28, 0.08 + row * 0.28))

# ── Head-pose directions (all at screen centre) ──────────────────────────

HP_DIRECTIONS = [
    ("Center",  0,  0),
    ("Left",   22,  0),
    ("Right", -22,  0),
    ("Up",      0, 15),
    ("Down",    0,-15),
]

HP_LABELS  = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
HP_ARROWS  = ["●", "⬅", "➡", "⬆", "⬇"]

# ── Timing ───────────────────────────────────────────────────────────────

FRAMES_WARMUP    = 50
FRAMES_COLLECT   = 30
FRAMES_DONE      = 20

DEFAULT_CALIB_PATH = os.path.join(os.path.dirname(__file__),
                                  "..", "calibration.npz")


class Calibrator:
    """Multi-dimensional gaze calibration.

    Phase 1 — 4×4 grid (16 points, gaze-only)
    Phase 2 — Head-pose directions (5 poses at centre)
    Model  — 7D Gaussian-weighted regression (Nadaraya–Watson)
    """

    def __init__(self, calib_path=None):
        self._calib_path = calib_path or DEFAULT_CALIB_PATH
        self.reset()

    # ── State ────────────────────────────────────────────────────────────

    def reset(self):
        self.active      = False
        self.phase       = None      # "grid" | "hp" | None
        self.step        = 0         # index within phase
        self.phase_      = None      # "warmup" | "collect" | "done"
        self.phase_ct_   = 0
        self.collected_  = []        # raw feature vectors for current sample

        # All calibration data
        self.features_   = []        # list of 7D feature vectors
        self.targets_    = []        # list of (screen_x, screen_y)
        self.labels_     = []        # "grid" or "hp"

        self.transform   = None      # will be set after _fit()
        self.ready       = False
        self.fw = self.fh = 0

        # Gaussian regression internals
        self._train_X    = None      # (n, 7)
        self._train_Y    = None      # (n, 2)
        self._sigma      = 0.15
        self._norm_min   = None      # (7,)  min per feature dim
        self._norm_max   = None      # (7,)  max per feature dim

    def start(self, frame_w, frame_h):
        self.reset()
        self.active = True
        self.fw = frame_w
        self.fh = frame_h
        self.phase = "grid"
        self.step = 0
        self.phase_ = "warmup"
        self.phase_ct_ = 0

    def stop(self):
        self.active = False

    def is_active(self):
        return self.active

    def is_calibrated(self):
        return self.ready

    # ── Target helper ────────────────────────────────────────────────────

    def _current_target(self):
        if self.phase == "grid":
            nx, ny = GRID_POINTS[self.step]
        else:
            nx, ny = 0.5, 0.5   # centre
        return (int(nx * self.fw), int(ny * self.fh))

    # ── Update state machine ─────────────────────────────────────────────

    def update(self, feature_vector):
        if not self.active:
            return None

        self.phase_ct_ += 1
        tv = self.phase_

        if tv == "warmup":
            if self.phase_ct_ >= FRAMES_WARMUP:
                self.phase_ = "collect"
                self.phase_ct_ = 0
                self.collected_ = []
            return "warmup"

        elif tv == "collect":
            self.collected_.append(np.array(feature_vector, dtype=np.float64))
            if self.phase_ct_ >= FRAMES_COLLECT:
                return self._commit()
            return "collect"

        elif tv == "done":
            if self.phase_ct_ >= FRAMES_DONE:
                self._advance()
                return "next"
            return "done"

        return None

    def _commit(self):
        n = len(self.collected_)
        if n < 3:
            return "error"

        avg = np.mean(self.collected_, axis=0)
        tx, ty = self._current_target()
        self.features_.append(avg)
        self.targets_.append((tx, ty))
        self.labels_.append(self.phase)

        self.phase_ = "done"
        self.phase_ct_ = 0
        return "done"

    def _advance(self):
        self.step += 1
        if self.phase == "grid":
            if self.step >= len(GRID_POINTS):
                # Move to head-pose phase
                self.phase = "hp"
                self.step = 0
            # else stay in grid, next point
        elif self.phase == "hp":
            if self.step >= len(HP_DIRECTIONS):
                self._fit()
                self.active = False
                return

        self.phase_ = "warmup"
        self.phase_ct_ = 0
        self.collected_ = []

    # ── Fit model ────────────────────────────────────────────────────────

    def _fit(self):
        n = len(self.features_)
        if n < 4:
            return

        X = np.array(self.features_, dtype=np.float64)  # (n, 7)
        Y = np.array(self.targets_, dtype=np.float64)   # (n, 2)

        # Normalisation: per-dimension min/max
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        span = mx - mn
        span[span < 1e-9] = 1.0
        X_norm = (X - mn) / span

        # Adaptive sigma: mean of nearest-neighbour distances
        dists = []
        for i in range(n):
            d = np.min(np.sum((X_norm - X_norm[i]) ** 2, axis=1)
                       [np.arange(n) != i])
            dists.append(np.sqrt(d))
        sigma = float(np.mean(dists)) * 1.5 if dists else 0.15
        sigma = max(sigma, 0.02)

        self._train_X  = X_norm          # (n, 7)
        self._train_Y  = Y               # (n, 2)
        self._norm_min = mn              # (7,)
        self._norm_max = mx              # (7,)
        self._sigma    = sigma
        self.transform = np.array([1.0])  # truthy sentinel
        self.ready     = True

        self.save(self._calib_path)

    # ── Apply (7D Gaussian regression) ──────────────────────────────────

    def apply(self, feature_vector):
        """Map 7D feature vector → (screen_x, screen_y) via Gaussian regression."""
        if self._train_X is None or len(self._train_X) < 4:
            fw = max(self.fw, 640)
            fh = max(self.fh, 480)
            x = np.array(feature_vector, dtype=np.float64).ravel()
            return float(x[0]) * fw, float(x[1]) * fh

        x = np.array(feature_vector, dtype=np.float64).ravel()
        span = self._norm_max - self._norm_min
        span[span < 1e-9] = 1.0
        x_norm = (x - self._norm_min) / span  # (7,)

        diff = self._train_X - x_norm        # (n, 7)
        dist2 = np.sum(diff ** 2, axis=1)    # (n,)
        w = np.exp(-dist2 / (2.0 * self._sigma ** 2))
        total_w = np.sum(w)
        if total_w < 1e-12:
            return float(x[0] * self.fw), float(x[1] * self.fh)

        pred = np.sum(w[:, None] * self._train_Y, axis=0) / total_w
        return float(pred[0]), float(pred[1])

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path=None):
        if not self.ready:
            return False
        path = path or self._calib_path
        np.savez_compressed(
            path,
            train_X=self._train_X,
            train_Y=self._train_Y,
            norm_min=self._norm_min,
            norm_max=self._norm_max,
            sigma=np.array([self._sigma]),
            frame_w=np.array([self.fw]),
            frame_h=np.array([self.fh]),
        )
        return True

    def load(self, path=None):
        path = path or self._calib_path
        try:
            data = np.load(path)
            self._train_X = data["train_X"]
            self._train_Y = data["train_Y"]
            self._norm_min = data["norm_min"]
            self._norm_max = data["norm_max"]
            self._sigma    = float(data["sigma"][0])
            self.fw = int(data["frame_w"][0])
            self.fh = int(data["frame_h"][0])
            self.ready = True
            self.transform = np.array([1.0])
            return True
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return False

    def delete_save(self, path=None):
        path = path or self._calib_path
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False

    # ── Drawing ──────────────────────────────────────────────────────────

    def draw(self, frame):
        if not self.active:
            return
        h, w = frame.shape[:2]

        tx, ty = self._current_target()

        # Target crosshair
        radius = 12 if self.phase_ == "collect" else 8
        cv2.circle(frame, (tx, ty), radius, (0, 255, 0), 2)
        cv2.line(frame, (tx - 30, ty), (tx + 30, ty), (0, 255, 0), 2)
        cv2.line(frame, (tx, ty - 30), (tx, ty + 30), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (0, 255, 0), -1)

        # Accumulated raw gaze dots
        for pt in self.targets_:
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

        # ── HUD ──────────────────────────────────────────────────────────

        if self.phase == "grid":
            total = len(GRID_POINTS)
            phase_msgs = {
                "warmup":  "1/3 : Look at the GREEN crosshair — hold still",
                "collect": "2/3 : Collecting gaze data... keep looking",
                "done":    "3/3 : Saved!",
            }
            title = "GRID CALIBRATION"
        else:
            total = len(HP_DIRECTIONS)
            dir_label  = HP_LABELS[self.step]
            dir_arrow  = HP_ARROWS[self.step]
            phase_msgs = {
                "warmup":  f"1/3 : Turn head {dir_label} {dir_arrow} — "
                           f"keep eyes on centre",
                "collect": "2/3 : Collecting... hold that pose",
                "done":    "3/3 : Saved!",
            }
            title = "HEAD-POSE CALIBRATION"

        progress = f"Point {self.step + 1} / {total}"
        phase_str = phase_msgs.get(self.phase_, "")

        draw_text_stroke(frame, title, (10, 30),
                         scale=0.75, color=(0, 255, 255), thickness=2)
        draw_text_stroke(frame, progress, (10, 60),
                         scale=0.6, color=(0, 255, 0), thickness=2)
        draw_text_stroke(frame, phase_str, (10, 90),
                         scale=0.55, color=(255, 255, 255), thickness=1)

        if self.phase == "hp":
            arrow = HP_ARROWS[self.step]
            draw_text_stroke(frame, arrow, (w // 2 - 20, h // 2 - 80),
                             scale=2.0, color=(0, 255, 255), thickness=3)

        guide = "System does the rest automatically — just follow the target"
        draw_text_stroke(frame, guide,
                         (10, h - 20),
                         scale=0.45, color=(200, 200, 200), thickness=1)

    @property
    def progress_str(self):
        if not self.active:
            return ""
        total = len(GRID_POINTS) if self.phase == "grid" else len(HP_DIRECTIONS)
        return f"{min(self.step + 1, total)}/{total}"

    @property
    def num_points(self):
        if self._train_X is not None:
            return len(self._train_X)
        if self.active:
            return len(self.features_)
        return 0
