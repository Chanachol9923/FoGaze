"""Face-lock: follow the face and feed only that region to the gaze model.

The eye landmarks MediaPipe returns get much sharper when the face fills the
input (the detector downsamples internally, so a small far-away face becomes
imprecise).  FaceLock keeps a smoothed square ROI around the face — taken from
the previous frame's landmarks — crops the next frame to it, and lets the gaze
pipeline run on that zoomed-in crop.  Because the crop is applied inside the
wrapped ``extract_features``, calibration and runtime use the SAME framing,
so the trained model stays valid.

The features eyetrax produces are already translation/rotation/scale
normalised, so the crop doesn't shift them — it only improves landmark
precision.  Robustness: generous padding + a miss counter that releases the
lock (back to the full frame) when the face is lost so it can re-acquire.
"""

from __future__ import annotations


class FaceLock:
    def __init__(self, enabled: bool = True, pad: float = 0.7,
                 smooth: float = 0.6, min_px: int = 64, max_miss: int = 8):
        self.enabled = enabled
        self.pad = pad          # extra margin around the face box (fraction)
        self.smooth = smooth    # ROI EMA (higher = steadier, laggier)
        self.min_px = min_px    # ignore a ROI smaller than this (px)
        self.max_miss = max_miss
        self.roi = None         # (x1, y1, x2, y2) full-frame px, float
        self.last_crop = None   # the crop most recently fed to the model
        self.last_origin = (0, 0)
        self._miss = 0

    def crop(self, frame):
        """Return (crop_bgr, (origin_x, origin_y)) for *frame*."""
        h, w = frame.shape[:2]
        if not self.enabled or self.roi is None:
            self.last_crop, self.last_origin = frame, (0, 0)
            return frame, (0, 0)
        x1, y1, x2, y2 = self.roi
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y2 = int(max(y1 + 1, min(h, y2)))
        if x2 - x1 < self.min_px or y2 - y1 < self.min_px:
            self.last_crop, self.last_origin = frame, (0, 0)
            return frame, (0, 0)
        crop = frame[y1:y2, x1:x2]
        self.last_crop, self.last_origin = crop, (x1, y1)
        return crop, (x1, y1)

    def update_from_landmarks(self, landmarks, crop_w, crop_h):
        """Recompute the ROI from landmarks (normalised to the last crop)."""
        ox, oy = self.last_origin
        xs = [p.x * crop_w + ox for p in landmarks]
        ys = [p.y * crop_h + oy for p in landmarks]
        fx1, fy1, fx2, fy2 = min(xs), min(ys), max(xs), max(ys)
        cx, cy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
        half = max(fx2 - fx1, fy2 - fy1) * (1.0 + self.pad) * 0.5
        new = (cx - half, cy - half, cx + half, cy + half)
        if self.roi is None:
            self.roi = new
        else:
            a = self.smooth
            self.roi = tuple(a * o + (1 - a) * n
                             for o, n in zip(self.roi, new))
        self._miss = 0

    def miss(self):
        """Call when no face was found; release the lock after enough misses."""
        self._miss += 1
        if self._miss > self.max_miss:
            self.roi = None
