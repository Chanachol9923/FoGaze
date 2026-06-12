"""Dark-theme UI components with animations.

All drawing functions operate on a BGR numpy canvas.
"""

from __future__ import annotations

import time
import math
from collections import deque

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Dark theme palette
# ═══════════════════════════════════════════════════════════════════════

class Theme:
    BG_PRIMARY     = (0x0A, 0x0A, 0x19)  # deep navy black
    BG_SECONDARY   = (0x14, 0x14, 0x28)  # slightly lighter
    BG_TERTIARY    = (0x1E, 0x1E, 0x3A)  # card surface
    BG_TOAST       = (0x14, 0x14, 0x28)  # toast bg
    ACCENT_CYAN    = (0xFF, 0xFF, 0x00)  # calibration targets
    ACCENT_RED     = (0x32, 0x32, 0xFF)  # gaze cursor
    ACCENT_GREEN   = (0x64, 0xC8, 0x00)  # calibrated / success
    ACCENT_ORANGE  = (0x00, 0xA5, 0xFF)  # warning / countdown
    TEXT_MAIN      = (0xE6, 0xDC, 0xDC)  # light grey text
    TEXT_DIM       = (0x80, 0x80, 0x80)  # dim text
    BORDER         = (0x40, 0x40, 0x60)  # subtle border
    GLOW           = (0x80, 0x80, 0x00)  # cyan glow (BGR)


# ═══════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════════════════════

def draw_text_stroke(canvas, text, pos, scale=0.6, color=Theme.TEXT_MAIN,
                     thickness=2, outline=True):
    """Text with black outline for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    if outline:
        ot = thickness + 2
        cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), ot,
                    cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def _lerp_color(c1, c2, t):
    """BGR tuple linear interpolation."""
    t = np.clip(t, 0, 1)
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _rounded_rect(canvas, x1, y1, x2, y2, color, radius=8, alpha=None):
    """Draw a filled rounded rectangle.  Optionally with alpha overlay."""
    h, w = canvas.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if alpha is not None and 0 <= alpha < 1:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)


def _glow_circle(canvas, cx, cy, radius, color, glow_intensity=0.3):
    """Draw a circle with a soft glow effect."""
    if glow_intensity > 0.01:
        for i in range(3, 0, -1):
            r = radius + i * 6
            alpha = glow_intensity / (i * 1.5)
            overlay = canvas.copy()
            cv2.circle(overlay, (cx, cy), r, color, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.circle(canvas, (cx, cy), radius, color, -1)


# ═══════════════════════════════════════════════════════════════════════
#  Animated Top Bar
# ═══════════════════════════════════════════════════════════════════════

class TopBar:
    HEIGHT = 56

    def __init__(self):
        self._fps_val = 0
        self._fps_str = "FPS: --"
        self._fps_alpha = 0.0
        self._toasts = []  # list of (text, color, expiry, x_offset)

    def set_fps(self, fps: int):
        """Call every frame with current FPS."""
        new = f"FPS: {fps}"
        if new != self._fps_str:
            self._fps_str = new
            self._fps_alpha = 1.0

    def toast(self, text: str, color=Theme.ACCENT_GREEN, duration=2.5):
        """Show a notification that slides in from the right."""
        self._toasts.append((text, color, time.time() + duration, 0.0))

    def draw(self, canvas, sw: int):
        """Draw the top bar and toasts."""
        # Bar background
        _rounded_rect(canvas, 0, 0, sw, self.HEIGHT, Theme.BG_SECONDARY,
                      alpha=0.92)

        # Left: app name
        draw_text_stroke(canvas, "FoGaze v3",
                         (16, self.HEIGHT - 16),
                         scale=0.6, color=Theme.ACCENT_CYAN, thickness=1)

        # Center: status / calibration state
        cx = sw // 2
        draw_text_stroke(canvas, self._fps_str,
                         (cx - 30, self.HEIGHT - 16),
                         scale=0.5, color=Theme.TEXT_DIM, thickness=1)

        # Right: toasts (slide in)
        now = time.time()
        self._toasts = [(t, c, e, _) for t, c, e, _ in self._toasts if e > now]
        offset = 0
        for i, (text, color, expiry, _) in enumerate(self._toasts):
            elapsed = expiry - now
            remaining = elapsed / 2.5  # 1→0

            # Slide in from right
            slide = max(0, 1 - (2.5 - elapsed) / 0.3) if elapsed > 2.2 else 1.0
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            tx = int(sw - (sw - tw - 32) * slide - tw - 16 - offset)
            ty = self.HEIGHT - 14
            _rounded_rect(canvas, tx - 8, ty - 20, tx + tw + 8, ty + 6,
                          Theme.BG_TOAST, alpha=0.85 * remaining)
            draw_text_stroke(canvas, text, (tx, ty), scale=0.5,
                             color=color, thickness=1)
            offset += tw + 32

        # Bottom separator line
        cv2.line(canvas, (0, self.HEIGHT), (sw, self.HEIGHT),
                 Theme.BORDER, 1)


# ═══════════════════════════════════════════════════════════════════════
#  Animated Gaze Cursor
# ═══════════════════════════════════════════════════════════════════════

class GazeCursor:
    def __init__(self, trail_len=15):
        self._trail = deque(maxlen=trail_len)
        self._alpha = 0.15  # always minimally visible
        self._alpha_step = 0.06
        self._pulse_t = 0.0
        self.display_x = None
        self.display_y = None

    def update(self, x: float, y: float, active: bool):
        """Update state.  Called every frame."""
        self._pulse_t += 0.1
        if active:
            self._alpha = min(self._alpha + self._alpha_step, 1.0)
        else:
            self._alpha = max(self._alpha - self._alpha_step, 0.15)
        dx = self.display_x if self.display_x is not None else x
        dy = self.display_y if self.display_y is not None else y
        self._trail.append((dx, dy))

    def draw(self, canvas: np.ndarray):
        """Draw cursor with trail and glow."""

        h, w = canvas.shape[:2]

        # Trail
        trail_pts = list(self._trail)
        for i, (px, py) in enumerate(trail_pts):
            frac = (i + 1) / len(trail_pts) if trail_pts else 0
            t_alpha = self._alpha * frac * 0.4
            if t_alpha > 0.01:
                r = int(3 + 4 * frac)
                overlay = canvas.copy()
                cv2.circle(overlay, (int(px), int(py)), r, Theme.ACCENT_RED, -1)
                cv2.addWeighted(overlay, t_alpha, canvas, 1 - t_alpha, 0, canvas)

        # Current position
        cx, cy = int(self._trail[-1][0]), int(self._trail[-1][1]) if self._trail else (w // 2, h // 2)
        pulse_r = 15 + int(3 * math.sin(self._pulse_t))

        _glow_circle(canvas, cx, cy, pulse_r, Theme.ACCENT_RED, 0.15 * self._alpha)
        cv2.circle(canvas, (cx, cy), 5, Theme.ACCENT_RED, -1)
        cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), 1)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration countdown arc
# ═══════════════════════════════════════════════════════════════════════

def draw_countdown_arc(canvas, center, radius, progress, color=Theme.ACCENT_CYAN,
                       thickness=4):
    """Smoothstepped arc from 0→360°."""
    p = np.clip(progress, 0, 1)
    eased = p * p * (3 - 2 * p)
    ang = int(360 * (1 - eased))
    cv2.ellipse(canvas, center, (radius, radius), 0, -90, -90 + ang,
                color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration target (pulsing circle)
# ═══════════════════════════════════════════════════════════════════════

def draw_pulse_target(canvas, cx, cy, phase_time, radius_base=20):
    """Pulsing circle target — sine wave radius oscillation."""
    r = radius_base + int(10 * abs(math.sin(2 * math.pi * phase_time)))
    cv2.circle(canvas, (cx, cy), r, Theme.ACCENT_CYAN, -1)
    cv2.circle(canvas, (cx, cy), r + 4, (255, 255, 255), 1)


# ═══════════════════════════════════════════════════════════════════════
#  Progress bar (bottom)
# ═══════════════════════════════════════════════════════════════════════

def draw_progress_bar(canvas, step, total, sw, sh,
                      color=Theme.ACCENT_CYAN):
    """Bottom progress bar with percentage."""
    bar_w = int(sw * 0.4)
    bar_h = 6
    x1 = (sw - bar_w) // 2
    y1 = sh - 30
    filled = bar_w * (step + 1) // total if total > 0 else 0
    cv2.rectangle(canvas, (x1, y1), (x1 + bar_w, y1 + bar_h),
                  (60, 60, 80), -1)
    cv2.rectangle(canvas, (x1, y1), (x1 + filled, y1 + bar_h),
                  color, -1)
    pct = f"{int(100 * (step + 1) // total)}%"
    draw_text_stroke(canvas, pct, (x1 + bar_w + 10, y1 + bar_h - 2),
                     scale=0.4, color=color, thickness=1)


# ═══════════════════════════════════════════════════════════════════════
#  PIP: face crop + iris crop
# ═══════════════════════════════════════════════════════════════════════

class PIPDisplay:
    def __init__(self, face_size=140, iris_size=80):
        self.face_size = face_size
        self.iris_size = iris_size
        self._border_glow = 0.0
        self._glow_dir = 0.02

    def draw(self, canvas, face_crop, iris_crop):
        """Draw PIPs at bottom-right corner."""
        h, w = canvas.shape[:2]
        gap = 12
        self._border_glow += self._glow_dir
        if self._border_glow >= 0.8 or self._border_glow <= 0.2:
            self._glow_dir *= -1

        y_offset = h - gap

        # Iris zoom (bottom-right)
        if iris_crop is not None:
            iz = self.iris_size
            y_offset -= iz
            x1 = w - iz - gap
            canvas[y_offset:y_offset+iz, x1:x1+iz] = iris_crop
            cv2.rectangle(canvas, (x1 - 1, y_offset - 1),
                          (x1 + iz, y_offset + iz),
                          _lerp_color(Theme.ACCENT_CYAN, Theme.ACCENT_RED,
                                      self._border_glow),
                          2)
            draw_text_stroke(canvas, "IRIS", (x1 + 3, y_offset + 14),
                             scale=0.3, color=Theme.ACCENT_CYAN, thickness=1,
                             outline=True)

        # Face crop (above iris)
        if face_crop is not None:
            fs = self.face_size
            y_offset -= fs + gap
            x1 = w - fs - gap
            pip = cv2.resize(face_crop, (fs, fs))
            canvas[y_offset:y_offset+fs, x1:x1+fs] = pip
            cv2.rectangle(canvas, (x1 - 1, y_offset - 1),
                          (x1 + fs, y_offset + fs),
                          _lerp_color(Theme.ACCENT_GREEN, Theme.ACCENT_CYAN,
                                      self._border_glow),
                          2)
            draw_text_stroke(canvas, "FACE", (x1 + 3, y_offset + 14),
                             scale=0.3, color=Theme.ACCENT_GREEN, thickness=1,
                             outline=True)


# ═══════════════════════════════════════════════════════════════════════
#  HUD info panel (top-left info card)
# ═══════════════════════════════════════════════════════════════════════

class HUDInfo:
    def draw(self, canvas, items: list[tuple[str, tuple]]):
        """Draw a semi-transparent info card.  items = [(label, color), ...]."""
        y = 70
        x = 12
        for text, color in items:
            draw_text_stroke(canvas, text, (x, y), scale=0.45,
                             color=color, thickness=1)
            y += 22
        # Semi-transparent bg behind info
        # (drawn first, but we draw on top for simplicity)


# ═══════════════════════════════════════════════════════════════════════
#  Gaze info overlay (bottom-left)
# ═══════════════════════════════════════════════════════════════════════

def draw_gaze_coords(canvas, gx, gy, label="GAZE"):
    """Show gaze coordinates at bottom-left."""
    h, w = canvas.shape[:2]
    txt = f"{label}: ({gx:.0f}, {gy:.0f})"
    draw_text_stroke(canvas, txt, (12, h - 16), scale=0.4,
                     color=Theme.TEXT_DIM, thickness=1)
