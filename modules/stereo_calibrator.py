"""Stereo calibration for the two-camera depth pipeline.

Mirrors the mono ``CameraCalibrator`` style (chessboard 9x6, 25 mm squares,
JSON under ``~/.cache/fogaze3/calib/``) but solves the *stereo* problem:
the relative rotation/translation between the left and right cameras.  From
that it builds rectification maps so every matching point lands on the same
image row — which is exactly what StereoSGBM needs to produce clean depth.

Saved file (``stereo_<L>_<R>.json``) holds only the calibration result
(per-camera intrinsics + the R/T between them + image size).  The heavy
rectify maps and the reprojection matrix ``Q`` are rebuilt from those on
load via ``cv2.stereoRectify`` — keeps the JSON small and portable.

Run standalone to capture and solve:

    python3 modules/stereo_calibrator.py -l 0 -r 2

Controls in the preview window:
    SPACE  capture the current pair (only if the board is seen in BOTH)
    c      calibrate from the captured pairs and save
    u      undo the last capture
    q      quit
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# Same cache location the mono CameraCalibrator uses.
CALIB_DIR = Path.home() / ".cache" / "fogaze3" / "calib"


def find_corners(gray: np.ndarray, pattern):
    """Robust chessboard corner finder.

    Prefers ``findChessboardCornersSB`` (sector-based) which copes far better
    with a board shown on a *screen* — glare, moiré and the lack of a white
    quiet-zone border that the classic detector needs.  Falls back to the
    classic detector + cornerSubPix on older OpenCV builds.
    """
    sb = getattr(cv2, "findChessboardCornersSB", None)
    if sb is not None:
        ret, corners = sb(
            gray, pattern,
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if ret:
            return corners  # SB already returns sub-pixel accurate corners
    ret, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return None
    return cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


class StereoCalibrator:
    CHESSBOARD = (9, 6)
    SQUARE_MM = 25.0

    def __init__(self, left_id: int, right_id: int):
        self.left_id = left_id
        self.right_id = right_id
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CALIB_DIR / f"stereo_{left_id}_{right_id}.json"

        # Calibration result (loaded from JSON or filled by calibrate()).
        self.mtxL = self.distL = None
        self.mtxR = self.distR = None
        self.R = self.T = None
        self._size = None  # (w, h) the calibration was solved at

        # Derived (rebuilt by _build_maps): rectify maps + reprojection.
        self.map1L = self.map2L = None
        self.map1R = self.map2R = None
        self.Q = None
        self._focal_px = None
        self._baseline_mm = None

        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _load(self):
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self.mtxL = np.array(d["mtxL"], dtype=np.float64)
            self.distL = np.array(d["distL"], dtype=np.float64)
            self.mtxR = np.array(d["mtxR"], dtype=np.float64)
            self.distR = np.array(d["distR"], dtype=np.float64)
            self.R = np.array(d["R"], dtype=np.float64)
            self.T = np.array(d["T"], dtype=np.float64)
            self._size = tuple(d["size"])
            self._build_maps()
            print(f"[StereoCalibrator] Loaded cal for pair L={self.left_id} "
                  f"R={self.right_id} (focal={self._focal_px:.1f}px, "
                  f"baseline={self._baseline_mm/10:.2f}cm)")
        except Exception as e:
            print(f"[StereoCalibrator] Load failed: {e}")

    def save(self):
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "mtxL": self.mtxL.tolist(), "distL": self.distL.tolist(),
            "mtxR": self.mtxR.tolist(), "distR": self.distR.tolist(),
            "R": self.R.tolist(), "T": self.T.tolist(),
            "size": list(self._size),
        }, indent=2))
        print(f"[StereoCalibrator] Saved -> {self._path}")

    def reset(self):
        self.mtxL = self.distL = self.mtxR = self.distR = None
        self.R = self.T = None
        if self._path.exists():
            self._path.unlink()

    # ── derived geometry ───────────────────────────────────────────────
    def _build_maps(self):
        w, h = self._size
        R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(
            self.mtxL, self.distL, self.mtxR, self.distR, (w, h),
            self.R, self.T, alpha=0)
        self.map1L, self.map2L = cv2.initUndistortRectifyMap(
            self.mtxL, self.distL, R1, P1, (w, h), cv2.CV_32FC1)
        self.map1R, self.map2R = cv2.initUndistortRectifyMap(
            self.mtxR, self.distR, R2, P2, (w, h), cv2.CV_32FC1)
        self.Q = Q
        # Rectified focal length (px) and baseline (mm) from the projection
        # matrices: P2[0,3] = -fx * Tx, so baseline = -P2[0,3] / fx.
        self._focal_px = float(P1[0, 0])
        self._baseline_mm = float(-P2[0, 3] / P2[0, 0])

    @property
    def ready(self) -> bool:
        return self.map1L is not None and self.Q is not None

    @property
    def focal_px(self) -> float | None:
        return self._focal_px

    @property
    def baseline_mm(self) -> float | None:
        return self._baseline_mm

    def rectify(self, left: np.ndarray, right: np.ndarray):
        """Return (rectified_left, rectified_right) aligned to common rows."""
        rl = cv2.remap(left, self.map1L, self.map2L, cv2.INTER_LINEAR)
        rr = cv2.remap(right, self.map1R, self.map2R, cv2.INTER_LINEAR)
        return rl, rr

    # ── solving ────────────────────────────────────────────────────────
    def _find(self, gray: np.ndarray):
        return find_corners(gray, self.CHESSBOARD)

    def calibrate(self, pairs: list[tuple[np.ndarray, np.ndarray]],
                  w: int, h: int) -> bool:
        """Solve stereo geometry from a list of (left, right) image pairs."""
        objp = np.zeros((self.CHESSBOARD[0] * self.CHESSBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self.CHESSBOARD[0], 0:self.CHESSBOARD[1]].T.reshape(-1, 2)
        objp *= self.SQUARE_MM

        objpoints, ptsL, ptsR = [], [], []
        for i, (fL, fR) in enumerate(pairs):
            cL = self._find(cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY))
            cR = self._find(cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY))
            if cL is not None and cR is not None:
                objpoints.append(objp)
                ptsL.append(cL)
                ptsR.append(cR)
                print(f"[StereoCalibrator] Pair {i+1}/{len(pairs)}: OK")
            else:
                print(f"[StereoCalibrator] Pair {i+1}/{len(pairs)}: "
                      f"board not in {'both' if cL is None and cR is None else ('left' if cL is None else 'right')}")

        if len(objpoints) < 8:
            print(f"[StereoCalibrator] Need >=8 good pairs, got {len(objpoints)}")
            return False

        # Per-camera intrinsics first (more stable than solving everything at once).
        _, mL, dL, *_ = cv2.calibrateCamera(objpoints, ptsL, (w, h), None, None)
        _, mR, dR, *_ = cv2.calibrateCamera(objpoints, ptsR, (w, h), None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        rms, self.mtxL, self.distL, self.mtxR, self.distR, self.R, self.T, *_ = \
            cv2.stereoCalibrate(
                objpoints, ptsL, ptsR, mL, dL, mR, dR, (w, h),
                criteria=crit, flags=flags)

        self._size = (w, h)
        self._build_maps()
        self.save()
        print(f"[StereoCalibrator] Stereo RMS reprojection error: {rms:.4f} px "
              f"(<1.0 is good, >2.0 means recapture)")
        print(f"[StereoCalibrator] baseline={self._baseline_mm/10:.2f}cm "
              f"focal={self._focal_px:.1f}px")
        return True


# ── Standalone capture + solve ──────────────────────────────────────────
def _open(index, w, h):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


# Guided poses shown on screen; press ENTER to capture each one and advance.
WIZARD_STEPS = [
    "Center, facing the cameras",
    "Move board to the LEFT side",
    "Move board to the RIGHT side",
    "Move board UP (top of view)",
    "Move board DOWN (bottom)",
    "Tilt board: top edge away",
    "Tilt board: turn to a side",
    "Bring board CLOSE to cameras",
    "Move board FAR from cameras",
    "One more, any new angle",
]


def _draw_overlay(view, lines, color=(255, 255, 255)):
    """Semi-transparent banner with wrapped instruction lines at the bottom."""
    h, w = view.shape[:2]
    pad, lh = 12, 28
    box_h = pad * 2 + lh * len(lines)
    y0 = h - box_h
    overlay = view.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, view)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(view, txt, (pad, y0 + pad + lh * (i + 1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Guided stereo calibration wizard")
    ap.add_argument("-l", "--left", type=int, default=4)
    ap.add_argument("-r", "--right", type=int, default=6)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    cal = StereoCalibrator(args.left, args.right)
    capL = _open(args.left, args.width, args.height)
    capR = _open(args.right, args.width, args.height)
    if not capL.isOpened() or not capR.isOpened():
        raise SystemExit(f"Cannot open cameras L={args.left} R={args.right}")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    step = 0           # index into WIZARD_STEPS
    solved = False
    win = "Stereo Calibration Wizard"
    print(f"Stereo calibration wizard  L={args.left}  R={args.right}")
    print("ENTER = capture & next step | u = undo | r = restart | q = quit")

    while True:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not okL or not okR:
            continue
        if fR.shape[:2] != fL.shape[:2]:
            fR = cv2.resize(fR, (fL.shape[1], fL.shape[0]))

        # Live board detection so the user knows when a capture will count.
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        cL = find_corners(gL, StereoCalibrator.CHESSBOARD)
        cR = find_corners(gR, StereoCalibrator.CHESSBOARD)
        okBoardL, okBoardR = cL is not None, cR is not None
        dispL, dispR = fL.copy(), fR.copy()
        if okBoardL:
            cv2.drawChessboardCorners(dispL, StereoCalibrator.CHESSBOARD, cL, okBoardL)
        if okBoardR:
            cv2.drawChessboardCorners(dispR, StereoCalibrator.CHESSBOARD, cR, okBoardR)

        both = okBoardL and okBoardR
        view = np.hstack([dispL, dispR])
        cv2.line(view, (dispL.shape[1], 0), (dispL.shape[1], view.shape[0]),
                 (80, 80, 80), 1)

        done = step >= len(WIZARD_STEPS)
        if solved:
            lines = [
                ("DONE - calibration saved.", (0, 255, 0)),
                ("Quit (q) and run: python3 modules/stereo_depth_estimator.py "
                 f"-l {args.left} -r {args.right}", (255, 255, 255)),
            ]
        elif done:
            lines = [
                (f"All {len(pairs)} poses captured. Press ENTER to CALIBRATE.",
                 (0, 255, 255)),
                ("(u = undo last,  r = restart,  q = quit)", (200, 200, 200)),
            ]
        else:
            ok_txt = "READY - press ENTER" if both else "Hold board so BOTH views see it"
            lines = [
                (f"Step {step+1}/{len(WIZARD_STEPS)}  (captured {len(pairs)})",
                 (255, 255, 0)),
                (WIZARD_STEPS[step], (255, 255, 255)),
                (ok_txt, (0, 255, 0) if both else (0, 0, 255)),
            ]
        _draw_overlay(view, lines)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("u") and pairs:
            pairs.pop()
            step = max(0, step - 1)
            solved = False
            print(f"  undo -> {len(pairs)} pairs (back to step {step+1})")
        elif key == ord("r"):
            pairs.clear()
            step = 0
            solved = False
            print("  restart")
        elif key in (13, 10):  # ENTER
            if solved:
                pass
            elif done:
                if cal.calibrate(pairs, fL.shape[1], fL.shape[0]):
                    solved = True
                    print("Calibration saved.")
                else:
                    print("  Calibration failed - press r to restart and recapture.")
            elif both:
                pairs.append((fL.copy(), fR.copy()))
                step += 1
                print(f"  captured pose {step}/{len(WIZARD_STEPS)}")
            else:
                print("  board not visible in BOTH cameras - reposition")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
