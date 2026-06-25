#!/usr/bin/env python3
"""Standalone stereo-camera tester for FoGaze.

Opens a left + right USB camera, runs StereoSGBM block matching, and shows
the depth map live — no GUI/ROS/MediaPipe, just the stereo pipeline so you can
check wiring, swap indices, and tune ``focal_px`` / ``baseline`` until the
reported distances look right.

Usage:
    python3 test_stereo.py                 # left=0 right=1
    python3 test_stereo.py --left 2 --right 4
    python3 test_stereo.py --width 640 --height 480

Windows:
    "controls"     trackbars for every parameter + checkboxes
    "stereo L|R"   side-by-side raw camera feeds (toggle via "show raw")
    "depth"        colourised depth map

Controls (drag the trackbars in the "controls" window):
    show raw    0/1  show or hide the raw "stereo L|R" window
    swap L/R    0/1  swap the two cameras (in case they're plugged in reversed)
    focal_px         pinhole focal length in pixels
    baseline x10     baseline in mm (i.e. cm * 10)
    num_disp         disparity search range (snapped to a multiple of 16)
    block_size       SGBM matched block size (forced odd, >=5)

Press 'q' or ESC to quit. Click anywhere in the "depth" window to print the
distance at that pixel.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


# Live-tunable stereo parameters (mirrors modules/stereo_depth_estimator.py).
PARAMS = {
    "focal_px": 500,     # pinhole focal length in pixels (per-camera)
    "baseline_cm": 6.0,  # distance between the two camera centres (cm)
    "num_disp": 64,      # disparity search range (multiple of 16)
    "block_size": 7,     # SGBM matched block size (odd, >=5)
}

CONTROLS_WIN = "controls"
RAW_WIN = "stereo L|R"
DEPTH_WIN = "depth"


def open_cam(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def build_matcher():
    num = max(16, (int(PARAMS["num_disp"]) // 16) * 16)
    block = max(5, int(PARAMS["block_size"]) | 1)  # force odd
    return cv2.StereoSGBM_create(
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


def compute_depth(grayL, grayR, matcher) -> np.ndarray:
    disp = matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    focal = float(PARAMS["focal_px"])
    baseline_mm = float(PARAMS["baseline_cm"]) * 10.0
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0.5
    depth[valid] = focal * baseline_mm / disp[valid]
    depth[depth > 1e5] = 0.0
    return depth


def colorise(depth: np.ndarray) -> np.ndarray:
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


# Shared state for the mouse callback (depth map + last click).
_state = {"depth": None, "click": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _state["click"] = (x, y)
        d = _state["depth"]
        if d is not None and 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
            mm = d[y, x]
            if mm > 0:
                print(f"[click] ({x},{y}) -> {mm / 10.0:.1f} cm")
            else:
                print(f"[click] ({x},{y}) -> no disparity")


def _noop(_):
    pass


def build_controls():
    """Create the 'controls' window with trackbars (and 0/1 checkboxes)."""
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 420, 260)
    cv2.createTrackbar("show raw", CONTROLS_WIN, 1, 1, _noop)
    cv2.createTrackbar("swap L/R", CONTROLS_WIN, 0, 1, _noop)
    cv2.createTrackbar("focal_px", CONTROLS_WIN, PARAMS["focal_px"], 2000, _noop)
    cv2.createTrackbar("baseline x10", CONTROLS_WIN,
                       int(PARAMS["baseline_cm"] * 10), 500, _noop)
    cv2.createTrackbar("num_disp", CONTROLS_WIN, PARAMS["num_disp"], 256, _noop)
    cv2.createTrackbar("block_size", CONTROLS_WIN, PARAMS["block_size"], 31, _noop)


def read_controls() -> tuple[bool, bool]:
    """Pull trackbar values into PARAMS; return (show_raw, swap)."""
    show_raw = cv2.getTrackbarPos("show raw", CONTROLS_WIN) == 1
    swap = cv2.getTrackbarPos("swap L/R", CONTROLS_WIN) == 1
    PARAMS["focal_px"] = max(10, cv2.getTrackbarPos("focal_px", CONTROLS_WIN))
    PARAMS["baseline_cm"] = max(
        0.5, cv2.getTrackbarPos("baseline x10", CONTROLS_WIN) / 10.0)
    PARAMS["num_disp"] = max(16, cv2.getTrackbarPos("num_disp", CONTROLS_WIN))
    PARAMS["block_size"] = max(5, cv2.getTrackbarPos("block_size", CONTROLS_WIN))
    return show_raw, swap


def main():
    ap = argparse.ArgumentParser(description="Standalone stereo camera tester")
    ap.add_argument("--left", type=int, default=0, help="left camera index")
    ap.add_argument("--right", type=int, default=1, help="right camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    capL = open_cam(args.left, args.width, args.height)
    capR = open_cam(args.right, args.width, args.height)
    if not capL.isOpened() or not capR.isOpened():
        raise SystemExit(
            f"Cannot open stereo pair (L={args.left}, R={args.right}). "
            "Check indices with: ls /dev/video*")

    print(f"[test_stereo] L={args.left} R={args.right} @ "
          f"{args.width}x{args.height}. Press 'q' to quit.")

    build_controls()
    cv2.namedWindow(DEPTH_WIN)
    cv2.setMouseCallback(DEPTH_WIN, on_mouse)
    raw_open = False

    while True:
        show_raw, swap = read_controls()

        okL, fL = capL.read()
        okR, fR = capR.read()
        if not okL or not okR:
            print("[test_stereo] frame grab failed, retrying...")
            cv2.waitKey(30)
            continue

        if swap:
            fL, fR = fR, fL

        grayL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        depth = compute_depth(grayL, grayR, build_matcher())
        _state["depth"] = depth

        depth_vis = colorise(depth)
        if _state["click"] is not None:
            cv2.drawMarker(depth_vis, _state["click"], (255, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)

        # Overlay the live params on the depth window.
        valid_ratio = float((depth > 0).mean()) * 100.0
        hud = (f"f={PARAMS['focal_px']}px  b={PARAMS['baseline_cm']:.1f}cm  "
               f"disp={PARAMS['num_disp']}  blk={PARAMS['block_size']}  "
               f"valid={valid_ratio:.0f}%  swap={swap}")
        cv2.putText(depth_vis, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(DEPTH_WIN, depth_vis)

        # Show/hide the raw side-by-side window via the "show raw" checkbox.
        if show_raw:
            cv2.imshow(RAW_WIN, np.hstack([fL, fR]))
            raw_open = True
        elif raw_open:
            cv2.destroyWindow(RAW_WIN)
            raw_open = False

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    print("[test_stereo] Closed.")


if __name__ == "__main__":
    main()
