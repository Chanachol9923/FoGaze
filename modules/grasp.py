"""Shared, ROS-free geometry + grasp-feasibility logic.

Imported by both the FoGaze app (``main.py``) and the ROS 2 manipulation
nodes (``fogaze_manip``) so the 3D projection and the "can the arm grab
this?" rules live in exactly one place.

Nothing here imports ROS or OpenCV — only ``numpy`` (lazily, for the
optional calibration loader), so it is safe to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


# Default PrimeSense colour-camera intrinsics (640x480).  These match the
# constants historically used in fogaze_node.py.  For accurate metric poses,
# load the real values from calibration.npz via CameraIntrinsics.from_npz().
DEFAULT_FX = 532.0
DEFAULT_FY = 532.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the colour camera (pixels)."""

    fx: float = DEFAULT_FX
    fy: float = DEFAULT_FY
    cx: float = DEFAULT_CX
    cy: float = DEFAULT_CY

    @classmethod
    def from_npz(cls, path) -> "CameraIntrinsics":
        """Load fx/fy/cx/cy from a calibration .npz holding a 3x3 K matrix."""
        import numpy as np

        data = np.load(path)
        for key in ("mtx", "camera_matrix", "K", "cameraMatrix"):
            if key in data:
                K = np.asarray(data[key], dtype=float)
                return cls(fx=float(K[0, 0]), fy=float(K[1, 1]),
                           cx=float(K[0, 2]), cy=float(K[1, 2]))
        raise KeyError(
            f"No camera matrix (mtx/camera_matrix/K) found in {path}; "
            f"keys present: {list(data.keys())}")


def project_bbox_to_3d(bbox, z_m: float,
                       intr: CameraIntrinsics = CameraIntrinsics()):
    """Back-project a bbox centre at depth ``z_m`` (metres) to a 3D point.

    Returns (x, y, z) in the camera *optical* frame (x right, y down,
    z forward) — the standard ROS ``*_color_optical_frame`` convention.
    """
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    x = (u - intr.cx) * z_m / intr.fx
    y = (v - intr.cy) * z_m / intr.fy
    return (x, y, z_m)


def bbox_size_m(bbox, z_m: float, intr: CameraIntrinsics = CameraIntrinsics()):
    """Approximate real-world (width, height) of a bbox in metres at ``z_m``."""
    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1) * z_m / intr.fx
    h = abs(y2 - y1) * z_m / intr.fy
    return (w, h)


@dataclass(frozen=True)
class GraspParams:
    """Feasibility thresholds for a single parallel-jaw gripper.

    Defaults are tuned for a Franka Emika Panda hand (≈8 cm max opening,
    ≈0.85 m practical reach).  Override per-arm in graspability.yaml.
    """

    min_reach_m: float = 0.20      # closer than this → inside the arm/base
    max_reach_m: float = 0.85      # farther than this → cannot be reached
    gripper_max_m: float = 0.08    # widest object the gripper can close on
    min_size_m: float = 0.02       # smaller → likely detection noise


def classify_graspable(z_m, size_wh_m, params: GraspParams = GraspParams()):
    """Decide whether an object can be picked up.

    Args:
        z_m:        distance to the object in metres (``None`` if unknown).
        size_wh_m:  (width, height) in metres (``None`` if unknown).
        params:     gripper/reach thresholds.

    Returns:
        (graspable: bool, reason: str) — ``reason`` is ``"ok"`` when graspable,
        otherwise a short machine-readable code for logging / UI.
    """
    if z_m is None or size_wh_m is None:
        return False, "no-depth"
    w, h = size_wh_m
    if z_m < params.min_reach_m:
        return False, "too-close"
    if z_m > params.max_reach_m:
        return False, "out-of-reach"
    if min(w, h) > params.gripper_max_m:
        return False, "too-wide-for-gripper"
    if max(w, h) < params.min_size_m:
        return False, "too-small"
    return True, "ok"
