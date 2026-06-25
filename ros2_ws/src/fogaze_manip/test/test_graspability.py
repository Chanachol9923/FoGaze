"""Unit tests for the shared graspability rules (pure, ROS-free)."""

import os
import sys

# Make modules.grasp importable without a sourced workspace.
_HERE = os.path.dirname(__file__)
_ROOT = _HERE
for _ in range(12):
    if os.path.isfile(os.path.join(_ROOT, "modules", "grasp.py")):
        break
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from modules.grasp import (  # noqa: E402
    CameraIntrinsics, GraspParams, bbox_size_m, classify_graspable,
    project_bbox_to_3d,
)


def test_graspable_can():
    ok, reason = classify_graspable(0.5, (0.06, 0.12))
    assert ok and reason == "ok"


def test_out_of_reach():
    ok, reason = classify_graspable(1.5, (0.06, 0.06))
    assert not ok and reason == "out-of-reach"


def test_too_close():
    ok, reason = classify_graspable(0.1, (0.06, 0.06))
    assert not ok and reason == "too-close"


def test_too_wide_for_gripper():
    ok, reason = classify_graspable(0.5, (0.30, 0.30))
    assert not ok and reason == "too-wide-for-gripper"


def test_thin_object_still_graspable_on_narrow_axis():
    # A pen is thin on one axis (graspable) but the long axis is fine.
    ok, _ = classify_graspable(0.5, (0.01, 0.14))
    assert ok  # min axis 1cm < gripper_max, max axis 14cm > min_size


def test_too_small_noise():
    ok, reason = classify_graspable(0.5, (0.005, 0.005))
    assert not ok and reason == "too-small"


def test_no_depth():
    ok, reason = classify_graspable(None, None)
    assert not ok and reason == "no-depth"


def test_custom_params_tighter_reach():
    params = GraspParams(max_reach_m=0.4)
    ok, reason = classify_graspable(0.5, (0.06, 0.06), params)
    assert not ok and reason == "out-of-reach"


def test_projection_centre_is_on_axis():
    # A bbox centred on the principal point projects to (0, 0, z).
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    x, y, z = project_bbox_to_3d((300, 220, 340, 260), 1.0, intr)
    assert abs(x) < 1e-9 and abs(y) < 1e-9 and z == 1.0


def test_size_scales_with_depth():
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    near = bbox_size_m((0, 0, 100, 100), 1.0, intr)
    far = bbox_size_m((0, 0, 100, 100), 2.0, intr)
    assert far[0] == near[0] * 2
