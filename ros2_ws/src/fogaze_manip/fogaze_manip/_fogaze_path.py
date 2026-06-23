"""Locate the FoGaze project root and expose it on sys.path.

Lets the ROS nodes reuse the app's shared, ROS-free logic
(``modules.grasp``) instead of duplicating the projection / graspability
rules.  Mirrors the discovery already used by fogaze_rviz/fogaze_node.py.
"""

import os
import sys


def fogaze_root() -> str:
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(12):
        if os.path.isfile(os.path.join(d, "modules", "grasp.py")):
            return d
        d = os.path.dirname(d)
    # Fallback: assume the standard ros2_ws/src/<pkg>/<pkg> layout.
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))


def ensure_on_path() -> str:
    root = fogaze_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
