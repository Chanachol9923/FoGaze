import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def _find_root():
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "modules", "depth_estimator.py")):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))


_FOGAZE_ROOT = _find_root()


def generate_launch_description():
    pkg_dir = os.path.join(os.path.dirname(__file__), "..")
    rviz_config = os.path.join(pkg_dir, "config", "fogaze.rviz")

    return LaunchDescription([
        ExecuteProcess(
            cmd=["python3", os.path.join(_FOGAZE_ROOT, "main.py")],
            output="screen",
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config],
            output="screen",
            additional_env={"LD_PRELOAD": "/lib/x86_64-linux-gnu/libpthread.so.0"},
        ),
    ])
