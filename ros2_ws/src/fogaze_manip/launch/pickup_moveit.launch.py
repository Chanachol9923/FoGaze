"""Tier 2 — simulated Panda arm driven by MoveIt 2.

Composes the official ``moveit_resources_panda_moveit_config`` demo (RViz +
MoveIt move_group + ros2_control mock hardware — i.e. a simulated arm) and
layers the FoGaze pickup pipeline on top:

  * panda demo.launch.py   (move_group + RViz + simulated controllers)
  * static panda_link0 -> camera TF   (placeholder eye-to-hand calibration)
  * pickup_planner          (graspability + TF into panda_link0)
  * moveit_pick_executor    (plans/executes the pick via pymoveit2)

Prerequisites (one-time, needs sudo — see fogaze_manip/README.md):
    sudo apt install ros-humble-moveit \
        ros-humble-moveit-resources-panda-moveit-config
    # pymoveit2 into the workspace (see moveit_pick_executor.py docstring)

For full Gazebo *physics*, bring up worlds/fogaze_table.sdf with ros_gz and
spawn the Panda there; this launch uses MoveIt's simulated controllers, which
is sufficient to validate the blink -> classify -> pick behaviour.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    manip_share = get_package_share_directory("fogaze_manip")
    cfg = os.path.join(manip_share, "config", "graspability.yaml")

    panda_demo = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "launch", "demo.launch.py")

    return LaunchDescription([
        # Simulated Panda + MoveIt move_group + RViz.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(panda_demo)),

        # Where the camera sits relative to the arm base.  REPLACE the
        # translation/rotation with your real eye-to-hand calibration —
        # otherwise the arm will reach the wrong place.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="arm_to_camera",
            arguments=["0.5", "0.0", "0.5", "0", "1.57", "0",
                       "panda_link0", "camera_color_optical_frame"],
        ),

        Node(
            package="fogaze_manip",
            executable="pickup_planner",
            name="pickup_planner",
            # use_tf must be true here so poses land in panda_link0.
            parameters=[cfg, {"use_tf": True}],
            output="screen",
        ),
        Node(
            package="fogaze_manip",
            executable="moveit_pick_executor",
            name="moveit_pick_executor",
            output="screen",
        ),
    ])
