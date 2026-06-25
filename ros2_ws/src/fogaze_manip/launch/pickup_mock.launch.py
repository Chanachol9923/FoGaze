"""Tier 1 demo — full pipeline with the mock arm (no Gazebo/MoveIt needed).

Brings up:
  * pickup_planner        (graspability + frame handling)
  * mock_arm_executor     (animates the pick in RViz)
  * static camera TF       so markers/poses have a frame to live in

Run the FoGaze app separately with ROS publishing enabled:
    python3 main.py --ros

Then triple-blink at a graspable object and watch fogaze/pickup_status
and the fogaze/arm_marker move in RViz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory("fogaze_manip"),
        "config", "graspability.yaml")

    return LaunchDescription([
        # A world->camera frame so RViz has something to anchor to.  Replace
        # with a real eye-to-hand calibration once a robot is in the loop.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_camera",
            arguments=["0", "0", "0.6", "0", "0", "0",
                       "world", "camera_color_optical_frame"],
        ),
        Node(
            package="fogaze_manip",
            executable="pickup_planner",
            name="pickup_planner",
            parameters=[cfg],
            output="screen",
        ),
        Node(
            package="fogaze_manip",
            executable="mock_arm_executor",
            name="mock_arm_executor",
            output="screen",
        ),
    ])
