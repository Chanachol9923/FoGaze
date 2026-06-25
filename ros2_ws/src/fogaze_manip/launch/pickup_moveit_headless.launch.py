"""Tier 2 (headless) — simulated Panda + MoveIt 2, no RViz.

Same pipeline as ``pickup_moveit.launch.py`` but builds the move_group +
ros2_control mock-hardware stack directly (instead of including the Panda
``demo.launch.py``) so it can run **without a display** — for CI, smoke tests,
or a headless robot box.  The arm still really plans and executes; you just
watch ``/fogaze/pickup_status`` and ``/joint_states`` instead of RViz.

    ros2 launch fogaze_manip pickup_moveit_headless.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    manip_share = get_package_share_directory("fogaze_manip")
    cfg = os.path.join(manip_share, "config", "graspability.yaml")

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    panda_share = get_package_share_directory(
        "moveit_resources_panda_moveit_config")
    ros2_controllers_path = os.path.join(
        panda_share, "config", "ros2_controllers.yaml")

    move_group = Node(
        package="moveit_ros_move_group", executable="move_group",
        output="screen", parameters=[moveit_config.to_dict()])

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        output="log", parameters=[moveit_config.robot_description])

    ros2_control = Node(
        package="controller_manager", executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[("/controller_manager/robot_description",
                     "/robot_description")],
        output="screen")

    def spawner(name):
        return Node(package="controller_manager", executable="spawner",
                    arguments=[name, "-c", "/controller_manager"])

    world_to_base = Node(
        package="tf2_ros", executable="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"])

    # Eye-to-hand: sim-default camera pose (matches pickup_moveit.launch.py).
    arm_to_camera = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="arm_to_camera", output="log",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.4",
            "--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
            "--frame-id", "panda_link0",
            "--child-frame-id", "camera_color_optical_frame"])

    pickup_planner = Node(
        package="fogaze_manip", executable="pickup_planner",
        name="pickup_planner", parameters=[cfg, {"use_tf": True}],
        output="screen")
    pick_executor = Node(
        package="fogaze_manip", executable="moveit_pick_executor",
        name="moveit_pick_executor", output="screen")
    scene_publisher = Node(
        package="fogaze_manip", executable="scene_publisher",
        name="scene_publisher", parameters=[cfg, {"use_tf": True}],
        output="screen")

    return LaunchDescription([
        robot_state_publisher, move_group, ros2_control,
        spawner("joint_state_broadcaster"),
        spawner("panda_arm_controller"),
        spawner("panda_hand_controller"),
        world_to_base, arm_to_camera,
        pickup_planner, pick_executor, scene_publisher,
    ])
