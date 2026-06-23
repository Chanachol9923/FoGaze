"""moveit_pick_executor — drive a real (simulated) arm via MoveIt 2.

Drop-in replacement for ``mock_arm_executor``: same input topic
(``fogaze/pick_pose``), but plans and executes an actual top-down pick with
MoveIt 2 — approach, descend, close gripper, lift.

Uses **pymoveit2** for a compact Python MoveGroup/gripper interface.  It is
not an apt package; install it into the workspace once::

    cd ros2_ws/src
    git clone https://github.com/AndrejOrsula/pymoveit2.git
    cd .. && rosdep install --from-paths src -y --ignore-src && colcon build

If pymoveit2 is missing the node stays alive and logs these instructions
instead of crashing the launch, so the rest of the pipeline keeps running.
"""

from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

try:
    from pymoveit2 import MoveIt2, MoveIt2Gripper
    from pymoveit2.robots import panda
    _HAVE_PYMOVEIT2 = True
except ImportError:
    _HAVE_PYMOVEIT2 = False


class MoveItPickExecutor(Node):
    APPROACH_HEIGHT = 0.15   # m above target for pre-grasp / lift
    # Top-down grasp: gripper pointing straight down (quaternion x,y,z,w).
    DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]

    def __init__(self):
        super().__init__("moveit_pick_executor")
        self.declare_parameter("synchronous", True)
        self._busy = False
        self._status = self.create_publisher(String, "fogaze/pickup_status", 1)

        if not _HAVE_PYMOVEIT2:
            self.get_logger().error(
                "pymoveit2 not found — install it (see module docstring). "
                "Node will idle so the launch stays up.")
            self.create_subscription(
                PoseStamped, "fogaze/pick_pose", self._warn_missing, 1)
            return

        cb = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=cb,
        )
        self.gripper = MoveIt2Gripper(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=cb,
        )
        self.create_subscription(
            PoseStamped, "fogaze/pick_pose", self._on_goal, 1)
        self.get_logger().info(
            "moveit_pick_executor ready — listening on fogaze/pick_pose")

    # ───────────────────────────────────────────────────────────────────
    def _warn_missing(self, _ps):
        self._say("REJECT — pymoveit2 not installed (cannot drive arm)")

    def _say(self, text):
        self._status.publish(String(data=text))
        self.get_logger().info(text)

    def _on_goal(self, ps: PoseStamped) -> None:
        if self._busy:
            self.get_logger().warn("arm busy — ignoring new pick goal")
            return
        # Run the (blocking) motion sequence off the executor thread.
        Thread(target=self._execute_pick, args=(ps,), daemon=True).start()

    def _execute_pick(self, ps: PoseStamped) -> None:
        self._busy = True
        try:
            x, y, z = (ps.pose.position.x, ps.pose.position.y,
                       ps.pose.position.z)
            self.moveit2.max_velocity = 0.3
            self.moveit2.max_acceleration = 0.3

            self._say(f"PICK approach -> [{x:.2f}, {y:.2f}, {z:.2f}]")
            self.gripper.open()
            self.gripper.wait_until_executed()
            self._move_to(x, y, z + self.APPROACH_HEIGHT)

            self._say("PICK descend")
            self._move_to(x, y, z)

            self._say("PICK grasp")
            self.gripper.close()
            self.gripper.wait_until_executed()

            self._say("PICK lift")
            self._move_to(x, y, z + self.APPROACH_HEIGHT)
            self._say("PICK COMPLETE")
        except Exception as e:  # noqa: BLE001 — surface any planning failure
            self._say(f"PICK FAILED — {e}")
        finally:
            self._busy = False

    def _move_to(self, x, y, z) -> None:
        self.moveit2.move_to_pose(position=[x, y, z], quat_xyzw=self.DOWN_QUAT)
        self.moveit2.wait_until_executed()


def main(args=None):
    rclpy.init(args=args)
    node = MoveItPickExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
