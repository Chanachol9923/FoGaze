"""mock_arm_executor — dependency-free stand-in for the real arm.

Subscribes to ``fogaze/pick_pose`` and "performs" a pick: it walks a gripper
marker through approach -> grasp -> lift in RViz and logs each phase.  This
lets the entire perception -> blink -> classify -> pick pipeline be demoed
after a plain ``colcon build``, before Gazebo or MoveIt are installed.

Swap this node for ``moveit_pick_executor`` (same input topic) to drive a
real simulated arm.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import String, ColorRGBA


class MockArmExecutor(Node):
    APPROACH_HEIGHT = 0.15   # m above the target to start from
    PHASE_DT = 0.4           # s between animation steps

    def __init__(self):
        super().__init__("mock_arm_executor")
        self.sub = self.create_subscription(
            PoseStamped, "fogaze/pick_pose", self._on_goal, 1)
        self.pub_marker = self.create_publisher(Marker, "fogaze/arm_marker", 1)
        self.pub_status = self.create_publisher(String, "fogaze/pickup_status", 1)
        self._busy = False
        self.get_logger().info(
            "mock_arm_executor ready — listening on fogaze/pick_pose")

    def _on_goal(self, ps: PoseStamped) -> None:
        if self._busy:
            self.get_logger().warn("arm busy — ignoring new pick goal")
            return
        self._busy = True
        self._frame = ps.header.frame_id
        self._target = (ps.pose.position.x, ps.pose.position.y,
                        ps.pose.position.z)
        self.get_logger().info(
            f"[mock arm] new goal {self._target} in {self._frame}")
        # Scripted phases: (label, z-offset above target, grasped?)
        self._phases = [
            ("approach", self.APPROACH_HEIGHT, False),
            ("descend", 0.02, False),
            ("grasp", 0.0, True),
            ("lift", self.APPROACH_HEIGHT, True),
            ("done", self.APPROACH_HEIGHT, True),
        ]
        self._i = 0
        self._timer = self.create_timer(self.PHASE_DT, self._step)

    def _step(self) -> None:
        label, dz, grasped = self._phases[self._i]
        x, y, z = self._target
        self._draw_gripper(x, y, z + dz, grasped)
        self.pub_status.publish(String(data=f"[mock arm] {label}"))
        self.get_logger().info(f"[mock arm] {label} @ z+{dz:.2f}")
        self._i += 1
        if self._i >= len(self._phases):
            self._timer.cancel()
            self._busy = False
            self.pub_status.publish(String(data="[mock arm] PICK COMPLETE"))

    def _draw_gripper(self, x, y, z, grasped) -> None:
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "mock_arm"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05
        # Green while carrying, amber while approaching.
        m.color = (ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.9) if grasped
                   else ColorRGBA(r=1.0, g=0.7, b=0.1, a=0.9))
        m.lifetime.sec = 2
        self.pub_marker.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MockArmExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
