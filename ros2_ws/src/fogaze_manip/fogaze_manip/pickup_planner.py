"""pickup_planner — decide-and-dispatch node.

Pipeline position::

    main.py --ros  --(fogaze/pickup, JSON)-->  [pickup_planner]
                                                   |  re-validate graspability
                                                   |  TF: camera -> arm base
                                                   v
                                       fogaze/pick_pose (PoseStamped)  --> executor
                                       fogaze/pickup_status (String)

The planner does **not** trust the app's graspable flag blindly: it re-runs
``classify_graspable`` with the arm's own configured limits, transforms the
target into the arm base frame, and only then emits a pick pose for an
executor (mock or MoveIt) to act on.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped do_transform)

from ._fogaze_path import ensure_on_path

ensure_on_path()
from modules.grasp import GraspParams, classify_graspable  # noqa: E402


class PickupPlanner(Node):
    def __init__(self):
        super().__init__("pickup_planner")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_base_frame", "panda_link0")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("min_reach_m", GraspParams.min_reach_m)
        self.declare_parameter("max_reach_m", GraspParams.max_reach_m)
        self.declare_parameter("gripper_max_m", GraspParams.gripper_max_m)
        self.declare_parameter("min_size_m", GraspParams.min_size_m)

        self.arm_base_frame = self.get_parameter("arm_base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.use_tf = self.get_parameter("use_tf").value
        self.params = GraspParams(
            min_reach_m=self.get_parameter("min_reach_m").value,
            max_reach_m=self.get_parameter("max_reach_m").value,
            gripper_max_m=self.get_parameter("gripper_max_m").value,
            min_size_m=self.get_parameter("min_size_m").value,
        )

        # ── TF ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── I/O ────────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            String, "fogaze/pickup", self._on_pickup, 1)
        self.pub_pose = self.create_publisher(PoseStamped, "fogaze/pick_pose", 1)
        self.pub_status = self.create_publisher(String, "fogaze/pickup_status", 1)

        self.get_logger().info(
            f"pickup_planner ready — arm_base='{self.arm_base_frame}', "
            f"camera='{self.camera_frame}', use_tf={self.use_tf}, "
            f"reach=[{self.params.min_reach_m},{self.params.max_reach_m}]m, "
            f"gripper_max={self.params.gripper_max_m}m")

    # ───────────────────────────────────────────────────────────────────
    def _status(self, text: str) -> None:
        self.pub_status.publish(String(data=text))
        self.get_logger().info(text)

    def _on_pickup(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"bad pickup JSON: {e}")
            return

        target = payload.get("target")
        frame = payload.get("frame_id", self.camera_frame)
        if not target:
            self._status("REJECT — empty pickup request")
            return

        cls = target.get("class", "object")
        pose = target.get("pose")      # [x, y, z] metres, camera optical frame
        size = target.get("size")      # [w, h] metres
        if pose is None:
            self._status(f"REJECT {cls} — no 3D pose (no depth)")
            return

        z_m = pose[2]
        size_wh = tuple(size) if size else None
        graspable, reason = classify_graspable(z_m, size_wh, self.params)
        if not graspable:
            self._status(f"REJECT {cls} — {reason} "
                         f"(z={z_m:.2f}m, size={size_wh})")
            return

        out = self._to_arm_frame(pose, frame)
        if out is None:
            self._status(f"REJECT {cls} — no TF {frame}->{self.arm_base_frame}")
            return

        ps = PoseStamped()
        ps.header.frame_id = out[3]
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = out[:3]
        # Top-down grasp orientation (identity here; the executor refines it).
        ps.pose.orientation.w = 1.0
        self.pub_pose.publish(ps)
        self._status(
            f"PICK {cls} -> [{out[0]:.2f}, {out[1]:.2f}, {out[2]:.2f}] "
            f"in {out[3]}")

    def _to_arm_frame(self, pose, src_frame):
        """Transform a camera-frame point to the arm base frame.

        Returns (x, y, z, frame_id) or ``None`` if TF is unavailable.
        Falls back to the original frame (with a warning) when ``use_tf`` is
        off, so the mock executor can still demo without a calibrated robot.
        """
        if not self.use_tf:
            return (pose[0], pose[1], pose[2], src_frame)
        pt = PointStamped()
        pt.header.frame_id = src_frame
        pt.point.x, pt.point.y, pt.point.z = pose
        try:
            out = self.tf_buffer.transform(
                pt, self.arm_base_frame,
                timeout=rclpy.duration.Duration(seconds=0.3))
            return (out.point.x, out.point.y, out.point.z, self.arm_base_frame)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PickupPlanner()
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
