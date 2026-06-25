"""scene_publisher — mirror YOLO detections into the MoveIt planning scene.

Pipeline position::

    main.py --ros  --(fogaze/objects, JSON)-->  [scene_publisher]
                                                     |  TF: camera -> arm base
                                                     v
                                      /collision_object (moveit_msgs/CollisionObject)
                                                     |
                                                     v
                                      move_group PlanningSceneMonitor + RViz

Every object the app sees (class, real-world 3D ``pose`` from depth, real-world
``size`` from the bbox) becomes an upright cylinder (height ~ bbox height, radius
~ half bbox width) in the arm base frame, so the detected scene "appears" in
simulation at its true position/distance and the arm plans around it.  Objects
that vanish between frames are REMOVEd so the scene tracks reality instead of
accumulating ghosts.

The camera->arm TF comes from the same (currently placeholder) eye-to-hand
transform used by ``pickup_planner`` — until that is calibrated, boxes land at
the right *relative* spot but the wrong absolute one.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped do_transform)


class ScenePublisher(Node):
    def __init__(self):
        super().__init__("scene_publisher")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("arm_base_frame", "panda_link0")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("use_tf", True)
        # Object footprint depth (the bbox gives width/height only).
        self.declare_parameter("default_depth_m", 0.05)
        # Clamp tiny/degenerate boxes — MoveIt rejects dims <= 0.
        self.declare_parameter("min_dim_m", 0.02)
        # Throttle planning-scene updates (the app publishes per video frame).
        self.declare_parameter("update_hz", 4.0)
        # Only mirror objects that have a real depth-derived pose.
        self.declare_parameter("id_prefix", "yolo_")
        # Static support surface (a "table") so the scene looks like a tabletop
        # and the arm plans ABOVE a real plane instead of through empty space.
        # Pose/size are in the arm base frame; defaults sit just under where the
        # camera->arm TF places typical 0.4-0.6 m detections.  Set
        # support_surface:=false to drop it (e.g. a wall-mounted arm).
        self.declare_parameter("support_surface", True)
        self.declare_parameter("table_id", "table_surface")
        self.declare_parameter("table_size", [0.5, 1.0, 0.3])     # x, y, z (m)
        self.declare_parameter("table_center", [0.55, 0.0, 0.0])  # x, y, z (m)

        self.arm_base_frame = self.get_parameter("arm_base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.use_tf = self.get_parameter("use_tf").value
        self.default_depth_m = self.get_parameter("default_depth_m").value
        self.min_dim_m = self.get_parameter("min_dim_m").value
        self.id_prefix = self.get_parameter("id_prefix").value
        self.support_surface = self.get_parameter("support_surface").value
        self.table_id = self.get_parameter("table_id").value
        self.table_size = list(self.get_parameter("table_size").value)
        self.table_center = list(self.get_parameter("table_center").value)
        period = 1.0 / max(self.get_parameter("update_hz").value, 0.1)

        # ── TF ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── I/O ────────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            String, "fogaze/objects", self._on_objects, 1)
        # While a pick is executing, clear the scene so the target object's box
        # doesn't block the gripper; resume mirroring once it finishes.
        self.sub_status = self.create_subscription(
            String, "fogaze/pickup_status", self._on_status, 10)
        self.pub_co = self.create_publisher(CollisionObject, "collision_object", 10)

        # Latest objects payload + throttle timer (decouple from frame rate).
        self._latest = None        # (objects, frame_id)
        self._prev_ids = set()     # ids published last tick (for REMOVE)
        self._picking = False      # paused while an arm pick is in progress
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"scene_publisher ready — mirroring fogaze/objects into "
            f"'{self.arm_base_frame}' at {self.get_parameter('update_hz').value}Hz "
            f"(camera='{self.camera_frame}', use_tf={self.use_tf})")

    # ───────────────────────────────────────────────────────────────────
    def _on_objects(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"bad objects JSON: {e}")
            return
        self._latest = (payload.get("objects") or [],
                        payload.get("frame_id", self.camera_frame))

    def _on_status(self, msg: String) -> None:
        s = msg.data
        # "PICK approach ..." starts a grasp; COMPLETE/FAILED/REJECT ends it.
        if s.startswith("PICK approach"):
            if not self._picking:
                self._picking = True
                for oid in self._prev_ids:   # clear so the gripper is unobstructed
                    self._remove(oid)
                self._prev_ids = set()
        elif s.startswith(("PICK COMPLETE", "PICK FAILED", "REJECT")):
            self._picking = False

    def _tick(self) -> None:
        # The support surface is part of the world, not a detection — keep it
        # present even while a pick runs (the arm must always avoid the table).
        if self.support_surface:
            self._publish_table()

        if self._latest is None or self._picking:
            return
        objects, frame = self._latest

        cur_ids = set()
        for i, obj in enumerate(objects):
            pose = obj.get("pose")        # [x, y, z] m, camera optical frame
            size = obj.get("size")        # [w, h] m
            if pose is None:
                continue                  # no depth -> can't place it
            xyz = self._to_arm_frame(pose, frame)
            if xyz is None:
                continue                  # TF not ready yet
            oid = f"{self.id_prefix}{i}"
            self._publish_shape(oid, xyz, size, obj.get("class", "object"))
            cur_ids.add(oid)

        # Remove shapes that were present last tick but are gone now.
        for stale in self._prev_ids - cur_ids:
            self._remove(stale)
        self._prev_ids = cur_ids

    # ───────────────────────────────────────────────────────────────────
    def _publish_table(self) -> None:
        """(Re)publish the static support surface as a BOX collision object."""
        from geometry_msgs.msg import Pose

        co = CollisionObject()
        co.header.frame_id = self.arm_base_frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = self.table_id

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [max(d, self.min_dim_m) for d in self.table_size]

        p = Pose()
        p.position.x, p.position.y, p.position.z = self.table_center
        p.orientation.w = 1.0

        co.primitives = [prim]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD     # ADD is idempotent (updates id)
        self.pub_co.publish(co)

    def _publish_shape(self, oid, xyz, size, cls) -> None:
        w, h = (size if size and len(size) == 2 else (0.05, 0.05))
        # Size proportional to the real object: height = bbox height, the
        # cylinder diameter = bbox width (so the gripper closes on the true
        # width).  Clamp only against degenerate/zero dims.
        height = max(h, self.min_dim_m)             # vertical extent ~ height
        radius = max(w, self.min_dim_m) / 2.0       # diameter ~ real width

        co = CollisionObject()
        co.header.frame_id = self.arm_base_frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = oid

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER       # upright cylinder (axis = +z)
        prim.dimensions = [height, radius]        # CYLINDER: [height, radius]

        from geometry_msgs.msg import Pose
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        p.orientation.w = 1.0          # axis-aligned (no orientation from a bbox)

        co.primitives = [prim]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD     # ADD also updates an existing id
        self.pub_co.publish(co)

    def _remove(self, oid) -> None:
        co = CollisionObject()
        co.header.frame_id = self.arm_base_frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = oid
        co.operation = CollisionObject.REMOVE
        self.pub_co.publish(co)

    def _to_arm_frame(self, pose, src_frame):
        """camera-frame point -> arm base frame, or None if TF unavailable."""
        if not self.use_tf:
            return (pose[0], pose[1], pose[2])
        pt = PointStamped()
        pt.header.frame_id = src_frame
        pt.point.x, pt.point.y, pt.point.z = pose
        try:
            out = self.tf_buffer.transform(
                pt, self.arm_base_frame,
                timeout=rclpy.duration.Duration(seconds=0.2))
            return (out.point.x, out.point.y, out.point.z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=5.0)
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ScenePublisher()
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
