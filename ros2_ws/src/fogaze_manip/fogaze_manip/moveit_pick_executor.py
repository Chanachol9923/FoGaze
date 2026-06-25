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

import json
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
    # Panda "ready" pose — the arm returns here after every pick so it always
    # waits for the next command from the same neutral configuration.
    READY_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    def __init__(self):
        super().__init__("moveit_pick_executor")
        self.declare_parameter("synchronous", True)
        # Group / EE names for the *moveit_resources* Panda config.  pymoveit2's
        # bundled panda preset assumes "arm"/"gripper"/"panda_hand_tcp", which do
        # NOT exist in moveit_resources_panda_moveit_config (groups are
        # "panda_arm"/"hand", and there is no _tcp frame) — leaving the defaults
        # makes every plan fail with "Cannot find planning configuration".
        self.declare_parameter("arm_group", "panda_arm")
        self.declare_parameter("gripper_group", "hand")
        self.declare_parameter("end_effector", "panda_hand")
        self._busy = False
        self._status = self.create_publisher(String, "fogaze/pickup_status", 1)

        if not _HAVE_PYMOVEIT2:
            self.get_logger().error(
                "pymoveit2 not found — install it (see module docstring). "
                "Node will idle so the launch stays up.")
            self.create_subscription(
                PoseStamped, "fogaze/pick_pose", self._warn_missing, 1)
            return

        arm_group = self.get_parameter("arm_group").value
        gripper_group = self.get_parameter("gripper_group").value
        end_effector = self.get_parameter("end_effector").value
        cb = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=end_effector,
            group_name=arm_group,
            callback_group=cb,
        )
        self.gripper = MoveIt2Gripper(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=gripper_group,
            callback_group=cb,
        )
        self.create_subscription(
            PoseStamped, "fogaze/pick_pose", self._on_goal, 1)
        # Listen to the raw pick request too, to learn the target's real size
        # ([w, h] m) so the gripper closes on the true width and we attach a
        # matching cylinder — no custom message needed.
        self._target_wh = None
        self.create_subscription(
            String, "fogaze/pickup", self._on_pickup, 1)
        # Gripper / object link names for the moveit_resources Panda.
        self.declare_parameter("finger_max_m", 0.04)   # per-finger travel
        self.declare_parameter("hand_link", "panda_hand")
        self.declare_parameter(
            "finger_links", ["panda_leftfinger", "panda_rightfinger"])
        self._finger_max = self.get_parameter("finger_max_m").value
        self._hand_link = self.get_parameter("hand_link").value
        self._finger_links = list(self.get_parameter("finger_links").value)
        self.get_logger().info(
            "moveit_pick_executor ready — listening on fogaze/pick_pose")

    def _on_pickup(self, msg: String) -> None:
        try:
            tgt = (json.loads(msg.data) or {}).get("target") or {}
            self._target_wh = tgt.get("size")
        except (ValueError, TypeError):
            self._target_wh = None

    # ───────────────────────────────────────────────────────────────────
    def _warn_missing(self, _ps):
        self._say("REJECT — pymoveit2 not installed (cannot drive arm)")

    def _say(self, text):
        self._status.publish(String(data=text))
        self.get_logger().info(text)

    def _on_goal(self, ps: PoseStamped) -> None:
        # Reject overlapping commands: one pick→place→reset cycle must finish
        # before another is accepted.  Set _busy synchronously here (not in the
        # worker thread) so two goals arriving back-to-back can't both start.
        if self._busy:
            self._say("BUSY — finish current pick first")
            return
        self._busy = True
        # Run the (blocking) motion sequence off the executor thread.
        Thread(target=self._execute_pick, args=(ps,), daemon=True).start()

    GRASP_ID = "grasped_object"

    def _execute_pick(self, ps: PoseStamped) -> None:
        self._busy = True                          # already set in _on_goal
        wh = self._target_wh                       # snapshot the object size
        try:
            x, y, z = (ps.pose.position.x, ps.pose.position.y,
                       ps.pose.position.z)
            self.moveit2.max_velocity = 0.3
            self.moveit2.max_acceleration = 0.3

            # Clear any object held from a previous pick.
            self.moveit2.detach_collision_object(self.GRASP_ID)
            self.moveit2.remove_collision_object(self.GRASP_ID)

            self._say(f"PICK approach -> [{x:.2f}, {y:.2f}, {z:.2f}]")
            self.gripper.open()
            self.gripper.wait_until_executed()
            self._move_to(x, y, z + self.APPROACH_HEIGHT)

            self._say("PICK descend")
            self._move_to(x, y, z)

            # Close the gripper onto the object's *real* width: each finger
            # travels to half the diameter (clamped to its limit).  Falls back
            # to a full close when the size is unknown.
            w = wh[0] if wh and len(wh) == 2 else None
            self._say("PICK grasp")
            if w is not None:
                finger = max(0.0, min(w / 2.0, self._finger_max))
                self.gripper.move_to_position(finger)
            else:
                self.gripper.close()
            self.gripper.wait_until_executed()

            # Attach a cylinder of the object's size so it is genuinely
            # "picked up" — it now moves/lifts with the gripper in the scene.
            if wh and len(wh) == 2:
                self.moveit2.add_collision_cylinder(
                    id=self.GRASP_ID,
                    height=max(wh[1], 0.02),
                    radius=max(wh[0], 0.02) / 2.0,
                    position=[x, y, z],
                    quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                    frame_id=ps.header.frame_id or "panda_link0",
                )
                self.moveit2.attach_collision_object(
                    id=self.GRASP_ID, link_name=self._hand_link,
                    touch_links=self._finger_links + [self._hand_link])

            self._say("PICK lift")
            self._move_to(x, y, z + self.APPROACH_HEIGHT)

            # ── Place it back where it came from ────────────────────────
            # The demo returns every object so the scene resets for the next
            # command.  Descend to the original spot, release, and let go of
            # the attached cylinder.
            self._say("PLACE descend")
            self._move_to(x, y, z)
            self._say("PLACE release")
            self.gripper.open()
            self.gripper.wait_until_executed()
            self.moveit2.detach_collision_object(self.GRASP_ID)
            self.moveit2.remove_collision_object(self.GRASP_ID)
            self._say("PLACE retreat")
            self._move_to(x, y, z + self.APPROACH_HEIGHT)

            # ── Reset to the ready pose to await the next command ───────
            self._say("RESET home")
            self.moveit2.move_to_configuration(self.READY_JOINTS)
            self.moveit2.wait_until_executed()

            # Emit COMPLETE last: scene_publisher keeps the scene paused until
            # this, so it resumes mirroring only once the arm is home again.
            self._say("PICK COMPLETE")
        except Exception as e:  # noqa: BLE001 — surface any planning failure
            # Drop any held object so a mid-sequence failure leaves no ghost.
            try:
                self.moveit2.detach_collision_object(self.GRASP_ID)
                self.moveit2.remove_collision_object(self.GRASP_ID)
            except Exception:
                pass
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
