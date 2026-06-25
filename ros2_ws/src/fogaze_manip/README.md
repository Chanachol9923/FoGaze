# fogaze_manip — gaze → blink → grasp

Turns a **triple-blink** pick request from the FoGaze app into a robot-arm
grasp in MoveIt 2 / Gazebo. Objects are classified **graspable / not** from
their **distance** and **real-world size** (from the PrimeSense depth camera).

```
main.py --ros ──fogaze/pickup (JSON)──▶ pickup_planner ──fogaze/pick_pose──▶ executor ──▶ arm
   │ gaze + YOLO + depth + 3-blink         │ re-check graspability             (mock | moveit)
   └ fogaze/objects (per-frame, debug)     └ TF: camera → arm base
```

Shared geometry + graspability rules live in `modules/grasp.py` (ROS-free), so
the app and the ROS nodes agree on a single definition. See
`test/test_graspability.py`.

## Graspability rule (`modules/grasp.py`)
An object is graspable when **all** hold (defaults = Franka Panda hand):
- `0.20 m ≤ distance ≤ 0.85 m`  (within reach)
- `min(width, height) ≤ 0.08 m` (fits the gripper)
- `max(width, height) ≥ 0.02 m` (not detection noise)

Tune in `config/graspability.yaml`.

---

## Tier 1 — works after a plain build (no Gazebo/MoveIt)
Demos the **entire** pipeline with a mock arm (animated marker in RViz).

```bash
cd ros2_ws
colcon build --packages-select fogaze_manip
source install/setup.bash

# terminal 1 — planner + mock arm + a placeholder camera TF
ros2 launch fogaze_manip pickup_mock.launch.py

# terminal 2 — the FoGaze app, publishing to ROS
python3 main.py --ros
```
Triple-blink at an object: boxes are **green** (graspable) / **red**
(not) / **cyan** (no depth) in the app. Watch `fogaze/pickup_status` and the
`fogaze/arm_marker` move in RViz.

```bash
ros2 topic echo /fogaze/pickup_status
```

---

## Tier 2 — simulated Panda arm via MoveIt 2
One-time install (**needs sudo**):
```bash
sudo apt update
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-moveit-resources-panda-moveit-config \
  ros-humble-ros2-control ros-humble-ros2-controllers \  # controller_manager for the Panda demo
  ros-humble-ros-gz            # only for full Gazebo physics

# Python MoveGroup interface used by moveit_pick_executor:
cd ros2_ws/src
git clone https://github.com/AndrejOrsula/pymoveit2.git
cd .. && colcon build && source install/setup.bash
```
Run:
```bash
ros2 launch fogaze_manip pickup_moveit.launch.py   # Panda + MoveIt + RViz
python3 main.py --ros
```
This brings up `move_group` + RViz + simulated controllers, **mirrors every YOLO
detection into the MoveIt planning scene** at its real depth-derived position
(`scene_publisher`), and runs the pick pipeline. The arm really plans + moves;
`scene_publisher` auto-clears the scene while a grasp executes so the target box
doesn't block the gripper, then restores it.

> **MoveIt group names:** this stack targets the `moveit_resources` Panda
> (groups `panda_arm` / `hand`, EE `panda_hand`). pymoveit2's bundled panda
> preset instead assumes `arm`/`gripper`/`panda_hand_tcp`, which don't exist
> here and make every plan fail — `moveit_pick_executor` overrides them via the
> `arm_group` / `gripper_group` / `end_effector` parameters.

### ⚠ Eye-to-hand calibration
`pickup_moveit.launch.py` publishes a **sim-default** static transform
`panda_link0 → camera_color_optical_frame` — camera 0.4 m above the base looking
straight forward, upright (REP-103 optical, quat `-0.5, 0.5, -0.5, 0.5`). This
puts objects detected at 0.3–0.7 m inside the Panda's reach. For a **real**
camera+arm rig, replace the 7 numbers with a measured / hand-eye calibration or
the arm reaches the wrong absolute spot.

### Full Gazebo physics (optional)
`worlds/fogaze_table.sdf` has a table + a can (graspable), a pen (too thin),
and a 30 cm box (too wide) to exercise the classifier:
```bash
ign gazebo src/fogaze_manip/worlds/fogaze_table.sdf
```
Spawn the Panda + `ros2_control` into this world and point MoveIt at it to run
the pick under physics.

---

## Topics
| topic | type | by | meaning |
|---|---|---|---|
| `fogaze/objects` | std_msgs/String (JSON) | app | all objects/frame (debug + scene) |
| `fogaze/pickup` | std_msgs/String (JSON) | app | focused object to pick |
| `fogaze/pick_pose` | geometry_msgs/PoseStamped | planner | target in arm frame |
| `fogaze/pickup_status` | std_msgs/String | planner/exec | human-readable status |
| `fogaze/arm_marker` | visualization_msgs/Marker | mock exec | gripper viz |
| `collision_object` | moveit_msgs/CollisionObject | scene_publisher | YOLO objects in the MoveIt scene |
