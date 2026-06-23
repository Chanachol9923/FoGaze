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

### ⚠ Eye-to-hand calibration (required for a real pick)
`pickup_moveit.launch.py` publishes a **placeholder** static transform
`panda_link0 → camera_color_optical_frame`. Replace the translation/rotation
with your measured camera-to-arm pose, or the arm will reach the wrong place.

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
| `fogaze/objects` | std_msgs/String (JSON) | app | all objects/frame (debug) |
| `fogaze/pickup` | std_msgs/String (JSON) | app | focused object to pick |
| `fogaze/pick_pose` | geometry_msgs/PoseStamped | planner | target in arm frame |
| `fogaze/pickup_status` | std_msgs/String | planner/exec | human-readable status |
| `fogaze/arm_marker` | visualization_msgs/Marker | mock exec | gripper viz |
