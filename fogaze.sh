#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[FoGaze] Starting main app + Rviz..."
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

python3 main.py &
PID_MAIN=$!

# Let main.py init ROS + publish static TF first
sleep 1

LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 rviz2 -d ros2_ws/install/fogaze_rviz/share/fogaze_rviz/config/fogaze.rviz &
PID_RVIZ=$!

trap "kill $PID_MAIN $PID_RVIZ 2>/dev/null; exit" INT TERM

echo ""
echo "=== Rviz เปิดแล้ว =="
echo "  ถ้าไม่มีอะไรขึ้นมา ให้เพิ่ม MarkerArray display ด้วยตนเอง:"
echo "  1. คลิก 'Add' ปุ่มล่างซ้าย"
echo "  2. เลือก 'MarkerArray'"
echo "  3. ตั้ง Marker Topic = /detections/markers"
echo "  4. เช็กที่ Fixed Frame (Global Options) ว่าเป็น 'map'"
echo "================================"
echo ""

wait
