#!/usr/bin/env bash
# FoGaze blink → grasp: run the robot-arm pipeline AND the real app together,
# in ONE command.  Triple-blink a graspable (green) object in the app to pick
# it; the arm plans + moves in RViz.
#
# Usage:
#   ./fogaze_arm.sh                    # Tier 2 (MoveIt Panda + RViz) + app
#   ./fogaze_arm.sh mock               # Tier 1 (mock arm marker)        + app
#   ./fogaze_arm.sh --sim              # Tier 2 + app in SIM mode (no depth cam)
#   ./fogaze_arm.sh mock --sim         # Tier 1 + app in SIM mode
#   ./fogaze_arm.sh moveit --sim --stereo ...   # any extra flags pass to main.py
#
# The arm launch and the app share this shell's env, so they land on the SAME
# ROS domain and talk automatically.  Ctrl+C stops both.
set -e
cd "$(dirname "$0")"

# First arg may be a mode (moveit|mock); anything else is passed to main.py.
MODE="moveit"
case "$1" in
    moveit|mock) MODE="$1"; shift ;;
esac
APP_ARGS=("$@")          # e.g. --sim, --stereo, --blink-count 3 ...

source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

# ── De-snap the environment so RViz (Qt/OGRE) can start ──────────────────
# When this is launched from the VS Code *snap* terminal, the snap rewrites
# GTK_EXE_PREFIX / XDG_* / GTK_* to point inside /snap/code, which makes
# rviz2 load the snap's old libpthread and die with
# "symbol lookup error: ... undefined symbol: __libc_pthread_init".
# VS Code stashes each original value in <VAR>_VSCODE_SNAP_ORIG, so restore
# those, drop the snap-only GUI vars, and strip /snap/ from the linker path.
# All no-ops when not running under a snap.
for _o in $(compgen -v | grep '_VSCODE_SNAP_ORIG$' || true); do
    _base=${_o%_VSCODE_SNAP_ORIG}
    if [ -n "${!_o}" ]; then export "$_base=${!_o}"; else unset "$_base"; fi
    unset "$_o"
done
unset GTK_EXE_PREFIX GTK_PATH GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR \
      GTK_IM_MODULE_FILE LOCPATH GCONV_PATH 2>/dev/null || true
if [[ ":${LD_LIBRARY_PATH:-}:" == *"/snap/"* ]]; then
    LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -v '/snap/' | paste -sd ':')
    export LD_LIBRARY_PATH
fi

if [ "$MODE" = "mock" ]; then
    LAUNCH="pickup_mock.launch.py";   WARMUP=3
else
    LAUNCH="pickup_moveit.launch.py"; WARMUP=8   # move_group + RViz + controllers
fi

echo "[FoGaze] (1/2) launching arm pipeline: $MODE ..."
ros2 launch fogaze_manip "$LAUNCH" &
PID_ARM=$!
trap 'echo; echo "[FoGaze] stopping..."; kill $PID_ARM 2>/dev/null; exit' INT TERM

# Wait for move_group + controllers (or the mock arm) to advertise topics.
sleep "$WARMUP"

echo "[FoGaze] (2/2) starting app (--ros ${APP_ARGS[*]})."
echo "         Triple-blink a GREEN object to pick. Status:"
echo "           ros2 topic echo /fogaze/pickup_status"
python3 main.py --ros "${APP_ARGS[@]}"

# App closed → tear down the arm pipeline too.
kill "$PID_ARM" 2>/dev/null || true
