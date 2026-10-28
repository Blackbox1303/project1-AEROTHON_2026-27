#!/bin/bash
# container_real.sh <run dir in the container> <geofence json> [arducam calibration yaml] [course] [corridor current|aerothon]: INSIDE the container -
# the REAL drone's mission stack (aerothon_nav container_real.launch.py) + loggers of /mission/log and
# /nav/altitude/state, all into the run dir. Started by real_mission.launch.py through docker exec. No auto-start:
# the mission is started from the ground station after the preflight.
RC=$1; GJ=$2; CAL=${3:-}; COURSE=${4:-v2}
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_udp.xml
unset DISPLAY                                                    # Argus (the Arducam) fails with a DISPLAY it can't use
# the mission code version chosen on the dashboard (code_version.py): its aerothon_nav first on the import path
[ -n "$AEROTHON_CODE_PATH" ] && export PYTHONPATH=$AEROTHON_CODE_PATH:$PYTHONPATH
mkdir -p "$RC"; rm -f ~/.ros/aerothon_*
echo "mission code: ${AEROTHON_CODE_PATH:-the working tree (current)}" > "$RC/code_version.txt"                           # a clean mission: no stored QR / map / geofence
# an empty 'calibration:=' is not a valid launch argument: only with a calibration file
ros2 launch aerothon_nav container_real.launch.py world_json:=$GJ course:=$COURSE corridor_impl:=${5:-current} \
    ${CAL:+calibration:=$CAL} \
    > "$RC/container.log" 2>&1 &
ros2 topic echo --full-length /mission/log std_msgs/msg/String > "$RC/mission_log.txt" 2>&1 &
ros2 topic echo --full-length /nav/altitude/state std_msgs/msg/String > "$RC/altitude_state.txt" 2>&1 &
wait
