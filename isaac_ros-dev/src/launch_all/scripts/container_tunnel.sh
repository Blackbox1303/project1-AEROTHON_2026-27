#!/bin/bash
# container_tunnel.sh <run dir in container> [world_json] [course] [vslam] [scan_band]: INSIDE the container -
# the REAL RealSense D435 tunnel + SLAM stack (tunnel_real.launch.py) + loggers into the run dir.
RC=$1; GJ=${2:-}; COURSE=${3:-v2}; VSLAM=${4:-false}; BAND=${5:-0.20}
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_udp.xml
unset DISPLAY
mkdir -p "$RC"
ros2 launch /workspaces/isaac_ros-dev/src/aerothon_nav/launch/tunnel_real.launch.py \
    course:=$COURSE vslam:=$VSLAM scan_band:=$BAND ${GJ:+world_json:=$GJ} \
    > "$RC/container.log" 2>&1 &
ros2 topic echo --full-length /nav/corridor/state std_msgs/msg/String > "$RC/corridor_state.txt" 2>&1 &
ros2 topic echo --full-length /nav/altitude/state std_msgs/msg/String > "$RC/altitude_state.txt" 2>&1 &
wait
