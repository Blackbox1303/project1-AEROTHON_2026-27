#!/bin/bash
# Start ONLY the Tunnel + SLAM web dashboard on port 8093 (does NOT auto-start at boot).
# Open http://10.42.0.247:8093/ (from the Lenovo) or http://127.0.0.1:8093/ (on the Jetson).
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/workspaces/isaac_ros-dev/fastdds_udp.xml
exec python3 "$(dirname "$(readlink -f "$0")")/tunnel_gcs.py" "$@"
