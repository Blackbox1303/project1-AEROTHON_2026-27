#!/bin/bash
# The AeroTHON ground station dashboard on the Jetson: http://10.42.0.247:8080 (open it on the Lenovo).
# Launch the sim / drone link from the dashboard (or sim_mission.launch.py / real_link.launch.py by hand).
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/workspaces/isaac_ros-dev/fastdds_udp.xml
exec python3 "$(dirname "$(readlink -f "$0")")/gcs_server.py" "$@"
