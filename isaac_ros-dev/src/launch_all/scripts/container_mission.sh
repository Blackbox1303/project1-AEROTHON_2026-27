#!/bin/bash
# container_mission.sh <run dir in the container> <world name> [autostart 1|0] [geofence json] [ardupilot|px4]
#   [corridor current|aerothon]: INSIDE the container - the mission stack
# (container_all) + loggers of /mission/log and /nav/altitude/state, all into the run dir; then /mission/start once
# the delivery Nav2 is up (autostart 1). Started by sim_mission.launch.py through docker exec.
RC=$1; W=$2; AUTO=${3:-1}; GJ=${4:-/workspaces/isaac_ros-dev/sim/worlds/$W.json}
export AEROTHON_AUTOPILOT=${5:-ardupilot}       # offboard_bridge / mission_supervisor: flight modes of that autopilot
source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_udp.xml
# the mission code version chosen on the dashboard (code_version.py): its aerothon_nav first on the import path
[ -n "$AEROTHON_CODE_PATH" ] && export PYTHONPATH=$AEROTHON_CODE_PATH:$PYTHONPATH
mkdir -p "$RC"; rm -f ~/.ros/aerothon_*
echo "mission code: ${AEROTHON_CODE_PATH:-the working tree (current)}" > "$RC/code_version.txt"                           # a clean mission: no stored QR / map / geofence
ros2 launch aerothon_nav container_all.launch.py vslam:=false course:=v2_slowsim \
    world_json:=$GJ corridor_impl:=${6:-current} > "$RC/container.log" 2>&1 &
ros2 topic echo --full-length /mission/log std_msgs/msg/String > "$RC/mission_log.txt" 2>&1 &
ros2 topic echo --full-length /nav/altitude/state std_msgs/msg/String > "$RC/altitude_state.txt" 2>&1 &
if [ "$AUTO" = 1 ]; then
    for _ in $(seq 180); do grep -q 'delivery.lifecycle_manager_navigation.*Managed nodes are active' "$RC/container.log" && break; sleep 2; done
    sleep 8
    # retried: the supervisor refuses until it has /nav/odom and a MAVROS connection (ArduCopter sim, 2026-10-27: the
    # one call came first, success=False, and the mission never started). The response line, not tail -1's blank line
    for _ in $(seq 20); do
        r=$(timeout 60 ros2 service call /mission/start std_srvs/srv/Trigger | grep -o 'Trigger_Response(.*)')
        echo "$(date +%T) $r" >> "$RC/mission_start.txt"
        case "$r" in *success=True*) break ;; esac
        sleep 10
    done
fi
wait
