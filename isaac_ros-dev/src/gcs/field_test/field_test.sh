#!/bin/bash
# FIELD TEST on the Jetson (props OFF, no course material): the cameras + detectors + guidance page
# http://10.42.0.247:8097 (field_vision.py in the dev container, with the RealSense node) and field_link.py on the host
# (the Pixhawk: GPS / heading in, motor cues out). The laptop's `aerothon fieldtest cable|radio` drives it; over the radio
# onboard/mav_agent.py runs this script (field codes 1 start / 2 stop).
#   field_test.sh start [field_vision args] | stop | status | log
# Not while a mission stack runs (it holds the cameras). Only one process holds the Arducam: start stops the calibration
# page (8090), the red / ArUco pages and the ROS Arducam first; stop starts the calibration page again (as at boot).
# Logs: runs/field_test/{vision,realsense,link}.log, events.log (every result / guidance event)
C=isaac_ros_dev-aarch64-pyzbar-container
WS=$HOME/workspaces/isaac_ros-dev
WSC=/workspaces/isaac_ros-dev
D=$(dirname "$(readlink -f "$0")")
S=$WS/src/launch_all/scripts
LOG=$WS/runs/field_test
CONF=$WS/runs/gcs/onboard_router.conf
VIS='field_test/field_vision.py'
mkdir -p "$LOG"

stop_all() {
    pkill -f 'gcs/field_test/field_link.py' 2>/dev/null
    P=$(pgrep -f "$VIS"); [ -n "$P" ] && kill $P
    if [ -f "$LOG/rs.pid" ]; then
        R=$(cat "$LOG/rs.pid"); kill $(pgrep -P "$R") "$R" 2>/dev/null; rm -f "$LOG/rs.pid"
    fi
    for _ in $(seq 25); do pgrep -f "$VIS" >/dev/null || break; sleep 0.2; done   # Argus is released (<= 3 s)
}

router_endpoint() {   # field_link's endpoint on the always-on router (install_onboard.sh writes it too)
    [ -f "$CONF" ] || { echo "no $CONF: run gcs/onboard/install_onboard.sh once - the motor cues need the router"; return; }
    grep -q '^\[UdpEndpoint field_test\]' "$CONF" && return
    printf '\n# gcs/field_test/field_link.py (the field test: GPS / heading in, motor cues out)\n[UdpEndpoint field_test]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14662\n' >> "$CONF"
    echo "router: field_test endpoint (UDP 14662) added - restarting aerothon-router"
    systemctl --user restart aerothon-router
}

case "${1:-start}" in
  stop)
    stop_all
    systemctl --user start arducam-calib 2>/dev/null          # back to the boot state
    echo "field test stopped" ;;
  log)
    tail -n 30 "$LOG/vision.log"; echo ---; tail -n 5 "$LOG/link.log" ;;
  status)
    curl -s -m 2 http://127.0.0.1:8097/api/info >/dev/null && echo "field test UP: http://10.42.0.247:8097" || echo "field test down"
    pgrep -fa 'field_link.py|field_vision.py|realsense2_camera_node' | cut -c1-120 ;;
  start|*)
    [ "${1:-start}" = start ] && shift
    if pgrep -f 'real_mission.launch.py|sim_mission.launch.py|container_real.launch.py|offboard_vrx.launch.py' >/dev/null; then
        echo "a mission stack is running (it holds the cameras): Stop it on the dashboard first"; exit 1
    fi
    stop_all
    router_endpoint
    "$S/arducam_calib.sh" stop >/dev/null 2>&1                # each of these holds the Arducam
    "$S/red_test.sh" stop >/dev/null 2>&1
    "$S/aruco_test.sh" stop >/dev/null 2>&1
    "$S/arducam_down.sh" >/dev/null 2>&1
    "$S/container_up.sh" || { echo "the dev container did not come up"; exit 1; }
    sleep 1.5                                                 # quick Argus reopens wedge nvargus-daemon
    ENV="source /opt/ros/humble/setup.bash; source $WSC/install/setup.bash; export FASTRTPS_DEFAULT_PROFILES_FILE=$WSC/fastdds_udp.xml; unset DISPLAY"
    # the D435 as the mission has it (container_real.launch.py): colour + depth 640x360@30, emitter on
    docker exec -d -u admin $C bash -c "$ENV; echo \$\$ > $WSC/runs/field_test/rs.pid; exec ros2 run realsense2_camera \
realsense2_camera_node --ros-args -r __node:=camera -r __ns:=/ -p enable_depth:=true -p enable_color:=true \
-p enable_infra1:=false -p enable_infra2:=false -p enable_gyro:=false -p enable_accel:=false \
-p depth_module.emitter_enabled:=1 -p depth_module.depth_profile:=640x360x30 -p rgb_camera.color_profile:=640x360x30 \
-p align_depth.enable:=false -p pointcloud.enable:=false > $WSC/runs/field_test/realsense.log 2>&1"
    docker exec -d -u admin $C bash -c "$ENV; cd $WSC; exec python3 $WSC/src/gcs/$VIS $* > $WSC/runs/field_test/vision.log 2>&1"
    setsid nohup python3 "$D/field_link.py" > "$LOG/link.log" 2>&1 < /dev/null &
    for _ in $(seq 40); do curl -s -m 1 http://127.0.0.1:8097/api/info >/dev/null && break; sleep 1; done
    if curl -s -m 1 http://127.0.0.1:8097/api/info >/dev/null; then
        echo "field test UP: http://10.42.0.247:8097 (logs $LOG)"
    else
        echo "field test did not come up - $LOG/vision.log:"; tail -n 15 "$LOG/vision.log"; exit 1
    fi ;;
esac
