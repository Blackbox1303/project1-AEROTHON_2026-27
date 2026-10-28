#!/bin/bash
# ArUco marker test page (http://10.42.0.247:8085) in the dev container, opening the Arducam on CSI CAM0 itself.
# Only ONE process can hold the camera: start stops the calibration page (8090) and the ROS camera first.
#   aruco_test.sh [start [args]] | stop | log     (args go to aruco_test.py, e.g. --marker-size 0.8 --width 3280 --height 2464)
# Log: runs/aruco_test/aruco.log
C=isaac_ros_dev-aarch64-pyzbar-container
PAT='^python3 /workspaces/isaac_ros-dev/src/vio_bringup/calibration/aruco_test.py'
D=$(dirname "$0")
case "${1:-start}" in
  stop)
    P=$(pgrep -f "$PAT"); [ -n "$P" ] && kill $P
    for _ in $(seq 25); do pgrep -f "$PAT" >/dev/null || break; sleep 0.2; done
    echo stopped ;;
  log)
    tail -n 40 ~/workspaces/isaac_ros-dev/runs/aruco_test/aruco.log ;;
  *)
    [ "${1:-start}" = start ] && shift
    "$D/arducam_calib.sh" stop >/dev/null                     # 8090 holds the camera
    "$D/arducam_down.sh" >/dev/null                           # so does the ROS camera
    P=$(pgrep -f "$PAT"); [ -n "$P" ] && kill $P
    for _ in $(seq 25); do pgrep -f "$PAT" >/dev/null || break; sleep 0.2; done   # Argus is released (<= 3 s)
    sleep 1.5                                                 # quick Argus reopens wedge nvargus-daemon
    mkdir -p ~/workspaces/isaac_ros-dev/runs/aruco_test
    docker exec -d -u admin $C bash -c "unset DISPLAY; cd /workspaces/isaac_ros-dev; exec python3 \
/workspaces/isaac_ros-dev/src/vio_bringup/calibration/aruco_test.py $* >> /workspaces/isaac_ros-dev/runs/aruco_test/aruco.log 2>&1"
    echo "ArUco page: http://10.42.0.247:8085 (log: $0 log)" ;;
esac
