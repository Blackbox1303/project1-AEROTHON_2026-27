#!/bin/bash
# The ArUco delivery mission dashboard: http://10.42.0.247:8084 - the 8080 ground station with --marker aruco (sim /
# offboard VRX / onboard radio, the same segments as 8080: whole mission, QR area, tunnels, QR area + drop; boards carry ArUco markers 3x3 .. 6x6). MAVLink from the routers'
# gcs_aruco sink (udpin :14651). Offboard (Jetson on the ground, VTX -> VRX video, radio): search at 1 m.
# Runs beside 8080; one stack at a time (either page sees and can stop it).
#   gcs_aruco.sh [start [extra gcs_server args, e.g. --radio /dev/ttyUSB0:57600]] | stop | log
# Log: runs/gcs/gcs_aruco.log
PAT='gcs_server.py --port 8084'
LOG=~/workspaces/isaac_ros-dev/runs/gcs/gcs_aruco.log
case "${1:-start}" in
  stop)
    P=$(pgrep -f "$PAT"); [ -n "$P" ] && kill $P
    for _ in $(seq 25); do pgrep -f "$PAT" >/dev/null || break; sleep 0.2; done
    echo stopped ;;
  log) tail -n 40 "$LOG" ;;
  *)
    [ "${1:-start}" = start ] && shift
    pgrep -f "$PAT" >/dev/null && { echo "already running: http://10.42.0.247:8084"; exit 0; }
    mkdir -p "$(dirname "$LOG")"
    setsid nohup ~/workspaces/isaac_ros-dev/src/gcs/run_gcs.sh --port 8084 --mavlink-port 14651 --marker aruco --offboard-alt 1.0 "$@" \
        >> "$LOG" 2>&1 < /dev/null &
    echo "ArUco delivery dashboard: http://10.42.0.247:8084 (log: $0 log)" ;;
esac
