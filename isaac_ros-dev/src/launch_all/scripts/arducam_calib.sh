#!/bin/bash
# Arducam calibration + QR test page (http://10.42.0.247:8090) in the dev container. A systemd --user service
# (arducam-calib) starts it at boot; only ONE process can use the camera, so the ROS camera (arducam_up.sh) and the real
# mission stack (real_mission.launch.py) stop this page first, and the real stack starts it again when it ends.
#   arducam_calib.sh [start [args]] | stop | fg [args]   (fg = what the service runs; args go to arducam_calib.py,
#   e.g. start --qr-max-exposure-ms 15 for an indoor QR test; kept in runs/arducam_real/calib_args until changed)
# Log: runs/arducam_real/calib.log
C=isaac_ros_dev-aarch64-pyzbar-container
PAT='^python3 /workspaces/isaac_ros-dev/src/vio_bringup/calibration/arducam_calib.py'
case "${1:-start}" in
  stop)
    systemctl --user stop arducam-calib 2>/dev/null
    P=$(pgrep -f "$PAT"); [ -n "$P" ] && kill $P
    echo stopped ;;
  fg)
    shift
    [ $# = 0 ] && [ -f ~/workspaces/isaac_ros-dev/runs/arducam_real/calib_args ] && \
        set -- $(cat ~/workspaces/isaac_ros-dev/runs/arducam_real/calib_args)
    for _ in $(seq 120); do                                   # the container comes up at boot (restart unless-stopped)
        [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" = true ] && break; sleep 2
    done
    "$(dirname "$0")/arducam_down.sh" >/dev/null               # the ROS camera holds the camera: one at a time
    P=$(pgrep -f "$PAT"); [ -n "$P" ] && kill $P && \
        for _ in $(seq 25); do pgrep -f "$PAT" >/dev/null || break; sleep 0.2; done   # it releases Argus (<= 3 s)
    mkdir -p ~/workspaces/isaac_ros-dev/runs/arducam_real
    exec docker exec -u admin $C bash -c "unset DISPLAY; cd /workspaces/isaac_ros-dev; exec python3 \
/workspaces/isaac_ros-dev/src/vio_bringup/calibration/arducam_calib.py $* >> /workspaces/isaac_ros-dev/runs/arducam_real/calib.log 2>&1" ;;
  *)
    [ "${1:-start}" = start ] && shift
    mkdir -p ~/workspaces/isaac_ros-dev/runs/arducam_real
    echo "$*" > ~/workspaces/isaac_ros-dev/runs/arducam_real/calib_args      # empty = the defaults
    systemctl --user restart arducam-calib
    echo "calibration page: http://10.42.0.247:8090 (log runs/arducam_real/calib.log)" ;;
esac
