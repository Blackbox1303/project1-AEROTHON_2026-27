#!/bin/bash
# VRX Explore (http://10.42.0.247:8094): QR scanning + frontier / A* exploration round red zones at 1 m, on PX4.
#   vrx_explore.sh start [--video /dev/videoN] [app args]   PX4 SIH SITL (if no PX4 is talking) + the 8094 app
#   vrx_explore.sh stop                                     stops the app and the SIH SITL
#   vrx_explore.sh status | log | sitl-log
# From the Lenovo:  ssh szyfrowac@10.42.0.247 '~/workspaces/isaac_ros-dev/src/launch_all/scripts/vrx_explore.sh start'
# The SITL is PX4's built-in SIH quad (no Gazebo): its MAVLink goes to the always-on aerothon-router (UDP 14540),
# the app talks to the router on 14551. Video: pick the USB device on the page (D435 / CSI nodes are never opened),
# or pass --video. Real drone later: --link /dev/ttyUSB0:57600 (radio) and no SITL (stop it).
WS=~/workspaces/isaac_ros-dev
PX4=~/PX4-Autopilot-1.16.2/build/px4_sitl_default
SIH=$WS/runs/sih
APP='^python3 .*src/gcs/vrx_explore/vrx_explore\.py'
SITL="^$PX4/bin/px4 -d $SIH"

case "${1:-start}" in
  start)
    shift
    mkdir -p $SIH $WS/runs/vrx_explore
    if pgrep -f "^$PX4/bin/px4" >/dev/null; then
        echo "a PX4 SITL is already running: using it"
    else
        ln -sfn $PX4/rootfs/etc $SIH/etc       # PX4 reads its startup files relative to its working dir
        (cd $SIH && PX4_SIM_MODEL=sihsim_quadx setsid nohup $PX4/bin/px4 -d $SIH > $SIH/px4.log 2>&1 < /dev/null &)
        echo "PX4 SIH SITL starting (log: $0 sitl-log)"
    fi
    if pgrep -f "$APP" >/dev/null; then
        echo "app already running: http://10.42.0.247:8094 (stop first to restart)"
        exit 0
    fi
    cd $WS
    LOG=runs/vrx_explore/vrx_explore_$(date +%F_%H-%M-%S).log
    setsid nohup python3 $WS/src/gcs/vrx_explore/vrx_explore.py "$@" > $LOG 2>&1 < /dev/null &
    echo "app starting: http://10.42.0.247:8094   (log: $0 log)" ;;
  stop)
    for p in "$APP" "$SITL"; do
        P=$(pgrep -f "$p"); [ -z "$P" ] && continue
        kill -INT $P
        for _ in $(seq 25); do pgrep -f "$p" >/dev/null || break; sleep 0.2; done
        pgrep -f "$p" >/dev/null && kill -TERM $(pgrep -f "$p")
    done
    echo stopped ;;
  status)
    printf "%-10s %s\n" app "$(pgrep -f "$APP" >/dev/null && echo running || echo -)" \
                         SITL "$(pgrep -f "^$PX4/bin/px4" >/dev/null && echo running || echo -)"
    echo "page: http://10.42.0.247:8094" ;;
  log)
    tail -f "$(ls -t $WS/runs/vrx_explore/vrx_explore_*.log | head -1)" ;;
  sitl-log)
    tail -f $SIH/px4.log ;;
  *)
    sed -n 2,10p "$0" ;;
esac
