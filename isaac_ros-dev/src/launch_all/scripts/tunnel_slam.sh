#!/bin/bash
# Tunnel + SLAM stack (D435 + cuVSLAM in the container, 2D SLAM, RTAB-Map, RViz) and its dashboard http://10.42.0.247:8093
#   tunnel_slam.sh start [launch args]   start (or restart) in the background; RViz + RTAB-Map open on the Jetson display
#   tunnel_slam.sh stop                  stop the stack; RViz / RTAB-Map windows stay open (close them yourself)
#   tunnel_slam.sh status | log
# Same from the Lenovo:  ssh szyfrowac@10.42.0.247 '~/workspaces/isaac_ros-dev/src/launch_all/scripts/tunnel_slam.sh start'
# Defaults: Pixhawk on TELEM2 (/dev/ttyTHS1), no flight nodes (nav:=false). Extra args are passed to the launch, e.g.
#   tunnel_slam.sh start link:=none        camera + SLAM only        tunnel_slam.sh start nav:=true   with flight nodes
WS=~/workspaces/isaac_ros-dev
LAUNCH='^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch .*tunnel_slam\.launch\.py'
latest() { ls -t $WS/runs/tunnel_slam_*_console.log 2>/dev/null | head -1; }

case "${1:-start}" in
  start)
    shift
    RUN=tunnel_slam_$(date +%F_%H-%M-%S)
    source /opt/ros/humble/setup.bash
    export DISPLAY=${DISPLAY:-:0}
    cd $WS
    # setsid: own session, survives the ssh / terminal closing. A running stack is stopped by the launch itself.
    setsid nohup ros2 launch src/launch_all/tunnel_slam.launch.py run:=$RUN link:=/dev/ttyTHS1:921600 \
        rviz:=true rtabmap:=true nav:=false "$@" > runs/${RUN}_console.log 2>&1 < /dev/null &
    echo "starting $RUN (camera up in ~20-60 s)"
    echo "dashboard: http://10.42.0.247:8093   log: $0 log" ;;
  stop)
    P=$(pgrep -f "$LAUNCH")
    [ -z "$P" ] && { echo "not running"; exit 0; }
    kill -INT $P
    for _ in $(seq 50); do pgrep -f "$LAUNCH" >/dev/null || break; sleep 0.3; done
    pgrep -f "$LAUNCH" >/dev/null && kill -TERM $(pgrep -f "$LAUNCH")
    echo "stopped (RViz / RTAB-Map stay open)" ;;
  status)
    for x in "launch:$LAUNCH" 'dashboard:^python3 .*tunnel_gcs\.py' 'camera:realsense2_camera_node' \
             'cuVSLAM:visual_slam_container' 'RViz:^/opt/ros/humble/lib/rviz2/rviz2 ' \
             'RTAB-Map:^/opt/ros/humble/lib/rtabmap_slam/rtabmap' 'RTAB-Map window:^/opt/ros/humble/lib/rtabmap_viz/'; do
        printf "%-16s %s\n" "${x%%:*}" "$(pgrep -f "${x#*:}" >/dev/null && echo running || echo -)"
    done
    echo "log: $(latest)" ;;
  log)
    tail -f "$(latest)" ;;
  *)
    sed -n 2,8p "$0" ;;
esac
