#!/bin/bash
# Stop the container mission stack (sim or real), the offboard VRX run (explorer + bench SITL) and the host side (Gazebo, PX4 / ArduCopter SITL, bridge, MAVROS, mavlink-router). Patterns live in this file, so a
# caller's command line never matches them. The always-on onboard router (runs/gcs/onboard_router.conf) is left alone.
pats='ros2 topic echo|container_all\.launch|container_real\.launch|tunnel_real\.launch|realsense2_camera_node|mavlink-routerd -c [^ ]*(sim|/router)\.conf|host_all\.launch|mavros_node|parameter_bridge|bin/px4|sitl/bin/arducopter --model (JSON|quad)|src/gcs/vrx_explore/vrx_explore\.py|gz sim|aerothon_nav/lib|slam_toolbox|nav2_|lifecycle_manager|static_transform_publisher|vio_bringup/lib|run_competition|cuvslam\.launch|visual_slam_container|aerothon_corridor_stack|aerothon_corridor\.sh'
mine=$$
# The tunnel + SLAM stack (tunnel_slam / tunnel_sim_saved launches, the container cuVSLAM launch, the dashboard's RViz /
# RTAB-Map) shares process names with the mission stack (mavros_node, slam_toolbox, realsense2_camera_node, ...): a
# mission-side stop (8080 Stop, radio "stop stack", a mission launch's Ctrl-C) used to kill it too, mid-run
# (2026-10-27). Its whole process tree is left alone unless called with --tunnel (the tunnel dashboard's Stop).
keep=$mine
if [ "$1" != "--tunnel" ]; then
    roots=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch .*(tunnel_slam|tunnel_sim_saved|tunnel_viz)\.launch\.py|ros2 launch vio_bringup cuvslam\.launch\.py|rviz2_tunnel|tunnel_gcs\.py' | tr '\n' ' ')
    if [ -n "$roots" ]; then
        keep="$keep $(ps -eo pid=,ppid= | awk -v r="$roots" 'BEGIN { n = split(r, a, " "); for (i = 1; i <= n; i++) k[a[i]] = 1 }
            { p[$1] = $2 } END { do { c = 0; for (x in p) if (!(x in k) && (p[x] in k)) { k[x] = 1; c = 1 } } while (c)
            for (x in k) printf "%s ", x }')"
    fi
fi
list() { pgrep -f "$pats" | grep -vxF -f <(tr ' ' '\n' <<< "$keep" | grep .); }
# STOP_ALL_DRY=1: print what would be stopped, stop nothing
[ -n "$STOP_ALL_DRY" ] && { p=$(list); [ -n "$p" ] && ps -o pid=,args= -p $p | cut -c1-110; exit 0; }
kill -INT $(list) 2>/dev/null; sleep 10
kill -TERM $(list) 2>/dev/null; sleep 4
kill -KILL $(list) 2>/dev/null; sleep 1
left=$(list); [ -z "$left" ] && echo "all stopped" || ps -o pid,args -p $left | cut -c1-100
