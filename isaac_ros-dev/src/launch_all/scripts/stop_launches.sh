#!/bin/bash
# Ctrl-C every sim_mission / real_link launch (they stop their stacks), wait, then stop_all.sh for anything left.
pids=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch .*(sim_mission|real_link)\.launch\.py')
[ -n "$pids" ] && kill -INT $pids
for i in $(seq 75); do alive=0; for p in $pids; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 2; done
"$(dirname "$0")/stop_all.sh"
