#!/bin/bash
# Stop the real-Arducam launch. The container shares the host's PIDs (pid=host): match only the launch's own python
# process (anchored), never a shell whose command line merely mentions it
P=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch vio_bringup arducam_real')
[ -z "$P" ] && { echo "not running"; exit 0; }
kill -INT $P && echo "stopped ($P)"
