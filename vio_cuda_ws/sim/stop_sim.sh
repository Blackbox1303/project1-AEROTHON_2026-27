#!/usr/bin/env bash
# Stop a simulation started with sim/run_sim.sh (PX4 SITL + our Gazebo world).
# Leaves other Gazebo instances (e.g. PX4's own default world) alone.
# Patterns are anchored to the start of the process command line: unanchored, they also
# matched (and killed) any shell whose command merely mentioned the PX4 build path.
HERE=$(cd "$(dirname "$0")" && pwd)
PX4='^\.\./bin/px4( |$)'                 # run_sim.sh starts PX4 as ../bin/px4 from rootfs
SERVER="^gz sim -r -s .*$HERE/worlds/"   # our Gazebo server
GUI='^gz sim -g *$'                      # its GUI (gz pads the command line with spaces)
for sig in INT KILL; do
    pkill -"$sig" -f "$PX4" 2>/dev/null
    pkill -"$sig" -f "$SERVER" 2>/dev/null
    pkill -"$sig" -f "$GUI" 2>/dev/null
    [ "$sig" = INT ] && sleep 3
done
echo "stopped; still running:"
ps -eo pid,args | grep -E "^ *[0-9]+ (gz sim|\.\./bin/px4)" || echo "  nothing"
