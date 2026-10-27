#!/usr/bin/env bash
# Browser view of a running sim/run_sim.sh (or run_competition.sh) simulation, for remote
# viewing without a Gazebo GUI on the Jetson: start the sim with HEADLESS=1, then run this.
# All 3D rendering happens in the viewer's browser; the Jetson only forwards scene + poses.
#   On the viewing machine:  ssh -N -L 9002:localhost:9002 szyfrowac@<jetson>
#   then open https://app.gazebosim.org/visualization and connect to ws://localhost:9002
# Ctrl+C stops the stream; the simulation keeps running.
HERE=$(cd "$(dirname "$0")" && pwd)
source "${PX4_DIR:-$HOME/PX4-Autopilot}/build/px4_sitl_default/gz_env.sh"
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HERE/models:$HERE/worlds GZ_IP=${GZ_IP:-127.0.0.1}
exec gz launch -v 3 "$HERE/websocket.gzlaunch"
