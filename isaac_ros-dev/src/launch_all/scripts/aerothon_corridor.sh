#!/bin/bash
# aerothon_corridor.sh <sim|real> <vio_fc 0|1> <log file>: the AeroTHON folder's corridor navigation + VIO
# (launch_all/aerothon_corridor_stack.py) in that folder's own container, aerothon-vio. Started by sim_mission /
# real_mission.launch.py with corridor:=aerothon. Its start_container.sh starts the container if needed and rebuilds
# their workspace when their src/ changed; the launcher goes in on stdin (the container doesn't mount ~/workspaces).
# Stopped by stop_all.sh (both containers share the host's PID namespace).
MODE=${1:-sim}; FC=${2:-0}; LOG=${3:-/dev/stdout}
AERO=$HOME/AeroTHON_2026/vio_cuda_ws
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$AERO/scripts/start_container.sh" ] || { echo "[aerothon_corridor] $AERO not found"; exit 1; }
echo "[aerothon_corridor] container aerothon-vio (build check) ..."
VIO_CONTAINER=aerothon-vio "$AERO/scripts/start_container.sh" true < /dev/null > "$LOG.build" 2>&1 \
    || { echo "[aerothon_corridor] start_container.sh failed: $LOG.build"; exit 1; }
echo "[aerothon_corridor] starting ($MODE, vio_fc=$FC): log $LOG"
exec docker exec -i -u admin -w /workspaces/vio_cuda_ws -e AEROTHON_AUTOPILOT=ardupilot -e OPENBLAS_NUM_THREADS=1 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/vio_cuda_ws/fastdds_udp.xml aerothon-vio bash -c '
    source /opt/ros/humble/setup.bash && source install_docker/setup.bash && exec python3 - "$@"' \
    _ aerothon_corridor_stack "mode=$MODE" "vio_fc=$FC" < "$HERE/aerothon_corridor_stack.py" > "$LOG" 2>&1
