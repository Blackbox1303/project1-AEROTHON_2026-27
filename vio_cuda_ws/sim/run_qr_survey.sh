#!/usr/bin/env bash
# Start the simulation and fly the QR survey (tests/fly_qr_survey.py). Run on the HOST.
#   sim/run_qr_survey.sh [world.sdf]       default sim/worlds/competition.sdf (+ its .json)
# Starts, in the background (logs in /tmp/qr_survey_*.log): Gazebo + PX4 (GUI visible),
# the Gazebo -> ROS bridge, MAVROS, the Arducam view (rqt_image_view) and the offboard bridge;
# runs the survey, then stops everything. Any sim already running is stopped first.
W=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/humble/setup.bash && source "$W/install/setup.bash"   # before set -u: ROS setup isn't -u safe
set -uo pipefail
WORLD=$(realpath "${1:-$W/sim/worlds/competition.sdf}")
TRUTH=${WORLD%.sdf}.json
[ -f "$TRUTH" ] || { echo "no ground truth $TRUTH (make the world with sim/make_competition_world.py)"; exit 1; }

"$W/scripts/sim_terminals.sh" stop > /dev/null 2>&1

PGIDS=()
start() {  # name, command...: own process group, so cleanup takes children with it
    local name=$1; shift
    setsid "$@" > "/tmp/qr_survey_$name.log" 2>&1 &
    PGIDS+=($!)
}
cleanup() {
    echo "stopping the simulation"
    for g in "${PGIDS[@]}"; do kill -INT -- -"$g" 2>/dev/null; done
    sleep 5
    for g in "${PGIDS[@]}"; do kill -KILL -- -"$g" 2>/dev/null; done
    "$W/sim/stop_sim.sh" > /dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start sim env DAEMON=1 systemd-inhibit --what=sleep:idle --why='QR survey' "$W/sim/run_sim.sh" "$WORLD"
echo "waiting for PX4 ..."
for _ in $(seq 60); do
    (cd ~/PX4-Autopilot/build/px4_sitl_default/rootfs && timeout 3 ../bin/px4-param show NAV_DLL_ACT 2>/dev/null | grep -q NAV_DLL_ACT) && break
    sleep 2
done
start bridge ros2 launch vio_bringup sim_bridge.launch.py
start mavros ros2 launch vio_bringup sim_view.launch.py rviz:=false
start arducam ros2 launch vio_bringup sim_arducam.launch.py
start offboard ros2 launch aerothon_nav offboard_bridge.launch.py
sleep 10
python3 "$W/tests/fly_qr_survey.py" --world "$TRUTH"
