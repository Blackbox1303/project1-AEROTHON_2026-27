#!/usr/bin/env bash
# Start the simulation and fly both corridors with corridor_nav (tests/fly_corridor.py). HOST.
#   sim/run_corridor_test.sh [world.sdf [fly_corridor.py args]]   default sim/worlds/competition.sdf (+ .json)
# Starts in the background (logs /tmp/corridor_test_*.log): Gazebo + PX4 (GUI), the Gazebo -> ROS
# bridge, MAVROS + RViz (sim_view, MAVROS TF off), cuVSLAM + obstacle scan in the container
# (vio_tf:=false), the Arducam (camera_info + TF), Nav2 + odom_tf + offboard_bridge + corridor_nav +
# banner_detector + entry_align; runs the test, stops everything.
#   sim/run_corridor_test.sh sim/worlds/competition.sdf --entrance banner --start-yaw 180
W=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/humble/setup.bash && source "$W/install/setup.bash"   # before set -u: ROS setup isn't -u safe
set -uo pipefail
WORLD=$(realpath "${1:-$W/sim/worlds/competition.sdf}")
[ $# -gt 0 ] && shift          # the rest goes to tests/fly_corridor.py (e.g. --laps return)
TRUTH=${WORLD%.sdf}.json
[ -f "$TRUTH" ] || { echo "no ground truth $TRUTH (make the world with sim/make_competition_world.py)"; exit 1; }

"$W/scripts/sim_terminals.sh" stop > /dev/null 2>&1

PGIDS=()
MARK=$(mktemp)                  # runs newer than this are ours
start() {  # name, command...: own process group, so cleanup takes children with it
    local name=$1; shift
    setsid "$@" > "/tmp/corridor_test_$name.log" 2>&1 &
    PGIDS+=($!)
}
stop_container_vio() {
    docker exec vio-test bash -c '
        for p in $(pgrep -f "^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch vio_bringup sim_vio.launch.py"); do
            kill -INT -- -$(ps -o pgid= -p $p | tr -d " ")
        done' 2>/dev/null
}
cleanup() {
    echo "stopping the simulation"
    stop_container_vio
    for g in "${PGIDS[@]}"; do kill -INT -- -"$g" 2>/dev/null; done
    sleep 6
    for g in "${PGIDS[@]}"; do kill -KILL -- -"$g" 2>/dev/null; done
    "$W/sim/stop_sim.sh" > /dev/null
    # keep this run's process logs with its report (the next run overwrites /tmp/corridor_test_*.log)
    local run
    run=$(find "$W/tests/runs" -mindepth 1 -maxdepth 1 -type d -newer "$MARK" 2>/dev/null | sort | tail -1)
    [ -n "$run" ] && mkdir -p "$run/logs" && cp /tmp/corridor_test_*.log "$run/logs/" && { cp /tmp/corridor_test_altitude_map.npz "$run/" 2>/dev/null; rm -f /tmp/corridor_test_altitude_map.npz; true; } && echo "logs in $run/logs"
    rm -f "$MARK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start sim env DAEMON=1 systemd-inhibit --what=sleep:idle --why='corridor test' "$W/sim/run_sim.sh" "$WORLD"
echo "waiting for PX4 ..."
for _ in $(seq 60); do
    (cd ~/PX4-Autopilot/build/px4_sitl_default/rootfs && timeout 3 ../bin/px4-param show NAV_DLL_ACT 2>/dev/null | grep -q NAV_DLL_ACT) && break
    sleep 2
done
start bridge ros2 launch vio_bringup sim_bridge.launch.py
start mavros ros2 launch vio_bringup sim_view.launch.py mavros_tf:=false
# cuVSLAM (still feeding PX4) + obstacle scan in the container; PX4's estimate owns base_link
start vio docker exec -u admin -w /workspaces/vio_cuda_ws \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/vio_cuda_ws/fastdds_udp.xml vio-test bash -c \
    'source /opt/ros/humble/setup.bash && source install_docker/setup.bash &&
     exec ros2 launch vio_bringup sim_vio.launch.py depth:=true vio_tf:=false scan_band:=0.3 '"${VIO_ARGS:-}"
start arducam ros2 launch vio_bringup sim_arducam.launch.py view:=false
start delivery ros2 launch aerothon_nav delivery.launch.py sim:=true world_json:="$TRUTH" \
    nadir_topic:=/arducam/nadir_hd/image_raw pylib:="$W/sim/.pylib" read_angle:=18.0
start delivery_nav ros2 launch aerothon_nav delivery_nav.launch.py sim:=true gain_radius:=3.2
start mission ros2 launch aerothon_nav mission.launch.py sim:=true
start nav ros2 launch aerothon_nav corridor_nav.launch.py sim:=true course:="${COURSE:-v2}" pylib:="$W/sim/.pylib" \
    qr_fallback:=/arducam/nadir_hd/image_raw altitude_dump:=/tmp/corridor_test_altitude_map.npz
sleep 15
python3 "$W/tests/${TEST:-fly_corridor.py}" --world "$TRUTH" "$@"   # TEST=vio_probe.py: VIO diagnostics
