#!/usr/bin/env bash
# Run PX4 SITL (your ~/PX4-Autopilot build) in our tunnel world. Run on the HOST.
#   sim/run_sim.sh [world.sdf]        default sim/worlds/tunnel.sdf (make it with make_tunnel_world.py)
# Env: HEADLESS=1   no Gazebo GUI
#      DAEMON=1     no interactive pxh> shell (for scripts)
#      PX4_DIR      default ~/PX4-Autopilot
# Gazebo is started here with our world + models; PX4 attaches in standalone mode,
# so the PX4 tree is not modified. MAVROS: fcu_url:=udp://:14540@127.0.0.1:14557
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PX4_DIR=${PX4_DIR:-$HOME/PX4-Autopilot}
WORLD=$(realpath "${1:-$HERE/worlds/tunnel.sdf}")
NAME=$(grep -o -m1 '<world name="[^"]*"' "$WORLD" | cut -d'"' -f2)
[ -n "$NAME" ] || { echo "no <world name=...> in $WORLD"; exit 1; }

source "$PX4_DIR/build/px4_sitl_default/gz_env.sh"
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HERE/models
export GZ_IP=${GZ_IP:-127.0.0.1}   # set to the LAN IP to view the GUI from another machine (sim/host_gui.sh)

# Server and GUI run as separate processes: started together (gz sim -r world), the GUI dying
# takes the server down with it. The GUI opens after the drone is spawned (see below).
setsid gz sim -r -s --headless-rendering "$WORLD" > /tmp/vio_sim_gz.log 2>&1 &
GZ=$!
GUI=
cleanup() {
    for g in $GUI $GZ; do kill -INT -- -"$g" 2>/dev/null || true; done
    sleep 3
    for g in $GUI $GZ; do kill -KILL -- -"$g" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "waiting for Gazebo world '$NAME' ..."
for _ in $(seq 60); do
    gz topic -l 2>/dev/null | grep -q "^/world/$NAME/clock$" && break
    sleep 1
done
gz topic -l | grep -q "^/world/$NAME/clock$" || { echo "Gazebo did not start, see /tmp/vio_sim_gz.log"; exit 1; }

# Spawn the drone ourselves: PX4's own spawn only looks in its PX4_GZ_MODELS folder.
# PX4 then attaches to it by name (PX4_GZ_MODEL_NAME).
MODEL=x500_d435_vio
INSTANCE=${MODEL}_0
gz service -s "/world/$NAME/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
    --timeout 5000 --req "sdf_filename: \"$HERE/models/$MODEL/model.sdf\", name: \"$INSTANCE\", \
    pose: {position: {x: 0, y: 0, z: 0.25}}" | grep -q "data: true" \
    || { echo "spawning $MODEL failed, see /tmp/vio_sim_gz.log"; exit 1; }
for _ in $(seq 20); do
    gz topic -l | grep -q "^/world/$NAME/model/$INSTANCE/link/base_link/sensor/imu_sensor/imu$" && break
    sleep 0.5
done
# Make sure physics is running: with the GUI the world was seen paused after start-up
# (drone frozen at the spawn height), despite gz sim -r.
gz service -s "/world/$NAME/control" --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'pause: false' > /dev/null

# GUI (unless HEADLESS) only now, so it sees the drone. Free camera; sim/gui.sh --follow to
# keep it with the drone, sim/gui.sh to reopen the window.
if [ -z "$HEADLESS" ]; then
    setsid gz sim -g > /tmp/vio_sim_gui.log 2>&1 &
    GUI=$!
fi

cd "$PX4_DIR/build/px4_sitl_default/rootfs"

# Apply sim/px4_sim_params.txt once PX4 is up, in the background so the pxh> shell stays
# usable. Results go to a log file: printing here would push the pxh> prompt out of view.
PARAM_LOG=/tmp/vio_sim_params.log
echo "PX4 sim params: see $PARAM_LOG (press Enter if the pxh> prompt is not visible)"
(
    for _ in $(seq 60); do ../bin/px4-param show SYS_AUTOSTART > /dev/null 2>&1 && break; sleep 1; done
    grep -v -E '^\s*(#|$)' "$HERE/px4_sim_params.txt" | while read -r name value; do
        ../bin/px4-param set "$name" "$value" > /dev/null 2>&1 \
            && echo "param $name = $value" || echo "WARN could not set $name"
    done
) > "$PARAM_LOG" 2>&1 &

PX4_ARGS=()
[ -n "$DAEMON" ] && PX4_ARGS+=(-d)
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$NAME" PX4_GZ_MODEL_NAME="$INSTANCE" PX4_SYS_AUTOSTART=4010 \
    ../bin/px4 "${PX4_ARGS[@]}"
