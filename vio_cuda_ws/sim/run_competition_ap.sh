#!/usr/bin/env bash
# ArduCopter SITL version of run_competition.sh (2026-10-27): Gazebo Harmonic headless + the iris_d435_vio model
# (ardupilot_gazebo's ArduPilotPlugin, lock-step JSON on UDP 9002) + arducopter (~/ardupilot, Copter-4.7.2).
# MAVLink goes out of SERIAL0 to UDP 14540 - the port PX4 SITL used - so mavlink-router (gcs/router/sim.conf or the
# onboard aerothon-router), MAVROS, the dashboard and QGC are wired as before.
#   run_competition_ap.sh [world.sdf]       env: GZ_IP, GZ_PARTITION, HEADLESS (1: no local GUI)
# Params: ardupilot_sim_params.parm (+ copter.parm, gazebo-iris.parm); the eeprom is wiped on every start.
# SITL work dir (eeprom.bin, dataflash logs/*.BIN): ~/workspaces/isaac_ros-dev/runs/ap_sitl
set -Eeuo pipefail

AP_DIR="${AP_DIR:-$HOME/ardupilot}"
AP_GZ_DIR="${AP_GZ_DIR:-$HOME/ardupilot_gazebo}"
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD_SRC="$(realpath "${1:-$SIM_DIR/worlds/competition.sdf}")"
AP_BIN="$AP_DIR/build/sitl/bin/arducopter"
WORK="$HOME/workspaces/isaac_ros-dev/runs/ap_sitl"
MODEL="${AP_MODEL:-iris_d435_vio}"      # AP_MODEL=iris_b19: the 8084 offboard sim (B19 analog camera only)
INSTANCE="${MODEL}_0"
STEP=0.0025                       # s: 400 Hz physics = SCHED_LOOP_RATE (PX4 ran the world at 4 ms)

test -x "$AP_BIN" || { echo "Missing ArduCopter SITL binary: $AP_BIN (./waf configure --board sitl && ./waf copter)"; exit 1; }
test -f "$AP_GZ_DIR/build/libArduPilotPlugin.so" || { echo "Missing $AP_GZ_DIR/build/libArduPilotPlugin.so (build ardupilot_gazebo)"; exit 1; }
test -f "$WORLD_SRC" || { echo "Missing world: $WORLD_SRC"; exit 1; }

# The generated worlds carry PX4's 4 ms step: a copy with the ArduPilot step, next to the original (same model://
# and relative paths). Rebuilt each start, so make_competition_world.py stays the only source of the course
mkdir -p "$WORK"
WORLD="$(dirname "$WORLD_SRC")/.ap_$(basename "$WORLD_SRC")"
sed -e "s#<max_step_size>[^<]*</max_step_size>#<max_step_size>$STEP</max_step_size>#" \
    -e "s#<real_time_update_rate>[^<]*</real_time_update_rate>#<real_time_update_rate>$(awk "BEGIN{print 1/$STEP}")</real_time_update_rate>#" \
    "$WORLD_SRC" > "$WORLD"

export GZ_SIM_RESOURCE_PATH="$SIM_DIR/models:$SIM_DIR/worlds:$AP_GZ_DIR/models:$AP_GZ_DIR/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$AP_GZ_DIR/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
# world systems (the worlds declare none; PX4 supplied them through its server.config): no baro / GPS / mag
# systems - ArduPilot SITL simulates those from the JSON state
export GZ_SIM_SERVER_CONFIG_PATH="$SIM_DIR/ap_server.config"
export GZ_IP=${GZ_IP:-127.0.0.1}

NAME="$(sed -n 's/.*<world name="\([^"]*\)".*/\1/p' "$WORLD" | head -1)"
test -n "$NAME" || { echo "Could not determine world name"; exit 1; }
# SITL home = the world's spherical_coordinates: GPS <-> odom (the ground station's geofence) as with PX4
LAT="$(sed -n 's#.*<latitude_deg>\s*\([^<]*\)</latitude_deg>.*#\1#p' "$WORLD" | head -1)"
LON="$(sed -n 's#.*<longitude_deg>\s*\([^<]*\)</longitude_deg>.*#\1#p' "$WORLD" | head -1)"
ELE="$(sed -n 's#.*<elevation>\s*\([^<]*\)</elevation>.*#\1#p' "$WORLD" | head -1)"
HOME_LOC="${LAT:-47.397971},${LON:-8.546164},${ELE:-488},90"      # yaw 90: the model spawns facing +x = east

GZ=""
GUI=""
AP=""
cleanup() {
    echo "Stopping simulation..."
    # arducopter ignores SIGINT, and SIGTERM too while it waits for a lock-step frame from a stopped Gazebo
    [ -z "$AP" ] || { kill -TERM "$AP" 2>/dev/null; sleep 2; kill -KILL "$AP" 2>/dev/null; } || true
    [ -z "$GUI" ] || kill -INT "$GUI" 2>/dev/null || true
    [ -z "$GZ" ] || kill -INT "$GZ" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Gazebo world: $NAME (step $STEP s)"
setsid gz sim -r -s --headless-rendering "$WORLD" > /tmp/vio_sim_gz.log 2>&1 &
GZ=$!

echo "Waiting for Gazebo..."
READY=0
for _ in $(seq 1 60); do
    if gz topic -l 2>/dev/null | grep -qx "/world/$NAME/clock"; then
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" != 1 ]; then
    echo "Gazebo failed to start. Check /tmp/vio_sim_gz.log"
    exit 1
fi

echo "Spawning $INSTANCE"
gz service -s "/world/$NAME/create" \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --timeout 10000 \
    --req "sdf_filename: \"$SIM_DIR/models/$MODEL/model.sdf\", name: \"$INSTANCE\", pose: {position: {x: ${AP_SPAWN_X:-0}, y: ${AP_SPAWN_Y:-0}, z: 0.25}}"

gz service -s "/world/$NAME/control" --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'pause: false' > /dev/null 2>&1 || true

if [ "${HEADLESS:-0}" = 0 ]; then                    # HEADLESS=1: GUI elsewhere (host_gui.sh)
    echo "Starting Gazebo GUI"
    setsid gz sim -g > /tmp/vio_sim_gui.log 2>&1 &
    GUI=$!
fi

D="$AP_DIR/Tools/autotest/default_params"
# AP_VIO=1 (sim_mission corridor:=aerothon vio_fc:=1): + the VIO -> EKF3 source set 2 params
echo "Starting ArduCopter SITL (home $HOME_LOC), MAVLink -> udp 127.0.0.1:14540${AP_VIO:+, VIO -> EKF3 source set 2}"
cd "$WORK"
# own TCP (SERIAL1/2: 5782/5783) and RC-in (5581) ports: a hand-started `arducopter -I0` (5760-5763, 5501) can run
# beside it. The JSON link stays on 9002 (iris_with_ardupilot's fdm_port_in)
"$AP_BIN" --model JSON --speedup 1 --wipe --base-port 5780 --rc-in-port 5581 -I0 --home "$HOME_LOC" \
    --defaults "$D/copter.parm,$D/gazebo-iris.parm,$SIM_DIR/ardupilot_sim_params.parm${AP_VIO:+,$SIM_DIR/ardupilot_vio_sim.parm}" \
    --serial0 udpclient:127.0.0.1:14540 &
AP=$!
wait "$AP"
