#!/usr/bin/env bash
set -Eeuo pipefail

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot-1.16.2}"
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD="$(realpath "${1:-$SIM_DIR/worlds/competition.sdf}")"
ROOTFS="$PX4_DIR/build/px4_sitl_default/rootfs"
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
MODEL="x500_d435_vio"
INSTANCE="${MODEL}_0"

test -x "$PX4_BIN" || { echo "Missing PX4 binary: $PX4_BIN"; exit 1; }
test -f "$ROOTFS/gz_env.sh" || { echo "Missing gz_env.sh"; exit 1; }
test -f "$WORLD" || { echo "Missing world: $WORLD"; exit 1; }

source "$ROOTFS/gz_env.sh"
export GZ_SIM_RESOURCE_PATH="$SIM_DIR/models:$SIM_DIR/worlds:$GZ_SIM_RESOURCE_PATH"
export GZ_IP=${GZ_IP:-127.0.0.1}   # set to the LAN IP to view the GUI from another machine (sim/host_gui.sh)

NAME="$(sed -n 's/.*<world name="\([^"]*\)".*/\1/p' "$WORLD" | head -1)"
test -n "$NAME" || { echo "Could not determine world name"; exit 1; }

GZ=""
GUI=""
cleanup() {
    echo "Stopping simulation..."
    [ -z "$GUI" ] || kill -INT "$GUI" 2>/dev/null || true
    [ -z "$GZ" ] || kill -INT "$GZ" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Gazebo world: $NAME"
setsid gz sim -r -s --headless-rendering "$WORLD" \
    > /tmp/vio_sim_gz.log 2>&1 &
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
    --req "sdf_filename: \"$SIM_DIR/models/$MODEL/model.sdf\", name: \"$INSTANCE\", pose: {position: {x: 0, y: 0, z: 0.25}}"

# Ensure physics is running
gz service -s "/world/$NAME/control" --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'pause: false' > /dev/null 2>&1 || true

if [ "${HEADLESS:-0}" = 0 ]; then                    # HEADLESS=1: GUI elsewhere (host_gui.sh)
    echo "Starting Gazebo GUI"
    setsid gz sim -g > /tmp/vio_sim_gui.log 2>&1 &
    GUI=$!
fi

cd "$ROOTFS"

# Apply simulation parameters in background once PX4 starts
if [ -f "$SIM_DIR/px4_sim_params.txt" ]; then
    (
        for _ in $(seq 60); do ../bin/px4-param show SYS_AUTOSTART > /dev/null 2>&1 && break; sleep 1; done
        grep -v -E '^\s*(#|$)' "$SIM_DIR/px4_sim_params.txt" | while read -r name value; do
            ../bin/px4-param set "$name" "$value" > /dev/null 2>&1 \
                && echo "param $name = $value" || echo "WARN could not set $name"
        done
    ) > /tmp/vio_sim_params.log 2>&1 &
fi

echo "Starting PX4 SITL attached to $INSTANCE"
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD="$NAME" \
PX4_GZ_MODEL_NAME="$INSTANCE" \
PX4_SYS_AUTOSTART=4010 \
"$PX4_BIN"
