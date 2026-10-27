#!/usr/bin/env bash
# Gazebo window helpers for a running sim/run_sim.sh simulation. The simulation keeps running;
# closing the window doesn't stop it.
#   sim/gui.sh            (re)open the Gazebo window, free camera
#   sim/gui.sh --follow   camera stays with the drone, but you can still rotate / zoom around it
#   sim/gui.sh --free     release the camera again (mouse moves the view freely)
# Plain FOLLOW mode is not used: it locks the view and ignores the mouse.
HERE=$(cd "$(dirname "$0")" && pwd)
source "${PX4_DIR:-$HOME/PX4-Autopilot}/build/px4_sitl_default/gz_env.sh"
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HERE/models GZ_IP=${GZ_IP:-127.0.0.1}
INSTANCE=x500_d435_vio_0

track() {
    gz topic -t /gui/track -m gz.msgs.CameraTrack -p "$1"
}

case "${1:-}" in
    --follow)
        track "track_mode: FOLLOW_FREE_LOOK, follow_target: {name: '$INSTANCE'}, \
            follow_offset: {x: -2.0, y: -2.0, z: 2.0}, follow_pgain: 1.0, track_pgain: 1.0"
        echo "camera following $INSTANCE (free look)" ;;
    --free)
        track "track_mode: NONE"
        echo "camera free" ;;
    "")
        pkill -INT -f "^gz sim -g *$" 2>/dev/null; sleep 2
        setsid gz sim -g > /tmp/vio_sim_gui.log 2>&1 < /dev/null &
        echo "Gazebo window (re)opened" ;;
    *)
        echo "usage: $0 [--follow|--free]"; exit 1 ;;
esac
