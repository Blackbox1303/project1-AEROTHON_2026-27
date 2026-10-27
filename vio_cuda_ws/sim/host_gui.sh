#!/usr/bin/env bash
# Gazebo GUI on another (x86) machine for a sim running headless on the Jetson, so all
# rendering happens on that machine. Copy this script to the HOST and run it there.
#
# Jetson:  ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/host_all.launch.py gz_ip:=10.42.0.247 [world:=...]
#          (Gazebo + ArduCopter / PX4 SITL + the ROS bridge on one GZ_IP / GZ_PARTITION=vio, no GUI on the Jetson), or by hand:
#          HEADLESS=1 GZ_IP=10.42.0.247 GZ_PARTITION=vio sim/run_competition.sh [world.sdf]
# Host:    ./host_gui.sh            sync models/worlds from the Jetson, then open the GUI
#          ./host_gui.sh --nosync   skip the sync
#          ./host_gui.sh --follow | --free   camera control, as in sim/gui.sh
#
# Host needs: gz-harmonic (gz sim 8, same as the Jetson), rsync, ssh access to the Jetson; its firewall open to the
# Jetson (gz-transport: UDP multicast 239.255.0.7:10317-10318 + random TCP ports; `sudo ufw allow from 10.42.0.0/24`).
# No GPU: the GUI renders on the CPU (Mesa llvmpipe) - GUI_ENGINE=ogre (Ogre 1) is usually lighter than ogre2.
# Worlds reference textures by absolute /home/szyfrowac/... paths, so the files are mirrored
# at the same paths on the host (one-time sudo to create /home/szyfrowac if it doesn't exist).
set -eo pipefail

JETSON=${JETSON:-szyfrowac@10.42.0.247}
JETSON_HOME=/home/szyfrowac
SIM=$JETSON_HOME/workspaces/vio_cuda_ws/sim
PX4_MODELS=$JETSON_HOME/PX4-Autopilot-1.16.2/Tools/simulation/gz/models
AP_MODELS=$JETSON_HOME/ardupilot_gazebo/models          # the ArduCopter SITL airframe (iris_d435_vio)

export GZ_PARTITION=${GZ_PARTITION:-vio}
export GZ_IP=${GZ_IP:-$(ip -4 route get "${JETSON#*@}" | grep -oP 'src \K\S+')}
export GZ_SIM_RESOURCE_PATH=$SIM/models:$SIM/worlds:$PX4_MODELS:$AP_MODELS
# GNOME on Wayland (the Lenovo): run the Qt / Ogre GUI through XWayland - native Wayland Qt + Ogre GL is unreliable
[ -n "$WAYLAND_DISPLAY" ] && export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

track() {
    gz topic -t /gui/track -m gz.msgs.CameraTrack -p "$1"
}

case "${1:-}" in
    --follow)
        # the drone: iris_d435_vio_0 (ArduCopter SITL, the default) or x500_d435_vio_0 (fcu:=px4)
        INSTANCE=$(gz model --list 2>/dev/null | grep -oE '(iris|x500)_d435_vio_0' | head -1)
        INSTANCE=${INSTANCE:-iris_d435_vio_0}
        track "track_mode: FOLLOW_FREE_LOOK, follow_target: {name: '$INSTANCE'}, \
            follow_offset: {x: -2.0, y: -2.0, z: 2.0}, follow_pgain: 1.0, track_pgain: 1.0"
        exit ;;
    --free)
        track "track_mode: NONE"; exit ;;
    ""|--nosync) ;;
    *)
        echo "usage: $0 [--nosync|--follow|--free]"; exit 1 ;;
esac

if [ "${1:-}" != --nosync ]; then
    [ -w "$JETSON_HOME" ] || { sudo mkdir -p "$JETSON_HOME" && sudo chown "$USER:" "$JETSON_HOME"; }
    mkdir -p "$SIM" "$PX4_MODELS" "$AP_MODELS"
    echo "syncing worlds / models from $JETSON ..."
    RS=(rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=5")
    "${RS[@]}" --exclude 'backup_*' --exclude __pycache__ --exclude '*.log' "$JETSON:$SIM/" "$SIM/" \
        && "${RS[@]}" "$JETSON:$PX4_MODELS/" "$PX4_MODELS/" && "${RS[@]}" "$JETSON:$AP_MODELS/" "$AP_MODELS/" \
        || { echo "sync from $JETSON failed (ssh key? install_lenovo_agent.sh)"; exit 1; }
fi

# the stack takes 1-2 min to bring Gazebo up: Open Gazebo right after Launch waits for it (GUI_WAIT s)
echo "GUI: partition=$GZ_PARTITION ip=$GZ_IP; waiting for the sim on the Jetson ..."
for _ in $(seq "${GUI_WAIT:-300}"); do
    gz topic -l 2>/dev/null | grep -q '^/world/.*/clock$' && break
    sleep 1
done
gz topic -l | grep -q '^/world/.*/clock$' \
    || { echo "no Gazebo world seen: check GZ_PARTITION/GZ_IP on both sides and the host firewall"; exit 1; }
exec gz sim -g ${GUI_ENGINE:+--render-engine-gui "$GUI_ENGINE"}
