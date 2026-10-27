#!/bin/bash
# save_run.sh <run dir on the host> [loop]: copy what lives elsewhere into the run dir - the autopilot's newest flight log (PX4 .ulg / ArduCopter .BIN), the
# Gazebo server log, the container's ~/.ros mission files (geofence + snapshot, start QR, delivery memory). loop:
# every 60 s until killed (sim_mission.launch.py runs it so).
R=$1; mkdir -p "$R"
RC=/workspaces/isaac_ros-dev/${R#$HOME/workspaces/isaac_ros-dev/}     # the same dir inside the container
REAL=0; grep -q '^link=' "$R/launch_args.txt" 2>/dev/null && REAL=1   # real_mission run: no SITL / Gazebo logs
save() {
    if [ $REAL = 0 ]; then
        if grep -q '^fcu=px4' "$R/launch_args.txt" 2>/dev/null; then
            cp -u "$(ls -t ~/PX4-Autopilot-1.16.2/build/px4_sitl_default/rootfs/log/*/*.ulg 2>/dev/null | head -1)" "$R/" 2>/dev/null
        else                                                    # ArduCopter SITL dataflash log (run_competition_ap.sh)
            cp -u "$(ls -t ~/workspaces/isaac_ros-dev/runs/ap_sitl/logs/*.BIN 2>/dev/null | head -1)" "$R/" 2>/dev/null
        fi
        cp -u /tmp/vio_sim_gz.log "$R/gz_server.log" 2>/dev/null
    fi
    docker exec -u admin isaac_ros_dev-aarch64-pyzbar-container bash -c "cp -u ~/.ros/aerothon_* '$RC/' 2>/dev/null"
}
save
trap "exit 0" INT TERM
[ "${2:-}" = loop ] && while sleep 60 & wait $!; do save; done
true
