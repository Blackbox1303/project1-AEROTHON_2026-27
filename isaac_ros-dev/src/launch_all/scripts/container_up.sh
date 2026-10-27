#!/bin/bash
# Start the dev container DETACHED and PERSISTENT if it isn't running (same options as isaac_ros_common/scripts/
# run_dev.sh with CONFIG_CONTAINER_NAME_SUFFIX=pyzbar, incl. ~/.isaac_ros_dev-dockerargs - but no --rm and
# --restart unless-stopped: it outlives the terminal that started it and comes back after a reboot, with everything
# installed in it; run_dev.sh's own --rm container was deleted with its terminal and took the mission stack along,
# 2026-10-24). run_dev.sh then just attaches to it (it reuses a running container). A stopped one is started again;
# then wait until its entrypoint hooks are done (install_ignition_deps.sh installs MAVROS into a new container).
# Shell in it: docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-pyzbar-container bash
set -e
C=isaac_ros_dev-aarch64-pyzbar-container
IMAGE=isaac_ros_dev-aarch64-pyzbar:latest
WS=$HOME/workspaces/isaac_ros-dev
if docker ps -a --format '{{.Names}}' | grep -qx "$C" && ! docker ps --format '{{.Names}}' | grep -qx "$C"; then
    echo "[container_up] starting the stopped $C again"
    docker start "$C" >/dev/null || docker rm -f "$C" >/dev/null
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$C"; then
    extra=()
    [ -f ~/.isaac_ros_dev-dockerargs ] && while read -r a; do [ -n "$a" ] && extra+=($(eval "echo $a")); done < ~/.isaac_ros_dev-dockerargs
    echo "[container_up] starting $C"
    docker run -d --restart unless-stopped --privileged --network host --ipc=host \
        -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/admin/.Xauthority:rw -e DISPLAY \
        -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all,nvidia.com/pva=all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e ROS_DOMAIN_ID -e USER -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \
        -e HOST_USER_UID=$(id -u) -e HOST_USER_GID=$(id -g) \
        -v /usr/bin/tegrastats:/usr/bin/tegrastats -v /tmp/:/tmp/ \
        -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra \
        -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api --pid=host \
        -v /usr/share/vpi3:/usr/share/vpi3 -v /dev/input:/dev/input "${extra[@]}" \
        -v $WS:/workspaces/isaac_ros-dev -v /etc/localtime:/etc/localtime:ro \
        --name "$C" --runtime nvidia --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
        --workdir /workspaces/isaac_ros-dev "$IMAGE" sleep infinity >/dev/null
fi
echo "[container_up] waiting for MAVROS in $C (installed by the entrypoint hook on a new container) ..."
for _ in $(seq 300); do
    docker exec "$C" bash -c 'dpkg -s ros-humble-mavros-msgs >/dev/null 2>&1' && { echo "[container_up] ready"; exit 0; }
    sleep 2
done
echo "[container_up] MAVROS still missing after 10 min - installing directly"
docker exec "$C" bash -lc 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ros-humble-mavros ros-humble-mavros-msgs'
