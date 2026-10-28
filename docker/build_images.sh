#!/bin/bash
# Build the two container images on a Jetson (aarch64, JetPack 6 / Isaac ROS 3.2):
#   isaac_ros_dev-aarch64-pyzbar:latest   the mission container (container_up.sh starts it as
#                                         isaac_ros_dev-aarch64-pyzbar-container; competition-container runs the same image)
#   aerothon_vio:latest                   the AeroTHON corridor VIO container (cuVSLAM, MAVROS, RealSense), for corridor:=aerothon
# Needs isaac_ros_common cloned at $ISAAC_ROS_COMMON (default ~/workspaces/isaac_ros-dev/src/isaac_ros_common):
#   git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
#   docker/build_images.sh [mission|vio|all]        default all
set -e
HERE=$(dirname "$(readlink -f "$0")")
IRC=${ISAAC_ROS_COMMON:-$HOME/workspaces/isaac_ros-dev/src/isaac_ros_common}
[ -x "$IRC/scripts/build_image_layers.sh" ] || { echo "isaac_ros_common not found at $IRC (set ISAAC_ROS_COMMON)"; exit 1; }
what=${1:-all}
if [ "$what" = all ] || [ "$what" = mission ]; then
    echo "== the Isaac ROS base (ROS 2 Humble + RealSense) -> isaac_ros_dev-aarch64"
    "$IRC/scripts/build_image_layers.sh" --image_key aarch64.ros2_humble.realsense --image_name isaac_ros_dev-aarch64
    echo "== + libzbar / pyzbar -> isaac_ros_dev-aarch64-pyzbar"
    docker build -t isaac_ros_dev-aarch64-pyzbar:latest --build-arg BASE_IMAGE=isaac_ros_dev-aarch64 \
        -f "$HERE/Dockerfile.pyzbar" "$HERE"
fi
if [ "$what" = all ] || [ "$what" = vio ]; then
    echo "== + the VIO layer (MAVROS, cuVSLAM, RealSense, Nav2) -> aerothon_vio"
    cp "$HERE/Dockerfile.vio" "$IRC/docker/Dockerfile.vio"       # isaac_ros_common builds layer <key> from Dockerfile.<key>
    "$IRC/scripts/build_image_layers.sh" --image_key aarch64.ros2_humble.realsense.vio --image_name aerothon_vio
fi
echo "done. Start the mission container:  isaac_ros-dev/src/launch_all/scripts/container_up.sh"
