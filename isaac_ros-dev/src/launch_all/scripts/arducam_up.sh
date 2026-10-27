#!/bin/bash
# The real Arducam (CSI) in the dev container, detached: /arducam/image_raw + camera_info + TF.
# Log: runs/arducam_real/arducam.log. Stop: scripts/arducam_down.sh. Args go to arducam_real.launch.py (sensor_id:=1 ...)
C=isaac_ros_dev-aarch64-pyzbar-container
systemctl --user stop arducam-calib 2>/dev/null   # the calibration page holds the camera: one process at a time
docker exec -d -u admin $C bash -c "source /opt/ros/humble/setup.bash; source /workspaces/isaac_ros-dev/install/setup.bash; \
  export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_udp.xml; unset DISPLAY; \
  exec ros2 launch vio_bringup arducam_real.launch.py $* > /workspaces/isaac_ros-dev/runs/arducam_real/arducam.log 2>&1"
echo "arducam launch started in $C (log runs/arducam_real/arducam.log)"
