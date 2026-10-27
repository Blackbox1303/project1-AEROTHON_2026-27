#!/bin/bash
cd ~/workspaces
OUT=isaac_ros-dev/runs/backups/aerothon_snapshot_$(date +%F_%H-%M).tar.gz
F=(isaac_ros-dev/fastdds_udp.xml isaac_ros-dev/install_ignition_deps.sh isaac_ros-dev/runs/gcs isaac_ros-dev/runs/tools
   isaac_ros-dev/sim/worlds isaac_ros-dev/src/aerothon_nav isaac_ros-dev/src/gcs isaac_ros-dev/src/launch_all
   isaac_ros-dev/src/vio_bringup isaac_ros-dev/calibration ip_launch vio_cuda_ws/sim/host_gui.sh vio_cuda_ws/sim/make_competition_world.py
   vio_cuda_ws/sim/models vio_cuda_ws/sim/run_competition.sh vio_cuda_ws/sim/worlds)
tar czf $OUT --exclude=__pycache__ -C ~/workspaces "${F[@]}" -C ~ .config/systemd/user .isaac_ros_dev-dockerargs $( [ -e ~/workspaces/lenovo_check.sh ] && echo -C ~/workspaces lenovo_check.sh ) 2>&1 | tail -3
ls -la $OUT
