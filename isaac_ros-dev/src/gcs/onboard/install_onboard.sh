#!/bin/bash
# ON THE JETSON, once (and again to change the Pixhawk link): the radio-only ground station's onboard half.
#   bash install_onboard.sh [link]     link: /dev/ttyACM0:921600 (the Pixhawk's USB port 1, the default)
# Installs two systemd --user services (they start at boot, like the dashboard - linger is on):
#   aerothon-router     mavlink-router on the Pixhawk link AND the sim's PX4 (UDP 14540), ALWAYS on: MAVROS 14640,
#                       onboard dashboard 14650, mav_agent 14660 (its fallback), field_link 14662, QGC 10.42.0.1:14550
#                       + any QGC on UDP 14551. sim_mission / real_mission then start no router of their own.
#   aerothon-mav-agent  onboard/mav_agent.py on the Pixhawk's USB port 2 (-if02): radio commands -> the onboard
#                       dashboard, status -> the radio
# The radio is on the PIXHAWK (GPS2 = SERIAL4, MAVLink 2, 57600), not the Jetson: ArduPilot routes between its ports
# (TELEM1 ESC telemetry, TELEM2 ELRS, GPS1 GPS). Nothing to set on the Pixhawk for it.
set -e
LINK=${1:-/dev/ttyACM0:921600}
DEV=${LINK%%:*}; BAUD=${LINK##*:}
GCS=$HOME/workspaces/isaac_ros-dev/src/gcs
CONF=$HOME/workspaces/isaac_ros-dev/runs/gcs/onboard_router.conf
mkdir -p "$(dirname "$CONF")" ~/.config/systemd/user
cat > "$CONF" <<CFG
# written by gcs/onboard/install_onboard.sh ($(date)): the onboard, always-on mavlink-router
[General]
TcpServerPort = 0
ReportStats = false

[UartEndpoint fcu]
Device = $DEV
Baud = $BAUD

# the SIM's PX4 (SITL sends to 14540): the same router serves Gazebo runs - sim_mission.launch.py starts none of its
# own while this one runs. Don't run the sim with a real Pixhawk connected (two system-1 autopilots)
[UdpEndpoint px4_sitl]
Mode = Server
Address = 0.0.0.0
Port = 14540

[UdpEndpoint lenovo_qgc]
Mode = Normal
Address = 10.42.0.1
Port = 14550

[UdpEndpoint mavros]
Mode = Normal
Address = 127.0.0.1
Port = 14640

[UdpEndpoint gcs_dashboard]
Mode = Normal
Address = 127.0.0.1
Port = 14650

[UdpEndpoint mav_agent]
Mode = Normal
Address = 127.0.0.1
Port = 14660

[UdpEndpoint qgc_any]
Mode = Server
Address = 0.0.0.0
Port = 14551

# gcs/field_test/field_link.py (the field test, props off: GPS / heading in, motor cues out)
[UdpEndpoint field_test]
Mode = Normal
Address = 127.0.0.1
Port = 14662

# a stand-in for the radio, to test the Radio GCS on the Jetson itself: radio_gcs.py --radio udpin:0.0.0.0:14670
[UdpEndpoint radio_test]
Mode = Normal
Address = 127.0.0.1
Port = 14670
CFG
echo "$LINK" > "$HOME/workspaces/isaac_ros-dev/runs/gcs/onboard_link"
cat > ~/.config/systemd/user/aerothon-router.service <<UNIT
[Unit]
Description=AeroTHON onboard mavlink-router (Pixhawk $LINK, always on)

[Service]
ExecStart=/usr/bin/mavlink-routerd -c $CONF
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNIT
cat > ~/.config/systemd/user/aerothon-mav-agent.service <<UNIT
[Unit]
Description=AeroTHON onboard MAVLink agent (radio commands -> the onboard dashboard)
After=aerothon-router.service gcs-dashboard.service

[Service]
ExecStart=/usr/bin/python3 $GCS/onboard/mav_agent.py --mavlink auto --link $LINK
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now aerothon-router.service aerothon-mav-agent.service
systemctl --user restart aerothon-router.service aerothon-mav-agent.service
sleep 2
systemctl --user --no-pager --lines=0 status aerothon-router.service aerothon-mav-agent.service | grep -E "service|Active"
echo "onboard radio GCS half installed (Pixhawk on $LINK)"
