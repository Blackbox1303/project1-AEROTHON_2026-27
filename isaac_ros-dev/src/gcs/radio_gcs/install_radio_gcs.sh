#!/bin/bash
# ON THE LENOVO, once (Ethernet cable to the Jetson connected for the copy):
#   scp szyfrowac@10.42.0.247:~/workspaces/isaac_ros-dev/src/gcs/radio_gcs/install_radio_gcs.sh ~/ && bash ~/install_radio_gcs.sh [radio]
#   radio: the ground telemetry radio's port and baud, default /dev/ttyUSB0:57600 (SiK / Holybro on USB; some show as
#          /dev/ttyACM0 - plug it in and run `ls /dev/serial/by-id/` to see)
# Installs the Radio GCS (http://127.0.0.1:8081) as a systemd --user service that starts at login, plus
# ~/radio_gcs.sh to run it by hand. QGroundControl keeps working beside it: it relays the radio to QGC's UDP 14550 -
# but QGC must NOT open the radio itself: QGC -> Application Settings -> General -> AutoConnect: untick "SiK Radio"
# (and "Pixhawk"), keep "UDP".
set -e
RADIO=${1:-/dev/ttyUSB0:57600}
JETSON=${JETSON:-szyfrowac@10.42.0.247}
DIR=~/.local/share/radio_gcs
mkdir -p "$DIR" ~/.config/systemd/user
scp "$JETSON:~/workspaces/isaac_ros-dev/src/gcs/radio_gcs/{radio_gcs.py,radio.html,qgc_mavlink_actions.json}" "$DIR/"
python3 -m pip install --user --quiet pymavlink aiohttp pyserial
id -nG | grep -qw dialout || echo "!! add yourself to dialout for the radio port: sudo usermod -aG dialout $USER (then log out / in)"
cat > ~/radio_gcs.sh <<RUN
#!/bin/bash
# Radio GCS by hand (stop the service first: systemctl --user stop radio-gcs): ~/radio_gcs.sh [/dev/ttyUSB0:57600]
exec python3 $DIR/radio_gcs.py --radio "\${1:-$RADIO}"
RUN
chmod +x ~/radio_gcs.sh
cat > ~/.config/systemd/user/radio-gcs.service <<UNIT
[Unit]
Description=AeroTHON Radio GCS (telemetry radio $RADIO -> http://127.0.0.1:8081, relayed to QGC UDP 14550)

[Service]
ExecStart=/usr/bin/python3 $DIR/radio_gcs.py --radio $RADIO
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now radio-gcs.service
# QGC's buttons file (optional, option A): QGC's save folder is under the Documents folder. Not fatal - a broken
# ~/Documents (a symlink loop on the T14, 2026-10-26) stopped the install here
DOCS=$(xdg-user-dir DOCUMENTS 2>/dev/null || echo "$HOME/Documents")
if mkdir -p "$DOCS/QGroundControl/MavlinkActions" 2>/dev/null; then
    cp "$DIR/qgc_mavlink_actions.json" "$DOCS/QGroundControl/MavlinkActions/aerothon.json"
else
    echo "note: QGC buttons file not installed ($DOCS not usable) - it is in $DIR/qgc_mavlink_actions.json"
fi
sleep 2; systemctl --user --no-pager --lines=3 status radio-gcs.service | head -5
echo "Radio GCS: open http://127.0.0.1:8081 (radio $RADIO). Change the radio: bash ~/install_radio_gcs.sh <port:baud>"
