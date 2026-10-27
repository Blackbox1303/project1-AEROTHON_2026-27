#!/bin/bash
# ON THE LENOVO, once (and again after changes on the Jetson): the dashboard's Gazebo GUI button needs this helper (a
# systemd --user service on 127.0.0.1:8765) and an ssh key to the Jetson (host_gui.sh syncs models, `aerothon` starts
# the Jetson's dashboard - both without a password prompt).
#   scp szyfrowac@10.42.0.247:~/workspaces/isaac_ros-dev/src/gcs/lenovo/install_lenovo_agent.sh ~/ && bash ~/install_lenovo_agent.sh
# Installs: ~/.local/bin/gcs_gui_agent.py (the helper), ~/host_gui.sh (Gazebo GUI), ~/.local/bin/aerothon (start the
# Jetson dashboard / launch / stop from here: `aerothon up`), an "AeroTHON GCS" app launcher (= aerothon up),
# ~/.local/share/aerothon_gcs (the final flight's radio dashboard: `aerothon radio`).
set -e
JETSON=szyfrowac@10.42.0.247
SRC=/home/szyfrowac/workspaces/isaac_ros-dev/src
mkdir -p ~/.local/bin ~/.config/systemd/user ~/.local/share/applications
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
ssh -o BatchMode=yes -o ConnectTimeout=5 $JETSON true 2>/dev/null \
    || ssh-copy-id -i ~/.ssh/id_ed25519.pub $JETSON          # asks the Jetson password one last time
scp $JETSON:$SRC/gcs/lenovo/gcs_gui_agent.py ~/.local/bin/gcs_gui_agent.py
scp $JETSON:$SRC/gcs/lenovo/aerothon.sh ~/.local/bin/aerothon
scp $JETSON:/home/szyfrowac/workspaces/vio_cuda_ws/sim/host_gui.sh ~/host_gui.sh
# the final flight's ground dashboard (`aerothon radio`): the same 8080 server + page, run here on the SiK radio
mkdir -p ~/.local/share/aerothon_gcs/static
scp $JETSON:$SRC/gcs/gcs_server.py ~/.local/share/aerothon_gcs/gcs_server.py
scp $JETSON:$SRC/gcs/static/index.html ~/.local/share/aerothon_gcs/static/index.html
python3 -c 'import aiohttp, pymavlink' 2>/dev/null || pip3 install --user aiohttp pymavlink
chmod +x ~/.local/bin/gcs_gui_agent.py ~/.local/bin/aerothon ~/host_gui.sh
cat > ~/.config/systemd/user/gcs-gui-agent.service <<UNIT
[Unit]
Description=AeroTHON ground station: Gazebo GUI + Jetson start helper (127.0.0.1:8765)

[Service]
ExecStart=/usr/bin/python3 %h/.local/bin/gcs_gui_agent.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
cat > ~/.local/share/applications/aerothon-gcs.desktop <<DESK
[Desktop Entry]
Type=Application
Name=AeroTHON GCS
Comment=Start the Jetson ground station dashboard if it is down, then open it
Exec=bash -c '$HOME/.local/bin/aerothon up || (echo; read -p "press Enter")'
Terminal=true
Icon=applications-science
DESK
systemctl --user daemon-reload
systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS || true
systemctl --user enable gcs-gui-agent.service
systemctl --user restart gcs-gui-agent.service
sleep 1; curl -s http://127.0.0.1:8765/status && echo "  <- helper up"
echo "done. Dashboard: http://10.42.0.247:8080 (Open Gazebo there). Jetson dashboard down? run 'aerothon up',"
echo "the 'AeroTHON GCS' app, or open http://127.0.0.1:8765 here."
