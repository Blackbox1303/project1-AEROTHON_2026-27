#!/bin/bash
# Run this ONCE on your Ground Laptop (with Ethernet/Wi-Fi connection to the Jetson to copy the files & WeChat CNN models):
#   scp szyfrowac@<jetson-ip>:~/workspaces/isaac_ros-dev/src/gcs/laptop_vrx/install_laptop_vrx.sh ~/ && bash ~/install_laptop_vrx.sh [/dev/video2] [/dev/ttyUSB0:57600]
#
# After installation, you can disconnect the Jetson completely and run the Delivery-Only mission + QR detection
# 100% on your laptop using your USB VRX and Ground Telemetry Radio:
#   ~/run_vrx_delivery.sh [/dev/video2] [/dev/ttyUSB0:57600]
#   Then open http://127.0.0.1:8082 in your laptop browser!
set -e

VRX_DEV=${1:-/dev/video2}
RADIO_DEV=${2:-/dev/ttyUSB0:57600}
JETSON=${JETSON:-szyfrowac@10.42.0.247}
DIR=~/.local/share/vrx_delivery

mkdir -p "$DIR/wechat_qrcode"
echo "[1/3] Copying laptop_vrx_mission.py and WeChat QR CNN models from $JETSON ..."
scp "$JETSON:~/workspaces/isaac_ros-dev/src/gcs/laptop_vrx/laptop_vrx_mission.py" "$DIR/"
scp "$JETSON:~/workspaces/isaac_ros-dev/calibration/wechat_qrcode/"* "$DIR/wechat_qrcode/"
chmod +x "$DIR/laptop_vrx_mission.py"

echo "[2/3] Installing Python packages on laptop (pymavlink, aiohttp, pyserial, opencv-contrib-python, pyzbar) ..."
python3 -m pip install --user --quiet pymavlink aiohttp pyserial opencv-contrib-python numpy pyzbar || true

# Stop radio-gcs service if running so it doesn't lock /dev/ttyUSB0
systemctl --user stop radio-gcs.service 2>/dev/null || true

cat > ~/run_vrx_delivery.sh <<EOF
#!/bin/bash
# Launch the Laptop VRX + Ground Telemetry Delivery-Only Mission & QR Detection Dashboard:
# Usage: ~/run_vrx_delivery.sh [/dev/video2] [/dev/ttyUSB0:57600] [fov_deg]
VRX=\${1:-$VRX_DEV}
RADIO=\${2:-$RADIO_DEV}
FOV=\${3:-90.0}
systemctl --user stop radio-gcs.service 2>/dev/null || true
echo "Starting Laptop VRX Delivery GCS -> http://127.0.0.1:8082 (VRX=\$VRX, Radio=\$RADIO, FOV=\${FOV} deg)"
exec python3 "$DIR/laptop_vrx_mission.py" --vrx "\$VRX" --radio "\$RADIO" --fov-deg "\$FOV" --wechat-dir "$DIR/wechat_qrcode" --port 8082
EOF
chmod +x ~/run_vrx_delivery.sh

echo "[3/3] Installed! To run the VRX + Ground Telemetry Delivery Mission on your laptop:"
echo "  ~/run_vrx_delivery.sh $VRX_DEV $RADIO_DEV"
echo "  Then open: http://127.0.0.1:8082"
