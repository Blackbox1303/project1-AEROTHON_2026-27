#!/bin/bash
# ON THE JETSON: pack the Lenovo app - no network between them in the field (user, 2026-10-28) and the Lenovo runs no
# ssh server, so it goes by USB stick, not install_lenovo_agent.sh's scp. Three ways to use it on the Lenovo:
#   radio.sh     THE FINAL MISSION: the dashboard on the ground SiK radio (the Jetson flies on the drone)
#   offboard.sh  this laptop as the OFFBOARD ground station: VRX receiver + SiK radio on the laptop, the VRX explorer
#                (vrx_explore.py, :8094) flies the QR area - no Jetson needed
#   jetson.sh    TESTING with an Ethernet cable: the Jetson's full dashboard (Sim / Offboard / Real, Open Gazebo)
#   make_radio_bundle.sh [out.tar.gz]   default ~/aerothon_radio_gcs.tar.gz - with the Python packages (aiohttp,
#                                       pymavlink, numpy, opencv-contrib-python-headless) as x86_64 wheels for
#                                       Python 3.10 / 3.11 / 3.12: no internet, pip or sudo on the Lenovo
#                                       (wheels cached in runs/gcs/lenovo_wheels)
# On the Lenovo:  tar xzf aerothon_radio_gcs.tar.gz -C ~ && ~/aerothon_radio_gcs/install.sh
set -e
GCS=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)        # src/gcs
SRC=$(cd "$GCS/.." && pwd)                                      # src
OUT=${1:-$HOME/aerothon_radio_gcs.tar.gz}
CACHE=$HOME/workspaces/isaac_ros-dev/runs/gcs/lenovo_wheels
# OpenCV 4.x: the ArUco API aruco_tools is built and tested on (pip picked 5.0 unpinned - untested); 4.10 takes numpy 2
PKGS="aiohttp pymavlink numpy opencv-contrib-python-headless==4.10.0.84"
for v in 3.10 3.11 3.12; do
    for p in aiohttp pymavlink numpy opencv_contrib_python_headless; do
        if [ -z "$(ls "$CACHE/py$v"/${p}-*.whl 2>/dev/null)" ]; then
            echo "downloading the Lenovo's Python $v packages (x86_64) ..."
            pip3 download --quiet --dest "$CACHE/py$v" --only-binary=:all: --implementation cp --python-version $v \
                --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 $PKGS
            break
        fi
    done
done
TMP=$(mktemp -d)
D=$TMP/aerothon_radio_gcs
mkdir -p "$D/static" "$D/wheels" "$D/src/gcs/vrx_explore" "$D/src/gcs/laptop_vrx" "$D/src/aerothon_nav/aerothon_nav"
cp "$GCS/gcs_server.py" "$D/"
cp "$GCS/static/index.html" "$D/static/"
cp "$GCS/vrx_explore/vrx_explore.py" "$GCS/vrx_explore/vrx_explore.html" "$D/src/gcs/vrx_explore/"
cp "$GCS/laptop_vrx/laptop_vrx_mission.py" "$D/src/gcs/laptop_vrx/"
cp "$SRC/aerothon_nav/aerothon_nav/__init__.py" "$SRC/aerothon_nav/aerothon_nav/aruco_tools.py" \
   "$SRC/aerothon_nav/aerothon_nav/fisheye.py" "$D/src/aerothon_nav/aerothon_nav/"
WECHAT=$HOME/workspaces/isaac_ros-dev/calibration/wechat_qrcode
[ -d "$WECHAT" ] && cp -r "$WECHAT" "$D/wechat_qrcode"           # QR boards offboard (ArUco needs none)
cp -r "$CACHE"/py* "$D/wheels/"

cat > "$D/env.sh" <<'EOF'
# sourced by the launchers: this bundle's Python packages for this Python
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ -d "$HERE/pylib/py$PYV" ] || { echo "run $HERE/install.sh first"; exit 1; }
export PYTHONPATH="$HERE/pylib/py$PYV${PYTHONPATH:+:$PYTHONPATH}"
EOF

cat > "$D/install.sh" <<'EOF'
#!/bin/bash
# ON THE LENOVO, once per new bundle: unpack the Python packages for this Python (no internet, pip or sudo), add the
# app icons, check the radio permission.
set -e
HERE=$(dirname "$(readlink -f "$0")")
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ARCH=$(uname -m)
[ "$ARCH" = x86_64 ] || { echo "this bundle is for x86_64, this PC is $ARCH"; exit 1; }
[ -d "$HERE/wheels/py$PYV" ] || { echo "no packages for Python $PYV in this bundle (has: $(ls "$HERE/wheels"))"; exit 1; }
LIB=$HERE/pylib/py$PYV
rm -rf "$LIB"; mkdir -p "$LIB"
echo "unpacking the Python $PYV packages ..."
python3 - "$HERE/wheels/py$PYV" "$LIB" <<'PY'
import sys, zipfile, glob
src, dst = sys.argv[1], sys.argv[2]
for w in sorted(glob.glob(src + '/*.whl')):
    zipfile.ZipFile(w).extractall(dst)
PY
PYTHONPATH=$LIB python3 -s -c '
import aiohttp, cv2, numpy
from pymavlink import mavutil
print("packages OK: aiohttp", aiohttp.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__,
      "| aruco", hasattr(cv2, "aruco"))'
mkdir -p ~/.local/share/applications
app() {   # app <file> <name> <command> <comment>
    cat > ~/.local/share/applications/$1.desktop <<DESK
[Desktop Entry]
Type=Application
Name=$2
Comment=$4
Exec=bash -c '$3; echo; read -p "stopped - Enter closes this window"'
Terminal=true
Categories=Utility;
DESK
}
app aerothon-radio "AeroTHON final mission (radio, ArUco)" "$HERE/radio.sh aruco" "The dashboard on the ground SiK radio"
app aerothon-offboard "AeroTHON offboard on this laptop (ArUco)" "$HERE/offboard.sh aruco drop" "VRX + SiK radio on this laptop"
app aerothon-jetson "AeroTHON Jetson dashboard (testing, cable)" "$HERE/jetson.sh 8084" "Sim / Offboard / Real on the Jetson, by Ethernet"
echo "app icons: final mission (radio) | offboard on this laptop | Jetson dashboard (testing)"
if id -nG | tr ' ' '\n' | grep -qx dialout; then
    echo "radio permission: OK (in the dialout group)"
else
    echo "radio permission: MISSING - run once:  sudo usermod -aG dialout $USER   then log out and back in"
fi
echo "done. Final mission:  $HERE/radio.sh aruco      (QGC: SiK auto-connect OFF, UDP 14550 ON)"
EOF

cat > "$D/radio.sh" <<'EOF'
#!/bin/bash
# THE FINAL MISSION, on the Lenovo: the dashboard on the ground SiK radio (the only link to the drone).
#   radio.sh [aruco|qr] [radio device]      defaults: aruco, /dev/ttyUSB0:57600 -> http://127.0.0.1:8080
# QGC: UDP 14550 auto-connect on, SiK auto-connect OFF (this program owns the radio and relays every packet to QGC).
source "$(dirname "$(readlink -f "$0")")/env.sh"
MARKER=${1:-aruco}
DEV=${2:-/dev/ttyUSB0:57600}
[ -e "${DEV%%:*}" ] || echo "WARNING: ${DEV%%:*} not found - plug the ground radio in (ls /dev/ttyUSB* /dev/serial/by-id/)"
systemctl --user stop radio-gcs 2>/dev/null       # the old 8081 Radio GCS would hold the radio
( sleep 3; xdg-open http://127.0.0.1:8080 >/dev/null 2>&1 ) &
echo "radio dashboard ($MARKER boards) on $DEV: http://127.0.0.1:8080 - Ctrl-C stops it"
exec python3 -s "$HERE/gcs_server.py" --radio "$DEV" --port 8080 --marker "$MARKER"
EOF

cat > "$D/offboard.sh" <<'EOF'
#!/bin/bash
# THIS LAPTOP AS THE OFFBOARD GROUND STATION: the drone carries only the flight controller + analog camera + VTX; the
# VRX receiver (USB video) and the ground SiK radio are plugged into this laptop. The VRX explorer reads the boards,
# maps red zones, explores the QR area (A* + frontiers, the batch / yaw-scan / scout rules) and drops or not.
#   offboard.sh [aruco|qr] [drop|nodrop] [radio device] [VRX video device] [target code]
#   defaults: aruco, drop, /dev/ttyUSB0:57600, '' (pick it on the page), '' (read the start board at take-off)
#   search height: ArUco 1 m (8084), QR 10 m (8080). Page: http://127.0.0.1:8094 - set the ZONE there (x / y in
#   metres from where the autopilot's EKF started) before Start. QGC on this laptop gets the telemetry (UDP 14550).
source "$(dirname "$(readlink -f "$0")")/env.sh"
MARKER=${1:-aruco}
DROP=${2:-drop}
DEV=${3:-/dev/ttyUSB0:57600}
VIDEO=${4:-}
TARGET=${5:-}
ALT=1.0; [ "$MARKER" = qr ] && ALT=10.0
ARGS=(--link "$DEV" --alt "$ALT" --marker "$MARKER" --qgc 127.0.0.1:14550 --port 8094)
[ "$DROP" = drop ] && ARGS+=(--drop)
[ -n "$VIDEO" ] && ARGS+=(--video "$VIDEO")
[ -n "$TARGET" ] && ARGS+=(--start-qr "$TARGET")
[ -d "$HERE/wechat_qrcode" ] && ARGS+=(--wechat-dir "$HERE/wechat_qrcode")
[ -e "${DEV%%:*}" ] || echo "WARNING: ${DEV%%:*} not found - plug the ground radio in (ls /dev/ttyUSB* /dev/serial/by-id/)"
systemctl --user stop radio-gcs 2>/dev/null
( sleep 4; xdg-open http://127.0.0.1:8094 >/dev/null 2>&1 ) &
echo "offboard explorer ($MARKER boards, $DROP, search at $ALT m) on $DEV: http://127.0.0.1:8094 - Ctrl-C stops it"
exec python3 -s "$HERE/src/gcs/vrx_explore/vrx_explore.py" "${ARGS[@]}"
EOF

cat > "$D/jetson.sh" <<'EOF'
#!/bin/bash
# TESTING ONLY, with an Ethernet cable to the Jetson: its full dashboard - Sim (Gazebo), Offboard Jetson, Real drone,
# Open Gazebo. The final mission never uses this (radio.sh).
#   jetson.sh [8084|8080]      8084 = ArUco boards (default), 8080 = QR codes
PORT=${1:-8084}
URL=http://10.42.0.247:$PORT
if curl -s -m 3 -o /dev/null "$URL/"; then
    echo "opening $URL"
    xdg-open "$URL" >/dev/null 2>&1 &
elif ping -c 1 -W 2 10.42.0.247 >/dev/null 2>&1; then
    echo "the Jetson answers but its $PORT dashboard is not up: on the Jetson run"
    [ "$PORT" = 8084 ] && echo "  ~/workspaces/isaac_ros-dev/src/launch_all/scripts/gcs_aruco.sh start" \
                       || echo "  systemctl --user restart gcs-dashboard"
else
    echo "no link to the Jetson (10.42.0.247): for testing, plug an Ethernet cable between this laptop and the Jetson."
    echo "The final mission needs no cable: use radio.sh."
fi
EOF
chmod +x "$D"/*.sh
{ git -C "$SRC/../.." log -1 --format='code %h %cd' 2>/dev/null || true; date '+packed %F %T on the Jetson'; } > "$D/VERSION"
tar czf "$OUT" -C "$TMP" aerothon_radio_gcs
rm -rf "$TMP"
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)) - copy it to a USB stick; on the Lenovo:"
echo "  rm -rf ~/aerothon_radio_gcs && tar xzf aerothon_radio_gcs.tar.gz -C ~ && ~/aerothon_radio_gcs/install.sh"
