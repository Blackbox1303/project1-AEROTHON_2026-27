#!/bin/bash
# ON THE JETSON: copy the current app code (and the offline Python packages) from the main workspace into this one.
# Run it after a code change there, then ./pack.sh for another computer.
#   ./update_code.sh [main workspace src]      default ~/workspaces/isaac_ros-dev/src
set -e
HERE=$(dirname "$(readlink -f "$0")")
SRC=${1:-$HOME/workspaces/isaac_ros-dev/src}
[ -f "$SRC/gcs/gcs_server.py" ] || { echo "no $SRC/gcs/gcs_server.py"; exit 1; }
D=$HERE/apps/dashboard O=$HERE/apps/offboard
mkdir -p "$D/static" "$O/src/gcs/vrx_explore" "$O/src/gcs/laptop_vrx" "$O/src/aerothon_nav/aerothon_nav"
cp "$SRC/gcs/gcs_server.py" "$D/"; cp "$SRC/gcs/static/index.html" "$D/static/"
cp "$SRC/gcs/vrx_explore/vrx_explore.py" "$SRC/gcs/vrx_explore/vrx_explore.html" "$O/src/gcs/vrx_explore/"
cp "$SRC/gcs/laptop_vrx/laptop_vrx_mission.py" "$O/src/gcs/laptop_vrx/"
F=$HERE/apps/fieldtest; mkdir -p "$F"                       # the field test's laptop half (+ the page)
cp "$SRC/gcs/field_test/field_ground.py" "$SRC/gcs/field_test/field_proto.py" "$SRC/gcs/field_test/field_test.html" "$F/"
cp "$SRC/aerothon_nav/aerothon_nav/__init__.py" "$SRC/aerothon_nav/aerothon_nav/aruco_tools.py" \
   "$SRC/aerothon_nav/aerothon_nav/fisheye.py" "$SRC/aerothon_nav/aerothon_nav/qr_area.py" \
   "$SRC/aerothon_nav/aerothon_nav/qr_match.py" "$O/src/aerothon_nav/aerothon_nav/"
W=$SRC/../calibration/wechat_qrcode; [ -d "$W" ] && rm -rf "$O/wechat_qrcode" && cp -r "$W" "$O/wechat_qrcode"
# the offline packages (x86_64, Python 3.10-3.12): the bundle's cache, filled if missing
CACHE=$HOME/workspaces/isaac_ros-dev/runs/gcs/lenovo_wheels
PKGS="aiohttp pymavlink numpy opencv-contrib-python-headless==4.10.0.84"   # OpenCV 4.x: the tested ArUco API
for v in 3.10 3.11 3.12; do
    ls "$CACHE/py$v"/opencv_contrib_python_headless-4*.whl >/dev/null 2>&1 && ls "$CACHE/py$v"/pymavlink-*.whl >/dev/null 2>&1 \
        || pip3 download --quiet --dest "$CACHE/py$v" --only-binary=:all: --implementation cp --python-version $v \
               --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 $PKGS
done
rm -rf "$HERE/wheels"; mkdir -p "$HERE/wheels"; cp -r "$CACHE"/py* "$HERE/wheels/"
{ git -C "$SRC/../.." log -1 --format='code %h %cd' 2>/dev/null || true; date '+copied %F %T from '"$SRC"; } > "$HERE/VERSION"
echo "updated from $SRC: $(cat "$HERE/VERSION" | tr '\n' ' ')"
