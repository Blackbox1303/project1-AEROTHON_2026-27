#!/bin/bash
# Install the AeroTHON ground apps on THIS computer (x86_64 Linux, Python 3.10-3.12): no internet, pip or sudo.
#   ./install.sh
# Unpacks the bundled Python packages into pylib/ (nothing system-wide), puts `aerothon` on the PATH
# (~/.local/bin), adds 7 app icons (the switcher + onboard / offboard / sim testing, each ArUco and QR), checks the
# radio permission. Run it again after copying a newer workspace over this one.
set -e
HERE=$(dirname "$(readlink -f "$0")")
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ARCH=$(uname -m)
[ "$ARCH" = x86_64 ] || { echo "the bundled packages are for x86_64, this computer is $ARCH"; exit 1; }
[ -d "$HERE/wheels/py$PYV" ] || { echo "no packages for Python $PYV (has: $(ls "$HERE/wheels"))"; exit 1; }
LIB=$HERE/pylib/py$PYV
rm -rf "$LIB"; mkdir -p "$LIB"
echo "unpacking the Python $PYV packages ..."
python3 - "$HERE/wheels/py$PYV" "$LIB" <<'PY'
import sys, zipfile, glob
for w in sorted(glob.glob(sys.argv[1] + '/*.whl')):
    zipfile.ZipFile(w).extractall(sys.argv[2])
PY
PYTHONPATH=$LIB python3 -s -c '
import aiohttp, cv2, numpy
from pymavlink import mavutil
print("packages OK: aiohttp", aiohttp.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__,
      "| aruco", hasattr(cv2, "aruco"))'
chmod +x "$HERE/aerothon" "$HERE"/*.sh
mkdir -p ~/.local/bin && ln -sf "$HERE/aerothon" ~/.local/bin/aerothon
mkdir -p ~/.local/share/applications
rm -f ~/.local/share/applications/aerothon-*.desktop          # the earlier bundle's icons
icon() {   # icon <id> <name> <aerothon args> <comment> [terminal]
    cat > ~/.local/share/applications/aerothon-$1.desktop <<DESK
[Desktop Entry]
Type=Application
Name=$2
Comment=$4
Exec=$HERE/aerothon $3
Terminal=${5:-false}
Categories=Utility;
DESK
}
icon switcher        "AeroTHON switcher"                     "switcher"          "One page to switch between all the apps"
icon onboard-aruco   "AeroTHON onboard - final mission, ArUco" "onboard aruco"   "The dashboard on the ground SiK radio"
icon onboard-qr      "AeroTHON onboard - final mission, QR"    "onboard qr"      "The dashboard on the ground SiK radio"
icon offboard-aruco  "AeroTHON offboard laptop, ArUco"        "offboard aruco"   "VRX + SiK radio on this laptop, search at 1 m"
icon offboard-qr     "AeroTHON offboard laptop, QR"           "offboard qr"      "VRX + SiK radio on this laptop, search at 10 m"
icon sim-aruco       "AeroTHON sim testing, ArUco (Jetson 8084)" "sim aruco"     "The Jetson's dashboard by Ethernet"
icon sim-qr          "AeroTHON sim testing, QR (Jetson 8080)"    "sim qr"        "The Jetson's dashboard by Ethernet"
icon fieldtest-cable "AeroTHON field test (cable, props off)"  "fieldtest cable"  "Jetson cameras + guidance by Ethernet"
icon fieldtest-radio "AeroTHON field test (radio, props off)"  "fieldtest radio"  "Jetson scans + guidance over the SiK radio"
icon stop            "AeroTHON stop the running app"          "stop"             "Close whichever AeroTHON app is running"
# THE FINAL MISSION, one click, auto (user, 2026-10-30): a terminal window with the countdown (Ctrl-C there cancels),
# left open after it ends so its last lines can be read
cat > ~/.local/share/applications/aerothon-mission.desktop <<DESK
[Desktop Entry]
Type=Application
Name=AeroTHON FINAL MISSION (auto)
Comment=Radio dashboard + geofence, Launch stack, preflight, 10 s countdown, Start - Ctrl-C in its window cancels
Exec=bash -c '"$HERE/aerothon" mission; echo; read -p "[aerothon] press Enter to close this window"'
Terminal=true
Categories=Utility;
DESK
echo "app icons: FINAL MISSION (auto), switcher, onboard / offboard / sim testing x ArUco / QR, field test cable / radio, stop"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "note: ~/.local/bin is not on PATH yet - log out and in, or use $HERE/aerothon";; esac
if id -nG | tr ' ' '\n' | grep -qx dialout; then
    echo "radio permission: OK (in the dialout group)"
else
    echo "radio permission: MISSING - run once:  sudo usermod -aG dialout $USER   then log out and back in"
fi
echo "done. Start with:  aerothon switcher      (or: aerothon onboard aruco)"
