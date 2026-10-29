# aerothon_here_we_come

The AeroTHON **ground apps** in one self-contained folder, installable on any x86_64 Linux computer (Python
3.10–3.12) with no internet, pip or sudo. One launcher, `aerothon`, switches between them; only one runs at a time.

## The apps

| app | command | runs on | what it is for |
|---|---|---|---|
| **Onboard: the final mission** | `aerothon onboard aruco` / `qr` | this computer + the ground SiK radio | The dashboard over the telemetry radio. The Jetson is the flight computer on the drone; this shows telemetry, the map with the boards read, VIO, the geofence (captured and sent over the radio), Launch / Start and LAND / RTH / RTL / KILL. Page: http://127.0.0.1:8080 |
| **Offboard: this laptop** | `aerothon offboard aruco` / `qr` `[drop\|nodrop] [radio] [video] [target]` | this computer + SiK radio + VRX receiver | This computer is the ground station: the drone carries only the flight controller, analog camera and VTX. Reads the boards from the VRX video, maps red zones and flies the QR area (drop or not). ArUco searches at 1 m, QR at 10 m. Set the zone on the page before Start. Page: http://127.0.0.1:8094 |
| **Sim testing: the Jetson** | `aerothon sim aruco` / `qr` | the Jetson (opened here by Ethernet cable) | Lab tests: the Jetson's full dashboard with Sim (Gazebo), Offboard Jetson, Real drone, Open Gazebo and every segment. ArUco = 8084, QR = 8080. |
| **Field test: props OFF** | `aerothon fieldtest cable [jetson-ip]` / `aerothon fieldtest radio [radio]` | this computer + the Jetson (cable) or the SiK radio alone | Test the real sensors and the mission's decisions without course material or flying: the RealSense and Arducam views (cable), Detect / Scan QR, read the START QR, red zone, green / blue banner, obstacles, and the walk-throughs (QR area search, obstacle nav) with the drone carried by hand: what to yaw, where to walk, height up / down, what the QR says and whether it matches, cued by spinning the motors (props OFF). Page: http://127.0.0.1:8096 |
| **Switcher** | `aerothon switcher` | this computer | One page, http://127.0.0.1:8079, with a button for each app above (ArUco and QR) and Stop: switch without starting each app yourself. |

**Course changes (2026-10-28), in every app:** the geofence is now GPS slots **Arena (4) + QR area (4) + Extra (2)**
– the organisers' 6 points are either the arena or the QR area, the other 2 go in Extra (shown only). Arena only: the
QR area's front wall is the entry tunnel's exit line (onboard / sim: found in flight; offboard: the line through the
take-off point). The dashboard's **Course** block holds the tunnel width / height / max height inside / length / gap and
the QR-area width (defaults 3.5 m, 10 ft, 8 ft, 10 m, 0 m, 30 m – edit when the organisers change them); Save + Apply
sends all of it over the radio. Once in the QR area the drone stays inside it until the exit tunnel. A board that won't
decode: decoder settings retried 5 frames → its image compared with the start QR's → only then 0.5 m lower (not
below 5.5 m) and again. **5 m is the lowest the drone goes in the QR area, and only to lower the payload** (the search
height trim stops at 5.5 m too). The offboard page has the same GPS slots (Apply GPS QR area).

**Field test (props OFF).** Cable: the Jetson on the drone AND on this laptop's Ethernet; `aerothon fieldtest cable`,
**Start on the Jetson** (runs `field_test.sh start` there over ssh: the RealSense node, the Arducam, the detectors,
http://10.42.0.247:8097, and the Pixhawk link for the motor cues). Radio: `aerothon fieldtest radio` - Start goes up the
radio to the onboard agent. Then the buttons; **QR area**: capture its 4 corners with the drone's GPS (or *from the
geofence*), **Read START QR** over the start board, **QR area search** and walk where it says. The height is virtual
(it can't change in your hands): the page says height up / down and the motors spin faster / slower; it is what the
drone would fly (search 10 m, unsure read: 0.5 m lower, never below 5.5 m; a match: down to 5 m and the payload). Tick
*props are OFF* before **Motor cues ON**; they never run while armed.

Other commands: `aerothon stop` (close the running app), `aerothon status`, `aerothon log` (follow its log).

## Install on another computer

On the Jetson, refresh the code and pack it:

```bash
~/aerothon_here_we_come/update_code.sh     # copy the current app code from ~/workspaces/isaac_ros-dev/src
~/aerothon_here_we_come/pack.sh            # -> ~/aerothon_here_we_come.tar.gz (about 255 MB)
```

Copy `aerothon_here_we_come.tar.gz` to the other computer (USB stick), then on it:

```bash
rm -rf ~/aerothon_here_we_come
tar xzf /path/to/aerothon_here_we_come.tar.gz -C ~
~/aerothon_here_we_come/install.sh
```

`install.sh` unpacks the bundled packages into the folder (nothing system-wide), puts `aerothon` on the PATH
(`~/.local/bin`; log out and in if the command isn't found) and adds app icons: the switcher, onboard / offboard /
sim testing in ArUco and QR versions, and Stop. If it says `radio permission: MISSING`, run once
`sudo usermod -aG dialout $USER` and log out and back in.

## Launch

```bash
aerothon switcher                 # the one-page switcher
aerothon onboard aruco            # the final mission, ArUco boards (radio /dev/ttyUSB0:57600)
aerothon offboard qr drop /dev/ttyUSB0:57600 /dev/video2
aerothon sim aruco                # with an Ethernet cable to the Jetson
aerothon fieldtest cable          # field test, props OFF: Jetson cameras + guidance by cable (press Start on the page)
aerothon fieldtest radio /dev/ttyUSB0:57600   # the same over the telemetry radio alone (no video)
aerothon stop
```

QGroundControl on the same computer: SiK auto-connect **off**, UDP 14550 **on**. The apps own the radio and relay
every packet to QGC.

## A busy port

Starting an app first closes whichever AeroTHON app is running, including copies from older installs
(`~/aerothon_radio_gcs`, the old radio GCS). If something else holds port 8080 or 8094, the launcher names it and
stops there. Add `--force` to close it too, e.g. `aerothon onboard aruco --force`. It never closes a system service
(e.g. the Jetson's own `gcs-dashboard`); it prints the `systemctl --user stop …` command instead.

## Layout

```
aerothon           the launcher            switcher.py   the switcher page (standard library only)
install.sh         install on this computer
update_code.sh     Jetson only: copy the app code in      pack.sh   the tar for another computer
apps/dashboard/    gcs_server.py + static/  (onboard; also the Jetson's dashboard code)
apps/fieldtest/    field_ground.py + field_test.html + field_proto.py (the field test's laptop half)
apps/offboard/     src/gcs/vrx_explore, src/gcs/laptop_vrx, src/aerothon_nav (ArUco, fisheye), wechat_qrcode/
wheels/py3.10..12  aiohttp, pymavlink, numpy, opencv-contrib-python-headless 4.10 (x86_64)
pylib/             made by install.sh for this computer's Python
VERSION            which code this is
```
