# AeroTHON: addresses, start-up and launch commands

Everything runs on the **Jetson** (onboard the drone) unless it says **Lenovo**. Run the commands in a normal terminal
on the Jetson host, not in the container. Lab link: Jetson `10.42.0.247`, Lenovo `10.42.0.1` (USB Ethernet cable).

## Addresses

| What | Where | Starts at boot? |
|---|---|---|
| Onboard dashboard: sim + real, geofence, preflight, mission | http://10.42.0.247:8080 (Lenovo browser, cable) | yes |
| Arducam calibration + QR test page | http://10.42.0.247:8090 | **no**: start it, below |
| Tunnel + SLAM dashboard (Intel RealSense D435 depth + 2D/3D SLAM) | http://10.42.0.247:8093 (or http://127.0.0.1:8093) | **no**: start it, below |
| VRX Explore: QR scanning + frontier/A* round red zones at 1 m (PX4 SITL now) | http://10.42.0.247:8094 | **no**: start it, below |
| Radio GCS: the drone over the telemetry radio only | http://127.0.0.1:8081 **on the Lenovo** | yes (at Lenovo login) |
| QGroundControl | UDP 14550 on the Lenovo (relayed by the Radio GCS) | - |

## What starts at boot (Jetson), and how to start it if it didn't

All three are systemd `--user` services, which run at boot without a login (linger is on).

```bash
# check
systemctl --user status gcs-dashboard aerothon-router aerothon-mav-agent --no-pager | grep -E "service|Active"

# restart one, or all three
systemctl --user restart gcs-dashboard        # the dashboard, :8080
systemctl --user restart aerothon-router      # mavlink-router: Pixhawk (/dev/ttyTHS1) + the sim's PX4 -> everything
systemctl --user restart aerothon-mav-agent   # radio commands from the Lenovo -> the dashboard
systemctl --user restart gcs-dashboard aerothon-router aerothon-mav-agent

# why did it fail?
journalctl --user -u gcs-dashboard -n 30 --no-pager       # or aerothon-router / aerothon-mav-agent
tail -30 ~/workspaces/isaac_ros-dev/runs/gcs/server.log   # the dashboard's own log
```

If `systemctl --user` itself fails, or a service is missing, start them by hand instead:

```bash
~/workspaces/isaac_ros-dev/src/gcs/run_gcs.sh &                                                  # dashboard
mavlink-routerd -c ~/workspaces/isaac_ros-dev/runs/gcs/onboard_router.conf &                     # router
python3 ~/workspaces/isaac_ros-dev/src/gcs/onboard/mav_agent.py --link /dev/ttyTHS1:921600 &    # radio agent

# reinstall the router + agent services (also: change the Pixhawk link, e.g. /dev/ttyACM0:921600 for USB)
bash ~/workspaces/isaac_ros-dev/src/gcs/onboard/install_onboard.sh /dev/ttyTHS1:921600
```

The dev container (`isaac_ros_dev-aarch64-pyzbar-container`) restarts by itself at boot, and **Launch stack** starts it
when it isn't running. By hand: `~/workspaces/isaac_ros-dev/src/launch_all/scripts/container_up.sh` (check: `docker ps`).

## Arducam calibration + QR test page (:8090), started by hand

```bash
~/workspaces/isaac_ros-dev/src/launch_all/scripts/arducam_calib.sh          # start  -> http://10.42.0.247:8090
~/workspaces/isaac_ros-dev/src/launch_all/scripts/arducam_calib.sh stop     # stop
```

* **Calibrate: capture** -> move the checkerboard (link on the page) over the whole image -> **Compute calibration**
  -> `calibration/arducam_fisheye_1640x1232.yaml` (the real stack uses it by itself) -> **Undistorted** to check.
* **QR test**: decodes live at full resolution (3280 x 2464) with the 5 ms exposure, like the mission's QR scan, on
  a straightened (non-fisheye) view of the middle of the lens - **QR view** buttons: 60 / 90 (default) / 120 deg or
  the whole fisheye - with OpenCV's WeChat decoder (much better on small / noisy codes) and zbar as backup.
  Indoors, start it with a longer exposure: `arducam_calib.sh start --qr-max-exposure-ms 15` (plain `start` = 5 ms).
  Reads are logged in `~/workspaces/isaac_ros-dev/calibration/qr_reads.log`.
* **Detect QRs**: bounding boxes only, no reading - green = WeChat's trained QR detection network (confidence),
  orange = found by the code's three corner squares, magenta = the mission's contrast detector's guess.
  **QR test (read)**: reads codes; a box it can't read is chased: 30 deg x2 zoom (3 frames averaged) -> 15 deg x3
  (5 frames) -> a longer exposure (12 ms), following the box; back to wide once read or gone.
* Calibration: open the page, click **How to calibrate the camera** - the steps are written there.
* Only one program can use the camera: stop this page before the ROS camera / the real stack. Both stop it by
  themselves, and the real stack starts it again when it ends.
* Why didn't it start: `systemctl --user status arducam-calib --no-pager`, `tail -20 ~/workspaces/isaac_ros-dev/runs/arducam_real/calib.log`

ROS camera alone (`/arducam/image_raw` for the dashboard's camera panel, no mission):

```bash
~/workspaces/isaac_ros-dev/src/launch_all/scripts/arducam_up.sh             # add force_qr:=true to test the QR mode
~/workspaces/isaac_ros-dev/src/launch_all/scripts/arducam_down.sh
```

**QR mode:** while the mission reads the start QR, searches the QR area and drops (`READ_START_QR`,
`SEARCH_DELIVERY`, `DELIVER`), the camera runs at full resolution (3280 x 2464 @ 21 fps) with its exposure capped at
5 ms instead of 20. `/arducam/image_raw` stays 1640 px colour (red zones), and the full frame goes in grey to the QR
readers on `/arducam/qr/image_raw` (~4 Hz). The rest of the flight is 1640 px, 20 ms.

## Tunnel + SLAM dashboard (:8093), started by hand

D435 + cuVSLAM in the Isaac container, 2D SLAM (slam_toolbox), RTAB-Map 3D map, RViz, and cuVSLAM -> PX4 when the
Pixhawk is connected. Open http://10.42.0.247:8093 (Launch / Stop / RViz / RTAB-Map / Save map / Fly drone in sim /
Reset sim, plus map, camera and stats).

```bash
S=~/workspaces/isaac_ros-dev/src/launch_all/scripts/tunnel_slam.sh
$S start                 # or restart; RViz + RTAB-Map windows open on the Jetson display
$S stop                  # stops the stack; RViz / RTAB-Map windows stay open until you close them
$S status                # what is running
$S log                   # follow the console log
$S start link:=none      # camera + SLAM only, no Pixhawk        ($S start nav:=true: with the flight nodes)
# from the Lenovo:
ssh szyfrowac@10.42.0.247 '~/workspaces/isaac_ros-dev/src/launch_all/scripts/tunnel_slam.sh start'
```
- Camera is up ~20-60 s after start (the D435 is USB-reset and re-synced into the container each time).
- **Save map** writes `slam_map.pgm/.yaml` (2D costmap), the slam_toolbox pose graph, the 3D points and `rtabmap.db`
  to `runs/saved_tunnel_map/` and `runs/maps/<time>/`; explore a saved one with
  `rtabmap-databaseViewer runs/saved_tunnel_map/rtabmap.db`. **Fly drone in sim** stops the live stack and flies a
  simulated drone on the saved map (click the map to send it).
- PX4 only fuses the VIO with `EKF2_EV_CTRL` bit 2 set (`vio_bringup/config/px4_6c_vio_params.txt`); the page shows it.

## VRX Explore (:8094), started by hand

QR scanning (the 8090 / laptop VRX decoder) on video from a USB device (VRX receiver; any webcam until the analog
camera exists), frontier exploration with A* round red zones, at 1 m, commands to PX4 over MAVLink. Runs on the
Jetson as the ground station; the PX4 is the SIH SITL (no Gazebo) for now.

```bash
X=~/workspaces/isaac_ros-dev/src/launch_all/scripts/vrx_explore.sh
$X start                       # SIH SITL + the app -> http://10.42.0.247:8094
$X start --video /dev/video2   # with a video device (or pick it on the page)
$X stop | status | log | sitl-log
```
- Page: Start explore, Return home (A* path, then land), Hold, Land, Kill; Draw red zone (drag on the map) to test
  avoidance without a camera; zone (m from home, x east / y north) and video device below the map.
- Without a camera it still explores (coverage from the pose) and avoids drawn red zones; QRs need video.
- Later on the real drone: `--link /dev/ttyUSB0:57600` (telemetry radio) and no SITL.

## Sim (Gazebo)

From the dashboard (http://10.42.0.247:8080): **🖥 Gazebo** -> world -> **Launch stack** -> wait for **READY**
(~1.5 min) -> **▶ Full mission**. Next test: **⟲ Reset sim + Gazebo + mission** once (don't press Stop after it).
Gazebo GUI on the Lenovo: **Gazebo GUI -> Open** (needs the Lenovo helper), or `~/host_gui.sh --follow` on the Lenovo.

From a terminal instead (not both at once). It is a ROS launch file: run it with `ros2 launch`, not `./`:

```bash
ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/sim_mission.launch.py                  # world:=competition
ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/sim_mission.launch.py world:=comp_far1 # far target
```

It starts the container, Gazebo, PX4, MAVROS and the mission stack, and starts the mission by itself. Ctrl-C stops
everything. Logs: `~/workspaces/isaac_ros-dev/runs/<world>_<date>/`.

## Real drone, with the Ethernet cable (dashboard)

1. Geofence: **Capture drone GPS** at each field corner and the 4 delivery corners -> **Save + Apply** (kept across
   reboots; redo only for a new field).
2. **📡 Real** -> link `/dev/ttyTHS1:921600` (TELEM2 on the Jetson UART) or `/dev/ttyACM0:921600` (USB).
3. **Launch stack** -> READY + preflight all green -> **▶ Full mission**. **■ Abort** / **LAND / RTL / KILL** any
   time; keep the RC transmitter in hand.

Terminal instead: `ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/real_mission.launch.py link:=/dev/ttyTHS1:921600`,
then `ros2 service call /mission/start std_srvs/srv/Trigger` (the dashboard is safer: preflight gate, abort).
Logs: `~/workspaces/isaac_ros-dev/runs/real_drone_<date>/`.

## Real drone over the telemetry radio only (no Wi-Fi): Radio GCS on the Lenovo

Once on the Pixhawk (QGC -> Parameters, reboot after): `MAV_0_CONFIG` TELEM 1 (radio, `SER_TEL1_BAUD` 57600),
`MAV_1_CONFIG` TELEM 2 (Jetson, `SER_TEL2_BAUD` 921600, `MAV_1_MODE` Onboard), `MAV_0_FORWARD` 1, `MAV_1_FORWARD` 1.

Once on the **Lenovo** (cable connected; use your radio's port, `ls /dev/serial/by-id/`):

```bash
scp szyfrowac@10.42.0.247:~/workspaces/isaac_ros-dev/src/gcs/radio_gcs/install_radio_gcs.sh ~/
bash ~/install_radio_gcs.sh /dev/ttyUSB0:57600
sudo usermod -aG dialout $USER      # if it says so; then log out / in
```

In QGC: Application Settings -> General -> AutoConnect: untick **SiK Radio** (the Radio GCS owns the radio and relays
it to QGC), keep **UDP**.

Each flight: plug the radio into the Lenovo, open http://127.0.0.1:8081 -> **PX4** + **Jetson** pills green ->
**Launch stack** -> **ready** + preflight green -> **▶ Full mission**. QGC runs beside it.

On the **Lenovo**, if the page doesn't open:

```bash
systemctl --user restart radio-gcs
systemctl --user status radio-gcs --no-pager | head -8
journalctl --user -u radio-gcs -n 30 --no-pager
```

## Stop everything

```bash
curl -s -X POST http://127.0.0.1:8080/api/stack_stop            # = the dashboard's Stop stack (sim or real)
~/workspaces/isaac_ros-dev/src/launch_all/scripts/stop_all.sh   # leftovers (keeps the always-on router)
~/workspaces/isaac_ros-dev/src/launch_all/scripts/status.sh     # what is still running
```

More detail: `~/workspaces/isaac_ros-dev/src/gcs/README.md`. Backups:
`bash ~/workspaces/isaac_ros-dev/src/launch_all/scripts/backup.sh` -> `~/workspaces/isaac_ros-dev/runs/backups/`.
