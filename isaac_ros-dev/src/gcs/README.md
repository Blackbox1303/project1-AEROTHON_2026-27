# AeroTHON ground station (GCS)

A web dashboard on the Jetson to set the geofence, check the drone, start and watch the mission, in Gazebo or on
the real drone. Open it in a browser on the Lenovo: **http://10.42.0.247:8080**

## Start

The dashboard runs on the Jetson (onboard the drone on the day) and starts by itself at boot (systemd user service
`gcs-dashboard`, linger on). Open it from the ground laptop's browser: **http://<jetson-ip>:8080** - on the lab USB
link http://10.42.0.247:8080; in the field the Jetson's Wi-Fi address (`hostname -I` on the Jetson, once). By hand:

```
systemctl --user restart gcs-dashboard      # status / stop the same way; log: runs/gcs/server.log
~/workspaces/isaac_ros-dev/src/gcs/run_gcs.sh   # or in the foreground
```

Everything after that is buttons; no terminal needed. **Dashboard not up after the Jetson's boot?** On the Lenovo:
`aerothon up` (or the **AeroTHON GCS** app, or http://127.0.0.1:8765) starts it over ssh and opens it.
`aerothon launch [world] [segment]`, `aerothon stop`, `aerothon status` work from there too.

Top bar: **mode**, **segment**, **▶ Launch stack**, **🖥 Open Gazebo (sync)**, **■ Stop stack**.
* **Stop** stops everything (Gazebo, autopilot SITL / link, MAVROS, nodes, the explorer) and closes the Gazebo GUI on
  the Lenovo. Nothing starts again until **Launch** (a Launch while stopping is refused).
* **Segments**: **Whole mission** | **QR area only** (search + read every QR, red zones avoided, no drop, home, land) |
  **Tunnels + banners only** (entry tunnel, straight to the exit tunnel, home, land) | **QR area + payload drop**.
* **LAND / RTH / RTL / KILL** go straight to ArduCopter (with its ACK); the mission then stops WITHOUT its failsafe
  flight, so nothing switches back to GUIDED over you. **RTH** = SmartRTL: retraces its own path home (through the
  tunnels it came by). **RTL** = climbs to RTL_ALT (15 m) and flies straight: never under a tunnel roof.

### Sim (Gazebo on the Jetson, GUI on the Lenovo only)
1. **🖥 Sim** mode, world, segment, **▶ Launch** -> READY (~1.5 min) -> the segment starts BY ITSELF once the
   preflight is green (pill: `qr_drop: waiting for READY` -> `qr_drop started`). QR-area segments first put the sim
   drone 1 m inside the zone (from the pad the straight line crosses the roofed tunnels) and use the world's start QR.
2. **🖥 Open Gazebo (sync)** whenever you want to watch (it waits for the sim if it is still starting).
3. **■ Stop**; a new run = **Launch** again.

### Offboard Jetson (Jetson on the ground, VRX + radio; the QR area only)
The drone carries the flight controller, an analog camera and its VTX. On the Jetson: the VRX receiver (USB video)
and the SiK radio (`/dev/ttyUSB0:57600`). No Arducam, no RealSense, no ROS stack.
1. **📺 Offboard Jetson** mode; link **SiK radio** (or **bench SITL**: ArduCopter SITL, no Gazebo, no video); VRX device
   (empty = pick it on the 8094 view; /dev/video1-5 are the D435 here); segment **QR area only** or **QR area + drop**.
2. **▶ Launch** -> `launch_all/offboard_vrx.launch.py`: onboard router paused, mavlink-router (radio -> this dashboard,
   the explorer, QGC on the Lenovo), `gcs/vrx_explore` (8094: red zones, QR, A* + frontier, 10 m, drop at 5 m) - its
   page is shown in the camera card. Zone: the applied geofence, else the sim world's.
3. Preflight green -> **▶ Start** (drop: the start QR is read at the pad, or type the payload).

### Final flight (Jetson onboard, only the telemetry radio to the ground)
On the Lenovo with the SiK radio: `aerothon radio` -> http://127.0.0.1:8080 - the same page in RADIO mode. This PC
owns the radio and relays every packet to QGC (UDP 14550 auto-connect; turn QGC's SiK auto-connect OFF). It shows
ArduCopter's telemetry and the onboard Jetson's status (mav_agent: stack phase, mission state, times, preflight).
Before take-off: segment + **Launch** (onboard stack) + **▶ Start** over the radio. Armed: only **LAND / RTH / RTL /
KILL**. The onboard dashboard's preflight and geofence rules apply (set the geofence once by cable).

### Real drone by cable (Pixhawk 6C Mini + RealSense D435i + Arducam, Jetson onboard)
1. On the drone: Pixhawk on the Jetson's USB (`/dev/ttyACM0`); D435i on USB 3; Arducam on CSI CAM0. Measure the mounts once:
   D435 in `vio_bringup/config/rig.yaml`, Arducam in `vio_bringup/launch/arducam_real.launch.py` (MOUNT).
2. Calibrate the Arducam once (`launch_all/scripts/arducam_calib.sh` -> http://<jetson-ip>:8090) - the stack uses
   `calibration/arducam_fisheye_1640x1232.yaml` by itself when it exists.
3. **📡 Real** mode; telemetry link `/dev/ttyACM0:921600`.
4. **Geofence**: carry / fly the drone to each field corner and the 4 delivery-region corners, **Capture drone GPS**
   each, **Save + Apply** (required: the real stack refuses to start without it).
5. **Launch stack** -> `launch_all/real_mission.launch.py` (router + MAVROS, then the container stack). READY ->
   preflight all green -> **▶ Start** (segment). Keep the RC transmitter in hand.
6. **Stop stack** when done. Logs: `runs/real_drone_<date>/`.

## Field test (props OFF, no course material)

`field_test/`: the real sensors and the mission's decisions, with the drone carried by hand. On the Lenovo:
`aerothon fieldtest cable` (Ethernet: live views) or `aerothon fieldtest radio` (the telemetry radio alone: text), page
http://127.0.0.1:8096; **Start on the Jetson** runs `field_test/field_test.sh start` (cable: over ssh; radio: through
`mav_agent`, action 11). By hand on the Jetson: `field_test/field_test.sh start | stop | status | log`, page :8097.

| | |
|---|---|
| `field_vision.py` | dev container: Arducam (GStreamer) + D435 (the RealSense node, ROS topics) + the detectors (WeChat / zbar, the deep read and image match vs the start QR, the delivery_mapper red rule, banner_detector's rule, a depth grid) + the walk-throughs: QR area search (lanes over the 4 corners, over each board, read, match -> down to 5 m + payload, unsure -> 0.5 m lower, never below 5.5 m - a VIRTUAL height) and obstacle nav. HTTP :8097 |
| `field_link.py` | Jetson host: the Pixhawk via the router's `field_test` endpoint (UDP 14662): GPS / heading / height in, MAV_CMD_DO_MOTOR_TEST out (one motor at a time; never armed): the spinning side = walk there, a chase = yaw, speed = height |
| `field_ground.py` + `field_test.html` | the Lenovo's page (cable: proxies the Jetson; radio: COMMAND_LONG 31010 / 11 to the agent, TUNNEL 32820 + STATUSTEXT 'FT ...' back, QGC relayed) |
| `field_proto.py` | the command codes / radio packet shared by all of them |

Logs and saved state: `runs/field_test/` (events.log, start_qr.json, corners.json, vision / realsense / link logs).
Not while a mission stack runs (the cameras); stop gives the Arducam back to the calibration page.

## Gazebo GUI on the Lenovo

The Jetson runs Gazebo headless; the GUI runs on the Lenovo only (never on the Jetson: the helper refuses to run there).

* One-time setup (and again after changes on the Jetson), **in a terminal on the Lenovo**:
  ```
  scp szyfrowac@10.42.0.247:~/workspaces/isaac_ros-dev/src/gcs/lenovo/install_lenovo_agent.sh ~/
  bash ~/install_lenovo_agent.sh
  ```
  Installs the helper on `127.0.0.1:8765` (systemd user service; finds the desktop's display by itself), `~/host_gui.sh`,
  `aerothon`, the radio dashboard, an ssh key to the Jetson.
* **Open Gazebo (sync)** in the dashboard; the pill shows syncing / waiting for the sim / open, or FAILED + why.
* By hand: `~/host_gui.sh [--nosync|--follow|--free]`; helper log `~/.cache/gcs_gui_agent.log`.

## Radio-only ground station (no Wi-Fi on the drone)

Commands and status over the telemetry radio alone, from the Lenovo: **Radio GCS** at http://127.0.0.1:8081 (on the
Lenovo itself). QGC keeps working beside it (relayed on UDP 14550).

```
Lenovo ── SiK radio ──> Pixhawk TELEM1 ─(MAVLink forwarding)─> TELEM2 ──> Jetson /dev/ttyTHS1
Radio GCS :8081 ─ relays to QGC                                  aerothon-router (always on) ─ MAVROS, dashboard,
                                                                 aerothon-mav-agent: commands -> this dashboard's API
```
* Jetson, once: `bash ~/workspaces/isaac_ros-dev/src/gcs/onboard/install_onboard.sh /dev/ttyTHS1:921600` (done
  2026-10-25): services `aerothon-router` (mavlink-router on the Pixhawk AND the sim's PX4, so neither stack starts
  its own) + `aerothon-mav-agent` (onboard/mav_agent.py, MAVLink component 1/194).
* PX4, once (QGC -> Parameters, reboot after): MAV_0_CONFIG = TELEM 1 (radio, SER_TEL1_BAUD 57600),
  MAV_1_CONFIG = TELEM 2 (Jetson, SER_TEL2_BAUD 921600, MAV_1_MODE = Onboard), MAV_0_FORWARD = 1, MAV_1_FORWARD = 1.
* Lenovo, once (Ethernet cable connected): `scp szyfrowac@10.42.0.247:~/workspaces/isaac_ros-dev/src/gcs/radio_gcs/install_radio_gcs.sh ~/ && bash ~/install_radio_gcs.sh /dev/ttyUSB0:57600`
  -> service `radio-gcs` (starts at login). In QGC: Application Settings -> General -> AutoConnect: untick
  SiK Radio and Pixhawk (the Radio GCS owns the radio), keep UDP. Optional QGC buttons:
  ~/Documents/QGroundControl/MavlinkActions/aerothon.json (QGC 4.3+, Fly View -> MAVLink Actions).
* Each flight: geofence with the cable on the onboard dashboard (Save + Apply), then on the Radio GCS:
  **Launch stack** -> READY + preflight green -> **▶ Full mission**; **■ Abort**, **LAND / RTL / KILL** any time.
  The onboard dashboard's rules apply (no real launch without the geofence, no start before READY).
* Test on the Jetson without a radio: `python3 radio_gcs/radio_gcs.py --radio udpin:0.0.0.0:14670 --port 8081`
  (the router's radio_test endpoint), sim stack up, tick "sim (test)" for Launch.

## MAVLink

mavlink-router (`router/sim.conf` / `router/real.conf`) splits PX4's link to MAVROS (UDP 14640), this dashboard
(14650) and QGroundControl on the Lenovo (10.42.0.1:14550). For the real drone, set `Device` / `Baud` in
`router/real.conf` to match the wiring.

## Files

| | |
|---|---|
| `gcs_server.py` | server: MAVLink sink, ROS 2 node, web / WebSocket / MJPEG / REST |
| `static/index.html` | the dashboard |
| `real_link.launch.py` | real drone, link only: router + MAVROS (the full stack: `launch_all/real_mission.launch.py`) |
| `router/*.conf` | mavlink-router configs |
| `lenovo/` | the Lenovo's Gazebo GUI helper + installer |
| `runs/gcs/` (data) | `geofence.json` (GPS), `mission_geofence.json` (for the nodes), `server.log` |

The sim stack without the dashboard:
`ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/sim_mission.launch.py world:=comp_far1 [world_json:=...] [autostart:=1]`.
Each run's logs go to `runs/<run>/`.
