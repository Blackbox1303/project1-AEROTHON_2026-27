# AeroTHON final mission — launch sequence and buttons (course of 2026-10-30)

The course: ArUco markers only (matched by image to the start marker, never read), green entry-tunnel banner, blue
exit-tunnel banner (a top banner + a doormat on the floor each), tunnels 20 ft 5 in (6.22 m) long, red start zone (take off
from it, land back on it), never above 10 m (the drone searches at 9.0 m, hard cap 9.5 m), 45 s at 5 m for the winch.

## Wiring (on the drone)

| Pixhawk 6C port | connected |
|---|---|
| USB (one cable, two ports) | Jetson: port 1 = mavlink-router (MAVROS, dashboard), port 2 = the radio agent |
| GPS1 | GPS · TELEM1 = ESC telemetry · TELEM2 = ELRS receiver · **GPS2 = SiK radio** |

The Jetson starts everything it needs at boot (`aerothon-router`, `aerothon-mav-agent`, `gcs-dashboard`).

## 1. Install / update on the Lenovo (once per new version)

```bash
aerothon stop
scp szyfrowac@10.42.0.247:~/aerothon_here_we_come.tar.gz ~/Downloads/
rm -rf ~/aerothon_here_we_come && tar xzf ~/Downloads/aerothon_here_we_come.tar.gz -C ~ && ~/aerothon_here_we_come/install.sh
```

## 2. Launch sequence — the final mission (over the telemetry radio)

1. **Drone on the red start zone**, nose roughly towards the entry tunnel; the **start ArUco 1 m ahead** of it (the
   drone takes off to 5 m, moves 1 m forward and reads it from straight above). Battery in, Jetson booted (~1 min).
2. **RC (ELRS) on**, mode switch in a pilot mode (LOITER). Don't touch the switch during the mission unless you are
   taking over — any pilot mode (LOITER / POSHOLD / ALT_HOLD / STABILIZE / BRAKE …) ends the autonomy at once.
3. **Lenovo**: ground SiK radio plugged in, then
   ```bash
   aerothon onboard aruco
   ```
   It finds the radio itself (`/dev/ttyUSB0`, else the one SiK / FTDI / CP210x plugged in). Page:
   **http://127.0.0.1:8080**. QGroundControl (optional): UDP 14550 on, SiK auto-connect **off**.
4. Header: **📻 RADIO** pill, **link** pill green (telemetry over the radio), **stack: stopped**.
5. **Geofence (GPS)** card: fill the organisers' points (Arena or QR area: 4, Extra: 2) — type them, or carry the drone
   to each corner and press **⊕ Capture drone GPS**. **Course**: tunnel length **6.22 m**, width as measured, height
   10 ft, max inside 8 ft. **Save + Apply** (sent up the radio; the drone saves it).
6. Segment drop-down: **Whole mission**. **▶ Launch stack** → 1–3 min → **stack: READY**, the **Preflight** list
   green.
7. Leave **target payload** empty and **read start QR at take-off** ticked (the start ArUco's image is taken in flight).
8. Everyone clear, **▶ Start**. The mission state shows in **Mission (supervisor FSM)** and the log:
   TAKEOFF (5 m) → READ_START_QR (the start marker's image) → OVERHEAD_GREEN (finds the green doormat from 5 m) →
   ALIGN_GREEN (2.3 m, wall fit) → ENTRY_CORRIDOR (1.5 m through the tunnel, end found by the walls) → climb to 9 m →
   SEARCH_DELIVERY (markers matched to the start marker; unsure ones: hover, deep read, 0.5 m lower) → DELIVER (down to
   5 m over the match, winch, **45 s** at 5 m) → back to 9 m → ALIGN_EXIT (the **blue** banner) → the exit tunnel →
   RETURN_HOME → LAND on the start zone.
9. After landing: **■ Stop stack**.

## 3. The buttons (8080, radio mode)

**Header**

| button / pill | what it does |
|---|---|
| segment drop-down | what Start flies: Whole mission · QR area only (no drop) · Tunnels + banners only · QR area + payload drop |
| ▶ Launch stack | starts the drone's whole stack (cameras, MAVROS, navigation, supervisor) over the radio; on the ground only |
| ■ Stop stack | stops it (ground only) |
| stack / link / state / mission / QR area pills | stack phase · radio link · supervisor state · mission time · time in the marker area |

**Mission card**

| button | what it does |
|---|---|
| ▶ Start | starts the chosen segment (needs READY + green preflight; "ignore preflight" overrides — don't) |
| Search height ▼ 0.5 m / ▲ 0.5 m | trims the search height in flight (never above 9.5 m, never below 5.5 m) |
| LAND | autopilot LAND where it is; the mission stops |
| RTH (retrace) | SmartRTL: flies its own path back (through the tunnels) and lands |
| RTL (straight) | climbs to RTL_ALT and flies straight home — **not inside / near a roofed tunnel** |
| KILL | motors off at once (asks twice) — the drone falls |
| **Take over: LOITER** | holds position + height (GPS); the mission stops; you fly on the RC |
| **POSHOLD** | like LOITER, brakes when the sticks are centred |
| **ALT_HOLD** | holds only the height (drifts with wind) |
| **STABILIZE** | fully manual — throttle from the stick at once (asks if the throttle is at hover) |
| **BRAKE** | stops dead and hovers |

Every one of those also tells the onboard mission "operator took over": it stops sending setpoints and does **no**
failsafe flight. Flipping the RC mode switch to a pilot mode does the same by itself.

**Geofence card**: ⊕ Capture drone GPS (the drone's GPS into a row) · + row · Course table · Save · **Save + Apply**.

## 4. Props-off test (field test page, 8096)

```bash
aerothon fieldtest radio          # page http://127.0.0.1:8096
```
**Start on the Jetson** → **Read START marker** over the chosen marker → **Scan / match** over each of the others and
the start one again (tiles: MATCH ✓ / NOT THE TARGET / UNSURE - come lower) → **Green banner** facing the entry banner →
tick *props are OFF* → **Spin all 4** / **Motor cues ON** → **QR area search** walk-through → **Stop it**.

## 5. Over the cable (Ethernet to the Jetson, for lab tests)

`aerothon sim aruco` opens the Jetson's own ArUco dashboard (8084) — sim, offboard, real drone and every segment.
