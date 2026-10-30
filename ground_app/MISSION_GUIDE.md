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

## 2. The final mission — ONE command, AUTO

The final mission is its own app (`aerothon mission`); offboard, sim and the field test are separate commands, one
at a time.

**Before the slot (once, any time before):** `aerothon onboard aruco` → http://127.0.0.1:8080 → **Geofence (GPS)**
card: the organisers' points (Arena or QR area: 4, Extra: 2 — typed, or **⊕ Capture drone GPS** carrying the drone to
each corner); **Course**: tunnel length **6.22 m**, the width as measured, height 10 ft, max inside 8 ft → **Save**.
It stays saved on this PC. Close it (`aerothon stop`).

**At the slot:**
1. **Drone on the red start zone**, nose roughly towards the entry tunnel; the **start ArUco 1 m ahead** (it takes off
   to 5 m, moves 1 m forward and keeps the marker's image). Battery in, Jetson booted (~1 min).
2. **RC (ELRS) on.** Don't touch the mode switch unless you take over: any pilot mode on it (STABILIZE, ALT_HOLD,
   LOITER …) hands the drone to the pilot at once.
3. **Lenovo**, ground SiK radio plugged in:
   ```bash
   aerothon mission
   ```
   It opens http://127.0.0.1:8080 and, in the terminal: radio + autopilot heard → Jetson heard → geofence Apply (up
   the radio) → **Launch stack** (1–3 min) → **READY** → **preflight green** → **10 s countdown** (Ctrl-C cancels:
   nothing flies) → **START, auto**. Then it prints every state / flight-mode change until landed and disarmed.
   To only test the link (nothing sent to the drone): with `aerothon onboard aruco` running,
   `python3 ~/aerothon_here_we_come/apps/dashboard/auto_mission.py --check`.
4. The mission flies itself: TAKEOFF (5 m) → READ_START_QR (the start marker's image) → OVERHEAD_GREEN (the green
   doormat from 5 m) → ALIGN_GREEN (2.3 m, wall fit) → ENTRY_CORRIDOR (1.5 m, end found by the walls) → climb to 9 m →
   SEARCH_DELIVERY (markers matched to the start marker; unsure: hover, deep read, 0.5 m lower) → DELIVER (5 m over the
   match, winch, **45 s** at 5 m) → 9 m → ALIGN_EXIT (the **blue** banner) → exit tunnel → RETURN_HOME → LAND on the
   start zone.
5. After landing: **■ Stop stack** on the page, `aerothon stop`.

**Reset:** `aerothon mission` RESETS by itself before every launch (stops a stack left up, clears the last run's start
marker, marker-area memory - red zones and markers rejected as not the target - and tunnel entrance). By hand: the
**↺ Reset mission** button in the Mission card (stack stopped, drone disarmed). Saving the geofence never launches
anything.

## 3. The page while it flies (8080)

Once the drone is armed the page shows only these (the rest is hidden):

| button | what it does |
|---|---|
| **✋ MANUAL (pilot takes over)** | **STABILIZE** on the autopilot — fully manual, the throttle comes from the RC stick AT ONCE (it asks: pilot ready, throttle at hover?). The onboard mission stops sending setpoints, no failsafe flight. |
| LAND | autopilot LAND where it is; the mission stops |
| RTH (retrace) | SmartRTL: its own path back (through the tunnels), lands |
| RTL (straight) | climbs to RTL_ALT, straight home — **not inside / near a roofed tunnel** |
| KILL | motors off at once (asks twice) — the drone falls |

The RC mode switch to any pilot mode does the same as MANUAL. Pills: stack · link (radio) · state (supervisor) ·
mission time · time in the marker area. The **Mission (supervisor FSM)** row lights the state; the log shows the events.

On the ground (disarmed) the page also has, for doing it by hand: segment drop-down, ▶ Launch stack, ■ Stop stack,
▶ Start, search height ▼▲ 0.5 m (9.5 m max), the Geofence card (Capture, + row, Course, Save, Save + Apply), Preflight.

## 4. Props-off test (field test page, 8096)

```bash
aerothon fieldtest radio          # page http://127.0.0.1:8096
```
**Start on the Jetson** → **Read START marker** over the chosen marker → **Scan / match** over each of the others and
the start one again (tiles: MATCH ✓ / NOT THE TARGET / UNSURE - come lower) → **Green banner** facing the entry banner →
tick *props are OFF* → **Spin all 4** / **Motor cues ON** → **QR area search** walk-through → **Stop it**.

## 5. Over the cable (Ethernet to the Jetson, for lab tests)

`aerothon sim aruco` opens the Jetson's own ArUco dashboard (8084) — sim, offboard, real drone and every segment.
