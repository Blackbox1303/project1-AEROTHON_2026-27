#!/usr/bin/env python3
"""AeroTHON ground station: one web dashboard to set the geofence, check the drone, run and watch the mission.

Runs on the Jetson (host, not the container); open http://10.42.0.247:8080 on the Lenovo.
    ~/workspaces/isaac_ros-dev/src/gcs/run_gcs.sh            (or: python3 gcs_server.py [--port 8080])
ArUco delivery mission: the same dashboard with --marker aruco (http://10.42.0.247:8084, MAVLink udpin :14651:
scripts/gcs_aruco.sh). The boards carry ArUco markers 3x3 .. 6x6 instead of QR codes: every stack it launches gets
AEROTHON_MARKER=aruco (qr_reader, qr_locator, vrx_explore read it); --radio sends it to the onboard agent (launch
param3 = 1). Everything else is the 8080 mission - every segment (whole mission, QR area, tunnels + banners, QR area
+ drop; the offboard Jetson: the two QR-area ones, at 1 m), and a sim world <w> flies as its ArUco twin <w>_aruco (vio_cuda_ws/sim/make_aruco_world.py, made if missing).

Data in:
  MAVLink   udpin :14650 - a sink of mavlink-router (router/sim.conf, router/real.conf), which also feeds MAVROS and
            QGC on the Lenovo: telemetry, GPS origin (GPS <-> odom), STATUSTEXT; emergency LAND / RTL / KILL go back
            the same way
  ROS 2     /mission/state, /mission/log (supervisor FSM), /delivery/map (costmap), /delivery/qrs, /delivery/target,
            /nav/odom (track), node states, /delivery/coverage, cameras (Gazebo) - and the supervisor's services:
            /mission/start, /mission/abort, /mission_supervisor/set_parameters, /mission/start_qr (payload)
  Camera    a ROS image topic (sim) or /dev/video2 (the Jetson's camera), as MJPEG
Geofence: GPS slots arena (4) + QR area (4) + extra (2) - the organisers' 6 points may be either set - typed in or
captured from the drone's GPS (move it there, press Capture), and the course dimensions (tunnel width / height /
max flight height / length / gap, QR-area width; defaults = the organisers' numbers). The corridors are not
recorded (GPS-less tunnels). Apply converts it to odom around the autopilot's GPS origin and writes
runs/gcs/mission_geofence.json - the nodes' world_json: delivery_zone (bounding box) + delivery_polygon (the QR
area; arena only: cut at the entry tunnel's exit line in flight), field_polygon (offboard_bridge's keep-in), course,
corridors (sim: the world's, for the tunnel ceiling).
Stack: sim = sim_mission.launch.py (Gazebo headless + PX4 + router + MAVROS + container), held on the ground;
real = mavlink-router (real.conf) + MAVROS on the flight controller's serial link (the drone's own sensor / mission
stack is started on the drone).
"""
import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from aiohttp import web
try:                                                  # not needed in --radio mode (the Lenovo may not have them)
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

OFFBOARD_ALT = 9.0           # (< 10 m: the airspace rule, 2026-10-30) --offboard-alt: the VRX explorer's search height, m (ArUco 8084: 1 m)
MARKER = 'qr'                # --marker: what the boards carry, qr | aruco (the 8084 ArUco delivery dashboard)
RADIO = None                 # --radio: this dashboard runs ON THE LENOVO on the telemetry radio (the final flight)
AGENT = 194                  # the onboard Jetson's mav_agent (MAV_COMP_ID_ONBOARD_COMPUTER2)
AGENT_CMD = 31010            # its MAV_CMD_USER_1; param1: 1 launch 2 stop 5 abort 6 status 7 start segment 8 operator
SEG_CODE = {'full': 1, 'qr': 2, 'tunnel': 3, 'qr_drop': 4}

WS = Path.home() / 'workspaces/isaac_ros-dev'
DATA = WS / 'runs/gcs'
WORLDS = Path.home() / 'workspaces/vio_cuda_ws/sim/worlds'
HERE = Path(__file__).resolve().parent
LAUNCH_ALL = WS / 'src/launch_all'
EARTH_R = 6378137.0

# the supervisor's FSM, in mission order (for the dashboard's pipeline)
FSM = {
    'full': ['TAKEOFF', 'READ_START_QR', 'OVERHEAD_GREEN', 'ALIGN_GREEN', 'GEOFENCE', 'ENTRY_CORRIDOR', 'SEARCH_DELIVERY', 'DELIVER',
             'DROP', 'TO_EXIT', 'ALIGN_EXIT', 'RETURN_CORRIDOR', 'RETURN_HOME', 'LAND', 'DONE'],
    'qr_drop': ['TAKEOFF', 'READ_START_QR', 'DIRECT_TO_ZONE', 'SEARCH_DELIVERY', 'DELIVER', 'DROP', 'ZONE_EXIT',
                'RETURN_HOME', 'LAND', 'DONE'],
    'qr': ['TAKEOFF', 'READ_START_QR', 'DIRECT_TO_ZONE', 'SEARCH_DELIVERY', 'ZONE_EXIT', 'RETURN_HOME', 'LAND', 'DONE'],
    'tunnel': ['TAKEOFF', 'READ_START_QR', 'OVERHEAD_GREEN', 'ALIGN_GREEN', 'GEOFENCE', 'ENTRY_CORRIDOR', 'TO_EXIT', 'ALIGN_EXIT',
               'RETURN_CORRIDOR', 'RETURN_HOME', 'LAND', 'DONE'],
}
# the mission segments: supervisor parameters. 'delivery' = the old name of qr_drop (mav_agent, radio GCS)
SEGMENTS = {
    'full': {'start_from': 'takeoff', 'drop': True, 'return_corridor': True},
    'qr': {'start_from': 'delivery', 'drop': False, 'return_corridor': False},
    'qr_drop': {'start_from': 'delivery', 'drop': True, 'return_corridor': False},
    'tunnel': {'start_from': 'tunnel', 'drop': False, 'return_corridor': True},
}
SEGMENT_ALIAS = {'delivery': 'qr_drop'}
CAMERAS = {
    'qr_hud': 'http://127.0.0.1:8091/annotated.jpg',
    'arducam': '/arducam/image_raw',
    'nadir': '/arducam/nadir_hd/image_raw',
    'video2': '/dev/video2',
}


def gps_to_local(lat, lon, lat0, lon0):
    """Equirectangular, ENU metres (fine over a few hundred metres)."""
    return (math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)), math.radians(lat - lat0) * EARTH_R)


def local_to_gps(x, y, lat0, lon0):
    return (lat0 + math.degrees(y / EARTH_R), lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0)))))


def inside_or_on(x, y, poly, tol=0.05):
    """In the polygon, or within tol m of its outline (a corner placed on the boundary counts as inside)."""
    if in_poly(x, y, poly):
        return True
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / max(dx * dx + dy * dy, 1e-9)))
        if math.hypot(x1 + t * dx - x, y1 + t * dy - y) <= tol:
            return True
    return False


def in_poly(x, y, poly):
    inside, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


class State:
    """Everything the dashboard shows; written by the MAVLink / ROS threads, read by the web side."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tel = {}                # MAVLink telemetry
        self.tel_time = {}           # message type -> monotonic time of the last one
        self.statustext = []         # (time, severity, text)
        self.origin = None           # (lat0, lon0, alt0) of PX4's local frame
        self.ros = {'state': '', 'log': [], 'qrs': [], 'candidates': [], 'pairwise_distances_m': [],
                    'drone_qrs': [], 'start_qr': None, 'target': None, 'states': {}, 'coverage': {},
                    'track': [], 'pose': None, 'mavros_connected': None}
        self.ros_time = {}
        self.costmap_png, self.costmap_meta, self.costmap_ver = None, None, 0
        self.tel_pose = None         # (x, y, z, yaw) ENU odom from MAVLink LOCAL_POSITION_NED + ATTITUDE
        self.tel_track = []
        self.mission_type = 'full'
        self.red_check = None        # why the red map is not the one 10 ft red zone (red_sanity), None = fine
        self.mission_t0 = None
        self.zone_t0 = None
        self.alt_offset = 0.0        # the operator's search-height trim (m), as last set on the explorer
        self.sim_clock = None        # (sim s, monotonic when received) from /clock: the sim's mission clock


def clock_s():
    """The mission clock: SIM time in the sim (the flight's own time - at RTF 0.2 the wall clock showed 12 min for a
    2.8 min flight, 2026-10-28), the wall clock on the real drone / offboard."""
    c = S.sim_clock
    if STACK.mode == 'sim' and c is not None and time.monotonic() - c[1] < 5.0:
        return c[0]
    return time.time()


S = State()


# ---------------------------------------------------------------- MAVLink (a mavlink-router sink)
def find_radio(path):
    """The ground SiK radio: `path` if it is there, else the one radio plugged in - by its USB id (SiK / FTDI / CP210x),
    else the only /dev/ttyUSB* or /dev/ttyACM* (a SiK shows as either). None: nothing found."""
    import glob
    import re
    if os.path.exists(path):
        return path
    ids = [p for p in sorted(glob.glob('/dev/serial/by-id/*'))
           if re.search(r'sik|3dr|rfd|radio|telem|ftdi|ft231|silicon_labs|cp210', p, re.I)
           and not re.search(r'ardupilot|pixhawk|px4|cube', p, re.I)]
    if ids:
        return os.path.realpath(ids[0])
    ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    return ports[0] if len(ports) == 1 else None


class Mavlink(threading.Thread):
    def __init__(self, port, radio=None, qgc='127.0.0.1:14550'):
        super().__init__(daemon=True)
        os.environ['MAVLINK20'] = '1'                 # TUNNEL (the agent's status) is MAVLink 2
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.radio, self.radio_err = radio, None
        self.agent, self.agent_detail, self.chunks = {}, {}, {}
        self.vio, self.boards = {}, {}   # the agent's VIO position (32802) and boards (32803, by payload)
        if radio:
            # the final flight: this PC owns the SiK radio and relays every packet both ways to QGroundControl
            # (its UDP auto-connect) - a mavlink-router in-process: QGC and this dashboard both see the drone
            import socket
            self.conn = None
            self.open_radio()
            self.qgc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.qgc.bind(('127.0.0.1', 0))
            self.qgc.setblocking(False)
            h, _, pt = qgc.partition(':')
            self.qgc_addr = (h, int(pt))
            threading.Thread(target=self.qgc_loop, daemon=True).start()
            threading.Thread(target=self.hb_loop, daemon=True).start()
        else:
            self.conn = mavutil.mavlink_connection(f'udpin:0.0.0.0:{port}', source_system=250)
        self.target = (1, 1)
        self.last_req = 0.0
        self.acks = {}               # command id -> (MAV_RESULT, monotonic time)

    def open_radio(self):
        """Never fatal: retried every 2 s (radio unplugged, no dialout permission) - the reason is on the page."""
        dev = self.radio
        try:
            if dev.startswith(('udpin:', 'udpout:', 'tcp:')):
                self.conn = self.mavutil.mavlink_connection(dev, source_system=255, source_component=190)
            else:
                path, _, baud = dev.partition(':')
                path = find_radio(path) or path                  # not there: the one radio plugged in
                if not Path(path).exists():
                    raise OSError(f'{path} not found (radio plugged in? ls /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/)')
                self.conn = self.mavutil.mavlink_connection(path, baud=int(baud or 57600), source_system=255,
                                                            source_component=190)
            self.radio_err = None
        except Exception as e:
            self.conn = None
            self.radio_err = str(e) + (' - sudo usermod -aG dialout $USER, log out / in' if 'ermission' in str(e) else '')

    def qgc_loop(self):
        """QGC -> the radio (its heartbeats, commands, parameters)."""
        while True:
            try:
                data = self.qgc.recv(4096)
                if self.conn is not None:
                    self.conn.write(data)
            except BlockingIOError:
                time.sleep(0.01)
            except OSError:
                time.sleep(0.2)

    def hb_loop(self):
        while True:
            try:
                self.conn.mav.heartbeat_send(6, 8, 0, 0, 4)   # MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, ACTIVE
            except Exception:
                pass
            time.sleep(1.0)

    def radio_msg(self, msg, t, now):
        """--radio: relay to QGC; the onboard agent's heartbeat / status / events."""
        try:
            self.qgc.sendto(msg.get_msgbuf(), self.qgc_addr)
        except OSError:
            pass
        if msg.get_srcComponent() != AGENT:
            return
        if t == 'HEARTBEAT':
            S.tel_time['agent'] = now
        elif t == 'TUNNEL' and msg.payload_type in (32800, 32801, 32802, 32803):
            try:
                d = json.loads(bytes(msg.payload[:msg.payload_length]).decode())
            except ValueError:
                return
            if msg.payload_type == 32800:
                self.agent = d
                S.tel_time['agent_status'] = now
            elif msg.payload_type == 32801:
                self.agent_detail = d
            elif msg.payload_type == 32802:               # VIO position (mav_agent --vio-period)
                self.vio = d
                S.tel_time['agent_vio'] = now
            elif d.get('p'):                              # a board (mav_agent --board-period): latest per payload
                self.boards[d['p']] = d
        elif t == 'STATUSTEXT':                       # chunked (MAVLink 2): join the chunks of one id
            txt = msg.text.rstrip('\0')
            if msg.id:
                self.chunks[msg.id] = self.chunks.get(msg.id, '') + txt
                if len(txt) == 50:
                    return
                txt = self.chunks.pop(msg.id)
            S.statustext = (S.statustext + [(time.time(), msg.severity, 'JETSON ' + txt)])[-40:]

    def run(self):
        m = self.mavutil.mavlink
        while True:
            if self.conn is None:
                time.sleep(2)
                self.open_radio()
                continue
            try:
                msg = self.conn.recv_match(blocking=True, timeout=1.0)
            except Exception as e:                    # the radio unplugged: reopen
                self.conn, self.radio_err = None, f'lost: {e}'
                continue
            now = time.monotonic()
            if now - self.last_req > 5.0 and self.conn.target_system:
                self.last_req = now
                self.request(49)                      # GPS_GLOBAL_ORIGIN
                self.request(242)                     # HOME_POSITION
            if msg is None:
                continue
            t = msg.get_type()
            if t == 'BAD_DATA':
                continue
            if self.radio:
                self.radio_msg(msg, t, now)
            with S.lock:
                S.tel_time[t] = now
                tel = S.tel
                if t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                    self.target = (msg.get_srcSystem(), msg.get_srcComponent())
                    tel['armed'] = bool(msg.base_mode & m.MAV_MODE_FLAG_SAFETY_ARMED)
                    tel['mode'] = self.mavutil.mode_string_v10(msg)
                    tel['system_status'] = msg.system_status
                elif t == 'GLOBAL_POSITION_INT':
                    tel.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7, alt_msl=msg.alt / 1e3,
                               alt_rel=msg.relative_alt / 1e3, heading=msg.hdg / 100 if msg.hdg != 65535 else None,
                               vx=msg.vx / 100, vy=msg.vy / 100, vz=msg.vz / 100)
                elif t == 'GPS_RAW_INT':
                    tel.update(gps_fix=msg.fix_type, sats=msg.satellites_visible,
                               hdop=msg.eph / 100 if msg.eph != 65535 else None)
                elif t == 'SYS_STATUS':
                    tel.update(batt_v=msg.voltage_battery / 1000 if msg.voltage_battery != 65535 else None,
                               batt_pct=msg.battery_remaining if msg.battery_remaining >= 0 else None,
                               sensors_health=msg.onboard_control_sensors_health,
                               sensors_enabled=msg.onboard_control_sensors_enabled)
                elif t == 'BATTERY_STATUS':
                    if msg.battery_remaining >= 0:
                        tel['batt_pct'] = msg.battery_remaining
                elif t == 'ATTITUDE':
                    tel.update(roll=math.degrees(msg.roll), pitch=math.degrees(msg.pitch), yaw=math.degrees(msg.yaw))
                    tel['yaw_enu'] = math.pi / 2 - msg.yaw            # NED heading -> ENU yaw
                elif t == 'VFR_HUD':
                    tel.update(groundspeed=msg.groundspeed, climb=msg.climb, throttle=msg.throttle)
                elif t == 'LOCAL_POSITION_NED':
                    tel.update(local_x=msg.y, local_y=msg.x, local_z=-msg.z)      # ENU
                    S.tel_pose = (msg.y, msg.x, -msg.z, tel.get('yaw_enu', 0.0))
                    tr = S.tel_track
                    if not tr or math.dist(tr[-1], (msg.y, msg.x)) > 0.3:
                        S.tel_track = (tr + [(round(msg.y, 2), round(msg.x, 2))])[-3000:]
                elif t == 'GPS_GLOBAL_ORIGIN':
                    S.origin = (msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1e3)
                elif t == 'HOME_POSITION':
                    tel.update(home_lat=msg.latitude / 1e7, home_lon=msg.longitude / 1e7)
                    if S.origin is None:
                        S.origin = (msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1e3)
                elif t == 'STATUSTEXT' and msg.get_srcComponent() == 1:
                    # the autopilot's only: onboard/mav_agent.py (component 194, same system) sends this dashboard's
                    # first alert as STATUSTEXT 'ALERT ...' - taken as an FCU message it became a new alert, which the
                    # agent sent again ('ALERT PX4: ALERT PX4: ...', every 0.5 s, 2026-10-27)
                    S.statustext = (S.statustext + [(time.time(), msg.severity, msg.text)])[-40:]
                elif t == 'COMMAND_ACK':
                    self.acks[msg.command] = (msg.result, now)
                elif t == 'ESTIMATOR_STATUS':
                    tel['ekf_flags'] = msg.flags
                elif t == 'RADIO_STATUS':
                    tel.update(rssi=msg.rssi, remrssi=msg.remrssi)

    def request(self, msg_id):
        self.conn.mav.command_long_send(*self.target, 512, 0, msg_id, 0, 0, 0, 0, 0, 0)

    def command(self, cmd, *p, comp=None):
        p = list(p) + [0] * (7 - len(p))
        if self.conn is None:
            return
        self.conn.mav.command_long_send(self.target[0], comp or self.target[1], cmd, 0, *p)

    def tunnel_to_agent(self, ptype, data):
        """A TUNNEL (<= 128 B) to the onboard agent: the geofence corners go up this way (radio_apply_gf)."""
        if self.conn is not None:
            self.conn.mav.tunnel_send(self.target[0], AGENT, ptype, len(data), list(data.ljust(128, b'\0')))

    RESULTS = {0: 'accepted', 1: 'temporarily rejected', 2: 'DENIED', 3: 'unsupported', 4: 'FAILED', 5: 'in progress'}

    def command_ack(self, cmd, *p, timeout=3.0, comp=None, final_within=0.0):
        """Send, resend (a radio drops packets), wait for the COMMAND_ACK -> (ok, text). final_within > 0: an
        IN_PROGRESS ack (the onboard agent launching a stack) waits up to that long for the final one."""
        if self.conn is None:
            return False, f'no link ({self.radio_err})' if self.radio else 'no link'
        t0 = time.monotonic()
        tries = 4 if self.radio else 2
        for attempt in range(tries):
            self.command(cmd, *p, comp=comp)
            end = time.monotonic() + timeout / tries
            while time.monotonic() < end:
                a = self.acks.get(cmd)
                if a and a[1] >= t0:
                    if a[0] == 5 and final_within and time.monotonic() - t0 < final_within:
                        end = time.monotonic() + 1.0          # in progress: keep waiting for the final answer
                    else:
                        return a[0] in (0, 5), self.RESULTS.get(a[0], f'result {a[0]}')
                time.sleep(0.05)
        return False, 'no ACK (link down?)'


# ---------------------------------------------------------------- ROS 2 (host side; DDS reaches the container)
class Ros(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ready = threading.Event()
        self.camera_topic = None
        self.jpeg, self.jpeg_time = None, 0.0
        self.video2 = None

    def run(self):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        from mavros_msgs.msg import State as MavState
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rcl_interfaces.srv import SetParameters
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        n = self.node = Node('gcs_dashboard' if MARKER == 'qr' else 'gcs_dashboard_aruco')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.Image, self.Trigger, self.SetParameters, self.String = Image, Trigger, SetParameters, String
        self.qos_img = qos_profile_sensor_data

        def stamp(key):
            S.ros_time[key] = time.monotonic()

        def on_state(m):
            with S.lock:
                stamp('state')
                S.ros['state'] = m.data
                if m.data == 'TAKEOFF' and S.mission_t0 is None:
                    S.mission_t0 = clock_s()
                if m.data.startswith('SEARCH_DELIVERY') and S.zone_t0 is None:
                    S.zone_t0 = clock_s()

        def on_log(m):
            with S.lock:
                stamp('log')
                try:
                    S.ros['log'] = (S.ros['log'] + [json.loads(m.data)])[-200:]
                except ValueError:
                    pass

        def on_qrs(m):
            with S.lock:
                stamp('qrs')
                try:
                    d = json.loads(m.data)
                    S.ros['qrs'] = d.get('qrs', [])
                    S.ros['candidates'] = d.get('candidates', [])
                    if d.get('start_qr'):
                        S.ros['start_qr'] = d.get('start_qr')
                    if d.get('pairwise_distances_m'):
                        S.ros['pairwise_distances_m'] = d.get('pairwise_distances_m', [])
                except ValueError:
                    pass

        def on_start_qr(m):
            with S.lock:
                stamp('start_qr')
                try:
                    S.ros['start_qr'] = json.loads(m.data).get('payload')
                except ValueError:
                    S.ros['start_qr'] = m.data.strip() or None

        def on_drone_qrs(m):
            with S.lock:
                stamp('drone_qrs')
                try:
                    d = json.loads(m.data)
                    S.ros['drone_qrs'] = d.get('detected_qrs', [])
                    if d.get('pairwise_distances_m'):
                        S.ros['pairwise_distances_m'] = d.get('pairwise_distances_m', [])
                except ValueError:
                    pass

        def on_target(m):
            with S.lock:
                S.ros['target'] = (m.pose.position.x, m.pose.position.y)

        def on_odom(m):
            p = m.pose.pose
            q = p.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            with S.lock:
                stamp('odom')
                S.ros['pose'] = (p.position.x, p.position.y, p.position.z, yaw)
                tr = S.ros['track']
                if not tr or math.dist(tr[-1], (p.position.x, p.position.y)) > 0.3:
                    S.ros['track'] = (tr + [(round(p.position.x, 2), round(p.position.y, 2))])[-3000:]

        def on_node_state(key):
            def cb(m):
                with S.lock:
                    stamp(key)
                    S.ros['states'][key] = m.data
            return cb

        def on_cov(m):
            with S.lock:
                stamp('coverage')
                try:
                    S.ros['coverage'] = json.loads(m.data)
                except ValueError:
                    pass

        def on_mavros(m):
            with S.lock:
                stamp('mavros')
                S.ros['mavros_connected'] = m.connected

        n.create_subscription(String, '/mission/state', on_state, 10)
        n.create_subscription(Clock, '/clock', lambda m: setattr(
            S, 'sim_clock', (m.clock.sec + m.clock.nanosec * 1e-9, time.monotonic())), qos_profile_sensor_data)
        n.create_subscription(String, '/mission/log', on_log, 50)
        n.create_subscription(String, '/delivery/qrs', on_qrs, 10)
        n.create_subscription(String, '/mission/start_qr', on_start_qr, latched)
        n.create_subscription(String, '/drone_qr/detections', on_drone_qrs, 10)
        n.create_subscription(PoseStamped, '/delivery/target', on_target, latched)
        n.create_subscription(Odometry, '/nav/odom', on_odom, 10)
        self.red_arr, self.last_map = None, None
        n.create_subscription(OccupancyGrid, '/delivery/map', self.on_map, latched)
        n.create_subscription(OccupancyGrid, '/delivery/red', self.on_red, latched)
        n.create_subscription(String, '/delivery/coverage', on_cov, 10)
        n.create_subscription(MavState, '/mavros/state', on_mavros, 10)
        for key, topic in (('explore', '/delivery/explore/state'), ('corridor', '/nav/corridor/state'),
                           ('entry', '/nav/entry/state'),
                           # corridor:=aerothon: its vio_to_px4 (cuVSLAM health gate) and EKF3 source set
                           ('vio', '/vio/health'), ('ekf_source', '/vio/ekf_source_state')):
            n.create_subscription(String, topic, on_node_state(key), 10)
        from sensor_msgs.msg import LaserScan
        n.create_subscription(LaserScan, '/vio/obstacle_scan', lambda m: stamp('scan'), qos_profile_sensor_data)

        def on_vio_pose(m):                           # cuVSLAM (corridor:=aerothon): its own frame, where it started
            p = m.pose.pose.position
            with S.lock:
                stamp('vio_pose')
                S.ros['vio_pose'] = (p.x, p.y, p.z)
        n.create_subscription(Odometry, '/visual_slam/tracking/odometry', on_vio_pose, qos_profile_sensor_data)
        self.img_sub = None
        self.cli_start = n.create_client(Trigger, '/mission/start')
        self.cli_abort = n.create_client(Trigger, '/mission/abort')
        self.cli_operator = n.create_client(Trigger, '/mission/operator')
        self.cli_params = n.create_client(SetParameters, '/mission_supervisor/set_parameters')
        self.cli_explore_params = n.create_client(SetParameters, '/frontier_explorer/set_parameters')
        self.pub_qr = n.create_publisher(String, '/mission/start_qr', latched)
        self.set_camera('qr_hud')
        ex = MultiThreadedExecutor()
        ex.add_node(n)
        self.ready.set()
        ex.spin()

    # ---- costmap -> PNG (+ where it lies in odom)
    def on_red(self, m):
        """The red zones themselves (no margin): drawn solid over the no-fly margin of /delivery/map."""
        self.red_arr = np.array(m.data, np.int16).reshape(m.info.height, m.info.width) >= 100
        S.red_check = red_sanity(self.red_arr, m.info.resolution)
        if self.last_map is not None:
            self.on_map(self.last_map)

    def on_map(self, m):
        self.last_map = m
        a = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
        img = np.zeros((m.info.height, m.info.width, 4), np.uint8)
        img[a < 0] = (90, 90, 90, 90)                 # unexplored
        img[a == 0] = (40, 160, 70, 70)               # explored
        img[(a > 0) & (a < 100)] = (255, 150, 0, 220)  # rejected QR boxes
        img[a >= 100] = (230, 90, 90, 110)            # no-fly: red zone margin, zone boundary (light)
        if self.red_arr is not None and self.red_arr.shape == a.shape:
            img[self.red_arr] = (235, 0, 0, 255)      # the red zone itself (solid)
        ok, png = cv2.imencode('.png', cv2.cvtColor(img[::-1], cv2.COLOR_RGBA2BGRA))
        with S.lock:
            S.costmap_png = png.tobytes() if ok else None
            S.costmap_meta = {'x0': m.info.origin.position.x, 'y0': m.info.origin.position.y,
                              'w': m.info.width * m.info.resolution, 'h': m.info.height * m.info.resolution}
            S.costmap_ver += 1
            S.ros_time['map'] = time.monotonic()

    # ---- camera
    def set_camera(self, name):
        if self.img_sub is not None:
            self.node.destroy_subscription(self.img_sub)
            self.img_sub = None
        if self.video2 is not None:
            self.video2.release()
            self.video2 = None
        self.camera_topic = name
        if name == 'video2':
            threading.Thread(target=self.video2_loop, daemon=True).start()
        elif name == 'qr_hud':
            threading.Thread(target=self.qr_hud_loop, daemon=True).start()
        else:
            self.img_sub = self.node.create_subscription(self.Image, CAMERAS[name], self.on_image, self.qos_img)

    def qr_hud_loop(self):
        import urllib.request
        urls = (
            'http://127.0.0.1:8091/annotated.jpg',
            'http://127.0.0.1:8091/frame.jpg',
            'http://127.0.0.1:8090/frame.jpg',
        )
        while self.camera_topic == 'qr_hud':
            got = False
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=0.6) as resp:
                        if resp.status == 200:
                            jpg = resp.read()
                            if jpg:
                                now = time.monotonic()
                                self.jpeg, self.jpeg_time = jpg, now
                                S.ros_time['camera'] = now
                                got = True
                                break
                except Exception:
                    continue
            time.sleep(0.22 if got else 0.5)

    def on_image(self, m):
        now = time.monotonic()
        if now - self.jpeg_time < 0.2:               # <= 5 Hz to the browser
            return
        a = np.frombuffer(m.data, np.uint8)
        if m.encoding in ('mono8', '8UC1'):
            img = a.reshape(m.height, m.width)
        else:
            img = a.reshape(m.height, m.width, -1)[..., :3]
            if m.encoding == 'rgb8':
                img = img[..., ::-1]
        scale = 640 / max(m.width, 1)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.jpeg, self.jpeg_time = jpg.tobytes(), now
            S.ros_time['camera'] = now

    def video2_loop(self):
        cap = cv2.VideoCapture('/dev/video2')
        self.video2 = cap
        while self.camera_topic == 'video2' and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            now = time.monotonic()
            if now - self.jpeg_time >= 0.2:
                ok, jpg = cv2.imencode('.jpg', cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))),
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg, self.jpeg_time = jpg.tobytes(), now
                    S.ros_time['camera'] = now
        cap.release()

    # ---- supervisor
    def call(self, client, req, timeout=15.0):
        if not client.wait_for_service(timeout_sec=3.0):
            return False, f'{client.srv_name} not available (mission stack not running?)'
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.05)
        if not fut.done():
            return False, f'{client.srv_name} timed out'
        r = fut.result()
        return getattr(r, 'success', True), getattr(r, 'message', 'ok')

    def trigger(self, name, timeout=10.0):
        cli = self.node.create_client(self.Trigger, name)
        try:
            return self.call(cli, self.Trigger.Request(), timeout)
        finally:
            self.node.destroy_client(cli)

    def set_supervisor(self, **kv):
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

        def pv(v):
            if isinstance(v, bool):
                return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v)
            return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(v))
        req = self.SetParameters.Request(parameters=[Parameter(name=k, value=pv(v)) for k, v in kv.items()])
        return self.call(self.cli_params, req)


    def set_alt_offset(self, offset):
        """The explorer's search-height trim (alt_offset, m) -> (ok, message)."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        req = self.SetParameters.Request(parameters=[Parameter(name='alt_offset', value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(offset)))])
        cli = self.cli_explore_params
        if not cli.wait_for_service(timeout_sec=3.0):
            return False, 'the explorer is not running (mission stack up?)'
        fut = cli.call_async(req)
        end = time.monotonic() + 5.0
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.05)
        if not fut.done():
            return False, 'the explorer did not answer'
        r = fut.result().results[0]
        return r.successful, r.reason or 'ok'


ALT_TRIM_STEP, ALT_TRIM_MIN, ALT_TRIM_MAX = 0.5, -9.0, 0.5   # m: up to 9.5 m (never above 10: airspace); the explorer keeps it above its alt_min


def alt_trim(delta):
    """The search height up / down by delta (the +-0.5 m buttons) -> (ok, message). Offboard: the VRX explorer."""
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        return False, 'delta: metres, e.g. -0.5'
    if STACK.mode == 'offboard':
        return vrx_post('alt', {'delta': delta})
    new = round(min(ALT_TRIM_MAX, max(ALT_TRIM_MIN, S.alt_offset + delta)), 2)
    success, msg = ROS.set_alt_offset(new)
    if not success:
        return False, f'height trim: {msg}'
    S.alt_offset = new
    return True, f'search height trim {new:+.1f} m'


# ---------------------------------------------------------------- the stack (sim / real)
class Stack:
    def __init__(self):
        self.proc, self.mode, self.run, self.started = None, None, None, None
        self.real_conf = HERE / 'router/real.conf'
        # one Launch / Stop at a time. A Stop pressed during the old Reset (stop + launch) finished first, then the
        # Reset's launch started a new stack - "it relaunches on its own" (2026-10-27 19:13). Now only a Launch
        # press starts a stack, and a Stop while one is starting stops it.
        self.lock = threading.Lock()
        self.stopping = False
        self.go = None               # sim: {'segment', ...} - the mission starts by itself once the stack is READY
        self.go_msg = ''
        self.video = ''              # offboard: the VRX's USB video device
        self.code = ''               # mission code version: '' = the working tree, else a commit (code_version.py)
        self.corridor = 'current'    # corridor navigation: current (ours) | aerothon (~/AeroTHON_2026, its container)
        self.vio_fc = False          # corridor aerothon: its VIO -> ArduPilot EKF3 (source set 2 in the tunnels)
        self.world = 'competition'
        self.offboard_seg = 'qr'
        self.marker = MARKER         # qr | aruco: the stack's boards (AEROTHON_MARKER in its environment)

    def running(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        pids = self.launches()
        if pids and self.mode is None:                # a stack this dashboard didn't start (it was restarted)
            try:
                cmd = Path(f'/proc/{pids[0]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                self.mode = 'real' if ('real_mission' in cmd or 'real_link' in cmd) else \
                    'offboard' if 'offboard_vrx' in cmd else 'sim'
                self.run = next((a.split(':=', 1)[1] for a in cmd.split() if a.startswith('run:=')), self.run)
                self.corridor = 'aerothon' if 'corridor:=aerothon' in cmd else 'current'
                self.vio_fc = 'vio_fc:=1' in cmd
                env = Path(f'/proc/{pids[0]}/environ').read_bytes().split(b'\0')
                self.marker = 'aruco' if b'AEROTHON_MARKER=aruco' in env else 'qr'
            except OSError:
                pass
        return bool(pids)

    def write_real_conf(self, link):
        """router/real.conf with the dashboard's Pixhawk link: '/dev/ttyACM0:921600' (USB), or 'udp:14555' (a UDP
        stream sent to this port). The SiK radio is on the Pixhawk (GPS2), not on the Jetson."""
        text = (HERE / 'router/real.conf').read_text()
        head, _, rest = text.partition('[UartEndpoint fcu]')
        rest = rest.split('\n\n', 1)[1] if '\n\n' in rest else ''
        if link.startswith('udp:'):
            ep = f'[UdpEndpoint fcu]\nMode = Server\nAddress = 0.0.0.0\nPort = {int(link[4:])}\n'
        else:
            dev, _, baud = link.partition(':')
            ep = f'[UartEndpoint fcu]\nDevice = {dev}\nBaud = {int(baud or 57600)}\n'
        self.real_conf = DATA / 'real_router.conf'
        self.real_conf.write_text(head + ep + '\n' + rest)

    def start(self, mode, world, geofence_json, link='/dev/ttyACM0:921600', gcs_ip=''):
        if self.stopping:
            return False, 'the stack is stopping - Launch again when it shows stopped'
        if not self.lock.acquire(blocking=False):
            return False, 'a Launch / Stop is in progress'
        try:
            return self._start(mode, world, geofence_json, link, gcs_ip)
        finally:
            self.lock.release()

    def _start(self, mode, world, geofence_json, link, gcs_ip):
        if self.running():
            return False, f'already running ({self.mode})'
        if mode == 'real':
            # the whole onboard stack (real_mission.launch.py: router + MAVROS on the Pixhawk, RealSense, Arducam,
            # navigation, supervisor); it needs the geofence - the real mission has no world file
            gf = DATA / 'mission_geofence.json'
            if not gf.exists():
                return False, 'no geofence applied: Geofence -> Save + Apply first (the real mission needs it)'
            geofence_json = str(gf)
            dev = link.partition(':')[0]
            onboard = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'aerothon-router']).returncode == 0
            if not onboard and not link.startswith('udp:') and not Path(dev).exists():
                return False, f'telemetry link {link!r}: {dev} does not exist (the Pixhawk USB: /dev/ttyACM0)'
        if mode == 'sim' and self.marker == 'aruco':
            world, err = aruco_world(world)
            if err:
                return False, err
        env = dict(os.environ, FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'), AEROTHON_MARKER=self.marker)
        self.run = f'{mode}_{world if mode == "sim" else "drone"}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        self.world = world
        self.offboard_seg = (self.go or {}).get('segment') or 'qr'
        log = open(WS / 'runs' / f'{self.run}_console.log', 'w')
        if mode == 'offboard':
            # the Jetson on the ground: VRX video + the radio (or the bench SITL), the 8094 explorer flies the QR area
            seg = (self.go or {}).get('segment') or 'qr'
            cmd = ['ros2', 'launch', str(LAUNCH_ALL / 'offboard_vrx.launch.py'), f'run:={self.run}',
                   f'link:={link}', f'segment:={seg}', f'alt:={OFFBOARD_ALT:g}'] + \
                ([f'video:={self.video}'] if self.video else [])
            if geofence_json:
                cmd.append(f'world_json:={geofence_json}')
        elif mode == 'sim':
            cmd = ['ros2', 'launch', str(LAUNCH_ALL / 'sim_mission.launch.py'), f'world:={world}', f'run:={self.run}',
                   'autostart:=0'] + ([f'code:={self.code}'] if self.code else []) + self.corridor_args()
            if geofence_json:
                cmd.append(f'world_json:={geofence_json}')
        else:
            cmd = ['ros2', 'launch', str(LAUNCH_ALL / 'real_mission.launch.py'), f'run:={self.run}', f'link:={link}',
                   f'world_json:={geofence_json}'] + ([f'code:={self.code}'] if self.code else []) + self.corridor_args()
            if gcs_ip:
                cmd.append(f'gcs_ip:={gcs_ip}')
        self.proc = subprocess.Popen(['bash', '-c', 'source /opt/ros/humble/setup.bash && exec "$@"', '_', *cmd],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.mode, self.started = mode, time.time()
        if mode == 'sim' and self.go and self.go.get('segment'):
            threading.Thread(target=self.launch_and_go, args=(self.run, dict(self.go)), daemon=True).start()
        with S.lock:
            S.mission_t0 = S.zone_t0 = None
            S.ros.update(track=[], log=[], target=None, qrs=[], state='')
            # the map starts clean: the last run's flight path (MAVLink track) and costmap stayed drawn (2026-10-27)
            S.tel_track = []
            S.costmap_png, S.costmap_meta = None, None
        return True, f'{mode} stack starting (run {self.run}{", ArUco boards" if self.marker == "aruco" else ""})'

    def corridor_args(self):
        return [f'corridor:={self.corridor}', f'vio_fc:={int(self.corridor == "aerothon" and self.vio_fc)}']

    def launch_and_go(self, run, go):
        """Sim: one Launch press = the stack + the chosen segment. Waits for READY and a green preflight (the GPS
        origin / supervisor come up a little after READY), then /mission/start. A Stop or another run ends it."""
        seg = go['segment']
        live = lambda: self.run == run and not self.stopping and self.running()  # noqa: E731
        end = time.monotonic() + 420
        self.go_msg = f'{seg}: waiting for READY'
        while live() and stack_phase() != 'ready' and time.monotonic() < end:
            time.sleep(2)
        msg = 'stack not READY in 7 min'
        while live() and time.monotonic() < end:
            success, msg = start_mission(seg, go.get('payload', ''), go.get('read_qr', True), go.get('override'))
            if success:
                self.go_msg = f'{seg} started'
                return
            self.go_msg = f'{seg}: {msg} - retrying'
            time.sleep(5)
        self.go_msg = '' if not live() else f'{seg} NOT started: {msg}'

    @staticmethod
    def launches():
        """PIDs of sim_mission / real_link launches, ours or not (a restarted dashboard doesn't own its stack)."""
        r = subprocess.run(['pgrep', '-f', r'^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch .*(sim_mission|real_mission|real_link|offboard_vrx)'
                            r'\.launch\.py'], capture_output=True, text=True)
        return [int(p) for p in r.stdout.split()]

    def stop(self):
        if self.stopping:
            return False, 'already stopping'
        self.stopping = True
        try:
            with self.lock:
                return self._stop()
        finally:
            self.stopping = False

    def _stop(self):
        pids = self.launches()
        if not pids and not self.running():
            self.restore_onboard_router()
            return False, 'not running'
        for p in pids:
            try:
                os.kill(p, signal.SIGINT)             # the launch's Ctrl-C: it stops everything (stop_all.sh)
            except ProcessLookupError:
                pass
        end = time.monotonic() + 150
        while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in pids):
            time.sleep(1)
        subprocess.run([str(LAUNCH_ALL / 'scripts/stop_all.sh')], timeout=90, check=False)   # anything left
        self.proc = None
        self.restore_onboard_router()
        return True, 'stopped (Gazebo, autopilot SITL, MAVROS, mission nodes)'

    @staticmethod
    def restore_onboard_router():
        """sim_mission pauses the always-on onboard router for a sim run and resumes it on stop - also here, in case
        the launch died before its stop handler ran."""
        sc = lambda *a: subprocess.run(['systemctl', '--user', *a, 'aerothon-router'], capture_output=True, timeout=20)  # noqa: E731
        if sc('is-enabled', '--quiet').returncode == 0 and sc('is-active', '--quiet').returncode != 0:
            sc('start')


STACK = Stack()


# ---------------------------------------------------------------- geofence
GF_FILE = DATA / 'geofence.json'
# GPS slots (user, 2026-10-28): the organisers give 6 points and don't say whether 4 of them are the whole ARENA or
# the QR AREA - fill whichever set it is; the 2 others go in EXTRA (meaning unknown: kept + drawn only)
GF_CAPS = {'arena': 4, 'qr': 4, 'extra': 2}
# the course dimensions: the organisers' defaults (user, 2026-10-28), editable when new info comes. Tunnel width /
# length / gap in m, heights in ft (as given); the gap = wall to wall between the entry and the exit tunnel
# tunnel_length: 20 ft 5 in = 6.22 m (user, 2026-10-30, "given inaccuracies"; was 10 m) - exit detection is on the walls
# ending, from 0.6 x it; the hard stop at the length + 5 m
COURSE_DEFAULT = {'tunnel_width': 3.5, 'tunnel_height_ft': 10.0, 'tunnel_ceiling_ft': 8.0, 'tunnel_length': 6.22,
                  'tunnel_gap': 0.0, 'qr_width': 30.0}
FT = 0.3048


def load_gf():
    try:
        gf = json.loads(GF_FILE.read_text())
    except (OSError, ValueError):
        gf = {}
    if 'arena' not in gf and ('field' in gf or 'delivery' in gf):   # the old field (<= 8) + delivery (4) file
        gf['arena'] = gf.get('field', []) if len(gf.get('field', [])) == 4 else []
        gf['qr'] = gf.get('delivery', [])
    gf.pop('field', None)
    gf.pop('delivery', None)
    for k in GF_CAPS:
        gf.setdefault(k, [])
    gf['course'] = {**COURSE_DEFAULT, **{k: float(v) for k, v in (gf.get('course') or {}).items() if k in COURSE_DEFAULT}}
    gf.setdefault('world', 'competition')
    return gf


def save_gf(gf):
    DATA.mkdir(parents=True, exist_ok=True)
    GF_FILE.write_text(json.dumps(gf, indent=2))


def ccw(pts):
    """The corners in order round their centroid (typed or captured in any order)."""
    if len(pts) < 3:
        return list(pts)
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def gf_local(gf):
    """The GPS slots in odom (needs the autopilot's GPS origin) + what the nodes use: `field` = the keep-in (the arena;
    QR area only: its hull with the start area, home +-6 m), `delivery` = the QR area (given, or the arena until the
    entry tunnel's exit line is known in flight: qr_source 'tunnel_exit'), `back` = its far two corners."""
    if S.origin is None:
        return None
    lat0, lon0, _ = S.origin
    loc = {k: [gps_to_local(p[0], p[1], lat0, lon0) for p in gf.get(k, [])] for k in GF_CAPS}
    arena, qr = ccw(loc['arena']), ccw(loc['qr'])
    loc['field'] = arena if len(arena) == 4 else hull(qr + [(sx * 6.0, sy * 6.0) for sx in (-1, 1) for sy in (-1, 1)]) \
        if len(qr) == 4 else []
    loc['delivery'] = qr if len(qr) == 4 else arena
    loc['qr_source'] = 'given' if len(qr) == 4 else 'tunnel_exit' if len(arena) == 4 else None
    # the back wall = the two corners farthest from home (the take-off point ~ the origin)
    loc['back'] = sorted(loc['delivery'], key=lambda p: -math.hypot(*p))[:2] if len(loc['delivery']) == 4 else []
    return loc


def aruco_world(world):
    """The ArUco twin of a sim world (<world>_aruco: same course, ArUco boards) -> (name, error or '')."""
    name = world if world.endswith('_aruco') else f'{world}_aruco'
    if not (WORLDS / f'{name}.sdf').exists():
        r = subprocess.run(['python3', str(WORLDS.parent / 'make_aruco_world.py'), world], capture_output=True,
                           text=True, timeout=60)
        if r.returncode:
            return name, f'could not make the ArUco world {name}: {(r.stderr or r.stdout).strip()[-300:]}'
    return name, ''


def gf_from_world(world):
    """Sim: the arena (start area + corridors + zone, 4 corners) and the QR area, in GPS around the origin."""
    w = json.load(open(WORLDS / f'{world}.json'))
    z = w['delivery_zone']
    ys = [c['centre_y'] + s * (c['width'] / 2 + 0.6) for c in w['corridors'] for s in (-1, 1)]
    m = 1.0                                           # the arena reaches 1 m beyond the zone
    ymin, ymax = min(ys + [-4.0, z['y_from'] - m]), max(ys + [4.0, z['y_to'] + m])
    arena = [(-4.0, ymin), (z['x_to'] + m, ymin), (z['x_to'] + m, ymax), (-4.0, ymax)]
    qr = [(z['x_from'], z['y_from']), (z['x_to'], z['y_from']), (z['x_to'], z['y_to']), (z['x_from'], z['y_to'])]
    lat0, lon0, _ = S.origin
    to = lambda pts: [list(local_to_gps(x, y, lat0, lon0)) for x, y in pts]  # noqa: E731
    gf = load_gf()
    gf.update({'arena': to(arena), 'qr': to(qr), 'extra': [], 'world': world})
    return gf


def apply_gf(gf, mode):
    """-> (ok, message, path): mission_geofence.json for the nodes (world_json schema)."""
    loc = gf_local(gf)
    if loc is None:
        return False, 'no GPS origin from the autopilot yet (MAVLink link up?)', None
    na, nq = len(loc['arena']), len(loc['qr'])
    if na not in (0, 4) or nq not in (0, 4) or not (na or nq):
        return False, f'need 4 arena corners and/or 4 QR-area corners (have arena {na}, QR area {nq})', None
    if na and nq and not all(inside_or_on(x, y, loc['field'], tol=1.0) for x, y in loc['qr']):
        return False, 'the QR area is not inside the arena', None
    c = gf['course']
    course = {'tunnel_width': c['tunnel_width'], 'tunnel_height': round(c['tunnel_height_ft'] * FT, 3),
              'tunnel_ceiling': round(c['tunnel_ceiling_ft'] * FT, 3), 'tunnel_length': c['tunnel_length'],
              'tunnel_gap': c['tunnel_gap'], 'qr_width': c['qr_width']}
    if course['tunnel_ceiling'] > course['tunnel_height']:
        return False, 'the tunnel ceiling (max flight height inside) is above the tunnel height', None
    r2 = lambda pts: [[round(x, 2), round(y, 2)] for x, y in pts]  # noqa: E731
    xs, ys = [p[0] for p in loc['delivery']], [p[1] for p in loc['delivery']]
    out = {'frame': 'odom = the autopilot local ENU around its GPS origin', 'origin_gps': list(S.origin),
           'source': 'ground station geofence', 'time': time.ctime(),
           # the QR area's bounding box; qr_source 'tunnel_exit': the arena's, until the supervisor cuts it at the
           # entry tunnel's exit line in flight (the QR area's front wall = that line, user 2026-10-28)
           'delivery_zone': {'x_from': round(min(xs), 2), 'x_to': round(max(xs), 2),
                             'y_from': round(min(ys), 2), 'y_to': round(max(ys), 2)},
           'delivery_polygon': r2(loc['delivery']), 'qr_source': loc['qr_source'], 'qr_back': r2(loc['back']),
           'field_polygon': r2(loc['field']), 'arena_polygon': r2(loc['arena']) if na else [],
           'extra_points': r2(loc['extra']), 'course': course,
           'corridors': [], 'red_zones': [], 'gps': gf}
    if mode == 'sim':                                 # the sim's tunnels are known: keep their ceiling boxes
        try:
            w = json.load(open(WORLDS / f'{gf.get("world", "competition")}.json'))
            out['corridors'] = w['corridors']
            # never the boards / target / red zones: they move on the day and the mission must find them itself
        except (OSError, ValueError, KeyError):
            pass
    path = DATA / 'mission_geofence.json'
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    # the container mounts the workspace: the same file there
    back = loc['back']
    width = f', back wall {math.dist(*back):.1f} m (QR width {c["qr_width"]:g})' if len(back) == 2 else ''
    return True, (f'geofence applied: QR area {"given" if nq else "= arena, front from the tunnel exit in flight"}'
                  f'{width}; tunnels {c["tunnel_width"]:g} x {c["tunnel_length"]:g} m, ceiling '
                  f'{course["tunnel_ceiling"]:.2f} m'), path


# ---------------------------------------------------------------- preflight + alerts
def fresh(key, age, table=None):
    # 1 Hz topics arrive ~2.5 s apart on the slow Jetson sim: limits of 5 s, not 3
    t = (table if table is not None else S.ros_time).get(key)
    return t is not None and time.monotonic() - t < age


def mem_available_mb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable'):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def cpu_temp():
    temps = []
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            temps.append(int(p.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def preflight_offboard():
    """The offboard Jetson: telemetry over the radio + the explorer's VRX video, no ROS stack."""
    tel, v = S.tel, vrx_state() or {}
    checks = [
        ('MAVLink link (radio -> router -> dashboard)', fresh('HEARTBEAT', 3, S.tel_time), 'heartbeat from ArduCopter', True),
        ('GPS 3D fix', (tel.get('gps_fix') or 0) >= 3, f'fix {tel.get("gps_fix")}, {tel.get("sats")} sats', True),
        ('GPS origin known', S.origin is not None, 'map <-> GPS', True),
        ('Battery >= 50 %', (tel.get('batt_pct') if tel.get('batt_pct') is not None else 0) >= 50,
         f'{tel.get("batt_pct")} %, {tel.get("batt_v")} V', True),
        ('Disarmed on the ground', tel.get('armed') is False and abs(tel.get('alt_rel') or 0) < 0.5,
         f'armed {tel.get("armed")}, alt {tel.get("alt_rel")}', True),
        ('VRX explorer up (:8094)', bool(v), 'vrx_explore.py', True),
        ('VRX video', bool(v) and not v.get('video_err'), v.get('video_err') or 'frames', True),
        ('Geofence applied', (DATA / 'mission_geofence.json').exists(), 'the zone for the explorer (else the sim '
         "world's)", False),
    ]
    return [{'name': n, 'ok': bool(ok), 'detail': d, 'critical': c} for n, ok, d, c in checks]


def preflight_radio():
    """The final flight, seen from the ground: the radio link + what the onboard Jetson reports."""
    tel, a, det = S.tel, MAV.agent, MAV.agent_detail
    checks = [
        ('Telemetry radio', MAV.conn is not None, MAV.radio_err or RADIO, True),
        ('Autopilot heartbeat over the radio', fresh('HEARTBEAT', 3, S.tel_time), 'ArduCopter', True),
        ('Onboard Jetson (mav_agent) heard', fresh('agent', 5, S.tel_time), 'heartbeat 1/194', True),
        ('Onboard stack READY', a.get('p') == 'ready', a.get('p') or 'no status', True),
        ('Onboard preflight green', fresh('agent_status', 8, S.tel_time) and not a.get('pf'),
         ', '.join(det.get('f') or []) or f'{a.get("pf")} failing', True),
        ('GPS 3D fix', (tel.get('gps_fix') or 0) >= 3, f'fix {tel.get("gps_fix")}, {tel.get("sats")} sats', True),
        ('Battery >= 50 %', (tel.get('batt_pct') if tel.get('batt_pct') is not None else 0) >= 50,
         f'{tel.get("batt_pct")} %, {tel.get("batt_v")} V', True),
        ('Disarmed on the ground', tel.get('armed') is False, f'armed {tel.get("armed")}', True),
    ]
    return [{'name': n, 'ok': bool(ok), 'detail': d, 'critical': c} for n, ok, d, c in checks]


def preflight(mode):
    if RADIO:
        return preflight_radio()
    if mode == 'offboard':
        return preflight_offboard()
    tel, gf = S.tel, load_gf()
    loc = gf_local(gf)
    applied = (DATA / 'mission_geofence.json').exists()
    mem = mem_available_mb()
    temp = cpu_temp()
    checks = [
        ('MAVLink link (router -> dashboard)', fresh('HEARTBEAT', 3, S.tel_time), 'heartbeat from PX4', True),
        ('MAVROS connected', bool(S.ros.get('mavros_connected')) and fresh('mavros', 5), '/mavros/state', True),
        ('GPS 3D fix', (tel.get('gps_fix') or 0) >= 3, f'fix {tel.get("gps_fix")}, {tel.get("sats")} sats, '
                                                         f'hdop {tel.get("hdop")}', True),
        ('GPS satellites >= 8', (tel.get('sats') or 0) >= 8, f'{tel.get("sats")} sats', mode == 'real'),
        ('GPS origin known', S.origin is not None, 'needed for GPS <-> map', True),
        ('Battery >= 50 %', (tel.get('batt_pct') if tel.get('batt_pct') is not None else 0) >= 50,
         f'{tel.get("batt_pct")} %, {tel.get("batt_v")} V', True),
        ('Disarmed on the ground', tel.get('armed') is False and abs(tel.get('alt_rel') or 0) < 0.5,
         f'armed {tel.get("armed")}, alt {tel.get("alt_rel")}', True),
        ('Arena and/or QR area: 4 corners each', len(gf['arena']) in (0, 4) and len(gf['qr']) in (0, 4)
         and bool(gf['arena'] or gf['qr']), f'arena {len(gf["arena"])}, QR area {len(gf["qr"])}, extra {len(gf["extra"])}', True),
        ('QR area inside the arena', bool(loc) and len(loc['field']) >= 3 and len(loc['delivery']) == 4
         and all(inside_or_on(x, y, loc['field'], tol=1.0) for x, y in loc['delivery']),
         (loc or {}).get('qr_source') == 'tunnel_exit' and 'front wall from the tunnel exit in flight' or 'given', True),
        ('Geofence applied', applied, 'mission_geofence.json written', True),
        ('Drone inside the field boundary', bool(loc) and len(loc['field']) >= 3 and drone_pose()[0] is not None
         and in_poly(drone_pose()[0][0], drone_pose()[0][1], loc['field']), f'from {drone_pose()[2]}', True),
        ('Mission supervisor up', fresh('state', 5), f'state {S.ros.get("state")!r}', True),
        ('Delivery search (Nav2 + explorer) up', fresh('explore', 5), S.ros['states'].get('explore', ''), True),
        ('Corridor navigation up', fresh('corridor', 5), S.ros['states'].get('corridor', ''), True),
        ('Delivery costmap', fresh('map', 5), 'delivery_mapper /delivery/map', True),
        ('Forward depth (D435 scan)', fresh('scan', 3), '/vio/obstacle_scan', True),
        *([('AeroTHON VIO up (cuVSLAM stereo + IMU)', fresh('vio', 3), S.ros['states'].get('vio', '') +
            ' - "OK" only with enough features in view: on the pad a REJECT is normal, it is reset at the tunnel', True),
           ('VIO -> EKF3', fresh('ekf_source', 3) and (STACK.vio_fc or S.ros['states'].get('ekf_source') == 'gps'),
            f'EKF source {S.ros["states"].get("ekf_source", "?")}; ' +
            ('sent, set 2 in the tunnels (FC needs EK3_SRC2_VELXY 6, VISO_TYPE 1)' if STACK.vio_fc else
             'off: checks only, the FC flies on GPS'), False)]
          if STACK.corridor == 'aerothon' else []),
        ('Camera', fresh('camera', 6), f'feed {ROS.camera_topic}', mode == 'real'),
        ('Arducam calibrated', mode != 'real' or (WS / 'calibration/arducam_fisheye_1640x1232.yaml').exists(),
         'calibration/arducam_fisheye_1640x1232.yaml (else the nominal lens: QR / red-zone positions off)', False),
        # a warning, not a gate: low RAM must not keep the drone from arming (user, 2026-10-28)
        ('Jetson RAM free >= 1 GB', mem is not None and mem >= 1024, f'{mem} MB', False),
        ('Jetson temperature < 85 C', temp is None or temp < 85, f'{temp} C', False),
    ]
    return [{'name': n, 'ok': bool(ok), 'detail': d, 'critical': c} for n, ok, d, c in checks]


# the course has ONE red zone, 10 ft x 10 ft. Red mapped beyond one such square (a second zone, or one far too big)
# means the red detection is wrong - the user's rule: then that code can't be trusted (2026-10-27)
RED_ZONE_M = 3.048
RED_FIT_M = RED_ZONE_M + 1.25                         # + mapping blur / margin: pieces of one zone still fit


def red_sanity(red, res):
    """-> None, or why the red map can't be the course's single 10 ft red zone."""
    if red is None or not red.any():
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    blobs = [st for st in stats[1:] if st[4] * res * res >= 0.25]          # specks below 0.25 m2 ignored
    if not blobs:
        return None
    x0 = min(b[0] for b in blobs) * res
    y0 = min(b[1] for b in blobs) * res
    x1 = max(b[0] + b[2] for b in blobs) * res
    y1 = max(b[1] + b[3] for b in blobs) * res
    w, h = x1 - x0, y1 - y0
    area = sum(b[4] for b in blobs) * res * res
    if w > RED_FIT_M or h > RED_FIT_M:
        return (f'red seen over {w:.1f} x {h:.1f} m in {len(blobs)} patch(es) - ONE {RED_ZONE_M:.2f} m zone '
                f'expected: red detection unreliable')
    if area > 1.6 * RED_ZONE_M ** 2:
        return f'red area {area:.1f} m2 (one zone is {RED_ZONE_M ** 2:.1f} m2): red detection unreliable'
    return None


def alerts():
    out = []
    if S.red_check:
        out.append(('critical', 'RED ZONE CHECK: ' + S.red_check))
    tel = S.tel
    armed = tel.get('armed')
    if armed and not fresh('HEARTBEAT', 3, S.tel_time):
        out.append(('critical', 'MAVLink LINK LOST while armed'))
    if tel.get('batt_pct') is not None and tel['batt_pct'] < 25:
        out.append(('critical', f'BATTERY {tel["batt_pct"]} %'))
    if armed and (tel.get('gps_fix') or 0) < 3:
        out.append(('critical', 'GPS FIX LOST'))
    gf = load_gf()
    loc = gf_local(gf)
    p = drone_pose()[0]
    if armed and loc and len(loc['field']) >= 3 and p and not in_poly(p[0], p[1], loc['field']):
        out.append(('critical', 'OUTSIDE THE FIELD GEOFENCE'))
    if (tel.get('alt_rel') or 0) > 11.5:
        out.append(('critical', f'ALTITUDE {tel["alt_rel"]:.1f} m'))
    st = S.ros.get('state', '')
    if st.startswith(('FAILSAFE', 'CRASH', 'FAILED')):
        out.append(('critical', f'MISSION {st}'))
    if armed and not fresh('state', 5):
        out.append(('warning', 'mission supervisor silent'))
    if S.zone_t0 and st not in ('DONE', 'LAND') and clock_s() - S.zone_t0 > 480 and \
            st.startswith(('SEARCH', 'DELIVER', 'DROP')):
        out.append(('warning', f'QR area {(clock_s() - S.zone_t0) / 60:.1f} min (limit 9)'))
    mem = mem_available_mb()
    if mem is not None and mem < 500:
        out.append(('warning', f'Jetson RAM {mem} MB free'))
    for t, sev, text in S.statustext[-5:]:
        if sev <= 3 and time.time() - t < 30:
            out.append(('critical' if sev <= 2 else 'warning', f'FCU: {text}'))
    return [{'level': lv, 'text': tx} for lv, tx in out]


# ---------------------------------------------------------------- the offboard VRX explorer (:8094)
VRX_URL = 'http://127.0.0.1:8094'
_vrx = {'t': 0.0, 'state': None}


def vrx_state():
    """The explorer's /state.json (offboard mode), cached 1 s; None while it is not up."""
    if STACK.mode != 'offboard':
        return None
    if time.monotonic() - _vrx['t'] > 1.0:
        _vrx['t'] = time.monotonic()
        try:
            with urllib.request.urlopen(VRX_URL + '/state.json', timeout=0.8) as r:
                _vrx['state'] = json.loads(r.read())
        except (OSError, ValueError):
            _vrx['state'] = None
    return _vrx['state']


def vrx_get(_path):
    return vrx_state()


def vrx_post(what, body=None):
    try:
        req = urllib.request.Request(f'{VRX_URL}/api/{what}', data=json.dumps(body or {}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read())
        return bool(d.get('ok')), str(d.get('message', ''))
    except (OSError, ValueError) as e:
        return False, f'VRX explorer: {e}'


# ---------------------------------------------------------------- web
def stack_phase():
    """stopped | starting: ... | ready - what the Launch / mission buttons go by."""
    if RADIO:                                         # the onboard Jetson's own phase, over the radio
        if not fresh('agent_status', 8, S.tel_time):
            return 'no onboard status (Jetson / radio?)'
        return MAV.agent.get('p') or '?'
    if STACK.stopping:
        return 'stopping'
    if not STACK.running():
        return 'stopped'
    if not fresh('HEARTBEAT', 3, S.tel_time):
        return {'real': 'starting: Pixhawk link', 'offboard': 'starting: radio / SITL link'}.get(
            STACK.mode, 'starting: Gazebo + ArduCopter SITL (1-2 min)')
    if STACK.mode == 'offboard':
        return 'ready' if vrx_get('/ws_state') is not None else 'starting: VRX explorer (:8094)'
    if STACK.mode == 'real' and not fresh('mavros', 5):
        return 'starting: MAVROS'
    if not (fresh('state', 5) and fresh('explore', 5)):
        return 'starting: mission stack (1-2 min)'
    return 'ready'


def drone_pose():
    """The drone in odom: from TELEMETRY (MAVLink, sim or real alike); ROS /nav/odom only if MAVLink has none."""
    if S.tel_pose is not None and fresh('LOCAL_POSITION_NED', 2, S.tel_time):
        return S.tel_pose, S.tel_track, 'telemetry'
    return S.ros.get('pose'), S.ros['track'], 'ros'


def snapshot():
    v = vrx_state()                                   # outside the lock: an HTTP call (cached 1 s)
    with S.lock:
        gf = load_gf()
        loc = gf_local(gf)
        p, track, pose_src = drone_pose()
        mode = STACK.mode or 'sim'
        out = {
            'time': time.time(), 'tel': dict(S.tel), 'origin': S.origin,
            'link_age': (time.monotonic() - S.tel_time['HEARTBEAT']) if 'HEARTBEAT' in S.tel_time else None,
            'ros': {'state': S.ros['state'], 'log': S.ros['log'][-60:], 'qrs': S.ros['qrs'],
                    'candidates': S.ros.get('candidates', []), 'start_qr': S.ros.get('start_qr'),
                    'drone_qrs': S.ros.get('drone_qrs', []),
                    'pairwise_distances_m': S.ros.get('pairwise_distances_m', []),
                    'target': S.ros['target'], 'states': S.ros['states'], 'coverage': S.ros['coverage'],
                    'track': track[-1500:], 'pose': p, 'pose_src': pose_src},
            'fsm': FSM[S.mission_type], 'mission_type': S.mission_type, 'vrx': v,
            'mission_s': (clock_s() - S.mission_t0) if S.mission_t0 else None,
            'zone_s': (clock_s() - S.zone_t0) if S.zone_t0 else None,
            'clock': 'sim' if STACK.mode == 'sim' and S.mission_t0 is not None and S.mission_t0 < 1e9 else 'wall',
            'costmap': {'ver': S.costmap_ver, **(S.costmap_meta or {})} if S.costmap_meta else None,
            'geofence': gf, 'geofence_local': loc,
            'stack': {'running': STACK.running(), 'mode': STACK.mode, 'run': STACK.run, 'phase': stack_phase(),
                      'go': STACK.go_msg, 'code': STACK.code, 'corridor': STACK.corridor,
                      'vio_fc': STACK.corridor == 'aerothon' and STACK.vio_fc,
                      'up_s': round(time.time() - STACK.started) if STACK.started and STACK.running() else None},
            'camera': ROS.camera_topic, 'preflight': preflight(mode), 'alerts': alerts(),
            'statustext': S.statustext[-15:],
            'alt_offset': S.alt_offset,
            'vio_pose': {'xyz': S.ros.get('vio_pose'), 'health': S.ros['states'].get('vio', ''),
                         'age': round(time.monotonic() - S.ros_time['vio_pose'], 1) if 'vio_pose' in S.ros_time else None},
            'ages': {k: round(time.monotonic() - t, 1) for k, t in S.ros_time.items()},
        }
        if RADIO:                                     # the onboard mission, from the agent's status
            a = MAV.agent
            out['radio'] = {'dev': RADIO, 'err': MAV.radio_err, 'rssi': S.tel.get('rssi'), 'remrssi': S.tel.get('remrssi'),
                            'agent_age': round(time.monotonic() - S.tel_time['agent'], 1) if 'agent' in S.tel_time else None}
            out['stack'].update(running=True, mode='radio', phase=stack_phase())
            out['ros']['state'] = a.get('s') if a.get('s') not in (None, '-') else ''
            out['ros']['pose'] = S.tel_pose
            out['mission_s'], out['zone_s'] = a.get('t'), a.get('z')
            out['alt_offset'] = a.get('ao')
            seg = {v: k for k, v in SEG_CODE.items()}.get(a.get('sg'), 'full')
            out['fsm'], out['mission_type'] = FSM[seg], seg
            # the boards and the VIO position the agent sends over the radio (32803 / 32802)
            out['ros']['qrs'] = [{'payload': b['p'], 'x': b.get('x', 0.0), 'y': b.get('y', 0.0), 'n': 0,
                                  'status': b.get('st') or 'pending'} for b in MAV.boards.values() if b['p'] != '*target']
            tb = MAV.boards.get('*target')
            out['ros']['target'] = (tb['x'], tb['y']) if tb else None
            vv, va = MAV.vio, S.tel_time.get('agent_vio')
            out['vio_pose'] = {'xyz': vv.get('v'), 'health': vv.get('h', ''),
                               'age': round(time.monotonic() - va + (vv.get('a') or 0), 1) if va else None}
        if v:                                         # offboard: the explorer is the mission
            out['ros'].update(state=v.get('state') or '', log=v.get('log') or [], qrs=[
                {**q, 'x': q.get('x') or 0.0, 'y': q.get('y') or 0.0, 'n': q.get('n') or 0, 'status': q.get('status') or 'pending'}
                for q in v.get('qrs') or []], target=v.get('target'), start_qr=v.get('start_qr'))
            out['ros']['coverage'] = {'explored': v.get('coverage')}
            out['fsm'] = v.get('fsm') or out['fsm']
        return out


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=10)
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str(json.dumps(snapshot(), default=float))
            await asyncio.sleep(0.2)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return ws


async def costmap(request):
    png = S.costmap_png
    if png is None:
        raise web.HTTPNotFound()
    return web.Response(body=png, content_type='image/png', headers={'Cache-Control': 'no-store'})


async def mjpeg(request):
    resp = web.StreamResponse(headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                                       'Cache-Control': 'no-store'})
    await resp.prepare(request)
    last = None
    try:
        while True:
            j = ROS.jpeg
            if j is not None and j is not last:
                last = j
                await resp.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(j)).encode() +
                                 b'\r\n\r\n' + j + b'\r\n')
            await asyncio.sleep(0.1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


def ok(success, message, **extra):
    return web.json_response({'ok': bool(success), 'message': message, **extra})


def gcs_ip(request):
    """The browser's address = the ground laptop: QGroundControl's UDP target on the real drone ('' on the Jetson)."""
    ip = request.remote or ''
    return '' if ip.startswith('127.') or ip == '::1' else ip


async def api(request):
    what = request.match_info['what']
    body = await request.json() if request.can_read_body else {}
    print(f'[api] {time.strftime("%H:%M:%S")} {request.remote} {what} {json.dumps(body)[:200]}', flush=True)
    loop = asyncio.get_running_loop()
    if RADIO and what not in ('geofence', 'capture'):   # those two stay here: the corners are kept on this PC
        return ok(*(await loop.run_in_executor(None, radio_api, what, body)))
    if what == 'operator':                            # mav_agent action 8: the ground took over over the radio
        return ok(*(await loop.run_in_executor(None, lambda: ROS.call(ROS.cli_operator, ROS.Trigger.Request(), 3.0))))
    if what == 'geofence':                            # save (lists of [lat, lon]; course: the dimensions)
        gf = load_gf()
        for k in (*GF_CAPS, 'world'):
            if k in body:
                gf[k] = body[k][:GF_CAPS[k]] if k in GF_CAPS else body[k]
        if isinstance(body.get('course'), dict):
            try:
                gf['course'].update({k: float(v) for k, v in body['course'].items() if k in COURSE_DEFAULT})
            except (TypeError, ValueError):
                return ok(False, 'course dimensions must be numbers')
        save_gf(gf)
        return ok(True, 'geofence saved')
    if what == 'capture':                             # the drone's GPS -> a corner of arena / QR area / extra
        lst = body.get('list')
        if lst not in GF_CAPS or S.tel.get('lat') is None:
            return ok(False, 'no GPS from the drone yet' if S.tel.get('lat') is None else 'list: ' + ' | '.join(GF_CAPS))
        if (S.tel.get('gps_fix') or 0) < 3:
            return ok(False, f'GPS fix {S.tel.get("gps_fix")}: not capturing')
        gf = load_gf()
        cap = GF_CAPS[lst]
        if len(gf[lst]) >= cap:
            return ok(False, f'{lst} already has {cap} corners')
        gf[lst].append([round(S.tel['lat'], 7), round(S.tel['lon'], 7)])
        save_gf(gf)
        return ok(True, f'{lst} corner {len(gf[lst])}: {gf[lst][-1]}')
    if what == 'from_world':                          # sim: fill from the world's ground truth
        if S.origin is None:
            return ok(False, 'no GPS origin from PX4 yet')
        gf = gf_from_world(body.get('world', 'competition'))
        save_gf(gf)
        return ok(True, f'filled from the sim world {gf["world"]}')
    if what == 'apply':
        success, msg, _ = apply_gf(load_gf(), STACK.mode or body.get('mode', 'sim'))
        return ok(success, msg)
    if what == 'stack_start':
        if STACK.running():
            return ok(False, f'a stack is already {stack_phase()} - wait, or Stop it first')
        mode, world = body.get('mode', 'sim'), body.get('world', 'competition')
        gj = DATA / 'mission_geofence.json'
        seg = SEGMENT_ALIAS.get(body.get('segment') or '', body.get('segment') or '')
        if seg and seg not in SEGMENTS:
            return ok(False, 'segment: ' + ' | '.join(SEGMENTS))
        # only the offboard Jetson is limited to the QR area (no tunnel sensors on that drone); 8084 (ArUco) flies every
        # segment in sim / real / radio like 8080 (user, 2026-10-28)
        if mode == 'offboard' and seg not in ('', 'qr', 'qr_drop'):
            return ok(False, 'the offboard Jetson flies the QR area only: QR area only | QR area + drop')
        STACK.go = {'segment': seg, 'payload': body.get('payload') or '', 'read_qr': body.get('read_qr', True),
                    'override': body.get('override')} if seg else None
        STACK.video = (body.get('video') or '').strip()
        STACK.code = (body.get('code') or '').strip()       # which mission code flies (sim / real; offboard: current)
        STACK.corridor = body.get('corridor') or 'current'  # corridor navigation: current | aerothon (sim / real)
        if STACK.corridor not in ('current', 'aerothon'):
            return ok(False, 'corridor: current | aerothon')
        STACK.vio_fc = bool(body.get('vio_fc'))
        STACK.marker = body.get('marker') or MARKER         # the onboard agent passes the radio page's marker
        if STACK.marker not in ('qr', 'aruco'):
            return ok(False, 'marker: qr | aruco')
        if mode == 'offboard' and body.get('link', '').startswith('/dev/ttyACM'):
            body['link'] = 'sitl'
        STACK.go_msg = ''
        return ok(*STACK.start(mode, world, str(gj) if body.get('use_geofence') and gj.exists() else '',
                               body.get('link', '/dev/ttyACM0:921600'), gcs_ip(request)))
    if what == 'code_versions':                       # the Launch row's "mission code" choice
        r = await loop.run_in_executor(None, lambda: subprocess.run(
            [str(LAUNCH_ALL / 'scripts/code_version.py'), 'list'], capture_output=True, text=True, timeout=20))
        try:
            return ok(True, 'versions', versions=json.loads(r.stdout))
        except ValueError:
            return ok(False, f'git log: {r.stderr.strip()[:200]}', versions=[])
    if what == 'status':                              # lenovo/aerothon.sh status
        return ok(True, stack_phase(), mode=STACK.mode, run=STACK.run, state=S.ros.get('state'))
    if what == 'stack_stop':
        return ok(*(await loop.run_in_executor(None, STACK.stop)))
    if what == 'alt_trim':                            # the search height +-0.5 m in flight (also from the radio)
        return ok(*(await loop.run_in_executor(None, alt_trim, body.get('delta'))))
    if what == 'mission_start':                       # full | qr | qr_drop | tunnel
        S.alt_offset = 0.0                            # the explorer drops its trim at the search start too
        return ok(*(await loop.run_in_executor(None, lambda: start_mission(
            body.get('type', 'full'), body.get('payload') or '', body.get('read_qr', True), body.get('override')))))
    if what == 'mission_abort':
        if STACK.mode == 'offboard':                  # the explorer's abort: BRAKE, hold where it is
            return ok(*(await loop.run_in_executor(None, vrx_post, 'abort')))
        return ok(*(await loop.run_in_executor(None, lambda: ROS.call(ROS.cli_abort, ROS.Trigger.Request()))))
    if what in EMERGENCY:                             # straight to the autopilot over MAVLink (the operator confirmed)
        return ok(*(await loop.run_in_executor(None, emergency, what)))
    if what == 'camera':
        name = body.get('name')
        if name not in CAMERAS:
            return ok(False, f'camera: {", ".join(CAMERAS)}')
        ROS.set_camera(name)
        return ok(True, f'camera {name}')
    raise web.HTTPNotFound()


# LAND / RTL / RTH / KILL and the hand-over modes: MAVLink command (id, params) - ArduCopter (sim and the real drone);
# fcu:=px4 sims take the same LAND / RTL / KILL, the ArduCopter custom modes (RTH and the hand-over ones) are refused
EMERGENCY = {
    'kill': (400, 0, 21196),                          # MAV_CMD_COMPONENT_ARM_DISARM, force: the motors stop in flight
    'land': (21,),                                    # MAV_CMD_NAV_LAND: down where it is
    'rtl': (20,),                                     # MAV_CMD_NAV_RETURN_TO_LAUNCH: climbs to RTL_ALT (15 m), straight home
    'rth': (176, 1, 21),                              # MAV_CMD_DO_SET_MODE SMART_RTL: retraces its own path home
}
# hand over to the pilot (user, 2026-10-30): MAV_CMD_DO_SET_MODE, ArduCopter's custom mode number - then the onboard
# mission stops sending setpoints and the ELRS pilot flies. LOITER / POSHOLD hold position on the GPS / EKF,
# ALT_HOLD holds only the height, STABILIZE is fully manual, BRAKE stops dead and hovers
MODES = {'loiter': 5, 'poshold': 16, 'alt_hold': 2, 'stabilize': 0, 'brake': 17}
EMERGENCY.update({m: (176, 1, n) for m, n in MODES.items()})


def radio_api(what, body):
    """--radio (the final flight, on the Lenovo): only what the radio carries. Before take-off: launch / stop the
    onboard stack, start a segment (mav_agent, 1/194); any time: LAND / RTH / RTL / KILL straight to ArduCopter
    (1/1), then the agent hands the mission over (no failsafe flight over them)."""
    if what == 'status':
        return True, stack_phase()
    if what in EMERGENCY:
        success, ack = MAV.command_ack(*EMERGENCY[what], timeout=4.0)
        _, sup = MAV.command_ack(AGENT_CMD, 8, comp=AGENT, timeout=4.0)
        return success, f'{what.upper()}: autopilot {ack}; onboard: {sup}'
    if S.tel.get('armed') and what in ('stack_start', 'stack_stop', 'mission_start'):
        return False, 'the drone is armed: in flight only LAND / RTH / RTL / KILL and the hand-over modes'
    if what == 'stack_start':                         # param3: the boards, 0 QR / 1 ArUco (this page's --marker)
        return MAV.command_ack(AGENT_CMD, 1, 0, int(MARKER == 'aruco'), comp=AGENT, timeout=10.0, final_within=240.0)
    if what == 'stack_stop':
        return MAV.command_ack(AGENT_CMD, 2, comp=AGENT, timeout=10.0, final_within=120.0)
    if what == 'mission_start':
        seg = SEGMENT_ALIAS.get(body.get('type'), body.get('type'))
        if seg not in SEG_CODE:
            return False, 'type: ' + ' | '.join(SEG_CODE)
        return MAV.command_ack(AGENT_CMD, 7, SEG_CODE[seg], comp=AGENT, timeout=10.0, final_within=30.0)
    if what == 'mission_abort':
        return MAV.command_ack(AGENT_CMD, 5, comp=AGENT, timeout=10.0, final_within=15.0)
    if what == 'apply':
        return radio_apply_gf()
    if what == 'alt_trim':                            # in flight too: agent action 10, param2 = the step (m)
        try:
            delta = float(body.get('delta'))
        except (TypeError, ValueError):
            return False, 'delta: metres, e.g. -0.5'
        press = next(TRIM_PRESS) % 1000000 + 1         # a new number per press: the agent runs each press once
        #                                               (from the clock, 2 presses in one 0.1 s were one, 2026-10-28)
        return MAV.command_ack(AGENT_CMD, 10, delta, press, comp=AGENT, timeout=8.0, final_within=10.0)
    return False, f'{what}: not over the radio'


TRIM_PRESS = __import__('itertools').count(__import__('random').randint(1, 500000))   # height-trim press numbers
T_GEOFENCE = 32810           # TUNNEL ground -> agent: [tid lo, tid hi, chunk, chunks] + up to 124 B of the JSON


def radio_apply_gf():
    """--radio: the geofence corners captured HERE (the drone's GPS over the radio) up to the onboard agent, which
    saves + applies them on the drone's dashboard (its GPS origin) - no cable or Wi-Fi to the Jetson exists
    (user, 2026-10-28). Chunks sent twice (the radio drops packets), then agent action 9 with the count + CRC."""
    import random
    import zlib
    gf = load_gf()
    a, q = gf['arena'], gf['qr']
    if len(a) not in (0, 4) or len(q) not in (0, 4) or not (a or q):
        return False, f'need 4 arena corners and/or 4 QR-area corners (have arena {len(a)}, QR area {len(q)})'
    raw = json.dumps({'a': a, 'q': q, 'e': gf['extra'], 'c': gf['course']}, separators=(',', ':')).encode()
    chunks = [raw[k:k + 124] for k in range(0, len(raw), 124)]
    tid, crc = random.randint(1, 65535), zlib.crc32(raw) & 0xFFFFFF     # 24 bits: exact in a float32 param
    for _ in range(2):
        for k, c in enumerate(chunks):
            MAV.tunnel_to_agent(T_GEOFENCE, bytes([tid & 255, tid >> 8, k, len(chunks)]) + c)
            time.sleep(0.08)
    t0 = time.time()
    success, res = MAV.command_ack(AGENT_CMD, 9, len(chunks), crc, tid, comp=AGENT, timeout=10.0, final_within=30.0)
    end = time.time() + 3.0                           # the agent's own words: its STATUSTEXT 'AGT geofence: ...'
    while time.time() < end:
        for t, _, text in reversed(S.statustext):
            if t >= t0 and 'AGT geofence' in text:
                return success, text.replace('JETSON ', '')
        time.sleep(0.2)
    return success, f'geofence over the radio: {res}'


def sim_place_at_zone(world):
    """Sim, QR-area segments: like the real drone carried to the zone, the sim drone is put down 1 m inside the
    zone's near edge - from the pad, the straight line to the zone crosses the roofed tunnels and the offboard
    geofence stopped it at x 4.9 (2026-10-27). -> (ok, message, the world's start QR payload)."""
    try:
        w = json.load(open(WORLDS / f'{world}.json'))
    except (OSError, ValueError) as e:
        return False, f'world {world}: {e}', None
    z = w['delivery_zone']
    x, y = z['x_from'] + 1.0, (z['y_from'] + z['y_to']) / 2
    payload = (w.get('start_qr') or {}).get('payload')
    p = S.tel_pose
    if p is not None and math.hypot(p[0] - x, p[1] - y) < 1.0:
        return True, 'already at the zone', payload
    if S.tel.get('armed'):
        return False, 'armed: not moving the sim drone', None
    env = dict(os.environ, GZ_IP=os.environ.get('GZ_IP', '10.42.0.247'), GZ_PARTITION=os.environ.get('GZ_PARTITION', 'vio'))
    out = ''
    for model in ('iris_d435_vio_0', 'x500_d435_vio_0'):  # ArduCopter SITL airframe / PX4's (sim_mission fcu:=px4)
        req = f'name: "{model}", position: {{x: {x}, y: {y}, z: 0.25}}, orientation: {{w: 1}}'
        r = subprocess.run(['gz', 'service', '-s', f'/world/{world}/set_pose', '--reqtype', 'gz.msgs.Pose',
                            '--reptype', 'gz.msgs.Boolean', '--timeout', '5000', '--req', req],
                           capture_output=True, text=True, env=env, timeout=20)
        out = r.stdout.strip() or r.stderr.strip()
        if 'true' in r.stdout:
            break
    else:
        return False, f'set_pose: {out}', None
    end = time.monotonic() + 60                       # the autopilot's estimate follows the move (GPS): wait for it
    while time.monotonic() < end:
        p = S.tel_pose
        if p is not None and math.hypot(p[0] - x, p[1] - y) < 0.7:
            time.sleep(5)                             # settle (EKF reset, pre-arm)
            return True, 'sim drone put at the zone edge', payload
        time.sleep(1)
    return False, 'the autopilot position did not follow the move in 60 s', None


def start_mission(kind, payload='', read_qr=True, override=False):
    """Preflight gate, the segment's supervisor parameters, /mission/start -> (ok, message)."""
    kind = SEGMENT_ALIAS.get(kind, kind)
    if stack_phase() != 'ready':
        return False, f'stack not ready ({stack_phase()}) - wait for READY'
    if kind not in SEGMENTS:
        return False, 'type: ' + ' | '.join(SEGMENTS)
    if STACK.mode == 'offboard':
        if kind not in ('qr', 'qr_drop'):
            return False, 'the offboard Jetson flies the QR area only: QR area only | QR area + drop'
        failing = [c['name'] for c in preflight('offboard') if c['critical'] and not c['ok']]
        if failing and not override:
            return False, 'preflight failing: ' + '; '.join(failing)
        success, msg = vrx_post('start', {'drop': kind == 'qr_drop', 'start_qr': payload.strip()})
        if success:
            with S.lock:
                S.mission_type, S.mission_t0, S.zone_t0 = kind, None, None
                S.tel_track = []
        return success, msg
    failing = [c['name'] for c in preflight(STACK.mode or 'sim') if c['critical'] and not c['ok']
               and c['name'] not in ('Mission supervisor up',)]
    if failing and not override:
        return False, 'preflight failing: ' + '; '.join(failing)
    payload = payload.strip()
    if STACK.mode == 'sim' and kind in ('qr', 'qr_drop'):
        placed, msg, start_payload = sim_place_at_zone(STACK.world)
        if not placed:
            return False, msg
        if start_payload and not payload:             # the pad's start QR is out of sight from the zone
            payload, read_qr = start_payload, False
    if payload:
        ROS.pub_qr.publish(ROS.String(data=json.dumps({'payload': payload, 'source': 'ground station'})))
    s1, m1 = ROS.set_supervisor(**SEGMENTS[kind], read_start_qr=bool(read_qr) or not payload)
    if not s1:
        return False, f'supervisor parameters: {m1}'
    success, msg = ROS.call(ROS.cli_start, ROS.Trigger.Request())
    if success:
        with S.lock:
            S.mission_type, S.mission_t0, S.zone_t0 = kind, None, None
            S.ros.update(track=[], log=[])
            S.tel_track = []                          # the map shows this mission's flight path only
    return success, f'{kind}: {msg}'


def emergency(what):
    """The command first (no delay), then the supervisor hands the mission over: aborted without its failsafe
    flight, which would put the autopilot back in GUIDED over this LAND / RTL / RTH."""
    success, ack = MAV.command_ack(*EMERGENCY[what])
    if STACK.mode == 'offboard':
        sup_ok, sup = vrx_post('release')
    else:
        sup_ok, sup = ROS.call(ROS.cli_operator, ROS.Trigger.Request(), timeout=3.0)
    msg = f'{what.upper()}: autopilot {ack}'
    if what in MODES and success:
        msg += ' - yours now: fly it on the RC'
    if sup_ok or 'not available' not in sup:
        msg += f'; mission: {sup}'
    if what == 'rth' and not success:
        msg += ' (SmartRTL needs a recorded path: try RTL or LAND)'
    return success, msg


async def index(request):
    if MARKER == 'qr':
        return web.FileResponse(HERE / 'static/index.html')
    # the ArUco delivery dashboard: its own title, a banner, the ArUco twins of the sim worlds
    html = (HERE / 'static/index.html').read_text()
    html = html.replace('<title>AeroTHON Ground Station</title>', '<title>AeroTHON ArUco Delivery</title>')
    html = html.replace('<option>competition</option><option>comp_far1</option>',
                        '<option>competition_aruco</option><option>comp_far1_aruco</option>')
    html = html.replace('<option value="full" class="seg-full">Whole mission</option>', '')     # QR area only
    html = html.replace('<option value="tunnel" class="seg-full">Tunnels + banners only</option>', '')
    html = html.replace('<body>', '<body><div style="background:#1d4ed8;color:#fff;font:600 13px system-ui;'
                        'padding:4px 10px">ArUco delivery (:8084) - the 8080 QR-area mission with ArUco markers 3x3 .. 6x6 '
                        'instead of QR codes</div>', 1)
    return web.Response(text=html, content_type='text/html')


def main():
    global ROS, MAV
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--mavlink-port', type=int, default=14650)
    ap.add_argument('--radio', default='', help='the final flight, ON THE LENOVO: the SiK radio (/dev/ttyUSB0:57600); '
                    'telemetry + RTH / RTL / LAND / KILL + segment start via the onboard mav_agent; QGC gets a relay')
    ap.add_argument('--qgc', default='127.0.0.1:14550', help="--radio: QGroundControl's UDP port (its auto-connect)")
    ap.add_argument('--marker', default='qr', choices=('qr', 'aruco'),
                    help='what the boards carry: qr (8080) | aruco (the ArUco delivery mission, 8084)')
    ap.add_argument('--offboard-alt', type=float, default=10.0,
                    help='offboard (VRX): search height, m - the drop is at min(5 m, this)')
    args = ap.parse_args()
    global RADIO, MARKER, OFFBOARD_ALT
    MARKER, OFFBOARD_ALT = args.marker, args.offboard_alt
    STACK.marker = MARKER
    RADIO = args.radio or None
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml'))
    MAV = Mavlink(args.mavlink_port, RADIO, args.qgc)
    MAV.start()
    if RADIO:                                         # no ROS on the ground in the final flight
        ROS = argparse.Namespace(camera_topic='', jpeg=None)
    else:
        ROS = Ros()
        ROS.start()
        ROS.ready.wait(20)
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/costmap.png', costmap)
    app.router.add_get('/camera.mjpg', mjpeg)
    app.router.add_post('/api/{what}', api)
    app.router.add_static('/static/', HERE / 'static')
    print(f'ground station: http://0.0.0.0:{args.port}  (MAVLink udpin :{args.mavlink_port})', flush=True)
    # shutdown_timeout: the MJPEG / WebSocket streams never end on their own - 60 s default stop time otherwise
    # Exit hard once shutdown starts: with browsers attached the cleanup took 5-8 s, then a C++ thread
    # (rclpy executor / OpenCV / serial) aborted the interpreter at exit ('terminate called without an
    # active exception', apport crash dump), so a stop + relaunch found the old process still on the port
    # (2026-10-27). Bounded to 2 s after SIGINT / SIGTERM.
    async def exit_soon(_app):
        threading.Timer(2.0, os._exit, args=(0,)).start()
    app.on_shutdown.append(exit_soon)
    web.run_app(app, host='0.0.0.0', port=args.port, print=None, shutdown_timeout=2.0)
    print('stopped', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
