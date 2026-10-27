#!/usr/bin/env python3
"""Radio ground station - runs ON THE LENOVO, talks to the drone over the TELEMETRY RADIO only (no Wi-Fi on the drone).
    python3 radio_gcs.py [--radio /dev/ttyUSB0:57600] [--port 8081]      then open http://127.0.0.1:8081
It owns the radio's serial port and relays every MAVLink packet both ways to QGroundControl on UDP (QGC's default
auto-connect, 127.0.0.1:14550), so QGC keeps working beside it.
  shows     PX4 telemetry (mode, armed, battery, GPS, position / height / speed, link age and radio RSSI), the track,
            and the onboard computer's status from the Jetson's mav_agent (gcs/onboard/mav_agent.py: TUNNEL 32800 /
            32801 - stack phase, mission state, mission and QR-area time, QRs read, target, failing preflight checks,
            alerts), and every STATUSTEXT (PX4's and the agent's events)
  commands  to the agent (COMMAND_LONG MAV_CMD_USER_1 -> 1/194, resent until acknowledged): Launch stack, Stop stack,
            Start FULL mission, Start DELIVERY only, ABORT (supervisor failsafe), Status now; and straight to PX4
            (1/1): LAND, RTL, KILL (flight termination via disarm-force, confirmed twice)
The onboard dashboard's rules apply to the agent's commands: no real launch without the geofence (set it once with
the Ethernet cable, onboard dashboard), no mission start before READY + a green preflight.
"""
import argparse
import asyncio
import json
import os
import math
import socket
import threading
import time
from pathlib import Path

from aiohttp import web
os.environ['MAVLINK20'] = '1'      # MAVLink 2 before pymavlink loads: TUNNEL (385) and chunked STATUSTEXT need it
from pymavlink import mavutil  # noqa: E402

HERE = Path(__file__).resolve().parent
AGENT, CMD = 194, 31010
ACTIONS = {'launch': 1, 'stop': 2, 'full': 3, 'delivery': 4, 'abort': 5, 'status': 6}
PX4_MAIN = {1: 'MANUAL', 2: 'ALTCTL', 3: 'POSCTL', 4: 'AUTO', 5: 'ACRO', 6: 'OFFBOARD', 7: 'STABILIZED', 8: 'RATTITUDE'}
PX4_AUTO = {1: 'READY', 2: 'TAKEOFF', 3: 'LOITER', 4: 'MISSION', 5: 'RTL', 6: 'LAND', 8: 'FOLLOW', 9: 'PRECLAND'}


class Radio:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.tel = {}
        self.t = {}                  # message type -> monotonic time last seen
        self.onboard, self.detail = {}, {}
        self.log = []                # (clock, source, severity, text)
        self.track = []
        self.home = None
        self.acks = {}               # command -> (result, time)
        self.radio_rx = 0
        self.mav, self.radio_err = None, 'not opened yet'
        self.open()                  # never fatal: the page opens and shows why the radio isn't there
        self.qgc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.qgc.bind(('127.0.0.1', 0))
        self.qgc.setblocking(False)
        host, port = a.qgc.split(':')
        self.qgc_addr = (host, int(port))

    def open(self):
        """Open the radio; on failure (not plugged in, wrong port, no 'dialout' permission) keep None + the reason -
        rx_loop retries every 2 s. A failed open at start used to kill the app before its page came up (2026-10-26)."""
        dev = self.a.radio
        try:
            if dev.startswith(('udpin:', 'udpout:', 'tcp:')):
                self.mav = mavutil.mavlink_connection(dev, source_system=254, source_component=190, dialect='common')
            else:
                path, _, baud = dev.partition(':')
                if not Path(path).exists():
                    raise OSError(f'{path} not found (radio plugged in? ls /dev/serial/by-id/)')
                self.mav = mavutil.mavlink_connection(path, baud=int(baud or 57600), source_system=254,
                                                      source_component=190, dialect='common')
            self.radio_err = None
        except Exception as e:
            self.mav = None
            msg = str(e)
            if 'ermission' in msg:
                msg += ' - add yourself to dialout: sudo usermod -aG dialout $USER, then log out / in'
            if msg != self.radio_err:
                self.note('gcs', 4, f'radio: {msg}')
            self.radio_err = msg

    # ---- receive (thread)
    def rx_loop(self):
        tun = {}
        while True:
            if self.mav is None:
                time.sleep(2)
                self.open()
                continue
            try:
                m = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception as e:                    # radio unplugged: reopen
                self.note('gcs', 4, f'radio lost: {e}')
                self.mav, self.radio_err = None, f'lost: {e}'
                continue
            if m is None:
                continue
            try:
                self.qgc.sendto(m.get_msgbuf(), self.qgc_addr)          # everything to QGC too
            except OSError:
                pass
            t = m.get_type()
            if t == 'BAD_DATA':
                continue
            now = time.monotonic()
            src, comp = m.get_srcSystem(), m.get_srcComponent()
            self.radio_rx += 1
            with self.lock:
                if t == 'HEARTBEAT':
                    if comp == 1:
                        self.t['px4'] = now
                        base, cm = m.base_mode, m.custom_mode
                        main, sub = (cm >> 16) & 0xFF, (cm >> 24) & 0xFF
                        mode = PX4_MAIN.get(main, str(main))
                        if main == 4:
                            mode = 'AUTO.' + PX4_AUTO.get(sub, str(sub))
                        self.tel.update(armed=bool(base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED), mode=mode,
                                        system_status=m.system_status)
                    elif comp == AGENT:
                        self.t['agent'] = now
                elif t == 'SYS_STATUS':
                    self.tel.update(batt_v=round(m.voltage_battery / 1000, 2) if m.voltage_battery != 65535 else None,
                                    batt_pct=m.battery_remaining if m.battery_remaining >= 0 else None)
                elif t == 'GPS_RAW_INT':
                    self.tel.update(gps_fix=m.fix_type, sats=m.satellites_visible,
                                    hdop=round(m.eph / 100, 2) if m.eph != 65535 else None)
                elif t == 'GLOBAL_POSITION_INT':
                    lat, lon = m.lat / 1e7, m.lon / 1e7
                    self.tel.update(lat=lat, lon=lon, alt_rel=round(m.relative_alt / 1000, 2),
                                    speed=round(math.hypot(m.vx, m.vy) / 100, 2), climb=round(-m.vz / 100, 2),
                                    heading=round(m.hdg / 100) if m.hdg != 65535 else None)
                    if lat or lon:
                        if self.home is None:
                            self.home = (lat, lon)
                        if not self.track or now - self.track[-1][2] > 1.0:
                            self.track = (self.track + [(lat, lon, now)])[-2000:]
                elif t == 'HOME_POSITION':
                    self.home = (m.latitude / 1e7, m.longitude / 1e7)
                elif t == 'RADIO_STATUS':
                    self.tel.update(rssi=m.rssi, remrssi=m.remrssi, noise=m.noise, txbuf=m.txbuf)
                elif t == 'STATUSTEXT':
                    key = (src, comp, m.id)
                    txt = m.text.rstrip('\0')
                    if m.id:                                  # chunked (MAVLink 2): join the chunks of one id
                        tun[key] = tun.get(key, '') + txt
                        if len(txt) == 50:
                            continue
                        txt = tun.pop(key)
                    self.log.append((time.strftime('%H:%M:%S'), 'jetson' if comp == AGENT else
                                     ('px4' if comp == 1 else f'{src}/{comp}'), m.severity, txt))
                    self.log = self.log[-300:]
                elif t == 'TUNNEL' and comp == AGENT and m.payload_type in (32800, 32801):
                    try:
                        d = json.loads(bytes(m.payload[:m.payload_length]).decode())
                    except ValueError:
                        continue
                    if m.payload_type == 32800:
                        self.onboard, self.t['onboard'] = d, now
                    else:
                        self.detail = d
                elif t == 'COMMAND_ACK':
                    self.acks[m.command] = (m.result, now)
            self.t[t] = now

    def qgc_loop(self):
        """QGC -> the radio (its heartbeats, its commands, parameter / mission traffic)."""
        while True:
            try:
                data = self.qgc.recv(4096)
                if self.mav is not None:
                    self.mav.write(data)
            except BlockingIOError:
                time.sleep(0.01)
            except OSError:
                time.sleep(0.2)

    def hb_loop(self):
        while True:
            try:
                if self.mav is None:
                    raise OSError('no radio')
                self.mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                                            mavutil.mavlink.MAV_STATE_ACTIVE)
            except Exception:
                pass
            time.sleep(1.0)

    def note(self, src, sev, text):
        with self.lock:
            self.log.append((time.strftime('%H:%M:%S'), src, sev, text))

    # ---- commands
    def command(self, target_comp, cmd, params, tries=4, wait=2.5):
        """COMMAND_LONG, resent until an ACK other than IN_PROGRESS (or the tries run out); (ok, message)."""
        t0 = time.monotonic()
        if self.mav is None:
            return False, f'no radio ({self.radio_err})'
        for k in range(tries):
            p = list(params) + [0] * (7 - len(params))
            self.mav.mav.command_long_send(1, target_comp, cmd, k, *p)
            end = time.monotonic() + wait
            while time.monotonic() < end:
                r = self.acks.get(cmd)
                if r and r[1] > t0:
                    if r[0] == mavutil.mavlink.MAV_RESULT_IN_PROGRESS:
                        end = max(end, time.monotonic() + 1.0)      # working on it (a launch takes a while)
                        if time.monotonic() - t0 > 240:
                            return False, 'no final answer in 240 s'
                    else:
                        ok = r[0] == mavutil.mavlink.MAV_RESULT_ACCEPTED
                        return ok, {0: 'accepted', 1: 'temporarily rejected', 2: 'denied', 3: 'unsupported',
                                    4: 'failed'}.get(r[0], f'result {r[0]}')
                time.sleep(0.05)
        return False, 'no answer over the radio'

    def snapshot(self):
        now = time.monotonic()
        age = lambda k: round(now - self.t[k], 1) if k in self.t else None  # noqa: E731
        with self.lock:
            h = self.home
            track = [[(la - h[0]) * 111320.0, (lo - h[1]) * 111320.0 * math.cos(math.radians(h[0]))]
                     for la, lo, _ in self.track[-1500:]] if h else []
            return {'tel': dict(self.tel), 'px4_age': age('px4'), 'agent_age': age('agent'),
                    'onboard_age': age('onboard'), 'onboard': dict(self.onboard), 'detail': dict(self.detail),
                    'log': self.log[-80:], 'track': track, 'radio': self.a.radio, 'rx': self.radio_rx,
                    'radio_err': self.radio_err}


def make_app(r):
    async def index(_):
        return web.FileResponse(HERE / 'radio.html')

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=10)
        await sock.prepare(request)
        try:
            while not sock.closed:
                await sock.send_str(json.dumps(r.snapshot()))
                await asyncio.sleep(0.5)
        except (ConnectionResetError, RuntimeError):
            pass
        return sock

    async def cmd(request):
        body = await request.json()
        what = body.get('action', '')
        loop = asyncio.get_running_loop()
        if what in ACTIONS:
            r.note('gcs', 6, f'-> jetson: {what}')
            p2 = 1.0 if body.get('sim') else 0.0
            ok, msg = await loop.run_in_executor(None, lambda: r.command(AGENT, CMD, [ACTIONS[what], p2]))
        elif what == 'land':
            ok, msg = await loop.run_in_executor(None, lambda: r.command(1, 21, []))
        elif what == 'rtl':
            ok, msg = await loop.run_in_executor(None, lambda: r.command(1, 20, []))
        elif what == 'kill':
            ok, msg = await loop.run_in_executor(None, lambda: r.command(1, 400, [0, 21196]))
        else:
            return web.json_response({'ok': False, 'message': f'unknown action {what!r}'})
        r.note('gcs', 6 if ok else 4, f'{what}: {msg}')
        return web.json_response({'ok': ok, 'message': f'{what}: {msg}'})

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws)
    app.router.add_post('/api/cmd', cmd)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--radio', default='/dev/ttyUSB0:57600',
                    help='the ground radio: /dev/ttyUSB0:57600 (SiK on USB), or udpin:0.0.0.0:14670 (a test)')
    ap.add_argument('--qgc', default='127.0.0.1:14550', help="QGroundControl's UDP port (its auto-connect)")
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to open it from other machines')
    a = ap.parse_args()
    r = Radio(a)
    for f in (r.rx_loop, r.qgc_loop, r.hb_loop):
        threading.Thread(target=f, daemon=True).start()
    print(f'radio GCS: {a.radio} -> http://{a.host}:{a.port} (QGC relay {a.qgc})', flush=True)
    app = make_app(r)
    # Exit hard once shutdown starts: with browsers attached the cleanup took 5-8 s, then a C++ thread
    # (rclpy executor / OpenCV / serial) aborted the interpreter at exit ('terminate called without an
    # active exception', apport crash dump), so a stop + relaunch found the old process still on the port
    # (2026-10-27). Bounded to 2 s after SIGINT / SIGTERM.
    async def exit_soon(_app):
        threading.Timer(2.0, os._exit, args=(0,)).start()
    app.on_shutdown.append(exit_soon)
    web.run_app(app, host=a.host, port=a.port, print=None, shutdown_timeout=2)
    print('stopped', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
