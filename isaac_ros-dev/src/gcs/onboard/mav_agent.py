#!/usr/bin/env python3
"""Onboard MAVLink agent (Jetson host, systemd --user aerothon-mav-agent): lets the ground run the mission over the
TELEMETRY RADIO alone - no Wi-Fi. A MAVLink component of the drone (system 1, component 194 = ONBOARD_COMPUTER2;
MAVROS is 191) on mavlink-router's agent endpoint (UDP 14660: router/sim.conf, onboard/install_onboard.sh).
  commands  COMMAND_LONG MAV_CMD_USER_1 (31010) to 1/194, param1 = action:
              1 launch stack (param2: 0 real drone, 1 sim world 'competition'; param3: 0 QR boards, 1 ArUco
                (the 8084 ArUco delivery radio page))   2 stop stack
              3 start FULL mission   4 start DELIVERY only   5 ABORT (supervisor failsafe: hold, land)   6 status now
              7 start a segment (param2: 1 full 2 qr 3 tunnel 4 qr_drop)   8 operator took over (after the ground's
              LAND / RTH / RTL / KILL: the supervisor stops without a failsafe flight)
              9 geofence (param2 chunks, param3 CRC-32 & 0xFFFFFF, param4 transfer id): the corners captured on the
                radio page came up as TUNNEL 32810 [tid lo, tid hi, chunk, chunks] + JSON {a: arena, q: QR area, e: extra, c: course};
                saved + applied on the onboard dashboard (no cable / Wi-Fi to the Jetson, user 2026-10-28)
              10 search height trim (param2: the step in m, e.g. -0.5): in flight too - the explorer flies its search
                 height + the trim at once, never below its alt_min (user, 2026-10-28)
              13 test heights for the NEXT start only (param2 start marker, param3 tunnels, param4 QR area, m): the
                 test app (aerothon test). Any start without it flies the mission's fixed heights (user, 2026-10-30)
              11 FIELD TEST (props off, carried by hand; gcs/field_test): param2 = its code (field_proto.CODES: 1 start
                 / 2 stop = field_test.sh, the rest -> field_vision's /api/cmd), param3 its argument, param4 a press
                 number. While it runs: TUNNEL 32820 its guidance ~1 Hz + STATUSTEXT 'FT <result / event>'
            -> the onboard dashboard's REST API (http://127.0.0.1:8080/api/...: its preflight gate, geofence check and
            stack logic apply unchanged) -> COMMAND_ACK (ACCEPTED / FAILED) + STATUSTEXT 'AGT: <its message>'
  status    TUNNEL (payload_type 32800, 128 B JSON) every status_period s: {p: stack phase, md: sim|real, s: FSM state,
            t: mission s, z: QR-area s, q: QRs read, tg: target found 0/1, pf: failing critical preflight checks,
            al: alerts}; payload_type 32801 every 3rd time: {f: failing preflight names, a: first alert}
  vio       TUNNEL 32802 every --vio-period s (0 = off): {v: [x, y, z] cuVSLAM position in its own frame
            (where it started; corridor:=aerothon only), a: its age s, h: /vio/health} - ~70 B a packet
  boards    TUNNEL 32803: {p: payload, x, y (odom, as the map), st: status} at once for a new board / a status change
            / a move > 0.5 m, plus one board again every --board-period s (0 = off; round robin: a packet lost on the
            radio comes back); the confirmed target as p '*target' - ~65 B a packet
  red       TUNNEL 32804: {r: [[x0, y0, x1, y1], ...]} the red zones as boxes (odom; the dashboard's red_boxes, 3 at most),
            at once on a change and every --board-period s; the costmap itself doesn't fit on the radio - ~50 B a packet
  events    STATUSTEXT on a change of stack phase / FSM state, a new QR, the target, a new alert (QGC shows them too)
  heartbeat 1 Hz, MAV_TYPE_ONBOARD_CONTROLLER
A radio at 57600 baud carries ~5 KB/s: the status adds ~70 B/s, VIO ~70 B/s at 1 Hz, the boards ~13 B/s at one per
5 s (MAVLink 2 drops TUNNEL's trailing zero bytes: a packet is ~12 B + its JSON). Tune the periods on the radio.
Wiring (ArduCopter, Pixhawk 6C): the SiK radio on GPS2 (SERIAL4, MAVLink 2, 57600), the Jetson on the Pixhawk's
USB - port 1 (SERIAL0, -if00) mavlink-router, port 2 (SERIAL7, -if02) mav_agent. ArduPilot's MAVLink routing
carries the radio's commands to 1/194 and the agent's packets down the radio. Nothing on the Jetson opens the radio.
"""
import argparse
import asyncio
import json
import os
import re
import time

import glob

import aiohttp
os.environ['MAVLINK20'] = '1'      # MAVLink 2 before pymavlink loads: TUNNEL (385) and chunked STATUSTEXT need it
from pymavlink import mavutil  # noqa: E402

COMP = 194                       # MAV_COMP_ID_ONBOARD_COMPUTER2
CMD = 31010                      # MAV_CMD_USER_1
ACTIONS = {1: 'launch', 2: 'stop', 3: 'full', 4: 'delivery', 5: 'abort', 6: 'status', 7: 'segment', 8: 'operator',
           9: 'geofence', 10: 'alt_trim', 11: 'field', 12: 'reset', 13: 'alts'}
FIELD_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'field_test', 'field_test.sh')
FIELD_URL = 'http://127.0.0.1:8097'   # field_test/field_vision.py
T_FIELD = 32820                  # out: the field test's guidance (field_test/field_proto.py)
SEGMENTS = {1: 'full', 2: 'qr', 3: 'tunnel', 4: 'qr_drop', 5: 'hop'}   # action 7, param2 (the 8080 page in --radio mode)
T_STATUS, T_DETAIL, T_VIO, T_BOARD, T_RED = 32800, 32801, 32802, 32803, 32804
T_GEOFENCE = 32810               # in: the radio page's geofence corners (action 9)
ALERT_REPEAT = 60.0              # s between two STATUSTEXTs of the same alert


ROUTER = 'udpin:127.0.0.1:14660'  # mavlink-router's agent endpoint (aerothon-router)


def pick_link(spec):
    """--mavlink auto: the Pixhawk's SECOND USB port (SERIAL7: ArduPilot relays it to and from the radio on GPS2)
    when it is there, else the router endpoint (USB port 1 through aerothon-router; the Pixhawk unplugged / one port)."""
    if spec != 'auto':
        return spec
    ports = sorted(glob.glob('/dev/serial/by-id/*ArduPilot*-if02') + glob.glob('/dev/serial/by-id/*Pixhawk*-if02') +
                   glob.glob('/dev/serial/by-id/*Holybro*-if02'))
    return f'{ports[0]}:115200' if ports else ROUTER


class Agent:
    def __init__(self, a):
        self.a = a
        self.link = pick_link(a.mavlink)
        dev, _, baud = self.link.partition(':') if self.link.startswith('/dev/') else (self.link, '', '')
        self.mav = (mavutil.mavlink_connection(dev, baud=int(baud or 115200), source_system=1, source_component=COMP,
                                               dialect='common') if self.link.startswith('/dev/') else
                    mavutil.mavlink_connection(self.link, source_system=1, source_component=COMP, dialect='common'))
        self.snap = None             # latest onboard dashboard snapshot
        self.last = {}               # what the events compare against
        self.n_status = 0
        self.done_cmds = {}          # (sysid, compid, action) -> time: a resent command isn't run twice
        self.alert_sent = {}         # alert without its numbers -> time sent
        self.boards_sent = {}        # payload -> (x, y, status) last sent
        self.gf_chunks = {}          # transfer id -> {chunk index: bytes} (TUNNEL 32810)
        self.board_rr = 0            # round-robin index of the periodic resend
        self.red_sent = None         # the red-zone boxes last sent
        self.next_alts = None        # the test app's heights, for the next start only
        self.bytes_out = {}          # packet type -> bytes sent (the radio load, printed every minute)
        self.field_on, self.field_n = False, None   # the field test answering; its last event relayed

    # ---- MAVLink out
    def text(self, msg, severity=6):
        """STATUSTEXT, chunked (MAVLink 2: 50 chars a chunk, same id)."""
        b = msg.encode()[:200]
        self.text_id = getattr(self, 'text_id', int(time.time()) % 60000) % 65534 + 1   # unique: several in one tick
        mid = self.text_id
        for k in range(0, max(1, len(b)), 50):
            self.mav.mav.statustext_send(severity, b[k:k + 50].ljust(50, b'\0'), mid, k // 50)

    def tunnel(self, ptype, obj):
        raw = json.dumps(obj, separators=(',', ':')).encode()[:128]
        self.mav.mav.tunnel_send(0, 0, ptype, len(raw), list(raw.ljust(128, b'\0')))
        self.bytes_out[ptype] = self.bytes_out.get(ptype, 0) + 12 + 5 + len(raw)   # MAVLink 2 header + fields

    def send_vio(self):
        v = (self.snap or {}).get('vio_pose') or {}
        if not v.get('xyz'):
            return
        self.tunnel(T_VIO, {'v': [round(c, 2) for c in v['xyz']], 'a': v.get('age'), 'h': (v.get('health') or '')[:24]})

    def boards(self):
        """[(payload, x, y, status)] from the onboard dashboard: every board read, and the confirmed target."""
        ros = (self.snap or {}).get('ros', {})
        out = [(str(q.get('payload', ''))[:24], round(q.get('x', 0.0), 2), round(q.get('y', 0.0), 2),
                str(q.get('status', ''))[:10]) for q in ros.get('qrs') or [] if q.get('payload')]
        if ros.get('target'):
            out.append(('*target', round(ros['target'][0], 2), round(ros['target'][1], 2), 'confirmed'))
        return out

    def send_board(self, b):
        self.tunnel(T_BOARD, {'p': b[0], 'x': b[1], 'y': b[2], 'st': b[3]})
        self.boards_sent[b[0]] = b[1:]

    def board_events(self):
        """A new board, a status change or a move > 0.5 m goes out at once."""
        for b in self.boards():
            last = self.boards_sent.get(b[0])
            if last is None or last[2] != b[3] or abs(last[0] - b[1]) + abs(last[1] - b[2]) > 0.5:
                self.send_board(b)

    def resend_board(self):
        bs = self.boards()
        if bs:
            self.board_rr = (self.board_rr + 1) % len(bs)
            self.send_board(bs[self.board_rr])

    def send_red(self):
        if self.snap is not None:                     # no snapshot yet: nothing known, don't clear the ground's map
            self.red_sent = self.snap.get('red_boxes') or []
            self.tunnel(T_RED, {'r': self.red_sent})

    def red_events(self):
        if self.snap is not None and (self.snap.get('red_boxes') or []) != self.red_sent:
            self.send_red()

    # ---- onboard dashboard
    async def ws_loop(self):
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(f'{self.a.dashboard}/ws', heartbeat=10) as ws:
                        async for m in ws:
                            if m.type == aiohttp.WSMsgType.TEXT:
                                self.snap = json.loads(m.data)
            except Exception:
                self.snap = None
                await asyncio.sleep(2)

    async def api(self, what, body):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f'{self.a.dashboard}/api/{what}', json=body,
                                  timeout=aiohttp.ClientTimeout(total=200)) as r:
                    d = await r.json()
                    return bool(d.get('ok')), str(d.get('message', ''))
        except Exception as e:
            return False, f'dashboard: {e}'

    async def mission_start(self, kind):
        """Over the radio: the mission's fixed heights, never what a sim / test left saved on the dashboard - unless the
        test app sent its own just before this start."""
        alts, self.next_alts = self.next_alts or 'mission', None
        return await self.api('mission_start', {'type': kind, 'alts': alts})

    async def apply_geofence(self, chunks, crc, tid):
        import zlib
        got = self.gf_chunks.get(tid, {})
        missing = [k for k in range(chunks) if k not in got]
        if chunks < 1 or missing:
            return False, f'chunks {missing} of {chunks} lost on the radio: Save + Apply again'
        raw = b''.join(got[k] for k in range(chunks))
        if zlib.crc32(raw) & 0xFFFFFF != crc:
            return False, 'checksum mismatch: Save + Apply again'
        try:
            gf = json.loads(raw.decode())
        except ValueError:
            return False, 'garbled corners: Save + Apply again'
        self.gf_chunks = {}
        # {a: arena, q: QR area, e: extra, c: course dimensions}
        ok, msg = await self.api('geofence', {'arena': gf.get('a', []), 'qr': gf.get('q', []), 'extra': gf.get('e', []),
                                              'course': gf.get('c', {})})
        if not ok:
            return ok, msg
        return await self.api('apply', {'mode': 'real'})

    async def run_action(self, action, p2, p3=0, p4=0):
        if action == 'launch':
            body = {'mode': 'sim', 'world': 'competition'} if int(p2) == 1 else {'mode': 'real', 'link': self.a.link}
            body['marker'] = 'aruco'                  # the course has ArUco markers only (2026-10-30), whatever p3
            return await self.api('stack_start', body)
        if action == 'stop':
            return await self.api('stack_stop', {})
        if action == 'alts':
            self.next_alts = {'takeoff': float(p2), 'tunnel': float(p3), 'search': float(p4)}
            return True, f'test heights for the next start: {self.next_alts}'
        if action in ('full', 'delivery'):
            return await self.mission_start(action)
        if action == 'abort':
            return await self.api('mission_abort', {})
        if action == 'segment':
            seg = SEGMENTS.get(int(p2))
            if seg is None:
                return False, f'segment code {p2}: 1 full 2 qr 3 tunnel 4 qr_drop'
            return await self.mission_start(seg)
        if action == 'reset':                         # 12: a clean start (the last mission's files + memory)
            return await self.api('mission_reset', {})
        if action == 'operator':                      # the ground's LAND / RTH / RTL / KILL: no failsafe flight over it
            return await self.api('operator', {})
        if action == 'status':
            self.send_status(detail=True)
            return True, 'status sent'
        if action == 'geofence':
            return await self.apply_geofence(int(p2), int(p3), int(p4))
        if action == 'alt_trim':
            return await self.api('alt_trim', {'delta': round(float(p2), 2)})
        if action == 'field':
            return await self.field(int(p2), float(p3))
        return False, 'unknown action'

    # ---- the field test (gcs/field_test): the radio's half
    async def field(self, code, arg):
        if code in (1, 2):                            # start / stop: the cameras + field_vision + field_link
            try:
                pr = await asyncio.create_subprocess_exec('bash', FIELD_SH, 'start' if code == 1 else 'stop',
                                                          stdout=asyncio.subprocess.PIPE,
                                                          stderr=asyncio.subprocess.STDOUT)
                out, _ = await asyncio.wait_for(pr.communicate(), 120)
            except Exception as e:
                return False, f'field_test.sh: {e}'
            lines = [ln for ln in out.decode(errors='replace').splitlines() if ln.strip()]
            self.field_on, self.field_n = code == 1 and pr.returncode == 0, None
            return pr.returncode == 0, (lines[-1] if lines else '')[:150]
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f'{FIELD_URL}/api/cmd', json={'code': code, 'arg': arg},
                                  timeout=aiohttp.ClientTimeout(total=60)) as r:
                    d = await r.json()
                    self.field_on = True
                    return bool(d.get('ok')), str(d.get('message', ''))[:150]
        except Exception as e:
            return False, f'field test not running ({type(e).__name__}): field code 1 starts it'

    async def field_loop(self):
        """While the field test answers: its guidance as TUNNEL 32820 every second, its events as STATUSTEXT."""
        n_probe = 0
        while True:
            await asyncio.sleep(1.0)
            n_probe += 1
            if not self.field_on and n_probe % 10:    # not running: look every 10 s (started by hand on the Jetson)
                continue
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f'{FIELD_URL}/api/radio?since={self.field_n or 0}',
                                     timeout=aiohttp.ClientTimeout(total=2)) as r:
                        d = await r.json()
            except Exception:
                self.field_on = False
                continue
            self.field_on = True
            if self.field_n is None:                  # first contact: from now on, not its whole history
                self.field_n = d.get('n', 0)
                continue
            self.tunnel(T_FIELD, d['status'])
            for e in d.get('events', [])[-3:]:       # <= 3 a second on the radio
                self.text(f'FT {e["text"]}', 5 if 'MATCH' in e['text'] else 6)
            self.field_n = d.get('n', self.field_n)

    # ---- status
    def summary(self):
        d = self.snap
        if d is None:
            return {'p': 'no dashboard'}, {}
        st = d.get('stack', {})
        ros = d.get('ros', {})
        fail = [c['name'] for c in d.get('preflight', []) if c.get('critical') and not c.get('ok')]
        qrs = ros.get('qrs') or []
        al = [a for a in d.get('alerts', [])]
        s = {'p': (st.get('phase') or '')[:24], 'md': st.get('mode') or '-', 's': (ros.get('state') or '-')[:22],
             't': int(d['mission_s']) if d.get('mission_s') else None,
             'z': int(d['zone_s']) if d.get('zone_s') else None,
             'q': len(qrs), 'tg': int(bool(ros.get('target'))), 'pf': len(fail), 'al': len(al),
             'ao': d.get('alt_offset') or 0,
             'sg': {v: k for k, v in SEGMENTS.items()}.get(d.get('mission_type'), 4 if d.get('mission_type') == 'delivery' else 0)}
        # (TUNNEL carries 128 B: the segment as its code)
        a0 = al[0] if al else None                    # the dashboard's alerts: {level, text}
        first_alert = (a0.get('text', '') if isinstance(a0, dict) else a0[1] if isinstance(a0, (list, tuple))
                       else str(a0 or ''))
        detail = {'f': [n[:22] for n in fail[:4]], 'a': first_alert[:40]}
        return s, detail

    def send_status(self, detail=False):
        s, det = self.summary()
        self.tunnel(T_STATUS, s)
        self.n_status += 1
        if detail or self.n_status % 3 == 0:
            self.tunnel(T_DETAIL, det)

    def events(self):
        s, det = self.summary()
        for key, label in (('p', 'stack'), ('s', 'state')):
            if s.get(key) != self.last.get(key) and s.get(key) not in (None, '-'):
                self.text(f'{label}: {s[key]}')
        if s.get('q', 0) > self.last.get('q', 0):
            qrs = (self.snap or {}).get('ros', {}).get('qrs') or []
            if qrs:
                q = qrs[-1]
                self.text(f'QR {str(q.get("payload", ""))[-16:]} {q.get("status", "")}', 6)
        if s.get('tg') and not self.last.get('tg'):
            self.text('TARGET FOUND', 5)
        # an alert once, then every ALERT_REPEAT s at most: its numbers ('Jetson RAM 494 MB free') change every
        # tick - each change went out as a new STATUSTEXT, twice a second (2026-10-27 sim)
        if det.get('a'):
            key, now = re.sub(r'[\d.]+', '#', det['a']), time.monotonic()
            if now - self.alert_sent.get(key, -1e9) > ALERT_REPEAT:
                self.alert_sent[key] = now
                self.text(f'ALERT {det["a"]}', 3)
        self.last = {**s, 'a': det.get('a')}

    # ---- loops
    async def mav_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            m = await loop.run_in_executor(None, lambda: self.mav.recv_match(type=['COMMAND_LONG', 'TUNNEL'],
                                                                                blocking=True, timeout=1.0))
            if m is not None and m.get_type() == 'TUNNEL':
                if m.payload_type == T_GEOFENCE and m.target_component == COMP and m.payload_length > 4:
                    b = bytes(m.payload[:m.payload_length])
                    tid = b[0] | b[1] << 8
                    self.gf_chunks = {tid: self.gf_chunks.get(tid, {})}       # a new transfer drops older ones
                    self.gf_chunks[tid][b[2]] = b[4:]
                continue
            if m is None or m.target_component != COMP or m.command != CMD:
                continue
            action = ACTIONS.get(int(m.param1))
            # (a geofence keyed by its transfer id: a new Save + Apply after a lost chunk must run)
            # (a height trim keyed by its press number: two quick presses are two steps, a resent one is one)
            key = (m.get_srcSystem(), m.get_srcComponent(), action,
                   int(m.param4) if action in ('geofence', 'field') else int(m.param3) if action == 'alt_trim' else 0)
            if action is None:
                self.mav.mav.command_ack_send(CMD, mavutil.mavlink.MAV_RESULT_UNSUPPORTED, 0, 0,
                                              m.get_srcSystem(), m.get_srcComponent())
                continue
            # a retry of the same command (its ACK got lost on the radio) within 5 s: ack, don't run it twice
            if time.monotonic() - self.done_cmds.get(key, -99) < 5.0 and action != 'status':
                self.mav.mav.command_ack_send(CMD, mavutil.mavlink.MAV_RESULT_ACCEPTED, 100, 0,
                                              m.get_srcSystem(), m.get_srcComponent())
                continue
            self.done_cmds[key] = time.monotonic()
            self.mav.mav.command_ack_send(CMD, mavutil.mavlink.MAV_RESULT_IN_PROGRESS, 0, 0,
                                          m.get_srcSystem(), m.get_srcComponent())
            ok, msg = await self.run_action(action, m.param2, m.param3, m.param4)
            print(f'{time.strftime("%H:%M:%S")} {action}: {ok} {msg}', flush=True)
            self.mav.mav.command_ack_send(CMD, mavutil.mavlink.MAV_RESULT_ACCEPTED if ok else
                                          mavutil.mavlink.MAV_RESULT_FAILED, 100, 0,
                                          m.get_srcSystem(), m.get_srcComponent())
            self.text(f'AGT {action}: {"OK" if ok else "FAILED"} {msg}', 6 if ok else 4)

    async def tick_loop(self):
        t_hb = t_st = t_vio = t_bd = 0.0
        t_load = time.monotonic()
        while True:
            now = time.monotonic()
            if now - t_hb >= 1.0:
                self.mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                                            mavutil.mavlink.MAV_STATE_ACTIVE)
                t_hb = now
            if now - t_st >= self.a.status_period:
                self.send_status()
                t_st = now
            if self.a.vio_period > 0 and now - t_vio >= self.a.vio_period:
                self.send_vio()
                t_vio = now
            if self.a.board_period > 0:
                self.board_events()
                self.red_events()
                if now - t_bd >= self.a.board_period:
                    self.resend_board()
                    self.send_red()
                    t_bd = now
            if now - t_load >= 60.0:                  # what this agent puts on the radio, per packet type
                names = {T_STATUS: 'status', T_DETAIL: 'detail', T_VIO: 'vio', T_BOARD: 'boards', T_RED: 'red', T_FIELD: 'field'}
                print('radio load B/s: ' + ', '.join(f'{names[k]} {v / (now - t_load):.0f}'
                                                    for k, v in sorted(self.bytes_out.items())), flush=True)
                self.bytes_out, t_load = {}, now
            self.events()
            await asyncio.sleep(min(0.5, self.a.vio_period) if self.a.vio_period > 0 else 0.5)

    async def main(self):
        print(f'mav_agent: {self.link} ({"USB port 2, the split" if self.link.startswith("/dev/") else "via the router"}) as 1/{COMP}, dashboard {self.a.dashboard}; periods status '
              f'{self.a.status_period} s, vio {self.a.vio_period} s, boards {self.a.board_period} s', flush=True)
        await asyncio.gather(self.ws_loop(), self.mav_loop(), self.tick_loop(), self.field_loop())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mavlink', default='auto', help='auto: the Pixhawk\'s 2nd USB port (-if02) if present, else '
                    f"mavlink-router's agent endpoint {ROUTER}; or any pymavlink link")
    ap.add_argument('--dashboard', default='http://127.0.0.1:8080')
    ap.add_argument('--link', default='/dev/ttyACM0:921600', help='the Pixhawk link, for "launch stack" (real): USB '
                    'port 1 (TELEM2 is the RC receiver on this drone)')
    ap.add_argument('--status-period', type=float, default=2.0)
    ap.add_argument('--vio-period', type=float, default=1.0, help='s between VIO position packets (0 = off)')
    ap.add_argument('--board-period', type=float, default=5.0,
                    help='s between periodic board resends; new / changed boards go at once (0 = no boards)')
    asyncio.run(Agent(ap.parse_args()).main())


if __name__ == '__main__':
    main()
