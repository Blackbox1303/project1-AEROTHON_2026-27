#!/usr/bin/env python3
"""Onboard MAVLink agent (Jetson host, systemd --user aerothon-mav-agent): lets the ground run the mission over the
TELEMETRY RADIO alone - no Wi-Fi. A MAVLink component of the drone (system 1, component 194 = ONBOARD_COMPUTER2;
MAVROS is 191) on mavlink-router's agent endpoint (UDP 14660: router/sim.conf, onboard/install_onboard.sh).
  commands  COMMAND_LONG MAV_CMD_USER_1 (31010) to 1/194, param1 = action:
              1 launch stack (param2: 0 real drone, 1 sim world 'competition')   2 stop stack
              3 start FULL mission   4 start DELIVERY only   5 ABORT (supervisor failsafe: hold, land)   6 status now
              7 start a segment (param2: 1 full 2 qr 3 tunnel 4 qr_drop)   8 operator took over (after the ground's
              LAND / RTH / RTL / KILL: the supervisor stops without a failsafe flight)
            -> the onboard dashboard's REST API (http://127.0.0.1:8080/api/...: its preflight gate, geofence check and
            stack logic apply unchanged) -> COMMAND_ACK (ACCEPTED / FAILED) + STATUSTEXT 'AGT: <its message>'
  status    TUNNEL (payload_type 32800, 128 B JSON) every status_period s: {p: stack phase, md: sim|real, s: FSM state,
            t: mission s, z: QR-area s, q: QRs read, tg: target found 0/1, pf: failing critical preflight checks,
            al: alerts}; payload_type 32801 every 3rd time: {f: failing preflight names, a: first alert}
  events    STATUSTEXT on a change of stack phase / FSM state, a new QR, the target, a new alert (QGC shows them too)
  heartbeat 1 Hz, MAV_TYPE_ONBOARD_CONTROLLER
A radio at 57600 baud carries ~5 KB/s: this adds ~70 B/s. PX4 must forward between its MAVLink ports
(MAV_0_FORWARD = MAV_1_FORWARD = 1) for the radio on TELEM1 to reach the Jetson on TELEM2.
"""
import argparse
import asyncio
import json
import os
import re
import time

import aiohttp
os.environ['MAVLINK20'] = '1'      # MAVLink 2 before pymavlink loads: TUNNEL (385) and chunked STATUSTEXT need it
from pymavlink import mavutil  # noqa: E402

COMP = 194                       # MAV_COMP_ID_ONBOARD_COMPUTER2
CMD = 31010                      # MAV_CMD_USER_1
ACTIONS = {1: 'launch', 2: 'stop', 3: 'full', 4: 'delivery', 5: 'abort', 6: 'status', 7: 'segment', 8: 'operator'}
SEGMENTS = {1: 'full', 2: 'qr', 3: 'tunnel', 4: 'qr_drop'}   # action 7, param2 (the 8080 page in --radio mode)
T_STATUS, T_DETAIL = 32800, 32801
ALERT_REPEAT = 60.0              # s between two STATUSTEXTs of the same alert


class Agent:
    def __init__(self, a):
        self.a = a
        self.mav = mavutil.mavlink_connection(a.mavlink, source_system=1, source_component=COMP, dialect='common')
        self.snap = None             # latest onboard dashboard snapshot
        self.last = {}               # what the events compare against
        self.n_status = 0
        self.done_cmds = {}          # (sysid, compid, action) -> time: a resent command isn't run twice
        self.alert_sent = {}         # alert without its numbers -> time sent

    # ---- MAVLink out
    def text(self, msg, severity=6):
        """STATUSTEXT, chunked (MAVLink 2: 50 chars a chunk, same id)."""
        b = msg.encode()[:200]
        mid = int(time.time() * 10) % 65535 + 1
        for k in range(0, max(1, len(b)), 50):
            self.mav.mav.statustext_send(severity, b[k:k + 50].ljust(50, b'\0'), mid, k // 50)

    def tunnel(self, ptype, obj):
        raw = json.dumps(obj, separators=(',', ':')).encode()[:128]
        self.mav.mav.tunnel_send(0, 0, ptype, len(raw), list(raw.ljust(128, b'\0')))

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

    async def run_action(self, action, p2):
        if action == 'launch':
            body = {'mode': 'sim', 'world': 'competition'} if int(p2) == 1 else {'mode': 'real', 'link': self.a.link}
            return await self.api('stack_start', body)
        if action == 'stop':
            return await self.api('stack_stop', {})
        if action in ('full', 'delivery'):
            return await self.api('mission_start', {'type': action})
        if action == 'abort':
            return await self.api('mission_abort', {})
        if action == 'segment':
            seg = SEGMENTS.get(int(p2))
            if seg is None:
                return False, f'segment code {p2}: 1 full 2 qr 3 tunnel 4 qr_drop'
            return await self.api('mission_start', {'type': seg})
        if action == 'operator':                      # the ground's LAND / RTH / RTL / KILL: no failsafe flight over it
            return await self.api('operator', {})
        if action == 'status':
            self.send_status(detail=True)
            return True, 'status sent'
        return False, 'unknown action'

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
            m = await loop.run_in_executor(None, lambda: self.mav.recv_match(type='COMMAND_LONG', blocking=True,
                                                                                timeout=1.0))
            if m is None or m.target_component != COMP or m.command != CMD:
                continue
            action = ACTIONS.get(int(m.param1))
            key = (m.get_srcSystem(), m.get_srcComponent(), action)
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
            ok, msg = await self.run_action(action, m.param2)
            print(f'{time.strftime("%H:%M:%S")} {action}: {ok} {msg}', flush=True)
            self.mav.mav.command_ack_send(CMD, mavutil.mavlink.MAV_RESULT_ACCEPTED if ok else
                                          mavutil.mavlink.MAV_RESULT_FAILED, 100, 0,
                                          m.get_srcSystem(), m.get_srcComponent())
            self.text(f'AGT {action}: {"OK" if ok else "FAILED"} {msg}', 6 if ok else 4)

    async def tick_loop(self):
        t_hb = t_st = 0.0
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
            self.events()
            await asyncio.sleep(0.5)

    async def main(self):
        print(f'mav_agent: {self.a.mavlink} as 1/{COMP}, dashboard {self.a.dashboard}', flush=True)
        await asyncio.gather(self.ws_loop(), self.mav_loop(), self.tick_loop())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mavlink', default='udpin:127.0.0.1:14660', help="mavlink-router's agent endpoint")
    ap.add_argument('--dashboard', default='http://127.0.0.1:8080')
    ap.add_argument('--link', default='/dev/ttyTHS1:921600', help='the Pixhawk link, for "launch stack" (real)')
    ap.add_argument('--status-period', type=float, default=2.0)
    asyncio.run(Agent(ap.parse_args()).main())


if __name__ == '__main__':
    main()
