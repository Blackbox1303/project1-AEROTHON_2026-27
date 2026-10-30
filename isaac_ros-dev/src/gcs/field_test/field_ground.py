#!/usr/bin/env python3
"""FIELD TEST, the laptop's page (http://127.0.0.1:8096) - props OFF, the drone carried by hand.

  aerothon fieldtest cable [jetson-ip]     the Ethernet cable to the Jetson (the Jetson also on the drone): the live
                                           Arducam / D435 colour / depth views + every button + the guidance. Start
                                           runs field_test.sh on the Jetson over ssh (the key `aerothon` installs).
  aerothon fieldtest radio [/dev/ttyUSB0:57600]   the telemetry radio ALONE: no video - the buttons go up as
                                           COMMAND_LONG 31010 action 11 to the onboard agent (1/194), the answers and
                                           the guidance come down as STATUSTEXT 'AGT field' / 'FT ...' + TUNNEL 32820;
                                           the Pixhawk's height / GPS / heading from its own telemetry. Every packet is
                                           relayed to QGroundControl (UDP 14550: QGC's SiK auto-connect OFF).
The same page (field_test.html) as the Jetson's own http://<jetson>:8097: /api/state, /api/cmd, /cam/*.jpg.
"""
import argparse
import asyncio
import json
import math
import os
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
import sys                                                # noqa: E402
sys.path.insert(0, str(HERE))
import field_proto as fp                                  # noqa: E402

JETSON_SH = '~/workspaces/isaac_ros-dev/src/gcs/field_test/field_test.sh'


def side_words(rel):
    a = abs(rel)
    if a < 20:
        return 'forward'
    if a > 160:
        return 'backward'
    lr = 'right' if rel > 0 else 'left'
    return f'forward-{lr}' if a < 60 else lr if a <= 120 else f'back-{lr}'


# the results panel: the latest answer of each check, from its command's reply and the Jetson's events (both links)
CHECKS = ('detect_qr', 'scan_qr', 'start_qr', 'walk', 'red', 'green', 'blue', 'obstacle')
PREFIX = (('detect QR', 'detect_qr'), ('scan QR', 'scan_qr'), ('start QR', 'start_qr'), ('red zone', 'red'),
          ('green', 'green'), ('blue', 'blue'), ('obstacles', 'obstacle'), ('QR ', 'walk'), ('board ', 'walk'))


def check_of(text):
    """An event's text -> the check it answers (None: not a result)."""
    return next((k for p, k in PREFIX if text.startswith(p)), None)


class WalkMap:
    """The walk-through on a map (user, 2026-10-30): the drone's GPS track (local metres from the first fix), the ground
    its camera has covered, and every detection where the drone was when it was seen."""

    def __init__(self, cover_r=1.2):
        self.o, self.track, self.hits, self.here, self.cover_r = None, [], [], None, cover_r
        self.lock = threading.Lock()

    def update(self, fc):
        lat, lon = fc.get('lat'), fc.get('lon')
        if not lat or not lon or (fc.get('fix') or 0) < 3:
            return
        if self.o is None:
            self.o = (lat, lon)
        x = (lon - self.o[1]) * 111320.0 * math.cos(math.radians(self.o[0]))
        y = (lat - self.o[0]) * 110540.0
        with self.lock:
            self.here = [round(x, 2), round(y, 2), fc.get('hdg')]
            if not self.track or math.dist(self.track[-1], (x, y)) > 0.3:
                self.track = (self.track + [[round(x, 2), round(y, 2)]])[-3000:]

    def hit(self, key, ok, text):
        with self.lock:
            if self.here is not None:
                self.hits = (self.hits + [{'x': self.here[0], 'y': self.here[1], 'key': key, 'ok': ok,
                                           'text': text[:160], 't': time.strftime('%H:%M:%S')}])[-200:]

    def get(self):
        with self.lock:
            return {'track': list(self.track), 'here': self.here, 'hits': list(self.hits), 'cover_r': self.cover_r,
                    'gps': self.o is not None}


class Results:
    def __init__(self, walk=None):
        self.r, self.lock, self.walk = {}, threading.Lock(), walk

    def put(self, key, ok, text):
        if key in CHECKS and text:
            with self.lock:
                old = self.r.get(key)
                if old and old['text'] == text and time.time() - old['ts'] < 3:   # the reply and its event: once
                    return
                self.r[key] = {'ok': ok, 'text': text, 't': time.strftime('%H:%M:%S'), 'ts': time.time(),
                               'n': (old or {}).get('n', 0) + 1}
            if self.walk is not None:
                self.walk.hit(key, ok, text)

    def event(self, text):
        self.put(check_of(text), True, text)

    def get(self):
        with self.lock:
            return {k: dict(v, age=round(time.time() - v['ts'])) for k, v in self.r.items()}


def say_of(st, yaw_tol=35):
    """The radio's compact guidance (TUNNEL 32820) -> words, as the Jetson says them."""
    parts = [st.get('s', '-')]
    y, g, d, h = st.get('y'), st.get('g'), st.get('d'), st.get('h', 0)
    if y is not None and abs(y) >= yaw_tol:
        parts.append(f'YAW {"RIGHT" if y > 0 else "LEFT"} {abs(y)} deg')
    elif g is not None:
        parts.append(f'WALK {side_words(g)}' + (f' {d} m' if d is not None else ''))
    if h:
        parts.append(f'HEIGHT {"UP" if h > 0 else "DOWN"}')
    if st.get('va') is not None and st.get('m') == 'q':
        parts.append(f'(virtual {st["va"]} m)')
    return ' - '.join(parts)


class Radio:
    """The SiK radio: this app owns it, relays everything to QGC, talks to the onboard agent."""

    def __init__(self, dev, qgc, results):
        self.results = results
        os.environ['MAVLINK20'] = '1'
        from pymavlink import mavutil
        self.mavutil, self.dev, self.conn, self.err = mavutil, dev, None, 'opening'
        self.chunks, self.texts = {}, []              # STATUSTEXT id -> text so far; [(time, text)]
        self.field, self.t_field, self.t_agent, self.t_fc = {}, 0.0, 0.0, 0.0
        self.fc = {}
        self.acks = {}                                # (command) -> (result, progress, time)
        self.press = int(time.time()) % 1000
        self.lock = threading.Lock()
        self.qgc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.qgc.bind(('127.0.0.1', 0))
        self.qgc.setblocking(False)
        h, _, p = qgc.partition(':')
        self.qgc_addr = (h, int(p))
        self.drop = fp.Rc5Drop(self.fc, self.send_cmd, self.rc5, self.request_param)
        for f in (self.rx_loop, self.qgc_loop, self.hb_loop):
            threading.Thread(target=f, daemon=True).start()

    # ---- the autopilot, straight over the radio (this app is the GCS, sysid 255): mode / winch tests
    def send_cmd(self, cmd, *p):
        if self.conn is not None:
            p = (list(p) + [0] * 7)[:7]
            self.conn.mav.command_long_send(1, 1, cmd, 0, *p)

    def rc5(self, pwm):
        if self.conn is not None:                   # channel 5 only; 65535 = leave the others alone, 0 = release
            self.conn.mav.rc_channels_override_send(1, 1, 65535, 65535, 65535, 65535, int(pwm), 65535, 65535, 65535)

    def request_param(self, name):
        if self.conn is not None:
            self.conn.mav.param_request_read_send(1, 1, name.encode(), -1)

    def set_mode(self, num):
        """The mode switch test: DO_SET_MODE, disarmed only -> (ok, message)."""
        num = int(num)
        if num not in fp.TEST_MODES:
            return False, 'modes: 0 STABILIZE, 3 AUTO, 6 RTL'
        if self.fc.get('armed') is not False:
            return False, f'disarmed only (armed {self.fc.get("armed")})'
        self.acks.pop(176, None)
        for _ in range(3):
            self.send_cmd(176, 1, num)
            t0 = time.time()
            while time.time() - t0 < 1.5 and 176 not in self.acks:
                time.sleep(0.05)
            if 176 in self.acks:
                break
        r = self.acks.get(176)
        name = fp.TEST_MODES[num]
        if r is None:
            return False, f'{name}: no answer from the autopilot'
        time.sleep(1.2)                             # its next heartbeat
        return r[0] == 0, f'{name}: {"accepted" if r[0] == 0 else "REFUSED (result " + str(r[0]) + ")"} - now ' \
                          f'{self.fc.get("mode")} ({self.fc.get("fc_text", "")})'

    def open(self):
        try:
            if self.dev.startswith(('udpin:', 'udpout:', 'tcp:')):
                self.conn = self.mavutil.mavlink_connection(self.dev, source_system=255, source_component=190)
            else:
                path, _, baud = self.dev.partition(':')
                path = fp.find_radio(path) or path               # not there: the one radio plugged in
                if not Path(path).exists():
                    raise OSError(f'{path} not found (radio plugged in? ls /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/)')
                self.conn = self.mavutil.mavlink_connection(path, baud=int(baud or 57600), source_system=255,
                                                            source_component=190)
                self.dev = f'{path}:{baud or 57600}'
            self.err = None
        except Exception as e:
            self.conn = None
            self.err = str(e) + (' - sudo usermod -aG dialout $USER, log out / in' if 'ermission' in str(e) else '')

    def rx_loop(self):
        t_req = 0.0
        while True:
            if self.conn is None:
                self.open()
                if self.conn is None:
                    time.sleep(2)
                    continue
            if time.time() - t_req > 15:              # the Pixhawk's height / GPS / heading, light on the radio
                for msg, hz in fp.FC_STREAMS:            # the panel: attitude, throttle, RC 5 / 7, motors, RPM
                    self.conn.mav.command_long_send(1, 1, 511, 0, msg, 1e6 / hz, 0, 0, 0, 0, 0)
                t_req = time.time()
            try:
                m = self.conn.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                self.err, self.conn = f'radio: {e}', None
                continue
            if m is None or m.get_type() == 'BAD_DATA':
                continue
            try:
                self.qgc.sendto(m.get_msgbuf(), self.qgc_addr)
            except OSError:
                pass
            self.handle(m)

    def handle(self, m):
        t, src, comp, now = m.get_type(), m.get_srcSystem(), m.get_srcComponent(), time.time()
        if src != 1:
            return
        if comp == 1:
            self.t_fc = now
            fp.fc_update(self.fc, m, self.mavutil.mode_string_v10)
            if t == 'STATUSTEXT':
                self.add_text('PIXHAWK ' + m.text.rstrip('\0'))
            elif t == 'COMMAND_ACK':
                self.acks[m.command] = (m.result, m.progress, now)
            return
        if comp != fp.AGENT_COMP:
            return
        self.t_agent = now
        if t == 'TUNNEL' and m.payload_type == fp.T_FIELD:
            try:
                self.field = json.loads(bytes(m.payload[:m.payload_length]).decode())
                self.t_field = now
            except ValueError:
                pass
        elif t == 'STATUSTEXT':
            txt = m.text.rstrip('\0')
            if m.id:
                for k in [k for k in self.chunks if k != m.id]:   # a text of exactly 50 x n chars has no end chunk
                    self.agent_text(self.chunks.pop(k))
                self.chunks[m.id] = self.chunks.get(m.id, '') + txt
                if len(txt) == 50:
                    return
                txt = self.chunks.pop(m.id)
            self.agent_text(txt)
        elif t == 'COMMAND_ACK' and m.command == fp.CMD:
            self.acks[fp.CMD] = (m.result, m.progress, now)

    def agent_text(self, txt):
        if txt.startswith(('FT ', 'AGT field')):
            self.add_text(txt)
        if txt.startswith('FT '):
            self.results.event(txt[3:])

    def add_text(self, txt):
        with self.lock:
            self.texts = (self.texts + [(time.strftime('%H:%M:%S'), txt)])[-80:]

    def qgc_loop(self):
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
                self.conn.mav.heartbeat_send(6, 8, 0, 0, 4)
            except Exception:
                pass
            time.sleep(1.0)

    def command(self, code, arg):
        """-> (ok, message): COMMAND_LONG to the agent, resent until it answers; its STATUSTEXT says the result."""
        if self.conn is None:
            return False, f'radio not open: {self.err}'
        self.press = (self.press + 1) % 100000
        self.acks.pop(fp.CMD, None)
        t0, t_send, n = time.time(), 0.0, 0
        long_one = code in (1, 2)                     # start / stop: field_test.sh takes up to a minute
        while time.time() - t0 < (120 if long_one else 40):
            a = self.acks.get(fp.CMD)
            if a and a[0] != self.mavutil.mavlink.MAV_RESULT_IN_PROGRESS:
                ok = a[0] == self.mavutil.mavlink.MAV_RESULT_ACCEPTED
                time.sleep(0.6)                       # its STATUSTEXT follows the ACK
                with self.lock:
                    msg = next((t for _, t in reversed(self.texts) if t.startswith('AGT field')), '')
                for pre in ('AGT field: OK ', 'AGT field: FAILED ', 'AGT field: OK', 'AGT field: FAILED'):
                    if msg.startswith(pre):
                        msg = msg[len(pre):]
                        break
                return ok, msg or ('accepted' if ok else 'FAILED')
            if a is None and time.time() - t_send > 1.5 and n < 4:   # no answer at all: send again (same press)
                self.conn.mav.command_long_send(1, fp.AGENT_COMP, fp.CMD, 0, fp.ACTION_FIELD, code, arg,
                                                self.press, 0, 0, 0)
                t_send, n = time.time(), n + 1
            time.sleep(0.05)
        if fp.CMD not in self.acks:
            return False, 'no answer from the onboard agent (radio link? Jetson up? aerothon-mav-agent running?)'
        return False, 'the agent is still on it - its answer comes in the log'

    def state(self):
        now = time.time()
        st = self.field
        fresh = now - self.t_field < 4.0
        g = {'mode': {'i': 'idle', 'q': 'walk_qr', 'o': 'walk_obstacle'}.get(st.get('m'), 'idle'),
             'state': st.get('s', '-'), 'say': say_of(st) if fresh else 'no field test guidance on the radio',
             'yaw': st.get('y'), 'go': st.get('g'), 'dist': st.get('d'), 'height': st.get('h', 0),
             'valt': st.get('va'),
             'motor': {'pattern': fp.CODE_PATTERN.get((st.get('mc') or 'om')[0], 'off'),
                       'level': {'l': 'low', 'm': 'mid', 'h': 'high'}.get((st.get('mc') or 'om')[-1], 'mid')}}
        with self.lock:
            ev = [{'t': t, 'text': x} for t, x in self.texts[-40:]]
        return {'link': 'radio', 'radio_err': self.err, 'up': fresh,
                'ages': {'agent': round(now - self.t_agent, 1) if self.t_agent else None,
                         'field': round(now - self.t_field, 1) if self.t_field else None,
                         'fc': round(now - self.t_fc, 1) if self.t_fc else None},
                'mode': g['mode'], 'guide': g, 'motors_on': bool(st.get('mo')), 'fc': self.fc,
                'settings': {'valt': st.get('va')}, 'boards_done': st.get('n'), 'events': ev,
                'results': self.results.get(), 'drop': dict(self.drop.state)}


def build(a):
    app = web.Application()
    walk = WalkMap()
    results = Results(walk)
    radio = Radio(a.radio, a.qgc, results) if a.radio else None
    seen = {'n': None}                                    # cable: the Jetson's events already taken
    jet = a.jetson.rstrip('/') if a.jetson else None
    host = urlparse(jet).hostname if jet else None

    async def page(_):
        return web.FileResponse(HERE / 'field_test.html')

    async def info(_):
        return web.json_response({'side': 'ground', 'link': 'radio' if radio else 'cable', 'jetson': jet,
                                  'codes': fp.CODES})

    async def jetson_sh(what):
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new',
               f'{a.jetson_user}@{host}', f'{JETSON_SH} {what}']
        try:
            p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(p.communicate(), 120)
        except Exception as e:
            return False, f'ssh: {e}'
        lines = [ln for ln in out.decode(errors='replace').splitlines() if ln.strip()]
        if p.returncode == 255:
            return False, (f'no ssh to {a.jetson_user}@{host} ({lines[-1] if lines else ""}) - on the Jetson run: '
                           f'{JETSON_SH} {what}')
        return p.returncode == 0, ' | '.join(lines[-2:])

    async def proxy_json(method, path, body=None):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.request(method, jet + path, json=body, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    return await r.json()
        except Exception as e:
            return {'ok': False, 'message': f'Jetson {jet}: {type(e).__name__} {e}'.strip()}

    async def state(_):
        if radio:
            st = radio.state()
            walk.update(st['fc'])
            st['walk'] = walk.get()
            return web.json_response(st)
        d = await proxy_json('GET', '/api/state')
        if 'mode' not in d:
            return web.json_response({'link': 'cable', 'up': False, 'error': d.get('message'), 'guide': None,
                                      'events': []})
        evs = [e for e in d.get('events') or [] if isinstance(e.get('n'), int)]
        if seen['n'] is None:                             # first look: its 'last' answers, not its whole history
            for k, v in (d.get('last') or {}).items():
                results.put(k, True, v)
        for e in evs:
            if seen['n'] is not None and e['n'] > seen['n']:
                results.event(e['text'])
        if evs:
            seen['n'] = max(e['n'] for e in evs)
        elif seen['n'] is None:
            seen['n'] = 0
        walk.update(d.get('fc') or {})
        d.update(link='cable', up=True, results=results.get(), walk=walk.get())
        return web.json_response(d)

    async def cmd(request):
        body = await request.json()
        name = body.get('name')
        code = fp.BY_NAME.get(name)
        if code is None:
            return web.json_response({'ok': False, 'message': f'unknown command {name}'})
        if radio and name in ('set_mode', 'winch'):    # straight to the autopilot over the radio (no Jetson needed)
            loop = asyncio.get_running_loop()
            if name == 'set_mode':
                ok, msg = await loop.run_in_executor(None, radio.set_mode, float(body.get('arg') or 0))
            else:
                ok, msg = radio.drop.start(float(body.get('arg') or 0))
            radio.add_text(('MODE ' if name == 'set_mode' else 'WINCH ') + msg)
            return web.json_response({'ok': ok, 'message': msg})
        if radio:
            if body.get('text'):
                return web.json_response({'ok': False, 'message': 'typing a start QR needs the cable (the radio '
                                                                  'carries numbers): read it with the camera instead'})
            loop = asyncio.get_running_loop()
            ok, msg = await loop.run_in_executor(None, radio.command, code, float(body.get('arg') or 0))
            results.put(name, ok, msg)
            return web.json_response({'ok': ok, 'message': msg})
        if name in ('start', 'stop'):
            ok, msg = await jetson_sh(name)
            return web.json_response({'ok': ok, 'message': msg})
        d = await proxy_json('POST', '/api/cmd', body)
        results.put(name, bool(d.get('ok')), str(d.get('message', '')))
        return web.json_response(d)

    async def levels(request):
        if radio:
            return web.json_response({'ok': False, 'message': 'spin levels: by cable (field_vision --spin-*)'})
        return web.json_response(await proxy_json('POST', '/api/levels', await request.json()))

    async def cam(request):
        if radio:
            return web.Response(status=503, text='no video over the radio')
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f'{jet}/cam/{request.match_info["name"]}.jpg',
                                 timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return web.Response(status=r.status, body=await r.read(), content_type=r.content_type,
                                        headers={'Cache-Control': 'no-store'})
        except Exception as e:
            return web.Response(status=503, text=str(e))

    app.router.add_get('/', page)
    app.router.add_get('/api/info', info)
    app.router.add_get('/api/state', state)
    app.router.add_post('/api/cmd', cmd)
    app.router.add_post('/api/levels', levels)
    app.router.add_get('/cam/{name}.jpg', cam)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--jetson', help='cable: the Jetson field test, e.g. http://10.42.0.247:8097')
    g.add_argument('--radio', help='radio: the SiK radio, e.g. /dev/ttyUSB0:57600 (not there: the one radio plugged in)')
    ap.add_argument('--jetson-user', default=os.environ.get('JETSON_USER', 'szyfrowac'))
    ap.add_argument('--qgc', default='127.0.0.1:14550', help="--radio: QGroundControl's UDP port")
    ap.add_argument('--port', type=int, default=fp.PORT_GROUND)
    a = ap.parse_args()
    print(f'field test (ground, {"radio " + a.radio if a.radio else "cable " + a.jetson}): '
          f'http://127.0.0.1:{a.port}', flush=True)
    web.run_app(build(a), host='127.0.0.1', port=a.port, print=None)


if __name__ == '__main__':
    main()
