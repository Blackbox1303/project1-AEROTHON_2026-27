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

    def __init__(self, dev, qgc):
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
        for f in (self.rx_loop, self.qgc_loop, self.hb_loop):
            threading.Thread(target=f, daemon=True).start()

    def open(self):
        try:
            if self.dev.startswith(('udpin:', 'udpout:', 'tcp:')):
                self.conn = self.mavutil.mavlink_connection(self.dev, source_system=255, source_component=190)
            else:
                path, _, baud = self.dev.partition(':')
                if not Path(path).exists():
                    raise OSError(f'{path} not found (radio plugged in? ls /dev/serial/by-id/)')
                self.conn = self.mavutil.mavlink_connection(path, baud=int(baud or 57600), source_system=255,
                                                            source_component=190)
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
                for msg, hz in ((33, 2), (24, 1), (30, 2)):
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
            f = self.fc
            if t == 'HEARTBEAT':
                f['armed'] = bool(m.base_mode & 128)
                f['mode'] = self.mavutil.mode_string_v10(m)
            elif t == 'GLOBAL_POSITION_INT':
                f.update(lat=m.lat * 1e-7, lon=m.lon * 1e-7, alt_rel=round(m.relative_alt / 1000.0, 2))
            elif t == 'GPS_RAW_INT':
                f.update(fix=m.fix_type, sats=m.satellites_visible)
            elif t == 'ATTITUDE':
                f['hdg'] = round(math.degrees(m.yaw) % 360, 1)
            elif t == 'STATUSTEXT':
                self.add_text('PIXHAWK ' + m.text.rstrip('\0'))
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
                self.chunks[m.id] = self.chunks.get(m.id, '') + txt
                if len(txt) == 50:
                    return
                txt = self.chunks.pop(m.id)
            if txt.startswith(('FT ', 'AGT field')):
                self.add_text(txt)
        elif t == 'COMMAND_ACK' and m.command == fp.CMD:
            self.acks[fp.CMD] = (m.result, m.progress, now)

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
                'settings': {'valt': st.get('va')}, 'boards_done': st.get('n'), 'events': ev}


def build(a):
    app = web.Application()
    radio = Radio(a.radio, a.qgc) if a.radio else None
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
            return web.json_response(radio.state())
        d = await proxy_json('GET', '/api/state')
        if 'mode' not in d:
            return web.json_response({'link': 'cable', 'up': False, 'error': d.get('message'), 'guide': None,
                                      'events': []})
        d.update(link='cable', up=True)
        return web.json_response(d)

    async def cmd(request):
        body = await request.json()
        name = body.get('name')
        code = fp.BY_NAME.get(name)
        if code is None:
            return web.json_response({'ok': False, 'message': f'unknown command {name}'})
        if radio:
            if body.get('text'):
                return web.json_response({'ok': False, 'message': 'typing a start QR needs the cable (the radio '
                                                                  'carries numbers): read it with the camera instead'})
            loop = asyncio.get_running_loop()
            ok, msg = await loop.run_in_executor(None, radio.command, code, float(body.get('arg') or 0))
            return web.json_response({'ok': ok, 'message': msg})
        if name in ('start', 'stop'):
            ok, msg = await jetson_sh(name)
            return web.json_response({'ok': ok, 'message': msg})
        return web.json_response(await proxy_json('POST', '/api/cmd', body))

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
    g.add_argument('--radio', help='radio: the SiK radio, e.g. /dev/ttyUSB0:57600')
    ap.add_argument('--jetson-user', default=os.environ.get('JETSON_USER', 'szyfrowac'))
    ap.add_argument('--qgc', default='127.0.0.1:14550', help="--radio: QGroundControl's UDP port")
    ap.add_argument('--port', type=int, default=fp.PORT_GROUND)
    a = ap.parse_args()
    print(f'field test (ground, {"radio " + a.radio if a.radio else "cable " + a.jetson}): '
          f'http://127.0.0.1:{a.port}', flush=True)
    web.run_app(build(a), host='127.0.0.1', port=a.port, print=None)


if __name__ == '__main__':
    main()
