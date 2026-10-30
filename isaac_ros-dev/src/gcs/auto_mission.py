#!/usr/bin/env python3
"""THE FINAL MISSION in one command, AUTO (user, 2026-10-30): `aerothon mission` starts the 8080 radio dashboard, then
this script drives it through its own API (the same checks and radio commands as its buttons):
  1. the radio: autopilot heartbeat + the onboard Jetson's agent heard
  2. the geofence saved on this PC (the page's Geofence card: Save) -> Apply (sent up the radio)
  3. Launch stack (ArUco) -> wait READY
  4. wait for every critical preflight check green
  5. a COUNTDOWN (Ctrl-C cancels: nothing flies) -> Start: the whole mission, autonomous
  6. watch it: the state, the flight mode, until landed + disarmed
After the start the only thing to do on the ground is the page's MANUAL button (STABILIZE: the RC pilot flies) - or
LAND / RTH / RTL / KILL in an emergency; the RC mode switch hands over too. Ctrl-C here then only stops watching: the
mission goes on. Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.request

OWN_MODES = {'GUIDED', 'LAND'}   # what the mission sets itself; anything else in flight = handed over


def say(msg):
    print(f'[mission {time.strftime("%H:%M:%S")}] {msg}', flush=True)


class Page:
    def __init__(self, url):
        self.url = url.rstrip('/')

    def post(self, what, body=None, timeout=300.0):
        req = urllib.request.Request(f'{self.url}/api/{what}', data=json.dumps(body or {}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def status(self):
        try:
            return self.post('auto_status', timeout=5.0)
        except Exception as e:
            return {'ok': False, 'message': f'the dashboard: {e}'}


def wait(page, what, cond, timeout, every=5.0):
    """Poll auto_status until cond(status) -> (True, status); print what is missing every `every` s."""
    t0, last = time.monotonic(), 0.0
    while True:
        st = page.status()
        ok, why = cond(st)
        if ok:
            return st
        if time.monotonic() - t0 > timeout:
            raise SystemExit(f'[mission] gave up waiting for {what} after {timeout:.0f} s: {why}')
        if time.monotonic() - last > every:
            say(f'waiting for {what}: {why}')
            last = time.monotonic()
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--countdown', type=int, default=10, help='s before the start (Ctrl-C cancels)')
    ap.add_argument('--segment', default='full', help='full (the whole mission) | qr | qr_drop | tunnel')
    ap.add_argument('--check', action='store_true', help='only the radio, the Jetson and the saved geofence: nothing is '
                                                         'sent to the drone')
    a = ap.parse_args()
    page = Page(a.url)

    # 1. the radio link
    st = wait(page, 'the dashboard', lambda s: (s.get('ok', False), s.get('message')), 60)
    if not st.get('radio'):
        raise SystemExit('[mission] the 8080 dashboard is not in radio mode (start it with `aerothon mission`)')
    wait(page, 'the autopilot over the radio', lambda s: (bool(s.get('link')), 'no heartbeat (radio plugged in? '
                                                         'drone powered?)'), 120)
    say('autopilot heard over the radio')
    wait(page, 'the onboard Jetson', lambda s: (bool(s.get('agent')), 'mav_agent not heard (Jetson booted? ~1 min)'),
         180)
    say('onboard Jetson heard')
    st = page.status()
    if st.get('armed'):
        raise SystemExit('[mission] the drone is ARMED - the auto start is for a drone on the ground, disarmed')

    # 2. the geofence
    gf = st.get('gf') or {}
    if gf.get('arena', 0) < 4 and gf.get('qr', 0) < 4:
        raise SystemExit(f'[mission] no geofence saved on this PC (arena {gf.get("arena", 0)}, QR area {gf.get("qr", 0)} '
                         'points): fill the Geofence card on http://127.0.0.1:8080 and press Save, then run again')
    say(f'geofence: arena {gf.get("arena")}, QR area {gf.get("qr")}, extra {gf.get("extra")} points')
    if a.check:
        say(f'check done: radio, Jetson and geofence OK; stack {st.get("message")}; preflight failing: '
            f'{"; ".join(st.get("failing") or []) or "none"} - nothing was sent to the drone')
        return
    say('geofence -> Apply (up the radio)')
    r = page.post('apply', {'mode': 'real'})
    say(f'apply: {"OK" if r.get("ok") else "FAILED"} {r.get("message", "")}')
    if not r.get('ok'):
        raise SystemExit('[mission] the geofence did not reach the drone - check the radio and run again')

    # 3. the stack
    phase = st.get('message', '')
    if phase != 'ready':
        say('Launch stack (ArUco) - 1 to 3 min')
        r = page.post('stack_start', {'mode': 'real'})
        say(f'launch: {"OK" if r.get("ok") else "FAILED"} {r.get("message", "")}')
        if not r.get('ok') and 'already' not in str(r.get('message', '')):
            raise SystemExit('[mission] the stack did not start')
    wait(page, 'stack READY', lambda s: (s.get('message') == 'ready', s.get('message')), 300, every=10)
    say('stack READY')

    # 4. the preflight
    wait(page, 'the preflight', lambda s: (not s.get('failing'), 'failing: ' + '; '.join(s.get('failing') or [])),
         180, every=5)
    say('preflight green')

    # 5. the countdown, then AUTO
    try:
        for k in range(a.countdown, 0, -1):
            print(f'\r[mission] STARTING THE MISSION (auto) in {k:2d} s - Ctrl-C cancels, nothing flies ', end='',
                  flush=True)
            time.sleep(1.0)
        print()
    except KeyboardInterrupt:
        print()
        say('cancelled - nothing started (the stack stays up: Stop stack on the page, or run again)')
        return
    st = page.status()
    if st.get('failing') or st.get('message') != 'ready' or st.get('armed'):
        raise SystemExit(f'[mission] not started: {st.get("message")} {st.get("failing")}')
    r = page.post('mission_start', {'type': a.segment})
    say(f'START: {"OK" if r.get("ok") else "FAILED"} {r.get("message", "")}')
    if not r.get('ok'):
        raise SystemExit(1)
    say('the mission is AUTO now. On the page: MANUAL (the RC pilot flies) or LAND / RTH / RTL / KILL only. '
        'Ctrl-C here stops watching, not the mission.')

    # 6. watch
    last, armed_seen, t0 = None, False, time.monotonic()
    try:
        while True:
            st = page.status()
            now = (st.get('state'), st.get('fc_mode'), st.get('armed'))
            if now != last:
                say(f'state {now[0] or "-"} | flight mode {now[1] or "-"} | {"ARMED" if now[2] else "disarmed"}')
                if now[2] and now[1] and now[1] not in OWN_MODES:
                    say(f'{now[1]}: the pilot / the RC switch / an autopilot failsafe has the drone - the mission '
                        f'stopped')
                last = now
            armed_seen = armed_seen or bool(st.get('armed'))
            if armed_seen and st.get('armed') is False:
                say(f'landed and disarmed after {time.monotonic() - t0:.0f} s - Stop stack on the page when done')
                return
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()
        say('stopped watching - the mission goes on (the page shows it)')


if __name__ == '__main__':
    sys.exit(main())
