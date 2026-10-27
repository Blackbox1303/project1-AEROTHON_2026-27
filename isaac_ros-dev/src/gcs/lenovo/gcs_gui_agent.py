#!/usr/bin/env python3
"""Runs ON THE LENOVO (systemd --user service, see install_lenovo_agent.sh): lets the ground station dashboard
(http://10.42.0.247:8080, open in this PC's browser) open / close the Gazebo GUI here - a web page can't start
programs itself - and starts the Jetson's dashboard when it is not up (http://127.0.0.1:8765 in the browser).
Listens on 127.0.0.1:8765 only.
  GET  /                 this PC's page: Jetson dashboard up?  Start / launch / stop buttons (aerothon.sh)
  GET  /status           {"gui": bool, "phase": "syncing" | "waiting for the sim" | "open", "error": why the last
                         start ended, "dashboard": bool}
  POST /gui/start        ~/host_gui.sh (sync first if ?sync=1, else --nosync); an open GUI is replaced
  POST /gui/follow       ~/host_gui.sh --follow       POST /gui/free   ~/host_gui.sh --free
  POST /gui/stop         close it
  POST /jetson/<cmd>     ~/.local/bin/aerothon <cmd>: up | restart | launch | stop | status
"""
import glob
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPT = Path.home() / 'host_gui.sh'
AEROTHON = Path.home() / '.local/bin/aerothon'
LOG = Path.home() / '.cache/gcs_gui_agent.log'
DASHBOARD = 'http://10.42.0.247:8080'
state = {'proc': None, 'error': None, 'log_pos': 0, 'started': 0.0}
lock = threading.Lock()


def gui_env():
    """The desktop session's display. Started by systemd at boot, before the login, this service had no DISPLAY /
    WAYLAND_DISPLAY: gz sim -g died at once and "Open Gazebo" did nothing (2026-10-27). Take them from the user
    manager (the session imports them at login) or from the sockets in /run/user/<uid>."""
    env = dict(os.environ)
    try:
        out = subprocess.run(['systemctl', '--user', 'show-environment'], capture_output=True, text=True,
                             timeout=5).stdout
        for line in out.splitlines():
            k, _, v = line.partition('=')
            if k in ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY', 'DBUS_SESSION_BUS_ADDRESS', 'XDG_SESSION_TYPE'):
                env.setdefault(k, v)
    except (OSError, subprocess.SubprocessError):
        pass
    run = f'/run/user/{os.getuid()}'
    env.setdefault('XDG_RUNTIME_DIR', run)
    if 'WAYLAND_DISPLAY' not in env and os.path.exists(f'{run}/wayland-0'):
        env['WAYLAND_DISPLAY'] = 'wayland-0'
    env.setdefault('DISPLAY', ':0')
    if 'XAUTHORITY' not in env:                       # GNOME Wayland: XWayland's cookie
        auth = sorted(glob.glob(f'{run}/.mutter-Xwaylandauth.*')) or \
            [p for p in (str(Path.home() / '.Xauthority'),) if os.path.exists(p)]
        if auth:
            env['XAUTHORITY'] = auth[-1]
    return env


def log_tail(n=6):
    try:
        with open(LOG, 'rb') as f:
            f.seek(state['log_pos'])
            lines = [ln for ln in f.read().decode(errors='replace').splitlines() if ln.strip()]
        return ' | '.join(lines[-n:])[-400:]
    except OSError:
        return ''


def gui_running():
    p = state['proc']
    if p is None:
        return False
    if p.poll() is None:
        return True
    if state['error'] is None and p.returncode not in (0, -signal.SIGTERM, 143):
        state['error'] = f'exit {p.returncode}: {log_tail() or "no output"}'
    state['proc'] = None
    return False


def phase():
    tail = log_tail(2)
    if 'waiting for the sim' in tail:
        return 'waiting for the sim'
    if time.monotonic() - state['started'] < 120 and 'GUI: partition' not in log_tail(40):
        return 'syncing'
    return 'open'


def stop_gui():
    p = state['proc']
    if p is not None and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    state['proc'] = None


def dashboard_up():
    try:
        urllib.request.urlopen(DASHBOARD + '/', timeout=1.5).close()
        return True
    except OSError:
        return False


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,
initial-scale=1"><title>AeroTHON Lenovo Helper</title><style>
:root { --bg:#0f1318; --panel:#171d24; --text:#dfe6ee; --dim:#8b98a6; --ok:#3fbf6f; --bad:#e5484d; }
body { margin:0; padding:16px; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
.card { background:var(--panel); border-radius:8px; padding:14px; max-width:640px; margin:0 auto; }
button { background:#26313d; color:var(--text); border:1px solid #2a333d; border-radius:5px; padding:8px 12px;
  margin:4px 4px 0 0; cursor:pointer; font:inherit; } .go { background:#1d5c36; } .stop { background:#6b1f22; }
pre { background:#0b0f13; padding:8px; border-radius:6px; white-space:pre-wrap; min-height:40px; }
.ok { color:var(--ok); } .bad { color:var(--bad); } a { color:#4aa3ff; }
</style></head><body><div class="card"><h2>AeroTHON - this Lenovo</h2>
<p>Jetson dashboard: <b id="d">?</b> &middot; Gazebo GUI here: <b id="g">?</b></p>
<button class="go" onclick="run('up')">Start dashboard + open it</button>
<button onclick="run('restart')">Restart dashboard</button>
<button class="go" onclick="run('launch')">Launch sim stack</button>
<button onclick="post('/gui/start?sync=1')">Open Gazebo (sync)</button>
<button class="stop" onclick="run('stop')">Stop stack</button>
<p><a href="http://10.42.0.247:8080" target="_blank">open http://10.42.0.247:8080</a></p><pre id="out"></pre></div>
<script>
async function post(u) { document.getElementById('out').textContent = u + ' ...';
  const j = await (await fetch(u, {method: 'POST'})).json(); document.getElementById('out').textContent = j.message; }
const run = c => post('/jetson/' + c);
async function poll() { try { const j = await (await fetch('/status')).json();
  const d = document.getElementById('d'), g = document.getElementById('g');
  d.textContent = j.dashboard ? 'up' : 'DOWN'; d.className = j.dashboard ? 'ok' : 'bad';
  g.textContent = j.gui ? j.phase : (j.error ? 'failed: ' + j.error : 'closed'); g.className = j.error ? 'bad' : '';
} catch (e) {} }
setInterval(poll, 2000); poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Private-Network', 'true')   # Chrome: a LAN page calling localhost

    def reply(self, code, obj, ctype='application/json'):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.cors()
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self.reply(200, PAGE, 'text/html; charset=utf-8')
        if path == '/status':
            with lock:
                gui = gui_running()
                return self.reply(200, {'gui': gui, 'phase': phase() if gui else None, 'error': state['error'],
                                        'script': SCRIPT.exists(), 'dashboard': dashboard_up()})
        self.reply(404, {'error': 'not found'})

    def do_POST(self):
        u = urlparse(self.path)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if u.path.startswith('/jetson/'):
            cmd = u.path.split('/')[-1]
            if cmd not in ('up', 'restart', 'launch', 'stop', 'status'):
                return self.reply(404, {'ok': False, 'message': 'up | restart | launch | stop | status'})
            if not AEROTHON.exists():
                return self.reply(500, {'ok': False, 'message': f'{AEROTHON} missing: re-run install_lenovo_agent.sh'})
            r = subprocess.run(['bash', str(AEROTHON), cmd], capture_output=True, text=True, timeout=240,
                               env=gui_env())
            return self.reply(200, {'ok': r.returncode == 0, 'message': (r.stdout + r.stderr).strip()[-1500:]})
        if not SCRIPT.exists():
            return self.reply(500, {'ok': False, 'message': f'{SCRIPT} missing: re-run install_lenovo_agent.sh'})
        with lock:
            if u.path == '/gui/start':
                # an open GUI may show an old, stopped world: replace it
                stop_gui()
                args = [] if parse_qs(u.query).get('sync') == ['1'] else ['--nosync']
                state['error'] = None
                state['log_pos'] = LOG.stat().st_size if LOG.exists() else 0
                state['started'] = time.monotonic()
                with open(LOG, 'a') as log:
                    log.write(f'--- {time.ctime()} start {args}\n')
                    state['proc'] = subprocess.Popen(['bash', str(SCRIPT), *args], stdout=log,
                                                     stderr=subprocess.STDOUT, env=gui_env(), start_new_session=True)
                return self.reply(200, {'ok': True, 'message': 'Gazebo GUI: syncing, then opening on this PC '
                                                               '(it waits for the sim if it is still starting)'})
            if u.path in ('/gui/follow', '/gui/free'):
                with open(LOG, 'a') as log:
                    subprocess.Popen(['bash', str(SCRIPT), '--' + u.path.split('/')[-1]], stdout=log,
                                     stderr=subprocess.STDOUT, env=gui_env())
                return self.reply(200, {'ok': True, 'message': 'camera ' + u.path.split('/')[-1]})
            if u.path == '/gui/stop':
                was = state['proc'] is not None and state['proc'].poll() is None
                stop_gui()
                state['error'] = None
                return self.reply(200, {'ok': True, 'message': 'Gazebo GUI closed' if was else 'Gazebo GUI was not open'})
        self.reply(404, {'ok': False, 'message': 'not found'})

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    # the Gazebo GUI belongs on the Lenovo only: on the Jetson it costs ~0.9 GB and CPU the sim needs
    if Path('/etc/nv_tegra_release').exists():
        raise SystemExit('gcs_gui_agent: this is the Jetson - the Gazebo GUI runs on the Lenovo only')
    ThreadingHTTPServer(('127.0.0.1', 8765), H).serve_forever()
