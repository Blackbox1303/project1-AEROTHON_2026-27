#!/usr/bin/env python3
"""AeroTHON app switcher: one page (http://127.0.0.1:8079) to switch between the ground apps without starting each one
separately - onboard (the final mission on the radio), offboard (this laptop + VRX + SiK), sim testing (the Jetson's
dashboard by Ethernet), each with ArUco or QR boards. Every button runs the `aerothon` launcher next to this file,
which closes the running app first (one app at a time). Python standard library only."""
import html
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
LAUNCHER = str(HERE / 'aerothon')
PORT = 8079

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>AeroTHON switcher</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f3f5f6;--panel:#fff;--ink:#18212b;--muted:#5a6876;--line:#d6dde3;--accent:#c2410c;--ok:#2f7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#11171c;--panel:#182129;--ink:#e3e9ee;--muted:#9aa8b4;--line:#2b3844;--accent:#f0803c;--ok:#6cc890}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
h1{margin:0;font-size:26px} p{margin:0;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:10px}
.card h2{margin:0;font-size:17px} .row{display:flex;gap:8px;flex-wrap:wrap}
button{font:600 14px system-ui;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:8px 12px;cursor:pointer}
button.go{background:var(--accent);border-color:var(--accent);color:#fff} button:disabled{opacity:.5}
label{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:3px}
input,select{font:14px ui-monospace,monospace;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
#st{white-space:pre-wrap;font:13px ui-monospace,monospace;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-height:3em}
#now{font-weight:600} #now a{color:var(--accent)} .live{color:var(--ok)}
</style></head><body><main>
<h1>AeroTHON switcher</h1>
<p>One app at a time: pressing a button closes the running app and starts the new one.</p>
<div id=now>-</div>
<div class=opts>
 <label>Ground radio <input id=radio value="/dev/ttyUSB0:57600"></label>
 <label>VRX video (offboard) <input id=video placeholder="/dev/videoN (empty: pick on the page)"></label>
 <label>Target code (offboard) <input id=target placeholder="empty: read the start board"></label>
 <label>Payload (offboard) <select id=drop><option value=drop>QR area + drop</option><option value=nodrop>QR area only</option></select></label>
</div>
<div class=grid>
 <div class=card><h2>Onboard: the final mission</h2><p>The dashboard on the ground SiK radio. The Jetson flies on the drone.</p>
  <div class=row><button class=go onclick="go('onboard','aruco')">ArUco</button><button class=go onclick="go('onboard','qr')">QR</button></div></div>
 <div class=card><h2>Offboard: this laptop</h2><p>VRX receiver + SiK radio on this laptop. QR area only. ArUco at 1 m, QR at 10 m.</p>
  <div class=row><button class=go onclick="go('offboard','aruco')">ArUco</button><button class=go onclick="go('offboard','qr')">QR</button></div></div>
 <div class=card><h2>Field test: props OFF</h2><p>Cameras, QR / red / banner / obstacle scans and the walk-through with motor cues. Cable: with video. Radio: text only.</p>
  <div class=row><button class=go onclick="go('fieldtest','cable')">By cable</button><button class=go onclick="go('fieldtest','radio')">Over the radio</button></div></div>
 <div class=card><h2>Sim testing: the Jetson</h2><p>Needs an Ethernet cable. The Jetson's full dashboard: Sim, Offboard Jetson, Real, Open Gazebo.</p>
  <div class=row><button class=go onclick="go('sim','aruco')">ArUco (8084)</button><button class=go onclick="go('sim','qr')">QR (8080)</button></div></div>
</div>
<div class=row><button onclick="go('stop')">Stop the running app</button><button onclick="go('status')">Status</button></div>
<div id=st></div>
</main><script>
const v = id => document.getElementById(id).value.trim();
async function go(mode, marker) {
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  document.getElementById('st').textContent = mode + (marker ? ' ' + marker : '') + ' ...';
  try {
    const q = new URLSearchParams({mode, marker: marker || '', radio: v('radio'), video: v('video'), target: v('target'), drop: v('drop')});
    const r = await (await fetch('/run?' + q)).json();
    document.getElementById('st').textContent = r.out;
  } catch (e) { document.getElementById('st').textContent = 'switcher: ' + e; }
  document.querySelectorAll('button').forEach(b => b.disabled = false);
  now();
}
async function now() {
  try {
    const s = await (await fetch('/now')).json();
    document.getElementById('now').innerHTML = s.name ? `Running: <span class=live>${s.name}</span> &middot; <a href="${s.url}" target=_blank rel=noopener>${s.url}</a>` : 'No app running';
  } catch (e) {}
}
now(); setInterval(now, 3000);
</script></body></html>"""


def run_dir():
    return Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / f'aerothon-{os.environ.get("USER", "")}'


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, body, ctype='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == '/':
            return self.send(PAGE, 'text/html')
        if u.path == '/now':
            d = run_dir()
            name = (d / 'app.name').read_text().strip() if (d / 'app.name').exists() else ''
            url = (d / 'app.url').read_text().strip() if (d / 'app.url').exists() else ''
            pid = (d / 'app.pid').read_text().strip() if (d / 'app.pid').exists() else ''
            alive = name.startswith('sim-') or (pid.isdigit() and Path(f'/proc/{pid}').exists())
            return self.send(json.dumps({'name': name if alive else '', 'url': url if alive else ''}))
        if u.path == '/run':
            mode, m = q.get('mode', ''), q.get('marker') or 'aruco'
            if mode == 'onboard':
                cmd = [LAUNCHER, 'onboard', m, q.get('radio') or '/dev/ttyUSB0:57600']
            elif mode == 'offboard':
                cmd = [LAUNCHER, 'offboard', m, q.get('drop') or 'drop', q.get('radio') or '/dev/ttyUSB0:57600',
                       q.get('video', ''), q.get('target', '')]
            elif mode == 'fieldtest':
                cmd = [LAUNCHER, 'fieldtest', 'radio' if q.get('marker') == 'radio' else 'cable'] + \
                      ([q.get('radio') or '/dev/ttyUSB0:57600'] if q.get('marker') == 'radio' else [])
            elif mode == 'sim':
                cmd = [LAUNCHER, 'sim', m]
            elif mode in ('stop', 'status'):
                cmd = [LAUNCHER, mode]
            else:
                return self.send(json.dumps({'out': f'unknown mode {html.escape(mode)}'}))
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                               env=dict(os.environ, NO_BROWSER='1'))
            out = (r.stdout + r.stderr).strip() or f'exit {r.returncode}'
            return self.send(json.dumps({'out': out}))
        self.send_error(404)


def main():
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'AeroTHON switcher: http://127.0.0.1:{PORT}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
