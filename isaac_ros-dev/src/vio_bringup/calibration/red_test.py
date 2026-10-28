#!/usr/bin/env python3
"""Standalone red-zone test page (http://10.42.0.247:8086) for the Arducam: what the mission calls red.

Opens the Arducam on CSI CAM0 itself (the Camera of aruco_test.py; scripts/red_test.sh runs it in the dev container
and stops 8090 / 8085 / the ROS camera first - only one process can hold the camera).

Red is the delivery_mapper rule, pixel by pixel on the raw fisheye frame: OpenCV hue in [h1] or [h2], S >= s_min,
V >= v_min (defaults = the mapper's red_h1 / red_h2 / red_s_min / red_v_min), and only within red_angle of straight
down (the mapper votes only there; the circle on the view). Red pixels are painted magenta, the rest dimmed.
The status gives the red share inside the cone and the biggest red blob; click the image for that pixel's HSV
(to tune the thresholds on the real mat under the real light). Thresholds are changed live from the page.
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aruco_test import Camera, load_fisheye, WS          # noqa: E402

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Red zone test 8086</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:8px}img{max-width:100%;border:1px solid #444;cursor:crosshair}
input{width:52px;font-size:14px}button{font-size:14px}#st{white-space:pre-wrap;font-family:monospace}</style>
</head><body>
<h3>Red zones on the Arducam (delivery_mapper rule)</h3>
<div>hue <input id=a1>-<input id=b1> or <input id=a2>-<input id=b2> &nbsp; S &ge; <input id=s> &nbsp; V &ge; <input id=v>
&nbsp; cone <input id=ang> deg &nbsp; exposure <input id=exp> ms
<button onclick=apply()>apply</button> <button onclick="q('reset')">mission defaults</button>
<label><input type=checkbox id=raw onchange="q('view?raw='+(this.checked?1:0))" style=width:auto> raw view</label></div>
<div id=st>starting</div>
<img id=im onclick=pick(event)>
<script>
const F=['a1','b1','a2','b2','s','v','ang','exp'];let filled=false;
function q(p){fetch('/'+p).then(r=>r.json()).then(show)}
function apply(){q('set?'+F.map(k=>k+'='+document.getElementById(k).value).join('&'))}
function pick(e){const i=e.target,r=i.getBoundingClientRect();
  q('hsv?x='+((e.clientX-r.left)/r.width).toFixed(4)+'&y='+((e.clientY-r.top)/r.height).toFixed(4))}
function show(s){document.getElementById('st').textContent=s.text+(s.pick?'\\nclicked: '+s.pick:'');
  if(!filled){F.forEach(k=>document.getElementById(k).value=s.th[k]);filled=true}}
setInterval(()=>{document.getElementById('im').src='/view.jpg?'+Date.now();q('status')},300);
</script></body></html>"""

DEFAULTS = {'a1': 0, 'b1': 10, 'a2': 170, 'b2': 180, 's': 100, 'v': 70, 'ang': 60.0}   # delivery_mapper params


class App:
    def __init__(self, a):
        self.a = a
        self.cam = load_fisheye(a.calib)
        self.camera = Camera(a)
        self.th = dict(DEFAULTS)
        self.raw = False
        self.lock = threading.Lock()
        self.view, self.text, self.pick, self.frame = None, 'starting', '', None
        self.cone, self.cone_key, self.fps = None, None, 0.0

    def cone_mask(self, w, h):
        """Pixels within th['ang'] of the optical axis (straight down when the drone is level)."""
        key = (w, h, self.th['ang'])
        if self.cone_key != key:
            s = 4                                              # rays on a 1/4 grid, then scaled up
            us, vs = np.meshgrid(np.arange(0, w, s) + s / 2, np.arange(0, h, s) + s / 2)
            px = np.column_stack([us.ravel(), vs.ravel()])
            ray = self.cam.rays(px, scale=w / self.cam.size[0])
            m = (ray[:, 2] > np.cos(np.radians(self.th['ang']))).reshape(us.shape).astype(np.uint8)
            self.cone = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            self.cone_key = key
        return self.cone

    def loop(self):
        n, t0 = 0, time.time()
        while True:
            try:
                img = self.camera.read()
            except Exception as e:
                self.text = f'no frame from {self.a.source}: {e}'
                time.sleep(2)
                continue
            if img is None:
                self.text = f'no frame from {self.a.source}'
                time.sleep(0.5)
                continue
            h, w = img.shape[:2]
            th = dict(self.th)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            red = ((((H >= th['a1']) & (H <= th['b1'])) | ((H >= th['a2']) & (H <= th['b2'])))
                   & (S >= th['s']) & (V >= th['v']))
            cone = self.cone_mask(w, h)
            red_in = red & cone
            share = red_in.sum() / max(1, cone.sum())
            nb, _, stats, _ = cv2.connectedComponentsWithStats(red_in.astype(np.uint8), connectivity=8)
            big = stats[1:, cv2.CC_STAT_AREA].max() if nb > 1 else 0
            blobs = int((stats[1:, cv2.CC_STAT_AREA] >= 0.001 * w * h).sum()) if nb > 1 else 0
            if self.raw:
                out = img.copy()
            else:
                out = (img * 0.35).astype(np.uint8)
                out[red_in] = (255, 0, 255)
                out[red & ~cone] = (120, 0, 120)                  # red, but outside the cone: the mapper ignores it
            cs = cv2.findContours(cone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(out, cs, -1, (0, 255, 255), max(2, w // 800))
            sc = 1600 / w
            jpg = cv2.imencode('.jpg', cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1
                               else out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            n += 1
            now = time.time()
            if now - t0 > 2:
                self.fps, n, t0 = n / (now - t0), 0, now
            with self.lock:
                self.view, self.frame = jpg, hsv
                self.text = (f'source {self.a.source}  {w}x{h}  {self.fps:.1f} fps  exposure <= {self.camera.exp_ms} ms\n'
                             f'RED {100 * share:.1f} % of the {th["ang"]:.0f} deg cone   {blobs} blob(s) >= 0.1 % of '
                             f'the frame, biggest {100 * big / (w * h):.2f} % of the frame\n'
                             f'rule: hue {th["a1"]}-{th["b1"]} | {th["a2"]}-{th["b2"]}, S >= {th["s"]}, V >= {th["v"]}'
                             f'   (magenta = red in the cone, dark purple = red outside it, yellow = cone)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8086)
    ap.add_argument('--source', default='csi', help="csi | http://.../frame.jpg | /dev/video index | file")
    ap.add_argument('--sensor-id', type=int, default=0)
    ap.add_argument('--width', type=int, default=1640)
    ap.add_argument('--height', type=int, default=1232)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--exposure-ms', type=float, default=20.0)
    ap.add_argument('--calib', default=os.path.join(WS, 'calibration/arducam_fisheye_1640x1232.yaml'))
    a = ap.parse_args()
    app = App(a)
    threading.Thread(target=app.loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, ctype='application/json'):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == '/':
                return self.send(PAGE.encode(), 'text/html')
            if u.path == '/view.jpg':
                with app.lock:
                    v = app.view
                if v is None:
                    self.send_error(503)
                    return
                return self.send(v, 'image/jpeg')
            try:
                if u.path == '/set':
                    for k in DEFAULTS:
                        if k in qs:
                            app.th[k] = float(qs[k]) if k == 'ang' else int(float(qs[k]))
                    if 'exp' in qs and float(qs['exp']) != app.camera.exp_ms:
                        app.camera.exp_ms = min(200.0, max(0.1, float(qs['exp'])))
                        app.camera.reopen = True
                elif u.path == '/reset':
                    app.th = dict(DEFAULTS)
                elif u.path == '/view':
                    app.raw = qs.get('raw') == '1'
                elif u.path == '/hsv':
                    with app.lock:
                        f = app.frame
                    if f is not None:
                        x = int(np.clip(float(qs['x']), 0, 0.9999) * f.shape[1])
                        y = int(np.clip(float(qs['y']), 0, 0.9999) * f.shape[0])
                        p = f[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)   # 5x5 median
                        hh, ss, vv = np.median(p, axis=0).astype(int)
                        app.pick = f'pixel ({x}, {y})  H {hh}  S {ss}  V {vv}'
                elif u.path != '/status':
                    self.send_error(404)
                    return
            except (ValueError, KeyError):
                pass
            th = dict(app.th, exp=app.camera.exp_ms)
            with app.lock:
                st = {'text': app.text, 'pick': app.pick, 'th': th}
            self.send(json.dumps(st).encode())

    srv = ThreadingHTTPServer(('0.0.0.0', a.port), H)
    srv.daemon_threads = True
    print(f'Red zone page: http://10.42.0.247:{a.port}  source {a.source}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    os._exit(0)   # don't wait on a blocked camera read (see dashboard exit hang); Argus is released by the exit


if __name__ == '__main__':
    main()
