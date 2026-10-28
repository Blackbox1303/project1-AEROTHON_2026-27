#!/usr/bin/env python3
"""Standalone ArUco marker test page (http://10.42.0.247:8085) for the Arducam (the ArUco delivery mission is 8084).

Opens the Arducam on CSI CAM0 itself (GStreamer nvarguscamerasrc, so it runs in the dev container:
scripts/aruco_test.sh) - only one process can hold the camera, so the script stops the calibration page (8090)
and the ROS camera first. --source http://.../frame.jpg, a /dev/video index or a file are still accepted.

Detection runs on the UNDISTORTED image: gnomonic views of the fisheye (one straight down at --centre-fov, --ring
views tilted --ring-tilt round it), the same aerothon_nav.aruco_tools the mission nodes use. 3x3 .. 6x6:
4x4 / 5x5 / 6x6 from the OpenCV dictionaries, 3x3 read bit by bit ('3X3#<code>'). With the marker's black
square side typed in (marker m) each marker gets distance / position from the undistorted view.
Detections are appended to runs/aruco_test/detections.csv.
"""
import argparse, csv, json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, os.path.join(WS, 'src/aerothon_nav'))
from aerothon_nav import aruco_tools                      # noqa: E402
from aerothon_nav.fisheye import Fisheye                  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>ArUco test 8085</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:8px}img{max-width:100%;border:1px solid #444}
table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #444;padding:3px 8px}
button,input,select{font-size:14px}#st{white-space:pre-wrap;font-family:monospace}</style>
</head><body>
<h3>ArUco markers 3x3 - 6x6 on the undistorted Arducam image</h3>
<div>view <select id=vm onchange="q('view?m='+this.value)"><option value=mosaic>undistorted views<option value=centre>centre view
<option value=fisheye>raw fisheye</select>
 &nbsp; marker m <input id=sz size=5> <button onclick=setsz()>set</button>
 &nbsp; exposure ms <input id=ex size=4> <button onclick="q('exposure?ms='+document.getElementById('ex').value)">set</button></div>
<div id=st></div>
<img id=v src=view.jpg>
<table id=t></table>
<script>
function q(p){return fetch(p).then(r=>r.json())}
function setsz(){q('size?m='+document.getElementById('sz').value)}
let first=1;
function poll(){q('status').then(s=>{
 document.getElementById('st').textContent=s.text;
 if(first){document.getElementById('sz').value=s.size;document.getElementById('ex').value=s.exposure;
   document.getElementById('vm').value=s.view;first=0}
 let h='<tr><th>#</th><th>marker</th><th>fisheye px</th><th>side px (undist.)</th><th>off-axis deg</th><th>dist m</th><th>x,y,z m (camera)</th><th>seen</th></tr>';
 s.markers.forEach((m,i)=>h+=`<tr><td>${i+1}</td><td><b>${m.id}</b></td><td>${m.c[0]},${m.c[1]}</td><td>${m.side}</td><td>${m.off}</td><td>${m.dist??'-'}</td><td>${m.xyz??'-'}</td><td>${m.seen}</td></tr>`);
 document.getElementById('t').innerHTML=h;}).catch(()=>{}).finally(()=>setTimeout(poll,500))}
function img(){let v=document.getElementById('v');let n=new Image();n.onload=()=>{v.src=n.src;setTimeout(img,100)};
 n.onerror=()=>setTimeout(img,1000);n.src='view.jpg?'+Date.now()}
poll();img();
</script></body></html>"""


def load_fisheye(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    info = SimpleNamespace(distortion_model=c['distortion_model'], k=c['camera_matrix']['data'],
                           d=c['distortion_coefficients']['data'], width=c['image_width'],
                           height=c['image_height'], header=SimpleNamespace(frame_id='arducam'))
    return Fisheye(info)


class Camera:
    """CSI CAM0 through Argus (or another --source); reopens with a new exposure cap."""

    def __init__(self, a):
        self.a, self.exp_ms, self.cap, self.reopen = a, a.exposure_ms, None, False

    def pipeline(self):
        a = self.a
        fps = min(a.fps, 21 if a.height > 2000 else 30)       # the full-resolution mode tops out at 21 fps
        return (f'nvarguscamerasrc sensor-id={a.sensor_id} exposuretimerange="13000 {int(self.exp_ms * 1e6)}" '
                f'gainrange="1 10.625" ! video/x-raw(memory:NVMM),width={a.width},height={a.height},'
                f'framerate={fps}/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false')

    def read(self):
        src = self.a.source
        if src.startswith('http'):
            data = urllib.request.urlopen(src, timeout=5).read()
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if self.reopen and self.cap is not None:
            self.cap.release()
            self.cap, self.reopen = None, False
            time.sleep(1.5)                  # quick Argus reopens wedge nvargus-daemon
        if self.cap is None:
            if src == 'csi':
                os.environ.pop('DISPLAY', None)      # Argus' EGL stream fails with a DISPLAY it can't use
                self.cap = cv2.VideoCapture(self.pipeline(), cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError('camera did not open (another process holding CSI CAM0? nvargus-daemon wedged?)')
        ok, f = self.cap.read()
        if not ok and src != 'csi' and not src.isdigit():   # a video / image file: start again
            self.cap.release()
            self.cap = None
        return f if ok else None


class App:
    def __init__(self, a):
        self.a = a
        self.cam = load_fisheye(a.calib)
        self.det = aruco_tools.Detector()
        self.camera = Camera(a)
        self.size_m = a.marker_size
        self.view_mode = 'mosaic'
        self.lock = threading.Lock()
        self.view, self.markers, self.seen = None, [], {}
        self.text, self.fps = 'starting', 0.0
        os.makedirs(a.log_dir, exist_ok=True)
        self.csv_path = os.path.join(a.log_dir, 'detections.csv')
        new = not os.path.exists(self.csv_path)
        self.csv = open(self.csv_path, 'a', newline='')
        self.w = csv.writer(self.csv)
        if new:
            self.w.writerow(['t', 'id', 'cx', 'cy', 'side_px', 'off_deg', 'dist_m', 'x', 'y', 'z'])

    def pose(self, q, meta):
        """Marker corners in an undistorted view -> (distance m, xyz in the camera frame) or None."""
        if self.size_m <= 0:
            return None
        h = self.size_m / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        K = np.array([[meta['f_p'], 0, meta['c_p']], [0, meta['f_p'], meta['c_p']], [0, 0, 1]])
        ok, _, tv = cv2.solvePnP(obj, q.reshape(-1, 1, 2), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        t = meta['R'] @ tv.ravel()                       # view frame -> camera optical frame
        return float(np.linalg.norm(t)), t

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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = time.time()
            views = []
            found = aruco_tools.detect_fisheye(self.det, gray, self.cam, centre_fov=self.a.centre_fov,
                                               ring_fov=self.a.ring_fov, ring_tilt=self.a.ring_tilt,
                                               ring=self.a.ring, max_size=self.a.view_px, views_out=views)
            dt = time.time() - t
            now = time.time()
            ms, done = [], set()
            for view, meta, hits in views:
                for pid, q in hits:
                    if pid in done or pid not in found:
                        continue
                    done.add(pid)
                    c = found[pid].mean(axis=0)
                    ray = self.cam.rays(c[None], scale=gray.shape[1] / self.cam.size[0])[0]
                    off = float(np.degrees(np.arccos(np.clip(ray[2], -1, 1))))
                    side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
                    p = self.pose(q, meta)
                    self.seen[pid] = self.seen.get(pid, 0) + 1
                    m = {'id': pid, 'c': [int(c[0]), int(c[1])], 'side': round(side, 1), 'off': round(off, 1),
                         'dist': round(p[0], 2) if p else None,
                         'xyz': ','.join(f'{v:.2f}' for v in p[1]) if p else None, 'seen': self.seen[pid]}
                    ms.append(m)
                    self.w.writerow([f'{now:.2f}', pid, *m['c'], m['side'], m['off'], m['dist'] or '',
                                     *(p[1].round(3) if p else ['', '', ''])])
            self.csv.flush()
            jpg = self.render(img, views, found)
            n += 1
            if now - t0 > 2:
                self.fps, n, t0 = n / (now - t0), 0, now
            with self.lock:
                self.view, self.markers = jpg, ms
                self.text = (f'source {self.a.source}  {img.shape[1]}x{img.shape[0]}  {self.fps:.1f} fps  '
                             f'detect {dt * 1000:.0f} ms ({len(views)} undistorted views)   exposure <= '
                             f'{self.camera.exp_ms} ms   marker size '
                             f'{self.size_m if self.size_m > 0 else "unset (no distance)"} m\n'
                             f'{len(ms)} marker(s) now: ' + (', '.join(m['id'] for m in ms) or '-') +
                             f'\nlog {self.csv_path}')

    def render(self, img, views, found):
        def draw(im, hits, th):
            for pid, q in hits:
                cv2.polylines(im, [q.astype(np.int32)], True, (0, 255, 0), th)
                cv2.circle(im, tuple(q[0].astype(int)), th * 3, (0, 0, 255), -1)       # marker's first corner
                cv2.putText(im, pid, (int(q[:, 0].min()), int(q[:, 1].min()) - 4 * th), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * th, (0, 255, 0), th)
        if self.view_mode == 'fisheye':
            out = img.copy()
            draw(out, list(found.items()), max(2, img.shape[1] // 800))
        else:
            tiles = []
            for view, meta, hits in (views[:1] if self.view_mode == 'centre' else views):
                v = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                draw(v, hits, 2)
                tiles.append(v)
            out = tiles[0]
            if len(tiles) > 1:                            # centre view big, the tilted views in 2 rows beside it
                hs = out.shape[0] // 2
                small = [cv2.resize(t, (hs, hs)) for t in tiles[1:]]
                cols = (len(small) + 1) // 2
                small += [np.zeros((hs, hs, 3), np.uint8)] * (2 * cols - len(small))
                side = np.vstack([np.hstack(small[:cols]), np.hstack(small[cols:])])
                side = cv2.resize(side, (side.shape[1] * out.shape[0] // side.shape[0], out.shape[0]))
                out = np.hstack([out, side])
        s = 1600 / out.shape[1]
        if s < 1:
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8085)
    ap.add_argument('--source', default='csi', help="csi | http://.../frame.jpg | /dev/video index | file")
    ap.add_argument('--sensor-id', type=int, default=0)
    ap.add_argument('--width', type=int, default=1640)
    ap.add_argument('--height', type=int, default=1232)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--exposure-ms', type=float, default=20.0)
    ap.add_argument('--calib', default=os.path.join(WS, 'calibration/arducam_fisheye_1640x1232.yaml'))
    ap.add_argument('--marker-size', type=float, default=0.0, help='black square side, m (0 = no distance)')
    ap.add_argument('--centre-fov', type=float, default=100.0, help='deg, the straight-down undistorted view')
    ap.add_argument('--ring-fov', type=float, default=90.0, help='deg, each tilted view (0 = centre only)')
    ap.add_argument('--ring-tilt', type=float, default=55.0, help='deg off axis of the tilted views')
    ap.add_argument('--ring', type=int, default=5)
    ap.add_argument('--view-px', type=int, default=1100, help='max side of an undistorted view')
    ap.add_argument('--log-dir', default=os.path.join(WS, 'runs/aruco_test'))
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
                if u.path == '/size':
                    app.size_m = max(0.0, float(qs.get('m', 0)))
                elif u.path == '/exposure':
                    app.camera.exp_ms = min(200.0, max(0.1, float(qs.get('ms', 20))))
                    app.camera.reopen = True
                elif u.path == '/view' and qs.get('m') in ('mosaic', 'centre', 'fisheye'):
                    app.view_mode = qs['m']
                elif u.path != '/status':
                    self.send_error(404)
                    return
            except ValueError:
                pass
            with app.lock:
                st = {'text': app.text, 'markers': app.markers, 'size': app.size_m, 'view': app.view_mode,
                      'exposure': app.camera.exp_ms}
            self.send(json.dumps(st).encode())

    srv = ThreadingHTTPServer(('0.0.0.0', a.port), H)
    srv.daemon_threads = True
    print(f'ArUco page: http://10.42.0.247:{a.port}  source {a.source}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    os._exit(0)   # don't wait on a blocked camera read (see dashboard exit hang); Argus is released by the exit


if __name__ == '__main__':
    main()
