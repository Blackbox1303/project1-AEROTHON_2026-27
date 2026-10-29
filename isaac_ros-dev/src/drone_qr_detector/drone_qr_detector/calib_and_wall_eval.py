#!/usr/bin/env python3
"""Camera Calibration (with Outlier Rejection) & 3m Wall Multi-QR Distance Evaluator (`calib_and_wall_eval`).

Supports three modes without interrupting another user's CSI session (pulls from http://127.0.0.1:8090/frame.jpg when
`arducam_calib.py` is running, or opens CSI directly if free):

1. `capture_board`:
   Auto-captures still 9x6 (10x7 squares) checkerboard frames from shared/direct CSI.
2. `calibrate`:
   Recomputes equidistant fisheye calibration from saved checkerboard captures (`cap_*.png`) with ITERATIVE PER-VIEW
   RMS OUTLIER REJECTION.
3. `wall_eval`:
   Captures and aligns/averages frames to suppress sensor grain, detects all QR codes on the wall at `--wall-distance`
   (default 3.0 m) using multi-scale QrNet (144/192/256 tiles) + gnomonic rectification + WeChat/zbar super-resolution,
   computes each QR code's 3D location in the camera and drone (`base_link`) frames, and prints all pairwise 3D
   distances between the QR codes on the wall.
"""
import argparse
import http.client
import math
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import yaml

from drone_qr_detector.fisheye_model import FisheyeModel, default_base_from_optical, rpy_to_rot
from drone_qr_detector import qr_tools


def fetch_frame(host='127.0.0.1', port=8090, sensor_id=0, width=3280, height=2464, exposure_ms=8.0):
    """Fetch a full-resolution frame non-intrusively from port 8090 if active, else from direct CSI."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2.0)
        conn.request('GET', '/frame.jpg')
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status == 200 and len(body) > 1000:
            img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img, f'http://{host}:{port}/frame.jpg (shared CSI)'
    except Exception:
        pass

    os.environ.pop('DISPLAY', None)
    exp_ns = int(exposure_ms * 1e6)
    pipe = (
        f'nvarguscamerasrc sensor-id={sensor_id} exposuretimerange="13000 {exp_ns}" gainrange="1 10.625" ! '
        f'video/x-raw(memory:NVMM),width={width},height={height},framerate=21/1,format=NV12 ! '
        f'nvvidconv ! video/x-raw,format=GRAY8 ! appsink drop=true max-buffers=1 sync=false'
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None, 'Failed to open camera (neither shared HTTP 8090 nor direct CSI available)'
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, 'Direct CSI opened but returned no frame'
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, 'direct_csi_nvargus'


def parse_board_inner_corners(board_str, gray_sample=None):
    """Accept either inner corners ('9x6') or square counts ('10x7' -> 9x6 inner corners)."""
    c, r = (int(v) for v in board_str.lower().split('x'))
    if (c, r) == (10, 7):
        if gray_sample is not None:
            small = cv2.resize(gray_sample, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ok96, _ = cv2.findChessboardCornersSB(small, (9, 6), cv2.CALIB_CB_EXHAUSTIVE)
            if ok96:
                return 9, 6
            ok107, _ = cv2.findChessboardCornersSB(small, (10, 7), cv2.CALIB_CB_EXHAUSTIVE)
            if ok107:
                return 10, 7
        return 9, 6
    return c, r


def calibrate_with_outlier_rejection(frames_dir, out_yaml, board='9x6', square=0.025, max_view_rms=1.80):
    files = sorted(Path(frames_dir).glob('cap_*.png'))
    if not files:
        return False, f'No cap_*.png files found in {frames_dir}'
    first_img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    cols, rows = parse_board_inner_corners(board, first_img)
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)

    captures = []
    size = None
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, c = cv2.findChessboardCornersSB(small, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            continue
        c = (c * 2.0).astype(np.float32)
        cv2.cornerSubPix(img, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        captures.append((f.name, c.reshape(1, -1, 2).astype(np.float64)))

    if len(captures) < 6:
        return False, f'Only {len(captures)} valid checkerboard captures found in {frames_dir}; need at least 8.'

    w, h = size
    keep = list(range(len(captures)))
    dropped_cond = []
    dropped_outlier = []
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)

    while len(keep) >= 6:
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                [objp] * len(keep),
                [captures[i][1] for i in keep],
                (w, h), K, D, flags=flags, criteria=crit,
            )
        except cv2.error as e:
            m = re.search(r'input array (\d+)', str(e))
            if not m:
                return False, f'cv2.fisheye.calibrate error: {e}'
            bad_idx = int(m.group(1))
            dropped_cond.append(captures[keep.pop(bad_idx)][0])
            continue

        view_errors = []
        for idx_in_keep, cap_idx in enumerate(keep):
            proj, _ = cv2.fisheye.projectPoints(objp, rvecs[idx_in_keep], tvecs[idx_in_keep], K, D)
            err = np.linalg.norm(proj.reshape(-1, 2) - captures[cap_idx][1].reshape(-1, 2), axis=1)
            view_rms = float(np.sqrt(np.mean(err ** 2)))
            view_errors.append((view_rms, idx_in_keep, cap_idx))

        worst_rms, worst_pos, worst_cap_idx = max(view_errors, key=lambda t: t[0])
        if worst_rms > max_view_rms and len(keep) > 6:
            dropped_outlier.append((captures[worst_cap_idx][0], round(worst_rms, 3)))
            keep.pop(worst_pos)
            continue
        break
    else:
        return False, 'Too many ill-conditioned or blurry views; please capture more still views.'

    data = {
        'image_width': w,
        'image_height': h,
        'camera_name': 'arducam_imx219_fisheye',
        'distortion_model': 'equidistant',
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in K.ravel()]},
        'distortion_coefficients': {'rows': 1, 'cols': 4, 'data': [float(v) for v in D.ravel()]},
        'rms_px': round(float(rms), 4),
        'rms_1640_equiv_px': round(float(rms) * 1640.0 / w, 4),
        'views': len(keep),
        'dropped_ill_conditioned': dropped_cond,
        'dropped_high_rms_outliers': dropped_outlier,
        'board': f'{cols}x{rows}',
        'date': time.strftime('%Y-%m-%d %H:%M'),
    }
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(out_yaml).write_text(yaml.safe_dump(data, sort_keys=False))
    msg = (
        f'Saved cleaned calibration to {out_yaml}\n'
        f'  Resolution : {w}x{h}\n'
        f'  Overall RMS: {rms:.4f} px at {w}x{h} (equiv to {rms * 1640.0 / w:.4f} px at 1640x1232) across {len(keep)} kept views\n'
        f'  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}\n'
        f'  D={[round(float(x), 5) for x in D.ravel()]}\n'
        f'  Dropped ill-conditioned: {dropped_cond}\n'
        f'  Dropped high-RMS outliers: {dropped_outlier}'
    )
    return True, msg


def try_decode_crop(gray_img, model, poly_px, wc, zbar, cv_qr):
    """Try decoding a detected QR bounding box using gnomonic ray rectification + multi-scale unsharp masking."""
    h, w = gray_img.shape[:2]
    m = model.scaled(w, h)
    ctr = poly_px.mean(axis=0)
    pad = max(35.0, 0.75 * float(max(poly_px.ptp(axis=0))))
    x0, y0 = int(max(0, ctr[0] - pad)), int(max(0, ctr[1] - pad))
    x1, y1 = int(min(w, ctr[0] + pad)), int(min(h, ctr[1] + pad))
    crop = gray_img[y0:y1, x0:x1]

    # 1. Try direct multi-scale + unsharp mask on the aligned crop
    for sc in (1.0, 2.0, 3.0):
        up = crop if sc == 1.0 else cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        for sharp in (0.0, 0.6, 1.2):
            img_s = cv2.addWeighted(up, 1.0 + sharp, cv2.GaussianBlur(up, (0, 0), 2.0), -sharp, 0) if sharp > 0 else up
            hits = qr_tools.decode_gray(img_s, wc=wc, zbar=zbar, cv_qr=cv_qr, upscale_small=False)
            if hits:
                pl, p_hit = next(iter(hits.items()))
                return pl, (p_hit / sc) + [x0, y0]

    # 2. Try gnomonic patch rectification along the box's 3D ray
    ray_d = m.rays(ctr.reshape(1, 2))[0]
    for zoom in (2.0, 3.0):
        patch, meta = m.rectify_patch(gray_img, ray_d=ray_d, fov_deg=18.0, zoom_scale=zoom, max_size=384)
        for sharp in (0.0, 0.7):
            p_s = cv2.addWeighted(patch, 1.0 + sharp, cv2.GaussianBlur(patch, (0, 0), 2.0), -sharp, 0) if sharp > 0 else patch
            hits = qr_tools.decode_gray(p_s, wc=wc, zbar=zbar, cv_qr=cv_qr, upscale_small=False)
            if hits:
                pl, p_hit = next(iter(hits.items()))
                return pl, m.patch_px_to_fisheye_px(p_hit, meta)

    return None, poly_px


def run_wall_eval(args):
    model = FisheyeModel.from_yaml(args.calib, focal_scale=args.focal_scale)
    wc = qr_tools.make_wechat(args.wechat_dir)
    zbar = qr_tools.make_zbar()
    cv_qr = cv2.QRCodeDetector()
    qnet = qr_tools.QrNet(args.wechat_dir)
    T_base_opt = default_base_from_optical()

    print(f'Capturing & aligning {args.frames} frame(s) at wall distance = {args.wall_distance:.2f} m ...')
    frames = []
    src_name = ''
    for i in range(args.frames):
        gray, src = fetch_frame(host=args.host, port=args.port, exposure_ms=args.exposure_ms)
        if gray is not None:
            frames.append(gray)
            src_name = src
        time.sleep(0.06)

    if not frames:
        print('Could not capture any frames from camera.')
        return

    avg, n_used = qr_tools.aligned_mean(frames) if len(frames) >= 2 else (frames[0], 1)
    h, w = avg.shape[:2]
    m = model.scaled(w, h)
    fmask = m.circle_mask(w, h, max_theta_deg=80.0)
    print(f'Aligned & averaged {n_used}/{len(frames)} frames from {src_name} ({w}x{h}).')

    # Multi-scale QrNet scan + contrast candidates inside the active fisheye circle
    cands = qr_tools.find_qr_candidates(avg, fisheye_mask=fmask, min_transitions=2.5, max_cands=25)
    centres = [c['center'] for c in cands]

    raw_boxes = []
    if qnet.net is not None:
        # 1. Fine-tile scan around contrast candidates (catches 40-75 px codes with 0.95-1.00 confidence)
        for tile in (144, 192, 256):
            if centres:
                for conf, poly in qnet.scan(avg, conf=0.35, centres=centres, tile=tile, overlap=tile // 3):
                    wb, hb = poly.ptp(axis=0)
                    if 22 <= max(wb, hb) <= 160 and 0.55 <= wb / max(hb, 1.0) <= 1.8:
                        raw_boxes.append((conf, poly))
        # 2. Central wall region grid scan at 128, 160, and 224 px tiles so even small/oblique QRs are found
        wx0, wy0, wx1, wy1 = int(w * 0.25), int(h * 0.20), int(w * 0.78), int(h * 0.75)
        wall_roi = avg[wy0:wy1, wx0:wx1]
        for tile, c_thr in ((128, 0.30), (160, 0.40), (224, 0.45)):
            for conf, poly in qnet.scan(wall_roi, conf=c_thr, tile=tile, overlap=tile // 3):
                wb, hb = poly.ptp(axis=0)
                if 24 <= max(wb, hb) <= 135 and 0.6 <= wb / max(hb, 1.0) <= 1.65:
                    x0b, y0b = poly.min(axis=0).astype(int)
                    x1b, y1b = poly.max(axis=0).astype(int)
                    tr_r, tr_c = qr_tools.transitions(wall_roi[max(0, y0b):y1b, max(0, x0b):x1b])
                    if min(tr_r, tr_c) >= 3.2:
                        raw_boxes.append((conf, poly + [wx0, wy0]))

    # Also include strong QR contrast candidates (e.g. oblique far-right wall QR codes) not already covered by QrNet
    for c in cands:
        if c['score'] >= 0.60:
            poly = c['poly']
            wb, hb = poly.ptp(axis=0)
            if 45 <= max(wb, hb) <= 125 and 0.65 <= wb / max(hb, 1.0) <= 1.55:
                raw_boxes.append((c['score'], poly))

    keep_boxes = []
    for conf, poly in sorted(raw_boxes, key=lambda t: -t[0]):
        ctr = poly.mean(axis=0)
        if fmask[int(min(h - 1, max(0, ctr[1]))), int(min(w - 1, max(0, ctr[0])))] == 0:
            continue
        if all(qr_tools.iou(poly, q) < 0.25 and math.dist(ctr, q.mean(axis=0)) > 35.0 for _, q in keep_boxes):
            keep_boxes.append((conf, poly))

    # Sort left-to-right by pixel X coordinate for intuitive wall layout comparison
    keep_boxes.sort(key=lambda t: float(t[1].mean(axis=0)[0]))

    results = []
    for idx, (conf, poly) in enumerate(keep_boxes):
        payload, refined_poly = try_decode_crop(avg, m, poly, wc, zbar, cv_qr)
        loc = m.localize_qr_3d(
            poly_fisheye_px=refined_poly,
            image_size=(w, h),
            qr_size_m=args.qr_size,
            plane_distance_m=args.wall_distance,
            T_base_from_opt=T_base_opt,
            modules=args.qr_modules,
        )
        label = f'QR_{idx+1}' + (f' ({payload})' if payload else f' [boxed conf={conf:.2f}]')
        results.append({'id': f'QR_{idx+1}', 'label': label, 'payload': payload, 'conf': conf, 'loc': loc})

    if not results:
        print('\nNo QR codes detected in view.')
        return

    print(f'\n=== Detected {len(results)} QR Code(s) on the Wall (Sorted Left -> Right in Image) ===')
    r_deg, p_deg, y_deg = args.cam_rpy_deg
    R_wall_cam = rpy_to_rot(math.radians(r_deg), math.radians(p_deg), math.radians(y_deg))
    for r in results:
        loc = r['loc']
        pd = loc['pos_drone']
        pc = loc['pos_camera']
        # Ray in camera body FLU (+x out of lens, +y left, +z up), rotated by camera tilt R_wall_cam
        ray_body = np.array([pc[2], -pc[0], -pc[1]], dtype=np.float64)
        ray_wall = R_wall_cam @ ray_body
        s_w = float(args.wall_distance) / max(ray_wall[0], 1e-4)
        y_left_wall = float(ray_wall[1] * s_w)
        z_up_wall = float(ray_wall[2] * s_w)
        r['wall_yz'] = (y_left_wall, z_up_wall)
        print(
            f'  * {r["label"]}:\n'
            f'      Image Center / Size     : center=({loc["center_px"][0]:.1f}, {loc["center_px"][1]:.1f}) px | '
            f'size={loc["side_px"]:.1f} px (~{loc["px_per_module"]:.1f} px/module, {loc["off_axis_deg"]:.1f} deg off-axis)\n'
            f'      Wall WRT Lens (levelled): y_left = {y_left_wall:+.4f} m ({y_left_wall*100:+.1f} cm), '
            f'z_up_above_lens = {z_up_wall:+.4f} m ({z_up_wall*100:+.1f} cm)\n'
            f'      Drone Frame (base_link) : x={pd[0]:+.4f} m, y={pd[1]:+.4f} m, z={pd[2]:+.4f} m\n'
            f'      Camera Optical Frame    : x={pc[0]:+.4f} m, y={pc[1]:+.4f} m, z={pc[2]:+.4f} m'
        )

    if len(results) >= 2:
        print('\n=== Pairwise 3D Distances Between Detected QR Codes on the Wall ===')
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                r1, r2 = results[i], results[j]
                p1 = np.array(r1['loc']['pos_camera'])
                p2 = np.array(r2['loc']['pos_camera'])
                dist_3d = float(np.linalg.norm(p2 - p1))
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                line = f'  * {r1["id"]} <-> {r2["id"]}: 3D_dist = {dist_3d:.4f} m  (horiz dx = {dx:+.4f} m, vert dy = {dy:+.4f} m)'
                if args.known_dist_m > 0:
                    err = dist_3d - args.known_dist_m
                    line += f' | diff vs {args.known_dist_m:.3f}m = {err:+.4f} m'
                print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')

    p_cap = sub.add_parser('capture_board', help='Auto-capture still 9x6 (10x7 squares) checkerboard frames')
    p_cap.add_argument('--frames-dir', default='/workspaces/isaac_ros-dev/calibration/arducam_calib_frames')
    p_cap.add_argument('--board', default='10x7')
    p_cap.add_argument('--target-captures', type=int, default=25)
    p_cap.add_argument('--still-px', type=float, default=1.5)
    p_cap.add_argument('--host', default='127.0.0.1')
    p_cap.add_argument('--port', type=int, default=8090)
    p_cap.add_argument('--exposure-ms', type=float, default=8.0)

    p_cal = sub.add_parser('calibrate', help='Recompute fisheye calibration with per-view RMS outlier rejection')
    p_cal.add_argument('--frames-dir', default='/workspaces/isaac_ros-dev/calibration/arducam_calib_frames')
    p_cal.add_argument('--out', default='/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml')
    p_cal.add_argument('--board', default='10x7')
    p_cal.add_argument('--square', type=float, default=0.025)
    p_cal.add_argument('--max-view-rms', type=float, default=1.80)

    p_wall = sub.add_parser('wall_eval', help='Detect wall QR codes at 3m, output drone-frame poses & inter-QR distances')
    p_wall.add_argument('--calib', default='/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml')
    p_wall.add_argument('--wall-distance', type=float, default=3.0, help='Perpendicular distance to wall (m)')
    p_wall.add_argument('--qr-size', type=float, default=0.30, help='Physical side length of wall QR codes (m)')
    p_wall.add_argument('--qr-modules', type=int, default=29, help='Modules per side of QR code (21=v1, 25=v2, 29=v3)')
    p_wall.add_argument('--known-dist-m', type=float, default=0.0, help='Known distance (m) between QR pair on wall')
    p_wall.add_argument('--focal-scale', type=float, default=1.046)
    p_wall.add_argument('--cam-rpy-deg', nargs=3, type=float, default=[2.55, -7.61, -2.25],
                        help='Camera roll, pitch, yaw (deg) relative to wall horizontal/normal')
    p_wall.add_argument('--frames', type=int, default=8)
    p_wall.add_argument('--host', default='127.0.0.1')
    p_wall.add_argument('--port', type=int, default=8090)
    p_wall.add_argument('--exposure-ms', type=float, default=8.0)
    p_wall.add_argument('--wechat-dir', default='')

    args = ap.parse_args()
    if args.cmd == 'capture_board':
        cols, rows = parse_board_inner_corners(args.board)
        fdir = Path(args.frames_dir)
        fdir.mkdir(parents=True, exist_ok=True)
        existing = sorted(fdir.glob('cap_*.png'))
        print(f'Starting with {len(existing)} existing captures in {fdir} (target: {args.target_captures})')
        prev_c = None
        descs = []
        while len(list(fdir.glob('cap_*.png'))) < args.target_captures:
            gray, src = fetch_frame(host=args.host, port=args.port, exposure_ms=args.exposure_ms)
            if gray is None:
                print(src)
                time.sleep(1.0)
                continue
            h, w = gray.shape[:2]
            small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ok, c = cv2.findChessboardCornersSB(small, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if not ok:
                prev_c = None
                print('No board detected in view... (hold board fully visible)', end='\r', flush=True)
                time.sleep(0.2)
                continue
            c = (c * 2.0).astype(np.float32)
            cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            moved = float(np.abs(c - prev_c).max() * 1640.0 / w) if prev_c is not None else 999.0
            prev_c = c
            p = c.reshape(-1, 2)
            desc = np.array([p[:, 0].mean() / w, p[:, 1].mean() / h, (p[:, 0].ptp() + p[:, 1].ptp()) / (w + h)])
            is_new = all(np.abs(desc - d).max() > 0.06 for d in descs)
            if moved < args.still_px and is_new:
                idx = len(list(fdir.glob('cap_*.png'))) + 1
                out_path = fdir / f'cap_{idx:03d}.png'
                cv2.imwrite(str(out_path), gray)
                descs.append(desc)
                print(f'\n[CAPTURED {idx}/{args.target_captures}] {out_path.name} (centre=({desc[0]:.2f}, {desc[1]:.2f}), motion={moved:.2f}px)')
            else:
                state = 'HOLD STILL' if moved >= args.still_px else 'MOVE TO A NEW POSITION/ANGLE'
                print(f'Board found ({state}, motion={moved:.1f}px)                    ', end='\r', flush=True)
            time.sleep(0.2)
        print('\nTarget captures reached! Now run `calibrate`.')
    elif args.cmd == 'calibrate':
        ok, msg = calibrate_with_outlier_rejection(
            args.frames_dir, args.out, args.board, args.square, args.max_view_rms
        )
        print(msg)
    else:
        if not hasattr(args, 'wall_distance'):
            args = ap.parse_args(['wall_eval'])
        run_wall_eval(args)


if __name__ == '__main__':
    main()
