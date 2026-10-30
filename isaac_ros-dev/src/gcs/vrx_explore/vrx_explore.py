#!/usr/bin/env python3
"""VRX Explore: QR scanning (the 8090 / laptop_vrx decoder) + frontier exploration with A* round red zones, at 1 m.

The ground station side of the analog link: video from a USB video device (VRX receiver, or any webcam until the analog
camera exists), commands to PX4 over MAVLink (the SITL now, the telemetry radio later). Page: http://<host>:8094.
Built on laptop_vrx_mission.VrxDeliverySystem (decoder, VRX capture, ground projection, MAVLink offboard), adding:
  map       grid over the zone (+ home, + 2 m): explored = cells the downward camera's read area covered (from the
            pose, so coverage also works without a camera); red = HSV red pixels voting for their ground cell (two hue
            bands; red after min_red_votes with red share >= red_share) and red rectangles drawn on the page; the
            planner keeps red_margin off both
  goals     unread QR candidates (boards seen but not decoded) first, in BATCHES (user, 2026-10-28, as the
            sim's frontier_explorer): the boards first seen together are one batch, boards seen later (on the way, or
            from a reject's yaw scan) go into the next; a batch is finished before the next is started, inside it
            the cheapest A* route first, re-costed from where the drone is at each pick. A frontier / scout leg is
            broken off as soon as a candidate is pending. Then frontiers: per cluster of unexplored zone cells next
            to explored ones, score = cluster area / (path length + 1 m)
  yaw scan  a board read as a REJECT (at a candidate, or in passing within --spin-near): turn in place to 3 more
            headings 90 deg apart for boards out of view; not again within --spin-skip of an earlier scan
  scout     no board in view --scout-wait s into the search: out to the line --scout-offset (at most a quarter of
            the short side) in from the nearer long wall and along it to the far end; dropped once a board is seen
  retrace   while a frontier is left every explorer route costs --retrace-cost extra per cell already flown over
            (--trail-radius round the track); the way home / to the target is the plain shortest A*
  A*        8-connected on the grid, red cells grown by red_margin blocked, string-pulled to line-of-sight waypoints;
            re-planned when a red zone appears on the remaining path, the goal stops being worth it (frontier seen on
            the way), or the goal is unreachable (then blacklisted)
  mission   IDLE -> TAKEOFF (1 m) -> EXPLORE -> RETURN_HOME (A*) -> LAND -> DONE; return at time_budget (15 min rule)
Lens (--lens): the analog camera is a B19 (2.1 mm, rated ~130 deg diagonal - retailer spec, not measured): an
equidistant (fisheye) model fitted to that rating projects pixels to the ground (marker positions, red votes,
coverage) and ArUco markers are found on undistorted gnomonic views of the frame. --lens pinhole = the old
--fov-deg pinhole projection. A checkerboard calibration of the real unit should replace the rated FOV.
ArUco delivery (--marker aruco, default the AEROTHON_MARKER environment variable; user, 2026-10-30: the course has ArUco
markers only, MATCHED by image, never read): the start marker's image is kept at take-off (aerothon_nav.aruco_match);
every marker-like square in the video is compared with it - crisp and >= --aruco-hi the target, crisp and <=
--aruco-lo another marker ('<marker>'), else (blurry / between) a candidate the drone hovers over for the deep read:
every camera setting (virtual focal length / zoom, FOV, resolution: aruco_match.SETTINGS) x filter (gamma = exposure,
CLAHE, sharpen, denoise) per frame, the median of the best 3 frames; unsure / too blurry -> 0.5 m lower and again.
QR AREA (user, 2026-10-28, as the 8080 dashboard): GPS slots arena 4 + QR area 4 + extra 2 (/api/zone_gps), through
the autopilot's GPS origin; the QR area given, or the arena with its front wall = the line through the take-off point
parallel to the back wall (the drone is carried into the QR area). Routes never leave it (A* keep-in; home +-2 m).
DEEP READ (QR boards): a candidate still unread after the hover -> --verify-rounds frames of undistorted views (3 FOV)
x the filter settings of aerothon_nav.qr_match, then its image vs the start QR's (stored at the start read; a start
QR without data = qr_match.IMAGE_ONLY): match = the target, different = another board, unsure / not seen -> 0.5 m
lower (not below --verify-floor, 5.5 m: 5 m is reached only to lower the payload) and again; the search height comes
back afterwards.
Usage:  vrx_explore.py --video /dev/video2 --link udpout:127.0.0.1:14551 [--alt 1.0] [--port 8094]
        (scripts/vrx_explore.sh starts it with the PX4 SIH SITL)
"""
import argparse
import asyncio
import heapq
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'laptop_vrx'))
import laptop_vrx_mission as base  # noqa: E402

# hand over to the RC pilot (user, 2026-10-30): ArduCopter custom modes for MAV_CMD_DO_SET_MODE; the explorer stops
HANDOVER = {'loiter': 5, 'poshold': 16, 'alt_hold': 2, 'stabilize': 0, 'brake': 17}
MAX_ALT = 9.5                  # m: the airspace rule there - never above 10 m (user, 2026-10-30); margin for overshoot
sys.path.insert(0, str(HERE.parents[1] / 'aerothon_nav'))
from aerothon_nav import aruco_match, aruco_tools, qr_area, qr_match  # noqa: E402
from aerothon_nav.fisheye import Fisheye  # noqa: E402
LENS_DFOV = {'b19': 130.0}   # rated diagonal FOV, deg

FSM = ['IDLE', 'TAKEOFF', 'EXPLORE', 'RETURN_HOME', 'LAND', 'DONE']
# the QR area with the payload drop (8080 offboard segment qr_drop): the start QR first, the drop on the match
FSM_HOP = ['IDLE', 'TAKEOFF', 'HOVER', 'LAND', 'DONE']                # pre-mission take-off / land check
FSM_DROP = ['IDLE', 'TAKEOFF', 'READ_START_QR', 'EXPLORE', 'DELIVER', 'DROP', 'RETURN_HOME', 'LAND', 'DONE']
SKIP_VIDEO = ('realsense', 'vi-output', 'tegra')   # D435 nodes and the CSI camera: never opened here


class GzCapture:
    """A Gazebo camera topic as a cv2.VideoCapture (read / release / isOpened): the 8084 sim's B19 camera
    (`--video gz:/b19/image_raw`). gz-transport Python bindings (Harmonic); GZ_IP / GZ_PARTITION from the environment."""

    def __init__(self, topic):
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node
        self.cond, self.frame, self.n = threading.Condition(), None, 0
        self.node = Node()
        self.ok = self.node.subscribe(Image, topic, self.on_image)

    def on_image(self, m):
        a = np.frombuffer(m.data, np.uint8)
        if a.size == m.width * m.height * 3:
            f = cv2.cvtColor(a.reshape(m.height, m.width, 3), cv2.COLOR_RGB2BGR)
        elif a.size == m.width * m.height:
            f = a.reshape(m.height, m.width)
        else:
            return
        with self.cond:
            self.frame, self.n = f, self.n + 1
            self.cond.notify_all()

    def isOpened(self):
        return self.ok

    def read(self):
        with self.cond:
            n = self.n
            if not self.cond.wait_for(lambda: self.n != n, timeout=3.0):
                return False, None
            return True, self.frame

    def release(self):
        self.node = None


def video_devices():
    """[(path, name)] of capture devices, without the D435 / CSI ones."""
    out = []
    for d in sorted(Path('/sys/class/video4linux').glob('video*'), key=lambda p: int(p.name[5:] or 0)):
        try:
            name = (d / 'name').read_text().strip()
        except OSError:
            name = '?'
        if not any(s in name.lower() for s in SKIP_VIDEO):
            out.append((f'/dev/{d.name}', name))
    return out


class ExploreMap:
    """Grid over the zone: explored, red (camera votes + drawn rectangles), planning, frontiers."""

    def __init__(self, zone, home, res=0.2, red_margin=0.5, min_red_votes=3, red_share=0.5, start_r=2.5):
        self.res, self.red_margin, self.start_r = res, red_margin, start_r
        self.min_red_votes, self.red_share = min_red_votes, red_share
        self.set_zone(zone, home)

    def set_zone(self, zone, home, poly=None):
        """zone = x_min, x_max, y_min, y_max (ENU, m); poly: the QR area's corners (the zone = their bounding box)."""
        if poly:
            zone = qr_area.bbox(poly)
        self.poly = [tuple(map(float, p)) for p in poly] if poly else None
        self.zone = tuple(float(v) for v in zone)
        x0, x1, y0, y1 = self.zone
        hx, hy = home
        self.ox, self.oy = min(x0, hx) - 2.0, min(y0, hy) - 2.0
        self.w = int(math.ceil((max(x1, hx) + 2.0 - self.ox) / self.res))
        self.h = int(math.ceil((max(y1, hy) + 2.0 - self.oy) / self.res))
        self.explored = np.zeros((self.h, self.w), bool)
        self.seen = np.zeros((self.h, self.w), np.int32)
        self.red_votes = np.zeros((self.h, self.w), np.int32)
        self.drawn = np.zeros((self.h, self.w), bool)
        self.rects = []
        self.in_zone = np.zeros((self.h, self.w), bool)
        self.trail = np.zeros((self.h, self.w), bool)           # cells flown over (the retrace cost)
        (i0, j0), (i1, j1) = self.cell(x0, y0), self.cell(x1, y1)
        self.in_zone[max(j0, 0):min(j1, self.h - 1) + 1, max(i0, 0):min(i1, self.w - 1) + 1] = True
        if self.poly:
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m, [np.round([((px - self.ox) / self.res - 0.5, (py - self.oy) / self.res - 0.5)
                                       for px, py in self.poly]).astype(np.int32)], 1)
            self.in_zone &= m.astype(bool)
        # the keep-in: routes stay in the zone (user, 2026-10-28: no leaving the QR area), home +-2 m for the way in/out
        hm = np.zeros((self.h, self.w), np.uint8)
        cv2.circle(hm, self.cell(hx, hy), int(2.0 / self.res), 1, -1)
        self.keep = self.in_zone | hm.astype(bool)
        # the START ZONE is red (course 2026-10-30): the drone takes off from it and lands back on it - the camera's
        # red votes within start_r of the take-off point are not a red zone (a rectangle drawn there still is)
        sm = np.zeros((self.h, self.w), np.uint8)
        cv2.circle(sm, self.cell(hx, hy), int(round(self.start_r / self.res)), 1, -1)
        self.start = sm.astype(bool)
        self.ver = 0

    def cell(self, x, y):
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def xy(self, i, j):
        return self.ox + (i + 0.5) * self.res, self.oy + (j + 0.5) * self.res

    def inside(self, i, j):
        return 0 <= i < self.w and 0 <= j < self.h

    def mark_explored(self, ground_poly):
        pts = np.array([[(x - self.ox) / self.res, (y - self.oy) / self.res] for x, y in ground_poly], np.int32)
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        self.explored |= m.astype(bool)
        self.ver += 1

    def mark_trail(self, x, y, radius):
        i, j = self.cell(x, y)
        cv2.circle(self.trail.view(np.uint8), (i, j), max(1, int(radius / self.res)), 1, -1)

    def add_red_samples(self, xy, is_red):
        ij = np.floor((xy - [self.ox, self.oy]) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.w) & (ij[:, 1] >= 0) & (ij[:, 1] < self.h)
        ij, is_red = ij[ok], is_red[ok]
        np.add.at(self.seen, (ij[:, 1], ij[:, 0]), 1)
        np.add.at(self.red_votes, (ij[:, 1], ij[:, 0]), is_red.astype(np.int32))
        self.ver += 1

    def add_rect(self, x0, y0, x1, y1):
        (i0, j0), (i1, j1) = self.cell(min(x0, x1), min(y0, y1)), self.cell(max(x0, x1), max(y0, y1))
        self.drawn[max(j0, 0):min(j1, self.h - 1) + 1, max(i0, 0):min(i1, self.w - 1) + 1] = True
        self.rects.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        self.ver += 1

    def red(self):
        cam = (self.red_votes >= self.min_red_votes) & (self.red_votes >= self.red_share * np.maximum(self.seen, 1))
        return (cam & ~self.start) | self.drawn

    def blocked(self, margin=None):
        red = self.red().astype(np.uint8)
        r = max(1, int(math.ceil((self.red_margin if margin is None else margin) / self.res)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(red, k).astype(bool) | ~self.keep

    def coverage(self):
        n = int(self.in_zone.sum())
        free = self.in_zone & ~self.red()
        return float((self.explored & free).sum()) / max(1, int(free.sum())), n

    def frontiers(self, blocked):
        """[(x, y, area_m2)] per cluster of unexplored zone cells that touch explored ones."""
        unexp = (self.in_zone & ~self.explored & ~blocked).astype(np.uint8)
        known = cv2.dilate(self.explored.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        front = (unexp.astype(bool) & known).astype(np.uint8)
        if not front.any():                       # nothing explored next to the zone yet: head into it
            front = unexp
        n, lab, stats, cent = cv2.connectedComponentsWithStats(front, connectivity=8)
        out = []
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 3:
                continue
            js, iis = np.nonzero(lab == k)
            c = np.argmin((iis - cent[k][0]) ** 2 + (js - cent[k][1]) ** 2)      # the cluster cell nearest its centre
            # gain: unexplored cells within 1 m of that cell (what the camera would reveal there)
            r = int(1.0 / self.res)
            i, j = int(iis[c]), int(js[c])
            win = unexp[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1]
            out.append((*self.xy(i, j), float(win.sum()) * self.res ** 2))
        return out

    def astar(self, start, goal, blocked, avoid=None, avoid_cost=0.0):
        """Waypoints [(x, y)] from start to goal round blocked cells, or None. A blocked start may leave its margin;
        a blocked goal moves to the nearest free cell within 1 m. avoid: cells costing avoid_cost extra per step."""
        si, sj = self.cell(*start)
        gi, gj = self.cell(*goal)
        if not (self.inside(si, sj) and self.inside(gi, gj)):
            return None
        if blocked[gj, gi]:
            js, iis = np.nonzero(~blocked)
            if not len(js):
                return None
            d = (iis - gi) ** 2 + (js - gj) ** 2
            k = int(np.argmin(d))
            if d[k] * self.res ** 2 > 1.0:
                return None
            gi, gj = int(iis[k]), int(js[k])
        free = ~blocked
        free[sj, si] = True
        g = {(si, sj): 0.0}
        prev = {}
        heap = [(0.0, si, sj)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while heap:
            _, i, j = heapq.heappop(heap)
            if (i, j) == (gi, gj):
                break
            for di, dj, c in steps:
                ni, nj = i + di, j + dj
                if not self.inside(ni, nj) or not (free[nj, ni] or blocked[sj, si] and blocked[nj, ni] and
                                                   g[(i, j)] < 1.0 / self.res):      # leave a start margin
                    continue
                ng = g[(i, j)] + c * (1.0 + avoid_cost if avoid is not None and avoid[nj, ni] else 1.0)
                if ng < g.get((ni, nj), 1e18):
                    g[(ni, nj)], prev[(ni, nj)] = ng, (i, j)
                    heapq.heappush(heap, (ng + math.hypot(gi - ni, gj - nj), ni, nj))
        if (gi, gj) not in g:
            return None
        cells = [(gi, gj)]
        while cells[-1] != (si, sj):
            cells.append(prev[cells[-1]])
        cells.reverse()
        # string-pull: keep the furthest cell in line of sight - with avoid, a shortcut may only cross flown-over
        # cells the A* path itself crossed (else it pulls the route straight back over the trail)
        pull = free
        if avoid is not None and avoid_cost > 0:
            on = np.zeros_like(free)
            on[tuple(np.array(cells).T[::-1])] = True
            on = cv2.dilate(on.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            pull = free & (~avoid | on)
        out, k = [cells[0]], 0
        while k < len(cells) - 1:
            n = len(cells) - 1
            while n > k + 1 and not self.line_free(cells[k], cells[n], pull):
                n -= 1
            out.append(cells[n])
            k = n
        return [self.xy(i, j) for i, j in out[1:]] or [self.xy(gi, gj)]

    @staticmethod
    def line_free(a, b, free):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        for t in np.linspace(0.0, 1.0, n):
            i, j = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
            if not free[j, i]:
                return False
        return True

    def path_blocked(self, path, blocked):
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            if not self.line_free(self.cell(x0, y0), self.cell(x1, y1), ~blocked):
                return True
        return False

    def png(self):
        img = np.zeros((self.h, self.w, 4), np.uint8)
        img[self.in_zone] = (40, 48, 58, 255)                        # zone, unexplored
        img[self.in_zone & self.explored] = (40, 110, 75, 255)       # explored
        img[self.blocked() & ~self.red()] = (120, 40, 45, 200)       # red margin
        img[self.red()] = (229, 72, 77, 255)                         # red zone
        ok, buf = cv2.imencode('.png', cv2.cvtColor(img[::-1], cv2.COLOR_RGBA2BGRA))
        return buf.tobytes() if ok else None


class VrxExplore(base.VrxDeliverySystem):
    def __init__(self, a):
        super().__init__(argparse.Namespace(qgc=a.qgc, wechat_dir=a.wechat_dir, vrx=a.video, fov_deg=a.fov_deg,
                                            start_qr='', radio=a.link))
        self.a = a
        a.alt = min(a.alt, MAX_ALT)                # the airspace rule: never above 10 m
        self.alt = a.alt
        self.search_alt = self.takeoff_alt = a.alt              # ground projection fallback on the bench
        self.vmax = a.speed
        self.zone_enu = tuple(a.zone)
        self.map = ExploreMap(self.zone_enu, (0.0, 0.0), res=a.res, red_margin=a.red_margin, start_r=a.start_zone_r)
        self.map_png, self.map_ver = None, -1
        self.goal = None          # {'x', 'y', 'kind'}
        self.path = []
        self.blacklist = []
        self.visited = []
        self.frame_size = (640, 480)
        self.video_err = 'no device selected' if not a.video else 'opening...'
        self.fsm = FSM
        self.go_home = False
        self.in_flight = False
        self.drop = a.drop
        # QR area: 5 m (the drop height) is the lowest the drone goes, and only to lower the payload (user, 2026-10-30)
        self.deliver_alt = a.drop_alt if a.marker == 'qr' else min(a.drop_alt, a.alt)
        self.start_qr = a.start_qr or None
        self.open_batch, self.batch_t = 0, None   # batch new candidates join; when the last one was started
        self.rejects, self.spins, self.spin_due = set(), [], None   # reject payloads; scan spots; a scan to do
        self.scout = None                         # scout waypoints left, None = no scout
        self.aruco, self.cam = None, None
        self.gf = {'arena': [], 'qr': [], 'extra': []}   # the GPS slots typed on the page
        self.template, self.templates = None, []         # the start QR's image (qr_match.signature)
        if a.marker == 'aruco':
            self.aruco = aruco_match                  # matched by image, never read (user, 2026-10-30)
            print('ArUco delivery: markers matched by IMAGE to the start marker (no id read)', flush=True)

    # ---- the lens: equidistant model of the analog camera at the frame's size
    def lens(self, w, h):
        if self.a.lens == 'pinhole':
            return None
        if self.cam is None or self.cam.size != (w, h):
            dfov = self.a.lens_dfov or LENS_DFOV[self.a.lens]
            f = math.hypot(w, h) / 2 / math.radians(dfov / 2)        # equidistant: r = f * theta
            info = argparse.Namespace(distortion_model='equidistant', k=[f, 0, (w - 1) / 2, 0, f, (h - 1) / 2, 0, 0, 1],
                                      d=[0.0, 0.0, 0.0, 0.0], width=w, height=h,
                                      header=argparse.Namespace(frame_id='vrx'))
            self.cam = Fisheye(info)
        return self.cam

    def pixel_to_ground_enu(self, px_pts, frame_w, frame_h, pose_enu, rpy_ned):
        """The base projection with the lens model's rays instead of the pinhole (same frames and conventions)."""
        cam = self.lens(frame_w, frame_h)
        if cam is None:
            return super().pixel_to_ground_enu(px_pts, frame_w, frame_h, pose_enu, rpy_ned)
        if pose_enu is None:
            return None, 1.0
        xe, yn, zu, _ = pose_enu
        if zu < 0.4:
            zu = float(self.search_alt)  # bench test fallback when drone is on the ground
        r = cam.rays(np.asarray(px_pts, dtype=np.float64).reshape(-1, 2))      # optical: x right, y down, z out
        rays_flu = np.column_stack([-r[:, 1], -r[:, 0], -r[:, 2]])             # mounted straight down
        rays_enu = rays_flu @ base.rpy_to_rot_enu(*rpy_ned).T
        if (rays_enu[:, 2] > -0.08).any():
            return None, 0.1
        cos_nadir = float(np.clip(-rays_enu[0, 2], 0.15, 1.0))
        k = zu / (-rays_enu[:, 2])
        return np.array([xe, yn]) + rays_enu[:, :2] * k[:, None], cos_nadir ** 4

    # ---- ArUco boards: matched by image (aruco_match), never read
    def aruco_views(self, img):
        """[(view, to-frame-px function)]: undistorted views through the lens model (100 deg down + 5 x 90 deg tilted
        55 deg), else the raw frame."""
        cam = self.lens(img.shape[1], img.shape[0])
        if cam is None:
            return [(img, lambda q: q)]
        out = []
        for ray in aruco_tools.view_rays(55.0, 5):
            view, meta = cam.rectify_patch(img, ray_d=ray, fov_deg=100.0 if ray is None else 90.0, zoom_scale=1.0,
                                           max_size=720)
            out.append((view, lambda q, meta=meta: cam.patch_px_to_fisheye_px(q, meta)))
        return out

    def aruco_seen(self, img):
        """[(frame px quad, score vs the start marker or None, sharpness, side px in the view)]: one per marker (the
        crisper view)."""
        am, seen = self.aruco, []
        for view, to_px in self.aruco_views(img):
            for q in am.marker_quads(view, min_px=8.0):
                px = to_px(q)
                if not np.isfinite(px).all():
                    continue
                c, side = px.mean(axis=0), am.side_px(px)
                score = am.similarity(self.template, am.signature(view, q)) if self.template is not None else None
                sharp = am.sharpness(view, q)
                k = next((k for k, o in enumerate(seen) if np.linalg.norm(o[0].mean(axis=0) - c) < side / 2), None)
                one = (px, score, sharp, am.side_px(q), am.glare_share(view, q))
                if k is None:
                    seen.append(one)
                elif sharp > seen[k][2]:
                    seen[k] = one
        return seen

    def _decode_patch(self, img):
        if self.aruco is None:
            return super()._decode_patch(img)
        return {}                                     # nothing is read: see aruco_seen

    def process_vrx_frame(self, gray, pose_now, rpy_now, now):
        if self.aruco is None:
            return super().process_vrx_frame(gray, pose_now, rpy_now, now)
        h, w = gray.shape[:2]
        am, target = self.aruco, self.start_qr or aruco_match.IMAGE_ONLY
        found, boxes = {}, []
        for px, score, sharp, side, glare in self.aruco_seen(gray):
            g_pts, weight = self.pixel_to_ground_enu(np.vstack([px.mean(axis=0), px]), w, h, pose_now, rpy_now)
            verdict = am.classify(score, sharp, side, self.a.aruco_hi, self.a.aruco_lo, glare)
            if verdict == 'match':
                payload = target
            elif verdict == 'different':
                payload = '<marker>'
            else:                                     # blurry / between: hover over it for the deep read
                if g_pts is not None:
                    self.add_candidate(float(g_pts[0, 0]), float(g_pts[0, 1]), conf=0.5, now=now)
                boxes.append((0.5, px, f'marker, unsure {score:.2f}' if score is not None else 'marker'))
                continue
            boxes.append((1.0, px, 'TARGET (image match)' if payload == target else 'marker: not the target'))
            found[payload if payload == target else f'<marker {len(found)}>'] = px
            if g_pts is not None:
                corners = g_pts[1:]
                self.add_qr_detection(payload, (float(g_pts[0, 0]), float(g_pts[0, 1])),
                                      [*corners.min(axis=0), *corners.max(axis=0)], weight, now, pose_now)
        return found, boxes, []

    def add_candidate(self, cx, cy, conf, now):
        """The base's averaged candidate, plus its batch on the first sighting."""
        super().add_candidate(cx, cy, conf, now)
        new = []
        with self.lock:
            t = time.time()
            for c in self.cands:
                if 'batch' not in c:
                    gather = self.batch_t is not None and t - self.batch_t < self.a.batch_gather
                    c['batch'] = self.open_batch - 1 if gather else self.open_batch
                    new.append(c)
        for c in new if self.mission_running else []:   # log_mission takes self.lock
            self.log_mission('EXPLORE', event=f'candidate at ({c["x"]:.1f}, {c["y"]:.1f}): batch {c["batch"]}')

    # ---- video: only the chosen device; colour frames also feed the red-zone votes
    def open_vrx(self):
        d = self.vrx_device
        if d and d.startswith('gz:'):                              # the 8084 sim's B19 camera
            cap = GzCapture(d[3:])
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok:
                self.video_err = None
                print(f'[VIDEO] {d} {frame.shape[1]}x{frame.shape[0]}', flush=True)
                return cap
            self.video_err = f'{d}: no frames (is the sim running? GZ_IP / GZ_PARTITION)'
            return None
        if not d or not Path(d).exists():
            self.video_err = f'{d or "no device"}: not found'
            return None
        cap = cv2.VideoCapture(d, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                self.video_err = None
                print(f'[VIDEO] {d} {frame.shape[1]}x{frame.shape[0]}', flush=True)
                return cap
        cap.release()
        self.video_err = f'{d}: opened but no frames'
        return None

    def vrx_loop(self):
        cap, dev, last_det = None, None, 0.0
        while True:
            if dev != self.vrx_device and cap is not None:          # device switched on the page
                cap.release()
                cap = None
            if cap is None:
                dev = self.vrx_device
                cap = self.open_vrx()
                if cap is None:
                    time.sleep(1.5)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                self.video_err = f'{dev}: lost'
                continue
            now = time.monotonic()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            with self.lock:
                pose_now, rpy_now = self.pose, self.rpy_ned
                self.latest_gray = gray
            self.frame_size = (gray.shape[1], gray.shape[0])
            if now - last_det >= 0.2:                                # 5 Hz, as the delivery GCS
                last_det = now
                found, boxes_viz, zoom = self.process_vrx_frame(gray, pose_now, rpy_now, now)
                if frame.ndim == 3:
                    self.red_votes(frame, pose_now, rpy_now)
                self.render_hud_frame(frame, gray, found, boxes_viz, zoom)

    def flying_level(self, pose, rpy):
        return pose is not None and pose[2] > 0.4 and abs(rpy[0]) < 0.26 and abs(rpy[1]) < 0.26

    def red_votes(self, frame, pose, rpy):
        if not self.flying_level(pose, rpy):
            return
        h, w = frame.shape[:2]
        us, vs = np.meshgrid(np.linspace(0.06, 0.94, 28) * w, np.linspace(0.06, 0.94, 21) * h)
        px = np.column_stack([us.ravel(), vs.ravel()])
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        p = hsv[px[:, 1].astype(int), px[:, 0].astype(int)]
        is_red = ((p[:, 0] <= 10) | (p[:, 0] >= 170)) & (p[:, 1] >= 110) & (p[:, 2] >= 60)
        g, _ = self.pixel_to_ground_enu(px, w, h, pose, rpy)
        if g is not None:
            with self.lock:
                self.map.add_red_samples(g, is_red)

    def watchdog_loop(self):
        """In flight, leaving OFFBOARD / GUIDED (the pilot / QGC took over, an autopilot failsafe) or disarming ends the
        mission: it used to carry on 'exploring' while PX4 sat disarmed on the ground (2026-10-27)."""
        bad_since = None
        while True:
            time.sleep(0.2)
            if not (self.mission_running and self.in_flight):
                bad_since = None
                continue
            with self.lock:
                z = self.pose[2] if self.pose else None
            if z is not None and z > self.alt + self.a.max_over and not self.abort_reason:
                self.abort_reason = f'height {z:.1f} m is over {self.alt + self.a.max_over:.1f} m: landing'
                self.offboard_active = False
                self.set_px4_mode('AUTO.LAND')
                continue
            ok = self.tel.get('armed') and self.tel.get('mode') == self.offboard_name
            bad_since = None if ok else (bad_since or time.monotonic())
            if bad_since and time.monotonic() - bad_since > 1.5 and not self.abort_reason:
                if self.tel.get('armed'):
                    # any other mode while armed: the RC switch (STABILIZE / AUTO / RTL), MANUAL, an autopilot failsafe -
                    # the explorer's failsafe must not put BRAKE / LAND over it
                    self.offboard_active = False
                    self.abort_reason = f'operator took over: {self.tel.get("mode")}'
                    continue
                self.abort_reason = (f'left {self.offboard_name} (now {self.tel.get("mode")}'
                                     f'{", disarmed" if not self.tel.get("armed") else ""})')

    def coverage_loop(self):
        """Mark the camera's read area explored from the pose (5 Hz): coverage without a camera too."""
        while True:
            time.sleep(0.2)
            with self.lock:
                pose, rpy = self.pose, self.rpy_ned
            if not self.flying_level(pose, rpy) or abs(pose[2] - self.alt) > 0.5:
                continue
            w, h = self.frame_size
            f = self.a.read_frac
            corners = [(w * (0.5 - f / 2), h * (0.5 - f / 2)), (w * (0.5 + f / 2), h * (0.5 - f / 2)),
                       (w * (0.5 + f / 2), h * (0.5 + f / 2)), (w * (0.5 - f / 2), h * (0.5 + f / 2))]
            g, _ = self.pixel_to_ground_enu(corners, w, h, pose, rpy)
            if g is not None:
                with self.lock:
                    self.map.mark_explored(g)
                    if self.mission_running:
                        self.map.mark_trail(pose[0], pose[1], self.a.trail_radius)

    def _update_qr_status(self, q):
        """(under self.lock) also: a reject read for the first time near the drone in the search -> a yaw scan."""
        super()._update_qr_status(q)
        q['in_zone'] = self.in_zone(q['x'], q['y'])          # the QR area polygon, not its bounding box
        q['status'] = 'found' if q['confirmed'] else 'reading'
        key = q['payload'] if q['payload'] != '<marker>' else f'<marker {round(q["x"] / 3)},{round(q["y"] / 3)}>'
        if (self.start_qr and q['payload'] != self.start_qr and key not in self.rejects
                and q.get('in_zone') and self.mission_running and self.in_flight):
            self.rejects.add(key)
            if self.pose and math.dist(self.pose[:2], (q['x'], q['y'])) < self.a.spin_near:
                self.spin_due = (q['x'], q['y'])

    # ---- the start QR's image + the deep read (QR boards; see the docstring)
    def view_at(self, gray, x, y, pose, rpy, fovs=((12.0, 3.0), (18.0, 2.2), (26.0, 1.6))):
        """Undistorted views (one per FOV / zoom) along the ray to the ground point (x, y), or None."""
        h, w = gray.shape[:2]
        cam = self.lens(w, h)
        if pose is None or cam is None:
            return None if cam is not None else [('centre', gray[h // 4:3 * h // 4, w // 4:3 * w // 4])]
        z = pose[2] if pose[2] > 0.4 else float(self.search_alt)
        ray = np.array([x - pose[0], y - pose[1], -z])
        flu = (ray / np.linalg.norm(ray)) @ base.rpy_to_rot_enu(*rpy)     # rays_enu = rays_flu @ R.T, inverted
        opt = np.array([-flu[1], -flu[0], -flu[2]])                       # FLU -> optical (camera straight down)
        if opt[2] < math.cos(math.radians(60.0)):
            return None
        return [(f'fov{v[0]:g}' if len(v) == 2 else v[0],          # (FOV, zoom) or aruco_match.SETTINGS rows
                 cam.rectify_patch(gray, ray_d=opt, fov_deg=v[-3] if len(v) == 4 else v[0],
                                   zoom_scale=v[-2] if len(v) == 4 else v[1], max_size=v[3] if len(v) == 4 else 400)[0])
                for v in fovs]

    def scan_start_qr_here(self, duration=5.0):
        """The base's read + the board's image for the deep read; a board seen but never decoded (no data) ->
        qr_match.IMAGE_ONLY."""
        if self.aruco is not None:
            return self.scan_start_aruco(duration)
        self.templates, stop = [], threading.Event()

        def grab():
            while not stop.is_set():
                with self.lock:
                    gray, pose, rpy = self.latest_gray, self.pose, self.rpy_ned
                views = self.view_at(gray, pose[0], pose[1], pose, rpy, fovs=((40.0, 1.0),)) \
                    if gray is not None and pose is not None else None
                sig = qr_match.signature(views[0][1]) if views else None
                if sig is not None:
                    self.templates.append(sig)
                time.sleep(0.3)
        t = threading.Thread(target=grab, daemon=True)
        t.start()
        try:
            got = super().scan_start_qr_here(max(duration, 3.0))
        finally:
            stop.set()
            t.join(1.0)
        if self.templates:
            self.template = self.templates[len(self.templates) // 2]
        if not got and len(self.templates) >= 2:
            with self.lock:
                self.start_qr = qr_match.IMAGE_ONLY
                for q in self.qrs:
                    self._update_qr_status(q)
            self.log_mission('READ_START_QR', event='a board but no data decoded: matching by image only')
            return qr_match.IMAGE_ONLY
        return got

    def scan_start_aruco(self, duration=5.0):
        """The start marker's IMAGE (no id read): the marker nearest straight down, the crispest x biggest view of
        duration s kept as the template; start = aruco_match.IMAGE_ONLY. None if no marker was seen twice."""
        am, views, last = self.aruco, [], None
        t_end = time.monotonic() + max(duration, 3.0)
        while time.monotonic() < t_end:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            with self.lock:
                gray = self.latest_gray
            if gray is None or gray is last:
                time.sleep(0.05)
                continue
            last = gray
            v = self.aruco_views(gray)[0][0]          # straight down
            qs = am.marker_quads(v, min_px=8.0)
            if qs:
                h, w = v.shape[:2]
                q = min(qs, key=lambda q: np.hypot(*(q.mean(axis=0) - [w / 2, h / 2])))
                sig = am.signature(v, q)
                if sig is not None:
                    side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
                    views.append((am.sharpness(v, q) * side, sig))
            time.sleep(0.15)
        if len(views) < 2:
            return None
        self.template = max(views, key=lambda t: t[0])[1]
        with self.lock:
            self.start_qr = aruco_match.IMAGE_ONLY
            for q in self.qrs:
                self._update_qr_status(q)
        self.log_mission('READ_START_QR', event=f'start marker image kept ({len(views)} views): matching by image')
        return aruco_match.IMAGE_ONLY

    def deep_read_aruco(self, x, y):
        """The ArUco deep read over a candidate: every setting x filter per frame, the best score kept."""
        am = self.aruco
        scores, sharp, frames, last, how, glare = [], 0.0, 0, None, '', 0.0
        while frames < self.a.verify_rounds:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            with self.lock:
                gray, pose, rpy = self.latest_gray, self.pose, self.rpy_ned
            if gray is None or gray is last:
                time.sleep(0.05)
                continue
            last, frames = gray, frames + 1
            views = self.view_at(gray, x, y, pose, rpy, fovs=am.SETTINGS)
            best = None
            for name, patch in views or []:
                r = am.best_in_patch(patch, self.template, min_px=8.0)
                if r is not None and (best is None or r[0] > best[0]):
                    best = (r[0], r[1], f'{name}/{r[2]}', r[3])
            if best is not None:
                scores.append(best[0])
                sharp = max(sharp, best[1])
                glare = max(glare, best[3])
                how = best[2] if best[0] >= max(scores) else how
        if self.template is None or len(scores) < 3:
            return 'unsure' if self.template is None else 'unseen'
        s = float(np.median(sorted(scores)[-3:]))
        res = 'unsure' if sharp < am.BLURRY else 'match' if s >= self.a.aruco_hi else \
            'different' if s <= self.a.aruco_lo and glare <= am.GLARE_MAX else 'unsure'
        if res == 'match':
            self.add_board(self.start_qr or am.IMAGE_ONLY, x, y, pose)
        elif res == 'different':
            self.add_board('<marker>', x, y, pose)
        self.log_mission('EXPLORE', event=f'deep read at ({x:.1f}, {y:.1f}): marker {res} ({s:.2f}, sharpness '
                                          f'{sharp:.2f}, best {how})')
        return res

    def deep_read(self, x, y):
        """Over an unread candidate: -> 'read' | 'match' | 'different' | 'unsure' | 'unseen'."""
        if self.aruco is not None:
            return self.deep_read_aruco(x, y)
        scores, frames, last = [], 0, None
        while frames < self.a.verify_rounds:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            with self.lock:
                gray, pose, rpy = self.latest_gray, self.pose, self.rpy_ned
            if gray is None or gray is last:
                time.sleep(0.05)
                continue
            last, frames = gray, frames + 1
            views = self.view_at(gray, x, y, pose, rpy)
            if not views:
                continue
            for name, patch in views:
                for fname, img in qr_match.variants(patch):
                    hit = self._decode_patch(img)
                    if hit:
                        payload = next(iter(hit))
                        if self.start_qr == qr_match.IMAGE_ONLY and \
                                (qr_match.similarity(self.template, qr_match.signature(views[1][1])) or -1) >= self.a.match_hi:
                            payload = qr_match.IMAGE_ONLY
                        self.add_board(payload, x, y, pose)
                        self.log_mission('EXPLORE', event=f'deep read at ({x:.1f}, {y:.1f}): {payload!r} ({name}/{fname})')
                        return 'read'
            s = qr_match.similarity(self.template, qr_match.signature(views[1][1]))
            if s is not None:
                scores.append(s)
        if self.template is None or len(scores) < 3:
            return 'unsure' if self.template is None else 'unseen'
        s = float(np.median(sorted(scores)[-3:]))
        res = 'match' if s >= self.a.match_hi else 'different' if s <= self.a.match_lo else 'unsure'
        if res == 'match':
            self.add_board(self.start_qr, x, y, pose)
        elif res == 'different':
            self.add_board(f'<board {x:.1f},{y:.1f}>', x, y, pose)
        self.log_mission('EXPLORE', event=f'deep read at ({x:.1f}, {y:.1f}): image {res} ({s:.2f})')
        return res

    def add_board(self, payload, x, y, pose):
        for _ in range(2):                                   # confirmed (n >= 2)
            self.add_qr_detection(payload, (x, y), [x - 0.5, y - 0.5, x + 0.5, y + 0.5], 1.0, time.monotonic(), pose)

    def verify_candidate(self, x, y):
        """The deep read, then 0.5 m lower and again while unsure (not below --verify-floor); height restored."""
        alt0 = self.alt
        try:
            while True:
                with self.lock:
                    read = any(math.dist((q['x'], q['y']), (x, y)) < 1.5 for q in self.qrs)
                if read:
                    return
                res = self.deep_read(x, y)
                if res in ('read', 'match', 'different'):
                    return
                floor = self.a.alt_min if self.aruco is not None else self.a.verify_floor   # ArUco flies at 1 m
                if self.alt - self.a.verify_step < floor - 1e-6:
                    self.log_mission('EXPLORE', event=f'deep read at ({x:.1f}, {y:.1f}): {res} down to '
                                                      f'{self.alt:.1f} m - giving up on it')
                    return
                self.set_alt(self.alt - self.a.verify_step)
                self.log_mission('EXPLORE', event=f'deep read at ({x:.1f}, {y:.1f}): {res} - down to {self.alt:.1f} m')
                time.sleep(self.a.verify_step / 0.5 + self.a.cand_hover)   # descend, settle, the normal decode
        finally:
            if self.alt != alt0:
                self.set_alt(alt0)

    def set_alt(self, alt):
        with self.lock:
            self.alt = round(alt, 2)
            if self.offboard_active and self.setpoint:
                x, y, _, yaw = self.setpoint
                self.setpoint = (x, y, self.alt, yaw)

    def zone_from_gps(self, gf):
        """GPS slots -> (QR area corners ENU, message) or (None, error)."""
        if self.origin is None:
            return None, 'no GPS origin from the autopilot yet'
        lat0, lon0 = self.origin[0], self.origin[1]
        loc = {k: [base.gps_to_local(float(p[0]), float(p[1]), lat0, lon0) for p in gf.get(k) or []]
               for k in ('arena', 'qr', 'extra')}
        if len(loc['qr']) == 4:
            return qr_area.ccw(loc['qr']), 'QR area given'
        if len(loc['arena']) != 4:
            return None, 'need 4 QR-area corners or 4 arena corners'
        with self.lock:
            home = (self.pose[0], self.pose[1]) if self.pose else (0.0, 0.0)
        back = sorted(loc['arena'], key=lambda p: -math.dist(p, home))[:2]
        mid = ((back[0][0] + back[1][0]) / 2, (back[0][1] + back[1][1]) / 2)
        poly = qr_area.front_from_tunnel_exit(loc['arena'], back, home, math.atan2(mid[1] - home[1], mid[0] - home[0]))
        if poly is None:
            return None, 'the arena and the take-off point give no closed QR area'
        return poly, 'arena: front wall through the take-off point'

    # ---- planning
    def pending(self):
        """(under self.lock) candidates still to visit: seen twice, in the zone, not visited / given up / in red."""
        red = self.map.red()
        out = []
        for c in self.cands:
            i, j = self.map.cell(c['x'], c['y'])
            if (c['n'] >= 2 and self.in_zone(c['x'], c['y']) and not red[j, i]
                    and not any(math.dist((c['x'], c['y']), v) < 1.0 for v in self.visited + self.blacklist)):
                out.append(dict(c))
        return out

    def route(self, pos, goal, blocked, fresh):
        """Explorer A*: flown-over cells cost --retrace-cost extra while a frontier is left (fresh)."""
        if fresh and self.a.retrace_cost > 0:
            return self.map.astar(pos, goal, blocked, avoid=self.map.trail, avoid_cost=self.a.retrace_cost)
        return self.map.astar(pos, goal, blocked)

    def pick_goal(self, pos, blocked):
        fronts = self.map.frontiers(blocked)
        with self.lock:
            cands = self.pending()
        if cands:
            # the earliest batch with any left, its cheapest route (re-costed from here) first
            b = min(c.get('batch', 0) for c in cands)
            best = None
            for c in (c for c in cands if c.get('batch', 0) == b):
                path = self.route(pos, (c['x'], c['y']), blocked, bool(fronts))
                if path:
                    d = self.path_len(pos, path)
                    if best is None or d < best[0]:
                        best = (d, {'x': c['x'], 'y': c['y'], 'kind': 'QR candidate', 'batch': b}, path)
                else:
                    self.blacklist.append((c['x'], c['y']))           # unreachable: dropped
            if best:
                with self.lock:
                    started = b >= self.open_batch
                    if started:
                        self.open_batch, self.batch_t = b + 1, time.time()
                if started:
                    self.log_mission('EXPLORE', event=f'batch {b}: {sum(c.get("batch", 0) == b for c in cands)} '
                                     'candidate(s), cheapest route first')
                return best[1], best[2]
            return self.pick_goal(pos, blocked)                      # that batch was unreachable: the next one
        if self.scout:
            tx, ty = self.scout[0]
            path = self.route(pos, (tx, ty), blocked, bool(fronts))
            if path:
                return {'x': tx, 'y': ty, 'kind': 'scout'}, path
            self.scout.pop(0)
        scored = []
        for x, y, gain in fronts:
            if any(math.dist((x, y), b) < 0.8 for b in self.blacklist):
                continue
            scored.append((gain / (math.dist(pos, (x, y)) + 1.0), x, y, gain))
        for _, x, y, gain in sorted(scored, reverse=True)[:6]:      # best few by straight distance; A* checks
            path = self.route(pos, (x, y), blocked, True)
            if path:
                return {'x': x, 'y': y, 'kind': 'frontier', 'gain_m2': round(gain, 2)}, path
            self.blacklist.append((x, y))
        return None, None

    def make_scout(self, pos):
        """Waypoints out to the line scout_offset in from the nearer long wall (45 deg), then along it to the far end."""
        x0, x1, y0, y1 = self.zone_enu
        mg = min(1.0, (min(x1 - x0, y1 - y0)) / 4)
        along_x = (x1 - x0) >= (y1 - y0)                  # the long walls are the y edges
        a0, a1, b0, b1 = (x0, x1, y0, y1) if along_x else (y0, y1, x0, x1)
        pa, pb = (pos[0], pos[1]) if along_x else (pos[1], pos[0])
        off = min(self.a.scout_offset, (b1 - b0) / 4)
        line = min((b0 + off, b1 - off), key=lambda v: abs(v - pb))
        far = a1 - mg if abs(a1 - pa) > abs(pa - a0) else a0 + mg
        step = math.copysign(min(abs(far - pa), abs(line - pb)), far - pa)
        pts = [(pa + step, line), (far, line)]
        return [p if along_x else (p[1], p[0]) for p in pts], f'{"y" if along_x else "x"}={line:.1f} ({off:.1f} m in)'

    def yaw_scan(self, x, y):
        """Turn in place to 3 more headings 90 deg apart, dwelling at each (boards out of view join the next batch)."""
        if any(math.dist(s, (x, y)) < self.a.spin_skip for s in self.spins):
            return
        self.spins.append((x, y))
        with self.lock:
            sx, sy, sz, yaw0 = self.setpoint
        self.log_mission('EXPLORE', event=f'reject at ({x:.1f}, {y:.1f}): yaw scan for more boards')
        for k in (1, 2, 3):
            yaw = math.atan2(math.sin(yaw0 + k * math.pi / 2), math.cos(yaw0 + k * math.pi / 2))
            with self.lock:
                self.setpoint = (sx, sy, self.alt, yaw)    # self.alt: an operator height trim holds here too
            t_end = time.monotonic() + 5.0
            while time.monotonic() < t_end:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                with self.lock:
                    cur = self.pose[3] if self.pose else yaw
                if abs(math.atan2(math.sin(cur - yaw), math.cos(cur - yaw))) < 0.15:
                    break
                time.sleep(0.1)
            time.sleep(self.a.spin_dwell)
        with self.lock:
            new = sum(1 for c in self.pending() if c.get('batch', 0) >= self.open_batch)
        self.log_mission('EXPLORE', event=f'yaw scan done: {new} candidate(s) for the next batch')

    def in_zone(self, x, y):
        i, j = self.map.cell(x, y)
        return self.map.inside(i, j) and bool(self.map.in_zone[j, i])

    @staticmethod
    def path_len(pos, path):
        return sum(math.dist(a, b) for a, b in zip([pos] + path[:-1], path))

    def follow(self, path, goal):
        """Fly the waypoints at vmax; returns 'arrived' | 'replan'."""
        dt = 0.1
        with self.lock:
            sp = self.setpoint
        cx, cy = sp[0], sp[1]
        for k, (tx, ty) in enumerate(path):
            yaw = math.atan2(ty - cy, tx - cx) if math.hypot(tx - cx, ty - cy) > 0.3 else sp[3]
            while True:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                with self.lock:
                    pose = self.pose
                    # one cell less margin than A* planned with: only a red zone that is new trips this, not the
                    # planner's own grid rounding (returning home it re-planned 5 times and landed away, 2026-10-27)
                    blocked = self.map.blocked(max(self.map.res, self.map.red_margin - self.map.res))
                    if goal['kind'] == 'frontier':
                        gi, gj = self.map.cell(goal['x'], goal['y'])
                        seen = self.map.explored[gj, gi]
                    else:
                        seen = False
                if self.go_home and goal['kind'] != 'home':
                    return 'replan'
                if goal['kind'] in ('frontier', 'scout'):
                    with self.lock:
                        if self.spin_due is not None or self.pending():
                            return 'replan'                           # a reject to scan / a board to go to first
                rest = [(cx, cy)] + path[k:]
                near = np.zeros_like(blocked, np.uint8)
                ci, cj = self.map.cell(cx, cy)
                cv2.circle(near, (ci, cj), int(1.0 / self.map.res) + 2, 1, -1)   # A* may leave a margin within 1 m
                if self.map.path_blocked(rest, blocked & ~near.astype(bool)):
                    self.log_mission('EXPLORE', event='red zone on the path: re-planning')
                    return 'replan'
                if seen and goal['kind'] == 'frontier':
                    return 'replan'                                   # revealed on the way: pick the next one
                d = math.hypot(tx - cx, ty - cy)
                step = self.vmax * dt
                if d > step:
                    cx, cy = cx + (tx - cx) / d * step, cy + (ty - cy) / d * step
                else:
                    cx, cy = tx, ty
                with self.lock:
                    self.setpoint = (cx, cy, self.alt, yaw)
                if (cx, cy) == (tx, ty) and pose and math.hypot(tx - pose[0], ty - pose[1]) < 0.3:
                    break
                time.sleep(dt)
        return 'arrived'

    def run_explore(self):
        self.mission_running = True
        self.abort_reason = None
        self.mission_t0 = time.time()
        self.blacklist, self.visited = [], []
        self.open_batch, self.batch_t, self.rejects, self.spins, self.spin_due = 0, None, set(), [], None
        self.scout = None
        with self.lock:
            for c in self.cands:
                c['batch'] = 0                            # seen before the start: the first batch
        self.go_home = False
        self.fsm = FSM_DROP if self.drop else FSM
        delivered = False
        with self.lock:
            self.target = None
            for q in self.qrs:
                self._update_qr_status(q)
        with self.lock:
            home_pose = self.pose or (0.0, 0.0, 0.0, 0.0)
        home = (home_pose[0], home_pose[1])
        try:
            self.log_mission('TAKEOFF', altitude_m=self.alt)
            self.arm_takeoff(home[0], home[1], self.alt, home_pose[3], timeout=30.0)   # PX4 or ArduCopter
            self.in_flight = True
            if self.drop and not self.start_qr:
                # the target payload: the start QR at the pad (8082's read, from the take-off height)
                self.log_mission('READ_START_QR')
                if not self.scan_start_qr_here(duration=8.0):
                    raise RuntimeError('start QR not read at the pad (type it on the page, or no drop)')
                self.log_mission('READ_START_QR', payload=self.start_qr)

            self.zone_t0 = time.time()
            self.log_mission('EXPLORE', zone=self.zone_enu)
            if self.a.scout_offset > 0:
                time.sleep(self.a.scout_wait)             # the first view
                with self.lock:
                    seen = bool(self.pending()) or any(q.get('in_zone') for q in self.qrs)
                    pos = (self.pose[0], self.pose[1])
                if not seen:
                    self.scout, line = self.make_scout(pos)
                    self.log_mission('EXPLORE', event=f'no board in view: scout line {line} from the long wall')
            while time.time() - self.mission_t0 < self.a.time_budget and not self.go_home:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                if self.drop and self.target is not None and not delivered:
                    self.deliver_here(self.target)
                    delivered = True
                    break
                with self.lock:
                    pos = (self.pose[0], self.pose[1])
                    blocked = self.map.blocked()
                    cov, _ = self.map.coverage()
                    due, self.spin_due = self.spin_due, None
                    seen = bool(self.pending()) or any(q.get('in_zone') for q in self.qrs)
                if self.scout and seen:
                    self.log_mission('EXPLORE', event='board seen: scout line dropped')
                    self.scout = None
                if due is not None and self.a.spin_reject:
                    self.yaw_scan(*due)
                    continue
                if cov >= self.a.done_coverage:
                    self.log_mission('EXPLORE', event=f'zone {cov:.0%} explored')
                    break
                goal, path = self.pick_goal(pos, blocked)
                if goal is None:
                    self.log_mission('EXPLORE', event=f'no reachable unexplored area left ({cov:.0%} explored)')
                    break
                with self.lock:
                    self.goal, self.path = goal, path
                self.log_mission('EXPLORE', goal=goal, waypoints=len(path))
                res = self.follow(path, goal)
                if res == 'arrived' and goal['kind'] == 'scout' and self.scout:
                    self.scout.pop(0)
                    if not self.scout:
                        self.log_mission('EXPLORE', event='scout line done')
                if res == 'arrived' and goal['kind'] == 'QR candidate':
                    time.sleep(self.a.cand_hover)                    # hover for the decode
                    if self.a.deep_read:
                        self.verify_candidate(goal['x'], goal['y'])
                    self.visited.append((goal['x'], goal['y']))
                    with self.lock:
                        q = min(self.qrs, key=lambda q: math.dist((q['x'], q['y']), (goal['x'], goal['y'])), default=None)
                        reject = (q is not None and self.start_qr and q['payload'] != self.start_qr
                                  and math.dist((q['x'], q['y']), (goal['x'], goal['y'])) < 1.5)
                        self.spin_due = None                          # this scan covers a passing read here too
                    if reject and self.a.spin_reject:
                        self.yaw_scan(q['x'], q['y'])
            else:
                self.log_mission('EXPLORE', event='operator: return home' if self.go_home else
                                 f'time budget {self.a.time_budget:.0f} s used: returning')

            self.log_mission('RETURN_HOME')
            for _ in range(20):
                with self.lock:
                    pos = (self.pose[0], self.pose[1])
                    path = self.map.astar(pos, home, self.map.blocked())
                    self.goal, self.path = {'x': home[0], 'y': home[1], 'kind': 'home'}, path or []
                if path is None:
                    raise RuntimeError('no red-zone-free path home')
                if self.follow(path, self.goal) == 'arrived':
                    break
            else:
                self.log_mission('RETURN_HOME', event='could not reach home: landing where it is')
            self.log_mission('LAND')
            self.in_flight = False
            self.offboard_active = False
            self.set_px4_mode('AUTO.LAND')
            t_end = time.monotonic() + 40
            while time.monotonic() < t_end and self.tel.get('armed'):
                time.sleep(0.5)
            self.log_mission('DONE', qrs=[q['payload'] for q in self.qrs if q.get('confirmed')],
                             total_s=round(time.time() - self.mission_t0, 1))
        except Exception as e:
            self.log_mission('FAILSAFE', reason=str(e))
            self.offboard_active = False
            if 'operator took over' not in str(e):    # else the operator's LAND / RTL / RTH stands
                self.set_px4_mode('AUTO.LOITER' if 'abort' in str(e).lower() and 'height' not in str(e) else 'AUTO.LAND')
        finally:
            self.mission_running = False
            self.in_flight = False
            with self.lock:
                self.goal, self.path = None, []

    def run_hop(self):
        """Pre-mission check (user, 2026-10-28): take off over home, hover, land - no exploring. The fly-away guard
        runs against the hop height (self.alt is swapped for the hop and put back after)."""
        self.mission_running = True
        self.abort_reason = None
        self.mission_t0 = time.time()
        self.fsm = FSM_HOP
        search_alt, hop_alt = self.alt, self.a.hop_alt
        with self.lock:
            home_pose = self.pose or (0.0, 0.0, 0.0, 0.0)
            self.alt = hop_alt
        try:
            self.log_mission('TAKEOFF', altitude_m=hop_alt, event='hop test')
            self.arm_takeoff(home_pose[0], home_pose[1], hop_alt, home_pose[3], timeout=30.0)
            self.in_flight = True
            self.log_mission('HOVER', seconds=self.a.hop_hold)
            end = time.monotonic() + self.a.hop_hold
            while time.monotonic() < end:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                time.sleep(0.2)
            self.log_mission('LAND')
            self.in_flight = False
            self.offboard_active = False
            self.set_px4_mode('AUTO.LAND')
            t_end = time.monotonic() + 40
            while time.monotonic() < t_end and self.tel.get('armed'):
                time.sleep(0.5)
            self.log_mission('DONE', event='hop test: landed' if not self.tel.get('armed') else 'hop test: still armed',
                             total_s=round(time.time() - self.mission_t0, 1))
        except Exception as e:
            self.log_mission('FAILSAFE', reason=str(e))
            self.offboard_active = False
            if 'operator took over' not in str(e):
                self.set_px4_mode('AUTO.LOITER' if 'abort' in str(e).lower() and 'height' not in str(e) else 'AUTO.LAND')
        finally:
            with self.lock:
                self.alt = search_alt
            self.mission_running = False
            self.in_flight = False

    def deliver_here(self, target):
        """A* to the matched QR (round the red zones), down to the drop height, release, back up."""
        tx, ty = target
        self.log_mission('DELIVER', target=[round(tx, 2), round(ty, 2)], payload=self.start_qr)
        for _ in range(5):
            with self.lock:
                pos = (self.pose[0], self.pose[1])
                path = self.map.astar(pos, (tx, ty), self.map.blocked())
                self.goal, self.path = {'x': tx, 'y': ty, 'kind': 'target'}, path or []
            if path is None:
                raise RuntimeError('no red-zone-free path to the target')
            if self.follow(path, self.goal) == 'arrived':
                break
        if self.abort_reason:
            raise RuntimeError(self.abort_reason)
        self.fly_to(tx, ty, self.deliver_alt, vmax=0.4, tol=0.15)
        time.sleep(1.5)                                   # settle over the target
        self.log_mission('DROP', at=[round(tx, 2), round(ty, 2)])
        self.trigger_winch_drop()
        self.fly_to(tx, ty, self.alt, vmax=0.6)

    def return_home_now(self):
        if not self.mission_running:
            return False, 'no mission running (use Land)'
        self.go_home = True
        return True, 'returning home along the A* path, then landing'

    def snapshot(self):
        s = super().snapshot()
        with self.lock:
            if self.map.ver != self.map_ver:
                self.map_png, self.map_ver = self.map.png(), self.map.ver
            cov, _ = self.map.coverage()
            m = self.map
            s.update(
                read_only=self.a.read_only, autopilot=self.autopilot, fsm=self.fsm, video_err=self.video_err, video_devices=video_devices(), alt=self.alt,
                map={'ver': m.ver, 'x0': m.ox, 'y0': m.oy, 'w': m.w * m.res, 'h': m.h * m.res, 'res': m.res},
                coverage=round(cov, 3), rects=m.rects, goal=self.goal, path=self.path,
                setpoint=self.setpoint if self.offboard_active else None, mission_running=self.mission_running,
                red_cells=int(m.red().sum()), blacklist=self.blacklist[-30:],
                gf=self.gf, zone=list(m.zone),
            )
        return s


PAGE = (HERE / 'vrx_explore.html').read_text


def make_app(sc):
    async def index(_):
        return web.Response(text=PAGE(), content_type='text/html')

    async def jpeg(request):
        with sc.lock:
            jpg = sc.annotated_jpeg or sc.raw_jpeg
        if not jpg:
            raise web.HTTPServiceUnavailable()
        return web.Response(body=jpg, content_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    async def map_png(_):
        sc.snapshot()
        if not sc.map_png:
            raise web.HTTPNotFound()
        return web.Response(body=sc.map_png, content_type='image/png', headers={'Cache-Control': 'no-store'})

    async def state_json(_):
        # the 8080 dashboard's offboard mode: the explorer's mission at a glance
        with sc.lock:
            cov, _c = sc.map.coverage()
            d = {'state': sc.state, 'mission_running': sc.mission_running, 'drop': sc.drop, 'start_qr': sc.start_qr,
                 'target': sc.target, 'coverage': round(cov, 3), 'video_err': sc.video_err, 'fsm': sc.fsm,
                 'qrs': [{k: q.get(k) for k in ('payload', 'x', 'y', 'n', 'status')} for q in sc.qrs],
                 'log': sc.log_entries[-60:]}
        return web.json_response(d, dumps=lambda o: json.dumps(o, default=float))

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        try:
            while not ws.closed:
                await ws.send_str(json.dumps(sc.snapshot(), default=float))
                await asyncio.sleep(0.25)
        except Exception:
            pass
        return ws

    def ok(success, message):
        return web.json_response({'ok': bool(success), 'message': message})

    async def api(request):
        what = request.match_info['what']
        body = await request.json() if request.can_read_body else {}
        if sc.a.read_only and (what in ('start', 'hop', 'home', 'abort', 'land') or what in HANDOVER):
            return ok(False, 'read-only mode: no commands to the autopilot (restart without --read-only to fly)')
        if what == 'start':
            if sc.mission_running:
                return ok(False, 'mission already running')
            if 'HEARTBEAT' not in sc.tel_time or time.monotonic() - sc.tel_time['HEARTBEAT'] > 3:
                return ok(False, f'no autopilot heartbeat on {sc.args.radio}')
            if 'drop' in body:
                sc.drop = bool(body['drop'])
            if body.get('start_qr'):
                sc.start_qr = str(body['start_qr']).strip()
            threading.Thread(target=sc.run_explore, daemon=True).start()
            return ok(True, f'{"QR area + drop" if sc.drop else "QR area"} mission started at {sc.alt} m')
        if what == 'hop':
            if sc.mission_running:
                return ok(False, 'mission already running')
            if 'HEARTBEAT' not in sc.tel_time or time.monotonic() - sc.tel_time['HEARTBEAT'] > 3:
                return ok(False, f'no autopilot heartbeat on {sc.args.radio}')
            if sc.tel.get('armed'):
                return ok(False, 'already armed: the hop test starts on the ground')
            threading.Thread(target=sc.run_hop, daemon=True).start()
            return ok(True, f'hop test: take off to {sc.a.hop_alt} m, hover {sc.a.hop_hold:.0f} s, land')
        if what == 'abort':
            sc.abort_reason = 'Operator abort'
            sc.offboard_active = False
            return ok(sc.set_px4_mode('AUTO.LOITER'), 'aborted: holding position'
                      + (' (BRAKE)' if sc.autopilot == 'ardupilot' else ' (AUTO.LOITER)'))
        if what == 'home':
            return ok(*sc.return_home_now())
        if what == 'release':                         # the operator flies (8080 LAND / RTL / RTH / KILL): no more setpoints
            sc.abort_reason = 'operator took over'
            sc.offboard_active = False
            return ok(True, 'mission handed to the operator' if sc.mission_running else 'no mission running')
        if what in HANDOVER:                          # the pilot takes over (user, 2026-10-30): the mode, no setpoints
            sc.abort_reason = 'operator took over'
            sc.offboard_active = False
            if sc.autopilot != 'ardupilot':
                return ok(sc.set_px4_mode('AUTO.LOITER'), 'PX4: AUTO.LOITER (the hand-over modes are ArduCopter)')
            done = sc.mav_cmd(176, 1, HANDOVER[what])
            return ok(done, f'{what.upper()} {"set - yours now: fly it on the RC" if done else "NOT acknowledged"}')
        if what in ('land', 'kill'):
            sc.abort_reason = f'Operator {what}'
            sc.offboard_active = False
            if what == 'land':
                return ok(sc.set_px4_mode('AUTO.LAND'), 'LAND sent')
            return ok(sc.mav_cmd(400, 0, 21196), 'KILL sent')
        if what == 'alt':
            # the operator's search height trim, in flight too (user, 2026-10-28): follow() flies self.alt at once
            try:
                delta = float(body.get('delta'))
            except (TypeError, ValueError):
                return ok(False, 'delta: metres, e.g. -0.5')
            lo = sc.a.alt_min if sc.a.marker == 'aruco' else max(sc.a.alt_min, sc.a.verify_floor)   # QR: above the drop
            new = round(min(sc.a.alt + 3.0, MAX_ALT, max(lo, sc.alt + delta)), 2)   # never above 10 m (airspace)
            with sc.lock:
                sc.alt = new
                if sc.mission_running and sc.offboard_active and sc.setpoint:
                    x, y, _, yaw = sc.setpoint
                    sc.setpoint = (x, y, new, yaw)        # hovering / scanning too, not only the next leg
            sc.log_mission('EXPLORE', event=f'search height {new:.1f} m (trim {new - sc.a.alt:+.1f} m)')
            return ok(True, f'search height {new:.1f} m (trim {new - sc.a.alt:+.1f} m, min {lo:.1f} m)')
        if what == 'zone':
            z = [float(v) for v in body.get('zone', [])]
            if len(z) != 4 or z[0] >= z[1] or z[2] >= z[3]:
                return ok(False, 'zone needs x_min < x_max, y_min < y_max')
            if sc.mission_running:
                return ok(False, 'stop the mission first')
            with sc.lock:
                home = (sc.pose[0], sc.pose[1]) if sc.pose else (0.0, 0.0)
                sc.zone_enu = tuple(z)
                sc.map.set_zone(z, home)
            return ok(True, f'zone {z} (map cleared)')
        if what == 'zone_gps':
            if sc.mission_running:
                return ok(False, 'stop the mission first')
            gf = {k: [p for p in body.get(k) or [] if len(p) == 2][:n] for k, n in (('arena', 4), ('qr', 4), ('extra', 2))}
            poly, msg = sc.zone_from_gps(gf)
            if poly is None:
                return ok(False, msg)
            with sc.lock:
                home = (sc.pose[0], sc.pose[1]) if sc.pose else (0.0, 0.0)
                sc.gf = gf
                sc.map.set_zone(None, home, poly)
                sc.zone_enu = sc.map.zone
            return ok(True, f'{msg}: {len(poly)} corners, box x {sc.zone_enu[0]:.1f}..{sc.zone_enu[1]:.1f}, '
                            f'y {sc.zone_enu[2]:.1f}..{sc.zone_enu[3]:.1f} m (map cleared)')
        if what == 'red_rect':
            r = [float(v) for v in body.get('rect', [])]
            if len(r) == 4:
                with sc.lock:
                    sc.map.add_rect(*r)
                return ok(True, 'red zone added')
            return ok(False, 'bad rectangle')
        if what == 'clear_red':
            with sc.lock:
                sc.map.drawn[:] = False
                sc.map.rects = []
                sc.map.ver += 1
            return ok(True, 'drawn red zones cleared (camera ones stay; Reset map clears those)')
        if what == 'reset_map':
            if sc.mission_running:
                return ok(False, 'stop the mission first')
            with sc.lock:
                home = (sc.pose[0], sc.pose[1]) if sc.pose else (0.0, 0.0)
                sc.map.set_zone(sc.zone_enu, home)
                sc.qrs, sc.cands, sc.track = [], [], []
            return ok(True, 'map, QRs and track cleared')
        if what == 'video':
            d = (body.get('device') or '').strip()
            sc.vrx_device = d or None
            sc.video_err = 'opening...' if d else 'no device selected'
            return ok(True, f'video: {d or "off"}')
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/camera.jpg', jpeg)
    app.router.add_get('/map.png', map_png)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/state.json', state_json)
    app.router.add_post('/api/{what}', api)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', default='', help='USB video device (VRX receiver / webcam), e.g. /dev/video2')
    ap.add_argument('--link', default='udpout:127.0.0.1:14551',
                    help='PX4 MAVLink: the SITL via the onboard router (default), or the radio /dev/ttyUSB0:57600')
    ap.add_argument('--alt', type=float, default=1.0, help='flight height, m')
    ap.add_argument('--hop-alt', type=float, default=1.5, help='hop test (take off / hover / land) height, m')
    ap.add_argument('--hop-hold', type=float, default=10.0, help='hop test hover time, s')
    ap.add_argument('--speed', type=float, default=0.5, help='cruise speed, m/s')
    ap.add_argument('--start-zone-r', type=float, default=2.5,
                    help='m: the red START zone round the take-off point (taken off from, landed on): no red zone')
    ap.add_argument('--alt-min', type=float, default=0.8, help='m: the operator\'s height trim never goes below this')
    ap.add_argument('--zone', type=float, nargs=4, default=[1.0, 9.0, -4.0, 4.0], metavar=('X0', 'X1', 'Y0', 'Y1'),
                    help='search zone in m from home (ENU: x east, y north)')
    ap.add_argument('--max-over', type=float, default=1.0, help='m above --alt that aborts and lands (fly-away guard)')
    ap.add_argument('--res', type=float, default=0.2, help='map cell, m')
    ap.add_argument('--red-margin', type=float, default=0.5, help='m kept off red zones')
    ap.add_argument('--fov-deg', type=float, default=120.0, help='camera horizontal FOV (analog FPV ~120-150)')
    ap.add_argument('--read-frac', type=float, default=0.6, help='share of the image (centre) where a QR reads')
    ap.add_argument('--cand-hover', type=float, default=3.0, help='s hovering over a QR candidate')
    ap.add_argument('--cand-std', type=float, default=25.0, help='ArUco: grey std a not-read square needs to be a candidate')
    ap.add_argument('--batch-gather', type=float, default=2.0, help='s: candidates first seen this soon after a batch '
                    'was started join it')
    ap.add_argument('--no-spin', dest='spin_reject', action='store_false', help='no yaw scan at a reject')
    ap.add_argument('--spin-dwell', type=float, default=1.0, help='s at each heading of the yaw scan')
    ap.add_argument('--spin-skip', type=float, default=2.0, help='m: no second yaw scan this near an earlier one')
    ap.add_argument('--spin-near', type=float, default=2.5, help='m: a reject read in passing this near gets a scan')
    ap.add_argument('--scout-offset', type=float, default=7.5, help='m in from the long wall for the start line '
                    '(at most a quarter of the short side; 0 = no scout)')
    ap.add_argument('--scout-wait', type=float, default=2.0, help='s at the search start for a board to show up')
    ap.add_argument('--retrace-cost', type=float, default=3.0, help='extra A* cost per cell already flown over (0 = off)')
    ap.add_argument('--trail-radius', type=float, default=0.5, help='m round the track counted as flown over')
    ap.add_argument('--done-coverage', type=float, default=0.95, help='explored share that ends the search')
    ap.add_argument('--time-budget', type=float, default=720.0, help='s before returning (15 min mission rule)')
    ap.add_argument('--no-deep-read', dest='deep_read', action='store_false',
                    help='QR: no deep read / image match / step down at an unread candidate')
    ap.add_argument('--verify-rounds', type=int, default=5, help='frames of the deep read')
    ap.add_argument('--verify-step', type=float, default=0.5, help='m lower per unsure deep read')
    ap.add_argument('--verify-floor', type=float, default=5.5, help='m: the deep read never steps below this (QR: '
                    'nor the height trim) - 5 m is the drop height, reached only to lower the payload')
    ap.add_argument('--match-hi', type=float, default=0.80, help='image match: >= this the start QR')
    ap.add_argument('--match-lo', type=float, default=0.50, help='image match: <= this another board')
    ap.add_argument('--aruco-hi', type=float, default=0.80, help='ArUco image match: >= this the start marker')
    ap.add_argument('--aruco-lo', type=float, default=0.65, help='ArUco image match: <= this another marker')
    ap.add_argument('--qgc', default='127.0.0.1:14550')
    ap.add_argument('--wechat-dir', default='')
    ap.add_argument('--port', type=int, default=8094)
    ap.add_argument('--drop', action='store_true', help='QR area + payload drop on the QR matching the start QR')
    ap.add_argument('--drop-alt', type=float, default=5.0, help='drop height, m (course rules: 5 m)')
    ap.add_argument('--start-qr', default='', help='the target payload (else read at the pad when --drop)')
    ap.add_argument('--lens', default='pinhole', choices=('pinhole', 'b19'),
                    help='analog camera lens: pinhole = --fov-deg (as 8080); b19 = equidistant at its rated diagonal FOV')
    ap.add_argument('--lens-dfov', type=float, default=0.0, help='diagonal FOV of the lens model, deg (0 = rated)')
    ap.add_argument('--marker', default=os.environ.get('AEROTHON_MARKER', 'qr'), choices=('qr', 'aruco'),
                    help='what the boards carry: QR codes, or ArUco markers 3x3 .. 6x6 (the 8084 mission)')
    ap.add_argument('--read-only', action='store_true',
                    help='link check on a real drone: listen only - no mode / arm / setpoint commands (Kill stays)')
    a = ap.parse_args()

    sc = VrxExplore(a)
    for fn in (sc.mav_rx_loop, sc.qgc_tx_loop, sc.offboard_tx_loop, sc.vrx_loop, sc.coverage_loop, sc.watchdog_loop):
        threading.Thread(target=fn, daemon=True).start()
    print(f'[VRX-EXPLORE] http://0.0.0.0:{a.port}  link={a.link} video={a.video or "none"} alt={a.alt} m', flush=True)
    app = make_app(sc)

    async def exit_soon(_app):       # same bounded exit as the other dashboards (2026-10-27)
        threading.Timer(2.0, os._exit, args=(0,)).start()
    app.on_shutdown.append(exit_soon)
    web.run_app(app, host='0.0.0.0', port=a.port, print=None, shutdown_timeout=2.0)
    print('stopped', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
