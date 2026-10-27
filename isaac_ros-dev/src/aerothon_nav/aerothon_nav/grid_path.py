"""A* on delivery_mapper's /delivery/map and straight-segment flight along the result, for the QR zone at 10 m.

At the search height nothing stands in the way - only red zones (+ their margin) and the zone boundary ring, both
cost 100 on /delivery/map - so the delivery Nav2 (MPPI) is not needed there. On the loaded Jetson sim its controller
missed its 20 Hz rate 3842 times in one search and the drone crawled at 0.7 m/s against a 2.5 m/s limit
(2026-10-25 sim); this flies the same A* route at vmax with a few line-of-sight segments, for ~nothing in CPU.
  plan      cells >= 100 grown by `clearance` are blocked; the start cell may be blocked (the escape gets the drone
            out of a margin, but a path from its edge must still exist); a blocked goal moves to the nearest free
            cell within snap_radius (a QR read from beside a red zone - the camera reads ~3 m off nadir). 8-connected
            A* on a grid coarsened to `step` m, then string-pulled to line-of-sight waypoints.
  PathFlight.step()  one DirectFlight command towards the current waypoint (call at ~10 Hz); facing the leg, so
            the fisheye's wide (across) axis sweeps sideways. Returns 'flying' | 'arrived' | 'idle'.
"""
import heapq
import math

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

from aerothon_nav.direct_flight import wrap


def grid_of(m):
    """(array h x w int8, resolution, origin x, origin y) of an OccupancyGrid."""
    return (np.array(m.data, np.int8).reshape(m.info.height, m.info.width), m.info.resolution,
            m.info.origin.position.x, m.info.origin.position.y)


def blocked_mask(m, clearance):
    a, r, _, _ = grid_of(m)
    lethal = (a >= 100).astype(np.uint8)
    if not lethal.any():
        return np.zeros(a.shape, bool)
    dist = cv2.distanceTransform(1 - lethal, cv2.DIST_L2, 3) * r
    return dist < max(clearance, r * 0.5)


def snap_free(m, blocked, x, y, radius):
    """(x, y), or the nearest free cell centre within radius of it, or None."""
    _, r, ox, oy = grid_of(m)
    h, w = blocked.shape
    i, j = int((x - ox) / r), int((y - oy) / r)
    if 0 <= i < w and 0 <= j < h and not blocked[j, i]:
        return x, y
    k = int(radius / r) + 1
    j0, j1, i0, i1 = max(0, j - k), min(h, j + k + 1), max(0, i - k), min(w, i + k + 1)
    js, iis = np.nonzero(~blocked[j0:j1, i0:i1])
    if not len(js):
        return None
    wx = ox + (iis + i0 + 0.5) * r
    wy = oy + (js + j0 + 0.5) * r
    d = np.hypot(wx - x, wy - y)
    n = int(np.argmin(d))
    return (float(wx[n]), float(wy[n])) if d[n] <= radius else None


def plan(m, start, goal, clearance=0.6, step=0.5, snap_radius=0.0):
    """Waypoints [(x, y), ...] from start to goal (goal last; start not included), or None (no path)."""
    _, r, ox, oy = grid_of(m)
    blocked = blocked_mask(m, clearance)
    if snap_radius > 0:
        goal = snap_free(m, blocked, goal[0], goal[1], snap_radius)
        if goal is None:
            return None
    f = max(1, int(round(step / r)))                  # coarse cell = f x f fine cells, blocked if any is
    h, w = blocked.shape
    H, W = (h + f - 1) // f, (w + f - 1) // f
    pad = np.zeros((H * f, W * f), bool)
    pad[:h, :w] = blocked
    coarse = pad.reshape(H, f, W, f).any(axis=(1, 3))
    cs = r * f

    def cell(p):
        return (min(H - 1, max(0, int((p[1] - oy) / cs))), min(W - 1, max(0, int((p[0] - ox) / cs))))

    s, g = cell(start), cell(goal)
    if coarse[g]:
        # the goal is free on the fine grid but its coarse cell isn't: the nearest free coarse cell
        js, iis = np.nonzero(~coarse)
        if not len(js):
            return None
        n = int(np.argmin((js - g[0]) ** 2 + (iis - g[1]) ** 2))
        if math.hypot(js[n] - g[0], iis[n] - g[1]) * cs > max(snap_radius, 1.0):
            return None
        g = (int(js[n]), int(iis[n]))
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    hfun = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
    best = {s: 0.0}
    came = {}
    q = [(hfun(s), 0.0, s)]
    while q:
        _, cost, c = heapq.heappop(q)
        if c == g:
            break
        if cost > best.get(c, math.inf):
            continue
        for dj, di, dc in moves:
            n = (c[0] + dj, c[1] + di)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or coarse[n]:
                continue
            if dj and di and (coarse[c[0] + dj, c[1]] or coarse[c[0], c[1] + di]):
                continue                              # no corner cutting
            nc = cost + dc
            if nc < best.get(n, math.inf):
                best[n], came[n] = nc, c
                heapq.heappush(q, (nc + hfun(n), nc, n))
    else:
        return None
    if g != s and g not in came:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    pts = [(ox + (c[1] + 0.5) * cs, oy + (c[0] + 0.5) * cs) for c in cells]
    pts[0], pts[-1] = tuple(start), tuple(goal) if not coarse[cell(goal)] else pts[-1]

    def clear(a, b):                                  # segment a-b over free fine cells only
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (r * 0.5)) + 1
        for t in np.linspace(0.0, 1.0, n + 1):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            i, j = int((x - ox) / r), int((y - oy) / r)
            if 0 <= i < w and 0 <= j < h and blocked[j, i]:
                if math.hypot(x - start[0], y - start[1]) > 1.0:   # leaving a blocked start cell is allowed
                    return False
        return True

    out, i = [], 0                                    # string pulling: furthest visible point each time
    while i < len(pts) - 1:
        k = len(pts) - 1
        while k > i + 1 and not clear(pts[i], pts[k]):
            k -= 1
        out.append(pts[k])
        i = k
    return out or [pts[-1]]


def route_clear(m, start, wps, clearance=0.6):
    """True if start -> wps stays off the blocked cells (red zones found after the plan was made)."""
    a, r, ox, oy = grid_of(m)
    blocked = blocked_mask(m, clearance)
    h, w = blocked.shape
    prev = start
    for p in wps:
        n = int(math.hypot(p[0] - prev[0], p[1] - prev[1]) / (r * 0.5)) + 1
        for t in np.linspace(0.0, 1.0, n + 1):
            x, y = prev[0] + (p[0] - prev[0]) * t, prev[1] + (p[1] - prev[1]) * t
            i, j = int((x - ox) / r), int((y - oy) / r)
            if 0 <= i < w and 0 <= j < h and blocked[j, i] and math.hypot(x - start[0], y - start[1]) > 1.0:
                return False
        prev = p
    return True


class PathFlight:
    """Fly a waypoint list with a DirectFlight (one step per call)."""

    def __init__(self, flight, pose_fn):
        self.flight, self.pose = flight, pose_fn
        self.wps, self.alt, self.vmax, self.tol = [], None, 2.5, 0.5

    def start(self, wps, alt, vmax=2.5, tol=0.5):
        self.wps, self.alt, self.vmax, self.tol = list(wps), alt, vmax, tol

    def stop(self):
        if self.wps:
            self.flight.stop()
        self.wps = []

    @property
    def active(self):
        return bool(self.wps)

    def step(self):
        if not self.wps:
            return 'idle'
        px, py, _, yaw = self.pose()
        x, y = self.wps[0]
        d = math.hypot(x - px, y - py)
        if d < (self.tol if len(self.wps) == 1 else max(self.tol, 1.0)):
            self.wps.pop(0)
            if not self.wps:
                self.flight.stop()
                return 'arrived'
            x, y = self.wps[0]
            d = math.hypot(x - px, y - py)
        face = math.atan2(y - py, x - px) if d > 1.0 else yaw
        if len(self.wps) == 1:
            self.flight.step(x, y, self.alt, face, vmax=self.vmax)     # slows down (0.8 m/s per m) for the end
            return 'flying'
        # a corner: full speed through it (the next leg is in the same free space) - DirectFlight.step would
        # brake for every waypoint
        m = Twist()
        vx, vy = (x - px) / d * self.vmax, (y - py) / d * self.vmax
        m.linear.x = vx * math.cos(yaw) + vy * math.sin(yaw)          # odom -> body
        m.linear.y = -vx * math.sin(yaw) + vy * math.cos(yaw)
        m.angular.z = max(-0.6, min(0.6, 1.5 * wrap(face - yaw)))
        if self.alt is not None:
            self.flight.alt.publish(Float32(data=float(self.alt)))
        self.flight.cmd.publish(m)
        return 'flying'
