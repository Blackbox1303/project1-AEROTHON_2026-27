"""The QR area as a polygon in odom (user, 2026-10-28): the organisers' 4 points are either the QR area or the whole
arena. Arena only: the QR area's back wall and side walls are the arena's, its front wall is the entry tunnel's exit
line - known only in flight, when the drone leaves the entry tunnel (front_from_tunnel_exit).

/mission/qr_area (std_msgs/String JSON, latched): {polygon: [[x, y], ...], lock: bool, source}. The supervisor
publishes it; delivery_mapper / qr_locator / frontier_explorer take the zone from it, offboard_bridge keeps the drone
inside it while lock is true (from the search start until the exit-tunnel sequence)."""
import math

TOPIC = '/mission/qr_area'
try:                                                  # the ground apps (no ROS) use the geometry only
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
except ImportError:
    QOS = None


def inside(x, y, poly):
    res, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            res = not res
    return res


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def ccw(pts):
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def ray_entry(origin, yaw, poly, min_t=0.5):
    """Distance along the ray (origin, yaw) to the first crossing of the polygon's outline beyond min_t, or None."""
    ox, oy, dx, dy = origin[0], origin[1], math.cos(yaw), math.sin(yaw)
    best = None
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
        if t > min_t and -1e-6 <= u <= 1 + 1e-6 and (best is None or t < best):
            best = t
    return best


def _cross(p, d, a, b):
    """Point where the line (p, direction d) meets the line through a, b; None if parallel."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = d[0] * ey - d[1] * ex
    if abs(den) < 1e-9:
        return None
    t = ((a[0] - p[0]) * ey - (a[1] - p[1]) * ex) / den
    return p[0] + t * d[0], p[1] + t * d[1]


def front_from_tunnel_exit(arena, back, exit_pt, yaw):
    """The QR area from the arena: back = its two far corners (= the QR area's back corners), the side walls run from
    each to its other arena neighbour, the front wall is the line through the entry tunnel's exit point exit_pt
    across the tunnel heading yaw. -> 4 corners (ccw), or None if the geometry doesn't close."""
    arena = ccw([tuple(p) for p in arena])
    back = [tuple(p) for p in back]
    if len(arena) != 4 or len(back) != 2:
        return None
    idx = [min(range(4), key=lambda i: math.dist(arena[i], b)) for b in back]
    if idx[0] == idx[1] or (idx[0] - idx[1]) % 4 not in (1, 3):
        return None                                   # the back corners must be neighbours
    across = (-math.sin(yaw), math.cos(yaw))
    front = []
    for i, j in zip(idx, idx[::-1]):
        other = next(k for k in ((i + 1) % 4, (i - 1) % 4) if k != j)   # the side wall from back corner i
        q = _cross(exit_pt, across, arena[i], arena[other])
        if q is None:
            return None
        front.append(q)
    poly = ccw([arena[idx[0]], arena[idx[1]], *front])
    return poly if inside(exit_pt[0] + 2.0 * math.cos(yaw), exit_pt[1] + 2.0 * math.sin(yaw), poly) else None
