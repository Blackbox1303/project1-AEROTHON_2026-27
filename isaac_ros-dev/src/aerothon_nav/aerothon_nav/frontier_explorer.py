"""Frontier exploration of the delivery zone with Nav2 (namespace `delivery`), until the target QR is found.

Adapted from the aeolus frontier_explorer (which needs an external /map + AMCL): here the map is delivery_mapper's
/delivery/map (-1 unexplored, 0 explored, 100 red zone + margin / zone boundary) and the pose is PX4's (/nav/odom).
  frontier  explored cell 8-adjacent to an unexplored one, at least `clearance` from any 100 cell
  goal      per 8-connected frontier cluster (>= min_cluster cells) the cell with the most unexplored cells within
            `gain_radius` (what the camera would reveal there); score = gain / (distance + dist_bias); the best
            cluster's cell, minus blacklisted spots (goals Nav2 aborted)
  re-plan   when the goal is reached / aborted, when its gain has dropped below min_gain (seen on the way), or
            after goal_timeout
  sweep     (mode sweep, default) first: lanes across the zone's short side, lane_spacing apart, from the corridor
            end outward, one Nav2 goal per lane end (A* routes round red zones; an end inside a red / boundary margin
            moves along the lane to the nearest free cell; a lane with < min_gain unexplored cells is skipped). The
            greedy frontier goals at 10 ft (1.25 m read radius) needed 30+ min for the zone; the 15 min mission
            limit needs a straight coverage pass (2026-10-22). Then the frontier logic below fills the gaps
  candidates (mode candidates, the default - user, 2026-10-24: go to the next QR at least cost, not a lawnmower):
            qr_locator's candidates (boards seen from afar, not decoded), visited along the SHORTEST OPEN ROUTE from the
            drone through all of them to route_end (the exit approach / home: where the drone goes next anyway) -
            exact (Held-Karp) up to 9 candidates, nearest-neighbour + 2-opt beyond. With the end fixed the far ones come
            first and the near ones on the way back: no doubling back (user, 2026-10-24). Re-planned on arrival / new
            candidates, but a leg more than commit_frac done is finished first. Each leg is a Nav2 goal (A* round red
            zones / rejected QRs); arrived, hover cand_hover s for the decode (-> rejected / target), then it is done. Unreachable / undecodable ones are dropped. No candidate left: the frontier goals below reveal
            more. (mode sweep: the old coverage lanes first)
  confirm   before any sweep / frontier goal: a QR in the zone read but not yet confirmed (qr_locator's /delivery/qrs) is flown to
            and hovered over (the start payload first, then the nearest), up to confirm_visits arrivals - read once in
            passing, the target's cells counted as explored and the explorer never came back (2026-10-21 sim)
  escape    the drone's own cell no-fly (a red zone's margin, the boundary ring): Nav2 can't plan out of a lethal start
            cell - every goal failed and its recoveries drifted the drone to a red zone's edge (232 failed plans: the
            search began 2 m past the corridor, where the margin of a red zone moved next to the corridors started,
            2026-10-24 sim). Fly straight (DirectFlight) to the nearest cell escape_clear from any no-fly cell - away
            from the red zone, not across it - then go on
  stop      /delivery/target (qr_locator) -> cancel, TARGET_FOUND; no frontier for `done_after` checks -> DONE
  flight    use_nav2 false (default): every goal is an A* route on /delivery/map flown straight at `speed`
            (aerothon_nav/grid_path.py) instead of a delivery Nav2 goal - its MPPI, CPU-starved on the Jetson sim,
            flew the search at 0.7 m/s (2026-10-25). A goal in a no-fly margin (a candidate beside a red zone) moves
            to the nearest free cell within snap_radius: the camera reads it from there
  order     candidates first (seen boards, cheapest route), then coverage lanes over what is still unread
            (sweep_after: a board the candidate pass missed or projected wrong is still read - the target 1.5 m from
            the red zone was never read, 2026-10-25 sim), then frontiers for the gaps
Holds `altitude` on /nav/altitude while exploring. Services /delivery/explore/start, /delivery/explore/stop;
state on /delivery/explore/state (String).
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
import rclpy.parameter
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from aerothon_nav.direct_flight import DirectFlight, yaw_of
from aerothon_nav.grid_path import PathFlight, plan, route_clear
from vio_bringup.spin import spin_node

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        p = self.declare_parameter
        p('altitude', 10.0)          # m: = the supervisor's search_altitude
        p('clearance', 1.0)          # m from red / boundary cells (the map already has a 1.5 m red margin)
        p('edge_ignore', 1.0)        # m: unexplored cells this close to a red / boundary cell don't count - the drone
        #                              can't get there, so they never get explored and the frontiers beside them never
        #                              went away: it patrolled the zone edge for 15 min, 45 goals at x 16-18, while the
        #                              target sat 6 m in (2026-10-21 sim). The camera footprint still covers them
        p('gain_radius', 3.25)       # m: ground the camera can READ QRs in around the drone (= delivery_mapper's
        #                              read_angle at the search altitude: sim 18 deg at 10 m -> 3.25 m; the v3 QRs
        #                              are 1 m without a quiet zone - decoded >= 70 px/m, offline 2026-10-24)
        p('min_cluster', 4)          # cells
        p('dist_bias', 3.0)          # m
        p('min_gain', 20)            # unexplored cells left around the goal, else re-plan
        p('goal_timeout', 60.0)      # s (wall clock: the sim runs at 0.3-0.5x on the Jetson)
        p('blacklist_radius', 1.5)   # m
        p('blacklist_after', 2)      # aborted attempts at a spot before it is skipped
        p('done_after', 50)          # checks without a frontier before giving up
        p('confirm_visits', 4)       # arrivals over an unconfirmed QR before giving up on confirming it
        p('mode', 'candidates')      # candidates: seen boards first, frontiers otherwise | sweep: coverage lanes first,
        #                              then frontiers | frontier: frontiers only
        p('cand_hover', 1.5)         # s over a candidate for qr_locator to decode it (3 Hz: ~4 tries)
        p('cand_reach', 1.2)         # m: a candidate this close counts as reached
        p('route_end_x', float('nan'))   # odom: where the drone heads after the search (set by the supervisor)
        p('route_end_y', float('nan'))
        p('commit_frac', 0.5)        # a leg this far done is finished before switching to a new plan
        p('zone', [15.0, 55.0, -15.0, 15.0])   # x_min, x_max, y_min, y_max (as delivery_mapper)
        p('world_json', '')          # sim: the zone from the world file (as delivery_mapper / qr_locator)
        p('lane_spacing', 5.5)       # m: 2 x gain_radius (8.0 for the 1.5 m QRs at 22 deg) (10 m: the sim nadir_hd reads a whole QR to ~26 deg off
        #                              nadir, 22 used): 5 lanes over the 40 m zone. (2.4 m: 7.0; 5.0 = 8 lanes ran
        #                              out of the 15 min budget on lane 8, 2026-10-22)
        p('escape_clear', 0.8)       # m from any no-fly cell for the escape point (> Nav2's inscribed 0.61 m)
        p('escape_speed', 0.5)       # m/s
        p('lane_margin', 2.0)        # m inside the zone edge for the lane ends (1.5: corner ends aborted in 4 s)
        p('sweep_after', True)       # mode candidates: coverage lanes once no candidate is left, then frontiers
        p('stop_on_target', True)    # False (the QR-area-only mission, no drop): keep exploring past the target
        p('use_nav2', False)         # False: A* on /delivery/map + straight flight (grid_path), True: delivery Nav2
        p('speed', 1.5)              # m/s along the A* route (rules: <= 10 km/h; 2.5 flew past boards, user 2026-10-25)
        p('look_radius', 1.5)        # m: an undecoded candidate gets a square of this half-size flown round it, the
        #                              heading turning 90 deg per leg, before it is given up (the board is often a
        #                              metre or two off its candidate fix, user 2026-10-25: 'yaw a bit more'); 0 = off
        p('path_clearance', 0.6)     # m from red-margin / boundary cells for the route (the map's red margin is
        #                              already 1.5 m)
        p('snap_radius', 3.0)        # m: a candidate goal in a no-fly cell is read from the nearest free cell this near
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        if self.g('world_json'):
            z = json.load(open(self.g('world_json')))['delivery_zone']
            self.set_parameters([rclpy.parameter.Parameter('zone', value=[float(z['x_from']), float(z['x_to']),
                                                                           float(z['y_from']), float(z['y_to'])])])
        self.map = self.odom = None
        self.state = 'IDLE'
        self.goal = None             # (x, y, sent time)
        self.goal_handle = None
        self.goal_id = 0             # results of goals that were replaced (Nav2 preempts them) are ignored
        self.goal_done = True
        self.blacklist = []
        self.failures = []           # (x, y) of aborted goals
        self.retried = False         # blacklist cleared once already
        self.no_frontier = 0
        self.qrs, self.start_qr = [], ''
        self.cands, self.cand_done, self.cand_goal, self.cand_t, self.cand_leg = [], [], None, None, None
        self.cand_look = None        # look-around points left round the current candidate, None = not looking
        self.visits = {}             # unconfirmed QR key -> arrivals over it
        self.confirming = None       # key of the QR the current goal confirms
        self.sweep, self.sweep_i, self.sweep_goal = [], 0, None   # lane-end waypoints, next one, the one sent
        self.rejected = False
        self.last_status, self.sweep_tries = None, 0
        self.nav = ActionClient(self, NavigateToPose, '/delivery/navigate_to_pose')
        self.alt_pub = self.create_publisher(Float32, '/nav/altitude', 10)
        self.state_pub = self.create_publisher(String, '/delivery/explore/state', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/delivery/explore/goal', 10)
        self.create_subscription(OccupancyGrid, '/delivery/map', lambda m: setattr(self, 'map', m), LATCHED)
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(PoseStamped, '/delivery/target', self.on_target, LATCHED)
        self.create_subscription(String, '/delivery/qrs', self.on_qrs, 10)
        self.create_service(Trigger, '/delivery/explore/start', self.on_start)
        self.create_service(Trigger, '/delivery/explore/stop', self.on_stop)
        self.create_timer(1.0, self.tick)
        self.flight = DirectFlight(self, self.pose_tuple)
        self.escape_to = None
        self.create_timer(0.1, self.escape_step)
        self.path = PathFlight(self.flight, self.pose_tuple)

    # ---- control
    def set_state(self, s):
        if s != self.state:
            self.get_logger().info(s)
        self.state = s

    def on_start(self, req, res):
        self.cand_done, self.cand_goal, self.cand_t, self.cand_look = [], None, None, None
        self.blacklist, self.failures, self.retried = [], [], False
        self.visits, self.confirming = {}, None
        self.sweep, self.sweep_i, self.sweep_goal = self.make_lanes(), 0, None
        self.no_frontier, self.goal, self.goal_done = 0, None, True
        self.set_state('EXPLORING')
        res.success, res.message = True, 'exploring'
        return res

    def on_stop(self, req, res):
        self.cancel()
        self.set_state('IDLE')
        res.success, res.message = True, 'stopped'
        return res

    def on_qrs(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        self.qrs, self.start_qr = d.get('qrs', []), d.get('start_qr') or ''
        self.cands = d.get('candidates', [])

    def route(self):
        """Unvisited candidates in visiting order: the shortest open route drone -> all -> route_end."""
        p = self.odom.pose.pose.position
        pts = [(c['x'], c['y']) for c in self.cands if all(math.dist((c['x'], c['y']), d) > 1.5 for d in self.cand_done)]
        if len(pts) <= 1:
            return pts
        start = (p.x, p.y)
        ex, ey = self.g('route_end_x'), self.g('route_end_y')
        end = (ex, ey) if math.isfinite(ex) and math.isfinite(ey) else None
        d = lambda a, b: math.dist(a, b)  # noqa: E731
        n = len(pts)
        if n <= 9:                                    # Held-Karp: best[mask][j] = shortest start -> mask, ending at j
            best = {(1 << j, j): (d(start, pts[j]), None) for j in range(n)}
            for size in range(2, n + 1):
                for mask in range(1 << n):
                    if bin(mask).count('1') != size:
                        continue
                    for j in range(n):
                        if not mask & (1 << j):
                            continue
                        prev = mask ^ (1 << j)
                        best[(mask, j)] = min(((best[(prev, k)][0] + d(pts[k], pts[j]), k) for k in range(n)
                                               if prev & (1 << k) and (prev, k) in best), default=(math.inf, None))
            full = (1 << n) - 1
            j = min(range(n), key=lambda j: best[(full, j)][0] + (d(pts[j], end) if end else 0.0))
            order, mask = [], full
            while j is not None:
                order.append(j)
                mask, j = mask ^ (1 << j), best[(mask, j)][1]
            return [pts[i] for i in reversed(order)]
        order, left, cur = [], list(range(n)), start   # nearest neighbour ...
        while left:
            k = min(left, key=lambda i: d(cur, pts[i]))
            order.append(k)
            left.remove(k)
            cur = pts[k]
        seq = [start] + [pts[i] for i in order] + ([end] if end else [])
        better = True                                 # ... + 2-opt (start and end fixed)
        while better:
            better = False
            for i in range(1, len(seq) - 2):
                for k in range(i + 1, len(seq) - (1 if end else 0)):
                    a, b = seq[i - 1], seq[i]
                    c2 = seq[k]
                    nx = seq[k + 1] if k + 1 < len(seq) else None
                    old = d(a, b) + (d(c2, nx) if nx else 0.0)
                    new = d(a, c2) + (d(b, nx) if nx else 0.0)
                    if new < old - 1e-6:
                        seq[i:k + 1] = reversed(seq[i:k + 1])
                        better = True
        return seq[1:len(seq) - (1 if end else 0)]

    def next_candidate(self):
        """(x, y) of the first candidate on the route, or None."""
        r = self.route()
        return r[0] if r else None

    def candidate_step(self):
        """Fly to / hover over the nearest candidate; False when there is none (frontiers then)."""
        if self.g('mode') != 'candidates':
            return False
        p = self.odom.pose.pose.position
        if self.cand_goal is not None:
            gx, gy = self.cand_goal
            gone = all(math.dist((c['x'], c['y']), (gx, gy)) > 2.0 for c in self.cands)   # decoded: not a candidate
            if gone:
                read = any(math.dist((q['x'], q['y']), (gx, gy)) < 3.0 for q in self.qrs)
                self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}) ' +
                                       ('read' if read else 'dropped by qr_locator (no decode) - the lanes cover it'))
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t, self.cand_look = None, None, None
            elif self.cand_look is not None:
                # looking around: next corner of the square once the last one is reached (or failed)
                if self.goal_done:
                    if self.cand_look:
                        self.send(*self.cand_look.pop(0), snap=self.g('snap_radius'))
                    else:
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode after looking round - next')
                        self.cand_done.append((gx, gy))
                        self.cand_goal, self.cand_t, self.cand_look = None, None, None
                return True
            elif math.hypot(gx - p.x, gy - p.y) < self.g('cand_reach') or (self.goal_done and self.last_status == 4):
                self.cand_t = self.cand_t or time.monotonic()
                if time.monotonic() - self.cand_t > self.g('cand_hover'):
                    r = self.g('look_radius')
                    if r > 0:
                        # a square round the fix, starting from the drone's side: each leg turns the heading 90 deg
                        # (PathFlight faces the leg), so the camera sees the spot from four sides
                        a0 = math.atan2(p.y - gy, p.x - gx)
                        self.cand_look = [(gx + r * math.sqrt(2) * math.cos(a0 + math.pi / 4 + k * math.pi / 2),
                                           gy + r * math.sqrt(2) * math.sin(a0 + math.pi / 4 + k * math.pi / 2))
                                          for k in range(4)]
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode - looking round it')
                        self.goal_done = True
                    else:
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode after the hover - next')
                        self.cand_done.append((gx, gy))
                        self.cand_goal, self.cand_t = None, None
                return True
            elif self.goal_done and self.last_status not in (4, None):
                self.get_logger().warn(f'candidate at ({gx:.1f}, {gy:.1f}) unreachable (status {self.last_status})')
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t = None, None
            else:
                # on the way: switch only if the plan changed AND this leg is less than commit_frac done
                nxt = self.next_candidate()
                if nxt is not None and math.dist(nxt, (gx, gy)) > 1.5 and self.cand_leg:
                    done = 1.0 - math.hypot(gx - p.x, gy - p.y) / max(self.cand_leg, 1e-3)
                    if done < self.g('commit_frac'):
                        self.get_logger().info(f'route changed ({len(self.cands)} candidates): ({nxt[0]:.1f}, '
                                               f'{nxt[1]:.1f}) first ({done:.0%} of this leg done)')
                        self.cand_goal, self.cand_t = nxt, None
                        self.cand_leg = math.hypot(nxt[0] - p.x, nxt[1] - p.y)
                        self.last_status = None
                        self.send(*nxt, snap=self.g('snap_radius'))
                return True
        nxt = self.next_candidate()
        if nxt is None:
            return False
        self.cand_goal, self.cand_t = nxt, None
        self.cand_leg = math.hypot(nxt[0] - p.x, nxt[1] - p.y)
        self.get_logger().info(f'candidate QR at ({nxt[0]:.1f}, {nxt[1]:.1f}): going ({len(self.cands)} seen)')
        self.last_status = None
        self.sweep_goal = None                        # a lane leg was preempted: resend it afterwards
        self.send(*nxt, snap=self.g('snap_radius'))
        return True

    def unconfirmed(self):
        """(key, x, y) of the QR to confirm next, or None."""
        p = self.odom.pose.pose.position
        todo = []
        done = {q['payload'] for q in self.qrs if q.get('confirmed')}
        for q in self.qrs:
            key = (q['payload'], round(q['x']), round(q['y']))
            if q.get('confirmed') or not q.get('in_zone') or q.get('in_red_zone'):
                continue
            if self.start_qr and q['payload'] != self.start_qr:
                # one read gives the payload: only a possible TARGET is worth a detour and a hover (user, 2026-10-24:
                # no aligning over every QR). Others are rejected when read enough in passing - or never, harmlessly
                continue
            if q['payload'] in done:
                # a scattered re-read of a QR already confirmed (rejected) elsewhere: one board per payload. A
                # stray GOLF fix sent the drone 20 m back from lane 4 (2026-10-23 sim)
                continue
            if self.visits.get(key, 0) >= self.g('confirm_visits'):
                continue
            todo.append((q['payload'] != self.start_qr, math.hypot(q['x'] - p.x, q['y'] - p.y), key, q['x'], q['y']))
        if not todo:
            return None
        _, _, key, x, y = min(todo)
        return key, float(x), float(y)

    def make_lanes(self):
        """Lane-end waypoints: lanes along y (the zone's short side), x stepping outward from the corridor end,
        alternate directions, the first starting at the end nearer the drone."""
        if self.g('mode') != 'sweep' and not (self.g('mode') == 'candidates' and self.g('sweep_after')):
            return []
        x0, x1, y0, y1 = self.g('zone')
        sp, mg = self.g('lane_spacing'), self.g('lane_margin')
        ya, yb = y0 + mg, y1 - mg
        p = self.odom.pose.pose.position if self.odom is not None else None
        up = p is None or abs(p.y - ya) <= abs(p.y - yb)
        pts, x = [], x0 + sp / 2
        while x <= x1 - mg + 1e-6:
            pts += [(x, ya), (x, yb)] if up else [(x, yb), (x, ya)]
            up = not up
            x += sp
        self.get_logger().info(f'sweep: {len(pts) // 2} lanes {sp:.1f} m apart')
        return pts

    def free_on_lane(self, x, y, toward):
        """(x, y) moved along the lane (towards y = toward) until it is clear of red / boundary cells; None if no
        free cell before `toward`."""
        m = self.map
        r = m.info.resolution
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        dist = cv2.distanceTransform((a < 100).astype(np.uint8), cv2.DIST_L2, 3) * r
        step = 0.25 if toward > y else -0.25
        while (step > 0 and y <= toward) or (step < 0 and y >= toward):
            i, j = int((x - m.info.origin.position.x) / r), int((y - m.info.origin.position.y) / r)
            if 0 <= i < m.info.width and 0 <= j < m.info.height and dist[j, i] >= self.g('clearance'):
                return x, y
            y += step
        return None

    def lane_unknown(self, x, ya, yb):
        """Unexplored (reachable) cells in the lane's strip."""
        m = self.map
        r = m.info.resolution
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        dist = cv2.distanceTransform((a < 100).astype(np.uint8), cv2.DIST_L2, 3) * r
        unknown = (a < 0) & (dist > self.g('edge_ignore'))
        h = self.g('lane_spacing') / 2
        i0 = max(0, int((x - h - m.info.origin.position.x) / r))
        i1 = min(m.info.width, int((x + h - m.info.origin.position.x) / r) + 1)
        j0 = max(0, int((min(ya, yb) - m.info.origin.position.y) / r))
        j1 = min(m.info.height, int((max(ya, yb) - m.info.origin.position.y) / r) + 1)
        return int(unknown[j0:j1, i0:i1].sum())

    def sweep_step(self):
        """Drive the sweep; False once it is finished (the frontier logic takes over)."""
        while self.sweep_i < len(self.sweep):
            i = self.sweep_i
            if self.sweep_goal == i:
                if not self.goal_done:
                    if time.monotonic() - self.goal[2] > self.g('goal_timeout'):
                        self.cancel()
                        self.sweep_i += 1
                        continue
                    return True                       # flying the lane leg
                if self.rejected:                     # server busy: send it again
                    self.sweep_goal = None
                    return True
                if self.last_status not in (4, None) and self.sweep_tries < 3:
                    # aborted (no plan, e.g. from behind the zone's boundary ring): try again, don't skip the lane -
                    # two lanes were lost that way (2026-10-22 sim)
                    self.sweep_tries += 1
                    self.sweep_goal = None
                    return True
                self.sweep_i += 1                     # reached (or given up): next lane end
                self.sweep_tries = 0
                continue
            x, y = self.sweep[i]
            other = self.sweep[i + 1][1] if i % 2 == 0 else self.sweep[i - 1][1]
            if i % 2 == 0 and self.lane_unknown(x, y, other) < self.g('min_gain'):
                self.get_logger().info(f'sweep lane x={x:.1f}: already seen, skipped')
                self.sweep_i += 2
                continue
            pt = self.free_on_lane(x, y, other)
            if pt is None:
                self.sweep_i += 1
                continue
            self.get_logger().info(f'sweep lane {i // 2 + 1}/{len(self.sweep) // 2}: to ({pt[0]:.1f}, {pt[1]:.1f})')
            self.sweep_goal = i
            self.send(*pt)
            return True
        if self.sweep:
            self.get_logger().info('sweep done: frontiers for the gaps')
            self.sweep, self.sweep_goal = [], None
        return False

    def on_target(self, m):
        if self.state == 'EXPLORING' and self.g('stop_on_target'):
            self.cancel()
            self.set_state(f'TARGET_FOUND at ({m.pose.position.x:.2f}, {m.pose.position.y:.2f})')

    def cancel(self):
        if self.goal_handle is not None and not self.goal_done and self.g('use_nav2'):
            self.goal_handle.cancel_goal_async()
        self.path.stop()
        self.goal_done = True

    # ---- frontiers
    def choose(self):
        m = self.map
        w, h, r = m.info.width, m.info.height, m.info.resolution
        a = np.array(m.data, np.int8).reshape(h, w)
        free = a == 0
        blocked = (a >= 100).astype(np.uint8)
        dist = cv2.distanceTransform((1 - blocked).astype(np.uint8), cv2.DIST_L2, 3) * r
        unknown = ((a < 0) & (dist > self.g('edge_ignore'))).astype(np.uint8)
        near_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8)).astype(bool)
        frontier = free & near_unknown & (dist >= self.g('clearance'))
        if not frontier.any():
            return None
        k = int(self.g('gain_radius') / r)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)).astype(np.float32)
        gain = cv2.filter2D(unknown.astype(np.float32), -1, disk, borderType=cv2.BORDER_CONSTANT)
        n, labels = cv2.connectedComponents(frontier.astype(np.uint8), connectivity=8)
        p = self.odom.pose.pose.position
        best = None
        for c in range(1, n):
            ys, xs = np.nonzero(labels == c)
            if len(xs) < self.g('min_cluster'):
                continue
            wx = m.info.origin.position.x + (xs + 0.5) * r
            wy = m.info.origin.position.y + (ys + 0.5) * r
            ok = np.ones(len(xs), bool)
            for bx, by in self.blacklist:
                ok &= np.hypot(wx - bx, wy - by) > self.g('blacklist_radius')
            if not ok.any():
                continue
            gi = gain[ys, xs] * ok
            i = int(np.argmax(gi))
            score = gi[i] / (math.hypot(wx[i] - p.x, wy[i] - p.y) + self.g('dist_bias'))
            if best is None or score > best[0]:
                best = (score, float(wx[i]), float(wy[i]), float(gi[i]))
        return best

    def cell_at(self, x, y):
        """/delivery/map value at (x, y): -1 unexplored, 0 free, > 0 rejected QR / red / boundary; 0 off the map."""
        m = self.map
        i = int((x - m.info.origin.position.x) / m.info.resolution)
        j = int((y - m.info.origin.position.y) / m.info.resolution)
        if 0 <= i < m.info.width and 0 <= j < m.info.height:
            return m.data[j * m.info.width + i]
        return 0

    def gain_at(self, x, y):
        m = self.map
        r = m.info.resolution
        i, j = int((x - m.info.origin.position.x) / r), int((y - m.info.origin.position.y) / r)
        k = int(self.g('gain_radius') / r)
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        dist = cv2.distanceTransform((a < 100).astype(np.uint8), cv2.DIST_L2, 3) * r
        win = (a < 0) & (dist > self.g('edge_ignore'))
        return int(win[max(0, j - k):j + k + 1, max(0, i - k):i + k + 1].sum())

    def send(self, x, y, snap=0.0):
        p = self.odom.pose.pose.position
        if not self.g('use_nav2'):
            self.goal_id += 1
            self.goal = (x, y, time.monotonic())
            self.rejected = False
            wps = plan(self.map, (p.x, p.y), (x, y), self.g('path_clearance'), 0.5, snap)
            if wps is None:
                self.get_logger().warn(f'no A* route to ({x:.1f}, {y:.1f})')
                self.path.stop()
                self.last_status, self.goal_done = 6, True        # as a Nav2 abort
                if self.state == 'EXPLORING' and self.sweep_goal is None:
                    self.failed((x, y), 6)
                return
            self.last_status, self.goal_done = None, False
            self.path.start(wps, float(self.g('altitude')), vmax=self.g('speed'))
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y = wps[-1]
            pose.pose.orientation.w = 1.0
            self.goal_pub.publish(pose)
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        yaw = math.atan2(y - p.y, x - p.x)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(goal.pose)
        self.goal, self.goal_done = (x, y, time.monotonic()), False
        self.goal_id += 1
        gid = self.goal_id
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(lambda f: self.on_goal_response(f, gid, (x, y)))

    def on_goal_response(self, fut, gid, xy):
        h = fut.result()
        if gid != self.goal_id:
            return
        if not h.accepted:
            # the server's problem, not the spot's (e.g. the delivery Nav2 still resuming from pause): retry next
            # tick, don't count it towards the blacklist
            self.get_logger().warn('Nav2 rejected the goal: retrying', throttle_duration_sec=5.0)
            self.goal_done, self.rejected = True, True
            return
        self.rejected = False
        self.goal_handle = h
        h.get_result_async().add_done_callback(lambda f: self.on_result(f, gid, xy))

    def on_result(self, fut, gid, xy):
        # a goal replaced by a newer one ends ABORTED (Nav2 preempts it): not a failure of either goal - blaming
        # the current goal for it blacklisted 46 good frontiers in one run (2026-10-19)
        if gid != self.goal_id:
            return
        status = fut.result().status
        self.last_status = status
        if status not in (4, 5) and self.state == 'EXPLORING' and self.sweep_goal is None:  # 4 SUCCEEDED, 5 CANCELED
            self.failed(xy, status)
        self.goal_done = True

    def failed(self, xy, status=None):
        self.failures.append(xy)
        n = sum(1 for f in self.failures if math.dist(f, xy) < self.g('blacklist_radius'))
        if n >= self.g('blacklist_after'):
            self.blacklist.append(xy)
            self.get_logger().warn(f'frontier goal ({xy[0]:.1f}, {xy[1]:.1f}) ended {status} ({n} x): blacklisted')
        else:
            self.get_logger().warn(f'frontier goal ({xy[0]:.1f}, {xy[1]:.1f}) ended {status}: will retry')

    def pose_tuple(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, p.position.z, yaw_of(p.orientation)

    def escape_point(self):
        """(x, y) to fly straight to when the drone's cell is no-fly, else None."""
        m = self.map
        r = m.info.resolution
        p = self.odom.pose.pose.position
        if self.cell_at(p.x, p.y) < 100:
            return None
        a = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        dist = cv2.distanceTransform((a < 100).astype(np.uint8), cv2.DIST_L2, 3) * r
        js, iis = np.nonzero(dist >= self.g('escape_clear'))
        if not len(js):
            return None
        wx = m.info.origin.position.x + (iis + 0.5) * r
        wy = m.info.origin.position.y + (js + 0.5) * r
        k = int(np.argmin(np.hypot(wx - p.x, wy - p.y)))
        return float(wx[k]), float(wy[k])

    def escape_step(self):
        if self.odom is None:
            return
        if self.escape_to is None:
            if self.path.active:
                if self.state != 'EXPLORING':
                    self.path.stop()
                elif self.path.step() == 'arrived':
                    self.last_status, self.goal_done = 4, True
            return
        x, y = self.escape_to
        px, py, _, yaw = self.pose_tuple()
        if math.hypot(x - px, y - py) < 0.3 or self.state != 'EXPLORING':
            self.flight.stop()
            self.get_logger().info(f'escaped the no-fly cells at ({px:.1f}, {py:.1f})')
            self.escape_to = None
            return
        self.flight.step(x, y, float(self.g('altitude')), yaw, vmax=self.g('escape_speed'))

    def tick(self):
        self.state_pub.publish(String(data=self.state))
        if self.state != 'EXPLORING':
            return
        self.alt_pub.publish(Float32(data=float(self.g('altitude'))))
        if self.map is None or self.odom is None:
            return
        if self.escape_to is not None:
            return                                    # flying out of the no-fly cells (escape_step)
        esc = self.escape_point()
        if esc is not None:
            self.cancel()
            self.sweep_goal = None                    # resend the lane leg afterwards
            self.escape_to = esc
            p = self.odom.pose.pose.position
            self.get_logger().warn(f'in a no-fly cell at ({p.x:.1f}, {p.y:.1f}): straight out to ({esc[0]:.1f}, '
                                   f'{esc[1]:.1f}) first')
            return
        if self.path.active and not route_clear(self.map, (self.odom.pose.pose.position.x,
                                                           self.odom.pose.pose.position.y), self.path.wps,
                                                self.g('path_clearance')):
            gx, gy, _ = self.goal
            self.get_logger().info(f'route to ({gx:.1f}, {gy:.1f}) now crosses a no-fly cell: re-routing')
            sweep_goal = self.sweep_goal
            self.send(gx, gy, snap=self.g('snap_radius'))
            self.sweep_goal = sweep_goal
            return
        if self.g('use_nav2') and not self.nav.server_is_ready():
            self.get_logger().warn('/delivery/navigate_to_pose not available', throttle_duration_sec=5.0)
            return
        # an unconfirmed QR first: fly over it and hover (one goal per tick while there) until qr_locator confirms it
        c = self.unconfirmed()
        if c is not None:
            key, x, y = c
            if self.confirming == key and self.goal_done:
                self.visits[key] = self.visits.get(key, 0) + 1          # arrived (or gave up): one visit
            if self.confirming != key or self.goal_done:
                if self.confirming != key:
                    self.get_logger().info(f'confirming QR {key[0]!r} at ({x:.1f}, {y:.1f})')
                self.confirming = key
                self.sweep_goal = None                # the sweep leg was preempted: resend it afterwards
                self.send(x, y, snap=self.g('snap_radius'))
            return
        self.confirming = None
        if self.candidate_step():
            return
        if self.sweep_step():
            return
        replan = self.goal_done or self.goal is None
        if not replan:
            gx, gy, t0 = self.goal
            if self.gain_at(gx, gy) < self.g('min_gain') or time.monotonic() - t0 > self.g('goal_timeout'):
                replan = True
            elif self.cell_at(gx, gy) > 0:
                # the goal's cell became a rejected QR / red zone after it was chosen (the map lags a decode by up
                # to 1 s: a goal sent the second GOLF was rejected took the drone right over it, 2026-10-20 sim)
                self.get_logger().info(f'goal ({gx:.1f}, {gy:.1f}) is no longer free: re-plan')
                replan = True
        if not replan:
            return
        best = self.choose()
        if best is None:
            self.no_frontier += 1
            if self.no_frontier >= self.g('done_after') and self.blacklist and not self.retried:
                self.get_logger().warn(f'no frontier left: retrying {len(self.blacklist)} blacklisted spots once')
                self.blacklist, self.failures, self.retried, self.no_frontier = [], [], True, 0
                return
            if self.no_frontier >= self.g('done_after'):
                self.cancel()
                self.set_state('DONE: no frontier left, target not found')
            return
        self.no_frontier = 0
        _, x, y, gain = best
        if self.goal and math.hypot(x - self.goal[0], y - self.goal[1]) < 0.5 and not self.goal_done:
            return                                    # same goal, keep flying
        self.get_logger().info(f'frontier goal ({x:.1f}, {y:.1f}), gain {gain:.0f} cells')
        self.send(x, y)


def main():
    spin_node(FrontierExplorer)


if __name__ == '__main__':
    main()
