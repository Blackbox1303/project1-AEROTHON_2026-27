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
            qr_locator's candidates (boards seen from afar, not decoded) in BATCHES (user, 2026-10-28): the boards first
            seen together (within batch_gather s of the batch being started) are one batch; boards seen later - on the
            way, or from a rejected board's yaw scan - go into the next batch. A batch is finished before the next is
            started (earlier sightings always first); inside it the cheapest A* route goes first, re-costed from where
            the drone is at each pick (a leg less than commit_frac done may switch to a cheaper one of the SAME batch).
            Arrived, hover cand_hover s for the decode (-> rejected / target); no decode: look round it. Unreachable /
            undecodable ones are dropped. No candidate left: the lanes / frontier goals below reveal more
  yaw scan  a candidate read as a REJECT: turn in place to 3 more headings (90 deg apart) for boards the fisheye's
            narrow axis missed - they join the next batch. Not again within spin_skip of an earlier scan. Also a
            reject read in passing (on a lane / scout / frontier leg, within spin_near of the drone; spin_passing):
            the leg is resent after the scan
  scout     the search starts with no board in view (none after scout_wait s): out to the line scout_offset (at most
            a quarter of the zone's short side) in from the nearer long wall (the zone's y edges) and along it to the far end - the most zone in the fisheye's
            view; dropped as soon as a board is seen
  no retrace every explorer route (A*) costs retrace_cost extra per cell already flown over (trail_radius round the
            track), while a frontier is left: fresh ground on the way, old ground only when nothing is left to explore.
            The way out after the search is the supervisor's plain shortest A* (fly_route)
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
Holds `altitude` + `alt_offset` on /nav/altitude while exploring: the operator's in-flight height trim (dashboard
+-0.5 m steps, e.g. down when the boards don't read from up there - user, 2026-10-28), applied at once, also to the
route being flown; never below `alt_min`; back to 0 at the next search start. Services /delivery/explore/start, /delivery/explore/stop;
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

from aerothon_nav import qr_area
from aerothon_nav.direct_flight import DirectFlight, wrap, yaw_of
from aerothon_nav.grid_path import PathFlight, path_length, plan, route_clear
from vio_bringup.spin import spin_node

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        p = self.declare_parameter
        p('altitude', 9.0)           # m: = the supervisor's search_altitude (under the 10 m airspace limit)
        p('alt_offset', 0.0)         # m: the operator's in-flight trim (dashboard / radio), + up / - down
        p('alt_max', 9.5)            # m: never above 10 m - the airspace rule there (user, 2026-10-30)
        p('alt_min', 5.5)            # m: the trimmed search height never goes below this. QR area: 5 m is the DROP
        #                              height only - scanning (search, trim, deep read) stays above it (user, 2026-10-30)
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
        # DEEP READ (user, 2026-10-28): a candidate still unread after the hover -> qr_locator's deep read (decoder
        # settings x verify_rounds frames, then the image vs the start QR's); 'unsure' / 'unseen' -> verify_step m
        # lower (the height trim, at once) and again, not below verify_floor (just above the drop height: 5 m is
        # reached only to lower the payload, user 2026-10-30); the trim is undone
        # when the candidate is settled. Only when that is exhausted: the look-round square below
        p('deep_read', True)
        p('verify_timeout', 30.0)    # s (wall) for one deep read answer
        p('verify_step', 0.5)        # m down per try
        p('verify_floor', 5.5)       # m: no lower than this (the supervisor's deliver_altitude 5 m is for the drop only)
        p('cand_reach', 1.2)         # m: a candidate this close counts as reached
        p('cand_verify_max', 12.0)   # s (sim / ROS time) from arriving over a candidate: hover + look-round, then it
        #                              is dropped - one false board (tunnel banner) cost 65 s of a 330 s search (2026-10-27)
        # negative evidence / decay (planner side - qr_locator only reports sightings): a candidate whose spot the
        # readable footprint (gain_radius) passed over cand_miss_passes times without a new sighting, or that has had
        # no sighting for cand_stale_s (sim / ROS time), is ignored - a one-off blob on grainy video stays a detour
        # target forever otherwise
        p('cand_miss_passes', 2)
        p('cand_stale_s', 60.0)
        p('lane_row_frac', 0.25)     # share of the lane strip's width unexplored that makes a lane row worth flying
        #                              (2 cells kept slivers alive: the drone flew 25 m back for 4 m of lane 1, 2026-10-27)
        p('route_end_x', float('nan'))   # odom: where the drone heads after the search (set by the supervisor; the
        p('route_end_y', float('nan'))   # batch order replaced the shortest route to it, 2026-10-28)
        p('commit_frac', 0.5)        # a leg this far done is finished before switching to a cheaper one of its batch
        p('batch_gather', 2.0)       # s (ROS time): boards first seen this soon after a batch was started join it (the
        #                              same view: qr_locator needs cand_min_hits before it reports one)
        p('spin_reject', True)       # yaw scan over a candidate read as a reject
        p('spin_dwell', 1.0)         # s (ROS time) at each heading of the scan
        p('spin_max', 20.0)          # s (ROS time) for the whole scan
        p('spin_skip', 4.0)          # m: no second scan this near an earlier one
        p('spin_passing', True)      # also a yaw scan at a reject read in passing (not flown to as a candidate)
        p('spin_near', 6.5)          # m: only when that board is this near the drone (2 x gain_radius)
        p('scout_offset', 7.5)       # m in from the long wall for the start line (0 = no scout)
        p('scout_wait', 2.0)         # s (ROS time) at the search start for a board to show up before scouting
        p('retrace_cost', 3.0)       # extra A* cost per cell already flown over (0 = off)
        p('trail_radius', 1.0)       # m round the track counted as flown over
        p('zone', [15.0, 55.0, -15.0, 15.0])   # x_min, x_max, y_min, y_max (as delivery_mapper)
        p('world_json', '')          # sim: the zone from the world file (as delivery_mapper / qr_locator)
        p('lane_spacing', 5.5)       # m: 2 x gain_radius (8.0 for the 1.5 m QRs at 22 deg) (10 m: the sim nadir_hd reads a whole QR to ~26 deg off
        #                              nadir, 22 used): 5 lanes over the 40 m zone. (2.4 m: 7.0; 5.0 = 8 lanes ran
        #                              out of the 15 min budget on lane 8, 2026-10-22)
        p('escape_clear', 0.8)       # m from any no-fly cell for the escape point (> Nav2's inscribed 0.61 m)
        p('escape_speed', 0.5)       # m/s
        p('lane_margin', 2.0)        # m inside the zone edge for the lane ends (1.5: corner ends aborted in 4 s)
        p('sweep_after', True)       # mode candidates: coverage lanes once no candidate is left, then frontiers
        p('use_nav2', False)         # False: A* on /delivery/map + straight flight (grid_path), True: delivery Nav2
        p('speed', 1.5)              # m/s along the A* route (rules: <= 10 km/h; 2.5 flew past boards, user 2026-10-25)
        p('look_radius', 1.5)        # m: an undecoded candidate gets a square of this half-size flown round it, the
        #                              heading turning 90 deg per leg, before it is given up (the board is often a
        #                              metre or two off its candidate fix, user 2026-10-25: 'yaw a bit more'); 0 = off
        p('path_clearance', 0.6)     # m from red-margin / boundary cells for the route (the map's red margin is
        #                              already 1.5 m)
        p('snap_radius', 3.0)        # m: a candidate goal in a no-fly cell is read from the nearest free cell this near
        # no candidate detours near the tunnels' zone-side ends: the banner / roof edge projects onto the ground just
        # inside the zone and drew the search there twice per run, ~110 s (2026-10-27 sim). Sim: the ends from
        # world_json; real flight: tunnel_end [x, y] set by the supervisor once the entry corridor is measured
        p('tunnel_clear', 3.5)       # m round each tunnel end
        p('tunnel_end', [0.0])       # [x, y, ...] extra tunnel ends (odom); [0] = none
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.tunnel_ends = []
        if self.g('world_json'):
            w = json.load(open(self.g('world_json')))
            z = w['delivery_zone']
            self.set_parameters([rclpy.parameter.Parameter('zone', value=[float(z['x_from']), float(z['x_to']),
                                                                           float(z['y_from']), float(z['y_to'])])])
            zc = ((z['x_from'] + z['x_to']) / 2, (z['y_from'] + z['y_to']) / 2)
            for c in w.get('corridors') or []:
                ends = [(c['x_from'], c['centre_y']), (c['x_to'], c['centre_y'])]
                self.tunnel_ends.append(min(ends, key=lambda e: math.dist(e, zc)))
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
        self.cand_arrive = None      # ROS time (s) the drone arrived over the current candidate
        self.cand_ev = []            # per candidate spot: {x, y, n, t_hit, inside, n_enter, misses, dead}
        self.visits = {}             # unconfirmed QR key -> arrivals over it
        self.confirming = None       # key of the QR the current goal confirms
        self.sweep, self.sweep_i, self.sweep_goal = [], 0, None   # lane-end waypoints, next one, the one sent
        self.rejected = False
        self.last_status, self.sweep_tries = None, 0
        self.open_batch, self.batch_t = 0, None      # batch new candidate spots join; ROS time the last one started
        self.spin, self.spins = None, []             # yaw scan in progress {x, y, yaws, t0, t_at}; where scans were
        self.rejects, self.spin_due = set(), None    # reject payloads seen; a passing reject's board to scan at
        self.scout, self.scout_goal, self.t_start = None, False, 0.0
        self.trail = None                            # bool, /delivery/map shape: cells flown over
        self.has_frontier = True
        self.nav = ActionClient(self, NavigateToPose, '/delivery/navigate_to_pose')
        self.alt_pub = self.create_publisher(Float32, '/nav/altitude', 10)
        self.state_pub = self.create_publisher(String, '/delivery/explore/state', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/delivery/explore/goal', 10)
        self.create_subscription(OccupancyGrid, '/delivery/map', lambda m: setattr(self, 'map', m), LATCHED)
        self.red = None              # /delivery/red: 100 = inside a red zone (no margin)
        self.create_subscription(OccupancyGrid, '/delivery/red', lambda m: setattr(self, 'red', m), LATCHED)
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(PoseStamped, '/delivery/target', self.on_target, LATCHED)
        self.create_subscription(String, '/delivery/qrs', self.on_qrs, 10)
        # the QR area cut at the entry tunnel's exit (supervisor): the lanes span its bounding box; the cells outside the
        # polygon are delivery_mapper's (no unexplored ground there, so no lane runs out of it)
        self.create_subscription(String, qr_area.TOPIC, lambda m: len(json.loads(m.data).get('polygon') or []) >= 3 and
                                 self.set_parameters([rclpy.parameter.Parameter('zone', value=[
                                     float(v) for v in qr_area.bbox(json.loads(m.data)['polygon'])])]), qr_area.QOS)
        self.verify_pub = self.create_publisher(String, '/delivery/verify/request', 10)
        self.verify_res = {}         # deep read id -> qr_locator's answer
        self.create_subscription(String, '/delivery/verify/result', lambda m: self.verify_res.__setitem__(
            json.loads(m.data)['id'], json.loads(m.data)), 10)
        self.verify_id, self.verify_n, self.verify_t, self.verify_hold, self.trim_before = None, 0, 0.0, 0.0, None
        self.create_service(Trigger, '/delivery/explore/start', self.on_start)
        self.create_service(Trigger, '/delivery/explore/stop', self.on_stop)
        self.create_timer(1.0, self.tick)
        self.flight = DirectFlight(self, self.pose_tuple)
        self.escape_to = None
        self.create_timer(0.1, self.escape_step)
        self.path = PathFlight(self.flight, self.pose_tuple)
        self.add_on_set_parameters_callback(self.on_params)

    # ---- control
    def set_state(self, s):
        if s != self.state:
            self.get_logger().info(s)
        self.state = s

    def on_start(self, req, res):
        if self.g('alt_offset'):
            self.set_parameters([rclpy.parameter.Parameter('alt_offset', value=0.0)])   # a new search: no trim
        self.cand_done, self.cand_goal, self.cand_t, self.cand_look = [], None, None, None
        self.cand_ev, self.open_batch, self.batch_t = [], 0, None
        self.verify_id, self.verify_hold, self.trim_before = None, 0.0, None
        self.spin, self.spins = None, []
        self.rejects, self.spin_due = set(), None
        self.scout, self.scout_goal, self.t_start = None, False, self.now_s()
        if self.g('scout_offset') > 0 and self.g('mode') == 'candidates':
            self.scout = []                           # waypoints made at scout_wait (the pose then)
        self.trail, self.has_frontier = None, True
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
        now = self.now_s()
        for c in self.cands:                          # new sightings: qr_locator's running count n went up
            ev = next((e for e in self.cand_ev if math.dist((e['x'], e['y']), (c['x'], c['y'])) < 1.5), None)
            if ev is None:
                gather = self.batch_t is not None and now - self.batch_t < self.g('batch_gather')
                batch = self.open_batch - 1 if gather else self.open_batch
                self.cand_ev.append({'x': c['x'], 'y': c['y'], 'n': c['n'], 't_hit': now, 'inside': False,
                                     'n_enter': c['n'], 'misses': 0, 'dead': False, 'batch': batch})
                if self.state == 'EXPLORING':
                    self.get_logger().info(f'candidate at ({c["x"]:.1f}, {c["y"]:.1f}) seen: batch {batch}')
            elif c['n'] > ev['n']:
                ev.update(x=c['x'], y=c['y'], n=c['n'], t_hit=now)
        for q in self.qrs:                            # a new reject read anywhere: the scan (tick) unless a candidate
            #                                           leg is on - its arrival scans
            if (not self.start_qr or q['payload'] == self.start_qr or q['payload'] in self.rejects
                    or not q.get('in_zone')):
                continue
            self.rejects.add(q['payload'])
            if (self.state == 'EXPLORING' and self.g('spin_passing') and self.cand_goal is None
                    and self.spin is None and self.odom is not None):
                p = self.odom.pose.pose.position
                if math.hypot(q['x'] - p.x, q['y'] - p.y) < self.g('spin_near'):
                    self.spin_due = (q['x'], q['y'])

    def cand_evidence(self):
        """Each tick: count footprint passes over every candidate spot without a new sighting; mark the dead ones."""
        p = self.odom.pose.pose.position
        now = self.now_s()
        for e in self.cand_ev:
            if e['dead']:
                continue
            inside = math.hypot(e['x'] - p.x, e['y'] - p.y) < self.g('gain_radius')
            if inside and not e['inside']:
                e['n_enter'] = e['n']
            elif e['inside'] and not inside and e['n'] == e['n_enter']:
                e['misses'] += 1
            e['inside'] = inside
            why = (f'passed over {e["misses"]} times without a sighting' if e['misses'] >= self.g('cand_miss_passes')
                   else f'no sighting for {now - e["t_hit"]:.0f} s' if now - e['t_hit'] > self.g('cand_stale_s') else '')
            if why:
                e['dead'] = True
                self.get_logger().info(f'candidate at ({e["x"]:.1f}, {e["y"]:.1f}) ignored: {why}')

    def near_tunnel(self, x, y):
        t = list(self.g('tunnel_end'))
        ends = self.tunnel_ends + [(t[k], t[k + 1]) for k in range(0, len(t) - 1, 2)]
        return any(math.dist(e, (x, y)) < self.g('tunnel_clear') for e in ends)

    def in_red(self, x, y):
        """Inside a red zone: never a QR there (the course rules) - a red-zone "board" drew the drone 22 m off its
        lane and round the no-fly margin for 100 s (2026-10-27 sim)."""
        m = self.red
        if m is None:
            return False
        i = int((x - m.info.origin.position.x) / m.info.resolution)
        j = int((y - m.info.origin.position.y) / m.info.resolution)
        return 0 <= i < m.info.width and 0 <= j < m.info.height and m.data[j * m.info.width + i] >= 100

    def cand_dead(self, x, y):
        return any(e['dead'] and math.dist((e['x'], e['y']), (x, y)) < 1.5 for e in self.cand_ev)

    def pending(self):
        """Candidate spots still to visit: qr_locator still reports them (not decoded), not done / dead / in the red
        zone / at a tunnel end."""
        out = []
        for e in self.cand_ev:
            xy = (e['x'], e['y'])
            if (e['dead'] or any(math.dist(xy, d) <= 1.5 for d in self.cand_done)
                    or not any(math.dist((c['x'], c['y']), xy) < 1.5 for c in self.cands)
                    or self.in_red(*xy) or self.near_tunnel(*xy)):
                continue
            out.append(e)
        return out

    def route_cost(self, x, y):
        """Length (m) of the explorer's A* route from the drone to (x, y); inf when there is none."""
        p = self.odom.pose.pose.position
        wps = self.plan_to(x, y, self.g('snap_radius'))
        return path_length((p.x, p.y), wps) if wps else math.inf

    def next_candidate(self, commit=True):
        """(x, y) of the next candidate: the cheapest route of the earliest batch with any left, or None. commit:
        that batch is started - boards seen from now on (after batch_gather) go into the next one."""
        todo = self.pending()
        if not todo:
            return None
        b = min(e['batch'] for e in todo)
        todo = [e for e in todo if e['batch'] == b]
        e = todo[0] if len(todo) == 1 else min(todo, key=lambda e: self.route_cost(e['x'], e['y']))
        if commit and b >= self.open_batch:
            self.open_batch, self.batch_t = b + 1, self.now_s()
            self.get_logger().info(f'batch {b}: {len(todo)} candidate(s), cheapest route first')
        return e['x'], e['y']

    def cand_batch(self, x, y):
        e = next((e for e in self.cand_ev if math.dist((e['x'], e['y']), (x, y)) < 1.5), None)
        return e['batch'] if e else None

    def start_spin(self, x, y):
        """Yaw scan over a rejected board: 3 more headings, 90 deg apart."""
        if not self.g('spin_reject') or any(math.dist(s, (x, y)) < self.g('spin_skip') for s in self.spins):
            return
        self.cancel()
        p = self.odom.pose.pose.position
        yaw = self.pose_tuple()[3]
        self.spins.append((x, y))
        self.spin = {'x': p.x, 'y': p.y, 'yaws': [yaw + k * math.pi / 2 for k in (1, 2, 3)], 't0': self.now_s(),
                     't_at': None}
        self.get_logger().info(f'reject at ({x:.1f}, {y:.1f}): yaw scan for more boards')

    def spin_step(self):
        """10 Hz: hold the scan's spot, turn to the next heading, dwell, next."""
        s = self.spin
        now = self.now_s()
        _, _, _, yaw = self.pose_tuple()
        if s['yaws'] and abs(wrap(s['yaws'][0] - yaw)) < 0.1:
            s['t_at'] = s['t_at'] or now
            if now - s['t_at'] > self.g('spin_dwell'):
                s['yaws'].pop(0)
                s['t_at'] = None
        if not s['yaws'] or now - s['t0'] > self.g('spin_max'):
            self.flight.stop()
            new = sum(1 for e in self.pending() if e['batch'] >= self.open_batch)
            self.get_logger().info(f'yaw scan done: {new} candidate(s) for the next batch')
            self.spin = None
            return
        self.flight.step(s['x'], s['y'], self.search_alt(), s['yaws'][0], vmax=0.5)

    def scout_step(self):
        """Search start with no board in view: the line scout_offset in from the nearer long wall, to its far end.
        False once done / dropped (a board seen)."""
        if self.scout is None:
            return False
        if self.cand_ev or any(q.get('in_zone') for q in self.qrs):
            self.get_logger().info('board seen: scout line dropped')
            self.scout, self.scout_goal = None, False
            return False
        if self.now_s() - self.t_start < self.g('scout_wait'):
            return True                               # hover: the first view
        p = self.odom.pose.pose.position
        if not self.scout and not self.scout_goal:
            x0, x1, y0, y1 = self.g('zone')
            off, mg = min(self.g('scout_offset'), min(x1 - x0, y1 - y0) / 4), self.g('lane_margin')
            # the nearer line; equally near (the drone on the centre line): the side with more unexplored ground
            yq = min((y0 + off, y1 - off), key=lambda v: (round(abs(v - p.y)), -self.gain_at(p.x, v)))
            far = x1 - mg if abs(x1 - p.x) > abs(p.x - x0) else x0 + mg
            step = math.copysign(min(abs(far - p.x), abs(yq - p.y)), far - p.x)   # 45 deg out to the line
            self.scout = [(p.x + step, yq), (far, yq)]
            self.get_logger().info(f'no board in view: scout line y={yq:.1f} m ({off:.1f} m in from the wall)')
        if self.scout_goal:
            if not self.goal_done:
                return True
            self.scout.pop(0)
            self.scout_goal = False
        if not self.scout:
            self.get_logger().info('scout line done')
            self.scout = None
            return False
        self.scout_goal = True
        self.send(*self.scout[0], snap=self.g('snap_radius'))
        return True

    def deep_read(self, gx, gy):
        """The deep read over an unread candidate (see the deep_read parameter) -> True while it is in hand, False when
        it is exhausted (at verify_floor without an answer): then the look-round as before."""
        now = time.monotonic()
        if now < self.verify_hold:
            return True                               # settled: wait for qr_locator's board to clear the candidate
        if self.verify_id is None:
            self.verify_n += 1
            self.verify_id, self.verify_t = f'{gx:.1f},{gy:.1f}#{self.verify_n}', now
            self.verify_pub.publish(String(data=json.dumps({'id': self.verify_id, 'x': gx, 'y': gy})))
            self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): unread - deep read at {self.search_alt():.1f} m')
            return True
        r = self.verify_res.pop(self.verify_id, None)
        if r is None:
            if now - self.verify_t < self.g('verify_timeout'):
                return True
            r = {'result': 'timeout'}
        self.verify_id = None
        if r['result'] in ('read', 'match', 'different'):
            self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): deep read {r["result"]} '
                                   f'{r.get("payload") or r.get("score", "")}')
            self.verify_hold = now + 4.0
            return True
        alt = self.search_alt()
        if alt - self.g('verify_step') < self.g('verify_floor') - 1e-6:
            self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): deep read {r["result"]} down to '
                                   f'{alt:.1f} m - giving up on it')
            return False
        if self.trim_before is None:
            self.trim_before = float(self.g('alt_offset'))
        self.set_parameters([rclpy.parameter.Parameter('alt_offset', value=float(self.g('alt_offset')) -
                                                       self.g('verify_step'))])
        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): deep read {r["result"]} '
                               f'{r.get("score", r.get("why", ""))} - {self.g("verify_step"):g} m lower')
        self.cand_t, self.cand_arrive = self.now_s(), self.now_s()   # hover at the new height, then again
        return True

    def candidate_step(self):
        """Fly to / hover over the nearest candidate; False when there is none (frontiers then)."""
        if self.g('mode') != 'candidates':
            return False
        if self.cand_goal is None and (self.trim_before is not None or self.verify_id is not None):
            if self.trim_before is not None:          # the deep read's steps down: back to the search height
                self.set_parameters([rclpy.parameter.Parameter('alt_offset', value=self.trim_before)])
            self.trim_before, self.verify_id, self.verify_hold = None, None, 0.0
        p = self.odom.pose.pose.position
        if self.cand_goal is not None:
            gx, gy = self.cand_goal
            gone = all(math.dist((c['x'], c['y']), (gx, gy)) > 2.0 for c in self.cands)   # decoded: not a candidate
            if gone:
                q = min(self.qrs, key=lambda q: math.dist((q['x'], q['y']), (gx, gy)), default=None)
                read = q is not None and math.dist((q['x'], q['y']), (gx, gy)) < 3.0
                self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}) ' +
                                       (f'read: {q["payload"]!r} {q.get("status", "")}' if read
                                        else 'dropped by qr_locator (no decode) - the lanes cover it'))
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t, self.cand_look, self.cand_arrive = None, None, None, None
                if read and self.start_qr and q['payload'] != self.start_qr:   # a reject (decoded, maybe not yet
                    #                                                                confirmed: the payload says it)
                    self.start_spin(q['x'], q['y'])
                    if self.spin is not None:
                        return True
            elif self.cand_arrive is None and self.cand_look is None and (self.cand_dead(gx, gy) or self.in_red(gx, gy)):
                # also when the red zone is mapped under it on the way: a red-zone "board" took 30 % of a search,
                # re-routed round the no-fly margin until unreachable (2026-10-27 sim)
                self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): ' +
                                       ('inside the red zone' if self.in_red(gx, gy) else 'no longer seen') + ' - not going')
                self.cancel()
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t, self.cand_look, self.cand_arrive = None, None, None, None
                return True
            elif (self.cand_arrive is not None and self.now_s() - self.cand_arrive > self.g('cand_verify_max')
                  and not (self.g('deep_read') and (self.verify_id is not None or self.trim_before is not None
                                                    or time.monotonic() < self.verify_hold))):
                self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode in '
                                       f'{self.g("cand_verify_max"):.0f} s - next')
                self.cancel()
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t, self.cand_look, self.cand_arrive = None, None, None, None
                return True
            elif self.cand_look is not None:
                # looking around: next corner of the square once the last one is reached (or failed)
                if self.goal_done:
                    if self.cand_look:
                        self.send(*self.cand_look.pop(0), snap=self.g('snap_radius'))
                    else:
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode after looking round - next')
                        self.cand_done.append((gx, gy))
                        self.cand_goal, self.cand_t, self.cand_look, self.cand_arrive = None, None, None, None
                return True
            elif math.hypot(gx - p.x, gy - p.y) < self.g('cand_reach') or (self.goal_done and self.last_status == 4):
                self.cand_t = self.cand_t or self.now_s()   # node clock: sim time in the sim
                self.cand_arrive = self.cand_arrive or self.now_s()
                if self.now_s() - self.cand_t > self.g('cand_hover'):
                    if self.g('deep_read') and self.deep_read(gx, gy):
                        return True
                    r = self.g('look_radius')
                    if r > 0:
                        # a square round the fix, starting from the drone's side: each leg turns the heading 90 deg
                        # (PathFlight faces the leg), so the camera sees the spot from four sides
                        a0 = math.atan2(p.y - gy, p.x - gx)
                        look = [(gx + r * math.sqrt(2) * math.cos(a0 + math.pi / 4 + k * math.pi / 2),
                                 gy + r * math.sqrt(2) * math.sin(a0 + math.pi / 4 + k * math.pi / 2))
                                for k in range(4)]
                        # only corners the drone can fly to: one outside the zone ("no A* route to (11.2, -1.6)")
                        # was waited out as a failed frontier goal (2026-10-27 sim)
                        self.cand_look = [q for q in look if self.reachable(*q)]
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode - looking round it')
                        self.goal_done = True
                    else:
                        self.get_logger().info(f'candidate at ({gx:.1f}, {gy:.1f}): no decode after the hover - next')
                        self.cand_done.append((gx, gy))
                        self.cand_goal, self.cand_t, self.cand_arrive = None, None, None
                return True
            elif self.goal_done and self.last_status not in (4, None):
                self.get_logger().warn(f'candidate at ({gx:.1f}, {gy:.1f}) unreachable (status {self.last_status})')
                self.cand_done.append((gx, gy))
                self.cand_goal, self.cand_t, self.cand_arrive = None, None, None
            else:
                # on the way: switch only to a cheaper one of the SAME batch, and only while this leg is less than
                # commit_frac done - a board seen on the way waits for its batch (user, 2026-10-28)
                nxt = self.next_candidate(commit=False)
                if (nxt is not None and math.dist(nxt, (gx, gy)) > 1.5 and self.cand_leg
                        and self.cand_batch(*nxt) == self.cand_batch(gx, gy)):
                    done = 1.0 - math.hypot(gx - p.x, gy - p.y) / max(self.cand_leg, 1e-3)
                    if done < self.g('commit_frac'):
                        self.get_logger().info(f'cheaper in this batch: ({nxt[0]:.1f}, {nxt[1]:.1f}) first '
                                               f'({done:.0%} of this leg done)')
                        self.cand_goal, self.cand_t, self.cand_arrive = nxt, None, None
                        self.cand_leg = math.hypot(nxt[0] - p.x, nxt[1] - p.y)
                        self.last_status = None
                        self.send(*nxt, snap=self.g('snap_radius'))
                return True
        nxt = self.next_candidate()
        if nxt is None:
            return False
        self.cand_goal, self.cand_t, self.cand_arrive = nxt, None, None
        self.cand_leg = math.hypot(nxt[0] - p.x, nxt[1] - p.y)
        self.get_logger().info(f'candidate QR at ({nxt[0]:.1f}, {nxt[1]:.1f}): going (batch {self.cand_batch(*nxt)}, '
                               f'{len(self.pending())} left)')
        self.last_status = None
        self.sweep_goal = None                        # a lane leg was preempted: resend it afterwards
        self.scout_goal = False
        self.send(*nxt, snap=self.g('snap_radius'))
        return True

    def unconfirmed(self):
        """(key, x, y) of the QR to confirm next, or None."""
        p = self.odom.pose.pose.position
        todo = []
        # zone boards only: the pad's start QR carries the target's payload, confirmed before the search - it made the
        # target "already confirmed elsewhere" and the explorer never flew over it to confirm (2026-10-28 sim)
        done = {q['payload'] for q in self.qrs if q.get('confirmed') and q.get('in_zone')}
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

    def lane_extent(self, x, ya, yb):
        """(y_first, y_last) of the lane's rows still unexplored, ordered from ya towards yb; None if none."""
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
        rows = np.nonzero(unknown[j0:j1, i0:i1].sum(axis=1) >= self.g('lane_row_frac') * max(i1 - i0, 1))[0]
        if not len(rows):
            return None
        ys = [m.info.origin.position.y + (j0 + k + 0.5) * r for k in (rows[0], rows[-1])]
        ys = [min(max(v, min(ya, yb)), max(ya, yb)) for v in ys]
        return (ys[0], ys[1]) if yb >= ya else (ys[1], ys[0])

    def lane_target(self, i):
        """The point to fly for lane leg i, the lane cut to its unexplored stretch (2026-10-27: after each candidate
        detour the drone went back to the lane's far start and re-flew seen ground). The drone joins the stretch at
        its nearer end; the next leg flies to the other. None: the lane is done."""
        i0 = i - i % 2
        x, ya = self.sweep[i0]
        yb = self.sweep[i0 + 1][1]
        ext = self.lane_extent(x, ya, yb)
        if ext is None:
            return None
        p = self.odom.pose.pose.position
        on_lane = abs(p.x - x) < 1.5
        if not (i % 2 == 1 and on_lane):
            # joining a lane (or back from a detour): the nearest unfinished stretch of ANY lane left, not this one
            # in order - from the target at (26, -9) the drone flew 25 m back to finish lane 1 (2026-10-27 sim)
            best = None
            for k in range(i0, len(self.sweep) - 1, 2):
                kx, ka = self.sweep[k]
                kb = self.sweep[k + 1][1]
                e = self.lane_extent(kx, ka, kb)
                if e is None or self.lane_unknown(kx, *e) < self.g('min_gain'):
                    continue
                d = min(math.hypot(p.x - kx, p.y - e[0]), math.hypot(p.x - kx, p.y - e[1]))
                if best is None or d < best[0]:
                    best = (d, k, e)
            if best is None:
                self.sweep_i = len(self.sweep)
                return None
            _, k, ext = best
            if k != i0:                               # bring that lane forward; the skipped ones come after it
                pair = self.sweep[k:k + 2]
                del self.sweep[k:k + 2]
                self.sweep[i0:i0] = pair
            x, ya, yb = self.sweep[i0][0], self.sweep[i0][1], self.sweep[i0 + 1][1]
        elif self.lane_unknown(x, *ext) < self.g('min_gain'):
            return None                               # only slivers left on this lane
        if i % 2 == 1 and on_lane:
            # flying the lane: on to the far end of what is unexplored, if it is still ahead
            ahead = (ext[1] - p.y) * (1 if yb >= ya else -1)
            return (x, ext[1]) if ahead > 0.5 else None
        # joining (or re-joining after a detour): the nearer end of the stretch; the lane then runs to the other
        if abs(p.y - ext[1]) < abs(p.y - ext[0]):
            self.sweep[i0], self.sweep[i0 + 1] = (x, yb), (x, ya)
            ext = (ext[1], ext[0])
        self.sweep_i = i0
        return (x, ext[0])

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
            tgt = self.lane_target(i)
            if tgt is None:
                if i % 2 == 1:
                    self.sweep_i += 1                 # the rest of this lane is seen: next lane
                else:
                    self.get_logger().info(f'sweep lane x={x:.1f}: already seen, skipped')
                    self.sweep_i += 2
                self.sweep_tries = 0
                continue
            i = self.sweep_i                          # lane_target may restart the lane at its nearer end
            x, y = tgt
            other = self.sweep[i + 1][1] if i % 2 == 0 else self.sweep[i - 1][1]
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
        if self.state == 'EXPLORING':
            self.cancel()
            self.set_state(f'TARGET_FOUND at ({m.pose.position.x:.2f}, {m.pose.position.y:.2f})')

    def cancel(self):
        if self.spin is not None:
            self.spin = None
            self.flight.stop()
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

    def search_alt(self):
        """The search height: `altitude` + the operator's `alt_offset`, not below `alt_min`, not above `alt_max`."""
        return min(float(self.g('alt_max')), max(float(self.g('alt_min')),
                                                 float(self.g('altitude')) + float(self.g('alt_offset'))))

    def on_params(self, params):
        """An alt_offset change applies at once: the route being flown takes the new height too."""
        from rcl_interfaces.msg import SetParametersResult
        for prm in params:
            if prm.name == 'alt_offset':
                alt = min(float(self.g('alt_max')), max(float(self.g('alt_min')), float(self.g('altitude')) + float(prm.value)))
                if self.path.active:
                    self.path.alt = alt
                self.get_logger().info(f'search height trim {float(prm.value):+.1f} m: flying at {alt:.1f} m')
        return SetParametersResult(successful=True)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9   # sim time in the sim (use_sim_time)

    def reachable(self, x, y):
        """Inside the zone and an A* route there from the drone."""
        x0, x1, y0, y1 = self.g('zone')
        if not (x0 + 0.5 <= x <= x1 - 0.5 and y0 + 0.5 <= y <= y1 - 0.5):
            return False
        return self.plan_to(x, y, self.g('snap_radius')) is not None

    def plan_to(self, x, y, snap=0.0):
        """A* route from the drone; cells already flown over cost retrace_cost extra while a frontier is left."""
        p = self.odom.pose.pose.position
        avoid = self.trail if self.has_frontier else None
        return plan(self.map, (p.x, p.y), (x, y), self.g('path_clearance'), 0.5, snap,
                    avoid=avoid, avoid_cost=self.g('retrace_cost'))

    def mark_trail(self):
        m = self.map
        if m is None or self.g('retrace_cost') <= 0:
            return
        shape = (m.info.height, m.info.width)
        if self.trail is None or self.trail.shape != shape:
            self.trail = np.zeros(shape, bool)
        r = m.info.resolution
        p = self.odom.pose.pose.position
        img = self.trail.view(np.uint8)
        cv2.circle(img, (int((p.x - m.info.origin.position.x) / r), int((p.y - m.info.origin.position.y) / r)),
                   max(1, int(self.g('trail_radius') / r)), 1, -1)

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
            wps = self.plan_to(x, y, snap)
            if wps is None:
                self.get_logger().warn(f'no A* route to ({x:.1f}, {y:.1f})')
                self.path.stop()
                self.last_status, self.goal_done = 6, True        # as a Nav2 abort
                if (self.state == 'EXPLORING' and self.sweep_goal is None and self.cand_goal is None
                        and not self.scout_goal):
                    self.failed((x, y), 6)
                return
            self.last_status, self.goal_done = None, False
            self.path.start(wps, self.search_alt(), vmax=self.g('speed'))
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
        if self.state == 'EXPLORING':
            self.mark_trail()
        if self.spin is not None:
            if self.state != 'EXPLORING':
                self.spin = None
                self.flight.stop()
            else:
                self.spin_step()
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
        self.flight.step(x, y, self.search_alt(), yaw, vmax=self.g('escape_speed'))

    def tick(self):
        self.state_pub.publish(String(data=self.state))
        if self.state != 'EXPLORING':
            return
        self.alt_pub.publish(Float32(data=self.search_alt()))
        if self.map is None or self.odom is None:
            return
        if self.escape_to is not None or self.spin is not None:
            return                                    # flying out of the no-fly cells / yaw scan (escape_step)
        self.has_frontier = self.choose() is not None
        esc = self.escape_point()
        if esc is not None:
            self.cancel()
            self.sweep_goal = None                    # resend the lane leg afterwards
            self.scout_goal = False
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
        if self.spin_due is not None:
            # a reject read in passing: scan here, then the interrupted lane / scout leg is sent again (cancel() ends
            # it "done", which the sweep / scout would take as reached)
            xy, self.spin_due = self.spin_due, None
            if self.cand_goal is None:
                self.start_spin(*xy)
                if self.spin is not None:
                    self.sweep_goal, self.scout_goal = None, False
                    return
        self.cand_evidence()
        if self.candidate_step():
            return
        if self.scout_step():
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
