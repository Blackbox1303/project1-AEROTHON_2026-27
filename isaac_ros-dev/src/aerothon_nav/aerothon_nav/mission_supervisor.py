"""Mission supervisor: the top-level state machine that runs the whole AeroTHON mission 2 from the ground.

  TAKEOFF         PX4: arm, AUTO.TAKEOFF to takeoff_altitude; ArduCopter: GUIDED, arm, MAV_CMD_NAV_TAKEOFF to
                  takeoff_altitude; then OFFBOARD / GUIDED (/nav/offboard); home = the take-off point
                  (autopilot: default $AEROTHON_AUTOPILOT, else ardupilot - flight mode names of px4 | ardupilot)
  READ_START_QR   1 m forward, /mission/read_qr (qr_reader stores the payload)
  OVERHEAD_GREEN  (user, 2026-10-28) still at takeoff_altitude 5 m: the downward Arducam (drone_qr_detector banner_node
                  as overhead_banner) finds the GREEN banner - floor 'doormat' at the mouth / top banner, 11.5 x 1.5 ft
                  across the tunnel (2026-10-28 a RED banner marked the other tunnel - gone 2026-10-30: red is ignored,
                  the start zone is red). Look points every overhead_step m towards the QR area (unknown: the take-off heading), then
                  lines overhead_lateral m to the sides. The banner's long edge on the ground -> the tunnel axis (away
                  from home); fly to standoff before the mouth at 5 m - never over the roof - then down to align_altitude
                  (< 8 ft) and ALIGN_GREEN in entry_align's geofence mode = the D435 wall fit refines the axis.
                  No green banner / no wall fit: the D435 banner search below, as before
  ALIGN_GREEN     /nav/entry/align, colour green: banner search at align_altitude, corridor axis from the walls
                  (/nav/entry/result)
  GEOFENCE        save the entrance geofence: entrance pose + width, GPS fix, local position, an Arducam snapshot
                  -> geofence_file (JSON) + .png, /mission/geofence (latched String JSON)
  ENTRY_CORRIDOR  corridor_nav from that entrance, wait for DONE length=<m>; the corridor's far end (entrance +
                  length along the axis) and the GPS there are added to the geofence
  COURSE v3 (organisers, 2026-10-24): tunnels roofed at 10 ft - flown at tunnel_altitude, never above 8 ft (2.44 m;
                  offboard_bridge caps 2.3 m in and at them); out of the entry tunnel fly clear of the roof, THEN climb
                  to search_altitude 10 m; drop at 5 m, back to 10 m; down to align_altitude at the exit approach. The
                  tunnel to fly into always shows GREEN (from the start area: entry; from the zone: exit), blue the other
  COURSE 2026-10-30 (user): the entry tunnel GREEN, the exit tunnel BLUE as seen from the QR / marker area (exit_colour),
                  both with a thin white border, an upper banner + a lower one flat on the floor at the mouth (doormat).
                  ONLY these two banners: none marks a tunnel to ignore. The start zone is RED: take off from it, land
                  back on it at the end
  (the start_from test entries below fly from home to search_start in a straight line over the roofed tunnels -
                  the bridge's tunnel ceiling stops that: use them only with the old worlds)
  (start_from delivery: the ground station's DELIVERY-ONLY mission - the drone stands at / beside the delivery zone:
                  TAKEOFF, READ_START_QR (or the payload the ground station published on /mission/start_qr; read_start_qr
                  false), climb to search_altitude, DIRECT_TO_ZONE (straight to 1 m inside the zone's nearest edge if
                  outside its map), SEARCH_DELIVERY, DELIVER, ZONE_EXIT (delivery Nav2 to the zone edge nearest home - red
                  zones avoided - then straight home), LAND. No corridors, banners or entrance geofence)
  QR-AREA TIME (user, 2026-10-24: <= 9 min in the QR area wherever the red zone / target is): zone_time_limit from
                  entering the zone; the search ends early enough for the drop (deliver_reserve) and the way out
                  (zone_exit_reserve, delivery-only: zone_home_reserve) to fit too
  (start_from search: test entry - after READ_START_QR, DIRECT_TO_SEARCH climbs to search_altitude, above the
                  corridor walls, and flies straight to search_start; no corridors, no geofence, so return_corridor
                  must be false: it lands on the target)
  (start_from exit: test entry for the return corridor - after TAKEOFF, climb to search_altitude, fly to
                  search_start (above the walls), ALIGN_ORANGE (banner only: no geofence), RETURN_CORRIDOR,
                  RETURN_HOME, LAND)
  SEARCH_DELIVERY climb to search_altitude, /delivery/map/reset, /delivery/search/start (qr_locator: only reads
                  from now on can be the target), /delivery/explore/start; wait for /delivery/target
  DELIVER         Nav2 (delivery) to the target, descend to deliver_altitude, settle deliver_hold s, DROP with
                  drop_mechanism:
                  rc5 (default, 2026-10-30): the transmitter's channel 5 - RC5_OPTION 28 Relay On/Off (RELAY1) -
                  switched by MAV_CMD_DO_AUX_FUNCTION high, rc5_drop_hold 45 s at the drop height, then low.
                  winch (default): N20 gear motor on a Cytron MDD20A, 6 m nylon line, passive hook that lets go when
                  the payload sits on the ground and the line goes slack. Two ArduPilot relays (MAV_CMD_DO_SET_RELAY
                  over /mavros/cmd/command): RELAY1 -> MDD20A PWM1 (run / brake), RELAY2 -> DIR1. Pays out
                  height - winch_stowed_drop + winch_slack (capped at the line), holding position, waits
                  winch_release_wait s, winds winch_retract_ratio of it back in, climbs back to search_altitude.
                  Open loop on time x winch_speed (encoder not read). An aborted drop brakes the motor and winds the
                  line in on a timer while the failsafe flies. /mission/winch (SetBool) = bench test.
                  gripper: release (MAV_CMD_DO_GRIPPER), wait drop_wait s, grab again.
                  A release the autopilot doesn't accept is retried drop_retries times, then logged (DROP_FAILED) and
                  the mission goes on to the exit - the drone still has to come home.
                  Hardware (ArduCopter): vio_bringup/config/ardupilot_vio_params.parm (winch relays); the sim:
                  vio_cuda_ws/sim/ardupilot_sim_params.parm (SITL relay pins, SIM_PIN_MASK shows them).
  ALIGN_EXIT      /nav/entry/align, colour exit_colour (blue since 2026-10-30; v3 green); the corridor found is checked against the entrance geofence: one
                  whose centre line is within exclude_radius of the entry corridor's IS the entry corridor -> rejected
  ALIGN_GEOFENCE  exit fallback (no exit banner, or it led back to the entry corridor): entry_align mode geofence,
                  predicted mouth = the entry corridor's far end, direction = entry yaw + 180 deg, entry corridor
                  excluded (the return corridor lies parallel beside it)
  RETURN_CORRIDOR corridor_nav from the exit found, wait for DONE
  RETURN_HOME     climb to home_altitude, fly to the take-off point
  LAND            AUTO.LAND, wait until disarmed -> DONE
CRASH (any time while armed: roll or pitch beyond flip_angle for flip_time s): force-disarm PX4 (MAV_CMD 400,
  21196), abort, no failsafe flight -> FAILED. A wall strike flipped the drone in the entry corridor; it stayed
  armed and slid along inverted while corridor_nav reported EXIT/DONE and the mission went on (2026-10-20 sim).
FAILSAFE (any FAILED step, a state's time limit, the mission timer `mission_time_limit`, or /mission/abort):
  stop the active nodes (entry / corridor / explorer cancel), hold failsafe_hold s, land where it is -> FAILED
  (failsafe_home: climb to failsafe_altitude and fly to the take-off point first - not with the 10 ft rule). Inside a corridor it first flies its own track
  back out (breadcrumbs, the path just flown is clear) before climbing: a straight climb under a window's lintel
  flipped the drone in the return corridor (2026-10-21 sim); if the retrace stalls it lands where it is. Once a
  window was passed it lands in place instead (the floor below is free): retracing a 1.06 m opening on the drifting
  GPS-only estimate clipped the frame (2026-10-21 sim).
Direct legs fly through offboard_bridge (aerothon_nav/direct_flight.py); the corridor and delivery legs through
their Nav2 instances. Services /mission/start, /mission/abort (std_srvs/Trigger), /mission/winch (std_srvs/SetBool:
true = pay out winch_test_length, false = wind it in), /mission/gripper (std_srvs/SetBool:
true = release, false = grab; bench test / loading, refused during a mission); state on /mission/state;
every transition on /mission/log (String JSON: time, state, info).
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandLong, SetMode
from sensor_msgs.msg import Image, NavSatFix
from rclpy.qos import qos_profile_sensor_data
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from aerothon_nav import qr_area
from aerothon_nav.direct_flight import DirectFlight, yaw_of
from aerothon_nav.grid_path import PathFlight, plan, route_clear

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Abort(Exception):
    pass


# flight modes (MAVROS custom_mode names) per autopilot; the take-offs: takeoff_px4 / takeoff_ardupilot
MODES = {'px4': {'offboard': 'OFFBOARD', 'land': 'AUTO.LAND'},
         'ardupilot': {'offboard': 'GUIDED', 'land': 'LAND'}}


def pval(v):
    if isinstance(v, bool):
        return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v)
    if isinstance(v, (int, float)):
        return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))
    if isinstance(v, (list, tuple)):
        return ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[float(x) for x in v])
    return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(v))


class MissionSupervisor(Node):
    def __init__(self):
        super().__init__('mission_supervisor')
        p = self.declare_parameter
        # ALTITUDES: 10 m in the tunnels and the search, 5 m for the drop (user, 2026-10-23 - the earlier 10 ft rule
        # was a misunderstanding). fly() clamps to max_altitude and offboard_bridge clamps its setpoint independently
        p('max_altitude', 9.5)       # m: HARD CAP - the airspace rule there: never above 10 m (user, 2026-10-30); 0.5 m
        #                              left for ArduCopter's height overshoot. Every fly() target is clamped to it
        p('takeoff_altitude', 5.0)   # m: take off to 5 m, move 1 m forward, scan the start QR, then descend to align_altitude for corridor VIO
        p('align_altitude', 2.3)     # m: banner search + wall fit outside the tunnels (below 8 ft: no climbing near them)
        p('tunnel_altitude', 1.5)    # m: both tunnels (v3: roofed 10 ft, never above 8 ft = 2.44 m)
        p('entry_speed', 1.0)        # m/s along the entry tunnel (no obstacles)
        p('return_speed', 0.5)       # m/s along the exit tunnel (the blocks: lateral plan + sidesteps)
        p('exit_colour', 'blue')     # banner of the tunnel to fly into from the zone (2026-10-30: blue with a thin
        #                              white border; v3: green; v2: orange)
        p('overhead_banner', True)   # OVERHEAD_GREEN: the Arducam banner fix from takeoff_altitude before ALIGN_GREEN
        p('overhead_step', 2.0)      # m between look points
        p('overhead_search_dist', 2.0)   # m out from the start QR along the search heading: from 5 m the camera sees
        #                              ~4-5 m ahead; further out the tunnel roofs begin (the sim's tunnel ceiling then
        #                              pulled the drone down to 2.3 m at the mouth, 2026-10-29) - farther: D435 search
        p('overhead_lateral', [0.0, 4.0, -4.0])  # m: search lines beside the first (start side only)
        p('overhead_look', 1.5)      # s hovering per look at least (the detector runs at 4 Hz, a track needs 2 hits) ...
        p('overhead_frames', 4)      # ... and until it has processed this many frames (the loaded sim: ~1 per 8 s)
        p('overhead_look_max', 40.0) # s (wall) per look at most
        p('overhead_banner_len', [2.0, 5.5])  # m: the floor banner (11.5 ft = 3.5 m) projected on the ground; the
        #                              top banner projected on the ground from 5 m comes out ~2.5 x longer
        p('overhead_hold_back', 1.5) # m on the start side of the banner where the drone looks from (never over the roof)
        p('overhead_lateral_max', 1.5)  # m: the wall fit's corridor must pass this close to the banner centre
        p('search_altitude', 9.0)    # delivery zone: under the 10 m airspace limit with margin (was 10 m; 2026-10-30)
        p('deliver_altitude', 5.0)   # m: the drop height
        p('deliver_tol', 0.15)       # m: arrival tolerance over the target (both levels)
        p('fine_wait', 8.0)          # s at the drop height for qr_locator's fine fix (reads from there) before release
        p('fine_min_reads', 3)
        p('fine_tol', 0.08)          # m: re-centre on the fine fix to within this
        p('fine_max_jump', 2.0)      # m: a fine fix further than this from the 10 m one is not the same board: ignored
        p('direct_fallback', 6.0)    # m: Nav2 can't reach a target this close (e.g. in a red zone's margin - never IN a
        #                              red zone, qr_locator checks): fly the short hop straight at search height
        p('deliver_hold', 1.5)       # s settling at deliver_altitude before the release (after the re-centring)
        p('deliver_min_time', 45.0)  # s at deliver_altitude at least, from arriving to the climb: the winch and the drop
        #                              mechanism need it (user, 2026-10-30)
        p('drop_wait', 1.5)          # s after the release is acknowledged (payload clear) before closing
        p('drop_retries', 3)
        p('delivery_nav2', False)    # False: the QR-zone legs (to the target, to the exit, zone exit) are A* routes on
        #                              /delivery/map flown straight (grid_path) - the delivery Nav2's MPPI crawled at
        #                              0.7 m/s on the loaded Jetson (2026-10-25); True: delivery Nav2 goals
        p('zone_speed', 1.5)         # m/s on those routes (rules: <= 10 km/h; 2.5 too fast, user 2026-10-25)
        p('direct_deliver_radius', 2.0)  # m: a target this close is flown to directly, not via Nav2
        p('gripper_timeout', 1.0)    # s = PX4 PD_GRIPPER_TO: PX4 acks the action as done after that
        # DROP MECHANISM: winch (N20 gear motor + 6 m nylon line + passive release hook, user 2026-10-28) | gripper.
        # Winch: Cytron MDD20A channel 1 driven by two ArduPilot relays (FMU outputs as GPIO, MAV_CMD_DO_SET_RELAY):
        # PWM1 = run (low = brake), DIR1 = direction. Open loop: paid-out length = winch_speed x run time (the N20's
        # hall encoder is not read - the Pixhawk can't count it in Copter).
        p('drop_mechanism', 'rc5')    # rc5 (the transmitter's channel 5: relay 1 via DO_AUX_FUNCTION, 2026-10-30) |
        #                               winch (the relay-driven N20 winch) | gripper (servo gripper, MAV_CMD_DO_GRIPPER)
        p('rc5_aux_function', 28)     # RC5_OPTION on the Pixhawk (read 2026-10-30): 28 = Relay On/Off (RELAY1, pin 54)
        p('rc5_drop_hold', 45.0)      # s at the drop height after RC5 high: the winch + drop mechanism (user, 2026-10-30)
        p('winch_relay_run', 0)       # ArduPilot relay instance (0 = RELAY1) on MDD20A PWM1
        p('winch_relay_dir', 1)       # relay instance (1 = RELAY2) on MDD20A DIR1
        p('winch_dir_down', 1)        # DIR level that unwinds (pays out) the line: swap to 0 if it winds the wrong way
        p('winch_speed', 0.3377)      # m/s of line: spool 21.5 mm x pi x 300 rpm / 60 (bare drum; calibrate)
        p('winch_line', 6.0)          # m of line on the spool
        p('winch_line_reserve', 0.3)  # m never paid out (the knot on the spool must not take the load)
        p('winch_stowed_drop', 0.25)  # m from the altitude reference (FC) down to the payload's bottom when wound in
        p('winch_slack', 0.5)         # m paid out past touch-down: the line goes slack, the hook lets go
        p('winch_release_wait', 2.0)  # s holding at the bottom (slack line) before winding in
        p('winch_retract_ratio', 0.95)  # wind in this share of the paid-out time: the hook must not jam the spool
        p('winch_test_length', 0.5)   # m out (then in) for /mission/winch (bench test, props off)
        p('home_altitude', 2.3)
        p('failsafe_hold', 5.0)
        p('failsafe_home', False)     # fly home after the failsafe hold: the straight line home crosses the tunnel
        #                               walls (taller than 10 m) - so land where it is (after a corridor retrace)
        p('failsafe_altitude', 2.3)   # m, if failsafe_home
        p('trail_step', 0.3)          # m between corridor breadcrumbs (failsafe retrace)
        p('retrace_speed', 0.3)       # m/s back along them
        p('no_timeout', True)         # disable state and mission time limits
        p('autopilot', os.environ.get('AEROTHON_AUTOPILOT', 'ardupilot'))   # px4 | ardupilot (offboard_bridge too)
        p('start_from', 'takeoff')    # takeoff (full mission) | delivery (delivery only) | tunnel (tunnels + banners
        #                               only: entry corridor, straight to the exit corridor, home) | search | exit (tests)
        p('qr_count', 5)              # boards in the QR area: once that many are read the search ends, target or not
        p('drop', True)               # False: the QR area without the drop - explore + read everything, then out
        p('zone_time_limit', 540.0)   # s in the QR area at most (entry -> the start of the exit align / the way home)
        p('zone_exit_reserve', 150.0) # s of it kept for TO_EXIT + ALIGN_EXIT (full mission)
        p('zone_home_reserve', 90.0)  # s of it kept for ZONE_EXIT to the zone edge (delivery only)
        p('search_start', [17.0, 0.0])
        p('read_start_qr', True)      # False (tests only): skip it, qr_locator keeps the stored payload  # m, odom: where start_from search enters the delivery zone
        p('return_corridor', True)    # after the drop: back to search altitude, exit corridor, home (False: land on the target)
        p('geofence_file', os.path.expanduser('~/.ros/aerothon_entrance_geofence.json'))
        p('world_json', '')           # the zone (= delivery_mapper's): delivery-only flies into / out of it
        p('snapshot_topic', '/arducam/image_raw')
        p('exclude_radius', 2.0)      # m: an exit centre line this close to the entry corridor's is the entry corridor
        #                               or the gap BETWEEN the corridors (taken for the exit at 1.58 and 1.72 m,
        #                               2026-10-21 sim; there the corridors are 3.05 m apart). Assumes corridors > 2 m
        #                               apart centre to centre: check on the real course
        p('exit_parallel_deg', 15.0)  # deg: the exit fit must be parallel to the entry corridor within this (exit fits
        #                               came within 3-7 deg; a fit 18.6 deg off was the gap between the corridors)
        p('geofence_fallback', True)  # exit from the entrance geofence when the exit banner fails
        p('mav_timeout', 30.0)        # s per MAVROS command: on a loaded link replies took > 10 s (2026-10-20)
        p('airborne_wait', 15.0)      # s the failsafe watches for armed / height before taking 'disarmed' as true
        p('flip_angle', 60.0)         # deg roll / pitch: beyond this the drone has crashed / flipped
        p('flip_time', 0.5)           # s sustained
        p('upset_angle', 35.0)        # deg roll / pitch, any duration: a collision -> abort (failsafe: land in place).
        #                               A hanging-box strike tumbled the drone (+-50 deg) without a sustained flip;
        #                               corridor_nav reported DONE and RETURN_HOME flew it out of the geofence
        #                               (2026-10-23 sim). Normal flight stays under ~15 deg
        p('mission_time_limit', 900.0)
        # MISSION BUDGET (rule: 15 min overall), on the node clock = sim time in sim (the slow sim's wall clock would
        # overstate it ~3x). Every state change logs t_min. The search gets what is left after return_reserve (exit
        # corridor + home + land) and deliver_reserve; out of time it stops and goes for the exit without a drop
        p('mission_budget', 840.0)    # s: the 15 min rule minus 1 min of buffer
        # corridor_nav altitude per corridor: 2D (slam_toolbox map, fixed height) in the entry corridor - it has no
        # obstacles; the 3D altitude planner only in the return corridor's windows / hanging boxes (user, 2026-10-22)
        p('into_zone_dist', 1.5)      # m past the entry corridor's end (= the zone boundary) where the climb starts
        p('exit_approach_dist', 2.5)  # m past the entry corridor's end (its heading): flown with the delivery Nav2.
        #                               v3 (2026-10-24): 2.5 - the exit's green banner is 3.6 m away there, and 6 m put
        #                               the point in the red zone's margin (moved next to the corridors)
        #                               (A* round the red zones) before ALIGN_ORANGE. From the zone's far end the
        #                               orange banner was 37 m off and never seen; the geofence fallback then flew a
        #                               straight line across the big red zone (2026-10-22 sim)
        p('corridor_nav2', False)     # the corridor Nav2 (A*) is not used: corridor_nav flies the corridors on the
        #                               2D SLAM pose (nav_mode direct, no A* in the corridors). False keeps it paused
        p('entry_altitude_mode', 'fixed')
        p('return_altitude_mode', 'planner')
        p('return_reserve', 240.0)    # s
        p('min_corridor_length', 3.7) # m: corridor_nav min_length (exit detection from here on): 0.6 x the 20 ft 5 in
        #                               (6.22 m) tunnels of 2026-10-30 - the course block sets it (v3: 4.0 of 6.7 m)
        p('length_slack', 1.5)        # m: return corridor min_length = entry corridor length - this
        p('deliver_reserve', 120.0)   # s: the climb down / up + deliver_min_time 45 s at 5 m (the winch: ~35 s of it)
        p('limit_takeoff', 90.0)     # s per state
        p('limit_read_qr', 45.0)
        p('limit_align', 300.0)
        p('limit_corridor', 240.0)
        p('limit_search', 900.0)
        p('limit_deliver', 200.0)
        p('limit_home', 180.0)
        p('limit_land', 90.0)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        if self.g('autopilot') not in MODES:
            raise ValueError(f"autopilot must be one of {sorted(MODES)}, not '{self.g('autopilot')}'")
        self.ardupilot, self.modes = self.g('autopilot') == 'ardupilot', MODES[self.g('autopilot')]
        self.odom = None
        self.mav = State()
        self.entry_state = self.corridor_state = self.explore_state = ''
        self.target = None
        self.entry_result = None
        self.gps = None
        self.geofence = None
        self.state = 'IDLE'
        self.running = False
        self.abort_reason = None
        self.t_start = None
        self.flight = DirectFlight(self, self.pose, lambda: self.abort_reason is not None)
        self.route = PathFlight(self.flight, self.pose)
        self.delivery_map = None
        self.create_subscription(OccupancyGrid, '/delivery/map', lambda m: setattr(self, 'delivery_map', m), LATCHED)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.log_pub = self.create_publisher(String, '/mission/log', 10)
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.mav_time = 0.0
        self.create_subscription(State, '/mavros/state', self.on_mav_state, 10)
        self.create_subscription(String, '/nav/entry/state', lambda m: setattr(self, 'entry_state', m.data), 10)
        self.create_subscription(String, '/nav/corridor/state', self.on_corridor_state, 10)
        self.create_subscription(String, '/delivery/explore/state', lambda m: setattr(self, 'explore_state', m.data), 10)
        self.create_subscription(PoseStamped, '/delivery/target', self.on_target, LATCHED)
        self.target_seen = False     # the target payload read in the zone (maybe not confirmed yet)
        self.qr_read = []            # (x, y) of the boards qr_locator has read in this search (the qr_count end):
        #                              by position, not payload - one board misread twice ('raw4x4#..' and its real
        #                              '3X3#107') counted as 2 and ended the search with the target unread (2026-10-28)
        self.create_subscription(String, '/delivery/qrs', self.on_qrs, 10)
        self.target_fine = None
        self.create_subscription(String, '/delivery/target_fine',
                                 lambda m: setattr(self, 'target_fine', json.loads(m.data)), LATCHED)
        self.create_subscription(String, '/nav/entry/result', lambda m: setattr(self, 'entry_result', json.loads(m.data)),
                                 10)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', lambda m: setattr(self, 'gps', m),
                                 qos_profile_sensor_data)
        self.geofence_pub = self.create_publisher(String, '/mission/geofence', LATCHED)
        # the QR area (qr_area.py): given by the ground station, or the arena cut at the entry tunnel's exit line
        self.qr_pub = self.create_publisher(String, qr_area.TOPIC, qr_area.QOS)
        self.qr_poly, self.qr_final, self.qr_src = None, False, None
        self.course, self.arena, self.qr_back = None, [], []
        cli = self.create_client
        self.srv = {name: cli(Trigger, name) for name in (
            '/mission/read_qr', '/nav/entry/align', '/nav/entry/cancel', '/nav/corridor/run', '/nav/corridor/cancel',
            '/delivery/map/reset',
            '/delivery/search/start', '/delivery/search/stop', '/delivery/explore/start', '/delivery/explore/stop')}
        self.offboard = cli(SetBool, '/nav/offboard')
        self.arming = cli(CommandBool, '/mavros/cmd/arming')
        self.set_mode = cli(SetMode, '/mavros/set_mode')
        self.px4_params = cli(SetParameters, '/mavros/param/set_parameters')
        # reentrant: /mission/gripper calls it from inside a service callback
        reentrant = ReentrantCallbackGroup()
        self.px4_command = cli(CommandLong, '/mavros/cmd/command', callback_group=reentrant)
        self.entry_params = cli(SetParameters, '/entry_align/set_parameters')
        self.overhead_params = cli(SetParameters, '/overhead_banner/set_parameters')
        self.overhead_dets, self.overhead_frames = [], 0
        self.create_subscription(String, '/nav/banner/overhead', self.on_overhead, 10)
        self.corridor_params = cli(SetParameters, '/corridor_nav/set_parameters')
        self.explorer_params = cli(SetParameters, '/frontier_explorer/set_parameters')
        # the Nav2 not in use is paused (lifecycle deactivate): two full Nav2 stacks plus the sim overloaded the
        # Jetson - the delivery Nav2's servers stopped acknowledging goals (2026-10-21 sim)
        self.nav2_managers = {'corridor': cli(ManageLifecycleNodes, '/lifecycle_manager_navigation/manage_nodes'),
                              'delivery': cli(ManageLifecycleNodes,
                                              '/delivery/lifecycle_manager_navigation/manage_nodes')}
        self.nav2_active = {'corridor': True, 'delivery': True}
        self.delivery_nav = ActionClient(self, NavigateToPose, '/delivery/navigate_to_pose')
        self.create_service(Trigger, '/mission/start', self.on_start)
        self.create_service(Trigger, '/mission/abort', self.on_abort)
        # the ground's LAND / RTL / KILL: the operator flies now - stop the mission without a failsafe flight (a
        # failsafe would put the autopilot back in GUIDED over the operator's LAND / RTL)
        self.create_service(Trigger, '/mission/operator', self.on_operator)
        self.create_service(SetBool, '/mission/gripper', self.on_gripper, callback_group=reentrant)
        # winch bench test: true = pay out winch_test_length, false = wind it in (props off, refused in a mission)
        self.create_service(SetBool, '/mission/winch', self.on_winch, callback_group=reentrant)
        self.winch_out = 0.0          # m of line out (estimate: run time x winch_speed)
        self.winch_lock = threading.Lock()
        self.winch_timer = None       # the background wind-in after an aborted drop
        self.create_timer(1.0, lambda: self.state_pub.publish(String(data=self.state)))
        self.crashed = False
        self.operator = False
        self.tilt_since = None
        self.create_timer(0.1, self.check_crash, callback_group=reentrant)
        self.create_timer(0.2, self.check_takeover, callback_group=reentrant)
        self.trail, self.in_corridor, self.window_seen = [], False, False
        self.t0_clock = None
        self.create_timer(0.2, self.record_trail, callback_group=reentrant)

    # ---- inputs / helpers
    def pose(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, p.position.z, yaw_of(p.orientation)

    def on_target(self, m):
        self.target = (m.pose.position.x, m.pose.position.y)

    def mission_elapsed(self):
        """s since take-off on the node clock (sim time in sim); 0 before take-off."""
        if self.t0_clock is None:
            return 0.0
        return (self.get_clock().now() - self.t0_clock).nanoseconds * 1e-9

    def set_state(self, s, **info):
        self.state = s
        if self.t0_clock is not None:
            info = {**info, 't_min': round(self.mission_elapsed() / 60, 2)}
        self.state_pub.publish(String(data=s))
        self.log_pub.publish(String(data=json.dumps({'time': round(time.time(), 2), 'state': s, **info})))
        self.get_logger().info(f'{s} {info if info else ""}')

    def check(self, deadline):
        if self.abort_reason:
            raise Abort(self.abort_reason)
        if not self.g('no_timeout'):
            if self.clock_s() - self.t_start > self.g('mission_time_limit'):
                raise Abort(f'mission time limit {self.g("mission_time_limit"):.0f} s')
            if self.clock_s() > deadline:
                raise Abort(f'{self.state} took longer than its limit')

    def wait(self, cond, deadline, period=0.2):
        while not cond():
            self.check(deadline)
            time.sleep(period)

    def call(self, client, req, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise Abort(f'{client.srv_name} not available')
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.05)
        if not fut.done():
            raise Abort(f'{client.srv_name} timed out')
        return fut.result()

    def mav_call(self, client, req):
        """A MAVROS command whose reply may come late or never although PX4 executes it: a timeout is logged, not
        raised - the caller checks the vehicle state instead (a timed-out arm DID arm PX4, which then took off with
        the mission already failed, 2026-10-20 sim). None on timeout."""
        try:
            return self.call(client, req, timeout=self.g('mav_timeout'))
        except Abort as e:
            if self.abort_reason:
                raise
            self.get_logger().warn(f'{e}: checking the vehicle state instead')
            return None

    def airborne(self):
        """Armed, watched for airborne_wait s: one /mavros/state can be stale. Height alone doesn't count: a
        disarmed drone resting where PX4's height reads > 0.5 m had the failsafe 'fly' it forever (2026-10-21)."""
        end = time.monotonic() + self.g('airborne_wait')
        while time.monotonic() < end:
            if self.mav.armed:
                return True
            time.sleep(0.2)
        return False

    def trigger(self, name, timeout=10.0, required=True):
        if not required:
            # housekeeping (search / explore stop): an answer that doesn't come must not abort the flight - a timed-out
            # /delivery/search/stop after the target was read landed the drone in the zone (2026-10-27 sim)
            try:
                return self.call(self.srv[name], Trigger.Request(), timeout)
            except Abort as e:
                self.get_logger().warn(f'{name}: {e} - carrying on')
                return Trigger.Response(success=False, message=str(e))
        r = self.call(self.srv[name], Trigger.Request(), timeout)
        if required and not r.success:
            raise Abort(f'{name}: {r.message}')
        return r

    def fly(self, x, y, alt, deadline, yaw=None, vmax=1.5, tol=0.4):
        yaw = self.pose()[3] if yaw is None else yaw
        alt = min(alt, self.g('max_altitude'))
        while not self.flight.fly(x, y, alt, yaw, vmax=vmax, tol=tol, timeout=10.0):
            self.check(deadline)

    # ---- mission
    def on_start(self, req, res):
        if self.running:
            res.success, res.message = False, f'already running ({self.state})'
            return res
        if self.odom is None or not self.mav.connected:
            res.success, res.message = False, 'no PX4 pose / MAVROS connection yet'
            return res
        self.running, self.abort_reason, self.target, self.geofence = True, None, None, None
        self.landing = False
        self.crashed = self.operator = False
        threading.Thread(target=self.run, daemon=True).start()
        res.success, res.message = True, 'mission started'
        return res

    def on_mav_state(self, m):
        # a MAVROS connection timeout reports disarmed, mode '': keep the last connected state (land() would take it for
        # touchdown). Not counted as fresh either, so warn_stale_mav still speaks up
        if not m.connected and self.mav.connected:
            return
        self.mav, self.mav_time = m, time.monotonic()

    def warn_stale_mav(self):
        """/mavros/state is 1 Hz: silent for long means host <-> container DDS lost it (2026-10-21 sim: the take-off
        loop kept re-arming an armed, hovering drone for 24 min with nothing in the log)."""
        age = time.monotonic() - self.mav_time
        if age > 5.0:
            self.get_logger().warn(f'/mavros/state silent for {age:.0f} s: the vehicle state may be stale',
                                   throttle_duration_sec=30.0)

    def on_corridor_state(self, m):
        self.corridor_state = m.data
        if self.in_corridor and m.data.startswith('WINDOW'):
            self.window_seen = True

    def nav2_use(self, which):
        """Activate the Nav2 `which` ('corridor' | 'delivery'), pause the other. Never fatal: a slow lifecycle call
        is retried, then logged - the mission goes on (the paused one only costs CPU)."""
        for name in sorted(self.nav2_managers, key=lambda n: n == which):      # pause first, then activate
            want = name == which and (name != 'corridor' or self.g('corridor_nav2'))
            if self.nav2_active[name] == want:
                continue
            if not want and not self.nav2_managers[name].wait_for_service(timeout_sec=2.0):
                # not launched (container_all corridor_nav2:=false, the default): nothing to pause. The retries below
                # would wait 3 x 120 s for it
                self.nav2_active[name] = False
                continue
            command = ManageLifecycleNodes.Request.RESUME if want else ManageLifecycleNodes.Request.PAUSE
            for attempt in range(1, 4):
                try:
                    r = self.call(self.nav2_managers[name], ManageLifecycleNodes.Request(command=command),
                                  timeout=120.0)
                    if r.success:
                        self.nav2_active[name] = want
                        self.get_logger().info(f'Nav2 {name} {"active" if want else "paused"}')
                        break
                    self.get_logger().warn(f'Nav2 {name} {"resume" if want else "pause"} refused: try {attempt}')
                except Abort as e:
                    self.get_logger().warn(f'Nav2 {name} {"resume" if want else "pause"}: {e} (try {attempt})')
                time.sleep(2.0)

    def record_trail(self):
        """Breadcrumbs of the corridor run in progress, for the failsafe to fly back out."""
        if not self.in_corridor or self.odom is None:
            return
        x, y, z, _ = self.pose()
        if not self.trail or math.dist(self.trail[-1], (x, y, z)) >= self.g('trail_step'):
            self.trail.append((x, y, z))

    def retrace(self):
        """Fly the corridor breadcrumbs backwards at their heights, heading kept. False if a leg stalls."""
        trail, yaw = list(reversed(self.trail)), self.pose()[3]
        self.get_logger().warn(f'FAILSAFE: retracing {len(trail)} breadcrumbs out of the corridor')
        for x, y, z in trail:
            if self.crashed or self.operator:
                return False
            if not self.flight.fly(x, y, z, yaw, vmax=self.g('retrace_speed'), tol=0.25, timeout=120.0):
                return False
        return True

    # the modes this mission sets itself (GUIDED / OFFBOARD, LAND; PX4's take-off): ANY other mode while it flies was set
    # by someone else - the RC mode switch (channel 7: STABILIZE / AUTO / RTL, user 2026-10-30), the ground's MANUAL /
    # LAND / RTL buttons, or an autopilot failsafe (battery / radio -> RTL / LAND)
    OWN_MODES = {'GUIDED', 'OFFBOARD', 'LAND', 'AUTO.LAND', 'AUTO.TAKEOFF', 'AUTO.LOITER'}

    def check_takeover(self):
        """Another mode while the mission flies: the pilot / the autopilot has it - stop at once, no failsafe flight
        (which would put the autopilot back in GUIDED and take the drone from the pilot / undo an RTL)."""
        own = self.OWN_MODES if getattr(self, 'landing', False) else self.OWN_MODES - {'LAND', 'AUTO.LAND'}
        if self.operator or not self.running or not self.mav.armed or not self.mav.mode or self.mav.mode in own:
            return
        self.operator = True
        self.abort_reason = f'operator took over: {self.mav.mode} (the RC switch, the ground or an autopilot failsafe)'
        self.get_logger().warn(f'{self.mav.mode} while the mission flies: handed over, no more setpoints')

    def check_crash(self):
        """Flipped / crashed while armed: kill the motors at once (non-blocking), abort the mission."""
        if self.odom is None or not self.mav.armed:
            self.tilt_since = None
            return
        q = self.odom.pose.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        tilt = max(abs(roll), abs(pitch))
        if tilt > math.radians(self.g('upset_angle')) and self.running and not self.abort_reason \
                and self.state not in ('FAILSAFE', 'LAND', 'FAILED', 'DONE'):
            self.get_logger().error(f'UPSET: roll {math.degrees(roll):.0f} deg, pitch {math.degrees(pitch):.0f} deg '
                                    f'(collision?): aborting')
            self.abort_reason = f'upset: tilt {math.degrees(tilt):.0f} deg (collision?)'
        if tilt < math.radians(self.g('flip_angle')):
            self.tilt_since = None
            return
        now = time.monotonic()
        self.tilt_since = self.tilt_since or now
        if now - self.tilt_since < self.g('flip_time'):
            return
        if not self.crashed:
            self.get_logger().error(f'CRASH: roll {math.degrees(roll):.0f} deg, pitch {math.degrees(pitch):.0f} deg '
                                    f'while armed: force disarm')
            self.log_pub.publish(String(data=json.dumps({'time': round(time.time(), 2), 'state': 'CRASH',
                                                         'roll_deg': round(math.degrees(roll)),
                                                         'pitch_deg': round(math.degrees(pitch))})))
        self.crashed = True
        self.abort_reason = self.abort_reason or 'crash: drone flipped'
        if self.px4_command.service_is_ready():          # repeated every 0.1 s until PX4 reports disarmed
            self.px4_command.call_async(CommandLong.Request(command=400, param1=0.0, param2=21196.0))

    def on_gripper(self, req, res):
        if self.running:
            res.success, res.message = False, f'mission running ({self.state})'
            return res
        try:
            res.success, res.message = self.gripper(release=req.data)
        except Abort as e:
            res.success, res.message = False, str(e)
        return res

    # ---- drop mechanism (PX4 gripper)
    GRIPPER_RELEASE, GRIPPER_GRAB = 0, 1             # MAVLink GRIPPER_ACTIONS
    ACK_TEXT = {0: 'accepted', 1: 'temporarily rejected', 2: 'denied', 3: 'unsupported', 4: 'failed',
                5: 'in progress', 6: 'cancelled'}

    def gripper(self, release):
        """MAV_CMD_DO_GRIPPER (211): param1 instance 1, param2 action. MAVROS returns PX4's FIRST ack: IN_PROGRESS
        (5) for an action it starts (ACCEPTED follows after PD_GRIPPER_TO, unseen here), ACCEPTED (0) if the gripper
        already is in that state. Either is OK; then wait gripper_timeout for the servo. -> (ok, message)"""
        action = self.GRIPPER_RELEASE if release else self.GRIPPER_GRAB
        req = CommandLong.Request(command=211, param1=1.0, param2=float(action))
        r = self.call(self.px4_command, req, timeout=self.g('mav_timeout'))
        text = self.ACK_TEXT.get(r.result, str(r.result))
        if not r.success and r.result not in (0, 5):
            return False, f'gripper {"release" if release else "grab"}: {text}'
        if r.result == 5:
            time.sleep(self.g('gripper_timeout') + 0.3)
        return True, f'gripper {"released" if release else "grabbed"} ({text})'

    def drop(self, deadline):
        """The payload drop with the configured mechanism (drop_mechanism) -> released?"""
        if self.g('drop_mechanism') == 'rc5':
            return self.drop_rc5(deadline)
        if self.g('drop_mechanism') == 'winch':
            return self.drop_winch(deadline)
        return self.drop_gripper(deadline)

    def aux(self, function, pos):
        """MAV_CMD_DO_AUX_FUNCTION (218): the autopilot does what an RC switch for that aux function would - param1
        the function (RCx_OPTION number), param2 the switch position 0 low / 1 middle / 2 high. -> (ok, message)"""
        req = CommandLong.Request(command=218, param1=float(function), param2=float(pos))
        r = self.call(self.px4_command, req, timeout=self.g('mav_timeout'))
        if not r.success and r.result != 0:
            return False, f'aux function {int(function)} {"high" if pos == 2 else "low"}: ' \
                          f'{self.ACK_TEXT.get(r.result, str(r.result))}'
        return True, f'aux function {int(function)} {"high" if pos == 2 else "low"}'

    def drop_rc5(self, deadline):
        """The drop on the transmitter's channel 5 (user, 2026-10-30): RC5 is RC5_OPTION 28 = Relay On/Off (RELAY1,
        pin 54) on this Pixhawk - the mission switches it the way RC5 high does (DO_AUX_FUNCTION rc5_aux_function,
        high), holds the drop height rc5_drop_hold s (the winch and drop mechanism need 45 s), then low. -> released?
        An abort / hand-over during the hold leaves the relay as it is: the pilot's RC5 switch decides."""
        alt, fn = self.g('deliver_altitude'), self.g('rc5_aux_function')
        for attempt in range(1, int(self.g('drop_retries')) + 1):
            self.check(deadline)
            try:
                ok, msg = self.aux(fn, 2)
            except Abort as e:                       # MAVROS command service missing / timed out
                ok, msg = False, str(e)
            self.set_state('DROP', attempt=attempt, result=msg, mechanism='RC5 (relay 1)')
            if ok:
                break
            self.flight.hold(1.0, alt)
        else:
            self.set_state('DROP_FAILED', reason=msg)
            return False
        end = self.clock_s() + self.g('rc5_drop_hold')   # node clock: sim time in the sim
        while self.clock_s() < end:
            self.check(deadline)
            self.flight.hold(0.2, alt)
        try:
            ok, msg = self.aux(fn, 0)
        except Abort as e:
            ok, msg = False, str(e)
        self.set_state('DROP', done=f'held {self.g("rc5_drop_hold"):.0f} s', result=msg)
        return True

    def drop_gripper(self, deadline):
        """Release the payload, wait drop_wait holding altitude, close the gripper again. -> released?"""
        alt = self.g('deliver_altitude')
        for attempt in range(1, int(self.g('drop_retries')) + 1):
            self.check(deadline)
            try:
                ok, msg = self.gripper(release=True)
            except Abort as e:                       # MAVROS command service missing / timed out
                ok, msg = False, str(e)
            self.set_state('DROP', attempt=attempt, result=msg)
            if ok:
                break
            self.flight.hold(1.0, alt)
        else:
            self.set_state('DROP_FAILED', reason=msg)
            return False
        end = self.clock_s() + self.g('drop_wait')       # node clock: sim time in the sim (t_min = real life)
        while self.clock_s() < end:
            self.check(deadline)
            self.flight.hold(0.2, alt)
        try:
            ok, msg = self.gripper(release=False)
        except Abort as e:
            ok, msg = False, str(e)
        if not ok:
            self.get_logger().warn(f'could not close the gripper after the drop: {msg}')
        return True

    # ---- drop mechanism (winch: N20 + MDD20A on two ArduPilot relays)
    def relay(self, instance, on):
        """MAV_CMD_DO_SET_RELAY (181): param1 relay instance (0 = RELAY1), param2 0 / 1. ArduPilot answers FAILED
        for a relay that isn't set up (RELAYn_FUNCTION 1 + RELAYn_PIN). -> (ok, message)"""
        req = CommandLong.Request(command=181, param1=float(instance), param2=1.0 if on else 0.0)
        r = self.call(self.px4_command, req, timeout=self.g('mav_timeout'))
        if not r.success and r.result != 0:
            return False, (f'relay {int(instance) + 1} {"on" if on else "off"}: '
                           f'{self.ACK_TEXT.get(r.result, str(r.result))} (RELAY{int(instance) + 1}_FUNCTION 1 and '
                           f'_PIN set on the autopilot?)')
        return True, ''

    def winch_stop(self):
        """Brake the motor (MDD20A PWM low), whatever else fails. -> ok"""
        with self.winch_lock:
            if self.winch_timer is not None:
                self.winch_timer.cancel()
                self.winch_timer = None
        for _ in range(3):
            try:
                if self.relay(self.g('winch_relay_run'), False)[0]:
                    return True
            except Abort:
                pass
        self.get_logger().error('WINCH: could not stop the motor (relay off refused)')
        return False

    def winch_run(self, down, length, deadline=None, alt=None):
        """Pay out (down) or wind in `length` m, open loop (time = length / winch_speed). In flight (alt given) it
        holds alt meanwhile and checks the deadline / abort; on the bench it just waits. Updates winch_out. The
        motor is braked on the way out, whatever happens. -> (ok, message)"""
        if length <= 0.0:
            return True, 'nothing to run'
        dir_level = bool(self.g('winch_dir_down')) == bool(down)
        ok, msg = self.relay(self.g('winch_relay_dir'), dir_level)
        if not ok:
            return False, msg
        time.sleep(0.05)                                  # DIR settled before the motor runs
        speed = self.g('winch_speed')
        sign = 1.0 if down else -1.0
        t0 = time.monotonic()
        last = t0
        try:
            ok, msg = self.relay(self.g('winch_relay_run'), True)
            if not ok:
                return False, msg
            end = t0 + length / speed
            while True:
                now = time.monotonic()
                self.winch_out = max(0.0, self.winch_out + sign * (now - last) * speed)
                last = now
                if now >= end:
                    break
                if alt is not None:
                    self.check(deadline)
                    self.flight.hold(min(0.2, end - now), alt)
                else:
                    time.sleep(min(0.05, end - now))
        finally:
            self.winch_stop()
            now = time.monotonic()
            self.winch_out = max(0.0, self.winch_out + sign * (now - last) * speed)
        return True, f'{"out" if down else "in"} {length:.2f} m in {time.monotonic() - t0:.1f} s'

    def winch_rewind_async(self):
        """After an aborted drop: wind the line in on a timer while the failsafe flies (the drone must not land on
        its own line / swinging hook). Not after a crash."""
        out = self.winch_out
        if out <= 0.05 or self.crashed:
            return
        try:
            if not self.relay(self.g('winch_relay_dir'), not bool(self.g('winch_dir_down')))[0]:
                return
            time.sleep(0.05)
            if not self.relay(self.g('winch_relay_run'), True)[0]:
                return
        except Abort:
            self.winch_stop()
            return
        t = out * self.g('winch_retract_ratio') / self.g('winch_speed')
        self.get_logger().warn(f'WINCH: winding in {out:.2f} m ({t:.0f} s) after the aborted drop')

        def done():
            with self.winch_lock:
                self.winch_timer = None
            self.winch_stop()
            self.winch_out = max(0.0, out - t * self.g('winch_speed'))
        with self.winch_lock:
            self.winch_timer = threading.Timer(t, done)
            self.winch_timer.daemon = True
            self.winch_timer.start()

    def drop_winch(self, deadline):
        """Lower the payload on the line until it sits on the ground and the line goes slack (the passive hook lets
        go), hold winch_release_wait, wind the line back in. Length = height - winch_stowed_drop + winch_slack,
        capped at winch_line - winch_line_reserve. -> released? (= the full length went out: no sensor confirms the
        hook opened)"""
        alt = self.g('deliver_altitude')
        z = self.pose()[2]
        need = max(0.0, z - self.g('winch_stowed_drop')) + self.g('winch_slack')
        cap = self.g('winch_line') - self.g('winch_line_reserve')
        length = min(need, cap)
        if need > cap:
            self.get_logger().warn(f'WINCH: {need:.2f} m needed from {z:.2f} m, only {cap:.2f} m of line: the payload '
                                   'may not reach the ground (the hook only opens on a slack line)')
        self.set_state('DROP', mechanism='winch', height=round(z, 2), lower_m=round(length, 2),
                       eta_s=round(length / self.g('winch_speed'), 1))
        try:
            msg = ''
            for attempt in range(1, int(self.g('drop_retries')) + 1):
                self.check(deadline)
                ok, msg = self.winch_run(True, length - self.winch_out, deadline, alt)
                if ok:
                    break
                self.set_state('DROP', attempt=attempt, result=msg)
                self.flight.hold(1.0, alt)
            else:
                self.set_state('DROP_FAILED', reason=msg, line_out=round(self.winch_out, 2))
                self.winch_run(False, self.winch_out * self.g('winch_retract_ratio'), deadline, alt)
                return False
            self.set_state('DROP', result=f'payload lowered ({msg}), line out {self.winch_out:.2f} m')
            end = time.monotonic() + self.g('winch_release_wait')
            while time.monotonic() < end:
                self.check(deadline)
                self.flight.hold(0.2, alt)
            back = self.winch_out * self.g('winch_retract_ratio')
            ok, msg = self.winch_run(False, back, deadline, alt)
            self.set_state('DROP', result=f'line wound in ({msg})' if ok else f'wind-in failed: {msg}',
                           line_out=round(self.winch_out, 2))
            return True
        except BaseException:
            self.winch_stop()
            self.winch_rewind_async()
            raise

    def on_winch(self, req, res):
        if self.running:
            res.success, res.message = False, f'mission running ({self.state})'
            return res
        try:
            n = self.g('winch_test_length')
            res.success, res.message = self.winch_run(bool(req.data), n)
            res.message += f' (line out ~{self.winch_out:.2f} m)'
        except Abort as e:
            self.winch_stop()
            res.success, res.message = False, str(e)
        return res

    def on_abort(self, req, res):
        self.abort_reason = 'abort requested'
        res.success, res.message = True, 'aborting'
        return res

    def on_qrs(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        qrs, start = d.get('qrs') or [], d.get('start_qr') or ''
        # the target's payload read but not yet confirmed: not "all read, none matched" - the 5th board read WAS the
        # target, 1 s before qr_locator's confirm_hits, and the search ended without the drop (2026-10-28 sim)
        self.target_seen = self.target_seen or any(start and q.get('payload') == start and q.get('in_zone', True)
                                                   for q in qrs)
        for q in qrs:                                 # in_zone: not the pad's start QR
            if (q.get('payload') and q.get('in_zone', True) and not q['payload'].startswith('raw')
                    and not any(math.dist((q['x'], q['y']), b) < 2.0 for b in self.qr_read)):
                self.qr_read.append((q['x'], q['y']))

    def on_operator(self, req, res):
        self.operator = True
        if self.running:
            self.abort_reason = 'operator took over (LAND / RTL / KILL / a hand-over mode from the ground)'
        res.success, res.message = True, 'mission handed to the operator' if self.running else 'no mission running'
        return res

    def clock_s(self):
        """The node clock in s: SIM time in the sim (use_sim_time), real time on the drone. Step limits and the
        mission timer run on it - on the wall clock a sim at 0.3x real time "took too long" landing a drone that was
        1.7 m up and descending normally (2026-10-27)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def limit(self, name):
        return self.clock_s() + self.g(f'limit_{name}')

    def set_params(self, client, **kv):
        self.call(client, SetParameters.Request(parameters=[Parameter(name=k, value=pval(v)) for k, v in kv.items()]))

    def align(self, align_state, **params):
        """entry_align with these parameters (it doesn't start corridor_nav: run_corridor() does, after the checks)
        -> its /nav/entry/result, or None if it FAILED."""
        self.set_params(self.entry_params, run_corridor=False, align_altitude=self.g('align_altitude'),
                        corridor_altitude=self.g('tunnel_altitude'), **params)
        self.set_state(align_state)
        deadline = self.limit('align')
        self.entry_state, self.entry_result = '', None
        self.trigger('/nav/entry/align')
        # THIS run active first (a 'DONE' of the last align still in flight was taken for this one's, 2026-10-24 sim)
        active = ('STARTING', 'SEARCH', 'LOCATE', 'STANDOFF', 'AXIS')
        end = time.monotonic() + 15.0                # capped: a run failing within ms may never show as active
        while not self.entry_state.startswith(active) and time.monotonic() < end:
            self.check(deadline)
            time.sleep(0.05)
        self.wait(lambda: self.entry_state.startswith(('DONE', 'FAILED')), deadline)
        if self.entry_state.startswith('FAILED'):
            self.get_logger().warn(f'{align_state}: {self.entry_state}')
            return None
        end = time.monotonic() + 5.0                 # published just before DONE (not self.wait: no_timeout)
        while self.entry_result is None and time.monotonic() < end:
            time.sleep(0.05)
        if self.entry_result is None:
            raise Abort(f'{align_state}: DONE but no /nav/entry/result')
        return self.entry_result

    def run_corridor(self, entrance, corridor_state):
        """corridor_nav from an entry_align result; returns the corridor length."""
        (ex, ey), yaw = entrance['entrance'], math.radians(entrance['yaw_deg'])
        params = dict(entry_x=float(ex), entry_y=float(ey), entry_yaw=yaw, altitude=self.g('tunnel_altitude'),
                      corridor_speed=self.g('entry_speed' if corridor_state == 'ENTRY_CORRIDOR' else 'return_speed'),
                      transit_altitude=self.g('align_altitude'),
                      altitude_mode=self.g('entry_altitude_mode') if corridor_state == 'ENTRY_CORRIDOR'
                      else self.g('return_altitude_mode'))
        if 'width_m' in entrance:                    # the measured corridor, not the default
            params['half_width'] = float(entrance['width_m']) / 2
        # exit detection not before: the return corridor runs parallel beside the entry one, so about as long. Its
        # walls dropped out of the SLAM map behind a window panel and the exit was called at 6.9 of 10 m
        # (2026-10-23 sim); out by luck - inside, RETURN_HOME's straight line home crosses the walls
        params['min_length'] = self.g('min_corridor_length')
        if corridor_state == 'ENTRY_CORRIDOR':
            # GPS + geofence: the entry corridor ends at the delivery zone - no exit before 1 m short of it, whatever
            # the SLAM map shows (missing walls called the exit at 4.0 of 6.7 m, 2026-10-24 sim)
            s_zone = self.axis_into_zone((ex, ey), yaw)
            if s_zone is not None:
                params['min_length'] = max(params['min_length'], s_zone - 1.0)
        gf = self.geofence
        if corridor_state == 'RETURN_CORRIDOR' and gf and gf.get('length_m'):
            params['min_length'] = max(params['min_length'], float(gf['length_m']) - self.g('length_slack'))
        self.set_params(self.corridor_params, **params)
        self.set_state(corridor_state, entrance=[ex, ey], yaw_deg=entrance['yaw_deg'])
        deadline = self.limit('corridor')
        self.trail, self.in_corridor, self.window_seen = [], True, False
        self.trigger('/nav/corridor/run')
        # wait for THIS run to be active (the topic still carries the last run's DONE). Not "not
        # startswith(('DONE', ..., ''))": every string starts with '' - the first mission run waited forever (2026-10-19)
        active = ('WAIT_NAV2', 'STAGE', 'DESCEND', 'VIO_RESET', 'CORRIDOR', 'EXIT')
        self.wait(lambda: self.corridor_state.startswith(active), deadline)
        self.wait(lambda: self.corridor_state.startswith(('DONE', 'FAILED')), deadline)
        if self.corridor_state.startswith('FAILED'):
            raise Abort(f'{corridor_state}: {self.corridor_state}')
        self.in_corridor = False
        try:
            return float(self.corridor_state.split('length=')[1].split()[0])
        except (IndexError, ValueError):
            return None

    # ---- entrance geofence
    def snapshot(self, path, timeout=3.0):
        """One frame of snapshot_topic -> path (PNG); False if none arrived."""
        frames = []
        sub = self.create_subscription(Image, self.g('snapshot_topic'), frames.append, 1)
        end = time.monotonic() + timeout
        while not frames and time.monotonic() < end:
            time.sleep(0.05)
        self.destroy_subscription(sub)
        if not frames:
            return False
        m = frames[0]
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = img[..., ::-1]
        return bool(cv2.imwrite(str(path), img))

    def gps_fix(self):
        g = self.gps
        if g is None or g.status.status < 0:
            return None
        return {'lat': g.latitude, 'lon': g.longitude, 'alt': round(g.altitude, 2)}

    def save_geofence(self):
        f = Path(self.g('geofence_file'))
        self.geofence_pub.publish(String(data=json.dumps(self.geofence)))
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(self.geofence, indent=2))
        except OSError as e:
            self.get_logger().error(f'could not save the geofence: {e}')

    def record_entrance(self, entrance):
        self.set_state('GEOFENCE')
        x, y, z, yaw = self.pose()
        png = Path(self.g('geofence_file')).with_suffix('.png')
        shot = self.snapshot(png)
        self.geofence = {
            'time': round(time.time(), 2),
            'entrance': entrance['entrance'], 'yaw_deg': entrance['yaw_deg'],
            'width_m': entrance.get('width_m'), 'banner': entrance.get('banner'),
            'gps': self.gps_fix(), 'local': [round(x, 3), round(y, 3), round(z, 3)],
            'heading_deg': round(math.degrees(yaw), 2), 'snapshot': str(png) if shot else None,
        }
        self.save_geofence()
        self.set_state('GEOFENCE', entrance=entrance['entrance'], gps=self.geofence['gps'], snapshot=shot)

    def axis_into_zone(self, entrance, yaw):
        """s along the corridor axis (from the entrance) where it enters the delivery zone rectangle, or None."""
        if self.qr_poly is not None:                 # the QR area polygon (None: not cut at the tunnel exit yet)
            return qr_area.ray_entry(entrance, yaw, self.qr_poly) if self.qr_final else None
        z = self.zone_rect()
        if z is None:
            return None
        x0, x1, y0, y1 = z
        ex, ey, c, sn = entrance[0], entrance[1], math.cos(yaw), math.sin(yaw)
        lo, hi = 0.0, 100.0
        for p, d, a, b in ((ex, c, x0, x1), (ey, sn, y0, y1)):      # slab intersection of the ray with the box
            if abs(d) < 1e-9:
                if not a <= p <= b:
                    return None
                continue
            t0, t1 = sorted(((a - p) / d, (b - p) / d))
            lo, hi = max(lo, t0), min(hi, t1)
        return lo if lo <= hi and lo > 0.5 else None

    def record_corridor_end(self, length):
        gf = self.geofence
        if gf is None:
            return
        yaw = math.radians(gf['yaw_deg'])
        if length is None:                           # no length reported: where the drone is now
            x, y = self.pose()[:2]
            length = (x - gf['entrance'][0]) * math.cos(yaw) + (y - gf['entrance'][1]) * math.sin(yaw)
        gf['length_measured_m'] = round(length, 2)
        # the corridor ends where its axis enters the delivery zone (the geofence given on the day): corridor_nav's
        # measure ran short (4.0 of 6.7 m: walls missing from the SLAM map) - the exit approach point then lay INSIDE
        # the entry tunnel and the drone came down at its mouth, blind to the exit banner (2026-10-24 sim)
        s_zone = self.axis_into_zone(gf['entrance'], yaw)
        if s_zone is not None:
            if abs(s_zone - length) > 0.5:
                self.get_logger().warn(f'corridor length: measured {length:.1f} m, zone boundary {s_zone:.1f} m - '
                                       f'using the zone boundary')
            length = s_zone
        elif self.qr_src == 'tunnel_exit' and self.course and abs(length - self.course['tunnel_length']) > 2.0:
            # the front wall of the QR area is cut here: a short measure (walls missing from the SLAM map) would move it
            # into the tunnel - the course length (dashboard) instead
            self.get_logger().warn(f'corridor length: measured {length:.1f} m, course {self.course["tunnel_length"]:g} m'
                                   f' - using the course length')
            length = float(self.course['tunnel_length'])
        gf['length_m'] = round(length, 2)
        gf['corridor_end'] = [round(gf['entrance'][0] + length * math.cos(yaw), 3),
                              round(gf['entrance'][1] + length * math.sin(yaw), 3)]
        gf['gps_end'] = self.gps_fix()
        if self.qr_src == 'tunnel_exit' and not self.qr_final:
            self.cut_qr_area(gf['corridor_end'], yaw)
            gf['qr_polygon'] = [[round(x, 2), round(y, 2)] for x, y in self.qr_poly] if self.qr_final else None
        self.save_geofence()
        try:                                         # the explorer makes no candidate detours next to it
            self.set_params(self.explorer_params, tunnel_end=[float(gf['corridor_end'][0]), float(gf['corridor_end'][1])])
        except Abort as e:
            self.get_logger().warn(f'explorer tunnel_end not set ({e})')

    def load_qr_area(self):
        """The QR area of the ground station's geofence: given (delivery_polygon) or to be cut from the arena at the
        entry tunnel's exit (qr_source tunnel_exit). Sim world files have neither: the zone rectangle as before."""
        self.qr_poly, self.qr_final, self.qr_src, self.arena, self.qr_back = None, False, None, [], []
        try:
            w = json.load(open(self.g('world_json')))
        except (OSError, ValueError):
            return
        if len(w.get('delivery_polygon') or []) < 3:
            return
        self.qr_src = w.get('qr_source') or 'given'
        self.qr_poly = [tuple(map(float, p)) for p in w['delivery_polygon']]
        self.qr_final = self.qr_src == 'given'
        self.arena = [tuple(map(float, p)) for p in w.get('arena_polygon') or []]
        self.qr_back = [tuple(map(float, p)) for p in w.get('qr_back') or []]
        self.publish_qr_area(False)

    def publish_qr_area(self, lock):
        """/mission/qr_area: lock = offboard_bridge keeps the drone inside it (search .. exit-tunnel sequence; user
        2026-10-28: once in the QR area it may not leave before that). Not locked before the area is final."""
        if self.qr_poly is None:
            return
        lock = bool(lock and self.qr_final)
        self.qr_pub.publish(String(data=json.dumps({'polygon': [[round(x, 3), round(y, 3)] for x, y in self.qr_poly],
                                                    'lock': lock, 'source': self.qr_src, 'final': self.qr_final})))
        self.get_logger().info(f'QR area {"LOCKED" if lock else "published"} ({self.qr_src})')

    def cut_qr_area(self, exit_pt, yaw):
        """Arena only: the QR area = the arena's back + side walls and the line across the entry tunnel's exit."""
        poly = qr_area.front_from_tunnel_exit(self.arena, self.qr_back, exit_pt, yaw)
        if poly is None:
            self.get_logger().error(f'QR area: no closed area from the arena {self.arena}, back {self.qr_back} and the '
                                    f'tunnel exit {exit_pt}: searching the whole arena (no lock)')
            return
        self.qr_poly, self.qr_final = poly, True
        width = math.dist(*self.qr_back)
        self.set_state(self.state, qr_area=[[round(x, 2), round(y, 2)] for x, y in poly], back_wall_m=round(width, 1))
        if self.course and abs(width - self.course['qr_width']) > 2.0:
            self.get_logger().warn(f'QR area back wall {width:.1f} m, course says {self.course["qr_width"]:g} m wide')
        self.publish_qr_area(False)

    def into_zone(self):
        """Fly straight along the entry corridor's heading until 2 m past its end, at the tunnel height (v3: the roof
        - climbing starts only out there). A corridor run ending inside the mouth (x 14.8) left the drone behind the
        delivery map's no-fly boundary ring: the delivery Nav2 found no plan and two sweep lanes were skipped
        (2026-10-22 sim)."""
        gf = self.geofence
        if not gf or 'corridor_end' not in gf:
            return
        yaw = math.radians(gf['yaw_deg'])
        # ALWAYS to into_zone_dist past the corridor end (the zone boundary), back if the corridor run overshot: it
        # used to climb wherever the run ended - 0.3 m from a red zone 4.5 m past the tunnels (2026-10-24 sim)
        d = self.g('into_zone_dist')
        tx = gf['corridor_end'][0] + d * math.cos(yaw)
        ty = gf['corridor_end'][1] + d * math.sin(yaw)
        if math.dist(self.pose()[:2], (tx, ty)) < 0.5:
            return
        self.get_logger().info(f'into the zone: ({tx:.1f}, {ty:.1f}) first, at the tunnel height')
        self.fly(tx, ty, min(self.pose()[2], self.g('align_altitude')), self.limit('search'), yaw=yaw, vmax=0.8)

    def zone_rect(self):
        """The delivery zone (x0, x1, y0, y1) in odom, from the ground station's / sim world's json (= the nodes').
        The QR area still to be cut at the entry tunnel's exit (arena only): None."""
        if self.qr_poly is not None:
            return qr_area.bbox(self.qr_poly) if self.qr_final else None
        try:
            z = json.load(open(self.g('world_json')))['delivery_zone']
            return z['x_from'], z['x_to'], z['y_from'], z['y_to']
        except (OSError, ValueError, KeyError):
            return None

    def zone_point(self, xy, inset=1.0):
        """The point of the zone (inset m inside its edge) nearest xy; xy itself if the zone is unknown."""
        z = self.zone_rect()
        if z is None:
            return xy
        x0, x1, y0, y1 = z
        return (min(max(xy[0], x0 + inset), x1 - inset), min(max(xy[1], y0 + inset), y1 - inset))

    def zone_home(self, home, t_in_zone):
        """Delivery only: Nav2 (red zones avoided) to the zone edge nearest home, then straight home."""
        self.set_state('ZONE_EXIT', qr_area_s=round(t_in_zone, 1))
        self.publish_qr_area(False)
        deadline = self.limit('home')
        edge = self.zone_point(home, inset=1.0)
        for attempt in range(1, 3):
            try:
                self.navigate_delivery(edge, deadline)
                break
            except Abort as e:
                if self.abort_reason:
                    raise
                self.get_logger().warn(f'ZONE_EXIT: {e} (try {attempt})')
        self.set_state('RETURN_HOME')
        x, y, z, _ = self.pose()
        self.fly(home[0], home[1], max(z, self.g('search_altitude')), deadline)

    def exit_approach(self):
        """The first exit approach point (corridor end + exit_approach_dist along the entry heading), or None."""
        gf = self.geofence
        if not gf or 'corridor_end' not in gf:
            return None
        yaw = math.radians(gf['yaw_deg'])
        return (gf['corridor_end'][0] + self.g('exit_approach_dist') * math.cos(yaw),
                gf['corridor_end'][1] + self.g('exit_approach_dist') * math.sin(yaw))

    def to_exit(self):
        """Back to the corridor side of the zone with the delivery Nav2 (red zones avoided). Never fatal."""
        gf = self.geofence
        if not gf or 'corridor_end' not in gf:
            return
        yaw = math.radians(gf['yaw_deg'])
        ax = gf['corridor_end'][0] + self.g('exit_approach_dist') * math.cos(yaw)
        ay = gf['corridor_end'][1] + self.g('exit_approach_dist') * math.sin(yaw)
        x, y = self.pose()[:2]
        if math.hypot(ax - x, ay - y) < 3.0:
            return
        # the measured corridor runs ~1 m long (7.6 of 6.7 m), so the point can land in a red zone's no-fly margin:
        # Nav2 refuses it - nearer to the corridor then (2026-10-24 sim: 3 refusals, a minute lost)
        # nearer only while still inside the zone: never back over the tunnels (corridor_end = the zone boundary)
        for dist in (self.g('exit_approach_dist'), 2.0, 1.5):
            ax = gf['corridor_end'][0] + dist * math.cos(yaw)
            ay = gf['corridor_end'][1] + dist * math.sin(yaw)
            self.set_state('TO_EXIT', to=[round(ax, 2), round(ay, 2)])
            try:
                self.navigate_delivery((ax, ay), self.limit('home'))
                break
            except Abort as e:
                if self.abort_reason:
                    raise
                self.get_logger().warn(f'TO_EXIT: {e} at {dist} m past the corridor: nearer')
                time.sleep(1.0)
        x, y, _, _ = self.pose()                     # down to the banner-search height here, in the open zone
        self.fly(x, y, self.g('align_altitude'), self.limit('home'))

    def is_entry_corridor(self, found):
        """True if an entry_align result is the entry corridor again (centre line within exclude_radius)."""
        gf = self.geofence
        if gf is None:
            return False
        yaw = math.radians(gf['yaw_deg'])
        n = (-math.sin(yaw), math.cos(yaw))
        dx, dy = found['entrance'][0] - gf['entrance'][0], found['entrance'][1] - gf['entrance'][1]
        parallel = abs(math.cos(math.radians(found['yaw_deg']) - yaw)) > 0.9
        return parallel and abs(dx * n[0] + dy * n[1]) < self.g('exclude_radius')

    def find_exit(self):
        """The return corridor: the exit banner (exit_colour), checked against the entrance geofence; else the geofence."""
        self.publish_qr_area(False)                  # the exit-tunnel sequence: the QR area may be left now
        # the corridors are parallel: with the entrance geofence, only wall pairs parallel to the entry corridor
        gf = self.geofence
        axis_yaw = math.radians(gf['yaw_deg']) if gf else float('nan')
        if gf:
            found = self.align('ALIGN_EXIT', colour=self.g('exit_colour'), mode='banner', axis_yaw=axis_yaw,
                               banner_exclude=True,
                               exclude_x=gf['entrance'][0], exclude_y=gf['entrance'][1],
                               exclude_radius=self.g('exclude_radius'))
        else:
            found = self.align('ALIGN_EXIT', colour=self.g('exit_colour'), mode='banner', axis_yaw=axis_yaw,
                               banner_exclude=False)
        if found and self.is_entry_corridor(found):
            self.set_state('ALIGN_EXIT', rejected=found['entrance'], reason='that is the entry corridor')
            found = None
        if found and gf and abs(math.cos(math.radians(found['yaw_deg'] - gf['yaw_deg']))) < \
                math.cos(math.radians(self.g('exit_parallel_deg'))):
            self.set_state('ALIGN_EXIT', rejected=found['entrance'], yaw_deg=found['yaw_deg'],
                           reason='not parallel to the entry corridor')
            found = None
        if found:
            return found
        gf = self.geofence
        if not (self.g('geofence_fallback') and gf and 'corridor_end' in gf):
            raise Abort(f'ALIGN_EXIT: {self.entry_state or "rejected"} and no entrance geofence to fall back on')
        yaw = math.radians(gf['yaw_deg'])
        found = self.align('ALIGN_GEOFENCE', mode='geofence',
                           predict_x=gf['corridor_end'][0], predict_y=gf['corridor_end'][1], predict_yaw=yaw + math.pi,
                           exclude_x=gf['entrance'][0], exclude_y=gf['entrance'][1], exclude_yaw=yaw,
                           exclude_radius=self.g('exclude_radius'), geofence_max_lateral=5.0)
        if not found:
            raise Abort(f'ALIGN_GEOFENCE: {self.entry_state}')
        return found

    def on_overhead(self, m):
        d = json.loads(m.data)
        if d.get('colour') == 'none':
            self.overhead_frames += 1
        elif d.get('colour') == 'green':            # 2026-10-30: no banner marks the other tunnel any more, and
            #                                           the start zone is a red area under the drone: red is ignored
            self.overhead_dets = (self.overhead_dets + [d])[-200:]

    @staticmethod
    def ground(origin, ray, z=0.0):
        """Where a ray from the camera meets the horizontal plane at height z (odom), or None."""
        if ray[2] > -0.05 or origin[2] <= z:
            return None
        t = (z - origin[2]) / ray[2]
        return np.array([origin[0] + t * ray[0], origin[1] + t * ray[1]])

    def overhead_look(self):
        """Hover overhead_look s -> (green fix or None, red centres). A green fix: the floor banner's centre and the
        direction of its long edge (mod 180 deg) on the ground, from its corner rays (the lower banner lies flat at the
        mouth like a doormat - user, 2026-10-30)."""
        self.overhead_dets, self.overhead_frames = [], 0
        t0, c0 = time.monotonic(), self.clock_s()
        while (self.clock_s() - c0 < self.g('overhead_look') or self.overhead_frames < self.g('overhead_frames')) \
                and time.monotonic() - t0 < self.g('overhead_look_max'):
            if self.abort_reason:
                raise Abort(self.abort_reason)
            time.sleep(0.05)
        lo, hi = self.g('overhead_banner_len')
        greens, reds, seen = [], [], []
        for d in self.overhead_dets:
            c = self.ground(d['origin'], d['ray'])
            if c is None:
                continue
            if d['colour'] == 'red':
                reds.append(c)
                continue
            q = [self.ground(d['origin'], r) for r in d.get('corner_rays') or []]
            if len(q) != 4 or any(v is None for v in q):
                seen.append(f'green@({c[0]:.1f},{c[1]:.1f}) corners {len(q)}')
                continue
            edges = [q[(i + 1) % 4] - q[i] for i in range(4)]
            k = 0 if np.linalg.norm(edges[0]) + np.linalg.norm(edges[2]) >= \
                np.linalg.norm(edges[1]) + np.linalg.norm(edges[3]) else 1
            e = edges[k] - edges[k + 2]                   # the two long edges run opposite ways round the quad
            length = np.linalg.norm(e) / 2
            seen.append(f'green@({c[0]:.1f},{c[1]:.1f}) {length:.1f}m')
            if lo <= length <= hi:
                greens.append((d['score'], c, math.atan2(e[1], e[0]), length))
        self.get_logger().info(f'overhead look: {self.overhead_frames} frames, {len(reds)} red, '
                               f'green {seen[:6]}')
        if not greens:
            return None, reds
        best = max(greens, key=lambda g: g[0])
        near = [g for g in greens if np.linalg.norm(g[1] - best[1]) < 1.0]
        a2 = sum(math.cos(2 * g[2]) for g in near), sum(math.sin(2 * g[2]) for g in near)
        return {'centre': np.mean([g[1] for g in near], axis=0), 'across': math.atan2(a2[1], a2[0]) / 2,
                'length': float(np.mean([g[3] for g in near])), 'n': len(near)}, reds

    def overhead_green(self, home, yaw0):
        """OVERHEAD_GREEN (see the docstring) -> an entry_align result (the D435 wall fit from the overhead fix), or
        None: then the D435 banner search as before."""
        alt = self.g('takeoff_altitude')
        deadline = self.limit('align')
        self.set_state('OVERHEAD_GREEN')
        try:
            self.set_params(self.overhead_params, active=True)
        except Abort as e:
            self.get_logger().warn(f'OVERHEAD_GREEN: overhead_banner not reachable ({e})')
            return None
        try:
            if self.qr_poly is not None:              # towards the QR area (the tunnels lie between)
                cx, cy = np.mean(self.qr_poly, axis=0)
                head = math.atan2(cy - home[1], cx - home[0])
            else:
                head = yaw0
            u, n = np.array([math.cos(head), math.sin(head)]), np.array([-math.sin(head), math.cos(head)])
            x0, y0, _, _ = self.pose()
            fix, reds = None, []
            for lat in self.g('overhead_lateral'):
                s = 0.0
                while fix is None and s <= self.g('overhead_search_dist') + 1e-6:
                    tx, ty = np.array([x0, y0]) + s * u + lat * n
                    self.fly(tx, ty, alt, deadline, yaw=head)
                    fix, r = self.overhead_look()
                    reds += r
                    s += self.g('overhead_step')
                if fix is not None:
                    break
            if fix is None:
                self.set_state('OVERHEAD_GREEN', result='no green banner seen from above')
                return None
            # the tunnel axis: across the banner's long edge, pointing away from home
            for _ in range(3):
                axis = fix['across'] + math.pi / 2
                if math.cos(axis) * (fix['centre'][0] - home[0]) + math.sin(axis) * (fix['centre'][1] - home[1]) < 0:
                    axis += math.pi
                a = np.array([math.cos(axis), math.sin(axis)])
                look = fix['centre'] - self.g('overhead_hold_back') * a     # start side: never over the roof
                self.fly(*look, alt, deadline, yaw=axis)
                again, r = self.overhead_look()
                reds += r
                if again is None:
                    break
                moved = float(np.linalg.norm(again['centre'] - fix['centre']))
                fix = again
                if moved < 0.3:
                    break
            axis = fix['across'] + math.pi / 2
            if math.cos(axis) * (fix['centre'][0] - home[0]) + math.sin(axis) * (fix['centre'][1] - home[1]) < 0:
                axis += math.pi
            a = np.array([math.cos(axis), math.sin(axis)])
            c = fix['centre']
            red = [q for q in reds if np.linalg.norm(q - c) > 1.5]
            red_c = np.mean(red, axis=0) if red else None
            self.set_state('OVERHEAD_GREEN', banner=[round(float(v), 2) for v in c], yaw_deg=round(math.degrees(axis), 1),
                           banner_len_m=round(fix['length'], 2), looks=fix['n'],
                           red=[round(float(v), 2) for v in red_c] if red_c is not None else None)
        finally:
            try:
                self.set_params(self.overhead_params, active=False)
            except Abort:
                pass
        # standoff before the mouth at 5 m, THEN down below 8 ft (outside the tunnel), then the D435 wall fit
        so = c - 3.0 * a
        self.fly(*so, alt, deadline, yaw=axis)
        self.fly(*so, self.g('align_altitude'), deadline, yaw=axis)
        ex = red_c if red_c is not None else c - 1000.0 * a          # far away: nothing excluded
        found = self.align('ALIGN_GREEN', mode='geofence', predict_x=float(c[0]), predict_y=float(c[1]),
                           predict_yaw=axis, exclude_x=float(ex[0]), exclude_y=float(ex[1]), exclude_yaw=axis,
                           exclude_radius=self.g('exclude_radius'), geofence_max_lateral=self.g('overhead_lateral_max'))
        if found:
            found.setdefault('banner', [round(float(c[0]), 3), round(float(c[1]), 3), 0.0])
            found['overhead'] = True
        return found

    def apply_course(self):
        """The course dimensions of the ground station's geofence (dashboard Course: the organisers' defaults, editable -
        user 2026-10-28) -> the heights near / in the tunnels, exit detection, the neighbour-tunnel test and the
        entry_align / corridor_nav widths. No course in the json (sim worlds): the parameters as they are."""
        try:
            c = json.load(open(self.g('world_json'))).get('course')
        except (OSError, ValueError):
            c = None
        if not isinstance(c, dict):                  # sim world files: 'course' is the course name ('v3')
            c = None
        self.course = c
        if not c:
            return
        cap = float(c['tunnel_ceiling']) - 0.14      # 8 ft (2.44 m) -> 2.3 m, as flown so far
        width, length = float(c['tunnel_width']), float(c['tunnel_length'])
        mine = {'align_altitude': min(self.g('align_altitude'), cap), 'home_altitude': min(self.g('home_altitude'), cap),
                'tunnel_altitude': min(self.g('tunnel_altitude'), cap - 0.3),
                'min_corridor_length': round(0.6 * length, 2),            # v3: 4.0 of 6.7 m
                'exclude_radius': round(0.6 * (width + float(c['tunnel_gap'])), 2)}   # < the centre-line spacing
        self.set_parameters([rclpy.parameter.Parameter(k, value=float(v)) for k, v in mine.items()])
        for name, client, kv in (('entry_align', self.entry_params, {'corridor_width': width}),
                                 ('corridor_nav', self.corridor_params, {'half_width': width / 2,
                                                                         'max_length': length + 5.0})):
            try:
                self.set_params(client, **kv)
            except Abort as e:
                self.get_logger().warn(f'course: {name} {kv} not set ({e})')
        self.get_logger().info(f'course {c}: {mine}')

    def run(self):
        self.t_start = self.clock_s()
        self.t0_clock = self.get_clock().now()
        home = None
        try:
            self.apply_course()
            self.load_qr_area()
            # TAKEOFF
            self.set_state('TAKEOFF')
            deadline = self.limit('takeoff')
            x, y, _, yaw0 = self.pose()
            home = (x, y)
            alt = self.g('takeoff_altitude')
            if self.ardupilot:
                self.takeoff_ardupilot(alt, deadline)
            else:
                self.takeoff_px4(alt, deadline)
            self.check(deadline)
            r = self.call(self.offboard, SetBool.Request(data=True))
            if not r.success:
                raise Abort(f'offboard refused: {r.message}')
            self.wait(lambda: self.mav.mode == self.modes['offboard'], deadline)
            x, y, _, _ = self.pose()
            self.fly(x, y, alt, deadline)

            # READ_START_QR: 1 m forward and read. Not seen: the QR may be behind us (the cameras look forward and
            # down) - the take-off heading isn't guaranteed (after a failsafe landing the drone faced 124 deg and
            # retried forever, 2026-10-20). Then 1 m out from the take-off point at each of 8 headings, twice round
            if self.g('start_from') == 'exit':
                self.nav2_use('corridor')
                x, y, _, _ = self.pose()
                self.fly(x, y, self.g('search_altitude'), self.limit('align'))
                self.fly(*self.g('search_start'), self.g('search_altitude'), self.limit('align'))
                self.run_corridor(self.find_exit(), 'RETURN_CORRIDOR')
                self.set_state('RETURN_HOME')
                deadline = self.limit('home')
                x, y, z, _ = self.pose()
                self.fly(x, y, max(z, self.g('home_altitude')), deadline)
                self.fly(home[0], home[1], max(z, self.g('home_altitude')), deadline)
                self.land()
                self.set_state('DONE', time_s=round(self.clock_s() - self.t_start, 1))
                return
            if self.g('read_start_qr'):
                self.set_state('READ_START_QR')
                deadline = self.limit('read_qr')
                payload = self.read_start_qr(home, alt, self.pose()[3], deadline)
                if not payload:
                    raise Abort('start QR not read at any heading')
                self.set_state('READ_START_QR', payload=payload)
            else:                                    # test runs away from the pad: qr_reader's stored read
                self.set_state('READ_START_QR', skipped='using the stored start QR')
            mode = self.g('start_from')
            overhead = mode not in ('search', 'delivery') and self.g('overhead_banner')
            if mode not in ('search', 'delivery') and not overhead:   # to the banner-search height where we are
                x, y, _, _ = self.pose()
                self.fly(x, y, self.g('align_altitude'), self.limit('align'))

            if mode not in ('search', 'delivery'):
                self.nav2_use('corridor')
            if mode == 'delivery':
                # DELIVERY ONLY: up where we are, then into the zone's map if outside it
                deadline = self.limit('search')
                x, y, _, _ = self.pose()
                self.fly(x, y, self.g('search_altitude'), deadline)
                inside = self.zone_point((x, y), inset=1.0)
                if math.dist(inside, (x, y)) > 0.5:
                    self.set_state('DIRECT_TO_ZONE', to=[round(v, 2) for v in inside])
                    self.fly(*inside, self.g('search_altitude'), deadline, vmax=1.5)
            elif mode == 'search':
                # test entry: straight to the delivery zone (a test-only route: above the walls)
                if self.g('return_corridor'):
                    raise Abort('start_from search needs return_corridor false (no entrance geofence)')
                self.set_state('DIRECT_TO_SEARCH', to=list(self.g('search_start')))
                deadline = self.limit('search')
                x, y, _, _ = self.pose()
                sa = self.g('search_altitude')
                self.fly(x, y, sa, deadline)
                self.fly(*self.g('search_start'), sa, deadline)
            else:
                # ENTRY CORRIDOR (green) + entrance geofence
                entrance = self.overhead_green(home, yaw0) if overhead else None
                if not entrance:
                    if overhead:                     # the D435 banner search, as before, from the banner-search height:
                        # back over the start QR first (the search may have ended over a tunnel roof), then down
                        z = self.pose()[2]
                        self.fly(home[0] + math.cos(yaw0), home[1] + math.sin(yaw0), z, self.limit('align'))
                        self.fly(home[0] + math.cos(yaw0), home[1] + math.sin(yaw0), self.g('align_altitude'),
                                 self.limit('align'))
                    entrance = self.align('ALIGN_GREEN', colour='green', mode='banner', axis_yaw=float('nan'),
                                          banner_exclude=False)
                if not entrance:
                    raise Abort(f'ALIGN_GREEN: {self.entry_state}')
                self.record_entrance(entrance)
                self.record_corridor_end(self.run_corridor(entrance, 'ENTRY_CORRIDOR'))

            if mode == 'tunnel':
                # TUNNELS + BANNERS ONLY: out of the entry corridor into the zone's edge, no search: the exit corridor
                self.into_zone()
                if self.g('delivery_nav2'):
                    self.nav2_use('delivery')
            else:
                # SEARCH_DELIVERY
                self.set_state('SEARCH_DELIVERY')
                self.into_zone()
                if self.g('delivery_nav2'):
                    self.nav2_use('delivery')             # else it stays paused: the search flies A* routes itself
                deadline = self.limit('search')
                x, y, _, _ = self.pose()
                self.fly(x, y, self.g('search_altitude'), deadline)
                self.target = None
                # the candidate route ends where the drone goes after the search: the exit approach / home's zone edge
                end = self.zone_point(home, inset=1.0) if mode == 'delivery' else self.exit_approach()
                if end is not None:
                    try:
                        self.set_params(self.explorer_params, route_end_x=float(end[0]), route_end_y=float(end[1]))
                    except Abort as e:
                        self.get_logger().warn(f'explorer route end not set ({e})')
                drop = self.g('drop')
                self.qr_read, self.target_seen = [], False
                self.publish_qr_area(True)               # in the QR area now: no leaving it before the exit sequence
                self.trigger('/delivery/map/reset')
                self.trigger('/delivery/search/start')
                self.trigger('/delivery/explore/start')
                search_end = self.g('mission_budget') - self.g('return_reserve') - self.g('deliver_reserve')
                zone_t0 = self.get_clock().now()
                zone_search = self.g('zone_time_limit') - self.g('deliver_reserve') - \
                    self.g('zone_home_reserve' if mode == 'delivery' else 'zone_exit_reserve')
                in_zone = lambda: (self.get_clock().now() - zone_t0).nanoseconds * 1e-9  # noqa: E731
                out_of_time = lambda: self.mission_elapsed() > search_end or in_zone() > zone_search  # noqa: E731
                # the target read ends the search (QR area only: then home and land; whole mission: the drop, then the
                # exit corridor), and so do all qr_count boards read - the target among them was missed then
                all_read = lambda: len(self.qr_read) >= self.g('qr_count') and not self.target_seen  # noqa: E731
                self.wait(lambda: self.target is not None or all_read() or self.explore_state.startswith('DONE')
                          or out_of_time(), deadline, 0.5)
                self.trigger('/delivery/explore/stop', required=False)
                delivered = False
                if self.target is not None and not drop:
                    self.set_state('SEARCH_DELIVERY', target=[round(v, 2) for v in self.target], drop='off (QR area only)')
                elif self.target is None:
                    # out of time or nothing left to explore: no drop, but still finish the course within the budget
                    self.set_state('SEARCH_DELIVERY', skipped=('QR-area time' if in_zone() > zone_search else 'time budget')
                                   if out_of_time() else f'all {len(self.qr_read)} QRs read, none matched the start QR'
                                   if all_read() else f'target not found ({self.explore_state})')
                else:
                    self.set_state('SEARCH_DELIVERY', target=[round(v, 2) for v in self.target])
                    delivered = True

                if not delivered:
                    self.trigger('/delivery/search/stop', required=False)
                else:
                    # DELIVER: Nav2 to the target (it knows the red zones), descend, hold/align, land
                    self.set_state('DELIVER', target=[round(v, 2) for v in self.target])
                    deadline = self.limit('deliver')
                    tx, ty = self.target
                    x, y, _, _ = self.pose()
                    if math.dist((x, y), (tx, ty)) > self.g('direct_deliver_radius'):
                        # the explorer cancels its own Nav2 goal on TARGET_FOUND; on a loaded system that cancel landed
                        # ~10 s later on OUR goal and aborted it (2026-10-20 sim): let it go idle first, retry an abort
                        end = time.monotonic() + 15.0
                        while not self.explore_state.startswith(('IDLE', 'TARGET_FOUND', 'DONE')) and time.monotonic() < end:
                            time.sleep(0.2)
                        for attempt in range(1, 4):
                            try:
                                self.navigate_delivery(self.target, deadline)
                                break
                            except Abort as e:
                                if self.abort_reason:
                                    raise
                                if attempt == 3:
                                    if math.dist(self.pose()[:2], (tx, ty)) > self.g('direct_fallback'):
                                        raise
                                    self.get_logger().warn(f'{e}: flying the short hop straight')
                                    break
                                self.get_logger().warn(f'{e}: retry {attempt}')
                                time.sleep(2.0)
                    # within direct_deliver_radius (the usual case: the target is read from right above it) a direct hop:
                    # the target is never in a red zone (qr_locator), so a short straight line can't cross one
                    # deliver_tol: the default 0.4 m arrival tolerance added ~0.25 m to a 0.61 m miss (2026-10-23 sim)
                    self.fly(tx, ty, self.g('search_altitude'), deadline, vmax=1.5, tol=self.g('deliver_tol'))
                    self.target_fine = None                  # reads from the drop height only
                    self.fly(tx, ty, self.g('deliver_altitude'), deadline, vmax=0.5, tol=self.g('deliver_tol'))
                    t_low = self.clock_s()                   # at the drop height: deliver_min_time from here
                    self.refine_over_target(tx, ty, deadline)
                    hold_end = self.clock_s() + self.g('deliver_hold')   # node clock (sim time in the sim)
                    while self.clock_s() < hold_end:
                        self.check(deadline)
                        self.flight.hold(0.2, self.g('deliver_altitude'))
                    released = self.drop(deadline)
                    # the winch + drop mechanism needs deliver_min_time at the drop height before the climb (user,
                    # 2026-10-30: 45 s) - whatever the open-loop winch timing says
                    while self.clock_s() < t_low + self.g('deliver_min_time'):
                        self.check(deadline)
                        self.flight.hold(0.2, self.g('deliver_altitude'))
                    self.set_state('DELIVER', delivered_at=[round(v, 2) for v in self.pose()[:2]], released=released)
                    self.trigger('/delivery/search/stop', required=False)

                    if mode == 'delivery':
                        self.fly(tx, ty, self.g('search_altitude'), deadline)
                    if not self.g('return_corridor') and mode != 'delivery':
                        self.get_logger().info('Target reached and aligned. Landing on target...')
                        self.land()
                        self.set_state('DONE', time_s=round(self.clock_s() - self.t_start, 1))
                        return

                    if mode != 'delivery':
                        self.fly(tx, ty, self.g('search_altitude'), deadline)
                if mode == 'delivery':
                    self.zone_home(home, in_zone())
                    self.land()
                    self.set_state('DONE', time_s=round(self.clock_s() - self.t_start, 1))
                    return
            self.to_exit()
            self.nav2_use('corridor')
            # RETURN CORRIDOR: exit banner, or the entrance geofence
            self.run_corridor(self.find_exit(), 'RETURN_CORRIDOR')

            # RETURN_HOME + LAND
            self.set_state('RETURN_HOME')
            deadline = self.limit('home')
            x, y, z, _ = self.pose()
            self.fly(x, y, max(z, self.g('home_altitude')), deadline)
            self.fly(home[0], home[1], max(z, self.g('home_altitude')), deadline)
            self.land()
            self.set_state('DONE', time_s=round(self.clock_s() - self.t_start, 1))
        except Abort as e:
            self.failsafe(str(e), home)
        except Exception as e:                       # any bug: still bring the drone home
            self.failsafe(f'error: {e}', home)
        finally:
            self.running = False

    def refine_over_target(self, tx, ty, deadline):
        """At the drop height: wait for qr_locator's fine fix (reads from here, hovering: no motion lag, twice the
        resolution of 10 m) and re-centre on it. No fix in fine_wait: drop at the 10 m fix (logged)."""
        alt = self.g('deliver_altitude')
        end = self.clock_s() + self.g('fine_wait')       # node clock (sim time in the sim)
        while self.clock_s() < end:
            self.check(deadline)
            f = self.target_fine
            if f and f['n'] >= self.g('fine_min_reads'):
                break
            self.flight.hold(0.2, alt)
        f = self.target_fine
        if not f or f['n'] < self.g('fine_min_reads'):
            self.set_state('DELIVER', fine='no fix from the drop height: the 10 m one')
            return
        jump = math.dist((f['x'], f['y']), (tx, ty))
        if jump > self.g('fine_max_jump'):
            self.set_state('DELIVER', fine=f'ignored ({jump:.2f} m from the 10 m fix)')
            return
        self.set_state('DELIVER', fine=[f['x'], f['y']], moved=round(jump, 2), reads=f['n'])
        self.fly(f['x'], f['y'], alt, deadline, vmax=0.3, tol=self.g('fine_tol'))

    def read_start_qr(self, home, alt, yaw0, deadline, rounds=2, steps=8):
        """Fly 1 m out from `home` facing each heading (yaw0 first, then 45 deg steps) and read; the payload or None."""
        for k in range(rounds * steps):
            yaw = yaw0 + 2 * math.pi * k / steps
            self.fly(home[0] + math.cos(yaw), home[1] + math.sin(yaw), alt, deadline, yaw=yaw)
            r = self.trigger('/mission/read_qr', timeout=30.0, required=False)
            if r and r.success and r.message:
                return r.message
            self.get_logger().info(f'start QR not seen facing {math.degrees(yaw) % 360:.0f} deg; next heading')
        return None

    def fly_route(self, xy, deadline):
        """A* on /delivery/map (red zones + margin, zone boundary) to xy, flown straight at zone_speed at the current
        height; re-routed when a red zone found on the way blocks it. A goal in a no-fly margin ends at the nearest
        free cell within 3 m (the caller's direct hop does the rest, as for a target beside a red zone)."""
        if self.delivery_map is None:
            raise Abort('no /delivery/map for the route')
        replan = True
        last_check = 0.0
        try:
            while True:
                self.check(deadline)
                if replan:
                    x, y = self.pose()[:2]
                    wps = plan(self.delivery_map, (x, y), xy, 0.6, 0.5, snap_radius=3.0)
                    if wps is None:
                        raise Abort(f'no A* route to ({xy[0]:.1f}, {xy[1]:.1f})')
                    self.route.start(wps, None, vmax=self.g('zone_speed'), tol=0.4)
                    replan = False
                if time.monotonic() - last_check > 1.0:
                    last_check = time.monotonic()
                    if not route_clear(self.delivery_map, self.pose()[:2], self.route.wps, 0.6):
                        self.get_logger().info('route crosses a newly mapped no-fly cell: re-routing')
                        replan = True
                        continue
                if self.route.step() == 'arrived':
                    return
                time.sleep(0.05)
        finally:
            self.route.stop()

    def navigate_delivery(self, xy, deadline):
        if not self.g('delivery_nav2'):
            return self.fly_route(xy, deadline)
        if not self.delivery_nav.wait_for_server(timeout_sec=5.0):
            raise Abort('/delivery/navigate_to_pose not available')
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.pose.orientation.w = 1.0
        fut = self.delivery_nav.send_goal_async(goal)
        self.wait(fut.done, deadline)
        handle = fut.result()
        if not handle.accepted:
            raise Abort('delivery Nav2 rejected the target goal')
        res = handle.get_result_async()
        try:
            self.wait(res.done, deadline)
        except Abort:
            handle.cancel_goal_async()
            raise
        if res.result().status != 4:
            raise Abort(f'delivery Nav2 could not reach the target (status {res.result().status})')

    def takeoff_px4(self, alt, deadline):
        # Not required: while MAVROS is still pulling PX4's parameter list over a busy link a set took > 10 s
        # and failed the mission on the ground (2026-10-20 sim). Without it PX4 climbs to its own
        # MIS_TAKEOFF_ALT (default 2.5 m) and the offboard climb below makes up the rest
        try:
            self.call(self.px4_params, SetParameters.Request(
                parameters=[Parameter(name='MIS_TAKEOFF_ALT', value=pval(alt))]), timeout=30.0)
        except Abort as e:
            self.get_logger().warn(f'MIS_TAKEOFF_ALT not set ({e}): PX4 takes off to its own value')
        while not self.mav.armed:
            self.warn_stale_mav()
            self.mav_call(self.set_mode, SetMode.Request(custom_mode='AUTO.TAKEOFF'))
            self.mav_call(self.arming, CommandBool.Request(value=True))
            self.check(deadline)
            time.sleep(2.0)
        # AUTO.LOITER is also the mode on the ground before take-off: wait for height too
        self.wait(lambda: self.pose()[2] > min(alt, 2.5) - 0.4 and self.mav.mode == 'AUTO.LOITER', deadline)

    def takeoff_ardupilot(self, alt, deadline):
        """GUIDED, arm, MAV_CMD_NAV_TAKEOFF (param7: metres above home) at once - ArduCopter disarms again after
        DISARM_DELAY on the ground. Arming is refused until the EKF has its origin (GPS lock, ~30 s after boot):
        retried every 2 s. A take-off sent while already climbing is refused, harmlessly."""
        refused = 0
        while self.pose()[2] < 0.3 or not self.mav.armed:
            self.check(deadline)                     # first: an operator LAND / RTL must not be undone by GUIDED
            self.warn_stale_mav()
            if refused >= 3 and self.mav.armed:
                # armed on the ground but every take-off refused (seen 2026-10-27 after ground LAND / SMART_RTL
                # requests): a fresh disarm + arm in GUIDED takes it at once
                self.get_logger().warn('take-off refused 3 times: disarm and arm again')
                self.mav_call(self.arming, CommandBool.Request(value=False))
                refused = 0
                time.sleep(1.0)
                continue
            if self.mav.mode != 'GUIDED':
                self.mav_call(self.set_mode, SetMode.Request(custom_mode='GUIDED'))
            r = self.mav_call(self.arming, CommandBool.Request(value=True)) if not self.mav.armed else None
            if self.mav.armed or (r is not None and r.success):
                r = self.mav_call(self.px4_command, CommandLong.Request(command=22, param7=float(alt)))
                if r is not None and not r.success:
                    refused += 1
                    self.get_logger().warn(f'take-off refused ({self.ACK_TEXT.get(r.result, r.result)})',
                                           throttle_duration_sec=5.0)
            elif r is not None:
                self.get_logger().warn(f'arming refused ({self.ACK_TEXT.get(r.result, r.result)}): '
                                       f'pre-arm checks (EKF / GPS) not passed yet', throttle_duration_sec=5.0)
            self.check(deadline)
            time.sleep(2.0)
        # ArduCopter's take-off ends at alt and holds there in GUIDED: let it finish
        self.wait(lambda: self.pose()[2] > alt - 0.5, deadline)

    def land(self):
        self.landing = True                          # LAND is this mission's own from here (check_takeover)
        self.set_state('LAND')
        deadline = self.limit('land')
        land = self.modes['land']
        self.call(self.offboard, SetBool.Request(data=False))
        self.mav_call(self.set_mode, SetMode.Request(custom_mode=land))
        sent = time.monotonic()
        while self.mav.armed:
            if self.operator:                        # the operator's RTL / LAND / KILL: no LAND re-requests
                raise Abort(self.abort_reason or 'operator took over')
            if self.clock_s() > deadline:
                raise Abort('landing took too long')
            if self.mav.mode != land and time.monotonic() - sent > 10.0:   # the request may have been lost
                self.mav_call(self.set_mode, SetMode.Request(custom_mode=land))
                sent = time.monotonic()
            time.sleep(0.5)

    def failsafe(self, why, home):
        if self.crashed or self.operator:            # nothing to fly: motors killed / the operator flies
            if self.crashed:
                self.winch_stop()                    # a pending wind-in must not run on a crashed drone
            self.set_state('CRASH' if self.crashed else 'OPERATOR', reason=why)
            for name in ('/delivery/explore/stop', '/nav/entry/cancel', '/nav/corridor/cancel'):
                try:
                    self.trigger(name, timeout=2.0, required=False)
                except Abort:
                    pass
            self.set_state('FAILED', reason=why)
            return
        in_corridor, self.in_corridor = self.in_corridor, False
        self.set_state('FAILSAFE', reason=why)
        self.abort_reason = None                     # let the failsafe legs fly
        for name in ('/delivery/explore/stop', '/nav/entry/cancel', '/nav/corridor/cancel'):
            try:
                self.trigger(name, timeout=2.0, required=False)
            except Abort:
                pass
        try:
            if self.odom is not None and self.airborne():
                if self.mav.mode != self.modes['offboard']:
                    self.call(self.offboard, SetBool.Request(data=True))
                self.t_start = self.clock_s()        # the failsafe has its own clock
                deadline = self.clock_s() + self.g('limit_home') + 60
                x, y, z, _ = self.pose()
                self.flight.hold(self.g('failsafe_hold'), z)
                if in_corridor and self.window_seen:
                    raise Abort('in the corridor past a window: no retrace')
                if in_corridor and self.trail:
                    if not self.retrace():
                        raise Abort('retrace out of the corridor stalled')
                    x, y, z, _ = self.pose()
                if self.g('failsafe_home') and home:
                    fa = max(z, self.g('failsafe_altitude'))
                    self.fly(x, y, fa, deadline)
                    self.fly(home[0], home[1], fa, deadline)
                self.land()
        except Exception as e:
            self.get_logger().error(f'failsafe could not finish ({e}): landing in place')
            try:
                self.landing = True
                self.mav_call(self.set_mode, SetMode.Request(custom_mode=self.modes['land']))
            except Exception:
                pass
        self.set_state('FAILED', reason=why)


def main():
    rclpy.init()
    node = MissionSupervisor()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:  # RCLError "context is not valid" when shutdown races the wait set (vio_bringup spin.py)
        if rclpy.ok():
            raise
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
