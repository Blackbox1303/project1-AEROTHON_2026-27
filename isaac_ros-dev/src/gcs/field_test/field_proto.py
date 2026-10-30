"""Field test (no materials, props OFF): the codes shared by the three halves - field_vision.py (Jetson, dev container:
cameras + detectors + guidance, HTTP 8097), field_link.py (Jetson host: the Pixhawk - altitude / GPS / heading in,
motor test out) and field_ground.py (the laptop, 8096: by the Ethernet cable or over the telemetry radio alone).
Over the radio a command is COMMAND_LONG MAV_CMD_USER_1 (31010) to 1/194 (onboard/mav_agent.py), param1 = 11 (field),
param2 = the code below, param3 = its argument, param4 = a press number (a resent command runs once).
Standard library only: the container, the Jetson host and the laptop all import it.
"""
import math
import os

PORT_VISION = 8097               # field_vision.py on the Jetson (container, host network)
PORT_GROUND = 8096               # field_ground.py on the laptop
ACTION_FIELD = 11                # mav_agent action (param1)
T_FIELD = 32820                  # TUNNEL payload_type: the field test's guidance, ~1 Hz while it runs (<= 128 B JSON)
CMD = 31010                      # MAV_CMD_USER_1
AGENT_COMP = 194                 # mav_agent

# code -> (name, what the argument is)
CODES = {
    1: ('start', ''),                      # mav_agent: field_test.sh start (cameras + this app on the Jetson)
    2: ('stop', ''),                       # mav_agent: field_test.sh stop
    3: ('detect_qr', ''),                  # boards in the Arducam view (no decode)
    4: ('scan_qr', ''),                    # decode the boards in view (+ the deep read), vs the start QR
    5: ('start_qr', ''),                   # read the START QR now: the payload + its image become the target
    6: ('red', ''),                        # red zone under the drone (Arducam, the mission's HSV rule)
    7: ('green', ''),                      # green banner ahead (D435 colour)
    8: ('blue', ''),                       # blue banner ahead (D435 colour)
    9: ('obstacle', ''),                   # obstacles ahead (D435 depth)
    10: ('walk_qr', ''),                   # walk-through of the QR area search (guidance + motors)
    11: ('walk_obstacle', ''),             # walk-through of obstacle avoidance on the D435 depth
    12: ('walk_stop', ''),
    13: ('motors', '1 on / 0 off'),        # the motor cues (props OFF): field_link spins them
    14: ('virtual_alt', 'm'),              # the height the walk-through pretends the drone is at
    15: ('corner', 'slot 1-4 (0 clears)'), # QR area corner here (the Pixhawk's GPS)
    16: ('area_geofence', ''),             # QR area = the geofence's QR slots (runs/gcs/geofence.json)
    17: ('next', ''),                      # walk_qr: skip the board in hand
    18: ('carry_height', 'm'),             # how high the drone is carried (the camera's ground projection)
    19: ('lane_spacing', 'm'),
    20: ('status', ''),                    # a status packet now
    21: ('spin_test', 's (0 = 6 s)'),      # props OFF: all four motors in turn FR RR RL FL (cw chase), mid level
    22: ('set_mode', 'ArduCopter mode number (0 STABILIZE, 3 AUTO, 6 RTL)'),   # disarmed only: the switch test
    23: ('winch', 's held (0 = 45, < 0 stops)'),   # props OFF, disarmed: the drop mechanism on RC5, then back
}
BY_NAME = {v[0]: k for k, v in CODES.items()}

# the motor cue (field_vision -> field_link), also in the radio status as 'mc'
#   pattern: off (stand still) | dir (spin the side to walk to, bearing deg clockwise from the nose) | cw / ccw (a
#            chase round the frame: yaw right / left) | all (criss-cross FR, RL, FL, RR: only the height to change)
#   level:   low = slow (height down - lowering the payload too) | mid (keep) | high = fast (height up)
MOTOR_PATTERNS = ('off', 'dir', 'cw', 'ccw', 'all')
LEVELS = ('low', 'mid', 'high')
PATTERN_CODE = {'off': 'o', 'dir': 'd', 'cw': 'r', 'ccw': 'l', 'all': 'a'}   # the radio's 'mc': this + the level's
CODE_PATTERN = {v: k for k, v in PATTERN_CODE.items()}                       # first letter (l / m / h)


def find_radio(path):
    """The ground SiK radio: `path` if it is there, else the one radio plugged in - by its USB id (SiK / FTDI / CP210x),
    else the only /dev/ttyUSB* or /dev/ttyACM* (a SiK shows as either). None: nothing found."""
    import glob
    import re
    if os.path.exists(path):
        return path
    ids = [p for p in sorted(glob.glob('/dev/serial/by-id/*'))
           if re.search(r'sik|3dr|rfd|radio|telem|ftdi|ft231|silicon_labs|cp210', p, re.I)
           and not re.search(r'ardupilot|pixhawk|px4|cube', p, re.I)]
    if ids:
        return os.path.realpath(ids[0])
    ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    return ports[0] if len(ports) == 1 else None


# ---- the autopilot panel of the test pages (user, 2026-10-30): the same in radio (field_ground) and cable (field_link)
FC_STREAMS = ((33, 2), (24, 1), (30, 4), (74, 1), (65, 1), (36, 1), (11030, 1))   # msg id, Hz (light on the radio)
SWITCH7 = ((0, 1230, 'STABILIZE'), (1491, 1620, 'AUTO'), (1750, 3000, 'RTL'))    # the Boxer's channel 7 positions
TEST_MODES = {0: 'STABILIZE', 3: 'AUTO', 6: 'RTL'}
CMD_AUX = 218                    # MAV_CMD_DO_AUX_FUNCTION (ardupilotmega)
DROP_HOLD = 45.0                 # s the drop mechanism needs after RC5 (user, 2026-10-30)


def switch7(pwm):
    return next((n for lo, hi, n in SWITCH7 if pwm is not None and lo <= pwm <= hi), '-')


def fc_update(s, m, mode_string):
    """One autopilot message (1/1) into the panel dict s; mode_string: mavutil.mode_string_v10."""
    t = m.get_type()
    if t == 'HEARTBEAT':
        s['armed'], s['mode'] = bool(m.base_mode & 128), mode_string(m)
    elif t == 'GPS_RAW_INT':
        s['fix'], s['sats'] = m.fix_type, m.satellites_visible
    elif t == 'GLOBAL_POSITION_INT':
        if m.lat or m.lon:
            s['lat'], s['lon'] = m.lat * 1e-7, m.lon * 1e-7
        s['alt_rel'] = round(m.relative_alt / 1000.0, 2)
    elif t == 'ATTITUDE':
        s['roll'], s['pitch'] = round(math.degrees(m.roll), 1), round(math.degrees(m.pitch), 1)
        s['hdg'] = round(math.degrees(m.yaw) % 360.0, 1)
    elif t == 'VFR_HUD':
        s['thr'] = m.throttle
    elif t == 'RC_CHANNELS':
        s['ch5'], s['ch7'] = m.chan5_raw or None, m.chan7_raw or None
        s['sw7'] = switch7(s['ch7'])
    elif t == 'SERVO_OUTPUT_RAW':
        s['mot'] = [m.servo1_raw, m.servo2_raw, m.servo3_raw, m.servo4_raw]
    elif t == 'ESC_TELEMETRY_1_TO_4':
        s['rpm'] = list(m.rpm)
    elif t == 'PARAM_VALUE':
        s.setdefault('params', {})[m.param_id.rstrip(chr(0))] = m.param_value
    elif t == 'STATUSTEXT':
        s['fc_text'] = m.text.rstrip(chr(0))[:80]


class Rc5Drop:
    """The drop mechanism test (props OFF, disarmed): RC5 'on' for hold s, then back. RC5_OPTION set (an aux function):
    MAV_CMD_DO_AUX_FUNCTION high, then low; else (RC5 passed through to the mechanism) an RC override of channel 5 at
    pwm_on, resent at 4 Hz, then released - sent as the GCS (sysid 255: ArduPilot takes overrides from SYSID_MYGCS only).
    send(cmd, *params) / override(ch5) are given by the caller (its MAVLink link); fc is the panel dict (armed,
    params)."""

    def __init__(self, fc, send, override, request_param, pwm_on=2000):
        self.fc, self.send, self.override, self.request_param, self.pwm_on = fc, send, override, request_param, pwm_on
        self.state = {'phase': 'idle', 'left_s': None, 'method': None, 'msg': ''}
        self.stop = False

    def start(self, hold=DROP_HOLD):
        if hold is not None and float(hold) < 0:     # the page's Stop: RC5 back at once
            self.stop = True
            return True, 'winch test: stopping - RC5 back to the transmitter'
        if self.state['phase'] not in ('idle', 'done', 'failed', 'stopped'):
            return False, f'already running: {self.state["phase"]}'
        if self.fc.get('armed') is not False:
            return False, f'the drone must be DISARMED for the winch test (armed {self.fc.get("armed")})'
        import threading
        self.stop = False
        threading.Thread(target=self.run, args=(float(hold or DROP_HOLD),), daemon=True).start()
        return True, f'winch test: RC5 on, {hold or DROP_HOLD:g} s, then off - PROPS OFF'

    def run(self, hold):
        import time
        st = self.state
        st.update(phase='reading RC5_OPTION', left_s=None, msg='')
        self.fc.setdefault('params', {}).pop('RC5_OPTION', None)
        for _ in range(3):
            self.request_param('RC5_OPTION')
            t0 = time.time()
            while time.time() - t0 < 1.5 and 'RC5_OPTION' not in self.fc['params']:
                time.sleep(0.05)
            if 'RC5_OPTION' in self.fc['params']:
                break
        opt = int(self.fc['params'].get('RC5_OPTION', 0) or 0)
        st['method'] = f'aux function {opt}' if opt else f'RC5 override {self.pwm_on}'
        end = time.time() + hold
        st['phase'] = 'RC5 ON - the drop mechanism runs'
        try:
            while time.time() < end and not self.stop:
                if self.fc.get('armed'):
                    st['msg'] = 'ARMED: stopped'
                    break
                if opt:
                    if st['left_s'] is None:
                        self.send(CMD_AUX, opt, 2)
                else:
                    self.override(self.pwm_on)
                st['left_s'] = round(end - time.time())
                time.sleep(0.25)
        finally:
            if opt:
                self.send(CMD_AUX, opt, 0)
            else:
                self.override(0)                      # 0 = back to the transmitter's channel 5
            st.update(phase='stopped' if self.stop else 'done', left_s=0)
