"""Field test (no materials, props OFF): the codes shared by the three halves - field_vision.py (Jetson, dev container:
cameras + detectors + guidance, HTTP 8097), field_link.py (Jetson host: the Pixhawk - altitude / GPS / heading in,
motor test out) and field_ground.py (the laptop, 8096: by the Ethernet cable or over the telemetry radio alone).
Over the radio a command is COMMAND_LONG MAV_CMD_USER_1 (31010) to 1/194 (onboard/mav_agent.py), param1 = 11 (field),
param2 = the code below, param3 = its argument, param4 = a press number (a resent command runs once).
Standard library only: the container, the Jetson host and the laptop all import it.
"""
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
