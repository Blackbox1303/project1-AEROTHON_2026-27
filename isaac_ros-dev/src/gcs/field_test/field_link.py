#!/usr/bin/env python3
"""FIELD TEST, the Jetson host's half: the Pixhawk (ArduCopter) for field_vision.py - PROPS OFF.

  in   GPS (GPS_RAW_INT fix / sats, GLOBAL_POSITION_INT lat / lon / height above home), heading (ATTITUDE yaw),
       rangefinder (DISTANCE_SENSOR), armed / mode (HEARTBEAT) -> POST field_vision /api/fc, 5 Hz
  out  the answer is the motor cue {pattern, deg, level, throttle %, on}: MAV_CMD_DO_MOTOR_TEST, ONE motor at a time
       (ArduCopter runs one motor test at a time; a new command takes over the running one), each with a short timeout -
       nothing sent = the motors stop within ~0.5 s. Quad X, test order A..D clockwise from the front right:
         dir  the motors on that side (within 67.5 deg of the bearing) alternate: walk towards the spinning side
         cw   A B C D chase (clockwise from above): yaw RIGHT      ccw  D C B A: yaw LEFT
         all  A C D B criss-cross: only the height is to change
       the throttle % says the height: slow = down (lowering the payload too), medium = keep, fast = up.
       Never while armed (ArduCopter refuses it then anyway); stops when field_vision stops answering.
Link: mavlink-router's field_test endpoint (UDP 14662, added by field_test.sh / onboard/install_onboard.sh) - the
Pixhawk's USB port 1: its commands are addressed to the Pixhawk (1/1), so ArduPilot keeps them off the radio (GPS2).
As MAVLink 1/195.
"""
import argparse
import json
import math
import os
import threading
import time
import urllib.request

os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil  # noqa: E402

COMP = 195                                     # MAV_COMP_ID_ONBOARD_COMPUTER3 (mav_agent is 194)
MOTOR_TEST = 209                               # MAV_CMD_DO_MOTOR_TEST
SEQ_DEG = {1: 45.0, 2: 135.0, 3: 225.0, 4: 315.0}   # test sequence A..D of a quad X: deg clockwise from the nose
NAMES = {1: 'FR', 2: 'RR', 3: 'RL', 4: 'FL'}
PATTERNS = {'cw': [1, 2, 3, 4], 'ccw': [4, 3, 2, 1], 'all': [1, 3, 4, 2]}


def motors_for(deg):
    """The motors on the side of `deg` (clockwise from the nose): 1 or 2 of them."""
    return [s for s, d in SEQ_DEG.items() if abs((deg - d + 180) % 360 - 180) <= 67.5]


class Link:
    def __init__(self, a):
        self.a = a
        self.mav = mavutil.mavlink_connection(a.mavlink, source_system=1, source_component=COMP, dialect='ardupilotmega')
        self.st = {'link': 'waiting for the Pixhawk', 'fix': 0, 'sats': None, 'lat': None, 'lon': None, 'alt_rel': None,
                   'hdg': None, 'range': None, 'armed': None, 'mode': None, 'motor_msg': '', 'fc_text': ''}
        self.t_fc = 0.0
        self.cue = {'pattern': 'off', 'on': False}
        self.t_cue = 0.0
        self.t_test = 0.0                      # the Pixhawk's last ACCEPTED motor test
        self.sent = 0

    def request_streams(self):
        for msg, hz in ((33, 5), (24, 2), (30, 10), (132, 5), (1, 1)):
            self.mav.mav.command_long_send(1, 1, 511, 0, msg, 1e6 / hz, 0, 0, 0, 0, 0)

    def rx_loop(self):
        t_req = 0.0
        while True:
            if time.time() - t_req > 10.0:            # again now and then: a Pixhawk rebooted forgets them
                self.request_streams()
                t_req = time.time()
            m = self.mav.recv_match(blocking=True, timeout=0.5)
            if m is None:
                if time.time() - self.t_fc > 3.0:
                    self.st['link'] = f'no Pixhawk on {self.a.mavlink} (router running? field_test endpoint?)'
                continue
            if m.get_srcSystem() != 1 or m.get_srcComponent() != 1:
                continue
            t, s = m.get_type(), self.st
            self.t_fc, s['link'] = time.time(), 'ok'
            if t == 'HEARTBEAT':
                s['armed'] = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                s['mode'] = mavutil.mode_string_v10(m)
            elif t == 'GPS_RAW_INT':
                s['fix'], s['sats'] = m.fix_type, m.satellites_visible
            elif t == 'GLOBAL_POSITION_INT':
                if m.lat or m.lon:
                    s['lat'], s['lon'] = m.lat * 1e-7, m.lon * 1e-7
                s['alt_rel'] = round(m.relative_alt / 1000.0, 2)
            elif t == 'ATTITUDE':
                s['hdg'] = round(math.degrees(m.yaw) % 360.0, 1)
            elif t == 'DISTANCE_SENSOR':
                s['range'] = round(m.current_distance / 100.0, 2)
            elif t == 'COMMAND_ACK' and m.command == MOTOR_TEST:
                if m.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    s['motor_msg'] = f'motor test refused by the Pixhawk (result {m.result}): {s["fc_text"]}'
                else:
                    s['motor_msg'] = 'motor test accepted'
                    self.t_test = time.time()
            elif t == 'STATUSTEXT':
                s['fc_text'] = m.text.rstrip('\0')[:80]

    def vision_loop(self):
        """5 Hz: the Pixhawk's state to field_vision, its motor cue back."""
        while True:
            body = json.dumps(dict(self.st, fc_age=round(time.time() - self.t_fc, 1) if self.t_fc else None)).encode()
            try:
                r = urllib.request.urlopen(urllib.request.Request(f'{self.a.vision}/api/fc', body,
                                                                  {'Content-Type': 'application/json'}), timeout=1.0)
                self.cue, self.t_cue = json.loads(r.read()), time.time()
            except Exception:
                self.cue = {'pattern': 'off', 'on': False}
            time.sleep(0.2)

    def motor_loop(self):
        """One motor at a time, step_s each, the sequence of the cue's pattern."""
        k = 0
        while True:
            c, step = self.cue, self.a.step_s
            # ArduCopter's HEARTBEAT says armed while its own motor test runs (1 Hz, so it lags): that 'armed' is ours
            # while the tests are being accepted. Really armed = the tests refused -> no fresh ACCEPT -> stop in ~2 s.
            testing = time.time() - self.t_test < step + 1.5
            disarmed = self.st['armed'] is False or (self.st['armed'] and testing)
            ok = (c.get('on') and c.get('pattern') not in (None, 'off') and time.time() - self.t_cue < 1.0
                  and disarmed and time.time() - self.t_fc < 2.0)
            if not ok:
                if c.get('on') and c.get('pattern') not in (None, 'off') and self.st['armed'] and not testing:
                    self.st['motor_msg'] = 'ARMED: no motor cues'
                k = 0
                time.sleep(0.1)
                continue
            if c['pattern'] == 'dir':
                seq = motors_for(float(c.get('deg', 0)))
                seq = seq + [None] if len(seq) == 1 else seq        # one motor: a pulse (on, off)
            else:
                seq = PATTERNS.get(c['pattern'], [])
            if not seq:
                time.sleep(0.1)
                continue
            motor = seq[k % len(seq)]
            k += 1
            if motor is not None:
                thr = max(0.0, min(self.a.max_throttle, float(c.get('throttle', 0))))
                self.mav.mav.command_long_send(1, 1, MOTOR_TEST, 0, motor, 0, thr, step + 0.15, 0, 0, 0)
                self.sent += 1
            time.sleep(step)

    def run(self):
        print(f'field_link: {self.a.mavlink} as 1/{COMP} -> {self.a.vision}', flush=True)
        for f in (self.rx_loop, self.vision_loop, self.motor_loop):
            threading.Thread(target=f, daemon=True).start()
        while True:
            time.sleep(10)
            s = self.st
            print(f'{time.strftime("%H:%M:%S")} link {s["link"]} fix {s["fix"]} sats {s["sats"]} hdg {s["hdg"]} '
                  f'armed {s["armed"]} cue {self.cue.get("pattern")}/{self.cue.get("level")} on {self.cue.get("on")} '
                  f'motor commands {self.sent} {s["motor_msg"]}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mavlink', default='udpin:127.0.0.1:14662', help="mavlink-router's field_test endpoint")
    ap.add_argument('--vision', default='http://127.0.0.1:8097')
    ap.add_argument('--step-s', type=float, default=0.3, help='s each motor of a pattern spins')
    ap.add_argument('--max-throttle', type=float, default=25.0, help='motor test %% never above this')
    Link(ap.parse_args()).run()


if __name__ == '__main__':
    main()
