#!/usr/bin/env python3
"""Compare / apply an ArduPilot parameter file (NAME VALUE lines, # comments) over MAVLink.

  ardupilot_params.py FILE [--link /dev/ttyUSB0:57600]            dry run: current vs new, nothing written
  ardupilot_params.py FILE --write                                 set the differing ones, read each back
Prints the board's serial-port layout first (MAVn_* stream groups count the serial ports whose SERIALn_PROTOCOL is
MAVLink, in order - they depend on the board's setup, so check that MAVn_ lines hit the port you mean). Refuses to
write while armed, and refuses a file that would turn the RC-input port into something else.
Most of these need a reboot of the flight controller to take effect.
"""
import argparse
import sys
import time

from pymavlink import mavutil


def connect(link):
    if link.startswith(('udp', 'tcp')):
        m = mavutil.mavlink_connection(link, source_system=255, source_component=191)
    else:
        dev, _, baud = link.partition(':')
        m = mavutil.mavlink_connection(dev, baud=int(baud or 57600), source_system=255, source_component=191)
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=10,
                      condition='HEARTBEAT.autopilot==%d' % mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA)
    if hb is None:
        sys.exit(f'no ArduPilot heartbeat on {link}')
    m.target_system, m.target_component = hb.get_srcSystem(), hb.get_srcComponent()
    return m, bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def get(m, name, tries=4):
    for _ in range(tries):
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        end = time.time() + 1.5
        while time.time() < end:
            p = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.3)
            if p is not None and p.param_id == name:
                return p.param_value
    return None


def put(m, name, value):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value),
                         mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    end = time.time() + 3
    while time.time() < end:
        p = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.3)
        if p is not None and p.param_id == name:
            return p.param_value
    return get(m, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--link', default='/dev/ttyUSB0:57600', help='radio, USB (/dev/ttyACM0:115200) or udpout:...')
    ap.add_argument('--write', action='store_true', help='apply (default: dry run)')
    a = ap.parse_args()

    want = []
    for line in open(a.file):
        line = line.split('#', 1)[0].split()
        if len(line) >= 2:
            want.append((line[0], float(line[1].rstrip(','))))
    m, armed = connect(a.link)
    print(f'ArduPilot system {m.target_system} on {a.link}{" (ARMED)" if armed else ""}')

    # the serial layout, and which MAVn group each MAVLink port is
    k, rcin = 0, []
    for i in range(8):
        proto, baud = get(m, f'SERIAL{i}_PROTOCOL'), get(m, f'SERIAL{i}_BAUD')
        if proto is None:
            break
        mav = ''
        if int(proto) in (1, 2):
            k += 1
            mav = f'  (MAVLink: MAV{k}_* streams)'
        if int(proto) == 23:
            rcin.append(f'SERIAL{i}_PROTOCOL')
        print(f'  SERIAL{i}: protocol {int(proto)}, baud {int(baud)}{mav}')
    clash = [k2 for k2, v in want if k2 in rcin and int(v) != 23]
    if clash:
        sys.exit(f'refusing: {clash} is the RC receiver port (protocol 23) - changing it would cut the RC link')

    todo = []
    print(f'{"parameter":18s} {"current":>10s} {"new":>10s}')
    for name, value in want:
        cur = get(m, name)
        mark = 'MISSING' if cur is None else ('' if abs(cur - value) < 1e-4 else 'change')
        print(f'{name:18s} {"-" if cur is None else f"{cur:g}":>10s} {value:>10g}  {mark}')
        if mark == 'change':
            todo.append((name, value))
    if not a.write:
        print(f'\ndry run: {len(todo)} to change. Run again with --write (props off) to apply, then reboot the FC.')
        return
    if armed:
        sys.exit('refusing to write parameters while ARMED')
    for name, value in todo:
        got = put(m, name, value)
        print(f'set {name} = {value:g}: read back {got}')
    print('done: reboot the flight controller for SERIAL / EK3_SRC / VISO changes to take effect')


if __name__ == '__main__':
    main()
