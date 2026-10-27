"""Keyboard flying for the PX4 SITL drone, through MAVROS offboard velocity setpoints.

SIMULATION ONLY. Needs MAVROS connected to SITL (e.g. sim_view.launch.py) and a real
terminal (it reads single key presses):
    ros2 run vio_bringup sim_teleop

  t  take off: arm, OFFBOARD, climb to the target altitude
  w / s   forward / back speed  +/- 0.2 m/s      (body frame, sticky)
  a / d   left / right speed    +/- 0.2 m/s
  q / e   yaw left / right rate +/- 15 deg/s
  r / f   target altitude       +/- 0.2 m
  space   stop: zero speed and yaw rate, hold altitude
  l       land (AUTO.LAND)
  x / Ctrl-C  quit (lands first if flying)

Speeds are relative to the drone's heading and capped at 1.5 m/s horizontal.
Altitude is held by a P-controller on the vertical speed.
"""
import math
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

MAX_XY = 1.5         # m/s
MAX_YAW = math.radians(60)
MAX_Z = 0.8          # m/s
ALT_GAIN = 1.0       # 1/s
STEP_XY = 0.2
STEP_YAW = math.radians(15)
STEP_ALT = 0.2
TAKEOFF_ALT = 1.0    # m; the tunnel walls are 2 m


def clamp(v, lim):
    return max(-lim, min(lim, v))


class SimTeleop(Node):
    def __init__(self):
        super().__init__('sim_teleop')
        self.fwd = self.left = self.yaw_rate = 0.0
        self.target_alt = TAKEOFF_ALT
        self.flying = False
        self.z = self.yaw = None
        self.state = State()
        self.pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(State, '/mavros/state', lambda m: setattr(self, 'state', m), 10)
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.create_timer(0.05, self.send)

    def on_pose(self, msg):
        q = msg.pose.orientation
        self.z = msg.pose.position.z
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def send(self):
        """20 Hz setpoints: PX4 leaves OFFBOARD if they stop for > 0.5 s."""
        t = Twist()
        if self.z is not None and self.yaw is not None:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            t.linear.x = c * self.fwd - s * self.left
            t.linear.y = s * self.fwd + c * self.left
            t.linear.z = clamp(ALT_GAIN * (self.target_alt - self.z), MAX_Z) if self.flying else 0.0
            t.angular.z = self.yaw_rate
        self.pub.publish(t)

    def call(self, cli, req):
        if not cli.wait_for_service(timeout_sec=2.0):
            return False
        fut = cli.call_async(req)
        end = time.monotonic() + 3.0
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        return fut.done() and fut.result() is not None

    def takeoff(self):
        if not self.state.connected:
            return 'FCU not connected - is MAVROS running against the sim?'
        self.fwd = self.left = self.yaw_rate = 0.0
        self.target_alt = TAKEOFF_ALT
        self.flying = True
        time.sleep(1.0)  # PX4 needs a stream of setpoints before it accepts OFFBOARD
        self.call(self.mode_cli, SetMode.Request(custom_mode='OFFBOARD'))
        self.call(self.arm_cli, CommandBool.Request(value=True))
        return 'takeoff: arm + OFFBOARD requested'

    def land(self):
        self.fwd = self.left = self.yaw_rate = 0.0
        self.flying = False
        self.call(self.mode_cli, SetMode.Request(custom_mode='AUTO.LAND'))
        return 'landing (AUTO.LAND)'

    def key(self, k):
        if k == 't':
            return self.takeoff()
        if k == 'l':
            return self.land()
        if k == ' ':
            self.fwd = self.left = self.yaw_rate = 0.0
            return 'stop'
        steps = {'w': ('fwd', STEP_XY, MAX_XY), 's': ('fwd', -STEP_XY, MAX_XY),
                 'a': ('left', STEP_XY, MAX_XY), 'd': ('left', -STEP_XY, MAX_XY),
                 'q': ('yaw_rate', STEP_YAW, MAX_YAW), 'e': ('yaw_rate', -STEP_YAW, MAX_YAW)}
        if k in steps:
            name, step, lim = steps[k]
            setattr(self, name, round(clamp(getattr(self, name) + step, lim), 3))
        elif k == 'r':
            self.target_alt = round(min(self.target_alt + STEP_ALT, 1.8), 2)
        elif k == 'f':
            self.target_alt = round(max(self.target_alt - STEP_ALT, 0.3), 2)
        return ''

    def status(self):
        z = f'{self.z:.2f}' if self.z is not None else '?'
        return (f'{self.state.mode or "?":>12s} {"ARMED" if self.state.armed else "disarmed":8s} '
                f'alt {z}/{self.target_alt:.1f} m  fwd {self.fwd:+.1f}  left {self.left:+.1f} m/s  '
                f'yaw {math.degrees(self.yaw_rate):+.0f} deg/s')


def main():
    if not sys.stdin.isatty():
        sys.exit('sim_teleop needs an interactive terminal')
    rclpy.init()
    node = SimTeleop()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(__doc__.split('\n\n', 2)[2])
    old = termios.tcgetattr(sys.stdin)
    msg = ''
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            print(f'\r{node.status()}  {msg:40s}', end='', flush=True)
            if select.select([sys.stdin], [], [], 0.2)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'x':
                    break
                msg = node.key(k) or msg
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        if node.state.armed:
            print('\n' + node.land())
            time.sleep(1.0)
        print()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
