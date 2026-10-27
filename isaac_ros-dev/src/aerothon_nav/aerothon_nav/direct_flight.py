"""Direct point-to-point flight through offboard_bridge (/cmd_vel body frame + /nav/altitude), for the
short legs Nav2 is not needed for (staging, search turns, strafing). The caller's thread blocks;
the node keeps spinning in its executor.
"""
import math
import time

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DirectFlight:
    def __init__(self, node, pose_fn, cancelled_fn=lambda: False):
        """pose_fn() -> (x, y, z, yaw) in odom; cancelled_fn() -> True aborts a leg."""
        self.cmd = node.create_publisher(Twist, '/cmd_vel', 10)
        self.alt = node.create_publisher(Float32, '/nav/altitude', 10)
        self.pose = pose_fn
        self.cancelled = cancelled_fn

    def stop(self):
        self.cmd.publish(Twist())

    def fly(self, x, y, alt, yaw, vmax=1.0, tol=0.25, yaw_tol=0.08, timeout=60.0, each=None, until=None):
        """Go to (x, y) at alt facing yaw; `yaw` may be a function (x, y) -> yaw, re-evaluated every
        step (e.g. keep facing a target while strafing). each(): called every step. True when there,
        or as soon as until() is True. alt None: leave the height to whoever else sets /nav/altitude
        (not published, not checked)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.cancelled():
            px, py, pz, pyaw = self.pose()
            target_yaw = yaw(px, py) if callable(yaw) else yaw
            ex, ey = x - px, y - py
            d = math.hypot(ex, ey)
            eyaw = wrap(target_yaw - pyaw)
            if alt is not None:
                self.alt.publish(Float32(data=float(alt)))
            if each:
                each()
            if until is not None and until():
                self.stop()
                return True
            if d < tol and (alt is None or abs(pz - alt) < 0.15) and abs(eyaw) < yaw_tol:
                self.stop()
                return True
            s = min(vmax, 0.8 * d)
            vx, vy = (ex / d * s, ey / d * s) if d > 1e-3 else (0.0, 0.0)
            m = Twist()
            m.linear.x = vx * math.cos(pyaw) + vy * math.sin(pyaw)       # odom -> body
            m.linear.y = -vx * math.sin(pyaw) + vy * math.cos(pyaw)
            m.angular.z = max(-0.6, min(0.6, 1.5 * eyaw))
            self.cmd.publish(m)
            time.sleep(0.05)
        self.stop()
        return False

    def step(self, x, y, alt, yaw, vmax=1.0):
        """One command toward (x, y) facing yaw, not waiting: for callers running their own loop (corridor_nav
        following the corridor line). alt None: height left to whoever else sets /nav/altitude."""
        px, py, pz, pyaw = self.pose()
        ex, ey = x - px, y - py
        d = math.hypot(ex, ey)
        if alt is not None:
            self.alt.publish(Float32(data=float(alt)))
        sp = min(vmax, 0.8 * d)
        vx, vy = (ex / d * sp, ey / d * sp) if d > 1e-3 else (0.0, 0.0)
        m = Twist()
        m.linear.x = vx * math.cos(pyaw) + vy * math.sin(pyaw)       # world -> body
        m.linear.y = -vx * math.sin(pyaw) + vy * math.cos(pyaw)
        m.angular.z = max(-0.6, min(0.6, 1.5 * wrap(yaw - pyaw)))
        self.cmd.publish(m)

    def hold(self, seconds, alt, each=None):
        """Hover in place (the bridge holds position once /cmd_vel stops)."""
        self.stop()
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.cancelled():
            self.alt.publish(Float32(data=float(alt)))
            if each:
                each()
            time.sleep(0.05)
