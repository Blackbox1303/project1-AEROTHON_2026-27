"""Guided hands-on bench tests that need someone to move the rig.

    ros2 run vio_bringup guided_test axes       # M1-5, needs mavros_imu.launch.py running
    ros2 run vio_bringup guided_test occlusion  # M4-5, needs cuvslam.launch.py running
    ros2 run vio_bringup guided_test rotation   # M4-6, needs cuvslam.launch.py running
    ros2 run vio_bringup guided_test line       # M4-2, walk a tape-measured straight line
    ros2 run vio_bringup guided_test loop       # M4-3, walk a loop back to a marked start

Each step prints an instruction, waits for Enter, records a few seconds while
you do the motion, then scores it against TESTING.md.
"""
import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self):
        super().__init__('guided_test')
        self.lock = threading.Lock()
        self.imu, self.odom, self.status = [], [], []
        self.create_subscription(Imu, '/vio/imu', lambda m: self.add(self.imu, m),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry',
                                 lambda m: self.add(self.odom, m), qos_profile_sensor_data)
        self.create_subscription(VisualSlamStatus, '/visual_slam/status',
                                 lambda m: self.add(self.status, m), qos_profile_sensor_data)

    def add(self, buf, msg):
        with self.lock:
            buf.append((time.monotonic(), msg))

    def window(self, seconds):
        """Clear buffers, wait `seconds`, return what arrived meanwhile."""
        with self.lock:
            self.imu, self.odom, self.status = [], [], []
        time.sleep(seconds)
        with self.lock:
            return list(self.imu), list(self.odom), list(self.status)


def record_until_enter(rec, prompt):
    """Clear buffers, record until the operator presses Enter."""
    with rec.lock:
        rec.imu, rec.odom, rec.status = [], [], []
    input(prompt)
    with rec.lock:
        return list(rec.imu), list(rec.odom), list(rec.status)


def path_xy(odom):
    return [(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z) for _, m in odom]


def path_length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def ask_float(prompt, default):
    text = input(f'{prompt} [{default}]: ').strip()
    return float(text) if text else default


def step(prompt, rec, seconds):
    input(f'\n{prompt}\n  Press Enter, then do it within {seconds:.0f} s... ')
    print('  recording', end='', flush=True)
    imu, odom, status = rec.window(seconds)
    print(' done')
    return imu, odom, status


def mean_acc(imu, last_s=0.5):
    if not imu:
        return (float('nan'),) * 3
    t_end = imu[-1][0]
    tail = [m for t, m in imu if t >= t_end - last_s]
    return tuple(sum(getattr(m.linear_acceleration, a) for m in tail) / len(tail) for a in 'xyz')


def gyro_integral(imu):
    total = [0.0, 0.0, 0.0]
    for (t0, m0), (t1, _) in zip(imu, imu[1:]):
        dt = t1 - t0
        total[0] += m0.angular_velocity.x * dt
        total[1] += m0.angular_velocity.y * dt
        total[2] += m0.angular_velocity.z * dt
    return total


def result(name, ok, detail):
    print(f'  {name}: {"PASS" if ok else "FAIL"}  ({detail})')
    return ok


def test_axes(rec):
    print('M1-5 axis signs (FLU). Keep the board flat on the desk to start.')
    base, _, _ = step('Leave the board LEVEL and still.', rec, 2)
    if not base:
        sys.exit('no /vio/imu data - is mavros_imu.launch.py running?')
    a0 = mean_acc(base, 1.5)
    ok = []
    imu, _, _ = step('Tilt the NOSE UP ~30 deg and HOLD it there.', rec, 3)
    g, a = gyro_integral(imu), mean_acc(imu)
    ok.append(result('nose up -> gyro y negative', g[1] < -0.3 and abs(g[1]) > abs(g[0]),
                     f'integrated gyro y {math.degrees(g[1]):+.0f} deg'))
    ok.append(result('nose up -> acc.x increases', a[0] - a0[0] > 2.5,
                     f'acc.x {a0[0]:+.2f} -> {a[0]:+.2f} m/s^2'))
    step('Put it back LEVEL.', rec, 2)
    imu, _, _ = step('ROLL RIGHT (right side down) ~30 deg and HOLD.', rec, 3)
    g, a = gyro_integral(imu), mean_acc(imu)
    ok.append(result('roll right -> gyro x positive', g[0] > 0.3 and abs(g[0]) > abs(g[1]),
                     f'integrated gyro x {math.degrees(g[0]):+.0f} deg'))
    ok.append(result('roll right -> acc.y increases', a[1] - a0[1] > 2.5,
                     f'acc.y {a0[1]:+.2f} -> {a[1]:+.2f} m/s^2'))
    step('Put it back LEVEL.', rec, 2)
    imu, _, _ = step('YAW LEFT (counter-clockwise seen from above) ~90 deg, flat on the desk.', rec, 3)
    g = gyro_integral(imu)
    ok.append(result('yaw left -> gyro z positive', g[2] > 1.0,
                     f'integrated gyro z {math.degrees(g[2]):+.0f} deg'))
    return all(ok)


def pose_change(odom):
    p0, p1 = odom[0][1].pose.pose, odom[-1][1].pose.pose
    d = math.dist((p0.position.x, p0.position.y, p0.position.z),
                  (p1.position.x, p1.position.y, p1.position.z))
    dy = math.degrees(yaw(p1.orientation) - yaw(p0.orientation))
    return d, (dy + 180) % 360 - 180


def tracking_stats(status):
    bad = [t for t, s in status if s.vo_state != 1]
    lost_s = (bad[-1] - bad[0]) if bad else 0.0
    return len(bad), len(status), lost_s, (status[-1][1].vo_state == 1) if status else False


def test_occlusion(rec):
    print('M4-5 lens cover. Rig stays still on the desk the whole time.')
    _, odom, _ = step('Leave the rig still (baseline).', rec, 2)
    if not odom:
        sys.exit('no odometry - is cuvslam.launch.py running?')
    _, odom, status = step('Fully cover BOTH IR lenses for ~2 s, then uncover. Do not move the rig.',
                           rec, 8)
    bad, n, lost_s, recovered = tracking_stats(status)
    d, dy = pose_change(odom)
    ok = [result('tracking resumes', recovered, f'{bad}/{n} status msgs not tracking, gap ~{lost_s:.1f} s'),
          result('pose jump after recovery < 5 cm, < 2 deg', d < 0.05 and abs(dy) < 2,
                 f'{d * 100:.1f} cm, yaw {dy:+.1f} deg')]
    if bad == 0:
        print('  note: tracking never dropped - lenses may not have been fully covered')
    return all(ok)


def test_rotation(rec):
    print('M4-6 fast rotation. Hold the rig in your hand, rotate about its own centre.')
    _, odom, _ = step('Hold the rig still, pointing at a textured scene.', rec, 2)
    if not odom:
        sys.exit('no odometry - is cuvslam.launch.py running?')
    imu, odom, status = step('Within ~1 s turn it ~180 deg LEFT, then back to the start pointing, '
                             'and hold still.', rec, 6)
    bad, n, _, _ = tracking_stats(status)
    peak = max((abs(m.angular_velocity.z) for _, m in imu), default=float('nan'))
    d, dy = pose_change(odom)
    ok = [result('rotation fast enough (>= 180 deg/s peak)', peak >= math.pi,
                 f'peak {math.degrees(peak):.0f} deg/s (from /vio/imu; nan if Pixhawk not attached)'),
          result('no tracking loss', bad == 0, f'{bad}/{n} status msgs not tracking'),
          result('back at start: < 5 cm, < 3 deg', d < 0.05 and abs(dy) < 3,
                 f'{d * 100:.1f} cm, yaw {dy:+.1f} deg')]
    return all(ok)


def test_line(rec):
    print('M4-2 straight line. Lay a tape measure on the ground and mark start and end.')
    true_len = ask_float('Tape length between the marks in metres', 10.0)
    input('Hold the rig at the START mark, pointing along the tape, still. Press Enter to start... ')
    _, odom, status = record_until_enter(
        rec, '  recording - walk steadily to the END mark, stop, hold still, then press Enter... ')
    if len(odom) < 10:
        sys.exit('no odometry - is cuvslam.launch.py running?')
    pts = path_xy(odom)
    (x0, y0, _), (x1, y1, _) = pts[0], pts[-1]
    est = math.dist(pts[0], pts[-1])
    ux, uy = (x1 - x0) / max(est, 1e-9), (y1 - y0) / max(est, 1e-9)
    lateral = max(abs((x - x0) * uy - (y - y0) * ux) for x, y, _ in pts)
    bad, n, _, _ = tracking_stats(status)
    err = (est - true_len) / true_len * 100
    ok = [result('length error < 2 %', abs(err) < 2, f'VO {est:.3f} m vs tape {true_len:.3f} m ({err:+.2f} %)'),
          result('max lateral deviation < 10 cm', lateral < 0.10, f'{lateral * 100:.1f} cm'),
          result('tracking', bad == 0, f'{bad}/{n} status msgs not tracking')]
    print(f'  path length walked {path_length(pts):.2f} m, height change {pts[-1][2] - pts[0][2]:+.3f} m')
    return all(ok)


def test_loop(rec):
    print('M4-3 closed loop. Mark a start spot on the ground; walk ~30 m and come back to it.')
    input('Hold the rig exactly at the START mark, still. Press Enter to start... ')
    _, odom, status = record_until_enter(
        rec, '  recording - walk the loop, return to the SAME mark and pose, hold still, press Enter... ')
    if len(odom) < 10:
        sys.exit('no odometry - is cuvslam.launch.py running?')
    pts = path_xy(odom)
    length = path_length(pts)
    end_err = math.dist(pts[0], pts[-1])
    bad, n, _, _ = tracking_stats(status)
    ok = [result('endpoint error < 1 % of path', end_err < 0.01 * length,
                 f'{end_err * 100:.1f} cm over {length:.1f} m ({end_err / max(length, 1e-9) * 100:.2f} %)'),
          result('tracking', bad == 0, f'{bad}/{n} status msgs not tracking')]
    return all(ok)


TESTS = {'axes': test_axes, 'occlusion': test_occlusion, 'rotation': test_rotation,
         'line': test_line, 'loop': test_loop}


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    if len(args) != 1 or args[0] not in TESTS:
        sys.exit(__doc__)
    rclpy.init()
    rec = Recorder()
    threading.Thread(target=rclpy.spin, args=(rec,), daemon=True).start()
    time.sleep(1.0)
    try:
        ok = TESTS[args[0]](rec)
        print(f'\n{args[0]}: {"PASS" if ok else "FAIL"}')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
