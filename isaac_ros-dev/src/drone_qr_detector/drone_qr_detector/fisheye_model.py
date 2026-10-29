"""Equidistant fisheye camera model, gnomonic patch rectifier, and 3D QR localizer in camera & drone frames.

Fixes over the original aerothon_nav/fisheye.py + vio_bringup/fisheye_camera_info.py:
1. Exact pixel-center scaling `(cx + 0.5) * s - 0.5` across all sensor modes (3280x2464 <-> 1640x1232).
2. Circular-fisheye active-FOV mask (ignores the black border outside the ~185 deg image circle on the IMX219).
3. Fast cached gnomonic (pinhole) rectification both for the central nadir view and for arbitrary off-axis candidate
   rays `rot_to(d)`, with exact inverse mapping from rectified patch pixels back to unit 3D camera rays.
4. Dual 3D localization in the camera frame (`arducam_optical_frame`) and drone body frame (`base_link`):
   - Ray-plane intersection using known plane distance / drone altitude (e.g. 10 m altitude or 3 m wall distance).
   - Metric PnP (`SOLVEPNP_IPPE_SQUARE`) in a ray-aligned local tangent frame using the known QR physical size
     (e.g. 1.0 m x 1.0 m), which works at any off-axis angle and requires zero external odometry.
"""
import math
from pathlib import Path
import cv2
import numpy as np
import yaml


def quat_to_rot(q):
    """Quaternion (x, y, z, w) or geometry_msgs/Quaternion -> 3x3 rotation matrix."""
    if hasattr(q, 'x'):
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    else:
        x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rpy_to_rot(roll, pitch, yaw):
    """Roll, pitch, yaw (rad, intrinsic ZYX / extrinsic XYZ) -> 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def default_base_from_optical(mount_xyz=(0.13, 0.0, -0.03), mount_rpy=(0.0, math.pi / 2, 0.0)):
    """Default static transform T_base_from_optical (4x4) matching arducam_real.launch.py:
    base_link -> arducam_link (mount_xyz, mount_rpy) -> arducam_optical_frame (x+0.018, q=(-0.5, 0.5, -0.5, 0.5)).
    For a nadir camera (pitch = +90 deg):
      optical +z (forward) -> base_link -z (down)
      optical +x (right)   -> base_link -y (right)
      optical +y (down)    -> base_link -x (backward)
    """
    R_base_link = rpy_to_rot(*mount_rpy)
    t_base_link = np.asarray(mount_xyz, dtype=np.float64)
    R_link_opt = quat_to_rot((-0.5, 0.5, -0.5, 0.5))
    t_link_opt = np.array([0.018, 0.0, 0.0], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_base_link @ R_link_opt
    T[:3, 3] = t_base_link + R_base_link @ t_link_opt
    return T


class FisheyeModel:
    """Equidistant fisheye camera model with gnomonic patch rectification."""

    def __init__(self, width=3280, height=2464, K=None, D=None, frame_id='arducam_optical_frame', focal_scale=1.0):
        self.size = (int(width), int(height))
        self.frame = frame_id
        self.focal_scale = float(focal_scale)
        if K is None:
            # Nominal 185 deg circular fisheye on IMX219: f ~ 705 px at 3280x2464
            f = 0.286 * height * self.focal_scale
            self.K = np.array([[f, 0.0, (width - 1) / 2.0],
                               [0.0, f, (height - 1) / 2.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            self.K = np.array(K, dtype=np.float64).reshape(3, 3).copy()
            self.K[0, 0] *= self.focal_scale
            self.K[1, 1] *= self.focal_scale
        self.D = np.zeros((4, 1), dtype=np.float64) if D is None else np.array(D[:4], dtype=np.float64).reshape(4, 1)
        self._rect_cache = {}
        self._mask_cache = {}

    @classmethod
    def from_camera_info(cls, info, focal_scale=1.0):
        return cls(
            width=info.width,
            height=info.height,
            K=info.k,
            D=info.d[:4] if len(info.d) >= 4 else [0.0, 0.0, 0.0, 0.0],
            frame_id=info.header.frame_id or 'arducam_optical_frame',
            focal_scale=focal_scale,
        )

    @classmethod
    def from_yaml(cls, path, target_width=None, target_height=None, focal_scale=1.0):
        p = Path(path)
        if not p.exists():
            w = target_width or 3280
            h = target_height or 2464
            return cls(width=w, height=h, focal_scale=focal_scale)
        c = yaml.safe_load(p.read_text())
        cw, ch = int(c['image_width']), int(c['image_height'])
        K = np.array(c['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
        D = np.array(c['distortion_coefficients']['data'][:4], dtype=np.float64).reshape(4, 1)
        model = cls(width=cw, height=ch, K=K, D=D, focal_scale=focal_scale)
        if target_width is not None and target_height is not None and (cw, ch) != (target_width, target_height):
            return model.scaled(target_width, target_height)
        return model

    def scaled(self, width, height):
        """Return a new FisheyeModel scaled to (width, height) using exact pixel-center math."""
        w, h = int(width), int(height)
        if (w, h) == self.size:
            return self
        cw, ch = self.size
        s = h / float(ch)
        crop = (cw * s - w) / 2.0
        K_new = np.array([
            [self.K[0, 0] * s, 0.0, (self.K[0, 2] + 0.5) * s - 0.5 - crop],
            [0.0, self.K[1, 1] * s, (self.K[1, 2] + 0.5) * s - 0.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return FisheyeModel(width=w, height=h, K=K_new, D=self.D.copy(), frame_id=self.frame, focal_scale=1.0)

    def to_camera_info_dict(self):
        return {
            'width': self.size[0],
            'height': self.size[1],
            'distortion_model': 'equidistant',
            'd': [float(x) for x in self.D.ravel()],
            'k': [float(x) for x in self.K.ravel()],
            'r': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'p': [
                float(self.K[0, 0]), 0.0, float(self.K[0, 2]), 0.0,
                0.0, float(self.K[1, 1]), float(self.K[1, 2]), 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        }

    def theta_d_from_theta(self, theta):
        t = np.asarray(theta, dtype=np.float64)
        k = self.D.ravel()
        t2 = t * t
        return t * (1.0 + k[0] * t2 + k[1] * (t2 ** 2) + k[2] * (t2 ** 3) + k[3] * (t2 ** 4))

    def theta_from_theta_d(self, theta_d):
        """Invert theta_d = theta * (1 + k1*theta^2 + ... + k4*theta^8) safely with Newton-Raphson + clamping."""
        rd = np.asarray(theta_d, dtype=np.float64)
        k = self.D.ravel()
        t = np.clip(rd.copy(), 0.0, 1.65)
        for _ in range(12):
            t2 = t * t
            f = t * (1.0 + k[0] * t2 + k[1] * (t2 ** 2) + k[2] * (t2 ** 3) + k[3] * (t2 ** 4)) - rd
            df = 1.0 + 3.0 * k[0] * t2 + 5.0 * k[1] * (t2 ** 2) + 7.0 * k[2] * (t2 ** 3) + 9.0 * k[3] * (t2 ** 4)
            step = np.where(np.abs(df) > 1e-4, f / df, 0.0)
            t = np.clip(t - step, 0.0, math.pi * 0.55)
        return t

    def rays(self, px, scale=1.0):
        """Pixels (N, 2) -> unit 3D rays (N, 3) in the optical frame (+z forward, +x right, +y down)."""
        d = np.asarray(px, dtype=np.float64).reshape(-1, 2) / float(scale)
        x = (d[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (d[:, 1] - self.K[1, 2]) / self.K[1, 1]
        rd = np.hypot(x, y)
        phi = np.arctan2(y, x)
        theta = self.theta_from_theta_d(rd)
        s = np.sin(theta)
        return np.column_stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)])

    def pixels(self, rays, scale=1.0):
        """Unit 3D rays (N, 3) -> pixels (N, 2)."""
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(r, axis=1, keepdims=True)
        r = r / np.maximum(norms, 1e-12)
        theta = np.arccos(np.clip(r[:, 2], -1.0, 1.0))
        phi = np.arctan2(r[:, 1], r[:, 0])
        td = self.theta_d_from_theta(theta)
        u = (self.K[0, 0] * td * np.cos(phi) + self.K[0, 2]) * float(scale)
        v = (self.K[1, 1] * td * np.sin(phi) + self.K[1, 2]) * float(scale)
        return np.column_stack([u, v])

    def circle_mask(self, width=None, height=None, max_theta_deg=82.0):
        """Binary mask (uint8, 255 inside active fisheye circle) for image size (width, height)."""
        w = int(width or self.size[0])
        h = int(height or self.size[1])
        key = (w, h, round(float(max_theta_deg), 1))
        if key not in self._mask_cache:
            m = self.scaled(w, h)
            td_max = float(m.theta_d_from_theta(math.radians(max_theta_deg)))
            rx = int(max(10, abs(m.K[0, 0] * td_max)))
            ry = int(max(10, abs(m.K[1, 1] * td_max)))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (int(round(m.K[0, 2])), int(round(m.K[1, 2]))),
                (rx, ry),
                0.0, 0.0, 360.0, 255, -1,
            )
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    @staticmethod
    def rot_to_ray(ray_d):
        """Rotation matrix R (camera -> virtual pinhole view) looking along unit ray `ray_d`."""
        z = np.asarray(ray_d, dtype=np.float64).ravel()
        z = z / max(np.linalg.norm(z), 1e-12)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(up, z))) > 0.95:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = np.cross(up, z)
        x /= max(np.linalg.norm(x), 1e-12)
        y = np.cross(z, x)
        return np.vstack([x, y, z])

    def rectify_patch(self, img, ray_d=None, fov_deg=75.0, zoom_scale=1.0, max_size=1600):
        """Warp a gnomonic (rectilinear pinhole) view of `img` centered along `ray_d` (None = optical axis).

        Returns:
            view: rectified image (n x n)
            meta: dict with ('f', 'cx', 'cy', 'R', 'size', 'img_size') to convert patch pixels -> 3D camera rays
        """
        h, w = img.shape[:2]
        m = self.scaled(w, h)
        d_key = None if ray_d is None else tuple(np.round(np.asarray(ray_d, float).ravel() / np.linalg.norm(ray_d), 3))
        key = (w, h, d_key, round(float(fov_deg), 1), round(float(zoom_scale), 2), int(max_size))
        if key not in self._rect_cache:
            f = float(m.K[0, 0]) * float(zoom_scale)
            n = int(min(max_size, max(128, round(2.0 * f * math.tan(math.radians(fov_deg) / 2.0)))))
            cx = (n - 1) / 2.0
            cy = (n - 1) / 2.0
            newK = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            R = np.eye(3, dtype=np.float64) if d_key is None else self.rot_to_ray(d_key)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(m.K, m.D, R, newK, (n, n), cv2.CV_16SC2)
            if len(self._rect_cache) > 24:
                self._rect_cache.clear()
            self._rect_cache[key] = (map1, map2, {
                'f': f, 'cx': cx, 'cy': cy, 'R': R, 'size': (n, n), 'img_size': (w, h), 'zoom_scale': float(zoom_scale)
            })
        map1, map2, meta = self._rect_cache[key]
        view = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return view, meta

    def patch_px_to_rays(self, px, meta):
        """Convert pixels (N, 2) from a rectified patch `meta` into unit 3D rays (N, 3) in the camera optical frame."""
        pts = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        v_cam = np.column_stack([
            (pts[:, 0] - meta['cx']) / meta['f'],
            (pts[:, 1] - meta['cy']) / meta['f'],
            np.ones(len(pts), dtype=np.float64),
        ])
        rays = v_cam @ meta['R']  # equivalent to (R.T @ v_cam.T).T
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.maximum(norms, 1e-12)

    def patch_px_to_fisheye_px(self, px, meta):
        """Convert pixels (N, 2) from a rectified patch `meta` back to raw fisheye image pixels (N, 2)."""
        rays = self.patch_px_to_rays(px, meta)
        w, h = meta['img_size']
        return self.scaled(w, h).pixels(rays)

    def localize_qr_3d(self, poly_fisheye_px, image_size, qr_size_m=1.0, plane_distance_m=0.0,
                       T_base_from_opt=None, modules=21):
        """Compute 3D position of a detected QR code from its 4 polygon corners in raw fisheye pixels.

        Args:
            poly_fisheye_px: (4, 2) array of corner pixels in `image_size = (w, h)`.
            image_size: (width, height) of the image where `poly_fisheye_px` was measured.
            qr_size_m: physical side length of the QR code in meters (e.g. 1.0 m for drone mission, or wall QR size).
            plane_distance_m: if > 0, perpendicular distance (m) from the camera to the target plane (e.g. 10.0 m
                              altitude or 3.0 m wall distance). Used for low-noise ray-plane intersection.
            T_base_from_opt: 4x4 homogeneous transform from `arducam_optical_frame` to `base_link` (drone frame).
            modules: number of QR modules per side (default 21 for v1, 25 for v2) to report px/module.

        Returns:
            dict with:
              - 'pos_camera': [x, y, z] in `arducam_optical_frame` (m)
              - 'pos_drone':  [x, y, z] in `base_link` (drone frame: +x forward, +y left, +z up) (m)
              - 'pos_camera_pnp': [x, y, z] from metric QR size PnP/angular subtense
              - 'pos_camera_plane': [x, y, z] from plane intersection (or None)
              - 'distance_m': Euclidean 3D distance from camera (m)
              - 'off_axis_deg': viewing angle off optical axis (deg)
              - 'confidence_weight': cos^4(theta) weight for multi-frame fusion
              - 'side_px': mean sensor side length in pixels
              - 'px_per_module': sensor pixels per QR module
              - 'corners_drone': 4x3 array of QR corners in `base_link`
        """
        w, h = int(image_size[0]), int(image_size[1])
        m = self.scaled(w, h)
        poly = np.asarray(poly_fisheye_px, dtype=np.float64).reshape(-1, 2)
        center_px = poly.mean(axis=0, keepdims=True)
        all_rays = m.rays(np.vstack([center_px, poly]))
        r_c = all_rays[0]
        r_corners = all_rays[1:]

        off_axis_rad = math.acos(max(-1.0, min(1.0, float(r_c[2]))))
        off_axis_deg = math.degrees(off_axis_rad)
        conf_weight = max(0.05, math.cos(min(off_axis_rad, math.radians(80.0))) ** 4)

        # Sensor pixel side length
        sides_px = [float(np.linalg.norm(poly[i] - poly[(i + 1) % len(poly)])) for i in range(len(poly))]
        side_px = float(np.mean(sides_px)) if sides_px else 0.0

        # Angular subtense of the 4 sides in 3D ray space
        side_angles = []
        for i in range(len(r_corners)):
            dot = float(np.dot(r_corners[i], r_corners[(i + 1) % len(r_corners)]))
            side_angles.append(math.acos(max(-1.0, min(1.0, dot))))
        # The maximum side angle corresponds to the un-foreshortened tangential edge of the square
        theta_tan = max(side_angles) if side_angles else 1e-3
        dist_ang = float(qr_size_m) / max(2.0 * math.tan(theta_tan / 2.0), 1e-4)

        # Metric PnP in the local ray-aligned tangent plane
        pos_cam_pnp = r_c * dist_ang
        if len(r_corners) == 4 and qr_size_m > 0:
            R_align = self.rot_to_ray(r_c)
            local_rays = r_corners @ R_align.T
            if np.all(local_rays[:, 2] > 0.1):
                uv_norm = local_rays[:, :2] / local_rays[:, 2:3]
                s2 = float(qr_size_m) / 2.0
                obj_pts = np.array([[-s2, -s2, 0.0], [s2, -s2, 0.0], [s2, s2, 0.0], [-s2, s2, 0.0]], dtype=np.float64)
                flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
                try:
                    ok, _, tvec = cv2.solvePnP(obj_pts, uv_norm.reshape(4, 1, 2), np.eye(3), np.zeros(4), flags=flag)
                    if ok:
                        p_local = tvec.ravel()
                        dist_pnp = float(np.linalg.norm(p_local))
                        if 0.2 * dist_ang <= dist_pnp <= 3.0 * dist_ang:
                            pos_cam_pnp = R_align.T @ p_local
                except cv2.error:
                    pass

        # Plane intersection (along optical axis z = plane_distance_m, e.g. nadir altitude or perpendicular wall at 3 m)
        pos_cam_plane = None
        corners_cam = None
        if plane_distance_m > 0.0 and r_c[2] > 0.08:
            scale_c = float(plane_distance_m) / float(r_c[2])
            pos_cam_plane = r_c * scale_c
            if np.all(r_corners[:, 2] > 0.05):
                scales = float(plane_distance_m) / r_corners[:, 2:3]
                corners_cam = r_corners * scales

        # Primary camera-frame position: use plane intersection when plane_distance_m is provided (much lower depth
        # variance at 10 m when the QR is ~70 px), otherwise use PnP metric range
        pos_cam = pos_cam_plane if pos_cam_plane is not None else pos_cam_pnp
        if corners_cam is None:
            dist_use = float(np.linalg.norm(pos_cam))
            corners_cam = r_corners * dist_use

        if T_base_from_opt is None:
            T_base_from_opt = default_base_from_optical()
        R_bo = T_base_from_opt[:3, :3]
        t_bo = T_base_from_opt[:3, 3]
        pos_drone = R_bo @ pos_cam + t_bo
        corners_drone = (corners_cam @ R_bo.T) + t_bo

        return {
            'pos_camera': [round(float(v), 4) for v in pos_cam],
            'pos_drone': [round(float(v), 4) for v in pos_drone],
            'pos_camera_pnp': [round(float(v), 4) for v in pos_cam_pnp],
            'pos_camera_plane': [round(float(v), 4) for v in pos_cam_plane] if pos_cam_plane is not None else None,
            'distance_m': round(float(np.linalg.norm(pos_cam)), 4),
            'distance_pnp_m': round(float(np.linalg.norm(pos_cam_pnp)), 4),
            'off_axis_deg': round(float(off_axis_deg), 2),
            'confidence_weight': round(float(conf_weight), 4),
            'side_px': round(float(side_px), 1),
            'px_per_module': round(float(side_px / max(modules, 1)), 2),
            'center_px': [round(float(center_px[0, 0]), 1), round(float(center_px[0, 1]), 1)],
            'poly_px': [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
            'corners_drone': [[round(float(v), 4) for v in pt] for pt in corners_drone],
        }
