"""Fisheye (equidistant / OpenCV fisheye) camera model from a sensor_msgs/CameraInfo.

The Arducam's camera_info (vio_bringup fisheye_camera_info, calibration/arducam_calib.py) uses
distortion_model 'equidistant' with D = k1..k4. Rays are in the camera's optical frame
(z forward, x right, y down), unit length. Works for rays up to and beyond 90 deg off axis
(a pinhole model cannot: the 175 deg lens sees past the image plane).

Includes gnomonic (pinhole) patch rectification (`rectify_patch`, `patch_px_to_fisheye_px`)
so QR detector nodes can steer a virtual pinhole camera along any candidate 3D ray and map
decoded polygon corners back to exact fisheye pixels and 3D rays.
"""
import math

import cv2
import numpy as np


def rot_to_ray(ray_d):
    """Orthonormal 3x3 rotation matrix R = [right, down, forward] whose +Z axis points along `ray_d`."""
    z_axis = np.asarray(ray_d, dtype=np.float64).reshape(3)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    ref_x = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(z_axis, ref_x)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    return np.column_stack([x_axis, y_axis, z_axis])


class Fisheye:
    def __init__(self, info):
        if info.distortion_model != 'equidistant':
            raise ValueError(f'expected an equidistant camera_info, got {info.distortion_model!r}')
        self.K = np.array(info.k, np.float64).reshape(3, 3)
        self.D = np.array(info.d[:4], np.float64).reshape(4, 1)
        self.size = (info.width, info.height)
        self.frame = info.header.frame_id
        self._map_cache = {}
        self._mask_cache = {}

    def _scaled_intrinsics(self, scale=1.0):
        s = float(scale)
        fx = self.K[0, 0] * s
        fy = self.K[1, 1] * s
        cx = (self.K[0, 2] + 0.5) * s - 0.5
        cy = (self.K[1, 2] + 0.5) * s - 0.5
        return fx, fy, cx, cy

    def rays(self, px, scale=1.0):
        """Pixels (N, 2) -> unit rays (N, 3). `scale`: the pixels come from an image resized by it."""
        d = np.asarray(px, np.float64).reshape(-1, 2)
        fx, fy, cx, cy = self._scaled_intrinsics(scale)
        x = (d[:, 0] - cx) / fx
        y = (d[:, 1] - cy) / fy
        rd, phi = np.hypot(x, y), np.arctan2(y, x)
        # invert theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8) by Newton, from theta = theta_d;
        # unlike cv2.fisheye.undistortPoints (which returns tan theta) this stays valid past 90 deg
        k = self.D.ravel()
        t = rd.copy()
        for _ in range(10):
            t2 = t * t
            f = t * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4) - rd
            deriv = 1 + 3 * k[0] * t2 + 5 * k[1] * t2 ** 2 + 7 * k[2] * t2 ** 3 + 9 * k[3] * t2 ** 4
            t -= f / np.where(np.abs(deriv) > 1e-6, deriv, 1e-6)
        t = np.clip(t, 0.0, math.pi * 0.98)
        s = np.sin(t)
        return np.column_stack([s * np.cos(phi), s * np.sin(phi), np.cos(t)])

    def pixels(self, rays, scale=1.0):
        """Rays (N, 3) in the optical frame -> pixels (N, 2)."""
        rays = np.asarray(rays, np.float64).reshape(-1, 3)
        fx, fy, cx, cy = self._scaled_intrinsics(scale)
        theta = np.arccos(np.clip(rays[:, 2] / np.linalg.norm(rays, axis=1), -1, 1))
        phi = np.arctan2(rays[:, 1], rays[:, 0])
        k = self.D.ravel()
        t2 = theta * theta
        td = theta * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4)
        return np.column_stack([fx * td * np.cos(phi) + cx, fy * td * np.sin(phi) + cy])

    def circle_mask(self, width, height, max_theta_deg=82.0):
        """Boolean mask of valid active fisheye image circle up to `max_theta_deg` off-axis."""
        key = (width, height, round(float(max_theta_deg), 1))
        if key in self._mask_cache:
            return self._mask_cache[key]
        scale = width / max(self.size[0], 1)
        fx, fy, cx, cy = self._scaled_intrinsics(scale)
        t = math.radians(max_theta_deg)
        t2 = t * t
        k = self.D.ravel()
        td = max(0.2, min(t * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4), 1.8))
        r_x = fx * td
        r_y = fy * td
        yy, xx = np.ogrid[:height, :width]
        mask = (((xx - cx) / max(r_x, 1.0)) ** 2 + ((yy - cy) / max(r_y, 1.0)) ** 2) <= 1.0
        if len(self._mask_cache) > 8:
            self._mask_cache.clear()
        self._mask_cache[key] = mask
        return mask

    def rectify_patch(self, img, ray_d=None, fov_deg=24.0, zoom_scale=2.2, max_size=512):
        """Steer a virtual gnomonic pinhole camera along `ray_d` to rectify a QR patch."""
        h, w = img.shape[:2]
        scale = w / max(self.size[0], 1)
        fx, fy, _, _ = self._scaled_intrinsics(scale)
        R = np.eye(3, dtype=np.float64) if ray_d is None else rot_to_ray(ray_d)
        f_p = 0.5 * (fx + fy) * float(zoom_scale)
        half_tan = math.tan(math.radians(fov_deg) / 2.0)
        size = int(min(max_size, max(160, round(2.0 * f_p * half_tan))))
        f_p = (size / 2.0) / half_tan
        c_p = (size - 1) / 2.0

        key = (
            w, h, size, round(f_p, 1),
            tuple(np.round(R[:, 2], 2)) if ray_d is not None else (0.0, 0.0, 1.0),
        )
        if key in self._map_cache:
            mx, my, R_cached = self._map_cache[key]
        else:
            u = (np.arange(size, dtype=np.float64) - c_p) / f_p
            gx, gy = np.meshgrid(u, u)
            rays_local = np.column_stack([gx.ravel(), gy.ravel(), np.ones(size * size, dtype=np.float64)])
            rays_cam = rays_local @ R.T
            px = self.pixels(rays_cam, scale=scale)
            mx = px[:, 0].reshape(size, size).astype(np.float32)
            my = px[:, 1].reshape(size, size).astype(np.float32)
            R_cached = R
            if len(self._map_cache) > 32:
                self._map_cache.clear()
            self._map_cache[key] = (mx, my, R_cached)

        patch = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        meta = {'R': R_cached, 'f_p': f_p, 'c_p': c_p, 'size': size, 'scale': scale}
        return patch, meta

    def patch_px_to_fisheye_px(self, pts_patch, meta):
        """Convert 2D pixels in a rectified patch directly back to original fisheye image pixels."""
        pts = np.asarray(pts_patch, dtype=np.float64).reshape(-1, 2)
        rays_local = np.column_stack([
            (pts[:, 0] - meta['c_p']) / meta['f_p'],
            (pts[:, 1] - meta['c_p']) / meta['f_p'],
            np.ones(len(pts), dtype=np.float64),
        ])
        rays_local /= np.linalg.norm(rays_local, axis=1, keepdims=True)
        rays_cam = rays_local @ meta['R'].T
        return self.pixels(rays_cam, scale=meta.get('scale', 1.0))


class Pinhole:
    """Same interface as Fisheye for a pinhole camera_info (the D435 colour camera): rays in the optical frame,
    distortion ignored (Gazebo's is zero, the D435's colour is ~1 px at the edge)."""

    def __init__(self, info):
        self.K = np.array(info.k, np.float64).reshape(3, 3)
        self.size = (info.width, info.height)
        self.frame = info.header.frame_id

    def rays(self, px, scale=1.0):
        d = np.asarray(px, np.float64).reshape(-1, 2) / scale
        r = np.column_stack([(d[:, 0] - self.K[0, 2]) / self.K[0, 0], (d[:, 1] - self.K[1, 2]) / self.K[1, 1],
                             np.ones(len(d))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def pixels(self, rays, scale=1.0):
        rays = np.asarray(rays, np.float64).reshape(-1, 3)
        z = np.where(rays[:, 2] > 1e-6, rays[:, 2], 1e-6)
        return np.column_stack([self.K[0, 0] * rays[:, 0] / z + self.K[0, 2], self.K[1, 1] * rays[:, 1] / z + self.K[1, 2]]) * scale


def camera_model(info):
    """Fisheye for an equidistant camera_info, else Pinhole."""
    return Fisheye(info) if info.distortion_model == 'equidistant' else Pinhole(info)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))))
