# IMX219 Wide-Angle Camera Calibration & `drone_qr_detector` Guide

## 1. Should You Use Checkerboard Calibration, the 3 m QR Wall, or Both?

For an **IMX219 circular fisheye lens ($\sim 185^\circ$ FOV, $f \approx 705\text{ px}$ at $3280 \times 2464$)** detecting $1\text{ m} \times 1\text{ m}$ QR codes from $10\text{ m}$ altitude, **you should use a two-step combination of both**:

| Step | Method | What It Calibrates Best | Why One Alone Is Not Enough |
| :--- | :--- | :--- | :--- |
| **Step 1: Intrinsic Distortion Curve** | **9×6 Checkerboard** (at $30\text{--}80\text{ cm}$ covering center + edges) | Principal point $(c_x, c_y)$ and 4th-order equidistant radial distortion $(k_1, k_2, k_3, k_4)$ across the full $185^\circ$ lens circle | QR codes on a wall 3 m away are only $\sim 60\text{--}80\text{ px}$ wide (4 corners each), which is too sparse to fit the 4th-order polynomial $(k_1..k_4)$ all the way to the lens periphery. |
| **Step 2: Metric Focal Scale & Extrinsic Verification** | **QR Codes on the Wall at $3.0\text{ m}$ with Known Inter-QR Distances** | Exact effective focal length scale (`focal_scale`) at $3\text{--}10\text{ m}$ infinity focus and end-to-end 3D metric distance accuracy | A close-up checkerboard ($40\text{ cm}$) can leave a $1\text{--}3\%$ scale bias at $3\text{--}10\text{ m}$, and any residual RMS error in $(f, k_1..k_4)$ shows up immediately when measuring the true distance $L_{ij}$ between two QR codes on your $3\text{ m}$ wall. |

---

## 2. Why the Previous Checkerboard Calibration Had High Error (`RMS = 4.44 px`) & How to Fix It

In `calibration/arducam_fisheye_1640x1232.yaml`, the 61-view calibration produced `rms_px: 4.4444` (whereas a clean fisheye calibration should be **$< 0.80\text{ px}$**). There are three physical causes on the IMX219:

1. **Rolling Shutter Warp (47 ms readout time at $3280 \times 2464$):**
   Even when exposure is capped at $5\text{--}8\text{ ms}$, the IMX219 sensor reads out its 2464 rows sequentially over **$\sim 47\text{ ms}$** (21 fps). If either the camera or the checkerboard is moving by hand while a frame is captured, the top of the checkerboard is sampled $20\text{--}30\text{ ms}$ before the bottom, skewing the corners by $3\text{--}10\text{ px}$.
2. **Paper Bowing on the Whiteboard:**
   When a printed checkerboard is taped only at the 4 corners, a $2\text{ mm}$ bow in the center of the sheet shifts off-axis fisheye rays by $2\text{--}4\text{ px}$.
3. **No Per-View RMS Outlier Rejection in `arducam_calib.py`:**
   `arducam_calib.py` only drops mathematically singular matrices (`CALIB_CHECK_COND`), keeping blurry/skewed frames that pull the polynomial fit off by 4+ pixels.

### Step-by-Step Procedure for a Sub-Pixel ($< 0.8\text{ px}$) Checkerboard Calibration
1. **Keep the board completely flat** (tape all 4 edges firmly to a rigid flat board or display `/board.png` fullscreen on a flat laptop screen with no glare).
2. **Pause completely for 1 full second at each pose** before capturing so the 47 ms rolling shutter has zero motion skew.
3. **Capture 20–30 views** covering:
   - Center (3–4 views, straight and tilted $\pm 30^\circ$)
   - Top, Bottom, Left, Right mid-radius ($30^\circ\text{--}55^\circ$ off-axis)
   - Outer ring ($55^\circ\text{--}75^\circ$ off-axis, tilted toward the camera so all 54 corners remain sharp)
4. **Run the Outlier-Rejecting Calibrator** included in `drone_qr_detector` (which automatically drops any frame with per-view RMS $> 0.9\text{ px}$):
   ```bash
   docker exec -u admin isaac_ros_dev-aarch64-pyzbar-container bash -c "
     source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash &&
     ros2 run drone_qr_detector calib_and_wall_eval calibrate \
       --frames-dir /workspaces/isaac_ros-dev/calibration/arducam_calib_frames \
       --out /workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml \
       --max-view-rms 0.90
   "
   ```

---

## 3. Calibrating & Verifying with Your QR Codes on the Wall at 3 m

A $30\text{ cm} \times 30\text{ cm}$ QR code at $3.0\text{ m}$ subtends the exact same angle ($0.1\text{ rad} \approx 70.5\text{ px}$) as a **$1\text{ m} \times 1\text{ m}$ QR code at $10.0\text{ m}$ altitude**!

With the camera placed **$3.0\text{ m}$ perpendicular distance** from the wall where multiple QR codes are attached at a known center-to-center distance (e.g. `1.20` m):

```bash
docker exec -u admin isaac_ros_dev-aarch64-pyzbar-container bash -c "
  source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash &&
  ros2 run drone_qr_detector calib_and_wall_eval wall_eval \
    --wall-distance 3.0 \
    --qr-size 0.30 \
    --known-dist-m 1.20 \
    --frames 5
"
```

What this does:
1. Pulls full-resolution $3280 \times 2464$ frames **non-intrusively** (works even while another person has `arducam_calib.py` open on port 8090, or uses direct CSI if free).
2. Rectifies the central view and local gnomonic patches around every candidate QR code on the wall.
3. Computes the 3D coordinates of each QR code in **both** the **Drone Frame (`base_link`)** and **Camera Optical Frame (`arducam_optical_frame`)**.
4. Computes the 3D Euclidean distance $\|\mathbf{p}_i - \mathbf{p}_j\|$ between every pair of detected QR codes on the wall, compares it against `--known-dist-m`, and prints the exact `focal_scale` parameter to eliminate any residual scale error!

---

## 4. Running `drone_qr_detector` (With Shared CSI Access)

### A. Flight Mode (10 m Altitude, 1 m × 1 m QR Codes)
```bash
docker exec -u admin isaac_ros_dev-aarch64-pyzbar-container bash -c "
  source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash &&
  ros2 launch drone_qr_detector drone_qr_detector.launch.py \
    target_distance_m:=10.0 \
    qr_size_m:=1.0
"
```

### B. 3 m Wall Test Mode
```bash
docker exec -u admin isaac_ros_dev-aarch64-pyzbar-container bash -c "
  source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash &&
  ros2 launch drone_qr_detector drone_qr_detector.launch.py \
    target_distance_m:=3.0 \
    qr_size_m:=0.30
"
```

### Published Topics
- `/drone_qr/detections` (`std_msgs/msg/String` JSON):
  Contains `detected_qrs` (each with `payload`, `position_drone_m: {x, y, z}` in `base_link`, `position_camera_m: {x, y, z}`, `position_camera_pnp_m: {x, y, z}`, `off_axis_deg`, `px_per_module`) and `pairwise_distances_m` between all detected QR codes.
- `/drone_qr/poses_drone` (`geometry_msgs/msg/PoseArray` in `base_link` frame).
- `/drone_qr/markers` (`visualization_msgs/msg/MarkerArray` in `base_link` frame for RViz).
- `/drone_qr/image_raw` & `/drone_qr/camera_info`: Full-resolution $3280 \times 2464$ `mono8` stream + calibrated equidistant `CameraInfo`.
