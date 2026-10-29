# `drone_qr_detector` — IMX219 Wide-Angle Fisheye QR Detector & 3D Drone-Frame Localizer (ROS 2 Humble)

Standalone ROS 2 package for detecting $1\text{ m} \times 1\text{ m}$ QR codes from $10\text{ m}$ altitude (and $30\text{ cm} \times 30\text{ cm}$ QR codes on a $3\text{ m}$ calibration wall) using an Arducam IMX219 wide-angle fisheye camera on NVIDIA Jetson, providing 3D coordinates of each detected QR code in the **drone body frame (`base_link`)** and **camera optical frame (`arducam_optical_frame`)**.

## Key Features
- **Shared / Non-Intrusive CSI Access (`csi_camera_node`):** Automatically shares full-resolution ($3280 \times 2464$) frames via `http://127.0.0.1:8090/frame.jpg` if `arducam_calib.py` is running, relays existing ROS 2 camera topics, or opens GStreamer `nvarguscamerasrc` (`GRAY8`) directly when free.
- **Gnomonic Patch Rectification (`fisheye_model`):** Unwarps both the central nadir FOV and off-axis candidate rays into rectilinear pinhole patches before running QR detection/decoding, then maps polygon corners back to exact unit 3D rays.
- **Multi-Scale `QrNet` + WeChat Super-Resolution + Phase-Correlated Frame Stacking (`qr_tools`, `qr_drone_locator`):** Detects and decodes low-px/module QR codes at $10\text{ m}$ altitude and outputs 3D drone-frame positions (`base_link`) + pairwise 3D distances.
- **Outlier-Rejecting Fisheye Calibrator & 3 m Wall Evaluator (`calib_and_wall_eval`):** Iteratively drops rolling-shutter outlier views during checkerboard calibration and verifies/calibrates metric focal scale against known inter-QR lengths on a 3 m wall.

See [`CALIBRATION_AND_USAGE_GUIDE.md`](./CALIBRATION_AND_USAGE_GUIDE.md) for full calibration and launch instructions.
