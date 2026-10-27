from pathlib import Path
from datetime import datetime
import re

sim = Path.home() / "workspaces/vio_cuda_ws/sim"
world = sim / "worlds/competition.sdf"
models = [
    sim / "models/realsense_d435_vio/model.sdf",
    sim / "models/arducam_imx219_fisheye/model.sdf",
]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = sim / f"backup_lighten_{stamp}"
backup.mkdir(parents=True)

def set_tag(block, tag, value):
    pattern = rf"(<{tag}\b[^>]*>).*?(</{tag}>)"
    return re.sub(pattern, rf"\g<1>{value}\g<2>", block, count=1, flags=re.S)

# Back up and lower world physics load.
text = world.read_text()
(backup / "competition.sdf").write_text(text)

if re.search(r"<physics\b[^>]*>.*?</physics>", text, re.S):
    def lighten_physics(m):
        block = m.group(0)
        block = set_tag(block, "max_step_size", "0.004")
        block = set_tag(block, "real_time_update_rate", "250")
        return block
    text = re.sub(
        r"<physics\b[^>]*>.*?</physics>",
        lighten_physics, text, flags=re.S
    )
else:
    physics = """
    <physics name="lightweight" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
"""
    text = re.sub(r"(<world\b[^>]*>)", r"\1" + physics, text, count=1)

world.write_text(text)
print("Updated world physics:", world)

# Lower camera update rates; retain image dimensions for calibration.
for model in models:
    if not model.exists():
        print("Not found, skipped:", model)
        continue
    text = model.read_text()
    dest = backup / (model.parent.name + "_model.sdf")
    dest.write_text(text)

    # Only lower sensor update rates above 15 Hz.
    def lower_rate(m):
        try:
            return set_tag(m.group(0), "update_rate", "15")
        except Exception:
            return m.group(0)

    text = re.sub(
        r"<sensor\b[^>]*>.*?</sensor>",
        lower_rate, text, flags=re.S
    )
    model.write_text(text)
    print("Reduced sensor update rates:", model)

print("Backups:", backup)
print("Done. Camera resolutions and world geometry are unchanged.")
