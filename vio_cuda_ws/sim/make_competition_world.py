"""Generate the AeroTHON 2026 Mission 2 (SkyScan) world for Gazebo Harmonic, after Figure 3 of
the rulebook ("flight path and layout are subject to change on event day", so all is a parameter).

    python3 sim/make_competition_world.py                       # -> sim/worlds/competition.sdf + .json
    python3 sim/make_competition_world.py --seed 4 --target random -o sim/worlds/comp4.sdf
    sim/run_sim.sh sim/worlds/competition.sdf

World frame: ENU, the drone spawns at the origin facing +x (the take-off pad).
  take-off pad          blue, centred on the origin
  start QR              flat on the ground, --start-qr-x ahead (rulebook: climb to 5 m, move ~1 m, scan)
  two corridors         3.5 m wide, --corridor-length long, walls taller than 10 ft (height not given):
                        forward lap on the LEFT (+y), return lap on the RIGHT (-y), --corridor-gap
                        between them; static obstacles only in the return corridor (entry corridor is
                        clear), each leaving a passable gap
  banners               "AeroTHON 2026", at each corridor entrance: GREEN background on the entry
                        (forward) corridor, ORANGE on the exit (return) one; the forward one faces the
                        pad, the return one the delivery zone. Organisers (2026-10-19): TWO per corridor,
                        one across the top of the entrance and one on the ground in front of it
                        (--banner-mount both, default); size not given. top / ground / side alone are
                        for tests (side: on a stand outside the outer wall - not on the day).
  delivery zone         40 x 30 m (geofence given on the day), starts where the corridors end
  QR boards             1.5 m, flat on the ground; one carries the start QR's message (the target),
                        the others decoys
  red zones             flat red sheets with a white border; flying over one is a violation

Course v3 (organisers, relayed 2026-10-24): two ROOFED tunnels 22 x 11 ft (6.71 x 3.35 m), 10 ft (3.05 m) high -
the drone stays below 8 ft inside; three floor-standing obstacles 6 ft tall x 3.5 ft wide in the exit tunnel (passed
sideways); banners 11.5 x 1 ft at the top and on the floor at BOTH ends: entry tunnel green outside (start side) /
blue inside (zone side), exit tunnel blue outside / green zone side - the tunnel to fly into always shows green;
QR boards 1 x 1 m with no white quiet zone, start QR 1 m from take-off, no landing pad; one red zone 10 x 10 ft;
QR zone 30 x 40 m.
    python3 sim/make_competition_world.py --course v3 -o sim/worlds/competition.sdf

The QR codes are real (segno, installed to sim/.pylib: pip install --target sim/.pylib segno)
and decode with pyzbar / cv2.QRCodeDetector.
Walls, obstacles and the corridor floor are textured for VIO. The ground truth (every object,
payload and the target) is written next to the world as <name>.json for scoring.
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_tunnel_world import PX4_DEFAULT_WORLD, make_texture, material

sys.path.insert(0, str(Path(__file__).resolve().parent / '.pylib'))
import segno  # noqa: E402

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# delivery-zone contents in zone coordinates (u: metres from the zone's corridor-side edge,
# v: metres left of the zone's centre line), read off Figure 3
FIG3_QRS = [(6.8, 10.9), (34.4, 10.9), (8.3, -3.0), (17.9, -11.6), (28.4, -8.0)]
FIG3_RED = [(27.9, 2.0, 10.0, 6.0),   # u, v, length along x, width along y
            (15.0, 4.5, 4.0, 3.0)]    # second, smaller zone between the corridor exit and the QRs


def visual(name, x, y, z, sx, sy, sz, rgb, texture=None):
    return (f'<visual name="{name}"><pose>{x} {y} {z} 0 0 0</pose>'
            f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{material(rgb, texture)}</visual>')


def collision(name, x, y, z, sx, sy, sz):
    return (f'<collision name="{name}"><pose>{x} {y} {z} 0 0 0</pose>'
            f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>')


def model(name, parts):
    """One static model holding many visuals/collisions: far cheaper than a model per panel."""
    return f'''
    <model name="{name}">
      <static>true</static>
      <link name="link">{''.join(parts)}</link>
    </model>'''


def textured_wall(prefix, x0, y, length, height, thick, tex, band=(-1e9, 1e9)):
    """Wall along x from x0, from 1 x 1 m textured panels (keeps the pattern dense), one collision. Only the metre
    rows overlapping `band` (z from, to: what the cameras see at flight height) are panels; below and above it one
    plain visual each - a 12 m tall-tunnel wall as panels tripled the world's visuals (render load on the Jetson)."""
    parts = [collision(f'{prefix}_col', x0 + length / 2, y, height / 2, length, thick, height)]
    rows = [k for k in range(int(np.ceil(height))) if k + 1 > band[0] and k < band[1]]
    for i in range(int(np.ceil(length))):
        w = min(1.0, length - i)
        for k in rows:
            h = min(1.0, height - k)
            parts.append(visual(f'{prefix}_{i}_{k}', x0 + i + w / 2, y, k + h / 2, w, thick, h,
                                (0.55, 0.45, 0.35), tex))
    if rows:
        for name, z0, z1 in (('below', 0.0, rows[0]), ('above', rows[-1] + 1.0, height)):
            if z1 - z0 > 1e-6:
                parts.append(visual(f'{prefix}_{name}', x0 + length / 2, y, (z0 + z1) / 2, length, thick, z1 - z0,
                                    (0.55, 0.45, 0.35)))
    return parts


def qr_texture(path, payload, px=1024, quiet=4):
    """Board texture: the QR plus `quiet` modules of white border filling it (v3: 0 - no border), crisp edges."""
    qr = segno.make(payload, error='m', micro=False)
    dark = np.array(qr.matrix, dtype=bool)                     # modules, no quiet zone
    modules = dark.shape[0]
    img = np.pad(np.where(dark, 0, 255).astype(np.uint8), quiet, constant_values=255)
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(img).convert('RGB').save(path)
    return modules


def banner_texture(path, background, text='AeroTHON 2026', size=(1400, 400)):
    img = Image.new('RGB', size, background)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 190)
    w, h = d.textbbox((0, 0), text, font=font)[2:]
    d.text(((size[0] - w) / 2, (size[1] - h) / 2 - 20), text, fill=(255, 255, 255), font=font)
    img.save(path)


def place_obstacles(rng, n, x0, length, half_w, min_gap, size):
    """Figure 3 style: blocks against alternating walls, each leaving >= min_gap free."""
    obs = []
    if n <= 0:
        return obs
    step = length / (n + 1)
    side = rng.choice((1, -1))
    for i in range(n):
        x = x0 + step * (i + 1) + rng.uniform(-0.2, 0.2) * step
        depth = rng.uniform(0.8, 2 * half_w - min_gap)            # how far it sticks out of the wall
        y = side * (half_w - depth / 2)
        obs.append((round(x, 2), round(y, 2), size, round(depth, 2)))
        side = -side
    return obs


def place_obstacles_v2(rng, n, x0, length, half_w, wall_h, kinds, drone=(0.6, 0.3), clear=0.15, cruise=1.4):
    """Course v2 exit-corridor obstacles. Each one BLOCKS the cruise line (corridor centre at `cruise` height),
    so the drone must change height and/or move sideways, and each stays passable for a drone (width, height)
    with `clear` sideways and >= 0.3 m above / below (PX4 height error included):
      box      on the floor, full width, top 1.2-1.7 m                       -> climb over
      beam     across the corridor, 0.3 m thick, spanning the cruise height    -> over or under
      hanging  block 0.6-0.8 m wide next to one wall, 1.0-1.9 m, rod to the wall top -> beside if room, else under
      partial  from one wall, full height, leaving a 0.95-1.05 m gap           -> move sideways
      window   full panel with an opening 0.95-1.1 m wide x 0.9-1.0 m high, at a random side and height
               -> sideways AND up / down
    Heights below are for cruise 1.4 m; another cruise shifts them all by cruise - 1.4 (a floor box grows to keep
    its top that far below / above the cruise line). Corridor-relative: y across from the centre line, z up from
    the ground. Spaced ~2 m along x.
    Returns dicts {kind, x, y, z, size_x/y/z (bounding box), boxes: [(x, y, z, sx, sy, sz)] for the sim}."""
    out = []
    if n <= 0:
        return out
    step = length / (n + 1)
    width = 2 * half_w
    dz = cruise - 1.4
    for i in range(n):
        x = round(x0 + step * (i + 1) + rng.uniform(-0.1, 0.1) * step, 2)
        kind = rng.choice(kinds)
        d = round(rng.uniform(0.25, 0.4), 2)                      # depth along the corridor
        side = rng.choice((1, -1))
        if kind == 'box':
            top = round(rng.uniform(1.2, 1.7) + dz, 2)
            boxes = [(x, 0.0, top / 2, d, width, top)]
        elif kind == 'beam':
            under = round(rng.uniform(cruise - 0.25, cruise + 0.05), 2)   # 0.3 m thick: covers the cruise height
            boxes = [(x, 0.0, under + 0.15, d, width, 0.3)]
        elif kind == 'hanging':
            bottom, w = round(rng.uniform(1.0, 1.2) + dz, 2), round(rng.uniform(0.6, 0.8), 2)
            top = round(rng.uniform(1.7, 1.9) + dz, 2)
            y = side * (half_w - 0.1 - w / 2)
            boxes = [(x, y, (bottom + top) / 2, d, w, top - bottom),
                     (x, y, (top + wall_h) / 2, 0.04, 0.04, wall_h - top)]                   # the rod
        elif kind == 'partial':
            gap = round(rng.uniform(0.95, 1.05), 2)
            depth_y = round(width - gap, 2)
            boxes = [(x, side * (half_w - depth_y / 2), wall_h / 2, d, depth_y, wall_h)]
        else:                                                      # window
            ow, oh = round(rng.uniform(0.95, 1.1), 2), round(rng.uniform(0.9, 1.0), 2)
            oy = side * round(half_w - ow / 2 - rng.uniform(0.0, 0.1), 2)       # opening off-centre
            # bottom edge: low (0.5-0.9) or high (1.4-1.7) - opening top stays <= ~2.7 m and its centre inside
            # the 0.5-2.3 m flight band with 0.1 m to spare (seeds 2 / 7 had openings out of reach at 2.1)
            oz = round((rng.uniform(0.5, 0.9) if rng.random() < 0.5 else rng.uniform(1.4, 1.7)) + dz, 2)
            y_lo, y_hi = oy - ow / 2, oy + ow / 2
            boxes = []
            if y_lo > -half_w + 0.01:                                            # left of the opening (-y)
                boxes.append((x, (-half_w + y_lo) / 2, wall_h / 2, d, y_lo + half_w, wall_h))
            if y_hi < half_w - 0.01:                                             # right of it (+y)
                boxes.append((x, (y_hi + half_w) / 2, wall_h / 2, d, half_w - y_hi, wall_h))
            boxes.append((x, oy, oz / 2, d, ow, oz))                             # below the opening
            if oz + oh < wall_h - 0.01:
                boxes.append((x, oy, (oz + oh + wall_h) / 2, d, ow, wall_h - oz - oh))  # above it
        boxes = [tuple(round(v, 3) for v in b) for b in boxes]
        xs = [b[0] - b[3] / 2 for b in boxes] + [b[0] + b[3] / 2 for b in boxes]
        ys = [b[1] - b[4] / 2 for b in boxes] + [b[1] + b[4] / 2 for b in boxes]
        zs = [b[2] - b[5] / 2 for b in boxes] + [b[2] + b[5] / 2 for b in boxes]
        o = {'kind': kind, 'x': x, 'y': round((min(ys) + max(ys)) / 2, 3), 'z': round((min(zs) + max(zs)) / 2, 3),
             'size_x': round(max(xs) - min(xs), 3), 'size_y': round(max(ys) - min(ys), 3),
             'size_z': round(max(zs) - min(zs), 3),
             'parts': [{'x': b[0], 'y': b[1], 'z': b[2], 'size_x': b[3], 'size_y': b[4], 'size_z': b[5]} for b in boxes],
             'boxes': boxes}
        if kind == 'window':
            o['opening'] = {'y': oy, 'z_bottom': oz, 'width': ow, 'height': oh}
        out.append(o)
    return out


def random_layout(rng, zl, zw, n, qr, red=3.048, edge=2.5, spacing=7.0, red_clear=1.5, red_u_min=4.0):
    """(QR (u, v) list, [(u, v, sx, sy)] red zone) in zone coordinates, drawn until every rule holds."""
    for _ in range(10000):
        pts = []
        while len(pts) < n:
            for _ in range(2000):
                p = (round(rng.uniform(edge, zl - edge), 2), round(rng.uniform(-zw / 2 + edge, zw / 2 - edge), 2))
                if all(math.dist(p, q) >= spacing for q in pts):
                    pts.append(p)
                    break
            else:
                break
        if len(pts) < n:
            continue
        lim = qr / 2 + red / 2 + red_clear          # board edge to red edge >= red_clear (Chebyshev on centres)
        for _ in range(2000):
            r = (round(rng.uniform(red_u_min, zl - red / 2 - 1.0), 2), round(rng.uniform(-zw / 2 + red, zw / 2 - red), 2))
            if all(max(abs(r[0] - u), abs(r[1] - v)) >= lim for u, v in pts):
                return pts, [(r[0], r[1], red, red)]
    raise RuntimeError('no random layout satisfies the rules')


def place_obstacles_v3(rng, n, length, half_w, height=1.8288, width=1.0668, depth=0.5, roof=3.048,
                       kinds=('standing', 'hanging', 'standing'), shift=0.0):
    """v3 exit-tunnel obstacles: n blocks `height` x `width` across against alternating walls (random first side),
    evenly spaced along the tunnel - each leaves 2.28 m beside it for the 0.85 m drone. kinds[i]: standing = on the
    floor (top 1.83 m: no way over below 8 ft); hanging = from the roof down to roof - height (1.22 m): beside or
    under (user, 2026-10-24: not all passable the same way). shift: the MIDDLE block moved this far off its wall
    towards the centre line (user, 2026-10-25) - the slot it leaves at the wall is too narrow to fly, the free side
    narrows by `shift`. Corridor-relative like place_obstacles_v2."""
    out = []
    side = rng.choice((1, -1))
    step = length / (n + 1)
    for i in range(n):
        x = round(step * (i + 1) + rng.uniform(-0.1, 0.1) * step, 2)
        y = round(side * (half_w - width / 2 - (shift if i == n // 2 else 0.0)), 3)
        kind = kinds[i % len(kinds)]
        zc = height / 2 if kind == 'standing' else roof - height / 2
        b = (x, y, zc, depth, width, height)
        out.append({'kind': kind, 'x': x, 'y': y, 'z': round(zc, 3), 'size_x': depth,
                    'size_y': round(width, 3), 'size_z': round(height, 3),
                    'parts': [{'x': b[0], 'y': b[1], 'z': b[2], 'size_x': b[3], 'size_y': b[4], 'size_z': b[5]}],
                    'boxes': [b]})
        side = -side
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='sim/worlds/competition.sdf')
    ap.add_argument('--name', default='competition', help='world name (PX4_GZ_WORLD)')
    ap.add_argument('--seed', type=int, default=0, help='payloads, obstacle placement, textures')
    ap.add_argument('--start-qr-x', type=float, default=None, help='start QR centre ahead of the pad, m (v3 1.0, '
                                                                     'else 1.5)')
    ap.add_argument('--corridor-x', type=float, default=5.0, help='corridor entrances, m from spawn')
    ap.add_argument('--corridor-length', type=float, default=None, help='m (v3 6.71 = 22 ft, else 10)')
    ap.add_argument('--course', choices=('v3', 'v2', 'v1'), default='v2',
                    help='v2 (organisers, 2026-10-19 evening): corridors 1.52-1.83 m wide, walls 2.74 m + 0.30 m '
                         'top banner, mixed exit-corridor obstacles passed over / under / beside; v1: the first '
                         'layout (3.5 m wide, 5 m walls, full-height blocks)')
    ap.add_argument('--corridor-width', type=float, default=None,
                    help='m; default v1 3.5, v2 random 1.52-1.83 per corridor (5-6 ft)')
    ap.add_argument('--corridor-gap', type=float, default=1.0, help='between the two corridors, m')
    ap.add_argument('--wall-height', type=float, default=None, help='m; default v1 5.0, v2 2.74 (9 ft)')
    ap.add_argument('--cruise', type=float, default=1.4,
                    help='v2: flight height in the corridors the exit obstacles are built around, m (tall tunnels '
                         'flown at 10 m: --wall-height 12 --cruise 10)')
    ap.add_argument('--obstacle-kinds', default='box,beam,hanging,partial,window',
                    help='v2 exit-corridor obstacle kinds to draw from (one kind = a focused test world)')
    ap.add_argument('--fwd-obstacles', type=int, default=0, help='entry corridor: none on the day')
    ap.add_argument('--ret-obstacles', type=int, default=4)
    ap.add_argument('--obstacle-size', type=float, default=0.5, help='block length along the corridor, m')
    ap.add_argument('--min-gap', type=float, default=1.5, help='free width left by each obstacle, m')
    ap.add_argument('--banner-mount', choices=('both', 'top', 'ground', 'side'), default='both')
    ap.add_argument('--zone-length', type=float, default=40.0)
    ap.add_argument('--zone-width', type=float, default=30.0)
    ap.add_argument('--qr-size', type=float, default=None, help='QR board edge, m (v3 1.0, else 1.5)')
    ap.add_argument('--obstacle-shift', type=float, default=0.0,
                    help='v3: the middle exit-tunnel block this far off its wall towards the centre, m')
    ap.add_argument('--layout', choices=('fig3', 'random'), default='fig3',
                    help='QR area: fig3 = the rulebook figure; random = 5 boards + the red zone placed from --seed '
                         '(boards >= 7 m apart and 2.5 m inside the edge, red zone >= 1.5 m clear of every board)')
    ap.add_argument('--elevation', type=float, default=488.0,
                    help="world origin height above sea level, m (PX4's GPS home; 488 = the Zurich default world)")
    ap.add_argument('--target', default='2',
                    help='index of the delivery QR that matches the start QR (0-4), or "random"')
    args = ap.parse_args()
    v3 = args.course == 'v3'
    args.start_qr_x = args.start_qr_x if args.start_qr_x is not None else (1.0 if v3 else 1.5)
    args.corridor_length = args.corridor_length or (6.706 if v3 else 10.0)
    args.qr_size = args.qr_size or (1.0 if v3 else 1.5)
    if v3 and args.ret_obstacles == 4:
        args.ret_obstacles = 3

    rng = random.Random(args.seed)
    out = Path(args.out).resolve()
    tex_dir = out.parent / 'textures'
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex = {}
    for i, (kind, tint) in enumerate((('wall', (1.0, 0.85, 0.7)), ('ground', (0.6, 0.8, 0.5)),
                                      ('obstacle', (0.45, 0.45, 0.5)))):
        tex[kind] = str(tex_dir / f'{out.stem}_{kind}.png')
        make_texture(tex[kind], args.seed * 10 + i, tint)
    for lap, colour in (('forward', (20, 140, 50)), ('return', (240, 120, 20)),     # green / orange (v1, v2)
                        ('green', (20, 140, 50)), ('blue', (25, 60, 200))):           # v3
        tex[f'banner_{lap}'] = str(tex_dir / f'{out.stem}_banner_{lap}.png')
        banner_texture(tex[f'banner_{lap}'], colour)

    models, truth = [], {'frame': 'world ENU, spawn at the origin facing +x', 'seed': args.seed}

    # ---- take-off pad and start QR
    if not v3:                                                  # v3: no landing pad
        models.append(model('takeoff_pad', [visual('pad', 0, 0, 0.002, 3.0, 3.0, 0.004, (0.15, 0.35, 0.8))]))

    words = ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO', 'FOXTROT', 'GOLF', 'HOTEL', 'KILO', 'LIMA']
    payloads = [f'AEROTHON26-{w}-{rng.randint(1000, 9999)}' for w in rng.sample(words, len(FIG3_QRS))]
    target = rng.randrange(len(FIG3_QRS)) if args.target == 'random' else int(args.target)

    def qr_board(name, x, y, payload):
        path = tex_dir / f'{out.stem}_{name}.png'
        modules = qr_texture(path, payload, quiet=0 if v3 else 4)
        s = args.qr_size
        models.append(model(name, [visual('board', 0, 0, 0, s, s, 0.01, (1, 1, 1), str(path))])
                      .replace('<static>true</static>', f'<static>true</static><pose>{x} {y} 0.006 0 0 0</pose>'))
        return {'name': name, 'x': round(x, 2), 'y': round(y, 2), 'size': s, 'payload': payload,
                'modules': modules}

    truth['start_qr'] = qr_board('qr_start', args.start_qr_x, 0.0, payloads[target])

    # ---- corridors
    v2 = args.course == 'v2'
    wall_h = args.wall_height or (3.0480 if v3 else 2.74 if v2 else 5.0)
    widths = {lap: args.corridor_width or (3.353 if v3 else round(rng.uniform(1.52, 1.83), 2) if v2 else 3.5)
              for lap in ('forward', 'return')}
    wall_t = 0.15
    x0, length = args.corridor_x, args.corridor_length
    truth['course'] = args.course
    truth['corridors'] = []
    for lap, sign, n_obs in (('forward', 1, args.fwd_obstacles), ('return', -1, args.ret_obstacles)):
        width = widths[lap]
        half = width / 2
        c = sign * (args.corridor_gap / 2 + wall_t + half)            # corridor centre line
        band = (args.cruise - 3.0, args.cruise + 2.0)           # cruise 1.4: every row (the old worlds)
        parts = (textured_wall('left', x0, c + half + wall_t / 2, length, wall_h, wall_t, tex['wall'], band) +
                 textured_wall('right', x0, c - half - wall_t / 2, length, wall_h, wall_t, tex['wall'], band))
        if v3:                                                        # the roof: tunnels, not open corridors
            parts += [collision('roof_col', x0 + length / 2, c, wall_h + 0.05, length, width + 2 * wall_t, 0.1),
                      visual('roof', x0 + length / 2, c, wall_h + 0.05, length, width + 2 * wall_t, 0.1,
                             (0.55, 0.45, 0.35), tex['wall'])]
        for i in range(int(np.ceil(length))):                         # textured floor (visual only)
            for j in range(int(np.ceil(width))):
                w, lx = min(1.0, width - j), min(1.0, length - i)
                parts.append(visual(f'floor_{i}_{j}', x0 + i + lx / 2, c - half + j + w / 2, 0.003, lx, w, 0.002,
                                    (0.5, 0.6, 0.4), tex['ground']))
        if v3:
            obs = place_obstacles_v3(rng, n_obs, length, half, shift=args.obstacle_shift)
            for o in obs:                                             # tunnel-relative x -> world
                o['x'] = round(x0 + o['x'], 2)
                o['parts'][0]['x'] = o['x']
                o['boxes'] = [(o['x'], *o['boxes'][0][1:])]
        elif v2:
            obs = place_obstacles_v2(rng, n_obs, x0, length, half, wall_h, args.obstacle_kinds.split(','),
                                     cruise=args.cruise)
        else:
            obs = [{'kind': 'block', 'x': ox, 'y': oy, 'z': wall_h / 2, 'size_x': sx, 'size_y': sy, 'size_z': wall_h}
                   for ox, oy, sx, sy in place_obstacles(rng, n_obs, x0, length, half, args.min_gap, args.obstacle_size)]
        for k, o in enumerate(obs):
            for j, (bx, by, bz, sx, sy, sz) in enumerate(o.pop('boxes', [(o['x'], o['y'], o['z'], o['size_x'],
                                                                         o['size_y'], o['size_z'])])):
                parts.append(collision(f'obstacle_{k}_{j}_col', bx, c + by, bz, sx, sy, sz))
                for zi in range(int(np.ceil(sz))):                  # 1 m textured layers (dense pattern)
                    h = min(1.0, sz - zi)
                    parts.append(visual(f'obstacle_{k}_{j}_{zi}', bx, c + by, bz - sz / 2 + zi + h / 2, sx, sy, h,
                                        (0.35, 0.35, 0.4), tex['obstacle']))
            o['y'] = round(c + o['y'], 2)
            for part in o.get('parts', []):                       # world frame, like the rest of the JSON
                part['y'] = round(c + part['y'], 3)
            if 'opening' in o:
                o['opening']['y'] = round(c + o['opening']['y'], 3)
        # banners, facing the drone coming in (outward = away from the corridor, along x). v1 / v2: at the entrance
        # only (green entry / orange exit). v3: both ends - entry green outside (start side) / blue zone side, exit
        # blue outside / green zone side: the tunnel to fly into always shows green
        if v3:
            colours = {'forward': ('green', 'blue'), 'return': ('blue', 'green')}[lap]
            ends = [(x0, -1, colours[0]), (x0 + length, 1, colours[1])]
        else:
            ends = [(x0, -1, lap) if lap == 'forward' else (x0 + length, 1, lap)]
        banners = []
        mounts = ('top', 'ground') if args.banner_mount == 'both' else (args.banner_mount,)
        for e, (ex, outward, colour) in enumerate(ends):
            yaw = 0 if outward < 0 else 3.14159
            for mount in mounts:
                if mount == 'top':                          # across the top of the entrance
                    bw, bh = (3.505, 0.3048) if v3 else (width + 2 * wall_t, (0.30 if v2 else 1.0))
                    # v2 / v3: 1 ft banner ON TOP of the walls (v3: on the roof edge); v1: the top metre
                    b = (ex + outward * 0.05, c, wall_h + 0.1 + bh / 2 if v3 else
                         wall_h + bh / 2 if v2 else wall_h - bh / 2)
                    normal = (outward, 0, 0)
                    geom = f'0.02 {bw} {bh}'
                elif mount == 'side':                       # on a stand outside the outer wall
                    bw, bh = 2.5, 1.0
                    b = (ex + outward * 0.3, c + sign * (half + wall_t + 1.5), 1.5)
                    normal = (outward, 0, 0)
                    geom = f'0.02 {bw} {bh}'
                    for k, dy in enumerate((-bw / 2, bw / 2)):
                        parts.append(visual(f'banner_pole_{e}_{k}', b[0], b[1] + dy, 1.0, 0.05, 0.05, 2.0,
                                            (0.3, 0.3, 0.3)))
                else:                                       # flat on the ground in front
                    bw, bh = (3.505, 0.3048) if v3 else (round(width + 0.3, 2), 0.6) if v2 else (3.0, 1.0)
                    b = (ex + outward * 1.5, c, 0.008)
                    normal = (0, 0, 1)
                    geom = f'{bh} {bw} 0.01'
                parts.append(f'<visual name="banner_{mount}_{e}"><pose>{b[0]} {b[1]} {b[2]} 0 0 {yaw}</pose>'
                             f'<geometry><box><size>{geom}</size></box></geometry>'
                             f'{material((1, 1, 1), tex[f"banner_{colour}"])}</visual>')
                banners.append({'mount': mount, 'end': 'start' if outward < 0 else 'zone', 'colour': colour,
                                'centre': [round(v, 3) for v in b], 'normal': normal, 'width': round(bw, 2),
                                'height': round(bh, 3)})
        models.append(model(f'corridor_{lap}', parts))
        truth['corridors'].append({
            'lap': lap, 'centre_y': round(c, 2), 'x_from': x0, 'x_to': x0 + length, 'width': width,
            'wall_height': wall_h, 'cruise': args.cruise, 'entrance_x': x0 if lap == 'forward' else x0 + length,
            'banner_colour': 'green' if v3 or lap == 'forward' else 'orange',   # the colour to fly into
            'roof': v3,
            'banners': banners,
            'obstacles': obs})

    # ---- delivery zone, QRs, red zones
    zx0 = x0 + length
    zl, zw = args.zone_length, args.zone_width
    border = 0.15
    zone = [visual('grass', zx0 + zl / 2, 0, 0.002, zl, zw, 0.004, (0.45, 0.7, 0.35))]
    for n, (bx, by, sx, sy) in {'n': (zx0 + zl / 2, zw / 2, zl, border), 's': (zx0 + zl / 2, -zw / 2, zl, border),
                                'w': (zx0, 0, border, zw), 'e': (zx0 + zl, 0, border, zw)}.items():
        zone.append(visual(f'line_{n}', bx, by, 0.003, sx, sy, 0.004, (1, 1, 1)))
    truth['delivery_zone'] = {'x_from': zx0, 'x_to': zx0 + zl, 'y_from': -zw / 2, 'y_to': zw / 2}
    truth['red_zones'] = []
    # v3: ONE red zone, 10 x 10 ft, 6 m into the zone between the corridors' line and the first QRs: on the way from
    # the entry corridor to the target / the sweep and back to the exit (user, 2026-10-24: see how it reacts)
    qr_uv, red_list = FIG3_QRS, [(6.0, 0.5, 3.048, 3.048)] if v3 else FIG3_RED   # v3 red: 1.5 m margin clear of qr_2
    if args.layout == 'random':
        qr_uv, red_list = random_layout(random.Random(args.seed + 1000), zl, zw, len(FIG3_QRS), args.qr_size)
    for k, (u, v, sx, sy) in enumerate(red_list):
        x, y = zx0 + u, v
        zone.append(visual(f'red_{k}_border', x, y, 0.005, sx + 1.0, sy + 1.0, 0.004, (1, 1, 1)))
        zone.append(visual(f'red_{k}', x, y, 0.007, sx, sy, 0.004, (0.85, 0.05, 0.05)))
        truth['red_zones'].append({'x': x, 'y': y, 'size_x': sx, 'size_y': sy,
                                   'x_from': x - sx / 2, 'x_to': x + sx / 2, 'y_from': y - sy / 2, 'y_to': y + sy / 2})
    models.append(model('delivery_zone', zone))
    truth['delivery_qrs'] = []
    for k, (u, v) in enumerate(qr_uv):
        q = qr_board(f'qr_{k}', zx0 + u, v, payloads[k])
        q['target'] = k == target
        truth['delivery_qrs'].append(q)
    truth['target'] = truth['delivery_qrs'][target]['name']

    sdf = PX4_DEFAULT_WORLD.read_text(encoding='utf-8-sig')
    sdf = re.sub(r'<world name="[^"]+">', f'<world name="{args.name}">', sdf, count=1)
    sdf = re.sub(r'<elevation>[^<]*</elevation>', f'<elevation>{args.elevation:g}</elevation>', sdf, count=1)
    meta = f'\n    <!-- generated by make_competition_world.py, seed {args.seed}; ground truth: {out.stem}.json -->'
    sdf = sdf.replace('</world>', meta + ''.join(models) + '\n  </world>', 1)
    out.write_text(sdf)
    out.with_suffix('.json').write_text(json.dumps(truth, indent=2))

    print(f'wrote {out} (world "{args.name}") and {out.with_suffix(".json").name}')
    print(f'  start QR at x={args.start_qr_x}: {payloads[target]!r} -> target {truth["target"]} '
          f'at ({truth["delivery_qrs"][target]["x"]}, {truth["delivery_qrs"][target]["y"]})')
    for c in truth['corridors']:
        print(f'  {c["lap"]:7s} corridor y={c["centre_y"]:+.2f}, x {c["x_from"]}..{c["x_to"]}, '
              f'obstacles {[(o["x"], o["y"]) for o in c["obstacles"]]}')
    print(f'  delivery zone x {zx0}..{zx0 + zl}, y +-{zw / 2}; red zones '
          f'{[(r["x"], r["y"], r["size_x"], r["size_y"]) for r in truth["red_zones"]]}')


if __name__ == '__main__':
    main()
