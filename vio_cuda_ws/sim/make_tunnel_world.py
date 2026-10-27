"""Generate a Gazebo (Harmonic) world with an open-roof tunnel and cylindrical pillars.

Built from PX4's own worlds/default.sdf (physics, light, GPS origin), so PX4 SITL
runs in it unchanged; PX4's world plugins come from its server.config.
The competition layout is unknown, so everything is a parameter. Pillars are
placed at random, but always leave one side gap of at least --min-gap so every
seed is passable.

    python3 sim/make_tunnel_world.py --seed 3 --pillars 3 -o sim/worlds/tunnel.sdf

World frame: ENU, drone spawns at the origin facing +x; the tunnel runs along +x.

Walls, pillars and the ground along the tunnel are textured (random high-contrast
patterns, built from 1 m panels so the texture isn't stretched) so feature-based VIO
has something to track; --plain gives single-colour surfaces, to test VIO without texture.
"""
import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

PX4_DEFAULT_WORLD = Path.home() / 'PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf'


def make_texture(path, seed, tint):
    """512x512 random pattern with plenty of corners: multi-scale blobs + rectangles + dots."""
    rng = np.random.default_rng(seed)
    n = 512
    img = np.zeros((n, n))
    for cells, weight in ((8, 1.0), (32, 0.6), (128, 0.3)):
        coarse = rng.random((cells, cells))
        img += weight * np.kron(coarse, np.ones((n // cells, n // cells)))
    img = (img > np.median(img)).astype(float) * 0.6 + 0.2
    for _ in range(60):
        x, y = rng.integers(0, n - 40, 2)
        w, h = rng.integers(6, 40, 2)
        img[y:y + h, x:x + w] = rng.choice([0.05, 0.95])
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(80):
        cx, cy, r = rng.integers(0, n), rng.integers(0, n), rng.integers(2, 7)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.choice([0.0, 1.0])
    rgb = np.clip(img[..., None] * np.array(tint)[None, None, :], 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def material(rgb, texture):
    r, g, b = rgb
    if texture is None:
        return f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>'
    return ('<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            f'<pbr><metal><albedo_map>{texture}</albedo_map><roughness>1</roughness>'
            '<metalness>0</metalness></metal></pbr></material>')


def box(name, x, y, z, sx, sy, sz, rgb, texture=None, collide=True):
    geom = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    col = f'<collision name="collision">{geom}</collision>' if collide else ''
    return f'''
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">{col}
        <visual name="visual">{geom}{material(rgb, texture)}</visual>
      </link>
    </model>'''


def cylinder(name, x, y, radius, height, rgb, texture=None):
    geom = f'<geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>'
    return f'''
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {height / 2} 0 0 0</pose>
      <link name="link">
        <collision name="collision">{geom}</collision>
        <visual name="visual">{geom}{material(rgb, texture)}</visual>
      </link>
    </model>'''


def place_pillars(rng, args):
    """Random pillars along the tunnel, each leaving a side gap >= min_gap."""
    half = args.width / 2
    pillars = []
    x = args.entry + args.first_pillar
    for i in range(args.pillars):
        if x > args.entry + args.length - 1.0:
            break
        r = rng.uniform(args.pillar_dia_min, args.pillar_dia_max) / 2
        # the pillar may sit anywhere that keeps the wider of the two gaps >= min_gap
        lim = half - r
        while True:
            y = rng.uniform(-lim, lim)
            gap_left = half - (y + r)
            gap_right = (y - r) + half
            if max(gap_left, gap_right) >= args.min_gap:
                break
        pillars.append((round(x, 3), round(y, 3), round(r, 3)))
        x += rng.uniform(args.spacing_min, args.spacing_max)
    return pillars


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='sim/worlds/tunnel.sdf')
    ap.add_argument('--name', default='tunnel', help='world name (PX4_GZ_WORLD)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=float, default=1.6, help='inner width, m (~2x a 12" quad)')
    ap.add_argument('--wall-height', type=float, default=2.0)
    ap.add_argument('--length', type=float, default=20.0)
    ap.add_argument('--entry', type=float, default=5.0, help='distance from spawn to tunnel mouth, m')
    ap.add_argument('--pillars', type=int, default=3)
    ap.add_argument('--first-pillar', type=float, default=4.0, help='mouth to first pillar, m')
    ap.add_argument('--spacing-min', type=float, default=3.0)
    ap.add_argument('--spacing-max', type=float, default=5.0)
    ap.add_argument('--pillar-dia-min', type=float, default=0.10)
    ap.add_argument('--pillar-dia-max', type=float, default=0.30)
    ap.add_argument('--min-gap', type=float, default=0.9, help='passable side gap: drone span + margin, m')
    ap.add_argument('--pillar-height', type=float, default=2.5)
    ap.add_argument('--plain', action='store_true', help='single-colour surfaces (no texture)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = {}
    if not args.plain:
        tex_dir = out.parent / 'textures'
        tex_dir.mkdir(exist_ok=True)
        for i, (kind, tint) in enumerate((('wall', (1.0, 0.85, 0.7)), ('ground', (0.6, 0.8, 0.5)),
                                          ('pillar', (0.7, 0.75, 0.9)))):
            path = tex_dir / f'{out.stem}_{kind}.png'
            make_texture(path, args.seed * 10 + i, tint)
            tex[kind] = str(path)

    wall_t = 0.1
    off = args.width / 2 + wall_t / 2
    models = []
    # walls from 1 m panels: one texture per panel keeps the pattern dense
    n_panels = int(round(args.length))
    for side, y in (('left', off), ('right', -off)):
        for k in range(n_panels):
            x = args.entry + k + 0.5
            models.append(box(f'wall_{side}_{k}', x, y, args.wall_height / 2, 1.0, wall_t, args.wall_height,
                              (0.55, 0.45, 0.35), tex.get('wall')))
    # textured ground tiles from the spawn point to past the tunnel (visual only)
    if not args.plain:
        for ix in range(-2, int(args.entry + args.length) + 3):
            for iy in range(-3, 3):
                models.append(box(f'ground_{ix}_{iy}', ix + 0.5, iy + 0.5, 0.001, 1.0, 1.0, 0.002,
                                  (0.5, 0.6, 0.4), tex['ground'], collide=False))
    pillars = place_pillars(rng, args)
    for i, (x, y, r) in enumerate(pillars):
        models.append(cylinder(f'pillar_{i}', x, y, r, args.pillar_height, (0.2, 0.2, 0.25), tex.get('pillar')))

    sdf = PX4_DEFAULT_WORLD.read_text(encoding='utf-8-sig')
    sdf = re.sub(r'<world name="[^"]+">', f'<world name="{args.name}">', sdf, count=1)
    meta = (f'\n    <!-- generated by make_tunnel_world.py: seed {args.seed}, width {args.width} m, '
            f'length {args.length} m, entry x={args.entry} m; pillars (x, y, r): {pillars} -->')
    sdf = sdf.replace('</world>', meta + ''.join(models) + '\n  </world>', 1)

    out.write_text(sdf)
    print(f'wrote {out}: world "{args.name}", tunnel x {args.entry}..{args.entry + args.length} m, '
          f'width {args.width} m, {len(pillars)} pillars {pillars}, '
          f'{"plain" if args.plain else "textured"}, {len(models)} models')


if __name__ == '__main__':
    main()
