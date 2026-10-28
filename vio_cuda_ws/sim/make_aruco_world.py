#!/usr/bin/env python3
"""ArUco twin of a make_competition_world.py world: same course, zone, red zone and board positions, but every QR
board (the start board and the delivery boards) carries an ArUco marker instead - the ArUco delivery mission (8040).

    python3 sim/make_aruco_world.py competition                 -> worlds/competition_aruco.sdf / .json
    python3 sim/make_aruco_world.py comp_far1 --dicts 4X4 --seed 3

Markers 3x3 .. 6x6 (--dicts, default all four sizes mixed over the boards): 4x4 / 5x5 / 6x6 from the OpenCV
dictionaries, 3x3 a raw 3x3 bit grid inside the black border (no OpenCV dictionary has it). Payloads are what
aerothon_nav.aruco_tools reads: '4X4#23', '3X3#87' (the smallest of the code's four rotations). The start board
carries the target board's marker. The marker fills the board up to --margin cells of white round it.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / 'workspaces/isaac_ros-dev/src/aerothon_nav'))
from aerothon_nav.aruco_tools import DICTS, canon      # noqa: E402

AR = cv2.aruco


def marker_bits(kind, rng, used):
    """(payload, n x n bool bits) of a marker not used yet."""
    n = int(kind[0])
    while True:
        if kind == '3X3':
            bits = np.array([rng.random() < 0.5 for _ in range(9)]).reshape(3, 3)
            if bits.all() or not bits.any():
                continue
            payload = f'3X3#{canon(bits)[0]}'
        else:
            i = rng.randrange(50)                                 # in every _50 .. _1000 set
            d = AR.getPredefinedDictionary(DICTS[kind])
            draw = AR.generateImageMarker if hasattr(AR, 'generateImageMarker') else AR.drawMarker
            bits = draw(d, i, n + 2)[1:-1, 1:-1] < 128          # True = black cell
            bits = ~bits                                          # aruco_tools: True = white cell
            payload = f'{kind}#{i}'
        if payload not in used:
            used.add(payload)
            return payload, bits


def marker_texture(path, bits, margin, px=1024):
    """Black border + bits (True = white), `margin` cells of white round it, crisp edges."""
    n = bits.shape[0]
    cells = np.zeros((n + 2, n + 2), np.uint8)
    cells[1:-1, 1:-1] = np.where(bits, 255, 0)
    k = 4                                                         # margin in quarter cells
    img = np.kron(cells, np.ones((k, k), np.uint8))
    img = np.pad(img, int(round(margin * k)), constant_values=255)
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('world', help='a generated world name in worlds/ (its .sdf and .json)')
    ap.add_argument('--out', default='', help='output world name (default <world>_aruco)')
    ap.add_argument('--dicts', default='3X3,4X4,5X5,6X6', help='marker sizes to draw from, comma separated')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--margin', type=float, default=0.5, help='white cells round the marker on the board')
    a = ap.parse_args()
    worlds = HERE / 'worlds'
    src_sdf, src_json = worlds / f'{a.world}.sdf', worlds / f'{a.world}.json'
    name = a.out or f'{a.world}_aruco'
    sdf, truth = src_sdf.read_text(), json.loads(src_json.read_text())
    kinds = [k.strip().upper() for k in a.dicts.split(',') if k.strip()]
    rng, used = random.Random(a.seed), set()
    tex = worlds / 'textures'

    boards = truth['delivery_qrs']
    target = next(i for i, q in enumerate(boards) if q.get('target'))
    for i, q in enumerate(boards):
        kind = kinds[i % len(kinds)] if len(kinds) > 1 else kinds[0]
        payload, bits = marker_bits(kind, rng, used)
        q['payload'], q['marker'] = payload, kind
        q.pop('modules', None)
        q['bits'] = int(bits.shape[0])
        if i == target:
            target_bits = bits
        path = tex / f'{name}_{q["name"]}.png'
        marker_texture(path, bits, a.margin)
        sdf = sdf.replace(str(tex / f'{a.world}_{q["name"]}.png'), str(path))
    s = truth['start_qr']
    s['payload'], s['marker'], s['bits'] = boards[target]['payload'], boards[target]['marker'], int(target_bits.shape[0])
    s.pop('modules', None)
    path = tex / f'{name}_{s["name"]}.png'
    marker_texture(path, target_bits, a.margin)
    sdf = sdf.replace(str(tex / f'{a.world}_{s["name"]}.png'), str(path))
    # the other textures (walls, ground, banners) stay the source world's
    left = re.findall(rf'{re.escape(str(tex))}/{re.escape(a.world)}_qr_[^<]*', sdf)
    if left:
        raise SystemExit(f'QR textures not replaced: {left}')
    sdf = re.sub(r'<world name="[^"]+">', f'<world name="{name}">', sdf, count=1)
    truth['marker'] = 'aruco'
    truth['source_world'] = a.world
    (worlds / f'{name}.sdf').write_text(sdf)
    (worlds / f'{name}.json').write_text(json.dumps(truth, indent=2))
    print(f'wrote worlds/{name}.sdf / .json (world "{name}")')
    print(f'  start marker {s["payload"]} -> target {boards[target]["name"]} at ({boards[target]["x"]}, {boards[target]["y"]})')
    for q in boards:
        print(f'  {q["name"]}: {q["payload"]:10s} at ({q["x"]}, {q["y"]})')


if __name__ == '__main__':
    main()
