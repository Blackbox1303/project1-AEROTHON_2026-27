#!/usr/bin/env python3
"""Which mission code flies: the working tree (current) or a committed version of the ~/workspaces git repo.

The mission nodes (aerothon_nav: supervisor, explorer, qr_locator, ...) run from source. A committed version gets its
own checkout (git worktree, detached, made once) under isaac_ros-dev/versions/<sha> - inside the container's mount -
and the container stack puts its aerothon_nav first on PYTHONPATH (container_mission.sh / container_real.sh,
AEROTHON_CODE_PATH). Only the Python nodes switch; launch files, configs and the dashboard stay current.
  code_version.py list            JSON: the recent commits [{sha, date, subject}]
  code_version.py ensure <rev>    make the checkout if needed; prints the container path for PYTHONPATH
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path.home() / 'workspaces'
VERSIONS = REPO / 'isaac_ros-dev/versions'
HOST_TO_CONTAINER = (str(REPO / 'isaac_ros-dev'), '/workspaces/isaac_ros-dev')


def git(*a):
    return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()


def versions(n=15):
    out = []
    for line in git('log', f'-n{n}', '--format=%h\t%cd\t%s', '--date=format:%Y-%m-%d %H:%M').splitlines():
        sha, date, subject = line.split('\t', 2)
        out.append({'sha': sha, 'date': date, 'subject': subject})
    return out


def ensure(rev):
    """-> the container path of that version's aerothon_nav (the PYTHONPATH entry)."""
    sha = git('rev-parse', '--short=7', f'{rev}^{{commit}}')
    path = VERSIONS / sha
    if not (path / 'isaac_ros-dev/src/aerothon_nav/aerothon_nav/__init__.py').exists():
        VERSIONS.mkdir(parents=True, exist_ok=True)
        (VERSIONS / 'COLCON_IGNORE').touch()          # its checkouts hold the same packages: colcon refused to build
        git('worktree', 'prune')
        git('worktree', 'add', '--detach', '--force', str(path), sha)
    host = path / 'isaac_ros-dev/src/aerothon_nav'
    return str(host).replace(*HOST_TO_CONTAINER, 1)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'list':
        print(json.dumps(versions()))
    elif len(sys.argv) == 3 and sys.argv[1] == 'ensure':
        print(ensure(sys.argv[2]))
    else:
        sys.exit(__doc__)
