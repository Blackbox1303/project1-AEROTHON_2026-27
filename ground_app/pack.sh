#!/bin/bash
# Pack this workspace for another computer (USB stick / scp):  ./pack.sh [out.tar.gz]   default ~/aerothon_here_we_come.tar.gz
# On that computer:  tar xzf aerothon_here_we_come.tar.gz -C ~ && ~/aerothon_here_we_come/install.sh
set -e
HERE=$(dirname "$(readlink -f "$0")")
OUT=${1:-$HOME/aerothon_here_we_come.tar.gz}
tar czf "$OUT" -C "$(dirname "$HERE")" --exclude="$(basename "$HERE")/pylib" --exclude='__pycache__' "$(basename "$HERE")"
echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"
echo "on the other computer:  tar xzf $(basename "$OUT") -C ~ && ~/aerothon_here_we_come/install.sh"
