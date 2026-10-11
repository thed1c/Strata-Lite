#!/bin/bash
# Strata-Lite: bring one grid column (numbers only) into the repo and refresh the README's results grid.
# usage: bash tests/grid_collect.sh TAG [ssh-host]   (no host: the column is on this machine)
# env GRID=grid-qwen: the Qwen3.6 grid (tests/grid_qwen.sh; README markers GRID-QWEN:START/END, 40 expert blocks)
set -u
TAG=$1; H=${2:-}; R=$(cd "$(dirname "$0")/.." && pwd); E=$WORK
GRID=${GRID:-grid}; MK=$([ $GRID = grid ] && echo GRID || echo GRID-QWEN); export GRID_BLOCKS=$([ $GRID = grid ] && echo 30 || echo 40)
D=$R/results/$GRID/$TAG; rm -rf $D; mkdir -p $D
list="cd $E/$GRID/$TAG && find . -path ./bad-port -prune -o \( -name speed.json -o -name 'speed-*.json' -o -name 'q-*.json' -o -name 'q-*.cmp' -o -name vision.json -o -name chunk -o -name ncpumoe -o -name vision_ncpumoe -o -name run.log \) -print0 | tar --null -T - -cf -"
if [ -n "$H" ]; then ssh $H "$list"; else bash -c "$list"; fi | tar -xf - -C $D
python3 $R/tests/grid_summary.py $R/results/$GRID > $R/results/$GRID/summary.md
python3 - "$R" "$GRID" "$MK" <<'PY'
import sys; R, G, MK = sys.argv[1:4]
s = open(f"{R}/README.md").read(); g = open(f"{R}/results/{G}/summary.md").read().strip()
if f"<!-- {MK}:START -->" not in s: sys.exit(f"README has no <!-- {MK}:START --> marker: the summary is in results/{G}/summary.md")
a = s.index(f"<!-- {MK}:START -->") + len(f"<!-- {MK}:START -->"); b = s.index(f"<!-- {MK}:END -->")
open(f"{R}/README.md", "w").write(s[:a] + "\n" + g + "\n" + s[b:])
PY
/bin/grep -rl '"text"' $D && echo "WARNING: text field in a result" || echo "collected $TAG: $(find $D -type f | wc -l) files, numbers only"
