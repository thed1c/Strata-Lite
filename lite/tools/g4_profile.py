#!/usr/bin/env python3
"""Strata-Lite: a static expert profile from route-log .routes files (stage d's resident set).

    g4_profile.py OUT.txt RUN1.routes [RUN2.routes ...]

OUT.txt: one line per layer, "layer: e e e ..." with all experts ranked by how often they were routed (prefill +
decode), most used first. The profile is aggregate counts of the maintainer's real inputs: it stays on machine A with the
route logs, never in the repo. Stdlib only.
"""
import collections, struct, sys
from pathlib import Path

out, runs = sys.argv[1], sys.argv[2:]
cnt = collections.defaultdict(collections.Counter)
NE = int(__import__("os").environ.get("PROFILE_NE", 128))   # Qwen3.6: 256
for r in runs:
    b = Path(r).read_bytes(); o = 0
    while o < len(b):
        phase, il, k, _ = b[o:o + 4]; n, _ = struct.unpack_from('<II', b, o + 4); o += 12
        cnt[il].update(b[o:o + n * k]); o += n * k
with open(out, 'w') as f:
    for il in sorted(cnt):
        rank = sorted(range(NE), key=lambda e: (-cnt[il][e], e))
        f.write(f"{il}: {' '.join(map(str, rank))}\n")
print(out, len(cnt), 'layers from', len(runs), 'runs')
