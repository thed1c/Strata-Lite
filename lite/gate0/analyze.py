#!/usr/bin/env python3
"""Strata-Lite Gate 0: routing skew and simulated expert-cache hit rates from route-log's .routes files.

Usage: analyze.py OUT.json RUN1.routes [RUN2.routes ...]
Pure Python (no numpy on machine A/machine B). Decode tokens are what the cache serves (prefill keeps the plain path),
so every hit rate below is over DECODE accesses; prefill only warms the cache.
"""
import collections, json, math, os, struct, sys
from pathlib import Path

# GATE0_NE / GATE0_K / GATE0_FRACS for other models (Qwen3.6-35B-A3B: 256 experts, 8 used)
NE, K = int(os.environ.get("GATE0_NE", 128)), int(os.environ.get("GATE0_K", 8))
FRACS = tuple(float(x) for x in os.environ["GATE0_FRACS"].split(",")) if os.environ.get("GATE0_FRACS") else \
        (0.25, 0.37, 0.50, 0.61, 0.65)   # Gemma 4: 0.61 = 5070 budget at 32k q4 KV, 0.37 = at 262k (measured VRAM)
WARM = 4096          # prefill tokens used to warm an adaptive cache (LRU/LFU state depends only on recent ones)


def load(path):
    """-> {layer: {'pre_n': int, 'pre_counts': Counter, 'pre_tail': [last WARM prefill tuples], 'dec': [tuples]}}.
    Prefill is streamed (157k-token prompts): only its counts and its last WARM tokens are kept."""
    b = Path(path).read_bytes(); o = 0
    L = collections.defaultdict(lambda: {'pre_n': 0, 'pre_counts': collections.Counter(),
                                         'pre_tail': collections.deque(maxlen=WARM), 'dec': []})
    while o < len(b):
        phase, il, k, _ = b[o:o + 4]; n, _pos0 = struct.unpack_from('<II', b, o + 4); o += 12
        ids = b[o:o + n * k]; o += n * k
        d = L[il]
        if phase:
            for r in range(n):
                d['dec'].append(tuple(ids[r * k:(r + 1) * k]))
        else:
            d['pre_n'] += n
            d['pre_counts'].update(ids)
            for r in range(max(0, n - WARM), n):
                d['pre_tail'].append(tuple(ids[r * k:(r + 1) * k]))
    for d in L.values():
        d['pre_tail'] = list(d['pre_tail'])
    return dict(L)


def freq(seq):
    c = collections.Counter()
    for t in seq: c.update(t)
    return c


def static_hits(seq, resident):
    h = sum(1 for t in seq for e in t if e in resident)
    return h, len(seq) * K


def lru(seq, warm, C):
    od = collections.OrderedDict()
    for t in warm:
        for e in t:
            od[e] = 1; od.move_to_end(e)
            if len(od) > C: od.popitem(last=False)
    h = 0; zero_miss = 0
    for t in seq:
        m = 0
        for e in t:
            if e in od: h += 1
            else: m += 1
        zero_miss += (m == 0)
        for e in t:
            od[e] = 1; od.move_to_end(e)
            if len(od) > C: od.popitem(last=False)
    return h, len(seq) * K, zero_miss


def lfu_decay(seq, warm, C, half_life, budget):
    """Frequency-weighted, decayed counts; resident set = top-C by score, at most `budget` swaps per token (None = unbounded)."""
    d = 0.5 ** (1.0 / half_life)
    s = [0.0] * NE
    for t in warm:
        for i in range(NE): s[i] *= d
        for e in t: s[e] += 1.0
    res = set(sorted(range(NE), key=lambda i: -s[i])[:C])
    h = 0; zero_miss = 0; swaps = 0
    for t in seq:
        m = 0
        for e in t:
            if e in res: h += 1
            else: m += 1
        zero_miss += (m == 0)
        for i in range(NE): s[i] *= d
        for e in t: s[e] += 1.0
        # swap in the best non-residents that beat the worst residents
        want = sorted((i for i in range(NE) if i not in res), key=lambda i: -s[i])
        worst = sorted(res, key=lambda i: s[i])
        n = 0
        for a, w in zip(want, worst):
            if budget is not None and n >= budget: break
            if s[a] <= s[w]: break
            res.discard(w); res.add(a); n += 1
        swaps += n
    return h, len(seq) * K, zero_miss, swaps


def main():
    out = sys.argv[1]; runs = sys.argv[2:]
    data = {Path(r).stem: load(r) for r in runs}
    layers = sorted(next(iter(data.values())).keys())
    R = {'runs': {}, 'per_layer': {}, 'overall': {}}
    for name, L in data.items():
        R['runs'][name] = {'prefill_tokens': L[layers[0]]['pre_n'], 'decode_tokens': len(L[layers[0]]['dec'])}

    acc = collections.defaultdict(lambda: [0, 0])     # policy -> [hits, total]
    zm = collections.defaultdict(lambda: [0, 0])      # policy -> [token-layers with zero misses, token-layers]
    sw = collections.defaultdict(int)
    for il in layers:
        PL = {}
        all_dec = freq(t for L in data.values() for t in L[il]['dec'])
        all_pre = sum((L[il]['pre_counts'] for L in data.values()), collections.Counter())
        tot = sum(all_dec.values()) or 1
        shares = sorted((c / tot for c in all_dec.values()), reverse=True) + [0.0] * (NE - len(all_dec))
        ent = -sum(p * math.log(p) for p in shares if p > 0)
        PL['decode_eff_experts'] = round(math.exp(ent), 1)               # 128 = uniform
        PL['decode_top_share'] = {f: round(sum(shares[:round(f * NE)]), 4) for f in FRACS}
        totp = sum(all_pre.values()) or 1
        sharesp = sorted((c / totp for c in all_pre.values()), reverse=True) + [0.0] * (NE - len(all_pre))
        PL['prefill_top_share'] = {f: round(sum(sharesp[:round(f * NE)]), 4) for f in FRACS}
        for f in FRACS:
            C = round(f * NE)
            for name, L in data.items():
                dec, warm = L[il]['dec'], L[il]['pre_tail']
                if not dec: continue
                # (a1) static oracle: top-C of this run's own decode (upper bound for any static set)
                own = freq(dec); res = {e for e, _ in own.most_common(C)}
                h, n = static_hits(dec, res); acc[(il, f, 'static_oracle')][0] += h; acc[(il, f, 'static_oracle')][1] += n
                # (a2) static profile, leave-one-out: top-C over every OTHER run (prefill + decode)
                other = collections.Counter()
                for n2, L2 in data.items():
                    if n2 != name: other.update(freq(L2[il]['dec'])); other.update(L2[il]['pre_counts'])
                if other:
                    res = {e for e, _ in other.most_common(C)}
                    h, n = static_hits(dec, res); acc[(il, f, 'static_profile_loo')][0] += h; acc[(il, f, 'static_profile_loo')][1] += n
                # (a3) static from this prompt's own prefill (chosen at prompt time)
                if L[il]['pre_n']:
                    res = {e for e, _ in L[il]['pre_counts'].most_common(C)}
                    h, n = static_hits(dec, res); acc[(il, f, 'static_prefill')][0] += h; acc[(il, f, 'static_prefill')][1] += n
                # (b) LRU, warmed by the prefill
                h, n, z = lru(dec, warm, C); acc[(il, f, 'lru')][0] += h; acc[(il, f, 'lru')][1] += n
                zm[(il, f, 'lru')][0] += z; zm[(il, f, 'lru')][1] += len(dec); sw[(il, f, 'lru')] += n - h   # every miss is copied in
                # (c) the planned policy: frequency-weighted decayed counts, bounded swaps
                for hl, bud in ((16, 2), (32, 2), (64, 2), (64, None), (256, 2)):
                    key = f'lfu_decay_hl{hl}_b{bud if bud is not None else "inf"}'
                    h, n, z, s = lfu_decay(dec, warm, C, hl, bud)
                    acc[(il, f, key)][0] += h; acc[(il, f, key)][1] += n
                    zm[(il, f, key)][0] += z; zm[(il, f, key)][1] += len(dec); sw[(il, f, key)] += s
        R['per_layer'][il] = PL
        print(f'layer {il}: eff experts {PL["decode_eff_experts"]}, top-50% share {PL["decode_top_share"][0.5]}', flush=True)

    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # (f, pol) -> hits, total, zero-miss, token-layers, swaps
    for (il, f, pol), (h, n) in acc.items():
        a = agg[(f, pol)]; a[0] += h; a[1] += n; a[2] += zm[(il, f, pol)][0]; a[3] += zm[(il, f, pol)][1]; a[4] += sw[(il, f, pol)]
        R['per_layer'][il].setdefault('hit', {}).setdefault(str(f), {})[pol] = round(h / n, 4) if n else None
    for (f, pol), (h, n, z, tl, s_) in sorted(agg.items()):
        R['overall'].setdefault(str(f), {})[pol] = {'hit_rate': round(h / n, 4) if n else None,
                                                     'zero_miss_token_layers': round(z / tl, 4) if tl else None,
                                                     'swaps_per_token_layer': round(s_ / tl, 3) if tl else None}
    Path(out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R['overall'], indent=1))


if __name__ == '__main__':
    main()
