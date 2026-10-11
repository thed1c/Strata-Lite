#!/usr/bin/env python3
"""Strata-Lite: the sameness turns (an agent tool turn and a shot-notes turn) ("one real agent turn and one shot-notes
turn through it by hand (directly), judged for sameness against the reference server's answers on the same inputs").

  sameness.py record BASE LABEL DIR req.json [req.json ...]   send each request (as captured, greedy, seed 42) and keep
                                                               the answer in DIR/<req>.<LABEL>.json (text stays on the host)
  sameness.py compare DIR REF TEST OUT.json                    numbers for the repo: identical or not, the common prefix,
                                                               a similarity ratio, lengths, finish reasons, tool calls
The maintainer judges sameness by reading DIR's two answers side by side; the numbers only say where they part.
"""
import difflib, json, os, sys, time, urllib.request, urllib.error


def send(base, body):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r: return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e), time.time() - t0
        except Exception: return e.code, {}, time.time() - t0


def record(base, label, d, reqs):
    os.makedirs(d, exist_ok=True)
    for p in reqs:
        body = json.load(open(p))
        body["model"] = "reference"; body["temperature"] = 0.0; body["seed"] = 42; body.pop("stream", None)
        body["max_tokens"] = min(int(body.get("max_tokens") or 1024), 1024)
        code, js, dt = send(base, body)
        name = os.path.basename(p)[:-5]
        json.dump({"code": code, "wall_s": round(dt, 2), "reply": js}, open(f"{d}/{name}.{label}.json", "w"), indent=1)
        print(f"{label} {name}: HTTP {code} in {dt:.1f} s", flush=True)


def msg(rec):
    ch = ((rec.get("reply") or {}).get("choices") or [{}])[0]
    m = ch.get("message") or {}
    calls = [(c.get("function") or {}).get("name") for c in (m.get("tool_calls") or [])]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), calls, ch.get("finish_reason")


def compare(d, ref, test, out):
    res = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(f".{ref}.json"): continue
        name = f[: -len(f".{ref}.json")]
        a = json.load(open(f"{d}/{f}")); b = json.load(open(f"{d}/{name}.{test}.json"))
        ca, ra, ta, fa = msg(a); cb, rb, tb, fb = msg(b)
        pre = os.path.commonprefix([ca, cb])
        res[name] = {"identical": ca == cb and ta == tb, "common_prefix_chars": len(pre), "chars": [len(ca), len(cb)],
                     "similarity": round(difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio(), 3),
                     "reasoning_chars": [len(ra), len(rb)], "tool_calls_same": ta == tb, "n_tool_calls": [len(ta), len(tb)],
                     "finish": [fa, fb], "wall_s": [a.get("wall_s"), b.get("wall_s")], "http": [a.get("code"), b.get("code")]}
        print(name, json.dumps(res[name]))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "record": record(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif sys.argv[1] == "compare": compare(*sys.argv[2:6])
    else: print(__doc__)
