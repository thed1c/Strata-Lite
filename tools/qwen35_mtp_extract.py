#!/usr/bin/env python3
"""Extract a qwen35/qwen35moe MTP head into an MTP-only GGUF (a drafter file for `-md` / `--spec-type draft-mtp`).

The MTP head of Qwen3.5/3.6 (`nextn_predict_layers = 1`) is the last block of a GGUF that carries it: a full attention +
MoE layer plus `nextn.eh_proj / enorm / hnorm / shared_head_norm`, using the model's token embedding and output head.
llama.cpp loads a GGUF without `blk.0` but with the nextn layer as an MTP-only drafter (`src/models/qwen35moe.cpp`,
`mtp_only`). This copies, byte for byte and without requantising: every metadata key (architecture, hyperparameters,
tokenizer), `token_embd`, `output_norm`, `output` (when present) and every `blk.<last>.*` tensor.

Pure Python (no numpy). Usage: qwen35_mtp_extract.py [--no-head] SOURCE OUT.gguf
SOURCE is a file or an https:// URL (a Hugging Face "resolve" link): from a URL only the header and the tensors kept are
read, with HTTP range requests (the MTP block is ~0.5 GB of a ~23 GB file). --no-head leaves out token_embd / output /
output_norm: Strata-Lite's server uses the target model's own (the default), llama.cpp's `-md` needs them.
"""
import io
import pathlib
import re
import struct
import sys
import time
import urllib.request


class Source:
    """A file or an HTTP(S) URL read by offset. URLs: HTTP range requests, retried; a header window is cached."""
    def __init__(self, src):
        self.url = src if str(src).startswith(("http://", "https://")) else None
        self.f = None if self.url else open(src, "rb")
        self.size = self._size()

    def _size(self):
        if not self.url:
            self.f.seek(0, 2); return self.f.tell()
        req = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": "strata-lite"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])

    def read(self, off, n):
        if not self.url:
            self.f.seek(off); return self.f.read(n)
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}", "User-Agent": "strata-lite"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == n: return data
            except Exception as e:
                err = e
            time.sleep(2 * (attempt + 1))
        sys.exit(f"cannot read {n} bytes at {off} from {self.url}")


class Cursor(io.RawIOBase):
    """A sequential reader over a Source for the header parse (reads ahead in 16 MB windows)."""
    def __init__(self, src):
        self.src, self.pos, self.buf, self.base = src, 0, b"", 0

    def read(self, n):
        if self.pos < self.base or self.pos + n > self.base + len(self.buf):
            self.base = self.pos
            self.buf = self.src.read(self.pos, min(max(n, 16 << 20), self.src.size - self.pos))
        out = self.buf[self.pos - self.base:self.pos - self.base + n]
        self.pos += n
        return out

    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off
        return self.pos

GGUF_MAGIC = 0x46554747
# value type ids (gguf.h) and their fixed sizes
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9


def read_str(fh):
    (n,) = struct.unpack("<Q", fh.read(8))
    return fh.read(n)


def skip_value(fh, vtype):
    """advance past a value of vtype; returns nothing (the caller slices the raw bytes)"""
    if vtype in FIXED:
        fh.read(FIXED[vtype])
    elif vtype == STRING:
        read_str(fh)
    elif vtype == ARRAY:
        etype, n = struct.unpack("<IQ", fh.read(12))
        if etype in FIXED:
            fh.read(FIXED[etype] * n)
        elif etype == STRING:
            for _ in range(n):
                read_str(fh)
        else:
            raise ValueError(f"array of type {etype} not handled")
    else:
        raise ValueError(f"value type {vtype} not handled")


def main():
    args = [a for a in sys.argv[1:] if a != "--no-head"]
    no_head = "--no-head" in sys.argv[1:]
    if len(args) != 2: sys.exit(__doc__)
    source, dst = Source(args[0]), pathlib.Path(args[1])
    src_name = args[0].rsplit("/", 1)[-1]
    fh = Cursor(source)
    if True:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fh.read(24))
        if magic != GGUF_MAGIC or version != 3:
            sys.exit(f"{src}: not a GGUF v3 file")
        kv_raw, meta = [], {}
        for _ in range(n_kv):
            start = fh.tell()
            key = read_str(fh).decode()
            (vtype,) = struct.unpack("<I", fh.read(4))
            vstart = fh.tell()
            skip_value(fh, vtype)
            end = fh.tell()
            fh.seek(start); raw = fh.read(end - start)
            kv_raw.append(raw)
            if vtype in (4, 5, 6) or vtype == 10:  # u32, i32, f32, u64: the few we read back
                fmt = {4: "<I", 5: "<i", 6: "<f", 10: "<Q"}[vtype]
                meta[key] = struct.unpack(fmt, raw[vstart - start:])[0]
            elif vtype == STRING:
                meta[key] = raw[vstart - start + 8:].decode(errors="replace")
        tensors = []
        for _ in range(n_tensors):
            name = read_str(fh).decode()
            (nd,) = struct.unpack("<I", fh.read(4))
            shape = struct.unpack(f"<{nd}Q", fh.read(8 * nd))
            ttype, off = struct.unpack("<IQ", fh.read(12))
            tensors.append([name, shape, ttype, off])
        align = meta.get("general.alignment", 32)
        data_start = (fh.tell() + align - 1) // align * align
        file_end = source.size

    arch = meta["general.architecture"]
    n_layer = meta[f"{arch}.block_count"]
    n_nextn = meta.get(f"{arch}.nextn_predict_layers", 0)
    if not n_nextn:
        sys.exit(f"{src_name}: no MTP head ({arch}.nextn_predict_layers is 0 or missing)")
    mtp_blocks = set(range(n_layer - n_nextn, n_layer))
    # tensor byte sizes from the offsets (sorted), the last one to the end of the file
    by_off = sorted(tensors, key=lambda t: t[3])
    size = {}
    for a, b in zip(by_off, by_off[1:] + [None]):
        size[a[0]] = (b[3] if b else file_end - data_start) - a[3]

    def keep(name):
        if name in ("token_embd.weight", "output_norm.weight", "output.weight"):
            return not no_head
        m = re.match(r"blk\.(\d+)\.", name)
        return bool(m) and int(m.group(1)) in mtp_blocks
    out = [t for t in tensors if keep(t[0])]
    if not any(".nextn." in t[0] for t in out):
        sys.exit("no nextn tensors found in the MTP block")

    def pad(n):
        return (n + align - 1) // align * align
    # new offsets, each tensor aligned
    off = 0
    for t in out:
        t.append(off)  # t[4] = new offset
        off = pad(off + size[t[0]])

    with dst.open("wb") as fo:
        fo.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(out), len(kv_raw)))
        for raw in kv_raw:
            fo.write(raw)
        for name, shape, ttype, _, noff in out:
            nb = name.encode()
            fo.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
                     + struct.pack("<IQ", ttype, noff))
        fo.write(b"\0" * (pad(fo.tell()) - fo.tell()))
        base = fo.tell()
        for name, shape, ttype, soff, noff in out:
            fo.write(b"\0" * (base + noff - fo.tell()))
            at, left = data_start + soff, size[name]
            while left:
                n = min(left, 64 << 20)
                chunk = source.read(at, n); fo.write(chunk); at += n; left -= n
    total = sum(size[t[0]] for t in out)
    print(f"{dst.name}: {len(out)} tensors ({total / 1e9:.2f} GB) from {src_name}: MTP block(s) {sorted(mtp_blocks)} + "
          f"{', '.join(t[0] for t in out if not t[0].startswith('blk.'))}; {len(kv_raw)} metadata keys copied")


if __name__ == "__main__":
    main()
