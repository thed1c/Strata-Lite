#!/usr/bin/env python3
"""Extract a qwen35/qwen35moe MTP head into an MTP-only GGUF (a drafter file for `-md` / `--spec-type draft-mtp`).

The MTP head of Qwen3.5/3.6 (`nextn_predict_layers = 1`) is the last block of a GGUF that carries it: a full attention +
MoE layer plus `nextn.eh_proj / enorm / hnorm / shared_head_norm`, using the model's token embedding and output head.
llama.cpp loads a GGUF without `blk.0` but with the nextn layer as an MTP-only drafter (`src/models/qwen35moe.cpp`,
`mtp_only`). This copies, byte for byte and without requantising: every metadata key (architecture, hyperparameters,
tokenizer), `token_embd`, `output_norm`, `output` (when present) and every `blk.<last>.*` tensor.

Pure Python (no numpy). Usage: qwen35_mtp_extract.py SOURCE.gguf OUT.gguf
"""
import pathlib
import re
import struct
import sys

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
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    with src.open("rb") as fh:
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
        file_end = src.stat().st_size

    arch = meta["general.architecture"]
    n_layer = meta[f"{arch}.block_count"]
    n_nextn = meta.get(f"{arch}.nextn_predict_layers", 0)
    if not n_nextn:
        sys.exit(f"{src.name}: no MTP head ({arch}.nextn_predict_layers is 0 or missing)")
    mtp_blocks = set(range(n_layer - n_nextn, n_layer))
    # tensor byte sizes from the offsets (sorted), the last one to the end of the file
    by_off = sorted(tensors, key=lambda t: t[3])
    size = {}
    for a, b in zip(by_off, by_off[1:] + [None]):
        size[a[0]] = (b[3] if b else file_end - data_start) - a[3]

    def keep(name):
        if name in ("token_embd.weight", "output_norm.weight", "output.weight"):
            return True
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

    with dst.open("wb") as fo, src.open("rb") as fi:
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
            fi.seek(data_start + soff)
            left = size[name]
            while left:
                chunk = fi.read(min(left, 64 << 20)); fo.write(chunk); left -= len(chunk)
    total = sum(size[t[0]] for t in out)
    print(f"{dst.name}: {len(out)} tensors ({total / 1e9:.2f} GB) from {src.name}: MTP block(s) {sorted(mtp_blocks)} + "
          f"{', '.join(t[0] for t in out if not t[0].startswith('blk.'))}; {len(kv_raw)} metadata keys copied")


if __name__ == "__main__":
    main()
