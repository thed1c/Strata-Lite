#!/usr/bin/env python3
"""Strata-Lite's part of the setup: Qwen3.6-35B-A3B and Gemma 4 26B-A4B on strata-lite-server (Linux and Windows,
NVIDIA RTX 30 / 40 / 50 cards). setup.py offers them in its "Which model?" question and starts an installed one like
its own models; this file does the rest with setup.py's helpers (output, questions, the PC checks, NVIDIA's CUDA
libraries from pip).

    python setup_lite.py --start lite-<model>.json     what run-<model>.bat / .sh run: the server, then the chat page

What an install does (each step is skipped when it is already done):
  1. asks which files (the uncensored release every number was measured with, the official quantization - experimental,
     not measured with this engine - or GGUF files you already have), the context, images, and an optional Hugging Face
     token (none of the files needs one)
  2. installs NVIDIA's CUDA libraries into .venv (pip)
  3. gets the engine: the ready-made strata-lite-server from the release (or, with --build, compiles lite/CMakeLists.txt)
  4. downloads the model files, 8 connections per file (resumable; aria2c when it is installed), and checks them
  5. Qwen with HauhauCS's file: reads the MTP head out of unsloth's release with range requests (0.5 GB, not 23 GB)
  6. writes lite-<model>.json and run-<model>.bat / .sh, and starts the model
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

import setup as S

ROOT = S.ROOT
WIN = S.WIN
# the published fork whose releases carry the ready-made engine (owner/repo; STRATA_LITE_REPO overrides it)
LITE_REPO = os.environ.get("STRATA_LITE_REPO", "thed1c/Strata-Lite")
LITE_ASSET = "strata-lite-windows-x64.zip" if WIN else "strata-lite-linux-x64.zip"
LITE_DIR = ROOT / "engine-lite"
LITE_EXE = "strata-lite-server.exe" if WIN else "strata-lite-server"
LITE_MIN_ENGINE = (0, 2, 0)
RAM_FLOOR_GIB = 15          # below this neither model is worth starting (low-RAM mode pages most of the model from disk)
LITE_ARCHS = ("86", "87", "89", "120")       # what the engine is built for: RTX 30, RTX 40, RTX 50
CONTEXTS = [32768, 131072, 262144]
CONNECTIONS = 8

HF_REVISIONS = {
    "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive": "f12a584fecbeb5f20001130d8ecd66c9327ae685",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
    "HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP": "f9093662a2e7ae0503f637088bc96f77a1a70c83",
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF": "7b92b5b28818151e8669af2e45e88d6086f490dd",
}
# SHA-256 of the files the engine was measured with (the rest: size checks only)
SHA256 = {
    "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf": "8d344a4336d8ea7da0cbfc12792d1471e568be7abe8930c52260698bfd01d731",
    "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf": "c8e702344a81f8c226a914aa980ed6e1f604bce9374f1fed8e65c896908af414",
}

FAMILIES = {
    "qwen36": {
        "title": "Qwen3.6-35B-A3B", "by": "Qwen, on the Strata-Lite engine",
        "about": "35B MoE (3B active), thinks, reads images: a 12 GB card and 32 GB of RAM (16 GB runs, slower)",
        "arch": "qwen35moe", "name": "qwen3.6-35b",
        # GiB as the PC reports it (a "32 GB" PC: ~31): enough to lock the experts (20 GiB) beside the OS and a browser
        "locked_gib": 30,
        "variants": {
            "uncensored": {"about": "HauhauCS Uncensored Aggressive, Q4_K_P (23 GB) - every number in lite/ was measured with it",
                           "repo": "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
                           "model": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
                           "mmproj": "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf",
                           # HauhauCS's file has no MTP head: it is read out of unsloth's release
                           "mtp_extract": ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
                           "mtp": "mtp-qwen3.6-35b-a3b-head.gguf"},
            "official": {"about": "unsloth UD-Q4_K_XL (23 GB), EXPERIMENTAL: not measured with this engine",
                         "repo": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "model": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                         "mmproj": "mmproj-F16.gguf", "mtp": "@model", "experimental": True},
        }},
    "gemma4": {
        "title": "Gemma 4 26B-A4B", "by": "Google, on the Strata-Lite engine",
        "about": "26B MoE (4B active), thinks, reads images: a 12 GB card and 24 GB of RAM (16 GB runs, slower)",
        "arch": "gemma4", "name": "gemma-4-26b",
        "locked_gib": 22,       # the experts: 14 GiB
        "variants": {
            "uncensored": {"about": "HauhauCS QAT Uncensored Balanced, Q4_K_M (17 GB) - every number in lite/ was measured with it",
                           "repo": "HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP",
                           "model": "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
                           "mmproj": "mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
                           "mtp": "mtp-gemma-4-26B-A4B-it.gguf"},
            "official": {"about": "unsloth QAT UD-Q4_K_XL (14 GB), EXPERIMENTAL: not measured with this engine",
                         "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "model": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                         "mmproj": "mmproj-BF16.gguf", "mtp": "mtp-gemma-4-26B-A4B-it.gguf", "experimental": True},
        }},
}


# ------------------------------------------------------------------------------------------------ Hugging Face
def hf_url(repo: str, path: str) -> str:
    return f"{S.hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/{path}"


def hf_token() -> str | None:
    """HF_TOKEN, else the token `hf auth login` saved. Never stored by this setup."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    f = home / "token"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


class _HFAuth(urllib.request.BaseHandler):
    """Sends the token to huggingface.co itself only, never on to the download servers it redirects to."""
    def __init__(self, token):
        self.token = token

    def http_request(self, req):
        if req.host.split(":")[0] in ("huggingface.co", "hf.co") and not req.has_header("Authorization"):
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return req
    https_request = http_request


def use_token(token: str | None) -> None:
    if token:
        urllib.request.install_opener(urllib.request.build_opener(_HFAuth(token)))


_ASKED = []


def ask_token(code: int) -> bool:
    """Hugging Face refused a download (busy or limited): a free account's token usually lifts that. Asked once."""
    if _ASKED or not sys.stdin.isatty():
        return False
    _ASKED.append(True)
    S.warn(f"Hugging Face refused the download (HTTP {code}): it limits downloads without an account at busy times.")
    S.say("  A free Hugging Face account's token fixes this: sign in at https://huggingface.co, open Settings > Access\n"
          "  Tokens, create a token with the 'Read' role and paste it here. It is used for this download only, not saved.")
    try:
        t = input("  Token (Enter to stop and try again later): ").strip()
    except EOFError:
        return False
    if not t:
        return False
    use_token(t)
    return True


# ------------------------------------------------------------------------------------------------ downloads
def _final_url_and_size(url: str) -> tuple[str, int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-lite-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl(), int(r.headers.get("Content-Length", 0))


def _fetch_range(url: str, path: Path, start: int, end: int, progress: list, i: int) -> None:
    """bytes [start, end] into `path` (a .part<i> file), resuming what it already holds."""
    for attempt in range(30):
        have = path.stat().st_size if path.exists() else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-lite-setup", "Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                if r.status != 206:
                    raise OSError("the server ignored the range")
                while True:
                    b = r.read(4 << 20)
                    if not b:
                        break
                    f.write(b)
                    progress[i] += len(b)
            if path.stat().st_size >= end - start + 1:
                return
        except (OSError, urllib.error.URLError):
            time.sleep(min(30, 2 * (attempt + 1)))
    raise OSError(f"bytes {start}-{end} did not arrive")


def download(url: str, dst: Path, what: str, sha256: str | None = None) -> None:
    """A model file over CONNECTIONS parallel range requests (aria2c when it is on the PATH), resumable; a finished
    file gets setup.py's .done mark and is checked against its SHA-256 when one is known."""
    if dst.exists() and S.done(dst):
        S.ok(f"{what} already downloaded")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        final, total = _final_url_and_size(url)
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403, 429) or not ask_token(e.code):
            raise
        final, total = _final_url_and_size(url)
    if total and S.free_gb(dst.parent) < total / 1e9 + 1:
        S.fail(f"not enough disk space for {what} ({total / 1e9:.1f} GB) in {dst.parent}")
    aria = shutil.which("aria2c")
    if aria and not hf_token():
        S.say(f"  Downloading {what} with aria2c ({total / 1e9:.1f} GB) ...")
        S.run([aria, "-x", str(CONNECTIONS), "-s", str(CONNECTIONS), "-c", "--console-log-level=warn",
               "--summary-interval=0", "-d", str(dst.parent), "-o", dst.name, url])
    elif total < (64 << 20):
        S.download(url, dst, what)
        return
    else:
        S.say(f"  Downloading {what} ({total / 1e9:.1f} GB, {CONNECTIONS} connections) ...")
        step = -(-total // CONNECTIONS)
        parts = [(i, i * step, min(total, (i + 1) * step) - 1) for i in range(CONNECTIONS) if i * step < total]
        paths = [dst.with_name(f"{dst.name}.part{i}") for i, _, _ in parts]
        progress = [p.stat().st_size if p.exists() else 0 for p in paths]
        errors = []

        def worker(k):
            i, a, b = parts[k]
            try:
                _fetch_range(url, paths[k], a, b, progress, k)   # the resolve URL: its signed redirect expires
            except Exception as e:      # noqa: BLE001 - reported below
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(len(parts))]
        for t in threads:
            t.start()
        t0, done0 = time.time(), sum(progress)
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            have = sum(progress)
            rate = (have - done0) / max(1e-3, time.time() - t0) / 1e6
            print(f"\r  {what}: {have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%), {rate:.0f} MB/s   ",
                  end="", flush=True)
        print()
        if errors:
            S.fail(f"{what}: the download stopped ({errors[0]})", "run it again: it continues where it stopped")
        with open(dst, "wb") as out:
            for p in paths:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 64 << 20)
        for p in paths:
            p.unlink()
    if total and dst.stat().st_size != total:
        dst.unlink()
        S.fail(f"{what}: {dst.stat().st_size if dst.exists() else 0} bytes instead of {total}", "run it again")
    if sha256:
        S.say(f"  Checking {what} ...")
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for b in iter(lambda: f.read(64 << 20), b""):
                h.update(b)
        if h.hexdigest() != sha256:
            dst.unlink()
            S.fail(f"{what} does not match its SHA-256 (a broken download): removed", "run it again")
    S.mark(dst)
    S.ok(f"{what} downloaded")


# ------------------------------------------------------------------------------------------------ the engine
def engine_version(eng: Path) -> tuple:
    try:
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
        return tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3])
    except (OSError, ValueError):
        return (0,)


def get_engine(build: bool) -> Path:
    exe = LITE_DIR / LITE_EXE
    if exe.exists() and engine_version(LITE_DIR) >= LITE_MIN_ENGINE and not build:
        S.ok("the Strata-Lite engine is installed")
        return exe
    if not build:
        # STRATA_LITE_RELEASE: another place with the release's files (a URL folder or a local folder; tests, mirrors)
        base = os.environ.get("STRATA_LITE_RELEASE") or f"https://github.com/{LITE_REPO}/releases/latest/download/"
        base = base if base.endswith(("/", "\\")) else base + "/"
        z = LITE_DIR / LITE_ASSET
        try:
            LITE_DIR.mkdir(exist_ok=True)
            if z.exists():
                z.unlink()
            S.download(base + LITE_ASSET, z, "the ready-made Strata-Lite engine")
            sums = (urllib.request.urlopen(base + "SHA256SUMS.txt", timeout=60).read().decode()
                    if base.startswith(("http://", "https://")) else Path(base, "SHA256SUMS.txt").read_text())
            want = next((l.split()[0] for l in sums.splitlines() if l.strip().endswith(LITE_ASSET)), None)
            got = hashlib.sha256(z.read_bytes()).hexdigest()
            if want != got:
                S.fail("the downloaded engine does not match the release's SHA-256", "run it again")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(LITE_DIR)
            z.unlink()
            if not WIN:
                exe.chmod(0o755)
            S.ok(f"the ready-made Strata-Lite engine {'.'.join(map(str, engine_version(LITE_DIR)))}")
            return exe
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as e:
            S.warn(f"no ready-made engine ({e}): compiling it instead")
    return build_engine()


def build_engine() -> Path:
    """lite/CMakeLists.txt (it fetches llama.cpp b6b003d itself). Needs the CUDA toolkit; on Windows also Visual
    Studio's C++ build tools (setup.py offers to install both)."""
    nvcc = S.find_nvcc()
    if not nvcc:
        S.fail("compiling the engine needs NVIDIA's CUDA toolkit (nvcc)", "install CUDA 13.0 and run it again, "
               "or use the ready-made engine (run without --build)")
    bdir = ROOT / "build-lite"
    S.say("  Compiling Strata-Lite (20-60 minutes) ...")
    S.cmake_build(ROOT / "lite", bdir, "strata-lite-server", ["-DSTRATA_LITE_PORTABLE=OFF", f"-DCMAKE_CUDA_COMPILER={nvcc}"],
                  S.find_vcvars() if WIN else None, "build-lite.bat")
    LITE_DIR.mkdir(exist_ok=True)
    shutil.copy2(bdir / "bin" / LITE_EXE, LITE_DIR / LITE_EXE)
    (LITE_DIR / "BUILD.json").write_text(json.dumps({"version": ".".join(map(str, LITE_MIN_ENGINE)), "source": "local"}))
    S.ok("Strata-Lite compiled")
    return LITE_DIR / LITE_EXE


# ------------------------------------------------------------------------------------------------ GGUF
def gguf_arch(path: Path) -> str:
    """general.architecture from a GGUF's header ("" when it is not one)."""
    import struct
    try:
        with open(path, "rb") as f:
            magic, ver, _, nkv = struct.unpack("<IIQQ", f.read(24))
            if magic != 0x46554747:
                return ""
            for _ in range(min(nkv, 64)):
                (n,) = struct.unpack("<Q", f.read(8)); key = f.read(n).decode(errors="replace")
                (t,) = struct.unpack("<I", f.read(4))
                if t != 8:
                    return ""   # general.architecture is the first key in every GGUF llama.cpp writes
                (n,) = struct.unpack("<Q", f.read(8)); val = f.read(n).decode(errors="replace")
                if key == "general.architecture":
                    return val
    except (OSError, struct.error):
        pass
    return ""


# ------------------------------------------------------------------------------------------------ install
def configs() -> list[Path]:
    return sorted(ROOT.glob("lite-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def recommend(gpu: dict | None, ram: float, hip: bool) -> str | None:
    """The Strata-Lite model for this PC: Qwen3.6 with 48 GB of RAM or more, else Gemma 4; none on a card it cannot run."""
    if hip or not gpu or str(gpu.get("arch")) not in LITE_ARCHS:
        return None
    return "qwen36" if ram >= FAMILIES["qwen36"]["locked_gib"] else "gemma4"


def menu_lines(start: int, rec: str | None = None) -> list[str]:
    return [f"  {start + i}) {d['title']:20s} {d['by']} - {d['about']}" + ("   (recommended for this PC)" if f == rec else "")
            for i, (f, d) in enumerate(FAMILIES.items())]


def install(a, family: str, gpu: dict, ram: float, models_dir: Path) -> int:
    fam = FAMILIES[family]
    S.ok(f"model: {fam['title']} (Strata-Lite)")
    # ---- the card and the PC
    if str(gpu.get("arch")) not in LITE_ARCHS:
        S.fail(f"{gpu.get('name')} (sm_{gpu.get('arch')}) is not an RTX 30, 40 or 50 card: the Strata-Lite engine is built for those",
               "choose a Strata model (Qwen3.8-Flash-Next), which runs on more cards")
    if gpu.get("vram_gb", 0) < 11:
        S.warn(f"{gpu['name']} has {gpu['vram_gb']:.0f} GB: Strata-Lite is measured on 12 GB cards; fewer experts fit on yours")
    if ram < RAM_FLOOR_GIB:
        S.confirm_risk(f"RAM: {ram:.0f} GB - {fam['title']} needs at least 16 GB; with less, most of the model is read from the disk "
                       "for every answer", False, a.yes, f"{fam['title']} needs at least 16 GB of RAM", "choose a PC with more RAM")
    # low-RAM mode (Strata's file tier, simply): the experts stay in the model file and are read in as needed instead of
    # being locked in RAM; the engine also falls back to it by itself if the OS will not lock them
    low_ram = ram < fam["locked_gib"]
    if low_ram:
        S.warn(f"RAM: {ram:.0f} GB - not enough to keep {fam['title']}'s experts locked in RAM (that needs about "
               f"{fam['locked_gib'] + 2} GB). It runs in low-RAM mode: the experts are read from the model file as needed, so "
               "it writes about half as fast and long prompts take longer. Use an SSD for the model files.")
    # ---- the questions: a first install takes the recommended answers (the measured files, the context the card holds,
    # images on); `setup --setup` (SETUP.bat) asks them
    simple = not a.setup
    vs = list(fam["variants"])
    if simple:
        pick = 1
    else:
        S.say("  Which files?")
        for i, v in enumerate(vs, 1):
            S.say(f"  {i}) {v:11s} {fam['variants'][v]['about']}")
        S.say(f"  {len(vs) + 1}) your own   GGUF files you already have (architecture {fam['arch']})")
        pick = int(S.ask("Which files?", [str(i) for i in range(1, len(vs) + 2)], "1", a.yes))
    own = pick == len(vs) + 1
    var = None if own else fam["variants"][vs[pick - 1]]
    if var and var.get("experimental"):
        S.warn("experimental: these files have not been measured with the Strata-Lite engine; if it refuses them, "
               "it says why (an unsupported quantization type)")
    rec_ctx = 262144 if gpu.get("vram_gb", 0) >= 11.5 else 131072
    ctx = a.context if a.context in CONTEXTS else (rec_ctx if simple else None)
    if ctx is None:
        for i, c in enumerate(CONTEXTS, 1):
            S.say(f"  {i}) {c // 1024}K tokens" + ("  (recommended)" if c == rec_ctx else ""))
        ctx = CONTEXTS[int(S.ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)], str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    vision = a.vision not in ("no", "none") if a.vision else (True if simple else S.ask(
        "Do you want images? (the image encoder is loaded only when an image comes)", ["y", "n"], "y", a.yes) == "y")
    if simple:
        S.ok(f"files: {vs[0]} ({fam['variants'][vs[0]]['about'].split(' - ')[0]}); context {ctx // 1024}K; images on "
             "(other choices: SETUP.bat / ./setup.sh --setup)")
    port = a.port or 8080
    # ---- files
    if own:
        mpath = Path(input("  The model GGUF (full path): ").strip().strip('"'))
        if gguf_arch(mpath) != fam["arch"]:
            S.fail(f"{mpath} is not a {fam['arch']} GGUF")
        mtp = input("  The MTP drafter GGUF (Enter: none, or the model itself if it carries an MTP head): ").strip().strip('"')
        mm = input("  The image projector (mmproj) GGUF (Enter: none): ").strip().strip('"') if vision else ""
        files = {"model": mpath, "mtp": Path(mtp) if mtp else None, "mmproj": Path(mm) if mm else None}
    else:
        use_token(hf_token())   # none of these files needs one; asked only if Hugging Face refuses a download
        mdir = models_dir / fam["name"]
        files = {"model": mdir / var["model"], "mtp": None, "mmproj": None}
        download(hf_url(var["repo"], var["model"]), files["model"], var["model"], SHA256.get(var["model"]))
        if vision and var.get("mmproj"):
            files["mmproj"] = mdir / var["mmproj"]
            download(hf_url(var["repo"], var["mmproj"]), files["mmproj"], var["mmproj"], SHA256.get(var["mmproj"]))
        if var.get("mtp") == "@model":
            files["mtp"] = files["model"]
        elif var.get("mtp_extract"):
            files["mtp"] = mdir / var["mtp"]
            if not (files["mtp"].exists() and S.done(files["mtp"])):
                repo, src = var["mtp_extract"]
                S.say(f"  Reading the MTP head out of {repo} (0.5 GB of the 23 GB file) ...")
                S.run([sys.executable, str(ROOT / "tools" / "qwen35_mtp_extract.py"), "--no-head", hf_url(repo, src), str(files["mtp"])])
                S.mark(files["mtp"])
        elif var.get("mtp"):
            files["mtp"] = mdir / var["mtp"]
            download(hf_url(var["repo"], var["mtp"]), files["mtp"], var["mtp"])
    # ---- NVIDIA's libraries and the engine
    S.pip_install(S.CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")
    exe = get_engine(getattr(a, "build", False))
    # ---- the config, the run script, the start
    args = ["-m", str(files["model"]), "-c", str(ctx), "--host", a.host or "127.0.0.1", "--port", str(port),
            "--alias", fam["name"]]
    if files["mtp"]:
        args += ["--mtp", str(files["mtp"])]
    if files["mmproj"]:
        args += ["--mmproj", str(files["mmproj"])]
    if low_ram:
        args += ["--no-async-swaps"]
    cfg = {"engine": "strata-lite", "model_name": f"{fam['title']} (Strata-Lite, {'own files' if own else vs[pick - 1]})",
           "family": family, "exe": str(exe), "args": args, "port": port, "context": ctx}
    cfg_path = ROOT / f"lite-{fam['name']}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    script = write_run_script(fam["name"], cfg_path)
    S.ok(f"installed: start it again with {script.name}")
    if a.host and a.host != "127.0.0.1":
        S.warn("the Strata-Lite server has no API key: on a network address anyone who can reach it can use it")
    return 0 if a.no_start else start(cfg_path)


def write_run_script(name: str, cfg_path: Path) -> Path:
    cmd = [sys.executable, str(ROOT / "setup_lite.py"), "--start", str(cfg_path)]
    if WIN:
        script = ROOT / f"run-{name}.bat"
        script.write_text("@echo off\r\ntitle Strata-Lite " + name + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in cmd) + "\r\nif errorlevel 1 pause\r\n", encoding="utf-8")
    else:
        script = ROOT / f"run-{name}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in cmd) + "\n", encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ start
def start(cfg_path: Path, port: int | None = None, open_browser: bool = True) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    args = list(cfg["args"])
    if port:
        args[args.index("--port") + 1] = str(port)
    port = int(args[args.index("--port") + 1])
    env = dict(os.environ)
    libs = S.cuda_lib_dirs(13)
    key = "PATH" if WIN else "LD_LIBRARY_PATH"
    env[key] = os.pathsep.join(libs + [env.get(key, "")]).rstrip(os.pathsep)
    S.say(f"\nStarting {cfg['model_name']} on http://127.0.0.1:{port}/ (loading takes a minute or two) ...")
    p = subprocess.Popen([cfg["exe"], *args], env=env)

    def watch():
        url = f"http://127.0.0.1:{port}/health"
        while p.poll() is None:
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    S.ok(f"ready: the chat page is http://127.0.0.1:{port}/ and the API http://127.0.0.1:{port}/v1")
                    if open_browser:
                        webbrowser.open(f"http://127.0.0.1:{port}/")
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
    threading.Thread(target=watch, daemon=True).start()
    try:
        return p.wait()
    except KeyboardInterrupt:
        p.terminate()
        try:
            return p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
            return 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--start":
        sys.exit(start(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None))
    print(__doc__)
    sys.exit(2)
