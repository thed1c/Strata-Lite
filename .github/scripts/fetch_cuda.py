#!/usr/bin/env python3
"""CI: NVIDIA's CUDA toolkit components for building Strata-Lite, from NVIDIA's redistributable archives (no installer,
no GPU needed), unpacked into one folder. Prints nothing secret; writes CUDA_PATH / PATH for the next GitHub steps.

    fetch_cuda.py VERSION DEST        e.g. fetch_cuda.py 13.0.2 $RUNNER_TEMP/cuda
"""
import json, os, platform, shutil, sys, tarfile, urllib.request, zipfile
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
PARTS = ["cuda_crt", "cuda_cudart", "cuda_nvcc", "cuda_nvrtc", "libcublas", "libnvvm", "cuda_nvtx", "cuda_profiler_api", "cuda_cccl"]
WIN = platform.system() == "Windows"
if WIN:
    PARTS.append("visual_studio_integration")
plat = "windows-x86_64" if WIN else "linux-x86_64"

ver, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
manifest = json.load(urllib.request.urlopen(BASE + f"redistrib_{ver}.json", timeout=120))
tmp = dest.parent / "cuda-archives"
tmp.mkdir(exist_ok=True)
for part in PARTS:
    rel = manifest[part][plat]["relative_path"]
    arc = tmp / Path(rel).name
    print(f"{part} {manifest[part]['version']}", flush=True)
    with urllib.request.urlopen(BASE + rel, timeout=600) as r, open(arc, "wb") as f:
        shutil.copyfileobj(r, f, 16 << 20)
    unpack = tmp / (part + "-x")
    if arc.suffix == ".zip":
        zipfile.ZipFile(arc).extractall(unpack)
    else:
        tarfile.open(arc).extractall(unpack)
    top = next(unpack.iterdir())                     # <name>-<plat>-<ver>-archive/
    shutil.copytree(top, dest, dirs_exist_ok=True)
    shutil.rmtree(unpack); arc.unlink()
if os.environ.get("GITHUB_ENV"):
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"CUDA_PATH={dest}\n")
    with open(os.environ["GITHUB_PATH"], "a") as f:
        f.write(str(dest / "bin") + "\n")
print("CUDA", ver, "in", dest)
