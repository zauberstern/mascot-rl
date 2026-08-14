#!/usr/bin/env python3
"""Build Layer-1 (cpp_rbergomi) and Layer-2 (polaris_pricer_cpp) extensions in-place."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("MASCOTRL_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ENGINE_DIR = ROOT / "src" / "mascotrl" / "engine"
PRICING_DIR = ROOT / "src" / "mascotrl" / "pricing"


def _march_flag() -> str:
    native = os.environ.get("MASCOTRL_NATIVE", "").strip().lower() in ("1", "true", "yes")
    if native:
        return "-march=native"
    return "-march=x86-64-v2"


CXX_FLAGS = [
    "-O3",
    _march_flag(),
    "-mavx2",
    "-mfma",
    "-fopenmp",
    "-ffast-math",
    "-std=c++20",
    "-fPIC",
]
LINK_FLAGS = ["-fopenmp"]
include_dirs = [
    str(ENGINE_DIR),
    str(ROOT / "third_party"),
    str(ROOT / "third_party" / "OpenCL"),
    pybind11.get_include(),
]
try:
    import pyarrow as _pa

    _pa_inc = str(Path(_pa.__file__).resolve().parent / "include")
    if Path(_pa_inc, "arrow", "c", "abi.h").is_file():
        include_dirs.append(_pa_inc)
except ImportError:
    pass

ext_rbergomi = Pybind11Extension(
    "cpp_rbergomi",
    sources=[
        "src/mascotrl/engine/bindings.cpp",
        "src/mascotrl/engine/rbergomi_engine.cpp",
        "src/mascotrl/engine/circulant_fft.cpp",
        "src/mascotrl/engine/dupire_pde.cpp",
        "src/mascotrl/engine/bs_iv.cpp",
        "src/mascotrl/engine/heston_cf.cpp",
        "src/mascotrl/engine/sabr_hagan.cpp",
        "src/mascotrl/engine/garch_duan.cpp",
        "src/mascotrl/engine/worlds.cpp",
    ],
    include_dirs=include_dirs,
    cxx_std=20,
    extra_compile_args=CXX_FLAGS,
    extra_link_args=LINK_FLAGS,
)


def main() -> None:
    polaris_only = "--polaris-only" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--polaris-only"]
    if argv:
        # Allow `python setup_extensions.py --polaris-only` without setuptools
        # treating unknown args as setup script args.
        sys.argv = [sys.argv[0], *argv]

    if not polaris_only:
        setup(
            name="volsurf_extensions",
            ext_modules=[ext_rbergomi],
            cmdclass={"build_ext": build_ext},
            zip_safe=False,
            script_args=["build_ext", "--inplace"],
        )
        # In-place build lands under src/mascotrl/; copy to repo root for import.
        import glob

        for path in glob.glob(str(ROOT / "src" / "mascotrl" / "cpp_rbergomi*.so")):
            dest = ROOT / Path(path).name
            shutil.copy2(path, dest)
            print(f"[mascotrl] cpp_rbergomi -> {dest}")

    # Layer 2: torch cpp extension
    import torch
    from torch.utils.cpp_extension import load

    polaris_dir = ROOT / "build" / "polaris"
    polaris_dir.mkdir(parents=True, exist_ok=True)
    load(
        name="polaris_pricer_cpp",
        sources=[str(PRICING_DIR / "polaris_pricer.cpp")],
        extra_cflags=CXX_FLAGS,
        extra_ldflags=LINK_FLAGS + ["-L/usr/lib/x86_64-linux-gnu", "-l:libOpenCL.so.1"],
        extra_include_paths=include_dirs + [str(PRICING_DIR)],
        build_directory=str(polaris_dir),
        verbose=True,
    )
    print("[mascotrl] polaris_pricer_cpp loaded/built")


if __name__ == "__main__":
    main()
