#!/usr/bin/env python3
"""Create a project-local CUDA toolkit shim for gsplat's JIT compiler.

This helper links the CUDA runtime and nvcc from the active Python environment. It does
not download or copy a toolkit and does not modify system CUDA paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = Path(sys.executable).resolve().parent.parent
PYTHON = f"python{sys.version_info.major}.{sys.version_info.minor}"
NVIDIA = PREFIX / "lib" / PYTHON / "site-packages/nvidia"
CUDA_RUNTIME = NVIDIA / "cuda_runtime"
TARGET = ROOT / ".cuda-toolkit"


def link(source: Path, destination: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def main():
    nvcc = PREFIX / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(
            f"nvcc is missing at {nvcc}. Install CUDA 12.1 nvcc into this environment first."
        )
    link(nvcc, TARGET / "bin/nvcc")
    link(PREFIX / "nvvm", TARGET / "nvvm")
    toolkit_include = PREFIX / "include" if (PREFIX / "include/crt/host_config.h").is_file() else CUDA_RUNTIME / "include"
    link(toolkit_include, TARGET / "include")
    runtime = PREFIX / "lib/libcudart.so.12"
    if not runtime.is_file():
        runtime = CUDA_RUNTIME / "lib/libcudart.so.12"
    link(runtime, TARGET / "lib64/libcudart.so.12")
    link(runtime, TARGET / "lib64/libcudart.so")
    print(f"CUDA_HOME={TARGET}")
    print(f"nvcc={nvcc}")
    print(f"runtime={runtime}")


if __name__ == "__main__":
    main()
