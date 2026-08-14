"""Torch CUDA extension cache under ``hound_nav/cache/`` (gitignored).

Set ``TORCH_EXTENSIONS_DIR`` before any ``torch.utils.cpp_extension.load``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    # hound_nav/hound_nav/cuda_cache.py → hound_nav/
    return Path(__file__).resolve().parents[1]


def cache_root() -> Path:
    return repo_root() / "cache"


def torch_extensions_dir() -> Path:
    return cache_root() / "torch_extensions"


def ensure_torch_extensions_dir(*, clean: bool = False) -> Path:
    """Point PyTorch JIT builds at ``hound_nav/cache/torch_extensions``.

    Must run before any ``cpp_extension.load`` (dynamics / cost / IGHA*).
    """
    d = torch_extensions_dir()
    if clean and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    # Absolute path; torch uses this when build_directory is omitted.
    os.environ["TORCH_EXTENSIONS_DIR"] = str(d)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    return d
