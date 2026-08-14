#!/usr/bin/env python3
"""CLI wrapper — see ``hound_nav.jit_build``."""

from __future__ import annotations

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from hound_nav.jit_build import main

if __name__ == "__main__":
    raise SystemExit(main())
