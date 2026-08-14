"""Deprecated single-process entry. Nav is the 3-node Dora graph."""

from __future__ import annotations

from typing import Optional


def main(args: Optional[list[str]] = None) -> None:
    raise SystemExit(
        "hound_nav is a 3-node Dora graph (manager / planner / controller). "
        "Enable nav in SSoT and run: ros2 launch hound_core hound_core.launch.py"
    )


if __name__ == "__main__":
    main()
