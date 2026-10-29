#!/usr/bin/env python3
"""Plot the latest nav drive log: command, /odom, and the cheapest rollout.

Writes plots/nav_drive.png. Pass a csv path to plot some other hold.

  python3 /root/colcon_ws/src/hound_nav/scripts/plot_nav_drive.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "logs" / "latest.csv"


def load(path: Path):
    base, horizons = [], []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        p = [float(x) for x in line.split(",")]
        base.append(p[:7])
        horizon = None
        if len(p) >= 9 and p[8] >= 1:
            steps = int(p[8])
            vals = p[9 : 9 + 3 * steps]
            if len(vals) == 3 * steps and np.isfinite(p[7]):
                horizon = (p[0], p[7], np.asarray(vals, dtype=float).reshape(steps, 3))
        horizons.append(horizon)
    a = np.asarray(base, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a, horizons


a, horizons = load(src)
t = a[:, 0]
# One forecast per second, from the cheapest rollout. Vertex i is timestep i.
shown = []
last = -1e9
for item in horizons:
    if item is None or item[0] - last < 1.0:
        continue
    last = item[0]
    shown.append(item)
pairs = (("vx", 1, 4, 0), ("vy", 2, 5, 1), ("wz", 3, 6, 2))
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
for ax, (name, c, o, col) in zip(axes, pairs):
    ax.plot(t, a[:, o], label="odom", color="k", lw=1.2)
    ax.plot(t, a[:, c], label="command", color="0.6", lw=1.0)
    first = True
    for t0, hdt, h in shown:
        ax.plot(
            t0 + np.arange(len(h)) * hdt,
            h[:, col],
            color=(1.0, 0.45, 0.05),
            alpha=0.75,
            lw=1.0,
            label="best rollout" if first else None,
        )
        first = False
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{name}  cmd [{a[:, c].min():+.2f}, {a[:, c].max():+.2f}]"
        f"  odom [{a[:, o].min():+.2f}, {a[:, o].max():+.2f}]"
    )
axes[0].legend(loc="upper right")
axes[2].set_xlabel("t (s)")
fig.tight_layout()
dest = root / "plots" / "nav_drive.png"
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest, dpi=120)
print(f"wrote {dest}  ({len(a)} rows, {t[-1]:.1f} s, from {src.name})")
