#!/usr/bin/env python3
"""Plot the holonomic gain-tau model against the last drive_sine recording.

Reads dt, the six filter coefficients, and the three drag terms from
Dynamics_config in config/gates_r6_f2.yaml, rolls /tmp/drive_sine.csv through
the same update as holonomic_mppi.cu, and writes plots/drive_model.png. Open
it from the opendubs workspace at ops/drive_model.png.

    python3 scripts/drive_model_plot.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "config" / "gates_r6_f2.yaml"
CSV_PATH = Path("/tmp/drive_sine.csv")
# Host path: /home/sid/hound_nav/plots/drive_model.png, opened from the
# opendubs workspace via ops/drive_model.png.
PLOT_PATH = ROOT / "plots" / "drive_model.png"
KEYS = (
    "ax_gain", "tau_ax", "ay_gain", "tau_ay", "omega_gain", "tau_omega",
    "drag_x", "drag_y", "drag_omega",
)
NAMES = ("vx", "vy", "wz")


def load_params() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    cfg = yaml.safe_load(YAML_PATH.read_text())["nav"]["Dynamics_config"]
    missing = [k for k in ("dt", *KEYS) if cfg.get(k) is None]
    if missing:
        raise SystemExit(f"{YAML_PATH}: set {', '.join(missing)}")
    dt = float(cfg["dt"])
    gain = np.array([float(cfg[k]) for k in ("ax_gain", "ay_gain", "omega_gain")])
    tau = np.array([float(cfg[k]) for k in ("tau_ax", "tau_ay", "tau_omega")])
    drag = np.array([float(cfg[k]) for k in ("drag_x", "drag_y", "drag_omega")])
    return dt, gain, tau, drag


def rollout(cmd: np.ndarray, t: np.ndarray, v0: np.ndarray, dt: float, gain: np.ndarray, tau: np.ndarray, drag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kernel step on flat ground: a = gain * (tau * new_a + (1 - tau) * a) - drag * v."""
    ts = np.arange(dt, t[-1] + 1e-9, dt)
    pred = np.empty((len(ts), 3))
    v = v0.astype(float).copy()
    a_cmd = np.zeros(3)
    for i, tq in enumerate(ts):
        j = int(np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1))
        new_a = (cmd[j] - v) / dt
        a_cmd = gain * (new_a * tau + a_cmd * (1.0 - tau)) - drag * v
        v = v + a_cmd * dt
        pred[i] = v
    return ts, pred


def main() -> None:
    dt, gain, tau, drag = load_params()
    rows = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    t, cmd, odom = rows[:, 0], rows[:, 1:4], rows[:, 4:7]
    ts, pred = rollout(cmd, t, odom[0], dt, gain, tau, drag)
    meas = np.vstack([np.interp(ts, t, odom[:, k]) for k in range(3)]).T
    rmse = np.sqrt(np.mean((pred - meas) ** 2, axis=0))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for k, ax in enumerate(axes):
        ax.plot(t, odom[:, k], color="k", lw=1.6, label="odom")
        ax.plot(t, cmd[:, k], color="0.7", lw=0.8, label="command")
        ax.plot(ts, pred[:, k], color="C0", lw=1.4, label="model")
        ax.set_ylabel(NAMES[k])
        ax.set_title(f"{NAMES[k]}  gain {gain[k]:.2f}  tau {tau[k]:.2f}  drag {drag[k]:.2f}  rmse {rmse[k]:.3f}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[2].set_xlabel("t (s)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=130)
    print(
        "gain",
        " ".join(f"{n} {g:.2f}" for n, g in zip(NAMES, gain)),
        " tau",
        " ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, tau)),
        " drag",
        " ".join(f"{n} {d:.2f}" for n, d in zip(NAMES, drag)),
    )
    print("rmse", " ".join(f"{n} {e:.3f}" for n, e in zip(NAMES, rmse)))
    print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
