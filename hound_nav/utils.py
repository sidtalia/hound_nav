"""Small nav helpers previously imported from BeamNGRL.utils.planning."""

from __future__ import annotations

import numpy as np


def _project_to_lookahead(pos, wp, lookahead):
    """If wp is farther than lookahead, snap it onto that circle toward wp."""
    pos = np.asarray(pos, dtype=np.float64)[:2]
    wp = np.asarray(wp, dtype=np.float64)[:2].copy()
    d = float(np.linalg.norm(wp - pos))
    if d > lookahead and lookahead > 0.0:
        angle = np.arctan2(wp[1] - pos[1], wp[0] - pos[0])
        wp[0] = pos[0] + lookahead * np.cos(angle)
        wp[1] = pos[1] + lookahead * np.sin(angle)
    return wp


def update_goal(
    goal,
    pos,
    target_WP,
    current_wp_index,
    lookahead,
    step_size=1,
    wp_radius=2.0,
):
    """Advance along mission waypoints; project goal onto the lookahead circle.

    Same carrot as IGHAStar BeamNG ``update_goal``: if the selected waypoint
    is farther than ``lookahead``, the planner goal is
    ``pos + lookahead * (wp - pos) / ‖wp - pos‖``.
    """
    target_WP = np.asarray(target_WP, dtype=np.float64)
    n = int(target_WP.shape[0]) if target_WP.ndim >= 1 else 0
    if n < 1:
        return pos, True, 0

    # Clamp: past-end index means mission already finished (esp. single-WP).
    if current_wp_index < 0:
        current_wp_index = 0
    if current_wp_index >= n:
        return np.asarray(pos, dtype=np.float64)[:2].copy(), True, n

    pos = np.asarray(pos, dtype=np.float64)[:2]
    if goal is None:
        return (
            _project_to_lookahead(pos, target_WP[current_wp_index, :2], lookahead),
            False,
            current_wp_index,
        )

    goal = np.asarray(goal, dtype=np.float64)[:2]
    d = float(np.linalg.norm(goal - pos))
    last = n - 1

    # Arrived at final waypoint → done (do not bump index past end for next tick).
    if current_wp_index >= last and d < wp_radius:
        return pos.copy(), True, n

    terminate = False
    while d < lookahead and current_wp_index < last:
        current_wp_index = min(current_wp_index + step_size, last)
        d = float(np.linalg.norm(target_WP[current_wp_index, :2] - pos))
        if current_wp_index >= last and d < wp_radius:
            terminate = True
            break

    if terminate:
        return pos.copy(), True, n
    return (
        _project_to_lookahead(pos, target_WP[current_wp_index, :2], lookahead),
        False,
        current_wp_index,
    )


def wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def pose_distances(
    path: np.ndarray,
    pos_xy: np.ndarray,
    yaw: float = 0.0,
    *,
    metric: str = "screw",
    screw_length_m: float = 1.0,
) -> np.ndarray:
    """Per-sample distance from robot pose to path rows (x,y[,yaw,...]).

    ``metric``:
      - ``xy``: Euclidean planar distance
      - ``screw``: SE(2) screw / weighted pose distance
        ``sqrt(dx^2 + dy^2 + (L * wrap(dθ))^2)`` with ``L = screw_length_m``
    """
    path = np.asarray(path, dtype=np.float64)
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    dxy = path[:, :2] - pos
    xy_d2 = np.sum(dxy * dxy, axis=1)
    m = str(metric).lower().strip()
    if m == "xy" or float(screw_length_m) <= 0.0 or path.shape[1] < 3:
        return np.sqrt(xy_d2)
    dth = wrap_pi(path[:, 2] - float(yaw))
    L = float(screw_length_m)
    return np.sqrt(xy_d2 + (L * dth) ** 2)


def find_closest_index(
    pos,
    target_WP,
    yaw: float = 0.0,
    *,
    metric: str = "xy",
    screw_length_m: float = 1.0,
):
    return int(
        np.argmin(
            pose_distances(
                target_WP,
                pos,
                yaw,
                metric=metric,
                screw_length_m=screw_length_m,
            )
        )
    )


def path_pose_to_start_state(robot_state: np.ndarray, path_row: np.ndarray) -> np.ndarray:
    """Copy plant state; overwrite xy / yaw / body speed from path sample (x,y,yaw,v)."""
    out = np.asarray(robot_state, dtype=np.float64).copy()
    row = np.asarray(path_row, dtype=np.float64).reshape(-1)
    if out.size >= 2 and row.size >= 2:
        out[0] = row[0]
        out[1] = row[1]
    if out.size >= 6 and row.size >= 3:
        out[5] = row[2]
    if out.size >= 8 and row.size >= 4:
        out[6] = float(row[3])
        out[7] = 0.0
    return out
