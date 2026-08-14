"""World-frame trajectory buffer (replace / append / truncate / horizon crop)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from hound_nav.utils import pose_distances


class TrajBuffer:
    """Latest world-frame path. First slice: replace-only; append/truncate for later."""

    def __init__(self) -> None:
        self._path: Optional[np.ndarray] = None

    def replace(self, path: np.ndarray) -> None:
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            self._path = None
            return
        if arr.shape[1] < 4:
            pad = np.zeros((arr.shape[0], 4), dtype=np.float64)
            pad[:, : arr.shape[1]] = arr
            arr = pad
        self._path = np.ascontiguousarray(arr[:, :4])

    def append(self, path: np.ndarray) -> None:
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            return
        if arr.shape[1] < 4:
            pad = np.zeros((arr.shape[0], 4), dtype=np.float64)
            pad[:, : arr.shape[1]] = arr
            arr = pad
        arr = arr[:, :4]
        if self._path is None or len(self._path) == 0:
            self.replace(arr)
            return
        self._path = np.concatenate([self._path, arr], axis=0)

    def truncate_after(self, index: int) -> None:
        if self._path is None:
            return
        if index <= 0:
            self._path = None
            return
        self._path = self._path[:index]

    def clear(self) -> None:
        self._path = None

    def get(self) -> Optional[np.ndarray]:
        if self._path is None:
            return None
        return np.copy(self._path)

    def empty(self) -> bool:
        return self._path is None or len(self._path) < 2

    def ref_index(
        self,
        pos_xy: np.ndarray,
        yaw: float = 0.0,
        *,
        metric: str = "screw",
        screw_length_m: float = 1.0,
    ) -> Optional[int]:
        if self.empty():
            return None
        return int(
            np.argmin(
                pose_distances(
                    self._path,
                    pos_xy,
                    float(yaw),
                    metric=metric,
                    screw_length_m=screw_length_m,
                )
            )
        )

    def ref_distance(
        self,
        pos_xy: np.ndarray,
        yaw: float = 0.0,
        *,
        metric: str = "screw",
        screw_length_m: float = 1.0,
    ) -> float:
        """Distance from robot pose to closest path sample (same metric as tracking)."""
        i = self.ref_index(
            pos_xy, yaw, metric=metric, screw_length_m=screw_length_m
        )
        if i is None:
            return float("inf")
        return float(
            pose_distances(
                self._path[i : i + 1],
                pos_xy,
                float(yaw),
                metric=metric,
                screw_length_m=screw_length_m,
            )[0]
        )

    def robot_too_far_from_ref(
        self,
        pos_xy: np.ndarray,
        yaw: float = 0.0,
        *,
        max_dist_m: float,
        metric: str = "screw",
        screw_length_m: float = 1.0,
    ) -> bool:
        if self.empty() or float(max_dist_m) <= 0.0:
            return True
        return (
            self.ref_distance(
                pos_xy, yaw, metric=metric, screw_length_m=screw_length_m
            )
            > float(max_dist_m)
        )

    def pose_after(
        self,
        pos_xy: np.ndarray,
        yaw: float = 0.0,
        *,
        dt_ahead_s: float,
        dt_s: float,
        metric: str = "screw",
        screw_length_m: float = 1.0,
    ) -> Optional[np.ndarray]:
        """World-frame path sample (x,y,yaw,v) ``dt_ahead_s`` after closest ref.

        Planner traj samples are already timed: consecutive states are ``dt_s`` apart.
        ``dt_ahead_s`` is typically average planning latency (SSoT / running avg).
        """
        i0 = self.ref_index(
            pos_xy, yaw, metric=metric, screw_length_m=screw_length_m
        )
        if i0 is None:
            return None
        path = self._path
        assert path is not None
        dt = max(float(dt_s), 1e-6)
        steps = max(float(dt_ahead_s), 0.0) / dt
        i_f = float(i0) + steps
        i = int(np.floor(i_f))
        a = float(i_f - i)
        if i >= len(path) - 1:
            return np.copy(path[-1, :4])
        if a <= 1e-9:
            return np.copy(path[i, :4])
        j = min(i + 1, len(path) - 1)
        row = (1.0 - a) * path[i, :4] + a * path[j, :4]
        row[2] = path[i, 2] if a < 0.5 else path[j, 2]
        return row

    def horizon(
        self,
        pos_xy: np.ndarray,
        timesteps: int,
        *,
        yaw: float = 0.0,
        metric: str = "screw",
        screw_length_m: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Body-relative T×4 crop around closest pose. stop=True if near end.

        Closest index uses ``pose_distances`` (``xy`` or SE(2) ``screw``).
        """
        ref_i = self.ref_index(
            pos_xy, yaw, metric=metric, screw_length_m=screw_length_m
        )
        if ref_i is None or timesteps < 1:
            return None, False
        world = np.copy(self._path)
        pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        body = np.copy(world)
        body[:, :2] -= pos
        t = int(timesteps)
        if ref_i < len(body) - t:
            return body[ref_i : ref_i + t, :4], False
        out = np.zeros((t, 4), dtype=np.float64)
        avail = body[ref_i:, :4]
        out[: len(avail)] = avail
        if len(avail) > 0:
            out[len(avail) :, :3] = out[len(avail) - 1, :3]
        stop = ref_i >= len(body) - 10
        return out, stop
