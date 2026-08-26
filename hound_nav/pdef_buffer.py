"""Thread-safe problem-definition buffer: map + state + mission waypoints."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class LocalMapSnapshot:
    elevation: np.ndarray  # HxW float32 world Z (robot-centered, odom axes)
    cost: np.ndarray  # HxW float32, 255 free .. 0 lethal (unobserved≈128)
    origin_x: float
    origin_y: float
    resolution: float
    stamp_sec: float = 0.0


@dataclass
class PDef:
    """Snapshot consumed by the planner/MPPI loop (BeamNG-compatible plant shape)."""

    state: np.ndarray  # (N,) plant state; N = control_state_dims (SSoT)
    height_bev: np.ndarray  # HxW robot-centric, relative Z
    normal_bev: np.ndarray  # HxW x 3
    costmap: np.ndarray  # HxW, 0/255 IGHA* convention
    map_center: np.ndarray  # (2,) world xy (= robot pos for body BEV)
    goal: Optional[np.ndarray]  # (2,) world xy
    target_wp: Optional[np.ndarray] = None  # Nx>=2 waypoints world
    map_res: float = 0.25
    map_size: float = 100.0


class PDefBuffer:
    """ROS callbacks write; manager reads under a lock."""

    def __init__(self, state_dims: int = 17) -> None:
        if state_dims < 2:
            raise ValueError(f"state_dims must be >= 2, got {state_dims}")
        self._state_dims = int(state_dims)
        self._lock = threading.Lock()
        self._state = np.zeros(self._state_dims, dtype=np.float64)
        self._state_valid = False
        self._map: Optional[LocalMapSnapshot] = None
        self._target_wp: Optional[np.ndarray] = None
        self._last_action = np.zeros(2, dtype=np.float64)
        self._state_stamp_sec: float = 0.0
        self._map_gen: int = 0
        self._wp_gen: int = 0
        # Set on new control_state; control loop waits on this when event-driven.
        self._state_event = threading.Event()
        # Kick background BEV rebuild (map).
        self._bev_event = threading.Event()
        self._ready_bev: Any = None

    @property
    def state_dims(self) -> int:
        return self._state_dims

    def notify_state(self) -> None:
        self._state_event.set()

    def wait_for_state(self, timeout: float = 0.1) -> bool:
        """Block until a new state arrives (or timeout). Clears the event."""
        ok = self._state_event.wait(timeout=timeout)
        if ok:
            self._state_event.clear()
        return ok

    def state_pending(self) -> bool:
        return self._state_event.is_set()

    def kick_bev(self) -> None:
        self._bev_event.set()

    def wait_for_bev_kick(self, timeout: float = 0.05) -> bool:
        ok = self._bev_event.wait(timeout=timeout)
        if ok:
            self._bev_event.clear()
        return ok

    def set_ready_bev(self, ready: Any) -> None:
        with self._lock:
            self._ready_bev = ready

    def get_ready_bev(self) -> Any:
        with self._lock:
            return self._ready_bev

    def get_state_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._state_valid:
                return None
            return np.copy(self._state)

    def set_state_vector(
        self,
        state: np.ndarray,
        stamp_sec: float = 0.0,
    ) -> None:
        """Plant state vector (length = control_state_dims from SSoT)."""
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.shape[0] < self._state_dims:
            return
        with self._lock:
            self._state[:] = arr[: self._state_dims]
            if self._state_dims >= 2:
                self._last_action = np.copy(self._state[-2:])
            self._state_valid = True
            if stamp_sec > 0.0:
                self._state_stamp_sec = float(stamp_sec)
        self._state_event.set()

    def get_state_stamp_sec(self) -> float:
        with self._lock:
            return self._state_stamp_sec

    def map_generation(self) -> int:
        with self._lock:
            return self._map_gen

    def set_odom_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
    ) -> None:
        with self._lock:
            if self._state_dims >= 9:
                self._state[0] = x
                self._state[1] = y
                self._state[2] = z
                self._state[3] = roll
                self._state[4] = pitch
                self._state[5] = yaw
                self._state[6] = vx
                self._state[7] = vy
                self._state[8] = vz
            self._state_valid = True
        self._state_event.set()

    def set_last_action(self, action: np.ndarray) -> None:
        with self._lock:
            self._last_action = np.asarray(action, dtype=np.float64).reshape(2)
            if self._state_dims >= 2:
                self._state[-2:] = self._last_action

    def set_local_map(self, snap: LocalMapSnapshot) -> None:
        with self._lock:
            self._map = snap
            self._map_gen += 1
        self._bev_event.set()

    def set_waypoints(self, wp_xy: np.ndarray) -> None:
        """wp_xy: Nx2+ in world frame. Bumps generation only when content changes."""
        wp = np.asarray(wp_xy, dtype=np.float64)
        with self._lock:
            if self._target_wp is not None and self._target_wp.shape == wp.shape:
                if np.allclose(self._target_wp, wp, rtol=0.0, atol=1e-3):
                    return
            self._target_wp = wp
            self._wp_gen += 1

    def waypoint_generation(self) -> int:
        with self._lock:
            return self._wp_gen

    def ready(self) -> bool:
        with self._lock:
            return self._state_valid and self._map is not None

    def missing_inputs(self) -> list[str]:
        """Names of inputs the planner still needs (empty ⇒ map+state present)."""
        with self._lock:
            missing: list[str] = []
            if not self._state_valid:
                missing.append("control_state")
            if self._map is None:
                missing.append("local_map")
            if self._target_wp is None or int(self._target_wp.shape[0]) < 1:
                missing.append("goal")
            return missing

    def snapshot_pdef(
        self,
        goal_xy: Optional[np.ndarray] = None,
    ) -> Optional[PDef]:
        """Mapper LocalMap elev+cost → relative-Z height + IGHA cost.

        Normals for dynamics API are finite-differenced from elevation (mapper
        no longer publishes them). Cost is free-ness [0,255]; unobserved≈128.
        """
        with self._lock:
            if not self._state_valid or self._map is None:
                return None
            state = np.copy(self._state)
            m = self._map
            elev = m.elevation
            cost = m.cost
            ox, oy, res_w = m.origin_x, m.origin_y, m.resolution
            target_wp = None if self._target_wp is None else np.copy(self._target_wp)

        h, w = int(elev.shape[0]), int(elev.shape[1])
        if h < 8 or w < 8 or cost.shape != elev.shape:
            return None

        center = np.array(
            [ox + 0.5 * w * res_w, oy + 0.5 * h * res_w],
            dtype=np.float64,
        )
        map_size = float(h) * float(res_w)

        height = np.ascontiguousarray(elev, dtype=np.float32)
        z0 = float(height[h // 2, w // 2]) if np.isfinite(height[h // 2, w // 2]) else float(
            state[2]
        )
        height = np.nan_to_num(height - z0, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )

        # Dynamics API still takes normals; FD from elev (unused in kernel body).
        nx = -cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        ny = -cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        nz = np.ones_like(height)
        normal = np.stack([nx, ny, nz], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
        normal = normal.astype(np.float32, copy=False)

        cost_s = np.ascontiguousarray(cost, dtype=np.float32)
        # Already 0..255 free-ness. Legacy: 0..1 lethal → invert to IGHA.
        if float(np.nanmax(cost_s)) <= 1.0 + 1e-3:
            costmap = np.where(cost_s < 0.5, 255.0, 0.0).astype(np.float32)
        else:
            costmap = np.clip(cost_s, 0.0, 255.0).astype(np.float32)

        return PDef(
            state=state,
            height_bev=height,
            normal_bev=normal,
            costmap=costmap,
            map_center=center,
            goal=None if goal_xy is None else np.asarray(goal_xy, dtype=np.float64),
            target_wp=target_wp,
            map_res=float(res_w),
            map_size=map_size,
        )
