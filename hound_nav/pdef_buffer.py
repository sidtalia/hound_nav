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
    cost: np.ndarray  # HxW float32, 0 free .. 1 lethal (semantic OR slope)
    origin_x: float
    origin_y: float
    resolution: float
    normals: Optional[np.ndarray] = None  # HxWx3 unit normals from mapper
    stamp_sec: float = 0.0


@dataclass
class PDef:
    """Snapshot consumed by the planner/MPPI loop (BeamNG example shape)."""

    state: np.ndarray  # (17,) pos,rpy,vel,acc,gyro,steer,thr
    height_bev: np.ndarray  # HxW robot-centric, relative Z
    normal_bev: np.ndarray  # HxW x 3
    costmap: np.ndarray  # HxW, 0/255 IGHA* convention
    map_center: np.ndarray  # (2,) world xy (= robot pos for body BEV)
    goal: Optional[np.ndarray]  # (2,) world xy
    target_wp: Optional[np.ndarray] = None  # Nx>=2 waypoints world
    map_res: float = 0.25
    map_size: float = 100.0


def _axis_aligned_window(
    img: np.ndarray,
    ox: float,
    oy: float,
    res_w: float,
    center_xy: np.ndarray,
    map_size_m: float,
    map_res: float,
    border_value: float = 0.0,
) -> np.ndarray:
    """Crop/resample robot-centered LocalMap into n×n at map_res (no yaw)."""
    n = int(round(map_size_m / map_res))
    scale = float(map_res) / float(res_w)
    half = 0.5 * n
    cx, cy = float(center_xy[0]), float(center_xy[1])
    M = np.array(
        [
            [scale, 0.0, (cx - ox) / res_w - scale * half],
            [0.0, scale, (cy - oy) / res_w - scale * half],
        ],
        dtype=np.float64,
    )
    src = img if img.dtype == np.float32 else img.astype(np.float32)
    if src.ndim == 2:
        return cv2.warpAffine(
            src,
            M,
            (n, n),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    # HxWxC (normals)
    out = cv2.warpAffine(
        src,
        M,
        (n, n),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return out


class PDefBuffer:
    """ROS callbacks write; control loop reads under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = np.zeros(17, dtype=np.float64)
        self._state_valid = False
        self._map: Optional[LocalMapSnapshot] = None
        self._target_wp: Optional[np.ndarray] = None
        self._injected_path: Optional[np.ndarray] = None  # Nx4 world (x,y,yaw,vel)
        self._last_action = np.zeros(2, dtype=np.float64)
        self._state_stamp_sec: float = 0.0  # receive time
        self._state_pub_stamp_sec: float = 0.0  # optional publisher time
        self._map_gen: int = 0
        self._inject_gen: int = 0
        # Set on new control_state; control loop waits on this when event-driven.
        self._state_event = threading.Event()
        # Kick background BEV rebuild (map/state).
        self._bev_event = threading.Event()
        self._ready_bev: Any = None

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
        state17: np.ndarray,
        stamp_sec: float = 0.0,
        pub_stamp_sec: float = 0.0,
    ) -> None:
        """Full BeamNG plant state: pos,rpy,vel,A,G,steer,wheelspeed (17).

        stamp_sec is receive time (used for state→cmd latency).
        pub_stamp_sec is optional publisher time for probe correlation only.
        """
        arr = np.asarray(state17, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 17:
            return
        with self._lock:
            self._state[:] = arr[:17]
            self._last_action = np.copy(self._state[15:17])
            self._state_valid = True
            if stamp_sec > 0.0:
                self._state_stamp_sec = float(stamp_sec)
            self._state_pub_stamp_sec = float(pub_stamp_sec)
        self._state_event.set()
        self._bev_event.set()

    def get_state_stamp_sec(self) -> float:
        with self._lock:
            return self._state_stamp_sec

    def get_state_pub_stamp_sec(self) -> float:
        with self._lock:
            return self._state_pub_stamp_sec

    def map_generation(self) -> int:
        with self._lock:
            return self._map_gen

    def inject_generation(self) -> int:
        with self._lock:
            return self._inject_gen

    def get_injected_path(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._injected_path is None:
                return None
            return np.copy(self._injected_path)

    def set_injected_path(self, path: Optional[np.ndarray]) -> None:
        with self._lock:
            if path is None:
                self._injected_path = None
            else:
                self._injected_path = np.asarray(path, dtype=np.float64)
            self._inject_gen += 1
        self._bev_event.set()

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
        self._bev_event.set()

    def set_last_action(self, action: np.ndarray) -> None:
        with self._lock:
            self._last_action = np.asarray(action, dtype=np.float64).reshape(2)
            self._state[15:17] = self._last_action

    def set_local_map(self, snap: LocalMapSnapshot) -> None:
        with self._lock:
            self._map = snap
            self._map_gen += 1
        self._bev_event.set()

    def set_waypoints(self, wp_xy: np.ndarray) -> None:
        """wp_xy: Nx2+ in world frame."""
        with self._lock:
            self._target_wp = np.asarray(wp_xy, dtype=np.float64)

    def ready(self) -> bool:
        with self._lock:
            return self._state_valid and self._map is not None

    def snapshot_pdef(
        self,
        map_size_m: float,
        map_res: float,
        goal_xy: Optional[np.ndarray] = None,
    ) -> Optional[PDef]:
        """Take mapper LocalMap (already robot-centered) → BeamNG BEV tensors.

        No yaw warp / slope recompute — only axis-aligned crop/resample to the
        nav map size/res, relative-Z, and [0,1]→IGHA 0/255 cost.
        """
        with self._lock:
            if not self._state_valid or self._map is None:
                return None
            state = np.copy(self._state)
            m = self._map
            elev = m.elevation
            cost = m.cost
            normals = m.normals
            ox, oy, res_w = m.origin_x, m.origin_y, m.resolution
            target_wp = None if self._target_wp is None else np.copy(self._target_wp)

        n = int(round(map_size_m / map_res))
        if n < 8:
            return None

        # Geometric center of LocalMap (== robot when mapper centers extract).
        center = np.array(
            [
                ox + 0.5 * elev.shape[1] * res_w,
                oy + 0.5 * elev.shape[0] * res_w,
            ],
            dtype=np.float64,
        )

        height = _axis_aligned_window(
            elev, ox, oy, res_w, center, map_size_m, map_res, border_value=0.0
        )
        cost_s = _axis_aligned_window(
            cost, ox, oy, res_w, center, map_size_m, map_res, border_value=1.0
        )

        z0 = float(height[n // 2, n // 2]) if np.isfinite(height[n // 2, n // 2]) else float(
            state[2]
        )
        height = np.nan_to_num(height - z0, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )

        if normals is not None and normals.ndim == 3 and normals.shape[2] >= 3:
            normal = _axis_aligned_window(
                normals[..., :3],
                ox,
                oy,
                res_w,
                center,
                map_size_m,
                map_res,
                border_value=0.0,
            )
            normal = normal.astype(np.float32, copy=False)
            norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
            normal = normal / norm
        else:
            # Fallback if older LocalMap without normals.
            nx = -cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
            ny = -cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
            nz = np.ones_like(height)
            normal = np.stack([nx, ny, nz], axis=-1)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
            normal = normal.astype(np.float32, copy=False)

        # Mapper cost is [0,1] lethal; IGHA*/TrackingCost expect 0 lethal / 255 free.
        costmap = np.where(cost_s < 0.5, 255.0, 0.0).astype(np.float32)

        return PDef(
            state=state,
            height_bev=height,
            normal_bev=normal,
            costmap=costmap,
            map_center=center,
            goal=None if goal_xy is None else np.asarray(goal_xy, dtype=np.float64),
            target_wp=target_wp,
            map_res=map_res,
            map_size=map_size_m,
        )
