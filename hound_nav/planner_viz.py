"""OpenCV planner debug viz (same logic as IGHAStar/examples/ROS/utils.py)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def paint_cost_elev_map(
    costmap: np.ndarray,
    elevation_map: np.ndarray,
    map_size: int,
) -> np.ndarray:
    """Manager cost/elev paint only (no path / robot). Y-up world, not flipped."""
    costmap = cv2.resize(costmap, (map_size, map_size))
    elevation_map = cv2.resize(elevation_map, (map_size, map_size))
    costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
    pink = np.array([255, 105, 180], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    color_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    mask_white = costmap_color >= 250
    color_map[mask_white] = white
    color_map[~mask_white] = pink
    elev_norm = np.clip((elevation_map + 4) / 8, 0, 1)
    elev_uint8 = (elev_norm * 255).astype(np.uint8)
    elev_color = np.stack([elev_uint8] * 3, axis=-1)
    display_img = color_map.copy()
    display_img[mask_white] = elev_color[mask_white]
    return display_img


def visualize_map_with_path(
    costmap: np.ndarray,
    elevation_map: np.ndarray,
    path: Optional[np.ndarray],
    goal: np.ndarray,
    state: np.ndarray,
    wp_radius: float,
    map_center: np.ndarray,
    map_size: int,
    resolution_inv: float,
) -> np.ndarray:
    """Cost (pink/white) + elev blend, path, goal circle, robot box.

    Same free/lethal paint as IGHAStar ``examples/standalone/utils.py``:
    cost >= 250 is free (elevation grey); below that is pink (obstacle).
    World XY relative to ``map_center``; image origin after flip matches
    cv2.imshow (top-left) vs planner map (bottom-left).
    """
    display_img = paint_cost_elev_map(costmap, elevation_map, map_size)

    goal_disp = np.asarray(goal[:2], dtype=np.float64) - np.asarray(
        map_center[:2], dtype=np.float64
    )
    goal_x = int(
        np.clip((goal_disp[0] * resolution_inv) + map_size // 2, 0, map_size - 1)
    )
    goal_y = int(
        np.clip((goal_disp[1] * resolution_inv) + map_size // 2, 0, map_size - 1)
    )
    radius = max(2, int(wp_radius * resolution_inv))
    cv2.circle(display_img, (goal_x, goal_y), radius, (255, 255, 255), 2)

    if path is not None and len(path) > 0:
        path_disp = np.copy(path)
        path_disp[..., :2] -= np.asarray(map_center[:2], dtype=np.float64)
        path_X = np.clip(
            (path_disp[..., 0] * resolution_inv) + map_size // 2, 0, map_size - 1
        ).astype(int)
        path_Y = np.clip(
            (path_disp[..., 1] * resolution_inv) + map_size // 2, 0, map_size - 1
        ).astype(int)
        car_width_px = max(1, int(0.15 * resolution_inv))

        direction = path[..., -1] if path.shape[-1] > 4 else np.ones(len(path))
        velocity = path[..., 3] if path.shape[-1] > 3 else np.ones(len(path))
        vmin = float(np.min(velocity))
        vmax = float(np.max(velocity))
        if vmax > vmin:
            velocity_norm = (velocity - vmin) / (vmax - vmin)
        else:
            velocity_norm = np.ones_like(velocity, dtype=np.float64)
        velocity_color = np.clip(velocity_norm * 255, 0, 255).astype(np.uint8)

        for i in range(len(path_X) - 1):
            if float(direction[i]) >= 0:
                color = (0, int(velocity_color[i]), 0)
            else:
                color = (int(velocity_color[i]), 0, 0)
            cv2.line(
                display_img,
                (path_X[i], path_Y[i]),
                (path_X[i + 1], path_Y[i + 1]),
                color,
                car_width_px,
            )

    if state is not None and len(state) >= 3:
        x = float(state[0]) - float(map_center[0])
        y = float(state[1]) - float(map_center[1])
        theta = np.pi - float(state[2])
        x_px = int(x * resolution_inv + map_size // 2)
        y_px = int(y * resolution_inv + map_size // 2)
        car_width_px = max(2, int(0.29 * resolution_inv))
        car_height_px = max(2, int(0.15 * resolution_inv))
        half_width = car_width_px // 2
        half_height = car_height_px // 2
        corners = np.array(
            [
                [x_px - half_width, y_px - half_height],
                [x_px + half_width, y_px - half_height],
                [x_px + half_width, y_px + half_height],
                [x_px - half_width, y_px + half_height],
            ],
            dtype=np.int32,
        )
        rotation_matrix = cv2.getRotationMatrix2D((x_px, y_px), np.degrees(theta), 1.0)
        rotated_corners = cv2.transform(np.array([corners]), rotation_matrix)[0]
        cv2.polylines(
            display_img, [rotated_corners], isClosed=True, color=(0, 0, 0), thickness=2
        )

    return cv2.flip(display_img, 0)


@dataclass
class PlannerVisFrame:
    costmap: np.ndarray
    height: np.ndarray
    path: Optional[np.ndarray]
    goal_xy: np.ndarray
    state_xy_yaw: np.ndarray
    map_center: np.ndarray
    map_res: float
    wp_radius: float
    ok: bool
    expansions: int
    note: str = ""


class PlannerVisWorker:
    """Background OpenCV window; manager only swaps the latest frame."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_name: str = "hound_planner_vis",
        map_size: int = 480,
        period_s: float = 0.05,
    ) -> None:
        self._enabled = bool(enabled)
        self._window_name = str(window_name)
        self._map_size = int(map_size)
        self._period_s = float(period_s)
        self._lock = threading.Lock()
        self._frame: Optional[PlannerVisFrame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._enabled:
            self._thread = threading.Thread(
                target=self._loop, name="hound_planner_vis", daemon=True
            )
            self._thread.start()

    def update(self, frame: PlannerVisFrame) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._frame = frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass

    def _loop(self) -> None:
        last_note = ""
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(self._period_s)
                continue
            try:
                extent_m = float(frame.costmap.shape[0]) * float(frame.map_res)
                resolution_inv = (
                    float(self._map_size) / extent_m if extent_m > 1e-6 else 1.0
                )
                img = visualize_map_with_path(
                    frame.costmap,
                    frame.height,
                    frame.path,
                    frame.goal_xy,
                    frame.state_xy_yaw,
                    frame.wp_radius,
                    frame.map_center,
                    self._map_size,
                    resolution_inv,
                )
                status = "OK" if frame.ok else "FAIL"
                label = (
                    f"{status} exp={frame.expansions} "
                    f"goal=({frame.goal_xy[0]:.2f},{frame.goal_xy[1]:.2f}) "
                    f"start=({frame.state_xy_yaw[0]:.2f},{frame.state_xy_yaw[1]:.2f})"
                )
                if frame.note:
                    label = f"{label} | {frame.note}"
                cv2.putText(
                    img,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(self._window_name, img)
                cv2.waitKey(1)
                if frame.note and frame.note != last_note:
                    last_note = frame.note
            except Exception as exc:
                # Headless / no DISPLAY: print once per distinct message.
                msg = f"[hound_planner_vis] {type(exc).__name__}: {exc}"
                if msg != last_note:
                    print(msg, flush=True)
                    last_note = msg
            time.sleep(self._period_s)
