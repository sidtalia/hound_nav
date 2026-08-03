# hound_nav

IGHA* + UW_mppi **ROS adapter** for HOUND. Consumes `hound_mapping/LocalMap`
(pre-fused elev / cost / normals) — no yaw warp and no nav-side slope rebuild.

## Hotpath

```
~/local_map → PDefBuffer (axis-aligned crop/resample)
  → optional async BEV worker (GPU tensors)
  → trackingCostCUDA + MPPI / IGHA*
  → control commands
```

Mapper BEV is robot-centered in **odom axes**. Nav only crops/resamples to
`Map_config.map_size` / `map_res`. Cost `[0,1]` → IGHA `0/255`.

## Layout

```
hound_nav/
  hound_nav/
    nav_node.py           ROS entry
    control_loop.py       planner / MPPI loop
    pdef_buffer.py        LocalMap + state + waypoints → PDef
    bev_worker.py         async BEV → GPU
    trackingCostCUDA.py   MPPI tracking cost (loads cuda/)
    cuda/                 tracking_cost.cu/.cpp
    nav_ipc_latency_probe.py
  config/                 nav_example.yaml
  scripts/                bench_mppi.py, bench_tracking_cost.py
```

Mission params: `hound_core` SSoT / launch. Example nav YAML under `config/`.
Slope threshold lives in mapping (`lethal_slope_deg`), not here.

## Build

```bash
cd /root/colcon_ws && colcon build --packages-select hound_nav --symlink-install
source install/setup.bash
```

Needs `IGHAStar` / MPPI on `PYTHONPATH` (see `deps_path.py`).

## Run

```bash
ros2 run hound_nav nav_node
```

Latency probe (with nav `skip_planner:=true event_driven:=true async_bev:=true`):

```bash
ros2 run hound_nav nav_ipc_latency_probe
```

## Notes

- Prefer mapper normals over finite-difference on nav.
- `Cost_config.lethal_w` weights the mapper costmap; it does not re-threshold slope.
