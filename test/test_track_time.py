"""Time index on a stamped path. No ROS."""

import numpy as np

from hound_nav.track_time import (
    clamp_twist,
    need_plan,
    open_loop_command,
    path_index,
    ref_window,
)
from hound_nav.traj_buffer import TrajBuffer


def _path():
    # Two edges. Eight rows share one g, then eight share the next.
    g = np.repeat([0.8, 1.6], 8)
    rows = np.zeros((16, 7), dtype=np.float64)
    rows[:, 0] = np.arange(16, dtype=np.float64)
    rows[:, 3] = 1.0
    rows[:, 4] = 0.2
    rows[:, 5] = 0.1
    rows[:, -1] = g
    return rows


def test_path_index_lands_on_the_edge_stamp():
    path = _path()
    # First edge is stamped with its end time, so t=0.8 is still that edge.
    assert path_index(path, 0.0) == 0
    assert path_index(path, 0.8) == 0
    assert path_index(path, 0.81) == 8
    assert path_index(path, 5.0) == 15


def test_ref_window_steps_by_stride_and_drops_g():
    path = _path()
    window = ref_window(path, 0.0, horizon=4, stride=2)
    assert window.shape == (4, 6)
    assert list(window[:, 0]) == [0.0, 2.0, 4.0, 6.0]


def test_need_plan_holds_until_the_last_edge():
    path = _path()
    assert need_plan(None, 0.0, 1) is True
    assert need_plan(path, 0.0, 1) is False
    # Last edge shares one g, so the index stays on its first row until the
    # clock passes that stamp. Then searchsorted falls off the end.
    assert need_plan(path, 1.6, 1) is False
    assert need_plan(path, 1.61, 1) is True


def test_open_loop_uses_the_next_row_and_stops_past_the_end():
    path = _path()
    controls = path[:, 3:6]
    # side=right: t == stamp takes the following row, not the row that just ended.
    at_boundary = open_loop_command(path, controls, 0.8)
    assert at_boundary is not None
    assert at_boundary[0] == 1.0
    assert np.allclose(at_boundary, controls[8])
    assert open_loop_command(path, controls, 1.61) is None
    assert open_loop_command(None, controls, 0.0) is None


def test_clamp_matches_planner_caps():
    cmd = clamp_twist([3.0, 2.0, -4.0], max_vel=1.55, vy_max=0.55, max_omega=1.5)
    assert np.allclose(cmd, [1.55, 0.55, -1.5])
    stopped = clamp_twist([-1.0, 0.0, 0.0], max_vel=1.55, vy_max=0.55, max_omega=1.5)
    assert stopped[0] == 0.0


def test_time_horizon_is_body_relative():
    buf = TrajBuffer()
    buf.replace(_path())
    window = buf.time_horizon([10.0, 0.0], 0.0, 4, stride=1)
    assert window is not None
    assert window.shape == (4, 6)
    assert np.allclose(window[:, 0], [-10.0, -9.0, -8.0, -7.0])
    assert np.allclose(window[:, 3], 1.0)


if __name__ == "__main__":
    test_path_index_lands_on_the_edge_stamp()
    test_ref_window_steps_by_stride_and_drops_g()
    test_need_plan_holds_until_the_last_edge()
    test_open_loop_uses_the_next_row_and_stops_past_the_end()
    test_clamp_matches_planner_caps()
    test_time_horizon_is_body_relative()
    print("ok")
