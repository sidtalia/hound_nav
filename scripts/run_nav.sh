#!/usr/bin/env bash
# Live Gates nav, inside opendubs-gates-sim. Ctrl-C stops the launch and the bridge.
# Do not source bashrc_common: it forces zenoh and localhost-only discovery.

source /opt/ros/jazzy/setup.bash
source /root/colcon_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/opendubs/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST=11.0
export HOUND_NAV_YAML=/root/colcon_ws/src/hound_nav/config/gates_r6_f2.yaml
export ROS_DISABLE_DAEMON=1

root=/root/colcon_ws/src/hound_nav
aruco=/root/colcon_ws/src/aruco_registration
ssot=/root/colcon_ws/src/hound_core/config/SSoT.yaml
# Color frames are published on the camera profile. UDPv4 alone drops them.
cam_xml="${aruco}/fastdds_camera.xml"

det_log=$(mktemp /tmp/aruco-detector.XXXX.log)
# shellcheck disable=SC2086
env -u FASTDDS_BUILTIN_TRANSPORTS \
  FASTRTPS_DEFAULT_PROFILES_FILE="$cam_xml" \
  FASTDDS_DEFAULT_PROFILES_FILE="$cam_xml" \
  PYTHONPATH="${aruco}${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u "${aruco}/aruco_registration/detector_node.py" --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p dictionary:=DICT_4X4_1000 \
  -p marker_id:=0 \
  -p marker_size_m:=0.18 \
  -p min_side_px:=12.0 \
  -p corner_refine:=apriltag \
  -p corner_refine_win_size:=10 \
  -p dense_refine:=all \
  -p dense_refine_optimize:=both \
  -p publish_overlay:=true \
  -p publish_detect_tf:=true \
  -p trigger_timeout_s:=8.0 \
  -p ssot_file:="$ssot" >"$det_log" 2>&1 &
det=$!

PYTHONPATH="${aruco}${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u "${aruco}/aruco_registration/marker_survey.py" --ros-args \
  -p ssot_file:="$ssot" &
survey=$!

python3 -u "${root}/scripts/robot_bridge.py" --config "${HOUND_NAV_YAML}" &
bridge=$!
python3 -u "${root}/scripts/nav_drive_log.py" &
logpid=$!

log=$(mktemp /tmp/hound-nav.XXXX.log)
ros2 launch "${root}/launch/hound_nav.launch.py" > >(tee -a "$log") 2>&1 &
navpid=$!

trap 'kill -TERM "$det" "$survey" "$bridge" "$logpid" "$navpid" 2>/dev/null || true' INT TERM EXIT

# Detector must stay up: the survey only averages poses the detector publishes.
sleep 2
if ! kill -0 "$det" 2>/dev/null; then
  echo "ArUco detector exited; see $det_log" >&2
  tail -n 40 "$det_log" >&2 || true
  exit 1
fi

ready=0
for _ in $(seq 1 180); do
  if grep -q "\[hound_planner\]" "$log" && grep -q "MPPI ready" "$log"; then
    ready=1
    break
  fi
  if ! kill -0 "$navpid" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$det" 2>/dev/null; then
    echo "ArUco detector died while waiting for planner; see $det_log" >&2
    tail -n 40 "$det_log" >&2 || true
    exit 1
  fi
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "planner and MPPI did not come up; ArUco trigger not sent" >&2
  exit 1
fi
if ! kill -0 "$det" 2>/dev/null; then
  echo "ArUco detector is down; trigger not sent. See $det_log" >&2
  tail -n 40 "$det_log" >&2 || true
  exit 1
fi

# Survey averages 8 s, then publishes /initialpose if the tag was seen.
ros2 topic pub --once -w 1 /aruco_registration/trigger std_msgs/msg/Empty "{}"
wait "$navpid"
