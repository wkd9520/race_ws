#!/usr/bin/env bash
set -u

failures=0

pass() {
  printf '[PASS] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1" >&2
  failures=$((failures + 1))
}

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command available: $1"
  else
    fail "command missing: $1"
  fi
}

check_package() {
  if ros2 pkg prefix "$1" >/dev/null 2>&1; then
    pass "ROS package available: $1"
  else
    fail "ROS package missing or environment not sourced: $1"
  fi
}

check_topic_type() {
  topic=$1
  expected=$2
  actual=$(timeout 4 ros2 topic type "$topic" 2>/dev/null || true)
  if [ "$actual" = "$expected" ]; then
    pass "$topic type=$expected"
  elif [ -z "$actual" ]; then
    fail "$topic is not visible (is the simulator running?)"
  else
    fail "$topic type mismatch: expected=$expected actual=$actual"
  fi
}

check_tf() {
  target=$1
  source=$2
  output=$(timeout 5 ros2 run tf2_ros tf2_echo "$target" "$source" 2>&1 || true)
  if printf '%s\n' "$output" | grep -Eq 'At time|Translation:'; then
    pass "TF available: $source -> $target"
  else
    fail "TF unavailable: $source -> $target"
  fi
}

check_command ros2
check_command timeout

if [ "${ROS_DISTRO:-}" = jazzy ]; then
  pass 'ROS_DISTRO=jazzy'
else
  fail "ROS_DISTRO is '${ROS_DISTRO:-unset}', expected jazzy"
fi

check_package physicar_camera_tf_correction
check_package physicar_track_perception_v2
check_package physicar_track_perception_v3

check_topic_type /camera/image_raw sensor_msgs/msg/Image
check_topic_type /joint_states sensor_msgs/msg/JointState
check_topic_type /scan sensor_msgs/msg/LaserScan
check_topic_type /clock rosgraph_msgs/msg/Clock

joint_sample=$(timeout 4 ros2 topic echo /joint_states --once 2>/dev/null || true)
if printf '%s\n' "$joint_sample" | grep -q 'camera_tilt_joint'; then
  pass '/joint_states contains camera_tilt_joint'
else
  fail '/joint_states does not show camera_tilt_joint in one sample'
fi

check_tf odom base_footprint

if ros2 node list 2>/dev/null | grep -qx '/camera_corrected_tf_broadcaster'; then
  check_tf base_footprint camera_optical_frame_corrected
else
  printf '[INFO] corrected TF node is not running yet; launch perception_v3.launch.py after this check.\n'
fi

if [ "$failures" -eq 0 ]; then
  printf '[RESULT] preflight PASS\n'
  exit 0
fi

printf '[RESULT] preflight FAIL (%d check(s))\n' "$failures" >&2
exit 1
