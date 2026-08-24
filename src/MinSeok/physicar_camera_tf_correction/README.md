# PhysiCar corrected camera TF broadcaster

This package implements only `camera_corrected_tf_broadcaster`. It does not
modify PhysiCar URDF/SDF, the original bringup launch files, `run.sh`, or
`lidar_camera_projection.py`.

## Published TF branch

```text
camera_pan_link
└── camera_tilt_link_corrected        dynamic, stamp copied from /joint_states
    └── camera_link_corrected         static
        └── camera_optical_frame_corrected  static
```

The dynamic transform uses the `camera_tilt_joint` position `q` from
`/joint_states` and applies `q_corrected = -q`. The node does not subscribe to
camera pan commands or republish `camera_pan_link`.

## Build and run in a ROS 2 workspace

```bash
cd /home/physicar/physicar_ws
colcon build --symlink-install --packages-select physicar_camera_tf_correction
source install/setup.bash
ros2 run physicar_camera_tf_correction camera_corrected_tf_broadcaster \
  --ros-args -p use_sim_time:=true
```

`/joint_states` is a fixed input name but remains remappable through normal ROS
2 remapping if the deployment requires it. Frame names and transform constants
are intentionally not parameters.

## Unit tests

The ROS-independent core tests implement the broadcaster cases from design
section 11.1 and can run without Gazebo:

```bash
cd /home/physicar/physicar_ws/src/physicar_camera_tf_correction
PYTHONPATH=. python3 -m unittest discover -s test -p 'test_corrected_tf_core.py' -v
```

In a sourced ROS 2 environment, run the package test suite with:

```bash
cd /home/physicar/physicar_ws
colcon test --packages-select physicar_camera_tf_correction --event-handlers console_direct+
colcon test-result --verbose
```

## Broadcaster-only TF integration checks

Start the original PhysiCar simulator and `robot_state_publisher`, then run this
node with `use_sim_time:=true`. The checks that do not require a startup
coordinator are:

1. Confirm `/joint_states` contains exactly one `camera_tilt_joint`, a position,
   and a nonzero header stamp.
2. Confirm there is exactly one `camera_corrected_tf_broadcaster` process.
3. Inspect the TF graph. The original and corrected branches must coexist; each
   corrected child has one parent and there is no cycle.
4. Command `/camera/tilt` to `0`, `-0.5224`, and `+0.5236` rad and record the
   steady-state JointState positions. Do not use the command value as TF input.
5. Record `/joint_states` and `/tf`; compare the matching corrected dynamic TF
   stamp with the source JointState stamp at exact `sec`/`nanosec` precision.
6. For `q=-0.5224`, verify the dynamic quaternion is approximately
   `(x,y,z,w)=(0,0.258240,0,0.966081)`.
7. Verify `camera_pan_link -> camera_optical_frame_corrected` against the design
   values. For `q=-0.5224`, the optical origin in the pan frame is approximately
   `(0.057984,0,0.010164)` m and optical `+Z` is approximately
   `(0.866624,0,-0.498961)`.
8. Compare optical `+Z` with the Gazebo camera link pose or image viewing
   direction. The vertical direction must agree.
9. Repeat at pan `+/-30` degrees. Both original and corrected branches must move
   under the original `camera_pan_link`; this node must not publish a pan TF.
10. Reset Gazebo simulation time and confirm a new, lower nonzero JointState
    stamp is accepted only after the clock-jump epoch reset.

Coordinator, perception/fusion lifecycle, controller fail-safe, sensor-pair
full-time lookup, and `run.sh` checks are intentionally outside this package's
current scope.
