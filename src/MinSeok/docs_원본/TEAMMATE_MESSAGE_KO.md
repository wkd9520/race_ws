# 팀원 전달용 문구

## 짧은 버전

장애물 회피 경로를 생성하거나 흰색으로 표시하는 부분은 제외하고, 카메라에서
만든 path와 LiDAR point를 한 BEV 화면에 동시에 겹쳐 보는 단계까지 묶었습니다.
압축을 푼 뒤 `INSTALL_KO.md` 순서대로 세 package를 workspace에 복사하고
build한 다음, simulator를 먼저 켜고
`ros2 launch physicar_track_perception_v3 perception_v3.launch.py`를 실행하면
됩니다. 통합 화면은 `/perception_v3/debug/path_lidar_overlay`를
`rqt_image_view`에서 선택해서 확인해 주세요.

## 호환성 확인까지 포함한 버전

이 묶음에는 `physicar_track_perception_v3`, V3가 사용하는 V2 metric-BEV 공통
모듈, corrected camera TF package가 들어 있습니다. obstacle avoidance 모듈과
`/avoidance_v3/**` 출력은 의도적으로 제외했습니다. 대상 simulator에서 camera,
joint state, scan topic과 `odom`, `base_footprint`, corrected camera TF 구조가
같은지 먼저 `scripts/preflight_runtime.sh`로 확인해 주세요. topic 이름만 다른
경우에는 launch argument로 바꿀 수 있지만, camera calibration이나 frame
geometry가 다르면 parameter/source adaptation 후 재검증이 필요합니다.

## 실행 명령만 전달하는 버전

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
cd ~/physicar_ws
colcon build --symlink-install --packages-select \
  physicar_camera_tf_correction \
  physicar_track_perception_v2 \
  physicar_track_perception_v3
source install/setup.bash

# simulator를 별도 터미널에서 먼저 실행한 뒤
ros2 launch physicar_track_perception_v3 perception_v3.launch.py

# 다른 터미널에서 통합 화면 확인
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
source ~/physicar_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view \
  /perception_v3/debug/path_lidar_overlay
```

상대방의 ROS/PhysiCar 설치 경로가 다르면 첫 두 source 경로를 실제 경로로
바꿔야 합니다.
