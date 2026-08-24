# 설치 및 실행 안내

## 1. 대상 simulator 호환 조건

이 코드는 아래 source interface를 사용합니다. 다른 simulator에서 이름이나
geometry가 다르면 실행 전에 대응 관계를 확인해야 합니다.

| 입력 | source에서 확인된 기본값 | message type |
|---|---|---|
| Camera | `/camera/image_raw` | `sensor_msgs/msg/Image` |
| Joint state | `/joint_states` | `sensor_msgs/msg/JointState` |
| LiDAR | `/scan` | `sensor_msgs/msg/LaserScan` |
| Simulation clock | `/clock` | `rosgraph_msgs/msg/Clock` |

Camera, JointState, LiDAR topic 이름은 launch argument로 바꿀 수 있습니다.

좌표변환 조건은 다음과 같습니다.

- Camera source frame: `camera_optical_frame_corrected`
- Camera target frame: `base_footprint`
- Camera source/target timestamp: image header timestamp `t_image`
- Camera transform direction: corrected camera frame의 점을 `base_footprint`로 변환
- LiDAR source frame/time: `LaserScan.header.frame_id` / `t_scan`
- LiDAR target frame/time: `base_footprint` / `t_image`
- LiDAR fixed frame: `odom`
- LiDAR transform direction: scan frame의 점을 `odom`을 경유해 image 시각의
  `base_footprint`로 변환
- latest-TF fallback은 사용하지 않습니다.

현재 PhysiCar source/interface에서는 `base_footprint`의 +X가 forward, +Y가
left입니다. 대상 simulator의 convention은 별도로 확인해야 합니다.

기본 image는 480x360이며 K/D, camera height correction(-0.018 m), projection
pitch offset(+2.8 deg)은 `config/perception_v3.yaml`에 있습니다. 대상 camera
model이나 mounting이 다르면 이 값을 그대로 사용하지 말고 재검증해야 합니다.

## 2. 압축 해제 및 무결성 확인

압축파일이 대상 workspace의 상위 디렉터리에 있다고 가정한 예시입니다.

```bash
mkdir -p ~/physicar_transfer
tar -xzf physicar_perception_v3_path_lidar_transfer_20260824.tar.gz \
  -C ~/physicar_transfer
cd ~/physicar_transfer/physicar_perception_v3_path_lidar_transfer_20260824
sha256sum -c MANIFEST.sha256
```

모든 파일이 `OK`인지 확인합니다.

## 3. package 복사

대상 workspace를 `~/physicar_ws`라고 가정합니다. 기존에 같은 이름의 package가
있으면 덮어쓰지 말고 먼저 비교하십시오.

```bash
mkdir -p ~/physicar_ws/src/MinSeok
for package in \
  physicar_camera_tf_correction \
  physicar_track_perception_v2 \
  physicar_track_perception_v3
do
  if [ -e "$HOME/physicar_ws/src/MinSeok/$package" ]; then
    echo "STOP: existing package: $package"
    exit 1
  fi
done

cp -a src/MinSeok/physicar_camera_tf_correction ~/physicar_ws/src/MinSeok/
cp -a src/MinSeok/physicar_track_perception_v2 ~/physicar_ws/src/MinSeok/
cp -a src/MinSeok/physicar_track_perception_v3 ~/physicar_ws/src/MinSeok/
```

## 4. ROS 2 build

대상 환경의 ROS 2와 PhysiCar underlay를 먼저 source합니다. 아래 경로는 기준
설치 예시이므로 상대 simulator 설치 경로가 다르면 바꾸십시오.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
cd ~/physicar_ws

colcon build --symlink-install --packages-select \
  physicar_camera_tf_correction \
  physicar_track_perception_v2 \
  physicar_track_perception_v3

source install/setup.bash
```

선택적으로 `rosdep`를 실행하려면 대상 시스템 정책과 설치 권한을 먼저
확인하십시오. 이 안내는 `sudo`나 system package 변경을 자동 실행하지 않습니다.

## 5. 비ROS 회귀 테스트

```bash
cd ~/physicar_ws
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/MinSeok/physicar_track_perception_v3:\
src/MinSeok/physicar_track_perception_v2 \
python3 -m pytest -q -p no:cacheprovider \
  src/MinSeok/physicar_track_perception_v3/test
```

제작 시점의 이식본 결과는 `22 passed`입니다. 이는 NON-ROS TEST 결과이며
Gazebo runtime이나 closed-loop driving 검증을 의미하지 않습니다.

## 6. simulator와 perception 실행

터미널 A에서 팀원이 평소 사용하는 simulator를 먼저 실행합니다. 기준 설치의
예시는 다음과 같지만 실제 launch entry는 상대 환경에서 확인해야 합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
ros2 launch physicar_bringup sim.launch.py
```

터미널 B에서 압축에 포함된 사전 점검을 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
source ~/physicar_ws/install/setup.bash
cd ~/physicar_transfer/physicar_perception_v3_path_lidar_transfer_20260824
./scripts/preflight_runtime.sh
```

터미널 C에서 corrected camera TF와 V3를 함께 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
source ~/physicar_ws/install/setup.bash
ros2 launch physicar_track_perception_v3 perception_v3.launch.py
```

입력 topic 이름이 다르면 다음처럼 remap합니다.

```bash
ros2 launch physicar_track_perception_v3 perception_v3.launch.py \
  camera_topic:=/other/camera/image_raw \
  joint_states_topic:=/other/joint_states \
  scan_topic:=/other/scan
```

simulation clock을 쓰지 않는 환경만 `use_sim_time:=false`를 추가합니다.

## 7. 결과 확인

주요 출력은 다음과 같습니다.

| 출력 | type | 의미 |
|---|---|---|
| `/perception_v3/path` | `nav_msgs/msg/Path` | `base_footprint` metric path |
| `/perception_v3/debug/path_valid` | `std_msgs/msg/Bool` | 현재 path 유효성 |
| `/perception_v3/debug/path_source` | `std_msgs/msg/String` | path provenance |
| `/perception_v3/debug/bev` | `sensor_msgs/msg/Image` | camera metric BEV |
| `/perception_v3/debug/path_overlay` | `sensor_msgs/msg/Image` | camera path/role overlay |
| `/perception_v3/debug/bev_lidar_overlay` | `sensor_msgs/msg/Image` | camera BEV + LiDAR |
| `/perception_v3/debug/path_lidar_overlay` | `sensor_msgs/msg/Image` | path + LiDAR 통합 화면 |
| `/perception_v3/debug/lidar_diagnostics` | `std_msgs/msg/String` | pairing/TF JSON 진단 |

통합 화면은 다음 명령으로 선택해 확인합니다.

```bash
ros2 run rqt_image_view rqt_image_view \
  /perception_v3/debug/path_lidar_overlay
```

`rqt_image_view`가 topic argument를 받지 않는 환경이면 실행 후 GUI의 topic
목록에서 `/perception_v3/debug/path_lidar_overlay`를 선택하십시오.

추가 확인 명령:

```bash
ros2 topic hz /perception_v3/path
ros2 topic echo /perception_v3/debug/path_valid --once
ros2 topic echo /perception_v3/debug/path_source --once
ros2 topic echo /perception_v3/debug/lidar_diagnostics --once
ros2 topic list | grep '^/avoidance_v3/'
```

마지막 명령은 아무것도 출력하지 않는 것이 이 이식본의 정상 상태입니다.

최소 runtime 확인 항목은 다음과 같습니다.

1. `camera_optical_frame_corrected` TF가 image timestamp에 제공된다.
2. `/perception_v3/debug/bev`가 지속 갱신된다.
3. path가 있을 때 `path_valid=true`이고 `/perception_v3/path`에 pose가 있다.
4. LiDAR diagnostics의 `tf_success`가 반복적으로 `true`이다.
5. 통합 화면에서 path와 LiDAR point가 같은 시각·방향·거리로 겹친다.
6. `/avoidance_v3/**` topic과 흰색 회피 경로가 생성되지 않는다.

대상 simulator에서 위 항목을 실제 확인하기 전에는 GAZEBO RUNTIME VERIFIED가
아니며, controller가 포함되지 않았으므로 CLOSED-LOOP DRIVING VERIFIED도
아닙니다.
