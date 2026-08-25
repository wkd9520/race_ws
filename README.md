# race_ws — 2026 AMET 자율주행 (PhysiCar)

주행 스택 하나만 남긴 상태다. 인지는 팀원 MinSeok 님 패키지를 **수정 없이**
쓰고, 그 뒤에 제어와 고깔 회피를 붙였다.

상세 설계·근거·실차 기록은 Obsidian `자이카2026/` 폴더에 있다.
처음 보는 사람은 `자이카2026/인수인계.md` 부터.

---

## 구조

```
race_ws/
├── scripts/preflight_runtime.sh      대상 시뮬레이터 호환 사전 점검
├── deploy/myapp.sh                   자체 부트스트랩 실행 (clone/build/launch)
├── src/
│   ├── MinSeok/                      ★ 수정 금지 (sha256 원본 일치)
│   │   ├── physicar_camera_tf_correction/    카메라 틸트 → 보정 TF
│   │   ├── physicar_track_perception_v2/     metric-BEV 전처리
│   │   └── physicar_track_perception_v3/     주황 중앙선 → 경로, LiDAR 오버레이
│   └── physicar_race/
│       ├── launch/
│       │   ├── perception_v3_race_launch.py   ★ 주력
│       │   └── hsv_tuner_launch.py            HSV 실측 도구
│       └── physicar_race/
│           ├── cone_bev_node.py               초록 고깔 → 미터 좌표
│           ├── perception_v3_follow_node.py   순수추종 + 고깔 회피
│           ├── race_overlay_node.py           디버그 오버레이 합성
│           └── hsv_tuner_node.py              슬라이더로 HSV 맞추기
└── test/                             ROS 없이 도는 자체 테스트 6개
```

---

## 데이터 흐름

```
/camera/image_raw ─┐
/joint_states ─────┼─> [MinSeok] camera_corrected_tf_broadcaster ─> TF
/scan ─────────────┘                    │
                                        v
                    [MinSeok] physicar_track_perception_v3
                       │            │              │
        /perception_v3/path    debug/bev      debug/path_overlay
                       │      + white_mask          │
                       │            │               │
                       │            v               │
                       │      cone_bev_node ─ /cones┤
                       │            │               │
                       v            v               v
              perception_v3_follow_node ──> race_overlay_node
                       │                          │
                /speed, /steering        /race/debug/path_overlay
```

- MinSeok 님 노드는 **경로까지만** 만든다 (본인 문서에 `controller integration
  are deferred` 명시). 그 뒤가 우리 몫이다.
- 기능을 붙일 때는 항상 **그가 이미 발행하는 토픽을 재사용**한다. 그래서
  `cone_bev_node` 가 IPM 을 복제하지 않고 `debug/bev` 를 받아 쓴다.

---

## 실행

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
source ~/physicar_ws/성규/install/setup.bash

# 사전 점검 (시뮬레이터 먼저 켜고)
bash scripts/preflight_runtime.sh

# 주행
ros2 launch physicar_race perception_v3_race_launch.py
```

디버그 시각화는 기본이 꺼져 있다. `debug_view:=true` 로 켜면
오버레이 합성과 `rqt_image_view` 가 같이 뜬다.

카메라 틸트는 이 launch 가 건드리지 않는다. V2 요구사항상 `-0.5236 rad`
(-30도)여야 하지만, 시뮬레이터가 이미 잡고 있으면 둘이 동시에 보내 값이
번갈아 들어간다. 필요하면 별도 터미널에서:

```bash
ros2 topic echo /joint_states --once | grep -A3 camera_tilt   # 현재 값
ros2 topic pub -r 10 /camera/tilt std_msgs/msg/Float64 "{data: -0.5235987756}"
```

| 자주 쓰는 인자 | 기본 | |
|---|---|---|
| `v_max` | 1.20 | 최고 속도 |
| `a_lat_max` | 3.0 | 횡가속 한계. 코너에서 밀리면 2.0 |
| `steer_sign` | 1.0 | 반대로 꺾이면 −1.0 |
| `avoid_enabled` | true | 고깔 회피 |
| `green_h_min/max` | 40 / 85 | 초록 HSV (**추정치, 실측 필요**) |
| `track_half_m` | 0.37 | 트랙 반폭 — 회피 안전 울타리 |
| `debug_view` | **false** | 오버레이·rqt. 실차는 꺼둔다 |

빌드는 워크스페이스 루트에서:

```bash
cd ~/physicar_ws/성규
colcon build --symlink-install
```

---

## 디버그

| 토픽 | |
|---|---|
| `/race/debug/path_overlay` | **주력.** 인지 + 우리 주행선 + 고깔 |
| `/cones/debug_image` | 초록 마스크와 고깔 검출 |
| `/perception_v3/debug/orange_mask` | 주황 중앙선 (초록 경로 안 나올 때) |
| `/perception_v3/debug/bev` | 펴진 BEV |

```bash
ros2 run rqt_image_view rqt_image_view
ros2 topic echo /perception_v3/debug/path_source --once   # DIRECT_CENTER_OBSERVED 기대
ros2 topic info /speed --verbose | grep -c "Node name"    # 2 이상이면 노드 충돌
```

HSV 를 실측하려면:

```bash
ros2 launch physicar_race hsv_tuner_launch.py
```

---

## 테스트 (Windows, ROS 없이)

```bash
for t in test/test_*.py; do python "$t"; done
bash test/test_myapp.sh deploy/myapp.sh
```

`test/ros_stubs.py` 가 rclpy·cv_bridge·메시지를 흉내 낸다.

> **구조 검사용이다.** 손계산되는 것(순수추종 공식, 횡가속 한계, 픽셀↔미터),
> 부호, 포화, 상태기계만 본다. **파라미터 값은 실차 결과로만 판단한다** —
> 합성 테스트가 통과하고 실차에서 실패한 일이 여러 번 있었다.

---

## 정리하면서 뺀 것

주행 스택이 넷이었는데 실제로 쓰는 하나만 남겼다. 전부 git 에 남아 있다.

| 뺀 것 | 되살리려면 |
|---|---|
| `los_drive_node` + `los_launch.py` | `git checkout 7f35d12 -- <경로>` |
| `centroid_follow_node` | 〃 |
| `bev_lane_node`, `bev_drive_node`, `bev_launch.py` | 〃 |
| `lane_detect_node`, `race_judgment_node`, `lane_obstacle_node`, `cone_detect_node`, `race_launch.py` | 〃 |
| `simple_drive_node`, `traffic_light_node` | 〃 |
| 위 노드들의 테스트 7개 | 〃 |

**신호등**(`traffic_light_node`)은 코스 규칙에 있으므로 실전 전에 되살려야 한다.
`los_drive_node` 는 perception_v3 가 안 될 때의 대안 스택이었다.
