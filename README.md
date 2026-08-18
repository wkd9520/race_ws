# physicar_race — 2차선 코스 레이스 스택

2026 AMET 자율주행 해커톤용 독립 워크스페이스. 단일 패키지 `physicar_race`
하나에 인지·판단 노드 4개가 들어있다.

## 코스 조건과 우선순위

| 조건 | 성격 |
|---|---|
| 바깥 경계 = 흰색 실선, 넘으면 **실격** | HARD 제약 |
| 중앙선 = 노란 점선, 2차선 | 주행 구조 |
| 2차선 중 랜덤 위치에 장애물 | 회피 대상 |
| 출발 시 신호등: 빨강 정지 / 초록 출발 | 1회성 게이트 |
| 최단 기록이 우승 | 목적 함수 |

```
실격 회피(흰선) > 충돌 회피 > 출발 게이트 > 기록 단축
```

흰선이 장애물보다 위에 있는 게 핵심이다. 장애물을 피하려다 흰선을 넘으면
사고가 아니라 실격이므로, 회피 조향은 언제나 흰선 여유의 제약을 받는다.

## 구조

```
race_ws/
├── src/physicar_race/
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/physicar_race
│   ├── launch/race_launch.py
│   └── physicar_race/
│       ├── __init__.py
│       ├── lane_detect_node.py      흰 실선 + 노란 점선 이중 마스크
│       ├── traffic_light_node.py    출발 신호등 적/녹 판정
│       ├── lane_obstacle_node.py    차선 단위 장애물 점유 판정
│       └── race_judgment_node.py    레이스 상태기계 -> /speed, /steering
├── test/                            ROS 없이 도는 검증
└── deploy/myapp.sh                  웹 UI(:5000) 업로드용
```

## 데이터 흐름

```
image_raw ──┬─> lane_detect_node    ──> lane/*
            └─> traffic_light_node  ──> traffic/*
                                                 ├─> race_judgment_node ──> /speed
scan ─────────> lane_obstacle_node  ──> obstacle/*┘                        /steering
                       ↑
                  lane/current_lane
```

`lane_obstacle_node` 가 `lane/current_lane` 을 되받는 게 유일한 순환이다.
"반대 차선이 좌측인가 우측인가"를 알아야 점유 판정을 할 수 있기 때문이다.

## 빌드와 실행

```bash
cd ~/race_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch physicar_race race_launch.py
```

등록 확인:

```bash
ros2 pkg executables physicar_race
# lane_detect_node / traffic_light_node / lane_obstacle_node / race_judgment_node
```

자주 쓰는 인자:

```bash
# 신호등 없는 맵 (이거 안 하면 영원히 출발 안 함)
ros2 launch physicar_race race_launch.py require_green:=false

# 토픽 이름이 다를 때 (ros2 topic list -t 로 먼저 확인)
ros2 launch physicar_race race_launch.py image_topic:=/image_raw

# HSV 튜닝
ros2 launch physicar_race race_launch.py publish_debug:=true
```

수동 출발 (신호등 게이트 건너뛰기):

```bash
ros2 topic pub --once /race/start std_msgs/msg/Bool "{data: true}"
```

## 토픽 계약

### lane_detect_node

| 토픽 | 타입 | 설명 |
|---|---|---|
| `lane/valid` | Bool | 흰선을 하나라도 봤는가 |
| `lane/offset_right` | Float64 | 오른쪽 차선 주행 시 정규화 횡오차, + = 차선중심보다 우측 |
| `lane/offset_left` | Float64 | 왼쪽 차선 주행 시 정규화 횡오차 |
| `lane/current_lane` | Int32 | 0=RIGHT, 1=LEFT, -1=UNKNOWN |
| `lane/margin_left` | Float64 | 좌측 흰선까지 정규화 거리, **음수 = 이미 넘음** |
| `lane/margin_right` | Float64 | 우측 흰선까지 정규화 거리 |
| `lane/curvature` | Float64 | 정규화 곡률, + = 우커브 |

### traffic_light_node

| 토픽 | 타입 | 설명 |
|---|---|---|
| `traffic/light_state` | String | `RED` / `GREEN` / `NONE` |
| `traffic/valid` | Bool | 카메라 살아있음 |

`valid=False`(카메라 사망)와 `NONE`(카메라는 살아있는데 신호등이 안 보임)은
반드시 구분된다. 전자만 정지 사유이고, 후자는 출발 후 정상 상태다.

### lane_obstacle_node

| 토픽 | 타입 | 설명 |
|---|---|---|
| `obstacle/blocked_current` | Bool | 현재 차선 전방 막힘 |
| `obstacle/blocked_other` | Bool | 반대 차선 전방 막힘 |
| `obstacle/emergency` | Bool | 코앞 장애물, 즉시 정지 |
| `obstacle/nearest_dist` | Float64 | 전방 최근접 거리 [m] |

### race_judgment_node

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/speed` | Float64 | 목표 속도 [m/s] |
| `/steering` | Float64 | 목표 조향각 [rad], Ackermann |
| `race/state` | String | `WAIT_GREEN` / `RACING` / `EMERGENCY` |
| `race/start` | Bool (구독) | 수동 출발 오버라이드 |

## 규약 두 가지

지키지 않으면 **에러 없이 조용히 동작하지 않는다.**

- 센서 구독은 **상대 토픽명**(`image_raw`, `scan`) + launch remapping
- 센서 구독 QoS 는 **`qos_profile_sensor_data`**(BEST_EFFORT).
  기본 QoS(RELIABLE)로 두면 BEST_EFFORT 퍼블리셔와 매칭되지 않아
  프레임이 한 장도 안 들어오는데 에러도 안 난다.

## 설계 판단 셋

**흰색/노란색 이중 마스크.** HSV 임계값 한 세트로는 흰 실선과 노란 점선의
의미 차이(실격 경계 vs 넘어도 되는 구분자)를 표현할 수 없다. 노란선은 점선이라
대시 공백에서 사라지므로 `yellow_hold_s` 동안 마지막 위치를 홀드하고, 그마저
만료되면 두 흰선 사이를 하나의 통로로 보고 차선 구분만 포기한다(주행은 계속).

**반응형 회피가 아니라 이산적 차선 변경.** 장애물이 차선 단위로 배치되므로
연속 스웨브는 흰선을 밟을 위험만 키우고 기록도 손해다. 장애물 노드는 조향을
만들지 않고 어느 차선이 막혔는지만 보고하며, 변경 결정은 판단 노드가 쿨다운과
타임아웃을 걸어 내린다. 회피 후 원래 차선으로 돌아오지 않는 것도 의도적이다 —
어느 차선이든 기록은 같고, 불필요한 차선 변경은 위험만 늘린다.

**출발 게이트는 1회 래치.** 신호등은 출발 시점에만 존재하므로, 한 번 초록을
확인해 출발하면 그 뒤로는 신호등 인지 결과를 아예 보지 않는다. 그래야 주행 중
빨간 물체(다른 차, 표지, 관중 옷)를 신호등으로 오인해 트랙 한복판에 서는 사고가
없다. 페널티가 "빨간불 출발"에만 붙으므로 늦게 출발하는 것보다 일찍 출발하는
게 훨씬 비싸다.

## 캘리브레이션

### 지금 (코스 규격 공개 직후)

| 항목 | 파라미터 | 방법 |
|---|---|---|
| 차선 폭 | `lane_width_m` | **코스 규격의 실제 값으로 교체.** 기본 0.50 은 추정치이고, 틀리면 반대 차선 점유 판정이 통째로 어긋난다 |
| 흰선 HSV | `white_s_max`, `white_v_min` | 코스 영상으로 실측 |
| 노란선 HSV | `yellow_h_min/max`, `yellow_s_min`, `yellow_v_min` | 위와 동일 |
| 점선 홀드 시간 | `yellow_hold_s` | 대시 길이·간격 / 예상 속도로 계산 |

### 실차 인수 후

| 항목 | 파라미터 | 비고 |
|---|---|---|
| 라이다 장착각 | `front_offset_deg` | 시뮬레이터는 보통 0, 연습 섀시는 180. 재측정 필수 |
| 조향 부호 | `lane_steer_sign` | 반대로 돌면 -1.0 |
| 카메라 광축 오프셋 | `cam_center_offset_px` | 직진 주행시켜 실측 |
| ROI 비율 | `roi_top_frac` 등 | 카메라 화각이 다르면 재조정 |
| 신호등 ROI | `roi_top_frac`, `roi_bottom_frac` | 출발선에 세워놓고 debug_image 로 확정 |
| 최고 속도 | `v_max` | 1.5 에서 시작해 단계적으로 3.0 까지 |

## 테스트

ROS나 하드웨어 없이 노트북에서 바로 돌아간다 (`test/ros_stubs.py` 가 rclpy 대체).
의존성은 `numpy`, `opencv-python` 뿐이다.

```bash
python3 test/test_race_logic.py
```

합성 도로 영상으로 차선 기하를, 합성 신호등 영상으로 적/녹 판정을, 합성
LaserScan 으로 차선 점유를, 가짜 인지 입력으로 레이스 상태기계를 검증한다.
파라미터를 바꿨을 때 회귀 확인용으로 쓸 것.

## 디버깅

```bash
ros2 launch physicar_race race_launch.py publish_debug:=true
ros2 run rqt_image_view rqt_image_view   # lane/debug_image, traffic/debug_image
ros2 topic echo /race/state
ros2 topic echo /lane/current_lane       # 계속 -1 이면 노란색 HSV 가 안 맞는 것
```

### 차가 출발하지 않을 때

`/race/state` 가 `WAIT_GREEN` 에서 안 넘어가면 신호등을 못 본 것이다.
원인을 두 줄로 가른다:

```bash
ros2 topic echo /traffic/valid
ros2 topic echo /traffic/light_state
```

| valid | light_state | 원인 |
|---|---|---|
| `false` | — | 카메라가 안 들어옴. `image_topic` 확인 |
| `true` | `NONE` | 색 검출 실패. 아래 probe 로 실측값 확인 |
| `true` | `RED` | 초록을 빨강으로 오인 |

### 신호등 HSV 실측 (probe)

임계값을 눈대중으로 돌리지 말고, 화면에 실제로 무슨 색이 보이는지 찍어본다:

```bash
ros2 launch physicar_race race_launch.py debug_probe:=true
```

1초에 한 번 이런 로그가 나온다:

```
[probe] 밝은 영역 (현재 기준: ROI y 0.00~0.55, sat_min=120, val_min=120, green H 40~90, min_blob_px=60)
  area=4900   H=60  S=75  V=255  위치 x=0.52 y=0.20
```

읽는 법 — 현재 기준과 실측값을 비교하면 원인이 바로 나온다:

| 증상 | 원인 | 조치 |
|---|---|---|
| `y` 가 ROI 범위 밖 | 신호등이 안 보이는 영역에 있음 | `tl_roi_bottom_frac:=1.0` |
| `S` 가 `sat_min` 미만 | LED 가운데가 하얗게 떠서 채도가 낮음 | `tl_sat_min:=60` |
| `V` 가 `val_min` 미만 | 화면이 어두움 | `tl_val_min:=80` |
| `H` 가 40~90 밖 | 초록 색상 범위가 안 맞음 | `tl_green_h_min/max` 조정 |
| `area` 가 `min_blob_px` 미만 | 신호등이 너무 작게 잡힘 | `tl_min_blob_px:=20` |

위 예시(`S=75` < `sat_min=120`)면 이렇게 하면 잡힌다:

```bash
ros2 launch physicar_race race_launch.py tl_sat_min:=60
```

값을 찾았으면 `deploy/myapp.sh` 의 `LAUNCH_ARGS` 에 넣어 고정한다.

**주의**: 파라미터는 노드 생성 때 한 번만 읽는다. `ros2 param set` 으로 바꿔도
안 먹으니 반드시 실행할 때 넘길 것.

### 일단 주행부터 보고 싶으면

```bash
ros2 launch physicar_race race_launch.py require_green:=false
```

이미 띄운 상태면:

```bash
ros2 topic pub --once /race/start std_msgs/msg/Bool "{data: true}"
```

## 아직 안 만든 것

- **결승선 감지 없음** — 완주 판정/정지가 없다.
- **IPM(bird's eye) 미적용** — 곡률 추정이 화면 좌표 기반 근사다.
- **흰선을 이미 넘은 상태의 복귀 전략** — 일반 조향 항이 밀어내는 것에 의존.
- **실측 검증 전무** — 위 테스트는 전부 합성 입력이다. 파라미터가 맞는지는
  실제 트랙에서만 알 수 있다.
