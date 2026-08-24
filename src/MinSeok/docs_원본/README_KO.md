# PhysiCar V3 path + LiDAR 통합 오버레이 이식본

이 압축파일은 장애물 회피 경로를 만들거나 흰색으로 출력하는 기능을 제외하고,
카메라 기반 path와 LiDAR point를 같은 확장 BEV에 동시에 표시하는 단계까지를
다른 PhysiCar 계열 시뮬레이터로 옮기기 위한 소스 묶음입니다.

## 포함 범위

- `physicar_track_perception_v3`: path 인식 및 path + LiDAR 통합 오버레이
- `physicar_track_perception_v2`: V3가 import하는 metric-BEV 공통 모듈
- `physicar_camera_tf_correction`: corrected camera TF broadcaster
- `scripts/preflight_runtime.sh`: 대상 simulator의 입력 topic과 TF 사전 점검
- `INSTALL_KO.md`: 설치, build, 실행 및 확인 방법
- `TEAMMATE_MESSAGE_KO.md`: 팀원에게 그대로 전달할 수 있는 안내 문구
- `MANIFEST.sha256`: 압축 내부 파일 무결성 목록

## 의도적으로 제외한 범위

- obstacle avoidance 계산 모듈과 관련 테스트
- 흰색 회피 경로 overlay
- `/avoidance_v3/**` topic 전체
- controller와 closed-loop driving 구성
- `/opt/physicar` 원본, `build/`, `install/`, `log/`, cache 및 runtime log

통합 화면은 `/perception_v3/debug/path_lidar_overlay`에서 확인합니다. 이 화면의
path 색은 기존 source 규칙을 유지하며, LiDAR point는 기본 BGR `[255,255,0]`
(cyan)으로 표시됩니다.

상세 명령과 호환 조건은 `INSTALL_KO.md`를 따르십시오.
