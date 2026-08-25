#!/bin/bash
# 웹 UI(:5000) MyApp 패널에 업로드하는 스크립트.
# physicar-myapp.service 가 이 파일을 실행한다 (실패 시 자동 재시작).
#
# BOOTSTRAP=1 이라 이 스크립트가 저장소를 스스로 clone/최신화하고, 커밋이
# 바뀐 경우에만 colcon build 한 뒤 스택을 띄운다. 대상 머신에 SSH 로 못
# 들어가도 이 파일 하나만 올리면 코드가 따라온다.
#
# 전제: 대상 머신에 git 과 인터넷 접근이 있을 것. 저장소가 public 이라
# 인증은 필요 없다. 이미 워크스페이스를 직접 넣어뒀다면 BOOTSTRAP=0 으로
# 바꾸면 clone/build 를 건너뛰고 실행만 한다.

set -u

# ===================== 설정 =====================
WS="$HOME/race_ws"              # 워크스페이스 경로 (없으면 clone 이 여기에 만든다)
BOOTSTRAP=1                     # 1 이면 아래 REPO/BRANCH 에서 clone + build
REPO="https://github.com/wkd9520/race_ws.git"
BRANCH="main"
ROS_DISTRO_SETUP="/opt/ros/jazzy/setup.bash"
PHYSICAR_SETUP="/opt/physicar/install/setup.bash"   # 없으면 건너뜀

# 실행할 launch 와 인자. 토픽 이름/신호등 유무/코스 치수는 여기만 고치면 된다.
LAUNCH_PKG="physicar_race"
LAUNCH_FILE="perception_v3_race_launch.py"
LAUNCH_ARGS=(
  "camera_topic:=/camera/image_raw"
  "scan_topic:=/scan"
  "v_max:=1.2"               # 안정되면 올릴 것
  "open_rqt:=false"          # 무인 실행이므로 GUI 는 끈다
)

log() { echo "[myapp] $*"; }

# systemd 가 실패 시 즉시 재시작하므로, 그냥 exit 하면 무한 반복한다.
die() { log "오류: $*"; sleep 15; exit 1; }

# ===================== ROS 환경 =====================
[ -f "$ROS_DISTRO_SETUP" ] || die "ROS 환경 없음 -> $ROS_DISTRO_SETUP"
# shellcheck disable=SC1090
source "$ROS_DISTRO_SETUP"

if [ -f "$PHYSICAR_SETUP" ]; then
  # shellcheck disable=SC1090
  source "$PHYSICAR_SETUP"
else
  log "경고: $PHYSICAR_SETUP 없음 - 건너뜀"
fi

# ===================== 소스 가져오기 (선택) =====================
if [ "$BOOTSTRAP" -eq 1 ]; then
  [ -n "$REPO" ] || die "BOOTSTRAP=1 인데 REPO 가 비어 있다"
  command -v git >/dev/null 2>&1 || die "git 이 없다"

  if [ ! -d "$WS/.git" ]; then
    log "저장소 클론: $REPO ($BRANCH) -> $WS"
    git clone --depth 1 --branch "$BRANCH" "$REPO" "$WS" || die "clone 실패"
  else
    log "저장소 최신화"
    # 오프라인이어도 기존 소스로 계속 간다.
    if git -C "$WS" fetch --depth 1 origin "$BRANCH"; then
      git -C "$WS" reset --hard "origin/$BRANCH" || log "경고: reset 실패, 기존 소스 사용"
    else
      log "경고: fetch 실패 (오프라인?), 기존 소스 사용"
    fi
  fi

  command -v colcon >/dev/null 2>&1 || die "colcon 이 없다"
  STAMP="$WS/.myapp_build_stamp"
  HEAD_SHA="$(git -C "$WS" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ ! -f "$WS/install/setup.bash" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$HEAD_SHA" ]; then
    log "빌드 시작 (커밋 $HEAD_SHA)"
    cd "$WS" || die "cd 실패: $WS"
    colcon build --symlink-install || die "colcon build 실패"
    echo "$HEAD_SHA" > "$STAMP"
    log "빌드 완료"
  else
    log "빌드 생략 (이미 빌드됨)"
  fi
fi

# ===================== 실행 =====================
[ -f "$WS/install/setup.bash" ] \
  || die "빌드 산출물 없음 -> $WS/install/setup.bash (colcon build 를 먼저 할 것)"
# shellcheck disable=SC1090
source "$WS/install/setup.bash"

# 랩실 네트워크의 다른 프로젝트와 토픽이 섞이지 않도록 팀 전원 동일하게 42.
export ROS_DOMAIN_ID=42

log "실행: ros2 launch $LAUNCH_PKG $LAUNCH_FILE ${LAUNCH_ARGS[*]}"
exec ros2 launch "$LAUNCH_PKG" "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}"
