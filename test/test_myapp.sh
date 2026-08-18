#!/bin/bash
# myapp.sh 의 분기들을 가짜 환경에서 실제로 실행해 검증한다.
set -u

REPO_SH="$1"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

fails=0
chk() {
  local label="$1" cond="$2" detail="${3:-}"
  if [ "$cond" = "1" ]; then echo "  [PASS] $label $detail"
  else echo "  [FAIL] $label $detail"; fails=$((fails+1)); fi
}

# --- 가짜 환경 ---
mkdir -p "$T/ros" "$T/ws/src" "$T/ws/install" "$T/bin"
echo 'export FAKE_ROS=1' > "$T/ros/setup.bash"
echo 'export FAKE_WS=1'  > "$T/ws/install/setup.bash"

cat > "$T/bin/ros2" <<'EOF'
#!/bin/bash
echo "ROS2_CALLED: $*"
EOF
cat > "$T/bin/colcon" <<'EOF'
#!/bin/bash
echo "COLCON_CALLED: $*"
EOF
chmod +x "$T/bin/ros2" "$T/bin/colcon"

# 설정만 가짜로 바꾼 사본을 만든다
mk() {  # mk <out> <bootstrap> <ros_setup>
  sed -e "s|^ROS_DISTRO_SETUP=.*|ROS_DISTRO_SETUP=\"$3\"|" \
      -e "s|^PHYSICAR_SETUP=.*|PHYSICAR_SETUP=\"$T/nonexistent.bash\"|" \
      -e "s|^WS=.*|WS=\"$T/ws\"|" \
      -e "s|^BOOTSTRAP=.*|BOOTSTRAP=$2|" \
      -e "s|sleep 15|sleep 0|" \
      "$REPO_SH" > "$1"
  chmod +x "$1"
}

echo "[myapp.sh 분기 검증]"

# 1) ROS 환경 없음 -> 명확한 오류 + exit 1
mk "$T/a.sh" 0 "$T/no_such_ros.bash"
out="$(bash "$T/a.sh" 2>&1)"; rc=$?
chk "ROS 없음 -> exit 1" "$([ $rc -eq 1 ] && echo 1 || echo 0)" "(rc=$rc)"
chk "ROS 없음 -> 원인 출력" \
    "$(echo "$out" | grep -q 'ROS 환경 없음' && echo 1 || echo 0)"

# 2) BOOTSTRAP=0, 빌드된 WS 있음 -> ros2 launch 까지 도달
mk "$T/b.sh" 0 "$T/ros/setup.bash"
out="$(PATH="$T/bin:$PATH" bash "$T/b.sh" 2>&1)"; rc=$?
chk "정상 경로 -> exit 0" "$([ $rc -eq 0 ] && echo 1 || echo 0)" "(rc=$rc)"
chk "ros2 launch 호출됨" \
    "$(echo "$out" | grep -q 'ROS2_CALLED: launch physicar_race race_launch.py' && echo 1 || echo 0)"
chk "launch 인자 전달됨" \
    "$(echo "$out" | grep -q 'require_green:=true' && echo 1 || echo 0)"
chk "colcon 안 부름 (BOOTSTRAP=0)" \
    "$(echo "$out" | grep -q 'COLCON_CALLED' && echo 0 || echo 1)"
chk "PHYSICAR_SETUP 없어도 계속 진행" \
    "$(echo "$out" | grep -q '경고' && echo 1 || echo 0)"

# 3) BOOTSTRAP=0 인데 빌드 산출물 없음 -> 명확한 오류
mk "$T/c.sh" 0 "$T/ros/setup.bash"
mv "$T/ws/install/setup.bash" "$T/ws/install/setup.bash.bak"
out="$(PATH="$T/bin:$PATH" bash "$T/c.sh" 2>&1)"; rc=$?
chk "빌드 산출물 없음 -> exit 1" "$([ $rc -eq 1 ] && echo 1 || echo 0)" "(rc=$rc)"
chk "빌드 산출물 없음 -> 원인 출력" \
    "$(echo "$out" | grep -q '빌드 산출물 없음' && echo 1 || echo 0)"
mv "$T/ws/install/setup.bash.bak" "$T/ws/install/setup.bash"

# 4) BOOTSTRAP=1 + git 없음 -> 명확한 오류
# /usr/bin 에는 sleep 등 coreutils 는 있고 git 은 없다(git 은 /mingw64/bin).
# 즉 이 PATH 면 스크립트는 정상 동작하되 git 만 사라진 상황이 된다.
mk "$T/d.sh" 1 "$T/ros/setup.bash"
out="$(PATH="$T/bin:/usr/bin" bash "$T/d.sh" 2>&1)"; rc=$?
chk "git 없음 -> exit 1" "$([ $rc -eq 1 ] && echo 1 || echo 0)" "(rc=$rc)"
chk "git 없음 -> 원인 출력" \
    "$(echo "$out" | grep -q 'git 이 없다' && echo 1 || echo 0)"

echo
if [ $fails -gt 0 ]; then echo "실패 ${fails}건"; exit 1; fi
echo "myapp.sh 분기 전부 통과"
