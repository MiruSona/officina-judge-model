#!/bin/bash
# watch-lib.sh — 감싸개 둘(podman-train.sh · cpu-train.sh)이 같이 쓰는 감시 고리. source 로 읽는다 (설계 3-3).
#
# 기계별 값은 git 밖 "$JT_ROOT/env.sh" 에서 읽는다 (예: WATCH_PROC=<같은 기계에서 도는 다른 GPU 작업의 실행 파일 이름>).
# WATCH_PROC 는 학습과 메모리를 나눠 쓰는 다른 프로세스다. 학습 중에 새로 뜨거나, 떠 있는 동안 가용 메모리가
# 10GB 아래로 내려가면 학습을 멈춘다. 비면 이 검사를 건너뛴다. 스왑 · 심장 박동 검사는 늘 한다.
# 옛 이름 JUDGEMODEL_GAME_PROC 로 준 값도 받는다.
#
# 멈추는 길 둘 :
#   - 스왑 · 다른 GPU 작업(WATCH_PROC) · 메모리 → 멈춤 파일에 까닭을 쓴다. 학습 스크립트가 검사점을 저장하고 스스로 끝낸다.
#     GRACE_SEC 안에 안 끝나면 죽인다 (bench.py 처럼 멈춤 파일을 안 보는 스크립트도 이 길로 끝난다)
#   - 심장 박동 끊김 → 바로 죽인다. 죽이기가 안 끝나면 GPU 멎음으로 보고 사람에게 알린다

JT_ROOT="${JT_ROOT:-$HOME/judgemodel-work}"   # 학습 기계의 작업 뿌리 (src · logs · runs · hf 가 이 아래)
if [ -f "$JT_ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  . "$JT_ROOT/env.sh"
fi
WATCH_PROC="${WATCH_PROC:-${JUDGEMODEL_GAME_PROC:-}}"
STALL_SEC="${JT_STALL_SEC:-60}"
STARTUP_SEC="${JT_STARTUP_SEC:-180}"
SWAP_MAX_MB="${JT_SWAP_MAX_MB:-256}"
AVAIL_MIN_MB="${JT_AVAIL_MIN_MB:-10240}"
GRACE_SEC="${JT_GRACE_SEC:-120}"
POLL_SEC=5

mem_line() {  # 가용 · 스왑 · GTT (MiB)
  local gtt=0 f
  for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
    [ -r "$f" ] && gtt=$((gtt + $(cat "$f") / 1048576))
  done
  free -m | awk -v gtt="$gtt" '/^Mem/{a=$7} /^Swap/{s=$3} END{print "avail=" a " swap=" s " gtt=" gtt}'
}
swap_mb() { free -m | awk '/^Swap/{print $3}'; }
avail_mb() { free -m | awk '/^Mem/{print $7}'; }

# 명령 줄 첫 낱말이 그 실행 파일인 것만 센다 — 그 낱말을 품은 ssh 셸 줄은 안 센다 (N1 함정 4-4).
watch_proc_up() {
  [ -n "$WATCH_PROC" ] || return 1
  pgrep -f "^[^ ]*${WATCH_PROC}" >/dev/null
}

log_line() { echo "[$(date +%T)] $*" >> "$log"; }

# watch_init : 시작 값을 잡는다. log 가 정해진 뒤 부른다.
watch_init() {
  swap0=$(swap_mb)
  proc0=0
  if watch_proc_up; then proc0=1; fi
  stop_reason=""
}

# watch_loop PID 심장박동파일 멈춤파일 죽이는함수
# 끝나면 stop_reason 에 까닭이 남는다 (스스로 끝났으면 빈 값).
watch_loop() {
  local pid=$1 hb=$2 stop_file=$3 kill_fn=$4
  local asked=0 started now age swap_now reason
  started=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    kill -0 "$pid" 2>/dev/null || break
    now=$(date +%s)
    log_line "mon $(mem_line)"
    if [ "$asked" -gt 0 ]; then
      # 검사점 저장 중엔 심장 박동이 멎을 수 있다 — 유예 동안은 끊김을 안 본다
      [ $(( now - asked )) -gt "$GRACE_SEC" ] && { "$kill_fn" "$stop_reason (grace ${GRACE_SEC}s over)"; break; }
      continue
    fi
    if [ -f "$hb" ]; then
      age=$(( now - $(stat -c %Y "$hb") ))
      [ "$age" -gt "$STALL_SEC" ] && { "$kill_fn" "stall (heartbeat ${age}s)"; break; }
    elif [ $(( now - started )) -gt "$STARTUP_SEC" ]; then
      "$kill_fn" "stall (no heartbeat in ${STARTUP_SEC}s)"; break
    fi
    reason=""
    swap_now=$(swap_mb)
    [ $(( swap_now - swap0 )) -gt "$SWAP_MAX_MB" ] && reason="swap (+$(( swap_now - swap0 ))MB)"
    if [ -z "$reason" ] && watch_proc_up; then
      if [ "$proc0" = 0 ]; then
        reason="watch_proc"
      elif [ "$(avail_mb)" -lt "$AVAIL_MIN_MB" ]; then
        reason="memory"
      fi
    fi
    if [ -n "$reason" ]; then
      stop_reason="$reason"
      echo "$reason" > "$stop_file"
      log_line "stop requested: $reason | $(mem_line)"
      asked=$now
    fi
  done
}

# exit_reason 종료코드 : 학습 스크립트 종료 코드 → 까닭 (train_nli.py 의 EXIT_* 와 같은 뜻)
exit_reason() {
  case "$1" in
    0) echo done ;;
    3) echo guard ;;
    4) echo nan ;;
    5) echo collapse ;;
    *) echo error ;;
  esac
}
