#!/bin/bash
# cpu-train.sh — 미니PC CPU 로 JudgeModel 학습 스크립트를 띄우고 감시한다 (설계 3-3 · 2순위 길).
#
#   run/cpu-train.sh [--name NAME] <src 기준 스크립트> [스크립트 인자...]
#   예) run/cpu-train.sh --name base-cpu train/train_nli.py --device cpu --dtype fp32 --batch 16 --accum 2 ...
#   넷 칸 연기 판 (판정 넷 확장) : run/cpu-train.sh --name smoke4 train/train_nli.py --smoke --grow-head 4 --out runs/smoke4
#   넷 칸 덧학습은 podman-train.sh 머리 주석의 r2-01 줄에서 --device cpu --dtype fp32 로 바꿔 쓴다.
#
# 파이썬은 JT_VENV(기본 $JT_ROOT/venv · 기계별 값은 env.sh)의 것을 쓴다 — 새로 깔지 않는다. 작업 폴더는 JT_ROOT.
# 메모리 상한 : systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 nice -n 19 (설계 3-3).
# 감시 규칙은 run/watch-lib.sh (podman-train.sh 와 같다).
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/watch-lib.sh"

VENV="${JT_VENV:-$JT_ROOT/venv}"
MEM_MAX="${JT_CPU_MEM_MAX:-12G}"

name="cpu-$(date +%Y%m%d-%H%M%S)"
if [ "${1:-}" = "--name" ]; then
  name="$2"; shift 2
fi
if [ $# -lt 1 ]; then
  echo "usage: $0 [--name NAME] <script.py> [args...]" >&2
  exit 2
fi
script="$1"; shift
if [ ! -f "$JT_ROOT/src/$script" ]; then
  echo "no script: $JT_ROOT/src/$script" >&2
  exit 2
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "no venv python: $VENV/bin/python" >&2
  exit 2
fi

mkdir -p "$JT_ROOT/logs"
log="$JT_ROOT/logs/$name.log"
hb="$JT_ROOT/logs/$name.heartbeat"
stop_file="$JT_ROOT/logs/$name.stop"
rm -f "$hb" "$stop_file"

watch_init
log_line "start name=$name script=$script args=$* venv=$VENV mem=$MEM_MAX game=$game0 | $(mem_line)"

cd "$JT_ROOT" || exit 2
systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
  nice -n 19 env HF_HOME="$JT_ROOT/hf" JT_HEARTBEAT="$hb" JT_STOP_FILE="$stop_file" JT_IMAGE=venv \
  "$VENV/bin/python" -I "$JT_ROOT/src/$script" "$@" >> "$log" 2>&1 &
pid=$!

kill_proc() {
  stop_reason="$1"
  log_line "kill: $stop_reason | $(mem_line)"
  kill -TERM "$pid" 2>/dev/null
  sleep 10
  kill -KILL "$pid" 2>/dev/null
}

watch_loop "$pid" "$hb" "$stop_file" kill_proc
wait "$pid"
code=$?
[ -z "$stop_reason" ] && stop_reason=$(exit_reason "$code")
log_line "end exit=$code stop_reason=$stop_reason | $(mem_line)"
echo "log=$log exit=$code stop_reason=$stop_reason"
[ "$stop_reason" = done ] || exit 1
