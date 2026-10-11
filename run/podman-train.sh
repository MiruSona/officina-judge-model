#!/bin/bash
# podman-train.sh — 미니PC GPU 로 JudgeModel 학습 스크립트를 띄우고 감시한다 (설계 3-3 · 8-1).
#
#   run/podman-train.sh [-e KEY=VAL]... [--name NAME] <src 기준 스크립트> [스크립트 인자...]
#   예) run/podman-train.sh -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 --name base-2 \
#           train/train_nli.py --device cuda --dtype bf16 --init hf/xlm-roberta-large ... --out runs/base-2
#   넷 칸 덧학습 (nli-studio-v3, 판정 넷 확장 설계 3-3) — r1-04 처방 그대로에 --grow-head 4 만 더한다.
#   기반은 3칸 runs/base-2/ckpt-best 그대로. train 쌍은 넷 라벨(a·b·c·d·cd) 판, replay 의 src 는 공개 출처 이름이어야
#   중립이 합친 칸 손실(c∪d)로 간다 (로그 「replay 중립 … 중 합친 칸 손실」 줄로 확인) :
#       run/podman-train.sh -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 -e HF_HUB_OFFLINE=1 --name r2-01 \
#           train/train_nli.py --device cuda --dtype bf16 --init runs/base-2/ckpt-best --grow-head 4 --stage studio \
#           --train mem/r2/train.jsonl --replay data/prep/replay-ko-en-r4.jsonl --mix-klue 1:1 --class-weight \
#           --dev mem/r2/dev-r2.jsonl --out runs/r2-01 --epochs 4 --batch 16 --warmup 0.10 --eval-every 0 \
#           --select last --collapse-acc 0 --max-len 320 --lr 1e-5 --seed 1
#
# 자리 : 호스트 ~/judgetool (JT_ROOT) 을 컨테이너 /work 로 붙이고 작업 폴더도 /work 다 — 상대 경로 인자는 JT_ROOT 기준.
# 코드는 /work/src, 로그는 logs/<NAME>.log, 심장 박동 logs/<NAME>.heartbeat, 멈춤 파일 logs/<NAME>.stop.
# 감시 규칙은 run/watch-lib.sh. 게임 서버 이름은 git 밖 $JT_ROOT/env.sh 의 JUDGEMODEL_GAME_PROC 로 받는다.
# -e 값은 로그인 셸이 /etc/profile.d 를 읽은 뒤 env 로 다시 건다. 이미지 profile 이
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 을 export 해서 podman -e 로는 0 이 덮여 버린다 (N1 실측).
# GPU 메모리는 GTT 라 cgroup 이 다 못 잡는다. --memory 는 호스트 RAM 몫 상한이고, 진짜 상한 노릇은 감시가 한다.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/watch-lib.sh"

IMAGE="${JT_IMAGE:-docker.io/kyuz0/amd-strix-halo-comfyui:latest}"
MEM_LIMIT="${JT_MEM_LIMIT:-20g}"

envs=()
name="run-$(date +%Y%m%d-%H%M%S)"
while [ $# -gt 0 ]; do
  case "$1" in
    -e) envs+=("$2"); shift 2 ;;
    --name) name="$2"; shift 2 ;;
    *) break ;;
  esac
done
if [ $# -lt 1 ]; then
  echo "usage: $0 [-e KEY=VAL]... [--name NAME] <script.py> [args...]" >&2
  exit 2
fi
script="$1"; shift
if [ ! -f "$JT_ROOT/src/$script" ]; then
  echo "no script: $JT_ROOT/src/$script" >&2
  exit 2
fi

mkdir -p "$JT_ROOT/logs"
log="$JT_ROOT/logs/$name.log"
hb_host="$JT_ROOT/logs/$name.heartbeat"
stop_host="$JT_ROOT/logs/$name.stop"
container="jt-$name"
rm -f "$hb_host" "$stop_host"
digest=$(podman image inspect --format '{{.Digest}}' "$IMAGE" 2>/dev/null || echo "")

watch_init
log_line "start name=$name script=$script args=$* envs=${envs[*]:-} image=$IMAGE digest=$digest mem=$MEM_LIMIT game=$game0 | $(mem_line)"

podman run --rm --name "$container" \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc=host \
  --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" \
  -v "$JT_ROOT:/work" -w /work -e HF_HOME=/work/hf \
  -e JT_HEARTBEAT="/work/logs/$name.heartbeat" -e JT_STOP_FILE="/work/logs/$name.stop" \
  -e JT_IMAGE="$IMAGE" -e JT_IMAGE_DIGEST="$digest" \
  --entrypoint "" "$IMAGE" \
  bash -lc 'cd /work && exec env "$@"' jt ${envs[@]+"${envs[@]}"} python -I "/work/src/$script" "$@" >> "$log" 2>&1 &
pid=$!

kill_container() {
  stop_reason="$1"
  log_line "kill: $stop_reason | $(mem_line)"
  if timeout 30 podman kill "$container" >/dev/null 2>&1; then
    return
  fi
  # 멎은 GPU 를 더 건드리지 않는다. wait 도 안 돌아올 수 있어 여기서 끝낸다.
  log_line "GPU_STUCK: podman kill 이 30초 안에 안 끝남 — 사람이 재부팅해야 한다"
  echo "log=$log stop_reason=$stop_reason GPU_STUCK"
  exit 3
}

watch_loop "$pid" "$hb_host" "$stop_host" kill_container
wait "$pid"
code=$?
sleep 3
[ -z "$stop_reason" ] && stop_reason=$(exit_reason "$code")
log_line "end exit=$code stop_reason=$stop_reason | $(mem_line)"
echo "log=$log exit=$code stop_reason=$stop_reason"
[ "$stop_reason" = done ] || exit 1
