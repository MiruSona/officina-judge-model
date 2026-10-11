#!/bin/bash
# seed3.sh — 한 조건을 씨앗 1·2·3 으로 차례로 학습 · 평가 · 문턱 고르기 (d 자료 층 설계 2-3, 미니PC 에서 돈다).
#
#   run/seed3.sh <조건> -- <train_nli 인자...>
#   예) run/seed3.sh B0 -- --device cuda --dtype bf16 --init runs/base-2/ckpt-best --grow-head 4 --stage studio \
#           --train mem/r2/train-r2.jsonl --replay data/prep/replay-ko-en-r4.jsonl --replay-n 3433 --class-weight \
#           --class-weight-scale b=1.2 --epochs 4 --batch 16 --warmup 0.10 --eval-every 0 --collapse-acc 0 \
#           --max-len 320 --lr 1e-5
#
# 판 이름 g1-<조건>-s<씨앗>. --seed · --out · --select last · --dev(dev-cmp) 는 이 스크립트가 붙인다 (인자에 넣지 않는다).
# 판마다 runs/<판>/eval-<셋>/{pred.jsonl,summary.json} · calibration.json(dev-cal 로) · eval-side.txt 가 생긴다.
# 그다음 measure/compare.py 로 견준다 (measure/README.md).
#
# 환경 변수 (자리는 JT_ROOT=~/judgetool 기준 상대 경로)
#   G1_DEV_CAL   문턱 고르는 셋        (기본 mem/g1/dev-cal-g1.jsonl)
#   G1_DEV_CMP   견주는 셋             (기본 mem/g1/dev-cmp-g1.jsonl)
#   G1_SIDE      곁 자 「이름:파일」 공백 목록 (기본 k48 · mnli-devm2000 · klue-dev)
#   G1_PROTOCOL  규약 파일             (기본 src/measure/protocol-g1.json)
#   G1_SEEDS     씨앗 목록             (기본 "1 2 3" · 같은 씨앗 재실행은 G1_SEEDS=1 G1_SUFFIX=r)
#   G1_SUFFIX    판 이름 꼬리          (기본 없음 → g1-B0-s1r)
#   G1_NEUTRAL   eval --public-neutral (기본 cd — 학습의 --replay-neutral 과 같게)
#   G1_TRAIN_ONLY 1 이면 학습만 하고 평가 · calibrate · 곁 자를 건너뛴다 (기본 0). dev-cal 은 안 찾고,
#                 학습 중 --dev 로 쓸 G1_DEV_CMP 만 있으면 된다
#   G1_EVAL_ONLY  1 이면 학습을 건너뛰고 이미 있는 runs/<판>/ckpt-best 로 평가 · calibrate · 곁 자만 돈다 (기본 0).
#                 G1_TRAIN_ONLY=1 로 학습만 한 판을 나중에 잴 때 쓴다. 학습 인자는 안 받는다 — run/seed3.sh B0 --
#                 (같은 씨앗 재실행 판은 G1_SEEDS=1 G1_SUFFIX=r 를 같이 준다)
set -u

if [ $# -lt 2 ] || [ "$2" != "--" ]; then
  echo "usage: $0 <조건> -- <train_nli 인자...>" >&2
  exit 2
fi
cond="$1"; shift 2
case "$cond" in
  *[!A-Za-z0-9_.-]*|"") echo "조건 이름은 영문·숫자·_.- 만: $cond" >&2; exit 2 ;;
esac
for arg in "$@"; do
  case "$arg" in
    --seed|--out|--select|--dev) echo "$arg 는 seed3.sh 가 붙인다 — 인자에서 뺀다" >&2; exit 2 ;;
  esac
done

JT_ROOT="${JT_ROOT:-$HOME/judgetool}"
DEV_CAL="${G1_DEV_CAL:-mem/g1/dev-cal-g1.jsonl}"
DEV_CMP="${G1_DEV_CMP:-mem/g1/dev-cmp-g1.jsonl}"
SIDE="${G1_SIDE:-k48:data/mem/k48.jsonl mnli-devm2000:data/prep/mnli-devm-eval2000.jsonl klue-dev:data/prep/klue-dev.jsonl}"
PROTOCOL="${G1_PROTOCOL:-src/measure/protocol-g1.json}"
SEEDS="${G1_SEEDS:-1 2 3}"
SUFFIX="${G1_SUFFIX:-}"
NEUTRAL="${G1_NEUTRAL:-cd}"
TRAIN_ONLY="${G1_TRAIN_ONLY:-0}"
EVAL_ONLY="${G1_EVAL_ONLY:-0}"
if [ "$TRAIN_ONLY" = 1 ] && [ "$EVAL_ONLY" = 1 ]; then
  echo "G1_TRAIN_ONLY 와 G1_EVAL_ONLY 는 같이 못 준다" >&2
  exit 2
fi
if [ "$EVAL_ONLY" = 1 ] && [ $# -gt 0 ]; then
  echo "G1_EVAL_ONLY=1 은 학습 인자를 안 받는다 — run/seed3.sh $cond --" >&2
  exit 2
fi
E="-e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 -e HF_HUB_OFFLINE=1"
export JT_SWAP_MAX_MB="${JT_SWAP_MAX_MB:-256}" JT_AVAIL_MIN_MB="${JT_AVAIL_MIN_MB:-10240}"

cd "$JT_ROOT" || exit 2
if [ "$TRAIN_ONLY" = 1 ]; then need="$DEV_CMP"; else need="$DEV_CAL $DEV_CMP $PROTOCOL"; fi
for f in $need; do
  [ -f "$f" ] || { echo "없는 파일: $JT_ROOT/$f" >&2; exit 2; }
done

eval_set() {  # eval_set <판> <셋 이름> <셋 파일>
  src/run/podman-train.sh $E --name "$1-$2" train/eval_nli.py --model "runs/$1/ckpt-best" --pairs "$3" \
    --out-dir "runs/$1/eval-$2" --max-len 320 --device cuda --dtype fp32 --batch 64 \
    --public-neutral "$NEUTRAL" --protocol "$PROTOCOL"
}

for seed in $SEEDS; do
  run="g1-$cond-s$seed$SUFFIX"
  if [ "$EVAL_ONLY" = 1 ]; then
    # 학습한 판을 잰다 — --select last 라 ckpt-best 가 마지막 바퀴 가중치다 (규약 ckpt last)
    [ -f "runs/$run/ckpt-best/model.safetensors" ] || { echo "학습한 판이 없다: runs/$run/ckpt-best" >&2; exit 2; }
    code=skip
  else
    if [ -e "runs/$run" ]; then
      echo "이미 있다: runs/$run — 지우거나 G1_SUFFIX 를 준다" >&2
      exit 2
    fi
    # 로그는 덧붙여 쓴다 — 멈췄다 다시 돈 판의 옛 「stop requested」 줄을 안 읽게 이번 판 줄만 본다 (45-라 함정)
    seen=0
    [ -f "logs/$run.log" ] && seen=$(wc -l < "logs/$run.log")
    src/run/podman-train.sh $E --name "$run" train/train_nli.py "$@" --dev "$DEV_CMP" \
      --out "runs/$run" --select last --seed "$seed"
    code=$?
    if tail -n +$((seen + 1)) "logs/$run.log" | grep -q "stop requested" || [ "$code" != 0 ]; then
      echo "STOPPED train=$code" > "logs/$run.done"
      exit 1
    fi
  fi
  if [ "$TRAIN_ONLY" = 1 ]; then
    echo "TRAINONLY train=$code" > "logs/$run.done"
    continue
  fi
  eval_set "$run" dev-cal "$DEV_CAL" || exit 1
  eval_set "$run" dev-cmp "$DEV_CMP" || exit 1
  python3 -I src/train/calibrate.py --pred "runs/$run/eval-dev-cal/pred.jsonl" --fitted-on "$DEV_CAL" \
    --out "runs/$run/calibration.json" > /dev/null || exit 1
  : > "runs/$run/eval-side.txt"
  for spec in $SIDE; do
    name="${spec%%:*}"; file="${spec#*:}"
    eval_set "$run" "$name" "$file" || exit 1
    # eval-side.txt 한 줄 : 이름 sha8 수 acc acc3 거짓지지 {재현} (measure/compare.py read_side 꼴)
    python3 -I - "runs/$run/eval-$name/summary.json" "$name" >> "runs/$run/eval-side.txt" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")); m = s["metrics"]
print(sys.argv[2], s["set_sha256"][:8], m["n"], m["acc"], m["acc3"], m["false_support"], m["recall"])
PY
  done
  echo "ALLDONE train=$code" > "logs/$run.done"
done
