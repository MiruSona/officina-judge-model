"""사전 등록 — 재기 직전에 모델 · 셋 · 문턱 · 온도 · 자 · 다음 걸음을 prereg.json 에 박제한다 (설계 6-2 · 6-3 · 6-4).

    python -I prereg.py --model-dir ~/.judgemodel/models/nli-studio-v1 \
        --set game90=<game90.jsonl> --set k48=<k48.jsonl> --once game90 --judge game90 --out <git 밖>/prereg.json

이미 있으면 덮지 않는다 — 잰 뒤 자를 옮기지 않게. 문턱은 model.json scores.thresholds 에서 읽는다.
"""
import argparse
import hashlib
import json
import os
import sys
import time

# 설계 6-3 의 자. 원문은 설계 문서이고 여기 값은 그 박제다.
CRITERIA = {
    "accuracy_min": 0.85,            # ② 모델 표 · 문턱 없이 최고 확률 답
    "contradict_recall_min": 0.75,   # ② 모델 표
    "false_support_max": 3,          # ③ 사다리 표 · 확신 A 인데 정답 B·C
    "unsure_max": 0.30,              # ③ 사다리 표 · Unsure / 전체
    "p95_ms_max": 1000.0,            # HTTP 포함 · 메인 PC
}
NEXT = {
    "pass": "nli_url 을 켠다 · mem history · 진행상황 · game90 은 개발용으로 내림 · 저장소 연결은 사용자에게 묻는다",
    "p95_only": "int8 → 그래도 넘으면 klue/roberta-base 로 같은 길 (p95 는 라벨을 안 보므로 다시 재도 된다)",
    "fail": "길 1 을 접는다 · nli_url 을 비운다 · 수치를 mem history 로 · 다시 열려면 새 blind 셋을 먼저",
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(model_spec, sets, once, judge):
    thresholds = (model_spec.get("scores") or {}).get("thresholds") or {}
    if thresholds.get("nli_sure") is None or thresholds.get("nli_sure_support") is None:
        raise SystemExit("model.json scores.thresholds 에 nli_sure · nli_sure_support 가 없다 — calibrate 뒤 내보낸다")
    return {
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"name": model_spec["name"], "weights_sha256": model_spec["weights_sha256"]},
        "temperature": model_spec["calibration"]["temperature"],
        "nli_sure": thresholds["nli_sure"], "nli_sure_support": thresholds["nli_sure_support"],
        "sets": {name: {"file": os.path.basename(path), "sha256": sha256_file(path),
                        "once": name in once, "judge": name in judge} for name, path in sets.items()},
        "criteria": CRITERIA, "next": NEXT,
    }


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="사전 등록 prereg.json 쓰기")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--set", dest="sets", action="append", required=True, help="이름=jsonl")
    p.add_argument("--once", action="append", default=[], help="한 번만 재는 셋 이름 (game90)")
    p.add_argument("--judge", action="append", default=[], help="자로 판정하는 셋 이름 (game90)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"이미 있다: {a.out} — 사전 등록은 덮지 않는다")
    with open(os.path.join(os.path.expanduser(a.model_dir), "model.json"), encoding="utf-8") as f:
        spec = json.load(f)
    sets = dict(s.split("=", 1) for s in a.sets)
    prereg = build(spec, sets, set(a.once), set(a.judge))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prereg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(prereg, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
