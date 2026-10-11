"""사전 등록 — 재기 직전에 모델 · 셋 · 문턱 · 온도 · 자 · 다음 걸음을 prereg.json 에 박제한다 (설계 6-2 · 6-3 · 6-4).

    python -I prereg.py --model-dir ~/.judgemodel/models/nli-studio-v1 \
        --set game90=<game90.jsonl> --set k48=<k48.jsonl> --once game90 --judge game90 --out <git 밖>/prereg.json
        [--protocol measure/protocol-g1.json]

이미 있으면 덮지 않는다 — 잰 뒤 자를 옮기지 않게. 문턱은 model.json scores.thresholds 에서 읽는다.
model.json letters 가 넷이면 자를 --protocol 규약의 release_gate 에서 읽어(gates.py 와 같은 파일) 문턱 셋
(nli_sure_related 포함)과 함께 박제한다. 숫자는 여기 박지 않는다 — 원문은 규약 파일 하나다.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare as cmp  # noqa: E402

DEFAULT_PROTOCOL = os.path.join(HERE, "protocol-g1.json")

# 설계 6-3 의 자. 원문은 설계 문서이고 여기 값은 그 박제다.
CRITERIA = {
    "accuracy_min": 0.85,            # ② 모델 표 · 문턱 없이 최고 확률 답
    "contradict_recall_min": 0.75,   # ② 모델 표
    "false_support_max": 3,          # ③ 사다리 표 · 확신 A 인데 정답 B·C
    "unsure_max": 0.30,              # ③ 사다리 표 · Unsure / 전체
    "p95_ms_max": 1000.0,            # HTTP 포함 · 메인 PC
}
# 4칸 자 : 규약 release_gate 칸 → run_measure.verdict 가 읽는 criteria 칸. 값은 규약에서만 온다
GATE_TO_CRITERIA = {
    "acc3_min": "accuracy3_min",                                 # 모델 표 · c·d 를 한 칸으로 접은 정확도
    "b_recall_min": "contradict_recall_min",                     # 모델 표
    "sure_false_support_share_max": "false_support_share_max",   # 확신 표 · 확신 A 인데 정답 B·C·D / 지지 아닌 쌍
    "unsure_max": "unsure_max",                                  # 확신 표 · Unsure / 전체
    "d_recall_min": "related_recall_min",                        # 모델 표 · 정답 d 중 d
    "d_sure_precision_min": "related_sure_precision_min",        # 확신 표 · 확신 D 중 정답 d
    "p95_ms_max": "p95_ms_max",                                  # HTTP 포함 · 한 번 묻기
}
GATE_OPTIONAL = ("fold_wrong_max",)  # 규약에 있을 때만 옮긴다 (fold40 · run_fold.py)
NEXT_V2 = {
    "pass": "nli-studio-v3 를 꽂는다 (응답 nli-v2) · 손님 판올림 · 진행상황 · mem history",
    "p95_only": "int8 → 그래도 넘으면 옛 서버(nli-studio-v2) 유지",
    "fail_old": "옛 일 자(정확도·반대·거짓 지지·보류) 미달 — 넷 칸을 꽂지 않는다 (설계 8절 ①)",
    "fail_related": "옛 일은 통과 · d 자만 미달 — --compat3 로 꽂아 옛 일만 쓰고 merge 2차는 양방향 지지만 (설계 8절 ②)",
}
NEXT = {
    "pass": "nli_url 을 켠다 · mem history · 진행상황 · game90 은 개발용으로 내림 · 저장소 연결은 사용자에게 묻는다",
    "p95_only": "int8 → 그래도 넘으면 klue/roberta-base 로 같은 길 (p95 는 라벨을 안 보므로 다시 재도 된다)",
    "fail": "길 1 을 접는다 · nli_url 을 비운다 · 수치를 mem history 로 · 다시 열려면 새 blind 셋을 먼저",
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def criteria_from_gate(gate):
    """규약 release_gate → 4칸 criteria. 빠진 칸이 있으면 멈춘다 (어림값으로 채우지 않는다)."""
    missing = [k for k in GATE_TO_CRITERIA if k not in gate]
    if missing:
        raise SystemExit(f"규약 release_gate 에 {', '.join(missing)} 가 없다")
    criteria = {dst: gate[src] for src, dst in GATE_TO_CRITERIA.items()}
    criteria.update({k: gate[k] for k in GATE_OPTIONAL if k in gate})
    return criteria


def build(model_spec, sets, once, judge, protocol=None, protocol_path="", protocol_sha=""):
    thresholds = (model_spec.get("scores") or {}).get("thresholds") or {}
    if thresholds.get("nli_sure") is None or thresholds.get("nli_sure_support") is None:
        raise SystemExit("model.json scores.thresholds 에 nli_sure · nli_sure_support 가 없다 — calibrate 뒤 내보낸다")
    letters = model_spec.get("letters") or ["a", "b", "c"]
    four = len(letters) == 4
    if four and thresholds.get("nli_sure_related") is None:
        raise SystemExit("4칸 모델인데 scores.thresholds 에 nli_sure_related 가 없다 — calibrate 뒤 내보낸다")
    out = {
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"name": model_spec["name"], "weights_sha256": model_spec["weights_sha256"]},
        "temperature": model_spec["calibration"]["temperature"],
        "nli_sure": thresholds["nli_sure"], "nli_sure_support": thresholds["nli_sure_support"],
        "sets": {name: {"file": os.path.basename(path), "sha256": sha256_file(path),
                        "once": name in once, "judge": name in judge} for name, path in sets.items()},
        "criteria": CRITERIA, "next": NEXT_V2 if four else NEXT,
    }
    if four:
        if not protocol or "release_gate" not in protocol:
            raise SystemExit("4칸 모델은 --protocol 규약의 release_gate 로 자를 박제한다 — 규약이 없다")
        out["criteria"] = criteria_from_gate(protocol["release_gate"])
        out["release_gate"] = protocol["release_gate"]
        out["protocol_file"] = os.path.basename(protocol_path)
        out["protocol_sha256"] = protocol_sha
        out["nli_sure_related"] = thresholds["nli_sure_related"]
        out["letters"] = list(letters)
        out["protocol"] = model_spec.get("protocol", "nli-v2")
    return out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="사전 등록 prereg.json 쓰기")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--set", dest="sets", action="append", required=True, help="이름=jsonl")
    p.add_argument("--once", action="append", default=[], help="한 번만 재는 셋 이름 (game90)")
    p.add_argument("--judge", action="append", default=[], help="자로 판정하는 셋 이름 (game90)")
    p.add_argument("--out", required=True)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL,
                   help="4칸 자를 읽을 규약 (release_gate) · 기본 measure/protocol-g1.json")
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"이미 있다: {a.out} — 사전 등록은 덮지 않는다")
    with open(os.path.join(os.path.expanduser(a.model_dir), "model.json"), encoding="utf-8") as f:
        spec = json.load(f)
    sets = dict(s.split("=", 1) for s in a.sets)
    protocol, protocol_sha = cmp.read_protocol(a.protocol)
    prereg = build(spec, sets, set(a.once), set(a.judge), protocol, a.protocol, protocol_sha)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prereg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(prereg, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
