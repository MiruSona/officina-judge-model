"""사전 등록 — 재기 직전에 모델 · 셋 · 문턱 · 온도 · 자 · 다음 걸음을 prereg.json 에 박제한다 (설계 6-2 · 6-3 · 6-4).

    python -I prereg.py --model-dir ~/.judgemodel/models/nli-studio-v1 \
        --set game90=<game90.jsonl> --set k48=<k48.jsonl> --once game90 --judge game90 --out <git 밖>/prereg.json
        [--protocol measure/protocol-g1.json] [--next <다음 걸음 json>]

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
import gates  # noqa: E402

DEFAULT_PROTOCOL = os.path.join(HERE, "protocol-g1.json")

# 설계 6-3 의 자. 원문은 설계 문서이고 여기 값은 그 박제다.
CRITERIA = {
    "accuracy_min": 0.85,            # ② 모델 표 · 문턱 없이 최고 확률 답
    "contradict_recall_min": 0.75,   # ② 모델 표
    "false_support_max": 3,          # ③ 사다리 표 · 확신 A 인데 정답 B·C
    "unsure_max": 0.30,              # ③ 사다리 표 · Unsure / 전체
    "p95_ms_max": 1000.0,            # HTTP 포함 · 서빙 기계
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
# g2 규약(vs_ref 칸이 있는 판) : 절대 자는 b 재현 · 확신 거짓 지지 · p95, 나머지는 기준 모델 대비 — 판정은 gates.py release
GATE_TO_CRITERIA_G2 = {
    "b_recall_min": "contradict_recall_min",
    "sure_false_support_share_max": "false_support_share_max",
    "unsure_vs_ref_max_diff": "unsure_vs_ref_max_diff",          # 보류 ≤ 기준 모델 보류 + 이 값
    "vs_ref": "vs_ref",                                          # [{metric, ci_low_gt | ci_low_ge}] 같은 쌍 차 아래끝
    "ref_model": "ref_model",
    "p95_ms_max": "p95_ms_max",
}
NEXT_G2 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다 (B판 설계 5-4)",
    "p95_only": "int8 → 그래도 넘으면 v1.1.0 유지",
    "fail": "B 판을 접고 결과만 적는다 · 8092 는 v1.1.0 그대로 (B판 설계 5-3)",
}
NEXT_G3 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K 판은 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K 판은 내지 않고 8092 는 v1.1.0 유지 · 다음 판은 새 blind 셋",
}
NEXT_G4 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K2 는 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K2 는 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
}
NEXT_G5 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K3 는 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K3 는 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
}
NEXT_G6 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K4 는 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K4 는 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
}
NEXT_G7 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K5 는 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K5 는 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
}
NEXT_G8 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K6 는 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K6 는 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
}
# g9 (K7 판) — 출시 관문 세 칸에 더해 짝 비교 멈춤 규칙(K7 설계 3절 · 사용자 확정 ⑦)을 같이 박제한다.
# 짝 비교 칸 이름은 compare 판정 낱말(먹힘 · 지킴 깨짐 · 같다)과 「두 판 연속」 — prereg 는 blind-g4 를 열기 직전에
# 쓰므로 먹힘 칸이 곧 이 박제의 앞 조건이다
NEXT_G9 = {
    "pass": "gates.py release 통과 — 8092 교체는 사용자가 정한다",
    "p95_only": "int8 → 그래도 넘으면 K7 은 내지 않고 8092 는 v1.1.0 유지",
    "fail": "출시 관문 미달 — K7 은 내지 않고 8092 는 v1.1.0 유지 · blind-g4 는 쓴 셋",
    "먹힘": "K7 먹힘(목표 통과 · 지킴 깨짐 0) — blind-g4 를 연다. 출시 관문 vs_ref 는 비공개 조사 E6(1:1 꼴) 뒤 "
            "짝 비교 지킴과 같은 F1 자로 고쳤다(2026-10-18)",
    "지킴 깨짐": "K7 지킴 깨짐 — 진단(pred.jsonl 혼동표 · 원천 × 칸 · 구간) → 사용자. 대체 판은 적지 않는다",
    "두 판 연속": "g9 지킴이 두 판 연속 깨짐 — v1.2 접음 · 8092 v1.1.0 유지 (사용자에게 알림). 연속 수는 K7 부터 센다",
    "같다": "목표만 미달 · 판별 불가 — 메인 판단 → 사용자",
}
# 기준 모델 대비 규약(vs_ref)의 다음 걸음 — 규약 파일 이름으로 고른다. 표에 없는 이름은 --next 로 준다.
NEXT_BY_PROTOCOL = {"protocol-g2.json": NEXT_G2, "protocol-g3.json": NEXT_G3, "protocol-g4.json": NEXT_G4,
                    "protocol-g5.json": NEXT_G5, "protocol-g6.json": NEXT_G6,
                    "protocol-g7.json": NEXT_G7, "protocol-g8.json": NEXT_G8, "protocol-g9.json": NEXT_G9}
NEXT_V2 = {
    "pass": "nli-studio-v3 를 꽂는다 (응답 nli-v2) · 손님 판올림 · 작업 기록",
    "p95_only": "int8 → 그래도 넘으면 옛 서버(nli-studio-v2) 유지",
    "fail_old": "옛 일 자(정확도·반대·거짓 지지·보류) 미달 — 넷 칸을 꽂지 않는다 (설계 8절 ①)",
    "fail_related": "옛 일은 통과 · d 자만 미달 — --compat3 로 꽂아 옛 일만 쓰고 merge 2차는 양방향 지지만 (설계 8절 ②)",
}
NEXT = {
    "pass": "nli_url 을 켠다 · 작업 기록 · game90 은 개발용으로 내림 · 저장소 연결은 사용자에게 묻는다",
    "p95_only": "int8 → 그래도 넘으면 klue/roberta-base 로 같은 길 (p95 는 라벨을 안 보므로 다시 재도 된다)",
    "fail": "길 1 을 접는다 · nli_url 을 비운다 · 수치를 mem history 로 · 다시 열려면 새 blind 셋을 먼저",
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_g2_gate(gate):
    """기준 모델 대비 자(vs_ref 목록)가 있는 규약인가 — protocol-g2 부터."""
    return "vs_ref" in gate


def criteria_from_gate(gate):
    """규약 release_gate → 4칸 criteria. 빠진 칸이 있으면 멈춘다 (어림값으로 채우지 않는다)."""
    mapping = GATE_TO_CRITERIA
    if is_g2_gate(gate):
        mapping = GATE_TO_CRITERIA_G2
    missing = [k for k in mapping if k not in gate]
    if missing:
        raise SystemExit(f"규약 release_gate 에 {', '.join(missing)} 가 없다")
    criteria = {dst: gate[src] for src, dst in mapping.items()}
    criteria.update({k: gate[k] for k in GATE_OPTIONAL if k in gate})
    if is_g2_gate(gate):
        criteria.update(slice_criteria(gate))
    return criteria


def slice_criteria(gate):
    """g3 꼴(거름 칸 lang · domain 이 달린 자)을 박제 문서에 읽기 쉽게 푼다.

    b_recall_min 이 목록 · dict 이면 contradict_recall_min 은 쌍 전체 b 자(없으면 None) — 거름이 없어도 숫자 하나로 푼다.
    걸러진 자는 recall_min_by_slice {b_recall@ko: .70}, vs_ref 는 vs_ref_rulers [{ruler, need}] 로 이름째 적는다.
    숫자 하나인 옛 꼴이면 빈 dict — g2 박제는 그대로.
    """
    mins = gates.recall_min_specs(gate["b_recall_min"])
    whole = next((s["min"] for s in mins if not cmp.is_slice(s)), None)
    vs_ref = gate.get("vs_ref", [])
    if not any(cmp.is_slice(s) for s in mins + vs_ref):
        if isinstance(gate["b_recall_min"], (int, float)):
            return {}
        return {"contradict_recall_min": whole}
    out = {"contradict_recall_min": whole,
           "recall_min_by_slice": {cmp.ruler_name(s): s["min"] for s in mins if cmp.is_slice(s)},
           "vs_ref_rulers": []}
    for spec in vs_ref:
        need = f"> {spec['ci_low_gt']}" if "ci_low_gt" in spec else f">= {spec['ci_low_ge']}"
        out["vs_ref_rulers"].append({"ruler": f"vs_ref_{cmp.ruler_name(spec)}", "need": f"차 95% 아래끝 {need}"})
    return out


def next_for_gate(protocol_path, next_text=None):
    """vs_ref 규약의 다음 걸음 — --next 로 받은 것 → 규약 파일 이름 표. 둘 다 없으면 멈춘다(다른 판 문구를 박지 않게)."""
    if next_text is not None:
        return next_text
    name = os.path.basename(protocol_path or "")
    if name not in NEXT_BY_PROTOCOL:
        raise SystemExit(f"규약 {name or '(이름 없음)'} 의 다음 걸음 표가 없다 — --next <json 파일> 로 준다")
    return NEXT_BY_PROTOCOL[name]


def read_next(path):
    """--next json 파일 → 다음 걸음 dict. dict 가 아니면 멈춘다."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not doc:
        raise SystemExit(f"--next 파일은 비지 않은 json 객체여야 한다 : {path}")
    return doc


def build(model_spec, sets, once, judge, protocol=None, protocol_path="", protocol_sha="", next_text=None):
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
        if is_g2_gate(protocol["release_gate"]):
            out["next"] = next_for_gate(protocol_path, next_text)
        out["protocol_file"] = os.path.basename(protocol_path)
        out["protocol_sha256"] = protocol_sha
        out["nli_sure_related"] = thresholds["nli_sure_related"]
        out["letters"] = list(letters)
        out["protocol"] = model_spec.get("protocol", "nli-v2")
    if next_text is not None:
        out["next"] = next_text
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
    p.add_argument("--next", dest="next_file", default=None,
                   help="다음 걸음 json 파일 · 없으면 규약 파일 이름 표(protocol-g2 · protocol-g3)에서 고른다")
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"이미 있다: {a.out} — 사전 등록은 덮지 않는다")
    with open(os.path.join(os.path.expanduser(a.model_dir), "model.json"), encoding="utf-8") as f:
        spec = json.load(f)
    sets = dict(s.split("=", 1) for s in a.sets)
    protocol, protocol_sha = cmp.read_protocol(a.protocol)
    next_text = read_next(a.next_file) if a.next_file else None
    prereg = build(spec, sets, set(a.once), set(a.judge), protocol, a.protocol, protocol_sha, next_text)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prereg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(prereg, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
