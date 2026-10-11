"""fold40 재기 — 양방향 「담김」 판정이 접어도 잃는 말이 없는 짝만 고르는지 (판정 넷 확장 설계 4-1 · 4-2 · 6-2).

    python -I run_fold.py --prereg <git 밖>/prereg.json --set fold40=<fold40.jsonl> \
        --url http://127.0.0.1:8092 --out-dir <git 밖>/nli2

fold40 은 문장 쌍이 아니라 **기억 짝**이다. 한 줄 꼴 :

    {"id": "f01",
     "x": {"id": "<기억 id>", "summary": "요약 한 문장", "sentences": ["본문 문장", …], "evidence": "정렬된 문장 묶음"},
     "y": { … 같은 꼴 … },
     "foldable": "x" | "y" | "both" | "none",     # 사람 라벨 : 접어도 잃는 말이 없는 쪽
     "group": "same" | "inside" | "overlap"?}     # 갈래 (같은 내용 · 한쪽 담김 · 겹침) — 셈에만 쓴다

「담김(작은 Z, 큰 W)」 = Z 의 요약 + 본문 문장(최대 5)을 하나씩 주장으로, W 의 정렬된 문장 묶음을 근거로 묻는다.
모든 문장이 확신 a(≥ nli_sure_support)이고 어느 문장도 확신 b(≥ nli_sure)가 아니면 담김이다.
fold40 의 sentences · evidence 는 손님(AIMemoryTool) FoldClaims · FoldEvidence 출력을 그대로 박아야 한다
(sentences 는 FoldClaims 의 둘째 줄부터 — 첫 줄 요약은 summary 칸이다).
손님은 근거를 닮은 순으로 줄 세우고 480룬에서 자르고 12자 미만 문장을 거르는데, 여기서는 그 일을 다시 하지 않는다.
그래서 `evidence` 가 없는 줄은 HTTP 를 부르기 전에 거절한다 — 손님과 다른 글로 재면 잰 값이 손님 판정을 대표하지 않는다.
먼저 요약 한 문장만 묻고, 확신 a 가 나온 방향만 문장마다 더 묻는다 — 대부분의 짝은 두 번에 끝난다.

접기 오판 = 모델이 「담김」이라 한 쪽(접을 쪽)을 사람은 「잃는 말 있음」이라 한 것. 양쪽 다 담김이면 두 쪽 다 맞아야 한다.
자는 prereg criteria 의 fold_wrong_max (넷 확장 설계 4-2, ≤ 1 짝). 문턱은 prereg 의 값을 그대로 쓴다.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

from calibrate import answer  # noqa: E402
from data_prep import strip_tags  # noqa: E402
from run_measure import check_server, check_set, http_json, read_rows  # noqa: E402

SUPPORT, CONTRADICT = 0, 1
MAX_SENTENCES = 5
FOLDABLE = {"x": {"x"}, "y": {"y"}, "both": {"x", "y"}, "none": set()}


def claims_of(memory):
    """주장으로 물을 문장 — 요약 + 본문 문장 최대 5. 빈 문장은 뺀다."""
    out = [memory.get("summary", "")] + list(memory.get("sentences") or [])[:MAX_SENTENCES]
    out = [strip_tags(s).strip() for s in out if s]
    return [s for s in out if s]


def evidence_of(memory):
    """근거로 쓸 글 — 손님 FoldEvidence 가 낸 evidence (check_rows 가 있는지 먼저 본다)."""
    return strip_tags(memory["evidence"])


def check_rows(rows):
    """HTTP 전에 fold40 줄을 본다 — foldable 값과 양쪽 evidence. 틀리면 SystemExit."""
    for row in rows:
        rid = row.get("id", "")
        if row.get("foldable", "none") not in FOLDABLE:
            raise SystemExit(f"foldable 을 못 읽는다: {rid} {row.get('foldable')!r}")
        for side in ("x", "y"):
            memory = row.get(side)
            if not isinstance(memory, dict):
                raise SystemExit(f"{rid} 에 {side} 기억이 없다")
            if not isinstance(memory.get("evidence"), str) or not memory["evidence"].strip():
                raise SystemExit(f"{rid} {side} 에 evidence 가 없다 — 손님 FoldEvidence 출력을 박아 넣는다")


def parse_set(text):
    """`이름=jsonl` → (이름, 경로). `=` 가 없으면 SystemExit."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"--set 은 이름=jsonl 꼴이다: {text!r}")
    return name, path


def sure_letter(probs, sure, sure_support):
    """확신 a 면 0, 확신 b 면 1, 아니면 None (Go 판정기와 같은 「모른다」 선)."""
    top, p, base_unsure = answer(probs)
    if base_unsure:
        return None
    if top == SUPPORT and p >= sure_support:
        return SUPPORT
    if top == CONTRADICT and p >= sure:
        return CONTRADICT
    return None


def contained(small, big, ask, sure, sure_support):
    """small 이 big 에 담겼나. (담김, 물은 횟수, 까닭). ask(evidence, claim) → 확률 목록."""
    evidence = evidence_of(big)
    claims = claims_of(small)
    if not claims or not evidence:
        return False, 0, "빈 기억"
    calls = 0
    for i, claim in enumerate(claims):
        calls += 1
        got = sure_letter(ask(evidence, claim), sure, sure_support)
        if got == CONTRADICT:
            return False, calls, f"문장 {i} 확신 b"
        if got != SUPPORT:
            return False, calls, "요약이 확신 a 아님" if i == 0 else f"문장 {i} 확신 a 아님"
    return True, calls, "모든 문장 확신 a"


def judge_pair(row, ask, sure, sure_support):
    """짝 하나 — 두 방향으로 담김을 묻고 접을 쪽 · 오판을 정한다."""
    x_in_y, calls_x, why_x = contained(row["x"], row["y"], ask, sure, sure_support)
    y_in_x, calls_y, why_y = contained(row["y"], row["x"], ask, sure, sure_support)
    fold = set()
    if x_in_y:
        fold.add("x")
    if y_in_x:
        fold.add("y")
    human = row.get("foldable", "none")
    if human not in FOLDABLE:
        raise SystemExit(f"foldable 을 못 읽는다: {row.get('id', '')} {human!r}")
    ok = FOLDABLE[human]
    return {"id": row.get("id", ""), "group": row.get("group", ""), "foldable": human,
            "x_in_y": x_in_y, "y_in_x": y_in_x, "why": {"x_in_y": why_x, "y_in_x": why_y},
            "calls": calls_x + calls_y, "fold": sorted(fold),
            "wrong": sorted(fold - ok), "missed": sorted(ok - fold)}


def fold_table(results):
    """접기 오판 짝 수 · 놓친 짝 수 · 갈래별 셈 · 묻기 횟수."""
    wrong = [r["id"] for r in results if r["wrong"]]
    missed = [r["id"] for r in results if r["missed"]]
    by_group = {}
    for r in results:
        g = by_group.setdefault(r["group"] or "-", {"n": 0, "folded": 0, "wrong": 0})
        g["n"] += 1
        g["folded"] += 1 if r["fold"] else 0
        g["wrong"] += 1 if r["wrong"] else 0
    calls = [r["calls"] for r in results]
    return {"n": len(results), "fold_wrong": len(wrong), "fold_wrong_ids": wrong,
            "fold_missed": len(missed), "fold_missed_ids": missed, "by_group": dict(sorted(by_group.items())),
            "calls_total": sum(calls), "pairs_done_in_two": sum(1 for c in calls if c <= 2)}


def fold_verdict(criteria, table):
    passed = table["fold_wrong"] <= criteria["fold_wrong_max"]
    return {"checks": {"fold_wrong": passed}, "passed": passed}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="fold40 접기 오판 재기 (양방향 담김)")
    p.add_argument("--prereg", required=True)
    p.add_argument("--set", required=True, help="이름=jsonl (prereg 에 있는 이름, 예 fold40=…)")
    p.add_argument("--url", default="http://127.0.0.1:8092")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    with open(a.prereg, encoding="utf-8") as f:
        prereg = json.load(f)
    if "fold_wrong_max" not in prereg["criteria"]:
        raise SystemExit("prereg 에 fold_wrong_max 가 없다 — 넷 확장 prereg(4칸 모델)로만 잰다")
    name, path = parse_set(a.set)
    rows = read_rows(path)
    check_rows(rows)
    info, result_path = check_set(a, prereg, name, path)
    letters = check_server(a, prereg)
    url = a.url.rstrip("/") + "/judge"

    def ask(evidence, claim):
        body = http_json(url, {"evidence": evidence, "claim": claim})
        return [body[letter] for letter in letters]

    results = [judge_pair(row, ask, prereg["nli_sure"], prereg["nli_sure_support"]) for row in rows]
    table = fold_table(results)
    result = {"set": name, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": prereg["model"],
              "thresholds": {"nli_sure": prereg["nli_sure"], "nli_sure_support": prereg["nli_sure_support"]},
              "fold_table": table}
    if info["judge"]:
        result["verdict"] = fold_verdict(prereg["criteria"], table)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"pairs-{name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(result_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
