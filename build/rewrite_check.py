"""rewrite_check — BoolQ · KLUE 물음 → 평서문 바꾸기(로컬 생성)의 입력 조각 · 거두기 · 기계 검사 (K 판 ③ · ⑤).

설계 `../Docs/Design/2026-10-13-v1.2-한국어b보강판설계.md` 3-2 · 3-5. 본문은 git 밖(`~/.judgemodel/build/g3/`)에만 쓴다.

    python -I rewrite_check.py prep  --kind boolq --src <g3>/boolq/train.jsonl --work <g3>/boolq/work [--size 40] [--tag t]
    python -I rewrite_check.py merge --kind boolq --src <g3>/boolq/train.jsonl --out <합친 파일> <모델 출력 …>
    python -I rewrite_check.py check --kind boolq --in <합친 파일> [--out 결과.jsonl]

- BoolQ : `needs_rewrite` 줄의 claim(물음)을 같은 명제의 평서문으로 → 새 칸 `claim_rewrite`. claim · hint 는 그대로.
- KLUE : 질문 + 답을 평서문으로(QACG). 모델은 짝(`pair`)마다 a(참 답) · b(바꾼 답) 두 문장을 낸다.
  a 는 `form: qa-true` 줄, b 는 `qa-alt` 줄의 `claim_rewrite` 가 된다. `ans_type: person` 은 빼고 넣는다(메인 결정 10-13).
- 검사 함수는 `seed_contrast` 의 토큰 꼴(`_TOKEN`) · 값 찾기(`has_value`) · 부정어 꼴(`_NEG`)을 그대로 쓴다.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_batches as lb  # noqa: E402
import seed_contrast as sc  # noqa: E402

STUDIO = os.path.dirname(os.path.dirname(HERE))
SIZES = {"boolq": 40, "klue": 20}
SKIP_TYPES = ("person",)                  # KLUE 사람 유형은 잡음이 커서 뺀다 (메인 결정 2026-10-13)
BOOLQ_TOKENS = 3                           # 물음 → 평서문 : 바뀐 토큰 수 — 끝맺음 낱말 · 물음표 · 하나 더까지
BOOLQ_LEN = (0.6, 1.5)
KLUE_PAIR_SLACK = 1                        # a ↔ b : 바뀐 토큰 ≤ 답 토큰 수 + 1 (조사가 따로 떨어진 꼴)
KLUE_LEN = (0.5, 2.5)                      # 질문 대비 — 「~는?」 꼴 질문은 평서문이 꽤 길어진다
_Q_TAIL = re.compile(r"(는가|은가|인가|던가|는지|을까|ㄹ까|나요|가요|까요|니까|습니까|냐|니|나)[.]?$")


# ---------------------------------------------------------------- 검사

def is_question(text):
    """물음꼴 — seed_contrast 꼴에 「~인가 · ~는가 · ~나」 같은 물음표 없는 끝맺음을 더한다."""
    tail = text.rstrip(" )」』'\"")
    return sc.is_question(text) or bool(_Q_TAIL.search(tail))


def changed_tokens(old, new):
    """바뀐 토큰 수 — 덩이 수(`sc.chunks`)는 문장 통째 바꿈도 1 로 세어, 덩이마다 긴 쪽 토큰 수를 더한다."""
    a, b = sc._TOKEN.findall(old), sc._TOKEN.findall(new)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")


def neg_count(text):
    return len(sc._NEG.findall(text))


def _length_bad(line, base, bounds, floor=8):
    ratio = len(line) / max(1, len(base))
    return not bounds[0] <= ratio <= bounds[1] or len(line) < floor


def check_boolq(claim, rewrite):
    """물음 claim → 평서문 rewrite 의 까닭 목록 (빈 목록이면 통과). 부정 방향은 늘지도 줄지도 않아야 한다."""
    if not rewrite:
        return ["empty"]
    out = []
    if rewrite.strip() == claim.strip():
        out.append("same_as_claim")
    if is_question(rewrite):
        out.append("question")
    if _length_bad(rewrite, claim, BOOLQ_LEN):
        out.append("length")
    if changed_tokens(claim, rewrite) > BOOLQ_TOKENS:
        out.append("chunks")
    n_old, n_new = neg_count(claim), neg_count(rewrite)
    if n_new > n_old:
        out.append("negation_added")
    elif n_new < n_old:
        out.append("negation_dropped")
    return out


def check_klue(question, answer, alt, a, b):
    """질문 + 참 답 a 문장 · 바꾼 답 b 문장 → {"a": 까닭, "b": 까닭, "pair": 까닭}."""
    why = {"a": [], "b": [], "pair": []}
    for role, line, value in (("a", a, answer), ("b", b, alt)):
        if not line:
            why[role].append("empty")
            continue
        if is_question(line):
            why[role].append("question")
        if not sc.has_value(value, line):
            why[role].append("value_missing")
        if _length_bad(line, question, KLUE_LEN):
            why[role].append("length")
        if neg_count(line) > neg_count(question):
            why[role].append("negation_added")
    if a and b:
        if answer not in alt and sc.has_value(answer, b):
            why["b"].append("b_has_answer")
        if alt not in answer and sc.has_value(alt, a):
            why["a"].append("a_has_alt")
        n = changed_tokens(a, b)
        limit = max(len(sc._TOKEN.findall(answer)), len(sc._TOKEN.findall(alt))) + KLUE_PAIR_SLACK
        if n == 0:
            why["pair"].append("same")
        elif n > limit:
            why["pair"].append("chunks")
    return {k: v for k, v in why.items() if v}


def check_rows(kind, rows):
    """합친 줄 → (줄별 결과, 셈)."""
    count = collections.Counter()
    results = []
    if kind == "boolq":
        for row in rows:
            if not row.get("needs_rewrite"):
                continue
            why = check_boolq(row["claim"], row.get("claim_rewrite"))
            count["rows"] += 1
            count["pass"] += not why
            count.update(why)
            results.append({"id": row["id"], "pass": not why, "why": why})
        return results, dict(count)
    for pair, two in _klue_pairs(rows).items():
        a, b = two.get("a"), two.get("b")
        base = a or b
        why = check_klue(base["question"], base["answer"], base["alt_answer"],
                         a and a.get("claim_rewrite"), b and b.get("claim_rewrite"))
        count["pairs"] += 1
        count["pass"] += not why
        for role, ws in why.items():
            count.update(f"{role}_{w}" for w in ws)
        results.append({"pair": pair, "pass": not why, "why": why})
    return results, dict(count)


def _klue_pairs(rows):
    pairs = collections.OrderedDict()
    for row in rows:
        role = "b" if row.get("hint") == "b" else "a"
        pairs.setdefault(row["pair"], {})[role] = row
    return pairs


# ---------------------------------------------------------------- 조각 · 거두기

GOAL_BOOLQ = """너는 한국어 문장 교정기다. `{inp}` 의 JSONL 줄을 Read 로 읽고, 줄마다 결과 한 줄씩 `{out}` 에 새로 쓴다 (Edit 으로 old_string 을 빈 문자열로 두고 만든다).

결과 한 줄의 꼴 (JSON 한 줄, 다른 칸 없음) : {{"id": "<입력 id 그대로>", "claim_rewrite": "<평서문>"}}

할 일 : claim(물음)을 **같은 명제의 평서문**으로 바꾼다.
- 끝맺음만 바꾼다 (「~인가?」 → 「~이다.」, 「~하나요?」 → 「~한다.」, 「~있는가?」 → 「~있다.」, 「~했습니까?」 → 「~했다.」).
- 낱말 · 수 · 이름 · 조사는 claim 그대로 둔다. 줄이거나 늘리거나 다른 말로 바꾸지 않는다. 문장 끝은 마침표.
- 긍정 · 부정 방향을 바꾸지 않는다. 「~않는가?」 → 「~않는다.」, 「~없나요?」 → 「~없다.」. 부정어(않다 · 안 · 못 · 아니다 · 없다)를 새로 넣거나 빼지 않는다.
- 참 · 거짓을 따지지 않는다. 틀린 말 같아도 그 말 그대로 평서문으로 쓴다.
- 입력 줄을 빠짐없이, 입력 차례대로 쓴다. 설명 · 머리말 · 코드펜스 없이 JSONL 줄만 쓴다."""

GOAL_KLUE = """너는 한국어 문장 교정기다. `{inp}` 의 JSONL 줄을 Read 로 읽고, 줄마다 결과 한 줄씩 `{out}` 에 새로 쓴다 (Edit 으로 old_string 을 빈 문자열로 두고 만든다).

결과 한 줄의 꼴 (JSON 한 줄, 다른 칸 없음) : {{"pair": "<입력 pair 그대로>", "a": "<answer 를 넣은 평서문>", "b": "<alt_answer 를 넣은 평서문>"}}

할 일 : question(물음)과 답을 합쳐 **평서문 하나**로 쓴다.
- a 는 answer 를 답 자리에 넣은 평서문이다. (「히로히토 천황이 항복을 선언한 해는?」 + 「1945년」 → 「히로히토 천황이 항복을 선언한 해는 1945년이다.」)
- b 는 a 와 **같은 문장**에서 answer 자리만 alt_answer 로 바꾼 것이다. 조사만 값에 맞춘다 (「대구에서」 → 「인천에서」, 「3개를」 → 「51개를」). 다른 글자는 a 와 같아야 한다.
- answer · alt_answer 는 글자 그대로 쓴다. 단위 · 숫자를 바꾸거나 풀어 쓰지 않는다.
- question 의 낱말은 그대로 쓰고, 묻는 말(「몇」 · 「언제」 · 「어디」 · 「누구」 · 「무엇」)만 값으로 바꾼다. 문장 끝은 마침표.
- 부정어(않다 · 안 · 못 · 아니다 · 없다)를 새로 넣지 않는다. 물음꼴로 끝내지 않는다. 참 · 거짓을 따지지 않는다.
- 입력 줄을 빠짐없이, 입력 차례대로 쓴다. 설명 · 머리말 · 코드펜스 없이 JSONL 줄만 쓴다."""


def prep_units(kind, rows):
    """모델에게 줄 줄인 꼴 목록 (BoolQ : 물음 줄 · KLUE : 짝 하나에 한 줄, 사람 유형 뺌)."""
    if kind == "boolq":
        return [{"id": r["id"], "claim": r["claim"]} for r in rows if r.get("needs_rewrite")]
    units, seen = [], set()
    for r in rows:
        if r.get("ans_type") in SKIP_TYPES or r["pair"] in seen:
            continue
        seen.add(r["pair"])
        units.append({"pair": r["pair"], "question": r["question"], "answer": r["answer"],
                      "alt_answer": r["alt_answer"], "ans_type": r["ans_type"]})
    return units


def make_task(kind, work, tag):
    inp, out = f"in/in-{tag}.jsonl", f"out/batch-{tag}.jsonl"
    goal = (GOAL_BOOLQ if kind == "boolq" else GOAL_KLUE).format(inp=inp, out=out)
    return {"id": f"20261013-K-{kind}-{tag}", "repo": work.replace("\\", "/"), "goal": goal,
            "allow": [out], "keep": [inp], "items": [{"id": "1", "file": out}]}


def merge_rows(kind, rows, outputs):
    """입력 줄 + 모델 출력 → (합친 줄, 셈). 모델 출력이 없는 줄은 claim_rewrite 없이 남는다."""
    got, count = {}, collections.Counter()
    key = "id" if kind == "boolq" else "pair"
    for obj in outputs:
        if not isinstance(obj, dict) or key not in obj:
            count["bad"] += 1
            continue
        got[obj[key]] = obj
    out = []
    for r in rows:
        if kind == "klue" and r.get("ans_type") in SKIP_TYPES:
            count["skip_type"] += 1
            continue
        r = dict(r)
        g = got.get(r[key])
        if kind == "boolq" and r.get("needs_rewrite"):
            if g and isinstance(g.get("claim_rewrite"), str):
                r["claim_rewrite"] = g["claim_rewrite"]
                count["merged"] += 1
            else:
                count["missing"] += 1
        elif kind == "klue":
            text = g and g.get("b" if r.get("hint") == "b" else "a")
            if isinstance(text, str):
                r["claim_rewrite"] = text
                count["merged"] += 1
            else:
                count["missing"] += 1
        out.append(r)
    return out, dict(count)


def read_outputs(paths):
    objs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    objs.append(json.loads(line))
                except ValueError:
                    objs.append(None)
    return objs


# ---------------------------------------------------------------- 명령

def _guard(path):
    """본문 · 조각은 스튜디오 저장소 밖에만 쓴다."""
    p = os.path.abspath(path)
    if os.path.commonpath([p, os.path.abspath(STUDIO)]) == os.path.abspath(STUDIO):
        sys.exit(f"{path} 가 스튜디오 저장소 안이다 — 본문은 git 밖에 둔다")


def cmd_prep(a):
    _guard(a.work)
    units = prep_units(a.kind, lb.read_jsonl(a.src))
    if a.limit:
        units = units[:a.limit]
    size = a.size or SIZES[a.kind]
    tasks = os.path.join(os.path.dirname(os.path.abspath(a.work)), "tasks")
    for i in range(0, len(units), size):
        tag = f"{a.tag}{i // size + 1:03d}"
        lb.write_jsonl(os.path.join(a.work, "in", f"in-{tag}.jsonl"), units[i:i + size])
        lb.write_json(os.path.join(tasks, f"task-{tag}.json"), make_task(a.kind, a.work, tag))
    print(json.dumps({"units": len(units), "batches": -(-len(units) // size), "size": size}, ensure_ascii=False))
    return 0


def cmd_merge(a):
    _guard(a.out)
    rows, count = merge_rows(a.kind, lb.read_jsonl(a.src), read_outputs(a.outputs))
    lb.write_jsonl(a.out, rows)
    print(json.dumps(count, ensure_ascii=False))
    return 0


def cmd_check(a):
    results, count = check_rows(a.kind, lb.read_jsonl(a.inp))
    if a.out:
        _guard(a.out)
        lb.write_jsonl(a.out, results)
    print(json.dumps(count, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="BoolQ · KLUE 물음 → 평서문 바꾸기 조각 · 거두기 · 기계 검사")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "merge", "check"):
        sub.add_parser(name).add_argument("--kind", choices=("boolq", "klue"), required=True)
    p = sub.choices["prep"]
    p.add_argument("--src", required=True)
    p.add_argument("--work", required=True, help="따로 git init 한 작업 폴더 (in/ · out/)")
    p.add_argument("--size", type=int)
    p.add_argument("--tag", default="t")
    p.add_argument("--limit", type=int, help="앞에서 이만큼만 (표본 판)")
    p = sub.choices["merge"]
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("outputs", nargs="+")
    p = sub.choices["check"]
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out")
    a = parser.parse_args(argv)
    return {"prep": cmd_prep, "merge": cmd_merge, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
