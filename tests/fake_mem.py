"""시험용 가짜 mem — use_eval sample · draw 가 부르는 명령만 흉내 낸다 (test_use_eval.py · test_use_draw.py).
review --nli 면 nli_url 로 정해진 문장 쌍을 묻고 b 확신이면 contradict 줄을 낸다. FAKE_MEM_* 환경 변수로 어긋남을 만든다.
"""
import json
import os
import re
import sys
import urllib.request

SURE, SURE_A = 0.82, 0.85
MEMS = {
    "m1": {"summary": "기본은 켠다", "body": "기본은 켠다. 끄려면 설정을 바꾼다.", "date": "2026-10-01", "scope": "s"},
    "m2": {"summary": "기본은 끈다", "body": "기본은 끈다. 켜려면 설정을 바꾼다.", "date": "2026-10-02", "scope": "s"},
    "m3": {"summary": "로그는 30일 둔다", "body": "로그는 30일 둔다. 그 뒤 지운다.", "date": "2026-10-01", "scope": "s"},
    "m4": {"summary": "로그는 오래 둔다", "body": "로그는 오래 보관한다. 지우지 않는다.", "date": "2026-10-03", "scope": "s"},
}
NEAR = [("m1", "m2", "기본은 켠다.", "기본은 끈다."), ("m3", "m4", "로그는 30일 둔다.", "로그는 오래 보관한다.")]


def nli_url():
    path = os.environ.get("MEM_LLM_CONFIG", "")
    if not path or not os.path.isfile(path):
        return ""
    text = open(path, encoding="utf-8").read()
    m = re.search(r'nli_url = "([^"]+)"', text)
    return m.group(1) if m else ""


def ask(evidence, claim):
    body = json.dumps({"evidence": evidence, "claim": claim}).encode("utf-8")
    req = urllib.request.Request(nli_url() + "/judge", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def flagged(ans):
    top = max(("a", "b", "c"), key=lambda k: ans[k])
    return top == "b" and ans["b"] >= SURE


def contradict_item(left, right, ev, cl):
    return {"kind": "contradict", "rule": "contradict-candidate", "id": left, "related": [right],
            "why": f"`{right}` 와 어긋날 수 있다 (NLI 반대 0.90) : 「{ev}」 ↔ 「{cl}」", "title": ""}


def review(args):
    items = []
    clash = set()
    nli = "--nli" in args
    if nli:
        for left, right, ev, cl in NEAR:
            if flagged(ask(ev, cl)):
                clash.add((left, right))
                items.append(contradict_item(left, right, ev, cl))
        if os.environ.get("FAKE_MEM_EXTRA_C19"):
            items.append(contradict_item("m1", "m3", "기본은 켠다.", "로그는 30일 둔다."))
    for left, right, _, _ in NEAR:
        if nli and os.environ.get("FAKE_MEM_DROP_MERGE"):
            continue
        if (left, right) not in clash:
            items.append({"kind": "merge", "rule": "merge-candidate", "id": left, "related": [right], "why": "", "title": ""})
    return {"checked": len(MEMS), "counts": {}, "items": items, "notes": []}


def judge_support(argv):
    """줄마다 한 줄. FAKE_MEM_NLI = skip(안 묻기) · fail(못 받음 · exit 2) · secret(비밀 꼴 거절 · exit 2)."""
    path = argv[argv.index("--file") + 1]
    stage = argv[argv.index("--stage") + 1]
    mode = os.environ.get("FAKE_MEM_NLI", "") if stage == "nli" else ""
    code = 0
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        out = {"id": row["id"], "letter": "", "stage": stage}
        if stage == "rules" and "켠다" in row["evidence"] and "끈다" in row["claim"]:
            out["letter"] = "B"
        if mode in ("fail", "secret"):
            out = {"id": row["id"], "problem": "timeout" if mode == "fail" else "secret-shape"}
            code = 2
        elif stage == "nli" and mode != "skip":
            ask(row["evidence"], row["claim"])
        print(json.dumps(out, ensure_ascii=False))
    return code


def main(argv):
    if argv[:2] == ["judge", "config"]:
        print(json.dumps({"path": "fake", "nli_url": "", "key": True, "nli_sure": SURE, "nli_sure_support": SURE_A}))
    elif argv[:2] == ["judge", "clean"]:
        print(json.dumps({"dir": ""}))
    elif argv[:1] == ["version"]:
        print("mem fake")
    elif argv[:1] == ["review"] and "retired" in argv:
        print("retired\tretired-unfolded\tm3\tm4")
    elif argv[:1] == ["review"]:
        print(json.dumps(review(argv), ensure_ascii=False))
    elif argv[:1] == ["show"]:
        m = dict(MEMS[argv[1]], id=argv[1])
        print(json.dumps(m, ensure_ascii=False))
    elif argv[:2] == ["judge", "support"]:
        return judge_support(argv)
    else:
        print("fake mem: 모르는 명령 " + " ".join(argv), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if "--repo" in args:
        i = args.index("--repo")
        args = args[:i] + args[i + 2:]
    sys.exit(main(args))
