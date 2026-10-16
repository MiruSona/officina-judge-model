"""시험용 가짜 mem — use_eval harvest-reports · log-report 가 부르는 search 만 흉내 낸다 (test_use_reports.py).

search --facet scope 는 scope 목록, 그 밖의 search 는 그 scope 의 결정을 날짜 차례로 낸다(질의는 안 본다).
"""
import json
import sys

DECISIONS = [
    {"id": "20261001-aaaa0001", "date": "2026-10-01", "scope": "judgemodel", "summary": "기본 서버 포트는 8092 로 둔다"},
    {"id": "20261002-aaaa0002", "date": "2026-10-02", "scope": "judgemodel", "summary": "측정 결과는 git 밖에만 둔다"},
    {"id": "20261005-aaaa0003", "date": "2026-10-05", "scope": "officina", "summary": "커밋은 사용자 승인 뒤에만 한다"},
    {"id": "20261005-aaaa0004", "date": "2026-10-05", "scope": "officina", "summary": "키는 sk-TESTABCDEFGHIJKLMNOPQR 를 쓴다"},
    {"id": "20261010-aaaa0005", "date": "2026-10-10", "scope": "judgemodel", "summary": "보고가 남긴 바로 그 결정이다"},
    {"id": "20261012-aaaa0006", "date": "2026-10-12", "scope": "judgemodel", "summary": "보고 날 뒤에 생긴 결정이다"},
]


def arg(argv, name, default=""):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


def main(argv):
    if argv[:1] != ["search"]:
        print("fake mem: 모르는 명령 " + " ".join(argv), file=sys.stderr)
        return 1
    if "--facet" in argv:
        names = sorted({d["scope"] for d in DECISIONS} | {"arttool"})
        print(json.dumps([{"name": n, "count": 1} for n in names]))
        return 0
    scope = arg(argv, "--scope")
    hits = [d for d in DECISIONS if d["scope"] == scope]
    hits.sort(key=lambda d: d["date"], reverse=True)
    print(json.dumps({"hits": hits[:int(arg(argv, "--limit", "10"))]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if "--repo" in args:
        i = args.index("--repo")
        args = args[:i] + args[i + 2:]
    sys.exit(main(args))
