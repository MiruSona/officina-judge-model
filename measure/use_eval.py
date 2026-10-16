"""실사용 측정 use1 — mem 이 실제로 묻는 자리에서 v1.0.0 · v1.1.0 견주기 (비공개 설계).
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
하위명령 차례는 measure/README.md 7항. 모듈마다 use_sites/ 에서 register 로 더하고, 쓰기는 ~/.judgemodel/measure/use1/<별명>/ 안뿐이다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from use_sites import common, register_all  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="실사용 측정 use1 (protocol-use1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    register_all(sub)
    return p


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    a = build_parser().parse_args(argv)
    try:
        a.func(a)
    except common.Refuse as e:
        print(f"멈춤: {e}", file=sys.stderr)
        return common.REFUSE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
