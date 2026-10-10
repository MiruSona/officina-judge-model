"""판정 셋 얼리기 — jsonl 의 SHA-256 과 갈래 수를 FROZEN.sha256 에 적고, 잴 때 바뀌지 않았는지 본다 (설계 6-2).

    python -I freeze.py write --pairs <셋.jsonl> --out <폴더>/FROZEN.sha256
    python -I freeze.py check --pairs <셋.jsonl> --frozen <폴더>/FROZEN.sha256     # 다르면 exit 1

저장소에는 SHA 와 갈래 수만 남긴다. 셋 본문은 git 밖에 둔다.
"""
import argparse
import collections
import hashlib
import json
import os
import sys


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def want_counts(path):
    counts = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_meta" in row:
                continue
            counts[str(row.get("want", "-"))] += 1
    return dict(sorted(counts.items()))


def read_frozen(path):
    """FROZEN.sha256 의 (sha, 파일 이름). 주석 줄은 건너뛴다."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                digest, name = line.split(None, 1)
                return digest, name
    raise SystemExit(f"SHA 줄이 없다: {path}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="판정 셋 얼리기")
    p.add_argument("cmd", choices=["write", "check"])
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="", help="write : FROZEN.sha256 경로")
    p.add_argument("--frozen", default="", help="check : FROZEN.sha256 경로")
    a = p.parse_args(argv)
    digest = sha256_file(a.pairs)
    counts = want_counts(a.pairs)
    if a.cmd == "write":
        if not a.out:
            raise SystemExit("--out 이 있어야 한다")
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# want {json.dumps(counts, ensure_ascii=False)}\n")
            f.write(f"{digest}  {os.path.basename(a.pairs)}\n")
        print(json.dumps({"sha256": digest, "want": counts}, ensure_ascii=False))
        return 0
    want, name = read_frozen(a.frozen)
    ok = want == digest
    print(json.dumps({"ok": ok, "sha256": digest, "frozen": want, "file": name, "want": counts}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
