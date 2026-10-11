"""scrub — 쌍을 넣기 전 누출 거르기 · 검사 (설계 d자료층재구축 4-2).

규칙
  1. 작성자 이름 · 메일이 든 줄은 버린다 (Author: · Signed-off-by: · 메일 주소).
  2. 사용자 경로(C:\\Users\\… · /home/… · /Users/…)는 `<경로>` 로 바꾼다.
  3. 키 · 토큰 꼴이 든 쌍은 통째로 버린다 (commit-prep 검사기와 같은 꼴을 이 툴 안에 따로 적음).
  4. 이름 목록(참고 프로젝트 이름 등, git 밖 파일)이 든 쌍은 버린다.
  5. 게임 저장소 이름은 `<게임>` 으로 바꾼다.

    python -I scrub.py apply --in pairs.jsonl --out clean.jsonl --names <목록> --game-names <목록>
    python -I scrub.py check --in clean.jsonl --names <목록> --game-names <목록>     # 0건이어야 exit 0

이름 목록 파일 : 한 줄에 이름 하나('#' 줄은 주석). 마크다운 표면 표마다 머리 줄을 빼고 첫 칸을 읽는다.
"""
import argparse
import collections
import json
import re
import sys

TEXT_FIELDS = ("evidence", "claim", "question", "answer", "removed")
PATH_MARK = "<경로>"
GAME_MARK = "<게임>"

_AUTHOR_LINE = re.compile(r"^\s*[*-]?\s*(authors?|author\(s\)|작성자|쓴 사람|signed-off-by|co-authored-by|reported-by|"
                          r"reviewed-by)\s*[:：]", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_USER_PATHS = (
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/\"'`]+(?:[\\/][^\s\"'`]*)?", re.I),
    re.compile(r"(?<![\w.])/home/[^\s/\"'`]+(?:/[^\s\"'`]*)?"),
    re.compile(r"(?<![\w.])/Users/[^\s/\"'`]+(?:/[^\s\"'`]*)?"),
)
_SECRETS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password|access[_-]?key)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{16,}"),
)


def read_names(paths):
    """이름 목록 파일들 → 이름 목록 (2글자 이상, 긴 것 먼저)."""
    names = set()
    for path in paths or []:
        in_table = False
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    in_table = False
                    continue
                if line.startswith("|"):
                    cells = [c.strip().strip("`*").strip() for c in line.strip("|").split("|")]
                    if not in_table:
                        in_table = True
                        continue
                    if cells and not re.fullmatch(r":?-{2,}:?", cells[0]) and len(cells[0]) >= 2:
                        names.add(cells[0])
                    continue
                in_table = False
                names.add(line.lstrip("-* ").strip())
    return sorted((n for n in names if len(n) >= 2), key=lambda n: (-len(n), n))


def _name_pattern(names):
    if not names:
        return None
    return re.compile("|".join(re.escape(n) for n in names), re.I)


class Scrubber:
    def __init__(self, names=(), game_names=()):
        self.names = _name_pattern(list(names))
        self.games = _name_pattern(list(game_names))

    def clean_text(self, text, counts):
        kept = []
        for line in text.split("\n"):
            if _AUTHOR_LINE.search(line) or _EMAIL.search(line):
                counts["author_line"] += 1
                continue
            kept.append(line)
        text = "\n".join(kept)
        for pattern in _USER_PATHS:
            text, n = pattern.subn(PATH_MARK, text)
            counts["user_path"] += n
        if self.games:
            text, n = self.games.subn(GAME_MARK, text)
            counts["game_name"] += n
        return text

    def apply(self, row, counts):
        """쌍 하나를 거른다. 버릴 쌍이면 None."""
        joined = "\n".join(str(row.get(k) or "") for k in TEXT_FIELDS)
        if any(p.search(joined) for p in _SECRETS):
            counts["drop_secret"] += 1
            return None
        if self.names and self.names.search(joined):
            counts["drop_name"] += 1
            return None
        out = dict(row)
        for key in TEXT_FIELDS:
            if isinstance(out.get(key), str):
                out[key] = self.clean_text(out[key], counts)
        if not (out.get("evidence") or "").strip():
            counts["drop_empty"] += 1
            return None
        return out

    def hits(self, row):
        """남은 누출 꼴 수 (check 모드)."""
        counts = collections.Counter()
        for key in TEXT_FIELDS:
            text = row.get(key)
            if not isinstance(text, str):
                continue
            for line in text.split("\n"):
                if _AUTHOR_LINE.search(line) or _EMAIL.search(line):
                    counts["author_line"] += 1
            counts["user_path"] += sum(len(p.findall(text)) for p in _USER_PATHS)
            counts["secret"] += sum(len(p.findall(text)) for p in _SECRETS)
            if self.names:
                counts["name"] += len(self.names.findall(text))
            if self.games:
                counts["game_name"] += len(self.games.findall(text))
        return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="누출 거르기 · 검사")
    parser.add_argument("mode", choices=("apply", "check"))
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out")
    parser.add_argument("--names", nargs="*", default=[], help="버릴 이름 목록 파일 (git 밖)")
    parser.add_argument("--game-names", nargs="*", default=[], help="<게임> 으로 바꿀 이름 목록 파일 (git 밖)")
    args = parser.parse_args(argv)
    scrubber = Scrubber(read_names(args.names), read_names(args.game_names))
    with open(args.inp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    counts = collections.Counter()
    if args.mode == "apply":
        if not args.out:
            parser.error("apply 는 --out 이 필요하다")
        kept = [r for r in (scrubber.apply(row, counts) for row in rows) if r is not None]
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"in": len(rows), "out": len(kept), **counts}, ensure_ascii=False))
        return 0
    for row in rows:
        counts.update(scrubber.hits(row))
    total = sum(counts.values())
    print(json.dumps({"rows": len(rows), "hits": total, **counts}, ensure_ascii=False))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
