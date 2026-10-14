"""filter_pairs — 고침 쌍 후보(jsonl)에서 판정할 값어치 없는 줄을 까닭별로 거른다 (v1.2 K 판 설계 3-4).

원천을 가리지 않는 일반 규칙이다 : 말(lang) · HTML 줄 · 목록(체크리스트 · 표 · 코드) · 짧음 · 같은 쌍.
남은 줄은 `--sort num_change` 면 수 바뀜 → 한 커밋 고침 수 적은 차례로 세우고, `--cap` 이면 수 바뀜은 다 두고 나머지를 씨앗 고정 무작위로 채운다.

    python -I filter_pairs.py --in <g3>/mit/pairs.jsonl --out <g3>/mit/pairs-f.jsonl \
        --lang ko,mix --drop-html --drop-list --min-chars 15 --sort num_change [--cap 600 --seed 20261013]
"""
import argparse
import collections
import json
import os
import random
import re
import sys

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_HTML_NAMES = ("a|b|i|u|s|em|strong|br|hr|img|sup|sub|p|div|span|summary|details|td|tr|th|table|thead|tbody|"
               "li|ul|ol|code|pre|kbd|mark|del|ins|blockquote|h[1-6]|font|center|figure|figcaption|small|big|tt")
_HTML_TAG = re.compile(r"</?(?:" + _HTML_NAMES + r")\b[^>]*>", re.IGNORECASE)
# 이 태그로 시작하는 줄은 글이 아니라 꾸밈 덩이(그림 · 표 칸 · 접기 제목)다
_HTML_BLOCK = re.compile(r"^\s*<(?:img|td|tr|th|table|thead|tbody|summary|details|div|figure|hr)\b", re.IGNORECASE)
_CHECKLIST = re.compile(r"^\s*(?:[-*+]\s*)?\[[ xX]\]")
_CODE_START = re.compile(r"^\s*(?:public|private|protected|static|class|interface|def|import|from|package|return|"
                         r"const|let|var|function|#include|@[A-Za-z]+)\b")
_CODE_MARKS = ("{", "}", ";", "==", "=>", "()", "::", "->", "!=")
_SPACES = re.compile(r"\s+")

REASONS = ("lang", "html", "checklist", "table", "code", "short", "same", "dup")


def korean_share(text):
    """글자 든 낱말 중 한글 낱말 비율. 한글은 낱말당 글자가 적어 글자 수로 재면 영어 쪽으로 치우친다."""
    words = [w for w in text.split() if _HANGUL.search(w) or _LATIN.search(w)]
    if not words:
        return 0.0
    return sum(1 for w in words if _HANGUL.search(w)) / len(words)


def strip_html(text):
    return _SPACES.sub(" ", _HTML_TAG.sub("", text)).strip()


def is_table(text):
    t = text.strip()
    return t.startswith("|") or t.count("|") >= 2


def is_code(text):
    t = text.strip()
    if t.endswith((";", "{", "}")) or t.startswith("}") or ");" in t or "//" in t.replace("://", ""):
        return True
    # 한글 낱말이 절반 넘는 줄은 설명 글이다(「루트 -> com」 「new(), malloc()」 같은 꼴을 코드로 안 봄)
    if korean_share(t) >= 0.5:
        return False
    return bool(_CODE_START.match(t)) or sum(t.count(m) for m in _CODE_MARKS) >= 2


def keep_lang(row, langs, mix_share):
    lang = row.get("lang", "ko")
    if lang not in langs:
        return False
    if lang != "mix":
        return True
    return min(korean_share(row["evidence"]), korean_share(row["claim"])) >= mix_share


def html_verdict(evidence, claim, min_chars):
    """(버림?, 태그 벗긴 근거, 태그 벗긴 주장). 태그가 없으면 그대로 돌려준다."""
    if not (_HTML_TAG.search(evidence) or _HTML_TAG.search(claim)):
        return False, evidence, claim
    if _HTML_BLOCK.match(evidence) or _HTML_BLOCK.match(claim):
        return True, evidence, claim
    ev, cl = strip_html(evidence), strip_html(claim)
    if ev == cl or min(len(ev), len(cl)) < min_chars:
        return True, evidence, claim
    return False, ev, cl


def list_reason(evidence, claim):
    for text in (evidence, claim):
        if _CHECKLIST.match(text):
            return "checklist"
    for text in (evidence, claim):
        if is_table(text):
            return "table"
    for text in (evidence, claim):
        if is_code(text):
            return "code"
    return ""


def drop_reason(row, opts, seen):
    """버릴 까닭(REASONS 하나) 또는 "". 남기는 줄은 HTML 태그를 벗긴 글로 row 를 고친다."""
    if not keep_lang(row, opts.langs, opts.mix_share):
        return "lang"
    evidence, claim = row["evidence"], row["claim"]
    if opts.drop_html:
        drop, evidence, claim = html_verdict(evidence, claim, opts.min_chars)
        if drop:
            return "html"
    if opts.drop_list:
        reason = list_reason(evidence, claim)
        if reason:
            return reason
    if min(len(evidence.strip()), len(claim.strip())) < opts.min_chars:
        return "short"
    key = (_SPACES.sub(" ", evidence).strip(), _SPACES.sub(" ", claim).strip())
    if key[0] == key[1]:
        return "same"
    if key in seen:
        return "dup"
    seen.add(key)
    if evidence != row["evidence"] or claim != row["claim"]:
        row["evidence"], row["claim"], row["html_stripped"] = evidence, claim, True
    return ""


def sort_key(row):
    return (not row.get("num_change", False), row.get("commit_edits", 0), row.get("id", ""))


def cap_rows(rows, cap, seed):
    """수 바뀜 줄은 다 두고(넘치면 앞 차례대로 자름) 나머지를 씨앗 고정 무작위로 cap 까지."""
    if len(rows) <= cap:
        return rows
    first = [r for r in rows if r.get("num_change")][:cap]
    rest = [r for r in rows if not r.get("num_change")]
    picked = random.Random(seed).sample(rest, cap - len(first))
    keep = {id(r) for r in first + picked}
    return [r for r in rows if id(r) in keep]


def filter_rows(rows, opts):
    seen = set()
    dropped = collections.Counter()
    kept = []
    for row in rows:
        row = dict(row)
        reason = drop_reason(row, opts, seen)
        if reason:
            dropped[reason] += 1
            continue
        kept.append(row)
    if opts.sort == "num_change":
        kept.sort(key=sort_key)
    filtered = len(kept)
    if opts.cap:
        kept = cap_rows(kept, opts.cap, opts.seed)
    summary = {
        "in": len(rows),
        "dropped": {r: dropped.get(r, 0) for r in REASONS},
        "kept": filtered,
        "kept_html_stripped": sum(1 for r in kept if r.get("html_stripped")),
        "out": len(kept),
        "out_num_change": sum(1 for r in kept if r.get("num_change")),
        "out_lang": dict(collections.Counter(r.get("lang", "ko") for r in kept)),
        "options": {"lang": sorted(opts.langs), "mix_share": opts.mix_share, "drop_html": opts.drop_html,
                    "drop_list": opts.drop_list, "min_chars": opts.min_chars, "sort": opts.sort,
                    "cap": opts.cap, "seed": opts.seed},
    }
    return kept, summary


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="고침 쌍 후보 거르기")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="", help="기본 : <out 폴더>/filter.summary.json")
    ap.add_argument("--lang", default="ko,mix", help="남길 lang 값, 쉼표로")
    ap.add_argument("--mix-share", type=float, default=0.5, help="mix 줄은 두 글 다 한글 낱말 비율이 이 값 이상일 때만")
    ap.add_argument("--drop-html", action="store_true")
    ap.add_argument("--drop-list", action="store_true", help="체크리스트 · 표 줄 · 코드 줄")
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--sort", choices=("none", "num_change"), default="none")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261013)
    opts = ap.parse_args(argv)
    opts.langs = set(x.strip() for x in opts.lang.split(",") if x.strip())
    rows, summary = filter_rows(read_jsonl(opts.inp), opts)
    write_jsonl(opts.out, rows)
    path = opts.summary or os.path.join(os.path.dirname(os.path.abspath(opts.out)), "filter.summary.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
