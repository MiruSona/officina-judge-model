"""rule_filter — v6 자리 규칙 ①~④ 에 걸리는 「거의 같은 문장」 쌍 거르기.

설계 `../Docs/Design/2026-10-12-v1.2-B판설계.md` 4-2, 기준 `../Docs/Guide/라벨기준support-v6.md` 3절.
근거에서 주장과 가장 닮은 문장을 고르고, 주장 쪽에서 바뀐 낱말이 무엇인지 보고 자리 하나를 붙인다.
지금 라벨(want)은 안 본다. 쌍 본문은 화면에 찍지 않는다.

    python -I rule_filter.py pick --sets <셋.jsonl> [<셋.jsonl> …] --out <걸린 쌍.jsonl> [--cap 200] [--seed 20261213]

rule 값 : 1 수 · 비교 한정어 · 때 비교 / 2 서수 · 순위 / 3 낱말 하나 바꿈 / 4 근거 표지 번호만 다름.
"""
import argparse
import collections
import difflib
import json
import os
import random
import re
import sys

SIM_MIN = 0.5            # 주장 낱말 중 근거 문장에도 있는 몫의 아래끝
SIM_MIN_QUALIFIED = 0.3  # 주장에 근거에 없는 비교 한정어가 있을 때의 아래끝 (한정어 주장은 말을 많이 바꾼다)
CHANGE_MAX = 6           # 주장에만 있는 낱말 수의 위끝 (넘으면 「거의 같은 문장」이 아니다)
CHANGE_MAX_QUALIFIED = 12
WORD_SWAP_MAX = 3        # ③ 낱말 하나 바꿈으로 볼 주장에만 있는 낱말 수의 위끝
RARE_MAX = 30            # cap 으로 자를 때 통째로 남기는 드문 자리의 쌍 수 위끝
SEED = 20261213

TOKEN = re.compile(r"[A-Za-z]*\d+(?:[-.,/:]\d+)*(?:%|[A-Za-z가-힣]+)?|\w+", re.UNICODE)
SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)|\n+")
HANGUL_END = re.compile(r"(으로|에서|에게|까지|부터|은|는|이|가|을|를|에|의|로|도|와|과|만)$")
STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "is", "are", "was",
             "were", "be", "been", "has", "have", "had", "it", "its", "this", "that", "as", "s", "from"}
EN_QUALIFIERS = {"less", "more", "fewer", "than", "before", "after", "over", "under", "above", "below",
                 "least", "most", "exceed", "exceeds", "since", "until", "prior", "higher", "lower"}
KO_QUALIFIERS = ("넘", "이상", "이하", "미만", "초과", "아래", "앞", "뒤", "이전", "이후", "전에", "후에", "까지", "보다")
ORDINAL = re.compile(r"^(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
                     r"|\d+(위|등|번째|차|째)|첫째|둘째|셋째|넷째)$")
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"}
MARKER = re.compile(r"^([a-z]{1,2}\d+|\d+-\d+)$")
NUMBER = re.compile(r"\d")


def normalize(word):
    """소문자 · 한글 낱말 끝 조사 하나 떼기."""
    word = word.lower()
    if re.match(r"^[가-힣]{3,}$", word):
        return HANGUL_END.sub("", word)
    return word


def tokens(text):
    """글 → 낱말 목록 (불용어 뺌)."""
    words = (normalize(w) for w in TOKEN.findall(str(text or "")))
    return [w for w in words if w and w not in STOPWORDS]


def name_words(text):
    """대문자로 시작하는 낱말 (주어를 이름으로 푼 말). 한정어 · 수는 빼고 센다."""
    return {w.lower() for w in TOKEN.findall(str(text or "")) if w[:1].isupper()}


def sentences(text):
    parts = [p.strip() for p in SENTENCE_END.split(str(text or "")) if p and p.strip()]
    return parts or [str(text or "")]


def is_count(word):
    return bool(NUMBER.search(word)) or word in NUMBER_WORDS


def is_signal(word):
    return word in EN_QUALIFIERS or is_count(word) or any(mark in word for mark in KO_QUALIFIERS)


def claim_cover(sentence_tokens, claim_tokens):
    """주장 낱말 중 근거 문장에도 있는 몫 (0 ~ 1). 차례는 안 본다 — 말 차례를 바꾼 주장도 잡으려고."""
    if not claim_tokens:
        return 0.0
    seen = set(sentence_tokens)
    return sum(1 for word in claim_tokens if word in seen) / len(claim_tokens)


def best_sentence(evidence, claim_tokens):
    """근거 문장 중 주장과 가장 닮은 것 → (낱말 목록, 닮음)."""
    best, best_cover = [], 0.0
    for sentence in sentences(evidence):
        sentence_tokens = tokens(sentence)
        cover = claim_cover(sentence_tokens, claim_tokens)
        if cover > best_cover:
            best, best_cover = sentence_tokens, cover
    return best, best_cover


def changed_words(sentence_tokens, claim_tokens):
    """근거 문장에만 있는 낱말(old) · 주장에만 있는 낱말(new)."""
    claim_set, sentence_set = set(claim_tokens), set(sentence_tokens)
    old = [word for word in sentence_tokens if word not in claim_set]
    new = [word for word in claim_tokens if word not in sentence_set]
    return old, new


def has_qualifier(words):
    """주장에만 있는 낱말에 비교 한정어가 있나."""
    for word in words:
        if word in EN_QUALIFIERS:
            return True
        if any(mark in word for mark in KO_QUALIFIERS):
            return True
    return False


def claim_words(evidence, claim):
    """주장 낱말 중 근거에 없는 이름 낱말은 뺀다 (VitaminC 주장은 대명사를 이름으로 푼다)."""
    evidence_set = set(tokens(evidence))
    names = name_words(claim)
    return [w for w in tokens(claim) if w in evidence_set or w not in names or is_signal(w)]


def rule_of(evidence, claim):
    """쌍 하나 → (rule, 닮음). 걸리지 않으면 rule 0. 차례 : ④ 표지 > ② 서수 > ① 수 · 한정어 > ③ 낱말 하나."""
    claim_tokens = claim_words(evidence, claim)
    sentence_tokens, cover = best_sentence(evidence, claim_tokens)
    old, new = changed_words(sentence_tokens, claim_tokens)
    if not new and not old:
        return 0, cover
    qualifier = has_qualifier(new)
    if qualifier:
        near = cover >= SIM_MIN_QUALIFIED and len(new) <= CHANGE_MAX_QUALIFIED
    else:
        near = cover >= SIM_MIN and len(new) <= CHANGE_MAX
    if not near:
        return 0, cover
    marker_new = any(MARKER.match(word) for word in new)
    marker_old = any(MARKER.match(word) for word in old)
    if (marker_new and marker_old) or ((marker_new or marker_old) and len(new) <= 2):
        return 4, cover
    if any(ORDINAL.match(word) for word in new):
        return 2, cover
    if any(ORDINAL.match(word) for word in old) and (qualifier or any(is_count(word) for word in new)):
        return 2, cover
    if qualifier or any(is_count(word) for word in new):
        return 1, cover
    if len(new) <= WORD_SWAP_MAX:
        return 3, cover
    return 0, cover


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cap_keep_rare(picked, cap, seed):
    """cap 으로 자르되 드문 자리(RARE_MAX 이하)는 통째로 남긴다 — 자리별 표가 비지 않게. 드문 쪽이 cap 을 넘으면 그냥 무작위."""
    by_rule = collections.defaultdict(list)
    for row in picked:
        by_rule[row["rule"]].append(row)
    rare = [row for rows in by_rule.values() if len(rows) <= RARE_MAX for row in rows]
    common = [row for rows in by_rule.values() if len(rows) > RARE_MAX for row in rows]
    rng = random.Random(seed)
    if len(rare) >= cap or not common:
        return rng.sample(picked, cap)
    return rare + rng.sample(common, cap - len(rare))


def pick(rows, cap=0, seed=SEED):
    """셋 줄들 → 걸린 줄(원래 줄 + rule · sim). cap 이 있으면 씨앗 고정으로 무작위 자른다."""
    picked = []
    for row in rows:
        rule, cover = rule_of(row.get("evidence", ""), row.get("claim", ""))
        if not rule:
            continue
        picked.append(dict(row, rule=rule, sim=round(cover, 3)))
    if cap and len(picked) > cap:
        picked = cap_keep_rare(picked, cap, seed)
    picked.sort(key=lambda r: r["id"])
    return picked


def cmd_pick(a):
    rows = []
    for path in a.sets:
        rows.extend(read_jsonl(path))
    found = pick(rows)
    kept = pick(rows, a.cap, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"rows": len(rows), "found": len(found), "kept": len(kept),
              "rule": dict(sorted(collections.Counter(r["rule"] for r in kept).items())),
              "want": dict(sorted(collections.Counter(r.get("want", "") for r in kept).items()))}
    print(json.dumps(report, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pick", help="셋 → 자리 규칙에 걸린 쌍")
    s.add_argument("--sets", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cap", type=int, default=0)
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_pick)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
