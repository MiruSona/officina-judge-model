"""pick_contrast — v1.2 B 판 학습 층 train-g2-b : VitaminC 대조쌍 짝 고르기 · 판정 뒤 짝 규칙.

비공개 설계 5-1. 쌍 본문은 화면에 찍지 않는다.

    python -I pick_contrast.py pick --cands <g2>/reserve/train-layer-cands.jsonl --sets <얼린 셋.jsonl> … \
        --forbid <셋.forbid.json> … --out <g2>/train-g2-b/cands.jsonl [--n 650] [--qual-share 0.55]
    python -I pick_contrast.py keep-pairs --set <assemble 결과.jsonl> --cands <위 cands.jsonl> --out <train-g2-b.jsonl>

pick : 같은 `pair`(같은 판 고침 · 같은 주장)의 a · b 두 줄을 짝째 고른다. 위키 문서(group) 하나에 한 짝,
얼린 평가 셋 · 금지 목록과 묶음 · 출처 id · 글이 겹치면 뺀다. 한정어 · 수 · 서수 주장(`qual`)을 qual-share 만큼 먼저,
나머지는 낱말 하나 바꿈(`swap`, rule_filter ③) → 그 밖(`other`) 차례로 채운다.
    python -I pick_contrast.py halve --set <train-g2-b.jsonl> --out <train-g2-b-half.jsonl> [--seed 20261303] [--by pick]

keep-pairs : 판정 뒤 a 줄이 a · b 줄이 b 로 둘 다 남은 짝만 넣는다. c · d · cd 가 하나라도 나오거나, 한쪽이 버려지거나,
칸이 hint 와 어긋나면 짝째 버리고 까닭별로 센다.
halve : 대체 판 B4h 층(설계 5-3) — 원천 칸(--by, 기본 pick : qual · swap)마다 짝 수의 절반(내림)을 고정 씨앗으로 짝째 고른다.
줄 차례는 원래 파일 차례 그대로 둔다.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rule_filter  # noqa: E402
from make_sets import norm, pair_key  # noqa: E402

SEED = 20261301
HALF_SEED = 20261303   # B4h 절반 뽑기 (고르기 20261301 · 묶음 20261302 다음)
QUAL_WORDS = {"less", "more", "fewer", "than", "before", "after", "over", "under"}   # 진단 1-4 의 낱말 표
GROUP_CAP = 1


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def forbid_hash(evidence, claim):
    """forbid.json 의 key_rule — sha256(evidence.strip() + '\\x1f' + claim.strip())."""
    return hashlib.sha256((evidence.strip() + "\x1f" + claim.strip()).encode("utf-8")).hexdigest()


def claim_kind(evidence_b, claim):
    """주장 꼴 : qual (한정어 · 수 · 서수) / swap (b 근거와 낱말 하나 바꿈) / other."""
    for word in rule_filter.TOKEN.findall(claim):
        low = word.lower()
        if low in QUAL_WORDS or rule_filter.is_count(low) or rule_filter.ORDINAL.match(low):
            return "qual"
    rule, _ = rule_filter.rule_of(evidence_b, claim)
    if rule == 3:
        return "swap"
    return "other"


class Forbid:
    """얼린 평가 셋 · 금지 목록에서 묶음 · 출처 id · 글(쌍 · 한쪽)을 모은다."""

    def __init__(self, set_paths, forbid_paths):
        self.groups, self.src_ids, self.keys, self.hashes, self.sides = set(), set(), set(), set(), set()
        for path in set_paths:
            for row in read_jsonl(path):
                self.groups.update(row.get("groups") or [row.get("group")])
                self.src_ids.update(row.get("src_ids", []))
                self.keys.add(pair_key(row))
                self.hashes.add(forbid_hash(row.get("evidence", ""), row.get("claim", "")))
                self.sides.update({norm(row.get("evidence")), norm(row.get("claim"))})
        for path in forbid_paths:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            self.groups.update(doc.get("groups", []))
            self.src_ids.update(doc.get("src_ids", []))
            self.hashes.update(doc.get("pair_keys", []))
        self.sides.discard("")

    def reasons(self, row):
        out = []
        if set(row.get("groups") or [row.get("group")]) & self.groups:
            out.append("group")
        if set(row.get("src_ids", [])) & self.src_ids:
            out.append("src")
        if pair_key(row) in self.keys or forbid_hash(row["evidence"], row["claim"]) in self.hashes:
            out.append("text")
        elif {norm(row["evidence"]), norm(row["claim"])} & self.sides:
            out.append("side")
        return out


def pairs_of(rows):
    """줄 → {pair: {hint: 줄}}. a · b 가 하나씩 다 있는 짝만."""
    by_pair = collections.defaultdict(dict)
    for row in rows:
        if row.get("g2_source", "vitaminc-train") != "vitaminc-train" or not row.get("pair"):
            continue
        by_pair[row["pair"]][row["hint"]] = row
    return {p: rows for p, rows in by_pair.items() if set(rows) == {"a", "b"}}


def pick(rows, forbid, n=650, qual_share=0.55, seed=SEED, group_cap=GROUP_CAP):
    """→ (고른 줄, 요약). 짝째 고르고 줄은 a · b 둘 다 낸다."""
    pairs = pairs_of(rows)
    dropped = collections.Counter()
    pools = {"qual": [], "swap": [], "other": []}
    for pid in sorted(pairs):
        a, b = pairs[pid]["a"], pairs[pid]["b"]
        why = set(forbid.reasons(a)) | set(forbid.reasons(b))
        if why:
            dropped.update(why)
            continue
        pools[claim_kind(b["evidence"], b["claim"])].append(pid)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    want = {"qual": math.ceil(n * qual_share)}
    used = collections.Counter()
    chosen = []

    def take(kind, limit):
        got = 0
        for pid in pools[kind]:
            if got >= limit:
                break
            group = pairs[pid]["b"]["group"]
            if used[group] >= group_cap:
                continue
            used[group] += 1
            chosen.append((pid, kind))
            got += 1
        return got

    got_qual = take("qual", want["qual"])
    got_swap = take("swap", n - got_qual)
    got_other = take("other", n - got_qual - got_swap)
    out = []
    for pid, kind in sorted(chosen):
        for hint in ("a", "b"):
            out.append(dict(pairs[pid][hint], pick=kind))
    summary = {"pairs_in": len(pairs), "pairs_ok": sum(len(p) for p in pools.values()),
               "pools": {k: len(v) for k, v in pools.items()}, "dropped": dict(dropped),
               "chosen": {"qual": got_qual, "swap": got_swap, "other": got_other, "pairs": len(chosen)},
               "qual_share": round(got_qual / len(chosen), 3) if chosen else 0.0,
               "groups": len(used), "seed": seed, "n": n, "group_cap": group_cap}
    return out, summary


def keep_pairs(set_rows, cand_rows):
    """판정 뒤 a 줄 a · b 줄 b 로 다 남은 짝만 → (남긴 줄, 요약)."""
    judged = {row["id"]: row for row in set_rows}
    pairs = pairs_of(cand_rows)
    kept, why = [], collections.Counter()
    kinds = collections.Counter()
    for pid in sorted(pairs):
        a, b = pairs[pid]["a"], pairs[pid]["b"]
        ja, jb = judged.get(a["id"]), judged.get(b["id"])
        if ja is None or jb is None:
            why["judged_dropped"] += 1
            continue
        labels = (ja["want"], jb["want"])
        if any(label in ("c", "d", "cd") for label in labels):
            why["c_or_d"] += 1
            continue
        if labels != ("a", "b"):
            why["cell_mismatch:" + "".join(labels)] += 1
            continue
        kept.extend([dict(ja, pair=pid, pick=a.get("pick", "")), dict(jb, pair=pid, pick=b.get("pick", ""))])
        kinds[a.get("pick", "")] += 1
    n_pairs = len(kept) // 2
    summary = {"pairs_in": len(pairs), "pairs_kept": n_pairs, "rows": len(kept), "dropped": dict(why),
               "want": dict(collections.Counter(r["want"] for r in kept)), "pick": dict(kinds),
               "qual_share": round(kinds["qual"] / n_pairs, 3) if n_pairs else 0.0}
    return kept, summary


def halve(rows, seed=HALF_SEED, by="pick"):
    """원천 칸마다 짝(pair)의 절반(내림)을 짝째 고른다. 돌려줌 (줄, 요약). 짝이 a · b 둘이 아니면 거부."""
    pairs = collections.OrderedDict()
    for row in rows:
        pairs.setdefault(row["pair"], []).append(row)
    bad = [k for k, v in pairs.items() if sorted(r["want"] for r in v) != ["a", "b"]]
    if bad:
        raise ValueError(f"a · b 짝이 아닌 pair {len(bad)}개")
    strata = collections.defaultdict(list)
    for key, members in pairs.items():
        strata[str(members[0].get(by))].append(key)
    rng = random.Random(seed)
    keep, summary = set(), {"seed": seed, "by": by, "strata": {}}
    for name in sorted(strata):
        keys = sorted(strata[name])
        chosen = rng.sample(keys, len(keys) // 2)
        keep.update(chosen)
        summary["strata"][name] = {"pairs": len(keys), "kept": len(chosen)}
    out = [r for r in rows if r["pair"] in keep]
    summary["pairs"] = len(keep)
    summary["want"] = dict(sorted(collections.Counter(r["want"] for r in out).items()))
    return out, summary


def cmd_halve(a):
    rows, summary = halve(read_jsonl(a.set), a.seed, a.by)
    write_jsonl(a.out, rows)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_pick(a):
    forbid = Forbid(a.sets, a.forbid)
    rows, summary = pick(read_jsonl(a.cands), forbid, a.n, a.qual_share, a.seed, a.group_cap)
    write_jsonl(a.out, rows)
    with open(a.out + ".summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["chosen"]["pairs"] == a.n else 1


def cmd_keep_pairs(a):
    rows, summary = keep_pairs(read_jsonl(a.set), read_jsonl(a.cands))
    write_jsonl(a.out, rows)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pick", help="대조쌍 짝 고르기")
    s.add_argument("--cands", required=True)
    s.add_argument("--sets", nargs="+", required=True, help="얼린 평가 셋 (dev-cal-g2 · dev-cmp-g2 · blind-g2)")
    s.add_argument("--forbid", nargs="*", default=[], help="<셋>.forbid.json")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=650)
    s.add_argument("--qual-share", type=float, default=0.55)
    s.add_argument("--group-cap", type=int, default=GROUP_CAP)
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_pick)
    s = sub.add_parser("keep-pairs", help="판정 뒤 짝 규칙 — a · b 둘 다 남은 짝만")
    s.add_argument("--set", required=True, help="label_batches assemble --mode train 결과")
    s.add_argument("--cands", required=True, help="pick 이 낸 후보")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_keep_pairs)
    s = sub.add_parser("halve", help="B4h 층 — 원천 칸마다 짝 절반을 고정 씨앗으로")
    s.add_argument("--set", required=True, help="train-g2-b.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=HALF_SEED)
    s.add_argument("--by", default="pick", help="원천 칸 (기본 pick : qual · swap)")
    s.set_defaults(func=cmd_halve)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
