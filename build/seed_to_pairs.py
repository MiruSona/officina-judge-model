"""seed_to_pairs — 다듬은 씨앗(b · a · d 세 줄)을 학습 쌍 꼴로 펴고, v6 두 판정을 합치고, g3 금지 목록을 낸다 (K 판 ⑨).

비공개 설계 3-3 · 4-1. 본문은 git 밖(`~/.judgemodel/build/g3/seed/`)에만 쓴다.

    python -I seed_to_pairs.py pairs  --in <out-450-opus.jsonl> --out <seed>/pairs.jsonl
    python -I seed_to_pairs.py judge  --pairs <pairs.jsonl> --j1 <j1.jsonl> --j2 <j2.jsonl> --out <seed>/judged.jsonl
    python -I seed_to_pairs.py forbid --in <out-450-opus.jsonl> --groups <seed-groups.txt> --pairs <pairs.jsonl> --out <seed>/forbid-g3.txt

judge 규칙 : 두 판이 같으면 그 칸(agree), 다르면 `disagree`(라벨 없음). 같은 씨앗의 줄 중 하나라도
disagree 이거나 확정 칸이 hint 와 다르면 그 씨앗 줄은 **짝째** 버린다(`keep: false`, 설계 3-3 함정 · 4-1 짝 규칙).
"""
import argparse
import collections
import hashlib
import json
import os
import sys

HINTS = ("b", "a", "d")
LABELS = ("a", "b", "c", "d", "cd")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_pairs(row):
    """씨앗 한 줄 → 쌍 줄들 (b 는 늘 · a · d 는 있을 때만). 근거 = 씨앗, 주장 = 다듬은 줄.
    씨앗 줄에 evidence(문단)가 있으면(seed_contrast --evidence paragraph, K4 archive) 근거 = 그 문단 · domain = 줄의 domain ·
    form = seed-<칸> · origin = g4-seed. 없으면 옛 꼴 그대로.
    줄에 lines 가 없고 drafts(seed_contrast make 의 규칙 초안)가 있으면 drafts 를 그대로 쓰고
    쌍 줄에 lines_from = rule-draft 를 남긴다. lines 가 있으면 옛 출력 그대로."""
    paragraph = row.get("evidence_mode") == "paragraph"
    from_draft = "lines" not in row and isinstance(row.get("drafts"), dict)
    lines = row["drafts"] if from_draft else (row.get("lines") or {})
    out = []
    for hint in HINTS:
        claim = lines.get(hint)
        if not isinstance(claim, str) or not claim.strip():
            continue
        out.append({
            "id": f"seed-{row['seed_id']}-{hint}", "evidence": row["evidence"] if paragraph else row["seed"],
            "claim": claim, "want": "", "hint": hint, "form": f"seed-{hint}" if paragraph else "contrast",
            "domain": row.get("domain", "archive") if paragraph else "seed", "group": row["group"], "groups": [row["group"]],
            "lang": "ko", "lic": "MIT", "split": "train", "src_ids": [row["seed_id"]],
            "origin": "g4-seed" if paragraph else "g3-seed",
            "pair": row["seed_id"], "slot_kind": (row.get("slot") or {}).get("kind"),
        })
        if from_draft:
            out[-1]["lines_from"] = "rule-draft"
    return out


def pair_key(evidence, claim):
    """blind-g2.forbid 의 key_rule 과 같은 열쇠."""
    return hashlib.sha256((evidence.strip() + "\x1f" + claim.strip()).encode("utf-8")).hexdigest()


def judge_rows(pairs, j1, j2):
    """두 판 답({id, label, reason}) → judged 줄 + 요약. 빠진 답이 있으면 ValueError."""
    a1 = {r["id"]: r for r in j1}
    a2 = {r["id"]: r for r in j2}
    missing = [p["id"] for p in pairs if p["id"] not in a1 or p["id"] not in a2]
    bad = [i for d in (a1, a2) for i, r in d.items() if r.get("label") not in LABELS]
    if missing or bad:
        raise ValueError(f"빠진 답 {missing[:5]} · 칸 밖 답 {bad[:5]}")
    rows = []
    for p in pairs:
        l1, l2 = a1[p["id"]]["label"], a2[p["id"]]["label"]
        final = l1 if l1 == l2 else "disagree"
        reason = a1[p["id"]].get("reason", "") if l1 == l2 else f"j1 {l1}: {a1[p['id']].get('reason', '')} / j2 {l2}: {a2[p['id']].get('reason', '')}"
        row = dict(p, judge={"label1": l1, "label2": l2, "final": final, "reason": reason}, label_guide="support-v6")
        if final != "disagree":
            row["want"] = final
        rows.append(row)
    bad_seed = {r["pair"] for r in rows if r["judge"]["final"] != r["hint"]}
    for r in rows:
        r["keep"] = r["pair"] not in bad_seed
    return rows, summarize(rows)


def summarize(rows):
    by = {}
    for hint in HINTS:
        sub = [r for r in rows if r["hint"] == hint]
        finals = collections.Counter(r["judge"]["final"] for r in sub)
        by[hint] = {
            "n": len(sub), "decided": sum(v for k, v in finals.items() if k != "disagree"),
            "labels": dict(sorted((k, v) for k, v in finals.items() if k != "disagree")),
            "disagree": finals.get("disagree", 0),
            "off_hint": sum(1 for r in sub if r["judge"]["final"] not in ("disagree", hint)),
            "kept": sum(1 for r in sub if r["keep"]),
        }
    seeds = collections.defaultdict(list)
    for r in rows:
        seeds[r["pair"]].append(r)
    return {
        "n": len(rows), "label_guide": "support-v6", "by_hint": by,
        "disagree": sum(1 for r in rows if r["judge"]["final"] == "disagree"),
        "seeds": len(seeds), "seeds_kept": sum(1 for s in seeds.values() if s[0]["keep"]),
        "kept_rows": sum(1 for r in rows if r["keep"]),
    }


def forbid_lines(seeds, groups, pairs):
    """g3 평가 금지 목록 줄. 칸은 blind-g2.forbid.json 과 같은 이름(groups · doc_paths · mem_ids · pair_keys)."""
    grp = set(groups) | {s["group"] for s in seeds}
    docs = sorted(g[len("doc:"):] for g in grp if g.startswith("doc:"))
    mems = sorted(g[len("mem:"):] for g in grp if g.startswith("mem:"))
    keys = sorted({pair_key(p["evidence"], p["claim"]) for p in pairs})
    lines = ["# g3 씨앗(K 판 3-3)으로 쓴 것 — g3 예약 · 평가 셋에서 문서째 뺀다. 줄 꼴 : <칸>\\t<값>",
             "# 칸 이름은 blind-g2.forbid.json 과 같다. pair_keys = sha256(evidence.strip() + '\\x1f' + claim.strip())",
             f"# groups {len(grp)} · doc_paths {len(docs)} · mem_ids {len(mems)} · pair_keys {len(keys)}"]
    lines += [f"groups\t{g}" for g in sorted(grp)]
    lines += [f"doc_paths\t{d}" for d in docs]
    lines += [f"mem_ids\t{m}" for m in mems]
    lines += [f"pair_keys\t{k}" for k in keys]
    return lines


def read_groups(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--pairs", required=True)
    j.add_argument("--j1", required=True)
    j.add_argument("--j2", required=True)
    j.add_argument("--out", required=True)
    f = sub.add_parser("forbid")
    f.add_argument("--in", dest="inp", required=True)
    f.add_argument("--groups", required=True)
    f.add_argument("--pairs", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "pairs":
        rows = [p for s in read_jsonl(args.inp) for p in seed_pairs(s)]
        write_jsonl(args.out, rows)
        print(json.dumps(collections.Counter(r["hint"] for r in rows), ensure_ascii=False))
    elif args.cmd == "judge":
        try:
            rows, summary = judge_rows(read_jsonl(args.pairs), read_jsonl(args.j1), read_jsonl(args.j2))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        write_jsonl(args.out, rows)
        with open(os.path.splitext(args.out)[0] + ".summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False))
    else:
        lines = forbid_lines(read_jsonl(args.inp), read_groups(args.groups), read_jsonl(args.pairs))
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print(lines[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
