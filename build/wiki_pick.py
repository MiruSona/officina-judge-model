"""wiki_pick — 거른 위키 후보(`wiki-all-{eval,train}.jsonl`)에서 판정할 줄을 뽑고, v6 두 판정을 합친다 (K 판 위키 몫).

비공개 설계 2-3 · 3-1 · 4-1. 본문은 git 밖(`~/.judgemodel/build/g3/kowiki/`)에만 쓴다.

    python -I wiki_pick.py pick  --in <wiki-all-eval.jsonl> --out <pick-eval.jsonl> --set eval --n-b 260 [--seed 20261013]
    python -I wiki_pick.py judge --pairs <pick-eval.jsonl> --j1 <j1.jsonl> --j2 <j2.jsonl> --out <judged-eval.jsonl>
    python -I wiki_pick.py finalize --in <judged-eval.jsonl> <judged-train-A.jsonl> … --out-dir <kowiki> [--override <id>=<label> …]

pick 규칙 : hint b 는 kind(num · date · ordinal) 비율대로 몫을 나눠(큰 나머지) 고정 씨앗으로 섞은 차례에서 고른다.
문서(group)마다 b 는 `--per-group`(2)개까지, flag 붙은 b 는 kind 몫마다 `--flag-cap`(0.2) 비율까지. hint d 는 전부 넣는다.
judge 규칙 : 두 판이 같으면 그 칸(`want` 에도 넣음), 다르면 `disagree`. disagree · noise 는 `keep: false`.
판 파일 줄은 `{id, label, reason}` 이고 label 은 a · b · c · d · noise 중 하나다.
finalize 규칙 : `keep` 인 줄만 남겨 `hint` 를 확정 칸으로 바꾼다(뽑을 때 hint 는 `hint_pick`). `--override` 는 갈린 줄을
사람이 정한 칸으로 되돌린다(`judge.override` 에 옛 final). `pick_set` 으로 평가 · 학습을 가르고, 두 셋의 문서가 겹치면 멈춘다.
"""
import argparse
import collections
import json
import os
import random
import sys

LABELS = ("a", "b", "c", "d", "noise")
GUIDE = "support-v6"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row):
    """옛 판 · 새 판 번호로 만든 id. 같은 판 쌍은 한 줄뿐이다(merge 가 같은 주장을 뺐다)."""
    revs = [s.split(":", 1)[-1] for s in row["src_ids"]]
    return "wiki-" + "-".join(revs)


def quotas(counts, n):
    """kind 별 몫. 비율대로 내림한 뒤 남는 수를 나머지 큰 kind 부터 1씩 준다(같으면 이름 차례)."""
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in counts}
    raw = {k: n * v / total for k, v in counts.items()}
    out = {k: int(x) for k, x in raw.items()}
    left = min(n, total) - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:max(left, 0)]:
        out[k] += 1
    return out


def pick_b(rows, n, seed, per_group=2, flag_cap=0.2):
    """hint b 줄에서 n 개. 돌려주는 값 : (뽑은 줄, kind 별 몫)."""
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    quota = quotas({k: len(v) for k, v in by_kind.items()}, n)
    rng = random.Random(seed)
    per_doc = collections.Counter()
    picked = []
    for kind in sorted(by_kind):
        pool = sorted(by_kind[kind], key=row_id)
        rng.shuffle(pool)
        want, flag_max = quota[kind], int(quota[kind] * flag_cap)
        got = flagged = 0
        for r in pool:
            if got >= want:
                break
            if per_doc[r["group"]] >= per_group:
                continue
            if r.get("flag") and flagged >= flag_max:
                continue
            per_doc[r["group"]] += 1
            flagged += bool(r.get("flag"))
            got += 1
            picked.append(r)
    return picked, quota


def pick_rows(rows, n_b, set_name, seed, per_group=2, flag_cap=0.2):
    """뽑은 b + 모든 d. 줄마다 `id` · `pick_set` 을 단다. b 몫을 못 채우면 ValueError."""
    b_rows = [r for r in rows if r["hint"] == "b"]
    d_rows = [r for r in rows if r["hint"] == "d"]
    picked, quota = pick_b(b_rows, n_b, seed, per_group, flag_cap)
    if len(picked) < n_b:
        raise ValueError(f"b 몫 {n_b} 중 {len(picked)} 만 뽑혔다 (문서 · flag 상한)")
    out = [dict(r, id=row_id(r), pick_set=set_name) for r in picked + d_rows]
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("id 가 겹친다")
    return out, quota


def pick_stats(rows):
    b = [r for r in rows if r["hint"] == "b"]
    return {
        "n": len(rows), "b": len(b), "d": len(rows) - len(b),
        "b_kind": dict(sorted(collections.Counter(r["kind"] for r in b).items())),
        "b_flagged": sum(1 for r in b if r.get("flag")),
        "b_docs": len({r["group"] for r in b}),
        "b_max_per_doc": max(collections.Counter(r["group"] for r in b).values(), default=0),
        "d_kind": dict(sorted(collections.Counter(r["kind"] for r in rows if r["hint"] == "d").items())),
        "docs": len({r["group"] for r in rows}),
    }


def judge_rows(pairs, j1, j2):
    """두 판 답 → judged 줄 + 요약. 빠진 답 · 칸 밖 답이 있으면 ValueError."""
    a1 = {r["id"]: r for r in j1}
    a2 = {r["id"]: r for r in j2}
    missing = [p["id"] for p in pairs if p["id"] not in a1 or p["id"] not in a2]
    bad = [i for d in (a1, a2) for i, r in d.items() if r.get("label") not in LABELS]
    if missing or bad:
        raise ValueError(f"빠진 답 {missing[:5]} · 칸 밖 답 {bad[:5]}")
    rows = []
    for p in pairs:
        r1, r2 = a1[p["id"]], a2[p["id"]]
        l1, l2 = r1["label"], r2["label"]
        final = l1 if l1 == l2 else "disagree"
        reason = r1.get("reason", "") if l1 == l2 else f"j1 {l1}: {r1.get('reason', '')} / j2 {l2}: {r2.get('reason', '')}"
        row = dict(p, judge={"label1": l1, "label2": l2, "final": final, "reason": reason}, label_guide=GUIDE)
        if final not in ("disagree", "noise"):
            row["want"] = final
        row["keep"] = final not in ("disagree", "noise")
        rows.append(row)
    return rows, summarize(rows)


def summarize(rows):
    by = {}
    for hint in sorted({r["hint"] for r in rows}):
        sub = [r for r in rows if r["hint"] == hint]
        finals = collections.Counter(r["judge"]["final"] for r in sub)
        by[hint] = {
            "n": len(sub),
            "labels": dict(sorted((k, v) for k, v in finals.items() if k != "disagree")),
            "disagree": finals.get("disagree", 0), "noise": finals.get("noise", 0),
            "kept": sum(1 for r in sub if r["keep"]),
            "kept_as_hint": sum(1 for r in sub if r["keep"] and r["judge"]["final"] == hint),
        }
    by_kind = {}
    for (hint, kind), sub in sorted(_group(rows, lambda r: (r["hint"], r["kind"])).items()):
        finals = collections.Counter(r["judge"]["final"] for r in sub)
        by_kind[f"{hint}:{kind}"] = dict(sorted(finals.items()))
    return {
        "n": len(rows), "label_guide": GUIDE, "by_hint": by, "by_hint_kind": by_kind,
        "disagree": sum(1 for r in rows if r["judge"]["final"] == "disagree"),
        "agree_rate": round(sum(1 for r in rows if r["judge"]["final"] != "disagree") / max(len(rows), 1), 4),
        "kept_rows": sum(1 for r in rows if r["keep"]),
    }


def _group(rows, key):
    out = collections.defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


FINAL_LABELS = ("a", "b", "c", "d")


def parse_overrides(items):
    """`<id>=<label>` 목록 → dict. 꼴이 틀리거나 칸이 a · b · c · d 밖이면 ValueError."""
    out = {}
    for item in items or ():
        rid, sep, label = item.partition("=")
        if not sep or not rid or label not in FINAL_LABELS:
            raise ValueError(f"--override 꼴이 틀렸다 : {item!r} (<id>=a|b|c|d)")
        out[rid] = label
    return out


def finalize_rows(rows, overrides=None):
    """판정 줄 → (평가 줄, 학습 줄, 요약). id 겹침 · 없는 override id · 평가 × 학습 문서 겹침이면 ValueError."""
    overrides = dict(overrides or {})
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("판정 파일 사이에 id 가 겹친다")
    missing = sorted(set(overrides) - set(ids))
    if missing:
        raise ValueError(f"override id 가 판정 파일에 없다 : {missing[:5]}")
    out = {"eval": [], "train": []}
    for r in rows:
        final = r["judge"]["final"]
        judge = dict(r["judge"])
        if r["id"] in overrides:
            judge["override"] = {"from": final, "to": overrides[r["id"]]}
            final = overrides[r["id"]]
        elif not r.get("keep"):
            continue
        if final not in FINAL_LABELS:
            raise ValueError(f"남길 줄의 칸이 a · b · c · d 밖이다 : {r['id']} {final}")
        if r.get("pick_set") not in out:
            raise ValueError(f"pick_set 이 eval · train 이 아니다 : {r['id']}")
        row = dict(r, hint_pick=r["hint"], hint=final, want=final, judge=judge, label_guide=GUIDE, keep=True)
        out[row["pick_set"]].append(row)
    docs = {k: {r["group"] for r in v} for k, v in out.items()}
    overlap = docs["eval"] & docs["train"]
    if overlap:
        raise ValueError(f"평가 · 학습 문서가 겹친다 : {sorted(overlap)[:5]}")
    return out["eval"], out["train"], finalize_summary(out, len(rows), overrides)


def finalize_summary(out, n_in, overrides):
    sets = {}
    for name, rows in out.items():
        n = len(rows)
        sets[name] = {
            "n": n,
            "labels": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
            "by_kind": {k: dict(sorted(collections.Counter(r["want"] for r in sub).items()))
                        for k, sub in sorted(_group(rows, lambda r: r["kind"]).items())},
            "by_hint_pick": {k: dict(sorted(collections.Counter(r["want"] for r in sub).items()))
                             for k, sub in sorted(_group(rows, lambda r: r["hint_pick"]).items())},
            "flagged": sum(1 for r in rows if r.get("flag")),
            "flag_rate": round(sum(1 for r in rows if r.get("flag")) / max(n, 1), 4),
            "docs": len({r["group"] for r in rows}),
        }
    return {
        "n_in": n_in, "n_out": sum(len(v) for v in out.values()), "label_guide": GUIDE,
        "overrides": dict(sorted(overrides.items())), "sets": sets,
        "doc_overlap_eval_train": len({r["group"] for r in out["eval"]} & {r["group"] for r in out["train"]}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", dest="set_name", required=True, choices=("eval", "train"))
    p.add_argument("--n-b", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--flag-cap", type=float, default=0.2)
    j = sub.add_parser("judge")
    j.add_argument("--pairs", required=True)
    j.add_argument("--j1", required=True)
    j.add_argument("--j2", required=True)
    j.add_argument("--out", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--in", dest="inp", nargs="+", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    if args.cmd == "finalize":
        try:
            rows = [r for path in args.inp for r in read_jsonl(path)]
            ev, tr, summary = finalize_rows(rows, parse_overrides(args.override))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        write_jsonl(os.path.join(args.out_dir, "wiki-labeled-eval.jsonl"), ev)
        write_jsonl(os.path.join(args.out_dir, "wiki-labeled-train.jsonl"), tr)
        with open(os.path.join(args.out_dir, "wiki-labeled.summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    if args.cmd == "pick":
        try:
            rows, quota = pick_rows(read_jsonl(args.inp), args.n_b, args.set_name, args.seed, args.per_group, args.flag_cap)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        write_jsonl(args.out, rows)
        print(json.dumps(dict(pick_stats(rows), quota=quota), ensure_ascii=False))
        return 0
    try:
        rows, summary = judge_rows(read_jsonl(args.pairs), read_jsonl(args.j1), read_jsonl(args.j2))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    write_jsonl(args.out, rows)
    with open(os.path.splitext(args.out)[0] + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
