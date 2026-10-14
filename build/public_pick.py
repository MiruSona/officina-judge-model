"""public_pick — 공개셋(KoBEST BoolQ · KLUE 답 있는 질문) 후보에서 판정할 줄을 뽑고, v6 두 판정을 합친다 (K 판).

설계 `../Docs/Design/2026-10-13-v1.2-한국어b보강판설계.md` 2-3 · 3-2 · 3-5 · 4-1. 본문은 git 밖(`~/.judgemodel/build/g3/boolq/` · `klue/`)에만 쓴다.
판정 합치기 · 합본은 `wiki_pick` 의 함수를 그대로 쓴다(같은 판 파일 꼴 · 같은 keep 규칙).

    python -I public_pick.py pick  --src <dev-rw.jsonl> <test-rw.jsonl> --check <dev-rw.check.jsonl> <test-rw.check.jsonl>
                                   --forbid <train-rw.jsonl> --set eval --n-b 300 --n-a 150 --out <pick-eval.jsonl>
    python -I public_pick.py stats --in <pick-eval.jsonl>
    python -I public_pick.py blind --in <pick-eval.jsonl> --seed 101 --out <judge-in-j1.jsonl>
    python -I public_pick.py judge --pairs <pick-eval.jsonl> --j1 <j1.jsonl> --j2 <j2.jsonl> --out <judged-eval.jsonl>
    python -I public_pick.py finalize --in <judged-eval.jsonl> <judged-train.jsonl> --out-dir <boolq> [--override <id>=<label> …]

pick 규칙 : 주장 = `claim_rewrite` 가 있으면 그것(옛 물음은 `claim_question`), 없으면 `claim`. 평서문 검사(`--check`)에서
떨어진 줄 · 물음꼴이 남은 줄은 뺀다. `--set eval` 이면 `near_train` 줄 · `--forbid` 파일의 문단(group)과 같은 문단도 뺀다.
`kind` = `<원천>-<split>`(원천은 줄 꼴로 안다 — `domain: klue-answered` 면 klue, 아니면 boolq).
KLUE 는 평서문 검사가 짝(`pair`) 단위라 떨어진 짝은 a · b 둘 다 빼고, 다시 쓴 것이 없는 줄도 뺀다(`no_rewrite`).
`--forbid` 파일은 맞은편 split 의 문단만 쓴다 — 평가 몫이면 `split: train` 줄, 학습 몫이면 그 밖의 줄. hint b 는 kind 비율대로(큰 나머지) 씨앗 고정으로 섞은 차례에서 문단마다 `--per-group`(2)개까지.
hint a 는 뽑은 b 와 같은 문단의 a 를 먼저(`a_mate: same`), 모자라면 kind 비율대로 다른 문단에서(`a_mate: other`), 문단마다 1개.
`id` 는 원래 id 그대로, `pick_set` 을 단다.
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rewrite_check  # noqa: E402
from label_batches import inside_git  # noqa: E402
import wiki_pick as wp  # noqa: E402


def failed_ids(check_rows):
    """평서문 검사 결과 줄 중 `pass: false` 인 id (KLUE 는 짝 `pair`)."""
    return {r.get("id") or r["pair"] for r in check_rows if not r.get("pass", True)}


def source_of(row):
    return "klue" if row.get("domain") == "klue-answered" else "boolq"


def forbid_groups_for(rows, set_name):
    """`--forbid` 줄에서 맞은편 split 의 문단 — 평가 몫은 학습(train) 문단, 학습 몫은 그 밖 문단."""
    return {r["group"] for r in rows if (r.get("split", "train") == "train") == (set_name == "eval")}


def prepare(rows, set_name, drop_ids=(), forbid_groups=()):
    """주장을 고르고 뺄 줄을 뺀다. 돌려주는 값 : (남은 줄, 뺀 까닭 셈)."""
    drop_ids, forbid_groups = set(drop_ids), set(forbid_groups)
    out, dropped = [], collections.Counter()
    for r in rows:
        rw = r.get("claim_rewrite")
        claim = rw or r["claim"]
        src = source_of(r)
        if r["id"] in drop_ids or r.get("pair") in drop_ids:
            dropped["check_fail"] += 1
        elif src == "klue" and not rw:
            dropped["no_rewrite"] += 1
        elif rewrite_check.is_question(claim):
            dropped["question"] += 1
        elif set_name == "eval" and r.get("near_train"):
            dropped["near_train"] += 1
        elif r["group"] in forbid_groups:
            dropped["forbid_group"] += 1
        else:
            row = dict(r, claim=claim, kind=f"{src}-{r['split']}")
            row.pop("claim_rewrite", None)
            if rw:
                row["claim_question"] = r["claim"]
            out.append(row)
    return out, dict(sorted(dropped.items()))


def _shuffled(rows, seed):
    pool = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    return pool


def pick_b(rows, n, seed, per_group=2):
    """hint b 줄에서 n 개 — kind 비율 몫, 문단마다 per_group 개까지. 돌려주는 값 : (뽑은 줄, kind 별 몫)."""
    by_kind = wp._group(rows, lambda r: r["kind"])
    quota = wp.quotas({k: len(v) for k, v in by_kind.items()}, n)
    per_doc = collections.Counter()
    picked = []
    for kind in sorted(by_kind):
        got = 0
        for r in _shuffled(by_kind[kind], seed):
            if got >= quota[kind]:
                break
            if per_doc[r["group"]] >= per_group:
                continue
            per_doc[r["group"]] += 1
            got += 1
            picked.append(r)
    return picked, quota


def pick_a(rows, n, seed, b_groups):
    """hint a 줄에서 n 개 — 뽑은 b 와 같은 문단 a 먼저, 나머지는 kind 비율 몫. 문단마다 1개."""
    used, picked = set(), []
    for r in _shuffled([r for r in rows if r["group"] in b_groups], seed):
        if len(picked) >= n:
            break
        if r["group"] not in used:
            used.add(r["group"])
            picked.append(dict(r, a_mate="same"))
    rest = [r for r in rows if r["group"] not in used and r["group"] not in b_groups]
    by_kind = wp._group(rest, lambda r: r["kind"])
    quota = wp.quotas({k: len(v) for k, v in by_kind.items()}, n - len(picked))
    for kind in sorted(by_kind):
        got = 0
        for r in _shuffled(by_kind[kind], seed + 1):
            if got >= quota[kind]:
                break
            if r["group"] in used:
                continue
            used.add(r["group"])
            got += 1
            picked.append(dict(r, a_mate="other"))
    return picked, quota


def pick_rows(rows, set_name, n_b, n_a, seed, per_group=2):
    """뽑은 b + a. 몫을 못 채우거나 id 가 겹치면 ValueError."""
    b_pick, b_quota = pick_b([r for r in rows if r["hint"] == "b"], n_b, seed, per_group)
    if len(b_pick) < n_b:
        raise ValueError(f"b 몫 {n_b} 중 {len(b_pick)} 만 뽑혔다")
    a_pick, a_quota = pick_a([r for r in rows if r["hint"] == "a"], n_a, seed, {r["group"] for r in b_pick})
    if len(a_pick) < n_a:
        raise ValueError(f"a 몫 {n_a} 중 {len(a_pick)} 만 뽑혔다")
    out = [dict(r, pick_set=set_name) for r in b_pick + a_pick]
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("id 가 겹친다")
    return out, {"b": b_quota, "a_other": a_quota}


def pick_stats(rows):
    per_group = collections.Counter(r["group"] for r in rows)
    return {
        "n": len(rows),
        "by_hint_kind": {f"{h}:{k}": n for (h, k), n in sorted(collections.Counter((r["hint"], r["kind"]) for r in rows).items())},
        "a_mate": dict(sorted(collections.Counter(r["a_mate"] for r in rows if r["hint"] == "a").items())),
        "rewritten": sum(1 for r in rows if "claim_question" in r),
        "shortened": sum(1 for r in rows if r.get("shortened")),
        "groups": len(per_group),
        "b_max_per_group": max(collections.Counter(r["group"] for r in rows if r["hint"] == "b").values(), default=0),
        "a_max_per_group": max(collections.Counter(r["group"] for r in rows if r["hint"] == "a").values(), default=0),
    }


def blind_rows(rows, seed):
    """판정 입력 — hint · split · near_train 같은 칸을 숨기고 씨앗 차례로 섞는다."""
    return [{"id": r["id"], "evidence": r["evidence"], "claim": r["claim"]} for r in _shuffled(rows, seed)]


def _dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--check", nargs="*", default=[])
    p.add_argument("--forbid", nargs="*", default=[], help="이 파일들의 맞은편 split 문단(group)은 뺀다")
    p.add_argument("--set", dest="set_name", required=True, choices=("eval", "train"))
    p.add_argument("--n-b", type=int, required=True)
    p.add_argument("--n-a", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--out", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--in", dest="inp", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--in", dest="inp", required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--pairs", required=True)
    j.add_argument("--j1", required=True)
    j.add_argument("--j2", required=True)
    j.add_argument("--out", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--in", dest="inp", nargs="+", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--override", action="append", default=[])
    f.add_argument("--name", default="boolq", help="합본 파일 이름 앞머리 (boolq · klue)")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "pick":
            rows = [r for path in args.src for r in wp.read_jsonl(path)]
            drop = {i for path in args.check for i in failed_ids(wp.read_jsonl(path))}
            forbid = forbid_groups_for([r for path in args.forbid for r in wp.read_jsonl(path)], args.set_name)
            rows, dropped = prepare(rows, args.set_name, drop, forbid)
            out, quota = pick_rows(rows, args.set_name, args.n_b, args.n_a, args.seed, args.per_group)
            wp.write_jsonl(args.out, out)
            summary = dict(pick_stats(out), dropped=dropped, pool=len(rows), quota=quota)
        elif args.cmd == "stats":
            summary = pick_stats(wp.read_jsonl(args.inp))
        elif args.cmd == "blind":
            out = blind_rows(wp.read_jsonl(args.inp), args.seed)
            wp.write_jsonl(args.out, out)
            summary = {"n": len(out), "seed": args.seed}
        elif args.cmd == "judge":
            out, summary = wp.judge_rows(wp.read_jsonl(args.pairs), wp.read_jsonl(args.j1), wp.read_jsonl(args.j2))
            wp.write_jsonl(args.out, out)
            _dump(os.path.splitext(args.out)[0] + ".summary.json", summary)
        else:
            if inside_git(args.out_dir):
                print("멈춤 : --out-dir 가 git 안이다 — 쌍 본문은 git 밖(~/.judgemodel/build/g3/…)에 둔다", file=sys.stderr)
                return 2
            rows = [r for path in args.inp for r in wp.read_jsonl(path)]
            ev, tr, summary = wp.finalize_rows(rows, wp.parse_overrides(args.override))
            os.makedirs(args.out_dir, exist_ok=True)
            wp.write_jsonl(os.path.join(args.out_dir, f"{args.name}-labeled-eval.jsonl"), ev)
            wp.write_jsonl(os.path.join(args.out_dir, f"{args.name}-labeled-train.jsonl"), tr)
            _dump(os.path.join(args.out_dir, f"{args.name}-labeled.summary.json"), summary)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
