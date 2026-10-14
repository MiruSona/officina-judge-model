"""layer_g4c — v1.2 K2 판 c 층 train-g4-c · train-g4-c-h (근접 c 후보 뽑기 → 판정 → 조립).

설계 `../Docs/Design/2026-10-14-v1.2-K2-한국어c보강판설계.md` 4-1 · 4-2 · 4-4 · 5절 ④. 쌍 본문은 화면에 찍지 않는다.

    python -I layer_g4c.py pick  --g4 ~/.judgemodel/build/g4 [--old-root ~/.judgemodel] [--pool boolq=<jsonl> …] [--force]
    → <g4>/train-g4-c/pick.jsonl · pick-summary.json   (label_batches.py make --mode train 에 넣는다)
    python -I layer_g4c.py build --g4 … [--dirs <판정 폴더> …] [--draft]
    → <g4>/train-g4-c/cands.jsonl (train-g4-c) · cands-h.jsonl (train-g4-c-h) · summary.json

꼴 둘 (원천 안에서만 짓는다 — BoolQ 문단 × 위키 주장은 안 만든다) :
  c-pair  train-g3-ko 의 b 주장 × 같은 원천 다른 묶음 문단 (BM25 10~60위). `pair` 에 b 줄 id. 먼저 짓는다
  near-x  학습 쪽 문단 × 같은 원천 다른 묶음 주장 (BM25 10~60위, train-g3-ko b 주장은 안 쓴다). train-g3-ko 문단을 먼저 쓰고 모자라면 원천 학습 줄로
규칙 : 근거 묶음당 2 · 주장 한 번 · KLUE 「답 없는 질문」(form unanswerable) 은 뺀다 ·
  금지 = G4Forbid 학습 몫 + blind-g4 · blind-g3 forbid.json + g4 평가 후보(blind-g4 · dev-cmp-g4) 겹침 + scrub.
build 는 두 판이 다 c 로 정한 줄만 넣고(셋째가 c 로 정한 것 포함), 원천 · 꼴 목표까지 묶음당 2 로 고른다.
train-g4-c-h 는 c-pair 만 원천별 절반(고정 씨앗) · near-x 는 그대로다 (설계 6-3).
"""
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scrub  # noqa: E402
from assemble_g3 import Stop, load_decisions, resolve  # noqa: E402
from label_batches import LABELS, inside_git, read_jsonl, write_json, write_jsonl  # noqa: E402
from layer_g3 import Guard, merge_targets, recheck  # noqa: E402
from make_sets import G4Forbid, _old_eval_index, near_c, norm, sha_key  # noqa: E402,F401

SEED = 20261402
LAYER, LAYER_H = "train-g4-c", "train-g4-c-h"
SOURCES = ("boolq", "wiki", "klue")
PAIR, NEAR = "c-pair", "near-x"
TARGETS = {
    "pick": {PAIR: {"boolq": 220, "klue": 110, "wiki": 110}, NEAR: {"boolq": 390, "wiki": 390, "klue": 220}},
    "build": {PAIR: {"boolq": 200, "klue": 100, "wiki": 100}, NEAR: {"boolq": 350, "wiki": 350, "klue": 200}},
    "per_group": 2,
}
# 원천 학습 줄 기본 자리 = 라벨 붙은 g3 학습 몫 파일(실물 약 1,018 · 1,027 · 356 줄). 설계 4-1 의 「안 쓴 학습 쪽 묶음」
# (BoolQ 2,923 · 위키 3,808 · KLUE 1,850)은 이 파일들이 아니다 — 그 묶음을 쓰려면 --pool 원천=경로 로 준다 (리뷰 #7)
POOLS = {"boolq": ("g3", "boolq/boolq-labeled-train.jsonl"), "klue": ("g3", "klue/klue-labeled-train.jsonl"),
         "wiki": ("g3", "reserve/train-layer-cands.jsonl")}


def source_of_pool(src, row):
    """원천 학습 줄 중 쓸 줄 — 위키는 train-layer-cands 의 wiki-train 몫만 · KLUE 답 없는 질문은 뺀다."""
    if row.get("keep") is False:
        return False
    if src == "wiki" and row.get("g3_source") not in (None, "wiki-train"):
        return False
    return not (src == "klue" and row.get("form") == "unanswerable")


def _ev_key(row):
    return norm(row.get("evidence"))


def _tag(rows, src, form, set_name, pair_of=None):
    out = []
    for row in rows:
        extra = {"set": set_name, "pick": src, "form": form}
        if pair_of is not None:
            extra["pair"] = pair_of[norm(row["claim"])]
        out.append(dict(row, **extra))
    return out


def _cap_groups(rows, per_group, used, limit=None):
    """근거 묶음(group)당 per_group 까지만 · 최대 limit 줄 — used 는 꼴 둘이 같이 쓴다. 고른 줄만 센다(리뷰 #11)."""
    out = []
    for row in rows:
        if limit is not None and len(out) >= limit:
            break
        if used[row["group"]] >= per_group:
            continue
        used[row["group"]] += 1
        out.append(row)
    return out


def pick(layer_rows, pools, guard, targets=None, seed=SEED):
    """원천마다 c-pair → near-x 후보. → (후보 줄, 요약). 금지 줄은 짓기 전에 · 지은 뒤에 두 번 거른다."""
    targets = merge_targets(TARGETS, targets)
    per_group = targets["per_group"]
    out, summary = [], {"seed": seed, "targets": targets["pick"], "sources": {}}
    for src in SOURCES:
        dropped = collections.Counter()
        layer, pool = [], []
        for row in sorted((r for r in layer_rows if r.get("pick") == src), key=lambda r: r["id"]):
            why = guard.reasons(row)
            dropped.update("layer:" + w for w in why)
            if not why:
                layer.append(row)
        for row in sorted((r for r in pools.get(src) or [] if source_of_pool(src, r)), key=lambda r: r["id"]):
            why = guard.reasons(row)
            dropped.update("pool:" + w for w in why)
            if not why:
                pool.append(row)
        rows_all = layer + pool
        used_group = collections.Counter()
        per_ev = collections.Counter()
        b_rows = [r for r in layer if (r.get("want") or r.get("hint")) == "b"]
        pair_of = {norm(r["claim"]): r["id"] for r in b_rows}
        made = near_c(rows_all, targets["pick"][PAIR].get(src, 0), random.Random(f"{seed}:{src}:pair"), per_ev,
                      f"g4c-p-{src}-", claim_rows=b_rows, split="train", origin="g4-train", hash_id=True)
        pairs = _cap_groups(_clean(made, guard, dropped), per_group, used_group)
        taken = {norm(r["claim"]) for r in b_rows}   # near-x 주장에 train-g3-ko b 주장을 안 섞는다 (c 짝과 갈리게, 리뷰 #3)
        near = []
        want = targets["pick"][NEAR].get(src, 0)
        for k, ev_rows in enumerate((layer, pool)):
            claims = [r for r in rows_all if norm(r.get("claim")) not in taken]
            made = near_c(ev_rows, want - len(near), random.Random(f"{seed}:{src}:near{k}"), per_ev,
                          f"g4c-x{k}-{src}-", claim_rows=claims, split="train", origin="g4-train", hash_id=True)
            got = _cap_groups(_clean(made, guard, dropped), per_group, used_group)
            taken |= {norm(r["claim"]) for r in got}
            near += got
        out += _tag(pairs, src, PAIR, LAYER, pair_of) + _tag(near, src, NEAR, LAYER)
        summary["sources"][src] = {"layer_rows": len(layer), "pool_rows": len(pool), "b_claims": len(b_rows),
                                   PAIR: len(pairs), NEAR: len(near),
                                   "near_from_layer": sum(1 for r in near if r["id"].startswith("g4c-x0-")),
                                   "dropped": dict(sorted(dropped.items()))}
    summary["rows"] = len(out)
    return out, summary


def _clean(rows, guard, dropped):
    ok = []
    for row in rows:
        why = guard.reasons(row)
        dropped.update("made:" + w for w in why)
        if not why:
            ok.append(row)
    return ok


def _drop_boundary(decisions):
    """판정 줄 중 칸이 a · b · c · d 한 글자가 아닌 줄(train 판정의 경계 칸 cd 같은 것)을 「c 아님 → 버림」으로 바꾼다.
    c 층은 두 판 다 c 인 줄만 넣으므로(설계 4-2) 버림이 맞다. assemble_g3.resolve 는 평가 셋 규칙이라 그런 칸에서
    멈추므로 넘기기 전에 여기서 거른다. → (고친 판정 dict, 버린 줄 수)."""
    out, n = {}, 0
    for rid, dec in decisions.items():
        if dec.get("label") is not None and dec.get("label") not in LABELS:
            dec = dict(dec, status="drop-boundary")
            n += 1
        out[rid] = dec
    return out, n


def build(cands, decisions, targets=None, seed=SEED, draft=False):
    """판정 끝 후보 → (train-g4-c 줄, train-g4-c-h 줄, 요약). c 로 정한 줄만 · 원천 × 꼴 목표 · 묶음당 2."""
    targets = merge_targets(TARGETS, targets)
    decisions, boundary = _drop_boundary(decisions)
    state = collections.Counter()
    decided = collections.defaultdict(list)
    for cand in sorted(cands, key=lambda r: r["id"]):
        label, votes, status, how = resolve(dict(cand, want=""), decisions)
        state[how if how in ("pending", "dropped") else f"label_{label}"] += 1
        if label != "c":
            continue
        row = {k: v for k, v in cand.items() if k != "judge"}
        row.update(want="c", hint="c", label_guide="support-v6", status=status, votes=votes or {})
        decided[(cand["pick"], cand["form"])].append(row)
    if state["pending"] and not draft:
        raise Stop(f"판정 전 줄 {state['pending']} — 판정을 끝내거나 --draft")
    used = collections.Counter()
    chosen = {}
    for form in (PAIR, NEAR):
        for src in SOURCES:
            rows = decided[(src, form)]
            random.Random(f"{seed}:{src}:{form}:build").shuffle(rows)
            chosen[(src, form)] = _cap_groups(rows, targets["per_group"], used, targets["build"][form].get(src, 0))
    rows = sorted((r for got in chosen.values() for r in got), key=lambda r: r["id"])
    half = set()
    for src in SOURCES:
        ids = sorted(r["id"] for r in chosen[(src, PAIR)])
        half |= set(random.Random(f"{seed}:{src}:half").sample(ids, len(ids) // 2))
    rows_h = [dict(r, set=LAYER_H) for r in rows if r["form"] == NEAR or r["id"] in half]
    summary = {"seed": seed, "targets": targets["build"], "cand_state": dict(sorted(state.items())),
               "decided_c": {f"{s} · {f}": len(v) for (s, f), v in sorted(decided.items())},
               "chosen": {f"{s} · {f}": len(v) for (s, f), v in sorted(chosen.items())},
               "short": {f"{s} · {f}": targets["build"][f].get(s, 0) - len(v) for (s, f), v in sorted(chosen.items())
                         if len(v) < targets["build"][f].get(s, 0)},
               "rows": len(rows), "rows_h": len(rows_h), "boundary_dropped": boundary}
    return rows, rows_h, summary


# ---------------------------------------------------------------- 명령


def make_guard(g4, old_root, no_old_forbid=False):
    """→ (Guard, 빠진 파일). 평가 = g4 예약 후보 둘(+ 조립한 g4 셋이 있으면 그것도) · forbid.json = blind-g4 · blind-g3."""
    reserve = os.path.join(g4, "reserve")
    eval_paths = [os.path.join(reserve, f"{n}-cands.jsonl") for n in ("blind-g4", "dev-cmp-g4")]
    eval_paths += [p for p in (os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("blind-g4", "dev-cmp-g4"))
                   if os.path.exists(p)]
    forbid_json = [os.path.join(reserve, "blind-g4.forbid.json"), os.path.join(old_root, "sets", "g3", "blind-g3.forbid.json")]
    names = [os.path.join(old_root, "build", "g1", n) for n in ("names.txt", "game-names.txt")]
    missing = [p for p in eval_paths + forbid_json + names if not os.path.exists(p)]
    old_forbid = old_index = None
    if not no_old_forbid:
        g2_build, g3_build = os.path.join(old_root, "build", "g2"), os.path.join(old_root, "build", "g3")
        old_forbid = G4Forbid(old_root, os.path.join(old_root, "build", "g1"), os.path.join(old_root, "measure", "use1"),
                              os.path.join(g2_build, "forbid-ruler-ids.txt"), g2_build,
                              os.path.join(g3_build, "seed", "forbid-g3.txt"), g3_build)
        missing += old_forbid.missing
        old_index = _old_eval_index(old_root)
    if missing:
        return None, missing
    scrubber = scrub.Scrubber(scrub.read_names([names[0]]), scrub.read_names([names[1]]))
    return Guard(eval_paths, forbid_json, old_forbid, old_index, scrubber), []


def _refuse_or_move(paths, force):
    """출력이 있으면 거부(→ 있는 이름 목록), --force 면 옛 파일을 <이름>.<날짜>.<확장자> 로 돌려 둔다(→ 빈 목록)."""
    clash = [p for p in paths if os.path.exists(p)]
    if not clash or not force:
        return [os.path.basename(p) for p in clash]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in clash:
        base, ext = os.path.splitext(path)
        os.replace(path, f"{base}.{stamp}{ext}")
    return []


def _targets(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_pick(a):
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    out_dir = os.path.join(g4, LAYER)
    if inside_git(out_dir):
        print(json.dumps({"stop": "출력 폴더가 git 안이다 — 쌍 본문을 git 안에 안 쓴다", "out": out_dir}, ensure_ascii=False))
        return 2
    outs = [os.path.join(out_dir, n) for n in ("pick.jsonl", "pick-summary.json")]
    clash = _refuse_or_move(outs, a.force)
    if clash:
        print(json.dumps({"stop": "출력이 이미 있다 — 덮지 않는다 (--force 면 날짜 꼬리로 돌려 둔다)", "files": clash},
                         ensure_ascii=False))
        return 2
    pool_paths = {s: os.path.join(old_root, "build", gen, rel) for s, (gen, rel) in POOLS.items()}
    for spec in a.pool or []:
        src, _, path = spec.partition("=")
        pool_paths[src] = path
    layer_path = os.path.join(old_root, "sets", "g3", "train-g3-ko.jsonl")
    guard, missing = make_guard(g4, old_root, a.no_old_forbid)
    missing += [p for p in [layer_path] + list(pool_paths.values()) if not os.path.exists(p)]
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    pools = {s: read_jsonl(p) for s, p in pool_paths.items()}
    rows, summary = pick(read_jsonl(layer_path), pools, guard, _targets(a.targets), a.seed)
    summary["pools"] = {s: os.path.basename(p) for s, p in pool_paths.items()}
    summary["recheck"] = recheck(rows, guard)
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(outs[0], rows)
    write_json(outs[1], summary)
    print(json.dumps({k: summary[k] for k in ("rows", "recheck")} |
                     {"sources": {s: {k: v for k, v in e.items() if k != "dropped"} for s, e in summary["sources"].items()}},
                     ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def cmd_build(a):
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    out_dir = os.path.join(g4, LAYER)
    if inside_git(out_dir):
        print(json.dumps({"stop": "출력 폴더가 git 안이다 — 쌍 본문을 git 안에 안 쓴다", "out": out_dir}, ensure_ascii=False))
        return 2
    outs = [os.path.join(out_dir, n) for n in ("cands.jsonl", "cands-h.jsonl", "summary.json")]
    clash = _refuse_or_move(outs, a.force)
    if clash:
        print(json.dumps({"stop": "출력이 이미 있다 — 덮지 않는다 (--force 면 날짜 꼬리로 돌려 둔다)", "files": clash},
                         ensure_ascii=False))
        return 2
    guard, missing = make_guard(g4, old_root, a.no_old_forbid)
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    dirs = a.dirs if a.dirs is not None else [os.path.join(g4, "judge", LAYER)]
    try:
        rows, rows_h, summary = build(read_jsonl(os.path.join(out_dir, "pick.jsonl")), load_decisions(dirs),
                                      _targets(a.targets), a.seed, a.draft)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    summary["recheck"] = recheck(rows, guard)
    summary["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    write_jsonl(outs[0], rows)
    write_jsonl(outs[1], rows_h)
    write_json(outs[2], summary)
    print(json.dumps({k: summary[k] for k in ("cand_state", "chosen", "short", "rows", "rows_h", "recheck")},
                     ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, func in (("pick", cmd_pick), ("build", cmd_build)):
        s = sub.add_parser(name)
        s.add_argument("--g4", default="~/.judgemodel/build/g4")
        s.add_argument("--old-root", default="~/.judgemodel")
        s.add_argument("--targets", help="TARGETS 를 덮을 json (일부 칸만 적어도 됨)")
        s.add_argument("--seed", type=int, default=SEED)
        s.add_argument("--force", action="store_true", help="출력이 있으면 옛 파일을 날짜 꼬리로 돌려 두고 쓴다")
        s.add_argument("--no-old-forbid", action="store_true", help="G4Forbid · 옛 평가 문단 검사를 끈다 (시험용)")
        s.set_defaults(func=func)
    sub.choices["pick"].add_argument("--pool", nargs="*", help="원천 학습 줄 바꾸기 (원천=경로, 기본 g3 학습 몫 파일)")
    sub.choices["build"].add_argument("--dirs", nargs="*", default=None,
                                      help="판정 폴더 (decided.jsonl). 기본 <g4>/judge/train-g4-c")
    sub.choices["build"].add_argument("--draft", action="store_true", help="판정 전 줄을 세어서 빼고 조립")
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
