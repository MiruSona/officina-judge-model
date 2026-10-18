"""dev_g9 — dev-cmp-g9 재료 만들기 (K7 설계 1-2 · 4-1 2단계). 학습 · 라벨 · 셋 얼리기는 안 한다.

    python -I dev_g9.py trim  [--root ~/.judgemodel] [--work ~/.judgemodel/work-g9]
    python -I dev_g9.py pick  [--root …] [--work …] [--pool 원천=파일 …] [--seed 20261801]
    python -I dev_g9.py check [--root …] [--work …]
    python -I dev_g9.py blind [--root …] [--work …] [--seed 20261802] [--size 100]

보충 판(앞 후보를 그대로 두고 더 뽑기) — 이름 · 피할 후보 · 묶음 시작 번호를 따로 준다 :
    python -I dev_g9.py pick  --name cands-g9-ko-extra --avoid cands-g9-ko --pools-only --pool 원천=파일 …
                              [--quota 원천=전체:b …] [--allow-form near-x]
    python -I dev_g9.py check --name cands-g9-ko-extra --avoid cands-g9-ko
    python -I dev_g9.py blind --name cands-g9-ko-extra --no-old --start 24 --map g9-blind-map-extra.jsonl

- trim  : dev-cmp-g4 에서 학습 층과 같은 묶음을 쓰는 쌍(빼기) · label_guide v5 쌍(v6 재라벨 대상) id 목록.
- pick  : 공개 한국어 원천 넷에서 학습 층 · dev-cmp-g4 · blind-g4 · 옛 셋 어디에도 없는 근거 묶음만 후보로.
          본문(id · src · form · evidence · claim)과 힌트(hint · 묶음 · 출처 id)를 다른 파일에 쓴다.
- check : 후보 ↔ 학습 층 · dev-cmp-g4 · blind-g4 열쇠 겹침을 다시 센다(pick 과 따로 읽음). 0 이 아니면 exit 1.
- blind : 후보 + v6 재라벨 대상을 섞어 라벨러용 묶음(bid · evidence · claim)과 되돌리는 표를 쓴다.

blind-g4 는 `blind-g4.forbid.json`(열쇠 목록)만 읽고 본문 파일은 열지 않는다. 출력은 모두 git 밖 work 폴더에 쓴다.
생성기 꼴(near-x · removed 등)은 후보에 넣지 않는다(설계 1-2 · 1-3). 보충 판의 `--allow-form near-x` 만 예외다 —
그 줄은 form 이 near-x 라 목표 자(generator_forms 제외)에서는 빠지고 지킴(F1 c)에는 든다(설계 1-3).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                   # python -I 는 스크립트 폴더를 import 길에 안 넣는다
    sys.path.insert(0, HERE)

import make_sets as ms  # noqa: E402

DEV = "sets/g4/dev-cmp-g4.jsonl"
DEV_SHA = "fa2f81cb4226475aebb825d61af8f36efc0be695ef9d32e78427d4ea0b1ae59b"
BLIND_FORBID = "sets/g4/blind-g4.forbid.json"
# K 처방(r2 · arch · g3-ko) + K4~K6 층. 빼기 목록은 이 여섯과 겹친 쌍이다(설계 1-2).
RECIPE_LAYERS = {
    "train-r2": "sets/2026-10-10/build/train-r2.jsonl",
    "train-g1-arch": "sets/g1/train-g1-arch.jsonl",
    "train-g3-ko": "sets/g3/train-g3-ko.jsonl",
    "train-g4-bal": "sets/g4/train-g4-bal.jsonl",
    "train-g4-bal2": "sets/g4/train-g4-bal2.jsonl",
    "train-g4-klb": "sets/g4/train-g4-klb.jsonl",
}
# 새 후보는 지난 학습 층 어디에도 안 닿게 한다 — 처방 밖 층까지 막는다.
OTHER_LAYERS = {
    "train-g1-II": "sets/g1/train-g1-II.jsonl",
    "train-g1-syn": "sets/g1/train-g1-syn.jsonl",
    "train-g2-b": "sets/g2/train-g2-b.jsonl",
    "train-g4-c": "sets/g4/train-g4-c.jsonl",
    "train-g4-c-h": "sets/g4/train-g4-c-h.jsonl",
    "train-g4-d": "sets/g4/train-g4-d.jsonl",
}
RESERVE_G4 = "build/g4/reserve/reserve-g4.json"
# 원천 → 후보 풀. 주장(claim)이 이미 평서문으로 있는 파일만 쓴다(새로 다시 쓰지 않는다).
POOLS = {
    "kowiki": ["build/g3/kowiki/wiki-all-f.jsonl"],
    "kobest-boolq": ["build/g3/boolq/train-rw.jsonl", "build/g3/boolq/dev-rw.jsonl", "build/g3/boolq/test-rw.jsonl"],
    "klue-answered": ["build/g3/klue/answered-rw.jsonl"],
    "klue-mrc": ["build/g1/train-cands-klue.jsonl", "build/g4/klb/pick-mrc-rw.jsonl"],
}
GENERATOR_FORMS = {"near-x", "removed", "boolq-removed", "klue-removed", "near"}
B_SHARE = {"kowiki": 100, "kobest-boolq": 100, "klue-answered": 60, "klue-mrc": 40}   # 설계 1-2 b 몫 [어림]
NEW_TOTAL = 1000
FACTOR = 1.6
GROUP_CAP = 2                              # 한 근거 묶음에서 쌍 2 까지(칸이 다를 때만)
PICK_SEED = 20261801
BLIND_SEED = 20261802
BID_SALT = "g9-blind-v1:"
CAND_NAME = "cands-g9-ko"
CAND_FILE = CAND_NAME + ".jsonl"
HINT_FILE = CAND_NAME + ".hints.jsonl"


def cand_paths(work, name=CAND_NAME):
    """이름 → (본문 파일, 힌트 파일, 요약 파일)."""
    base = os.path.join(work, name)
    return base + ".jsonl", base + ".hints.jsonl", base + ".summary.json"


def merged_rows(work, name):
    """본문 + 힌트를 id 로 합친 줄 — 피할 후보(앞 판)를 Marks 에 넣을 때 쓴다."""
    cpath, hpath, _ = cand_paths(work, name)
    meta = {h["id"]: h for h in ms.read_jsonl(hpath)}
    out = []
    for c in ms.read_jsonl(cpath):
        row = dict(c, **{k: v for k, v in meta.get(c["id"], {}).items() if k != "id"})
        row.setdefault("lang", "ko")
        out.append(row)
    return out


def _p(root, rel):
    return os.path.join(root, *rel.split("/"))


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_ids(path, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(i + "\n" for i in ids)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Marks:
    """파일 여럿의 (묶음 · 출처 id · 글 열쇠 · sha 열쇠) 와 한국어 문단 색인."""

    def __init__(self):
        self.groups, self.srcs, self.keys, self.shas = set(), set(), set(), set()
        self.para = ms.ParaIndex()

    def add_rows(self, rows, tag):
        for row in rows:
            self.groups |= ms._groups_of(row)
            self.srcs.update(row.get("src_ids") or [])
            self.keys |= ms._both_keys(row)
            if row.get("evidence") or row.get("claim"):
                self.shas.add(ms.sha_key(row))
            if row.get("lang") == "ko":
                self.para.add(row, tag)

    def add_forbid(self, doc):
        """blind-g4.forbid.json — 열쇠만(본문 없음)."""
        self.groups |= set(doc.get("groups", []))
        self.srcs |= set(doc.get("src_ids", []))
        self.shas |= set(doc.get("pair_keys", []))

    def reasons(self, row, para=True):
        out = []
        if ms._groups_of(row) & self.groups:
            out.append("group")
        if set(row.get("src_ids") or []) & self.srcs:
            out.append("src")
        if ms._both_keys(row) & self.keys:
            out.append("text")
        flip = ms.sha_key({"evidence": row.get("claim"), "claim": row.get("evidence")})
        if {ms.sha_key(row), flip} & self.shas:
            out.append("sha")
        if para and row.get("lang") == "ko" and self.para.hits(row):
            out.append("para")
        return out


def load_marks(root, with_reserve=True, avoid=None):
    """학습 층 열두 · dev-cmp-g4 · blind-g4 열쇠 · reserve-g4 예약 묶음 → {이름: Marks}, 빠진 파일 목록.
    avoid = {이름: 줄 목록} — 앞 판 후보처럼 더 피할 것(그 이름의 Marks 가 하나 더 생긴다)."""
    marks, missing = {"train": Marks(), "dev-cmp-g4": Marks(), "blind-g4": Marks()}, []
    for name, rows in (avoid or {}).items():
        marks[name] = Marks()
        marks[name].add_rows(rows, name)
    for name, rel in {**RECIPE_LAYERS, **OTHER_LAYERS}.items():
        path = _p(root, rel)
        if os.path.exists(path):
            marks["train"].add_rows(ms.read_jsonl(path), "train")
        else:
            missing.append(name)
    path = _p(root, DEV)
    if os.path.exists(path):
        marks["dev-cmp-g4"].add_rows(ms.read_jsonl(path), "dev")
    else:
        missing.append("dev-cmp-g4")
    path = _p(root, BLIND_FORBID)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            marks["blind-g4"].add_forbid(json.load(f))
    else:
        missing.append("blind-g4.forbid.json")
    path = _p(root, RESERVE_G4)
    if with_reserve and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            marks["blind-g4"].groups |= set(json.load(f).get("reserved", []))
    return marks, missing


# ---------------------------------------------------------------- trim


def trim(dev_rows, layer_rows):
    """→ (빼기 id, v5 재라벨 id, 빼기 까닭 셈). 빼기 = 묶음 · 출처 id 가 처방 층과 겹친 쌍. v5 는 빼기를 뺀 나머지."""
    groups, srcs, why = set(), set(), collections.Counter()
    owner = {}
    for name, rows in layer_rows.items():
        for row in rows:
            for g in ms._groups_of(row):
                groups.add(g)
                owner.setdefault(g, name)
            srcs.update(row.get("src_ids") or [])
    drop = []
    for row in dev_rows:
        hit = sorted(ms._groups_of(row) & groups)
        if hit or set(row.get("src_ids") or []) & srcs:
            drop.append(row["id"])
            why[owner.get(hit[0], "src") if hit else "src"] += 1
    gone = set(drop)
    v5 = [r["id"] for r in dev_rows if r.get("label_guide") == "support-v5" and r["id"] not in gone]
    return drop, v5, why


def trim_main(args):
    root = os.path.expanduser(args.root)
    dev_path = _p(root, DEV)
    sha = _sha_file(dev_path)
    if sha != DEV_SHA:
        print(json.dumps({"stop": "dev-cmp-g4 sha 가 얼린 값과 다르다", "sha": sha}, ensure_ascii=False))
        return 2
    dev_rows = ms.read_jsonl(dev_path)
    layers = {n: ms.read_jsonl(_p(root, rel)) for n, rel in RECIPE_LAYERS.items()}
    drop, v5, why = trim(dev_rows, layers)
    work = os.path.expanduser(args.work)
    _write_ids(os.path.join(work, "dev-g4-drop.ids.txt"), drop)
    _write_ids(os.path.join(work, "dev-g4-v5.ids.txt"), v5)
    v5_set = set(v5)
    lang = collections.Counter(r.get("lang") for r in dev_rows if r["id"] in v5_set)
    summary = {"dev_sha256": sha, "dev_pairs": len(dev_rows), "drop": len(drop), "drop_by_layer": dict(why),
               "v5_relabel": len(v5), "v5_by_lang": dict(lang), "kept": len(dev_rows) - len(drop)}
    with open(os.path.join(work, "trim.summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- pick


def claim_of(row):
    """평서문 주장 — claim_rewrite 가 있으면 그것, 없으면 claim. 물음표로 끝나거나 비면 None(쓰지 않는다)."""
    text = (row.get("claim_rewrite") or row.get("claim") or "").strip()
    if not text or text.endswith("?"):
        return None
    return text


def quotas(total=NEW_TOTAL, factor=FACTOR, b_share=None):
    """원천별 후보 수와 b 후보 수 — 원천 몫은 b 몫 비, 합은 total × factor (큰 나머지 나눔)."""
    b_share = b_share or B_SHARE
    n = ms._largest_remainder(round(total * factor), b_share)
    return {s: {"total": n[s], "b": round(b_share[s] * factor)} for s in sorted(b_share)}


def _clean(row, source, forbid, g4marks):
    """한 줄의 금지 까닭 — 값싼 열쇠(묶음 · 출처 · 글) 먼저, 걸리면 비싼 문단 검사는 건너뛴다."""
    why = [f"{w}_{n}" for n, m in g4marks.items() for w in m.reasons(row, para=False)]
    if why:
        return why
    why = forbid.reasons(row, "eval") if forbid else []
    if why:
        return why
    return [f"para_{n}" for n, m in g4marks.items() if row.get("lang", "ko") == "ko" and m.para.hits(row)]


def pick_source(source, rows, quota, forbid, g4marks, rng, allow_forms=()):
    """한 원천 후보 고르기 → (고른 줄, 버린 까닭 셈). 묶음을 섞어 차례로 보며(금지 검사는 볼 때만),
    1 바퀴는 b 힌트 묶음을 b 몫만큼(b + 같은 묶음 다른 칸 하나), 2 바퀴는 나머지 묶음의 b 아닌 칸."""
    dropped = collections.Counter()
    by_group = collections.defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("form") in GENERATOR_FORMS and row.get("form") not in allow_forms:
            dropped["generator_form"] += 1
            continue
        claim = claim_of(row)
        if claim is None or not (row.get("evidence") or "").strip():
            dropped["no_claim"] += 1
            continue
        row = dict(row, claim=claim, domain=row.get("domain") or source)
        row.setdefault("lang", "ko")
        key = ms.sha_key(row)
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        by_group[row["group"]].append(row)
    order = sorted(by_group)
    rng.shuffle(order)
    out, b_left, total_left = [], quota["b"], quota["total"]
    used_para = ms.ParaIndex()
    checked = {}
    for want_b in (True, False):
        for g in order:
            if total_left <= 0:
                break
            rows_g = by_group[g]
            if rows_g is None:
                continue
            has_b = any(r.get("hint") == "b" for r in rows_g)
            if want_b and (not has_b or b_left <= 0):
                continue
            if g not in checked:
                good = []
                for r in rows_g:
                    why = _clean(r, source, forbid, g4marks)
                    if why:
                        dropped.update(set(why))
                    else:
                        good.append(r)
                checked[g] = good
            pool = sorted(checked[g], key=lambda r: (r.get("hint") != "b", r.get("id") or ms.sha_key(r)))
            if not want_b:
                pool = [r for r in pool if r.get("hint") != "b"]
            take, hints = [], set()
            for r in pool:
                if len(take) >= min(GROUP_CAP, total_left) or r.get("hint") in hints:
                    continue
                take.append(r)
                hints.add(r.get("hint"))
            if not take:
                continue
            if used_para.hits(take[0]):
                dropped["para_within"] += 1
                by_group[g] = None
                continue                     # 다른 묶음의 같은 문단 — 한 문단은 한 번만
            for r in take:
                used_para.add(r, "picked")
            out += take
            total_left -= len(take)
            b_left -= sum(1 for r in take if r.get("hint") == "b")
            by_group[g] = None
    dropped["groups_checked"] = len(checked)
    return out, dropped


def new_id(row):
    """후보 id — 원천 · 칸 흔적이 없는 쌍 해시."""
    return "g9n-" + ms.sha_key(row)[:12]


def pick_main(args):
    root = os.path.expanduser(args.root)
    work = os.path.expanduser(args.work)
    if ms._inside_git(work):
        print(json.dumps({"stop": "work 폴더가 git 안이다"}, ensure_ascii=False))
        return 2
    pools = {} if args.pools_only else {s: [_p(root, rel) for rel in rels] for s, rels in POOLS.items()}
    for item in args.pool or []:
        name, path = item.split("=", 1)
        pools.setdefault(name, []).append(os.path.expanduser(path))
    missing = [p for ps in pools.values() for p in ps if not os.path.exists(p)]
    forbid, _ = ms._g4_forbid(argparse.Namespace(old_root=root, g1_build=None, use1=None, ruler_ids=None,
                                                 g2_build=None, g3_build=None, seed_forbid=None))
    print("forbid loaded", file=sys.stderr, flush=True)
    avoid = {"cands-g9": merged_rows(work, args.avoid)} if args.avoid else None
    g4marks, miss2 = load_marks(root, avoid=avoid)
    print("marks loaded", file=sys.stderr, flush=True)
    missing += list(forbid.missing) + miss2
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    rng = random.Random(args.seed)
    plan = quotas()
    for item in args.quota or []:
        name, nums = item.split("=", 1)
        total, b = nums.split(":")
        plan[name] = {"total": int(total), "b": int(b)}
    allow = set(args.allow_form or [])
    if allow - {"near-x"}:
        print(json.dumps({"stop": "생성기 꼴은 near-x 만 허용한다", "asked": sorted(allow)}, ensure_ascii=False))
        return 2
    cands, hints, report = [], [], {}
    for source in sorted(pools):
        rows = [r for p in pools[source] for r in ms.read_jsonl(p)]
        got, dropped = pick_source(source, rows, plan[source], forbid, g4marks, rng, allow)
        print(f"{source}: {len(got)}", file=sys.stderr, flush=True)
        report[source] = {"pool_rows": len(rows), "quota": plan[source], "picked": len(got),
                          "groups": len({r["group"] for r in got}),
                          "hints": dict(collections.Counter(r.get("hint") for r in got)),
                          "forms": dict(collections.Counter(r.get("form") for r in got)),
                          "dropped": dict(sorted(dropped.items()))}
        for r in got:
            cid = new_id(r)
            cands.append({"id": cid, "src": source, "form": r.get("form"), "evidence": r["evidence"],
                          "claim": r["claim"]})
            hints.append({"id": cid, "hint": r.get("hint"), "group": r["group"],
                          "groups": sorted(ms._groups_of(r)), "src_ids": r.get("src_ids") or [],
                          "lang": r.get("lang", "ko"), "lic": r.get("lic", ""), "orig_id": r.get("id"),
                          "split": r.get("split"), "origin": r.get("origin")})
    if len({c["id"] for c in cands}) != len(cands):
        print(json.dumps({"stop": "후보 id 가 겹친다"}, ensure_ascii=False))
        return 2
    cpath, hpath, spath = cand_paths(work, args.name)
    if args.name != CAND_NAME and any(os.path.exists(p) for p in (cpath, hpath)):
        print(json.dumps({"stop": "보충 후보 파일이 이미 있다 — 덮지 않는다", "file": cpath}, ensure_ascii=False))
        return 2
    _write_jsonl(cpath, cands)
    _write_jsonl(hpath, hints)
    summary = {"seed": args.seed, "name": args.name, "avoid": args.avoid, "total": len(cands), "sources": report,
               "generator_forms_skipped": sorted(GENERATOR_FORMS - allow), "generator_forms_allowed": sorted(allow)}
    with open(spath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- check


def check(cands, hints, marks):
    """후보 ↔ 각 Marks 겹침 셈 + 후보끼리 글 · 묶음 겹침(묶음은 GROUP_CAP 넘을 때만)."""
    meta = {h["id"]: h for h in hints}
    counts = collections.Counter()
    keys, per_group = set(), collections.Counter()
    for c in cands:
        row = dict(c, **{k: v for k, v in meta.get(c["id"], {}).items() if k != "id"})
        row.setdefault("lang", "ko")
        if c["id"] not in meta:
            counts["no_hint_row"] += 1
        for name, m in marks.items():
            for why in m.reasons(row):
                counts[f"{why}_{name}"] += 1
        both = ms._both_keys(row)
        if both & keys:
            counts["text_within"] += 1
        keys |= both
        per_group[row.get("group")] += 1
    counts["group_over_cap_within"] = sum(1 for n in per_group.values() if n > GROUP_CAP)
    names = [f"{w}_{n}" for n in marks for w in ("group", "src", "text", "sha", "para")]
    names += ["text_within", "group_over_cap_within", "no_hint_row"]
    return {k: counts.get(k, 0) for k in names}


def check_main(args):
    root = os.path.expanduser(args.root)
    work = os.path.expanduser(args.work)
    avoid = {"cands-g9": merged_rows(work, args.avoid)} if args.avoid else None
    marks, missing = load_marks(root, with_reserve=True, avoid=avoid)
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    cpath, hpath, _ = cand_paths(work, args.name)
    cands = ms.read_jsonl(cpath)
    hints = ms.read_jsonl(hpath)
    out = check(cands, hints, marks)
    doc = {"name": args.name, "avoid": args.avoid, "cands": len(cands), "check": out, "total": sum(out.values())}
    out_name = "check-g9.json" if args.name == CAND_NAME else f"check-{args.name}.json"
    with open(os.path.join(work, out_name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 1 if doc["total"] else 0


# ---------------------------------------------------------------- blind


def bid_of(row_id):
    return hashlib.sha256((BID_SALT + row_id).encode("utf-8")).hexdigest()[:12]


def blind(new_rows, old_rows, seed=BLIND_SEED, size=100, start=1):
    """→ (묶음 목록, 되돌리는 표). 묶음 줄은 bid · evidence · claim 만. 옛 dev 줄의 옛 라벨은 표에만."""
    items = []
    for r in new_rows:
        items.append(({"bid": bid_of(r["id"]), "evidence": r["evidence"], "claim": r["claim"]},
                      {"bid": bid_of(r["id"]), "id": r["id"], "from": "new-ko", "src": r["src"]}))
    for r in old_rows:
        items.append(({"bid": bid_of(r["id"]), "evidence": r["evidence"], "claim": r["claim"]},
                      {"bid": bid_of(r["id"]), "id": r["id"], "from": "old-dev", "src": r.get("domain"),
                       "old_want": r.get("want"), "old_guide": r.get("label_guide")}))
    bids = [b for b, _ in items]
    if len({b["bid"] for b in bids}) != len(bids):
        raise ValueError("bid 가 겹친다")
    items.sort(key=lambda it: it[0]["bid"])
    random.Random(seed).shuffle(items)
    batches = [[b for b, _ in items[i:i + size]] for i in range(0, len(items), size)]
    table = []
    for n, i in enumerate(range(0, len(items), size), start=start):
        for _, m in items[i:i + size]:
            table.append(dict(m, batch=f"g9-b{n:03d}"))
    return batches, table


def blind_main(args):
    root = os.path.expanduser(args.root)
    work = os.path.expanduser(args.work)
    if ms._inside_git(work):
        print(json.dumps({"stop": "work 폴더가 git 안이다"}, ensure_ascii=False))
        return 2
    new_rows = ms.read_jsonl(cand_paths(work, args.name)[0])
    old_rows = []
    if not args.no_old:
        with open(os.path.join(work, "dev-g4-v5.ids.txt"), encoding="utf-8") as f:
            v5 = {line.strip() for line in f if line.strip()}
        old_rows = [r for r in ms.read_jsonl(_p(root, DEV)) if r["id"] in v5]
        if len(old_rows) != len(v5):
            print(json.dumps({"stop": "v5 id 중 dev-cmp-g4 에 없는 것이 있다"}, ensure_ascii=False))
            return 2
    batches, table = blind(new_rows, old_rows, args.seed, args.size, args.start)
    out = os.path.join(work, "batches")
    names = [f"g9-b{n:03d}.jsonl" for n in range(args.start, args.start + len(batches))]
    map_path = os.path.join(work, args.map)
    clash = [n for n in names if os.path.exists(os.path.join(out, n))]
    if os.path.exists(map_path):
        clash.append(args.map)
    if clash or (args.start == 1 and os.path.isdir(out) and os.listdir(out)):
        print(json.dumps({"stop": "이미 있는 묶음 · 표는 덮지 않는다", "clash": clash, "dir": out}, ensure_ascii=False))
        return 2
    for name, rows in zip(names, batches):
        _write_jsonl(os.path.join(out, name), rows)
    _write_jsonl(map_path, table)
    doc = {"seed": args.seed, "size": args.size, "start": args.start, "batches": len(batches), "rows": len(table),
           "from": dict(collections.Counter(m["from"] for m in table))}
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    home = os.path.join(os.path.expanduser("~"), ".judgemodel")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trim", "pick", "check", "blind"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=home)
        p.add_argument("--work", default=os.path.join(home, "work-g9"))
        if name in ("pick", "check", "blind"):
            p.add_argument("--name", default=CAND_NAME, help="후보 파일 이름 앞부분 (work 폴더 안)")
        if name in ("pick", "check"):
            p.add_argument("--avoid", help="이 이름의 앞 판 후보와도 겹치지 않게 (본문 + 힌트를 같이 읽는다)")
        if name == "pick":
            p.add_argument("--pool", action="append", help="원천=파일 (그 원천 풀에 더한다)")
            p.add_argument("--seed", type=int, default=PICK_SEED)
            p.add_argument("--pools-only", action="store_true", help="기본 풀(POOLS)은 안 쓰고 --pool 만")
            p.add_argument("--quota", action="append", help="원천=전체:b (그 원천 몫을 바꾼다)")
            p.add_argument("--allow-form", action="append", help="받아 줄 생성기 꼴 (near-x 만)")
        if name == "blind":
            p.add_argument("--seed", type=int, default=BLIND_SEED)
            p.add_argument("--size", type=int, default=100)
            p.add_argument("--start", type=int, default=1, help="첫 묶음 번호 (보충 판은 앞 판 다음 번호)")
            p.add_argument("--no-old", action="store_true", help="v5 재라벨 대상(옛 dev 줄)을 섞지 않는다")
            p.add_argument("--map", default="g9-blind-map.jsonl", help="되돌리는 표 파일 이름")
    args = ap.parse_args(argv)
    return {"trim": trim_main, "pick": pick_main, "check": check_main, "blind": blind_main}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
