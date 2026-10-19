"""dev_g9 — dev-cmp-g9 재료 만들기와 얼리기 (K7 설계 1-2 · 4-1 2단계 · 4단계). 학습 · 라벨 판정은 안 한다.

    python -I dev_g9.py trim  [--root ~/.judgemodel] [--work ~/.judgemodel/work-g9]
    python -I dev_g9.py pick  [--root …] [--work …] [--pool 원천=파일 …] [--seed 20261801]
    python -I dev_g9.py check [--root …] [--work …]
    python -I dev_g9.py blind [--root …] [--work …] [--seed 20261802] [--size 100]

얼리기 (4단계, 라벨 판 · merge_g9 final 뒤) — 출력은 git 밖 `~/.judgemodel/sets/g9/` :
    python -I dev_g9.py assemble  [--root …] [--work …] [--dest ~/.judgemodel/sets/g9] [--seed 20261803]
    python -I dev_g9.py check-set [--root …] [--dest …]
    python -I dev_g9.py freeze    [--dest …]

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
- assemble : dev-cmp-g4 에서 빼기 쌍을 빼고, v5 쌍의 want 를 최종 라벨(labels-g9-final.jsonl)로 바꾸고,
          new-ko 최종 라벨에서 칸 몫(NEW_QUOTA)만큼 씨앗 고정으로 고른다 → dev-cmp-g9.jsonl + .summary.json.
          칸 안은 원천마다 고르게(b 는 B_SHARE 비), 원천 안은 생성기 꼴이 아닌 쌍 먼저. 모자란 칸은 있는 만큼만.
          보충 판(LABEL_ONLY)은 그 판의 칸만 풀에 넣는다 — 앞서 얼린 a · b · c 고르기가 같은 씨앗으로 그대로 남게.
- check-set : 조립한 셋 ↔ 학습 층 · blind-g4 열쇠 겹침(새 줄은 dev-cmp-g4 와도) · 셋 안 글 겹침. 0 아니면 exit 1.
- freeze : assemble_g3 와 같은 길(train/freeze.py write → check)로 FROZEN-dev-cmp-g9.sha256 을 쓴다.

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
# 얼리기 (4단계)
G9_SET = "dev-cmp-g9"
G9_DEST = "sets/g9"
FINAL_LABELS = "labels/labels-g9-final.jsonl"
CAND_NAMES = (CAND_NAME, "cands-g9-ko-extra", "cands-g9-ko-extra2")   # 새 줄 본문 · 힌트를 찾는 후보 파일 (본판 · 보충 판 둘)
NEW_QUOTA = {"b": 300, "a": 250, "c": 200, "d": 250}   # 설계 1-2 칸 몫
# 칸을 채우려고 뒤에 더한 보충 판 → 그 판에서 쓰는 칸. 다른 칸 줄은 풀에 안 넣는다 —
# 그래야 앞서 얼린 셋의 a · b · c 고르기(같은 씨앗)가 그대로고 모자란 칸만 는다 (10-18 d 보충)
LABEL_ONLY = {"cands-g9-ko-extra2": ("d",)}
ASSEMBLE_SEED = 20261803
GUIDE = "support-v6"
STATUS_OF = {"2": "agree", "3": "third", "split3": "split3"}
PROTOCOL_G9 = os.path.join(HERE, "..", "measure", "protocol-g9.json")


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


def avoid_rows(work, spec):
    """--avoid 값(쉼표로 여러 이름) → {"cands-g9": 앞 판 후보 줄 전부} 또는 None."""
    names = [n.strip() for n in (spec or "").split(",") if n.strip()]
    if not names:
        return None
    return {"cands-g9": [r for n in names for r in merged_rows(work, n)]}


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
    avoid = avoid_rows(work, args.avoid)
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
    avoid = avoid_rows(work, args.avoid)
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


# ---------------------------------------------------------------- assemble (얼리기)


def spread(total, avail, weights=None):
    """칸 몫 total 을 원천에 나눈다 → {원천: 수}. weights 비(없으면 고르게)로 큰 나머지 나눔을 하고,
    풀이 모자란 원천의 남는 몫은 아직 남은 원천에 같은 비로 다시 나눈다(다 모자라면 있는 만큼)."""
    keys = sorted(avail)
    weights = weights or {k: 1 for k in keys}
    got = {k: 0 for k in keys}
    while True:
        left = total - sum(got.values())
        open_ = {k: weights.get(k, 0) for k in keys if got[k] < avail[k] and weights.get(k, 0) > 0}
        if left <= 0 or not open_:
            return got
        share = ms._largest_remainder(left, open_)
        moved = 0
        for k, n in share.items():
            add = min(n, avail[k] - got[k])
            got[k] += add
            moved += add
        if not moved:
            return got


def _votes(f):
    out = {"j1": f["l1"], "j2": f["l2"]}
    if f.get("l3"):
        out["j3"] = f["l3"]
    return out


def _old_part(dev_rows, drop, v5, finals):
    """옛 dev-cmp-g4 줄 → (g9 줄, 셈). drop 은 빼고 v5 쌍은 최종 라벨로 다시 단다."""
    ids = {r["id"] for r in dev_rows}
    if set(drop) - ids:
        raise ValueError(f"빼기 id 중 dev-cmp-g4 에 없는 것 {len(set(drop) - ids)}개")
    if set(v5) - ids or set(v5) & set(drop):
        raise ValueError("v5 id 가 dev-cmp-g4 에 없거나 빼기 목록과 겹친다")
    old_final = {f["id"]: f for f in finals if f.get("set") == "old-dev"}
    if set(old_final) != set(v5):
        raise ValueError(f"old-dev 최종 라벨 {len(old_final)}쌍과 v5 목록 {len(v5)}쌍이 같지 않다")
    gone, redo = set(drop), set(v5)
    rows, changed = [], collections.Counter()
    for r in dev_rows:
        if r["id"] in gone:
            continue
        out = dict(r, set=G9_SET, g4_set=r.get("set"))
        if r["id"] in redo:
            f = old_final[r["id"]]
            out.update(g4_want=r["want"], g4_guide=r.get("label_guide"), want=f["label"], label_guide=GUIDE,
                       votes=_votes(f), status=STATUS_OF[f["agree"]], g9_label="v6-relabel")
            if f["label"] != r["want"]:
                changed[f"{r['want']}>{f['label']}"] += 1
        else:
            if r.get("label_guide") != GUIDE:
                raise ValueError(f"재라벨 대상이 아닌데 기준이 {r.get('label_guide')} 다 ({r['id']})")
            out["g9_label"] = "g4-carry"
        rows.append(out)
    rep = {"pairs_in": len(dev_rows), "dropped": len(drop), "kept": len(rows), "relabeled": len(v5),
           "changed": sum(changed.values()), "changed_by": dict(sorted(changed.items()))}
    return rows, rep


def _new_row(f, c):
    """최종 라벨 줄 + 후보 본문 · 힌트 → dev-cmp-g4 새 줄(v6-new)과 같은 꼴."""
    return {"id": c["id"], "evidence": c["evidence"], "claim": c["claim"], "want": f["label"],
            "hint": c.get("hint"), "form": c.get("form"), "domain": c["src"], "group": c.get("group"),
            "groups": c.get("groups") or [c.get("group")], "lang": c.get("lang", "ko"), "lic": c.get("lic", ""),
            "split": c.get("split"), "src_ids": c.get("src_ids") or [], "origin": c.get("origin"),
            "orig_id": c.get("orig_id"), "label_guide": GUIDE, "votes": _votes(f),
            "status": STATUS_OF[f["agree"]], "set": G9_SET, "g9_label": "v6-new"}


def _new_part(finals, cands, quota, seed, label_only=None):
    """new-ko 최종 라벨에서 칸 몫만큼 고르기 → (g9 줄, 셈). 칸은 이름 차례, 원천도 이름 차례로 같은 rng 를 쓴다.
    label_only = {후보 파일 이름: 쓰는 칸} — 그 파일 줄 중 다른 칸은 풀에서 뺀다(cands 줄의 cand_file 로 가린다)."""
    label_only = LABEL_ONLY if label_only is None else label_only
    news = [f for f in finals if f.get("set") == "new-ko"]
    missing = [f["id"] for f in news if f["id"] not in cands]
    if missing:
        raise ValueError(f"new-ko 최종 라벨 중 후보 파일에 없는 id {len(missing)}개")
    unused = collections.Counter()
    for f in news:
        name = cands[f["id"]].get("cand_file")
        if name in label_only and f["label"] not in label_only[name]:
            unused[f"{name}:{f['label']}"] += 1
    news = [f for f in news if not (cands[f["id"]].get("cand_file") in label_only
                                    and f["label"] not in label_only[cands[f["id"]]["cand_file"]])]
    bad = [f["id"] for f in news if f["label"] not in NEW_QUOTA]
    if bad:
        raise ValueError(f"모르는 라벨 {len(bad)}개")
    rng = random.Random(seed)
    picked, rep = [], {"pool": {}, "quota": dict(quota), "picked": {}, "shortfall": {}, "by_src": {}}
    for label in sorted(quota):
        pool = collections.defaultdict(list)
        for f in sorted((f for f in news if f["label"] == label), key=lambda f: f["id"]):
            pool[cands[f["id"]]["src"]].append(f)
        avail = {s: len(v) for s, v in pool.items()}
        if label == "b" and set(avail) - set(B_SHARE):
            raise ValueError(f"b 몫(B_SHARE)에 없는 원천 {sorted(set(avail) - set(B_SHARE))}")
        plan = spread(quota[label], avail, B_SHARE if label == "b" else None) if avail else {}
        took = 0
        for src in sorted(plan):
            plain = [f for f in pool[src] if cands[f["id"]].get("form") not in GENERATOR_FORMS]
            gen = [f for f in pool[src] if cands[f["id"]].get("form") in GENERATOR_FORMS]
            rng.shuffle(plain)
            rng.shuffle(gen)
            chosen = (plain + gen)[:plan[src]]
            picked += [_new_row(f, cands[f["id"]]) for f in chosen]
            took += len(chosen)
        rep["pool"][label] = sum(avail.values())
        rep["picked"][label] = took
        rep["by_src"][label] = {s: plan[s] for s in sorted(plan)}
        if took < quota[label]:
            rep["shortfall"][label] = quota[label] - took
    if unused:
        rep["label_only_unused"] = dict(sorted(unused.items()))
    picked.sort(key=lambda r: r["id"])
    return picked, rep


def assemble(dev_rows, drop, v5, finals, cands, quota=None, seed=ASSEMBLE_SEED, label_only=None):
    """dev-cmp-g9 줄 목록과 셈. cands = {id: 후보 본문 + 힌트}. 옛 줄(옛 차례) 다음에 새 줄(id 차례)."""
    # 최종 라벨 줄의 꼴을 먼저 본다 — 옛 줄 쪽에서 KeyError 로 죽지 않고 exit 2 로 멈추게
    bad = [f.get("id") for f in finals if f.get("label") not in NEW_QUOTA or f.get("agree") not in STATUS_OF]
    if bad:
        raise ValueError(f"최종 라벨 줄 중 label · agree 가 모르는 값인 것 {len(bad)}개 (첫 id {bad[0]})")
    ids = [f.get("id") for f in finals]
    if len(set(ids)) != len(ids):
        raise ValueError("최종 라벨 id 가 겹친다")
    old, old_rep = _old_part(dev_rows, drop, v5, finals)
    new, new_rep = _new_part(finals, cands, quota or NEW_QUOTA, seed, label_only)
    rows = old + new
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("dev-cmp-g9 id 가 겹친다")
    return rows, {"seed": seed, "old": old_rep, "new": new_rep}


def gen_counts(rows, forms):
    """규약 generator_forms_counts 꼴 — near-x · removed 는 전체와 @ko, 나머지 생성기 꼴은 others 하나로."""
    out = {"near-x": 0, "near-x@ko": 0, "removed": 0, "removed@ko": 0, "others": 0}
    for r in rows:
        form = r.get("form")
        if form in ("near-x", "removed"):
            out[form] += 1
            out[form + "@ko"] += r.get("lang") == "ko"
        elif form in forms:
            out["others"] += 1
    return out


def set_summary(rows, rep, sha, forms):
    def by(key):
        table = collections.defaultdict(collections.Counter)
        for r in rows:
            table[str(r.get(key))][r["want"]] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}

    return {"set": G9_SET, "draft": False, "pairs": len(rows), "sha256": sha,
            "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
            "lang_want": by("lang"), "domain_want": by("domain"),
            "form": dict(collections.Counter(str(r.get("form")) for r in rows).most_common()),
            "g9_label": dict(sorted(collections.Counter(r["g9_label"] for r in rows).items())),
            "generator_forms_counts": gen_counts(rows, forms), **rep}


def load_cands(work, names=CAND_NAMES):
    """후보 본문 + 힌트 → {id: 줄}. 파일 사이 id 가 겹치면 멈춘다."""
    out = {}
    for name in names:
        for row in merged_rows(work, name):
            if row["id"] in out:
                raise ValueError(f"후보 id 가 파일 사이에 겹친다 ({name})")
            out[row["id"]] = dict(row, cand_file=name)   # LABEL_ONLY 가리기용 — _new_row 는 안 옮긴다
    return out


def _read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _protocol_forms():
    with open(PROTOCOL_G9, encoding="utf-8") as f:
        return set(json.load(f).get("generator_forms") or [])


def assemble_main(args):
    root = os.path.expanduser(args.root)
    work = os.path.expanduser(args.work)
    dest = os.path.expanduser(args.dest or _p(root, G9_DEST))
    if ms._inside_git(dest):
        print(json.dumps({"stop": "dest 폴더가 git 안이다"}, ensure_ascii=False))
        return 2
    if os.path.exists(os.path.join(dest, f"FROZEN-{G9_SET}.sha256")):
        print(json.dumps({"stop": "이미 얼린 셋이다 — 다시 조립하지 않는다", "dest": dest}, ensure_ascii=False))
        return 2
    dev_path = _p(root, DEV)
    if _sha_file(dev_path) != DEV_SHA:
        print(json.dumps({"stop": "dev-cmp-g4 sha 가 얼린 값과 다르다"}, ensure_ascii=False))
        return 2
    try:
        rows, rep = assemble(ms.read_jsonl(dev_path), _read_ids(os.path.join(work, "dev-g4-drop.ids.txt")),
                             _read_ids(os.path.join(work, "dev-g4-v5.ids.txt")),
                             ms.read_jsonl(os.path.join(work, *FINAL_LABELS.split("/"))), load_cands(work),
                             seed=args.seed)
    except ValueError as e:
        print(json.dumps({"stop": str(e)}, ensure_ascii=False))
        return 2
    path = os.path.join(dest, G9_SET + ".jsonl")
    _write_jsonl(path, rows)
    summary = set_summary(rows, rep, _sha_file(path), _protocol_forms())
    with open(os.path.join(dest, G9_SET + ".summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("pairs", "sha256", "want", "lang_want", "g9_label",
                                              "generator_forms_counts", "old", "new")}, ensure_ascii=False, indent=1))
    return 0


def check_set(rows, marks):
    """조립한 셋 ↔ Marks 겹침 셈. 옛 줄(g9_label ≠ v6-new)은 dev-cmp-g4 에서 왔으니 그 Marks 는 건너뛴다.
    셋 안 글 겹침(text_within)도 센다."""
    counts, keys = collections.Counter(), set()
    for row in rows:
        for name, m in marks.items():
            if name == "dev-cmp-g4" and row.get("g9_label") != "v6-new":
                continue
            for why in m.reasons(row):
                counts[f"{why}_{name}"] += 1
        both = ms._both_keys(row)
        if both & keys:
            counts["text_within"] += 1
        keys |= both
    names = [f"{w}_{n}" for n in marks for w in ("group", "src", "text", "sha", "para")] + ["text_within"]
    return {k: counts.get(k, 0) for k in names}


def old_is_soft(key):
    """옛 줄의 물려받은 겹침 — 막지 않고 보고만. 문단 겹침(trim 규칙 밖)과 K7 처방 밖 학습 층 겹침."""
    return key.startswith("para_") or key.endswith("_other-train")


def judge_check(new_out, old_out):
    """(막는 수, 보고만 수). 새 줄은 모두 막고, 옛 줄은 old_is_soft 만 보고만."""
    block = sum(new_out.values()) + sum(v for k, v in old_out.items() if not old_is_soft(k))
    return block, sum(v for k, v in old_out.items() if old_is_soft(k))


def old_marks(root):
    """옛 줄용 Marks — K7 처방 층 · 처방 밖 층 · blind-g4 열쇠(forbid 만). reserve-g4 예약은 dev-cmp-g4 자기 몫이
    들어 있어(124쌍) 옛 줄에는 쓰지 않는다."""
    marks = {"recipe": Marks(), "other-train": Marks(), "blind-g4": Marks()}
    for name, layers in (("recipe", RECIPE_LAYERS), ("other-train", OTHER_LAYERS)):
        for rel in layers.values():
            marks[name].add_rows(ms.read_jsonl(_p(root, rel)), name)
    with open(_p(root, BLIND_FORBID), encoding="utf-8") as f:
        marks["blind-g4"].add_forbid(json.load(f))
    return marks


def check_set_main(args):
    root = os.path.expanduser(args.root)
    dest = os.path.expanduser(args.dest or _p(root, G9_DEST))
    marks, missing = load_marks(root, with_reserve=True)
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    rows = ms.read_jsonl(os.path.join(dest, G9_SET + ".jsonl"))
    new_rows = [r for r in rows if r.get("g9_label") == "v6-new"]
    new_out = check_set(new_rows, marks)
    old_out = check_set([r for r in rows if r.get("g9_label") != "v6-new"], old_marks(root))
    within = check_set(rows, {})["text_within"]
    for out in (new_out, old_out):
        out.pop("text_within")                       # 셋 안 글 겹침은 셋 전체로 한 번만 센다
    block, soft = judge_check(new_out, old_out)
    block += within
    doc = {"set": G9_SET, "pairs": len(rows), "new": new_out, "old": old_out, "text_within_all": within,
           "old_soft": {k: v for k, v in old_out.items() if v and old_is_soft(k)},
           "soft_note": "옛 줄(얼린 dev-cmp-g4 에서 온 줄)의 문단 겹침 · K7 처방 밖 층 겹침은 막지 않고 보고만",
           "total": block, "soft": soft}
    with open(os.path.join(dest, f"check-{G9_SET}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 1 if doc["total"] else 0


def freeze_main(args):
    from assemble_g3 import cmd_freeze   # build/ 안 모듈 — 얼릴 때만 부른다
    dest = os.path.expanduser(args.dest or _p(os.path.expanduser(args.root), G9_DEST))
    return cmd_freeze(argparse.Namespace(dest=dest, set=G9_SET))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    home = os.path.join(os.path.expanduser("~"), ".judgemodel")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trim", "pick", "check", "blind", "assemble", "check-set", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=home)
        p.add_argument("--work", default=os.path.join(home, "work-g9"))
        if name in ("assemble", "check-set", "freeze"):
            p.add_argument("--dest", help="셋 폴더 (기본 <root>/sets/g9)")
        if name == "assemble":
            p.add_argument("--seed", type=int, default=ASSEMBLE_SEED)
        if name in ("pick", "check", "blind"):
            p.add_argument("--name", default=CAND_NAME, help="후보 파일 이름 앞부분 (work 폴더 안)")
        if name in ("pick", "check"):
            p.add_argument("--avoid", help="이 이름의 앞 판 후보와도 겹치지 않게 (쉼표로 여럿, 본문 + 힌트를 같이 읽는다)")
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
    return {"trim": trim_main, "pick": pick_main, "check": check_main, "blind": blind_main, "assemble": assemble_main,
            "check-set": check_set_main, "freeze": freeze_main}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
