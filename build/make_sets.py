"""make_sets — 열쇠 묶음 예약 · 학습 후보 고르기 · 겹침 검사 (설계 d자료층재구축 2-4 · 3-2 · 3-5).

나누기 차례가 먼저다 : 원천마다 묶음을 dev-cal · dev-cmp · blind-g1 몫으로 먼저 예약하고(reserve),
학습 후보는 예약 묶음 밖에서만 고른다(pool). check 는 학습 후보가 예약 묶음 · ood · 옛 셋과
묶음 · 글 · 기억 id 어느 것으로도 안 겹치는지 본다.

    python -I make_sets.py reserve --build <build>/g1
    python -I make_sets.py pool    --build <build>/g1
    python -I make_sets.py check   --build <build>/g1 --cands <build>/g1/train-cands.jsonl --old-root ~/.judgemodel
    python -I make_sets.py attach-rewrite --cands <train-cands> --rewrite <rewrite-train-*.jsonl> --out <klue 줄>
    python -I make_sets.py pool --archive --build <build>/g1      # 아카이브 · mem 학습 후보 (소단계 48)
    python -I make_sets.py reserve-g2 --build <build>/g2           # v1.2 B 판 b 자 예약 · 학습 층
    python -I make_sets.py check-g2   --build <build>/g2
    python -I make_sets.py reserve-g3 --build <build>/g3           # v1.2 K 판 g3 평가 셋 새 쌍 예약 · 학습 층
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata

SEED = 20261011
FACTOR = 2.5                     # 라벨 남김률 60% · 칸 맞추기를 보고 후보를 넉넉히 예약
SET_SIZES = {"dev-cal": 320, "dev-cmp": 1000, "blind-g1": 700}
DEV_SHARE = {"proposals": 0.15, "vitaminc": 0.15, "klue-mrc": 0.13, "jira": 0.12}   # 설계 2-4 공개 몫
DEV_POOLS = {"proposals": ("any",), "vitaminc": ("heldout",), "klue-mrc": ("heldout",), "jira": ("any",)}
DEV_GROUP_CAP = {"proposals": 6, "vitaminc": 4, "klue-mrc": 3, "jira": 4}
SOURCES = ("jira", "proposals", "vitaminc", "klue-mrc", "minicheck")

TRAIN_TOTAL = 4000
TRAIN_SHARE_JIRA = {"jira": 0.25, "proposals": 0.25, "vitaminc": 0.20, "klue-mrc": 0.20, "minicheck": 0.10}
TRAIN_SHARE_NO_JIRA = {"proposals": 0.35, "vitaminc": 0.25, "klue-mrc": 0.25, "minicheck": 0.15}
TRAIN_GROUP_CAP = 8
# 원천 안 (hint, form) 몫 — d 꼴 다섯이 d 의 10~30% 에 들고, 한 칸이 원천의 50% 를 안 넘게 (설계 3-2)
MIX = {
    "proposals": {("d", "link"): 0.143, ("d", "same-doc"): 0.15, ("d", "near"): 0.21,
                  ("a", "same-sec"): 0.18, ("b", "rejected"): 0.14, ("c", "other-doc"): 0.177},
    "vitaminc": {("d", "same-doc"): 0.28, ("d", "near"): 0.12, ("a", "labeled"): 0.20,
                 ("b", "labeled"): 0.25, ("c", "same-page"): 0.15},
    "klue-mrc": {("d", "unanswerable"): 0.35, ("d", "removed"): 0.15, ("a", "qa-answer"): 0.25,
                 ("c", "other-doc"): 0.25},
    "minicheck": {("d", "removed"): 0.42, ("a", "doc-top"): 0.36, ("c", "other-doc"): 0.22},
    "jira": {("d", "link"): 0.40, ("a", "duplicate"): 0.20, ("b", "wontfix"): 0.15,
             ("c", "other-component"): 0.25},
}

OLD_SETS = {
    "r1-04": "twins/2026-10-09/train-sg-twins.jsonl",
    "train-r2": "sets/2026-10-10/build/train-r2.jsonl",
    "synth-r2": "sets/2026-10-10/build/synth-r2/synth-r2.jsonl",
    "synth-r4": "sets/2026-10-10/build/synth-r4/synth-r4.jsonl",
    "dev-r1": "sets/2026-10-10/build/dev-r1-fixed.jsonl",
    "dev-r0": "sets/2026-10-09/dev.jsonl",
    "blind-v1": "sets/2026-10-09/blind-v1.jsonl",
    "blind-v2": "sets/2026-10-09/blind.jsonl",
    "blind-v2-ids": "sets/2026-10-10/build/blind-v2-ids.txt",
    "blind-v3": "sets/2026-10-10/blind-v3.jsonl",
    "blind-v4": "sets/2026-10-10/blind-v4.jsonl",
    "fold40": "sets/2026-10-10/fold40.jsonl",
}

_MEM_ID = re.compile(r"\b\d{8}-[0-9a-f]{8}\b")
_SPACE = re.compile(r"\s+")


def norm(text):
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip().lower()


def pair_key(row):
    """(근거, 주장) 정규화 열쇠. 주장이 아직 없는 후보(다시 쓸 것)는 질문으로 대신한다."""
    claim = row.get("claim") or row.get("hypothesis") or row.get("question") or ""
    evidence = row.get("evidence") or row.get("premise") or ""
    return norm(evidence), norm(claim)


def text_keys(row):
    """(근거, 주장) 정규형 열쇠 양쪽 차례. evidence · claim 칸만 본다."""
    ev, cl = norm(row.get("evidence")), norm(row.get("claim"))
    return {(ev, cl), (cl, ev)}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_source(build, source):
    folder = os.path.join(build, source)
    cand_path = os.path.join(folder, "candidates.jsonl")
    group_path = os.path.join(folder, "groups.json")
    if not os.path.exists(cand_path) or not os.path.exists(group_path):
        return None, None
    with open(group_path, encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    return read_jsonl(cand_path), groups


# ---------------------------------------------------------------- reserve

def dev_shares(has_jira):
    shares = dict(DEV_SHARE)
    if not has_jira:
        half = shares.pop("jira") / 2
        shares["proposals"] += half
        shares["vitaminc"] += half
    return shares


def reserve(build, seed=SEED, factor=FACTOR):
    rng = random.Random(seed)
    loaded = {s: load_source(build, s) for s in SOURCES}
    has_jira = bool(loaded["jira"][0])
    shares = dev_shares(has_jira)
    out = {"seed": seed, "factor": factor, "jira": has_jira, "shares": shares, "sets": {}, "reserved": {}}
    taken = set()
    for source, share in shares.items():
        rows, groups = loaded[source]
        if not rows:
            continue
        inside = collections.Counter(r["group"] for r in rows if len(r["groups"]) == 1)
        eligible = sorted(g for g, meta in groups.items()
                          if meta["pool"] in DEV_POOLS[source] and inside.get(g, 0) > 0)
        rng.shuffle(eligible)
        cursor = 0
        for set_name, size in SET_SIZES.items():
            need = math.ceil(size * share * factor)
            picked = []
            usable = 0
            while usable < need and cursor < len(eligible):
                gid = eligible[cursor]
                cursor += 1
                picked.append(gid)
                usable += min(inside[gid], DEV_GROUP_CAP[source])
            out["sets"].setdefault(set_name, {})[source] = {
                "pairs": round(size * share), "need_candidates": need, "usable_candidates": usable,
                "groups": sorted(picked)}
            taken.update(picked)
    ood_rows, ood_groups = load_source(build, "ood")
    if ood_groups:
        out["sets"]["ood-g1"] = {"ood": {"groups": sorted(ood_groups), "candidates": len(ood_rows)}}
        taken.update(ood_groups)
    out["reserved"] = sorted(taken)
    return out


# ---------------------------------------------------------------- reserve (아카이브 · mem)

ARCH_SHARE = {"mem": 0.30, "archive": 0.15}          # 설계 2-4 우리 도메인 몫
ARCH_SOURCES = {"mem": ("mem",), "archive": ("archive/git", "archive/docs")}
ARCH_GROUP_CAP = 6
ARCH_ORDER = ("blind-g1", "dev-cmp", "dev-cal")


def _arch_rows(build, domain):
    rows, groups = [], {}
    for source in ARCH_SOURCES[domain]:
        got, meta = load_source(build, source)
        rows.extend(got or [])
        groups.update(meta or {})
    return rows, groups


def _row_fits(row, set_name):
    """blind-g1 은 dev-r2 다시 붙이기 줄을 못 쓴다 (옛 셋에 없는 기억만)."""
    if set_name == "blind-g1":
        return row.get("split") == "any" and not row.get("relabel")
    return row.get("split") in ("any", "dev")


def _relabel_blocks(rows):
    """dev-r2 다시 붙이기 줄이 잇는 묶음 덩이 [(묶음 id 목록, 줄 수)] — 덩이째 한 셋에 넣어야 줄이 안 잘린다. 작은 것 먼저."""
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    relabel = [r for r in rows if r.get("relabel")]
    for row in relabel:
        for gid in row["groups"]:
            a, b = find(row["groups"][0]), find(gid)
            if a != b:
                parent[b] = a
    blocks = collections.defaultdict(set)
    sizes = collections.Counter()
    for row in relabel:
        root = find(row["group"])
        blocks[root].update(row["groups"])
        sizes[root] += 1
    return [(sorted(blocks[root]), sizes[root]) for root in sorted(blocks, key=lambda r: (sizes[r], r))]


def reserve_archive(build, seed=SEED, factor=FACTOR):
    """mem 30% · 아카이브 15% 몫을 묶음 단위로 예약한다. 줄의 다른 묶음 · 출처 id 도 같은 셋이 가져간다.

    mem 차례 : ① dev-r2 다시 붙이기 묶음을 dev 둘에 ② 새 기억을 blind-g1 에 ③ 남은 새 기억을 dev 에.
    다시 붙이기 줄은 이미 뽑힌 쌍이라 한 줄을 한 쌍으로 센다(새 후보는 factor 배).
    mem 이 모자란 몫은 같은 셋의 아카이브 몫으로 넘긴다 (설계 2-4 「모자라면 아카이브 몫을 늘린다」).
    """
    rng = random.Random(seed)
    out = {"seed": seed, "factor": factor, "shares": dict(ARCH_SHARE), "sets": {}, "shift": {}}
    owner_group, owner_src = {}, {}
    carry = collections.Counter()
    for domain in ("mem", "archive"):
        rows, _ = _arch_rows(build, domain)
        by_group = collections.defaultdict(list)
        for row in rows:
            by_group[row["group"]].append(row)
        order = sorted(by_group)
        rng.shuffle(order)
        relabel_blocks = _relabel_blocks(rows) if domain == "mem" else []
        state = {s: {"pairs": round(SET_SIZES[s] * ARCH_SHARE[domain]) + carry[s], "got": 0.0,
                     "picked": [], "row_ids": [], "usable": 0} for s in ARCH_ORDER}
        if domain == "mem":
            cal, cal_pairs = [], state["dev-cal"]["pairs"]
            for block in list(relabel_blocks):
                if sum(n for _, n in cal) + block[1] <= cal_pairs * 1.2:
                    cal.append(block)
                    relabel_blocks.remove(block)
            phases = [("dev-cal", [g for b, _ in cal for g in b]), ("dev-cmp", [g for b, _ in relabel_blocks for g in b]),
                      ("blind-g1", order), ("dev-cmp", order), ("dev-cal", order)]
        else:
            phases = [(s, order) for s in ARCH_ORDER]
        for set_name, groups in phases:
            st = state[set_name]
            relabel_phase = groups is not order
            for gid in groups:
                if st["got"] >= st["pairs"] and not relabel_phase:
                    break
                if gid in st["picked"] or owner_group.get(gid, set_name) != set_name:
                    continue
                fit = [r for r in by_group[gid] if _row_fits(r, set_name)
                       and all(owner_group.get(g, set_name) == set_name for g in r["groups"])
                       and all(owner_src.get(s, set_name) == set_name for s in r["src_ids"])]
                fresh = [r for r in fit if not r.get("relabel")][:ARCH_GROUP_CAP]
                fit = [r for r in fit if r.get("relabel")] + fresh
                if not fit:
                    continue
                st["picked"].append(gid)
                for row in fit:
                    st["row_ids"].append(row["id"])
                    owner_group.update({g: set_name for g in row["groups"]})
                    owner_src.update({s: set_name for s in row["src_ids"]})
                    st["got"] += 1.0 if row.get("relabel") else 1.0 / factor
                st["usable"] += len(fit)
        for set_name in ARCH_ORDER:
            st = state[set_name]
            short = max(0, st["pairs"] - int(st["got"])) if domain == "mem" else 0
            if short:
                carry[set_name] += short
                out["shift"][set_name] = short
            out["sets"].setdefault(set_name, {})[domain] = {
                "pairs": st["pairs"] - short, "pairs_covered": round(st["got"], 1),
                "usable_candidates": st["usable"], "groups": sorted(st["picked"]), "row_ids": sorted(st["row_ids"])}
    out["reserved"] = sorted(owner_group)
    out["reserved_src_ids"] = sorted(owner_src)
    return out


def arch_cross_check(doc):
    """셋 사이에 같은 줄 · 묶음이 없는지 (아카이브 예약 자체 검사)."""
    seen_rows, seen_groups, counts = {}, {}, collections.Counter()
    for set_name, per in doc["sets"].items():
        for entry in per.values():
            for rid in entry["row_ids"]:
                if seen_rows.setdefault(rid, set_name) != set_name:
                    counts["row_across_sets"] += 1
            for gid in entry["groups"]:
                if seen_groups.setdefault(gid, set_name) != set_name:
                    counts["group_across_sets"] += 1
    return {k: counts.get(k, 0) for k in ("row_across_sets", "group_across_sets")}


def reserved_groups(reserve_doc):
    return set(reserve_doc["reserved"])


# ---------------------------------------------------------------- pool

def _quotas(source_total, mix, available):
    """몫대로 나누고, 모자란 칸은 같은 hint 의 다른 꼴 → 원천 안 다른 칸 차례로 메운다."""
    want = {key: round(source_total * frac) for key, frac in mix.items()}
    final = {key: min(n, available.get(key, 0)) for key, n in want.items()}
    short = sum(want.values()) - sum(final.values())
    while short > 0:
        hints_short = [k for k in want if final[k] < want[k]]
        spare = [k for k in want if final[k] < available.get(k, 0)]
        if not spare:
            break
        same_hint = [k for k in spare if any(k[0] == h[0] for h in hints_short)]
        target = (same_hint or spare)
        target.sort(key=lambda k: final[k] / max(1, want[k]))
        final[target[0]] += 1
        short -= 1
    return final


def held_keys(build, banned):
    """예약 묶음(· ood)에 든 후보의 (근거, 주장) 열쇠 — 양쪽 차례 다."""
    keys = set()
    for source in SOURCES + ("ood",):
        rows, _ = load_source(build, source)
        for row in rows or []:
            if set(row["groups"]) & banned:
                key = pair_key(row)
                keys.add(key)
                keys.add((key[1], key[0]))
    return keys


def pool(build, reserve_doc, seed=SEED, total=TRAIN_TOTAL, group_cap=TRAIN_GROUP_CAP):
    rng = random.Random(seed)
    banned = reserved_groups(reserve_doc)
    held = held_keys(build, banned)
    shares = TRAIN_SHARE_JIRA if reserve_doc.get("jira") else TRAIN_SHARE_NO_JIRA
    chosen = []
    report = {}
    for source, share in shares.items():
        rows, _ = load_source(build, source)
        if not rows:
            continue
        free = [r for r in rows if r["split"] in ("train", "any") and not set(r["groups"]) & banned
                and pair_key(r) not in held]
        buckets = collections.defaultdict(list)
        for row in free:
            buckets[(row["hint"], row["form"])].append(row)
        mix = MIX[source]
        quotas = _quotas(round(total * share), mix, {k: len(v) for k, v in buckets.items() if k in mix})
        per_group = collections.Counter()
        picked_here = collections.Counter()
        for key in sorted(quotas):
            bucket = sorted(buckets.get(key, []), key=lambda r: r["id"])
            rng.shuffle(bucket)
            for row in bucket:
                if picked_here[key] >= quotas[key]:
                    break
                if per_group[row["group"]] >= group_cap:
                    continue
                per_group[row["group"]] += 1
                picked_here[key] += 1
                chosen.append(row)
        report[source] = {
            "target": round(total * share), "picked": sum(picked_here.values()),
            "groups": len(per_group), "free_candidates": len(free),
            "by_hint_form": {f"{h}:{f}": picked_here[(h, f)] for h, f in sorted(mix)},
        }
    d_rows = [r for r in chosen if r["hint"] == "d"]
    d_forms = collections.Counter(r["form"] for r in d_rows)
    report["d_form_share"] = {form: round(n / max(1, len(d_rows)), 3) for form, n in sorted(d_forms.items())}
    report["hint_total"] = dict(sorted(collections.Counter(r["hint"] for r in chosen).items()))
    report["total"] = len(chosen)
    return chosen, report


# ---------------------------------------------------------------- check

def _walk_pairs(node, keys, ids):
    if isinstance(node, dict):
        if ("evidence" in node and "claim" in node) or ("premise" in node and "hypothesis" in node):
            key = pair_key(node)
            keys.add(key)
            keys.add((key[1], key[0]))
        for value in node.values():
            _walk_pairs(value, keys, ids)
    elif isinstance(node, list):
        for value in node:
            _walk_pairs(value, keys, ids)


def load_old(old_root, names=OLD_SETS):
    """옛 셋 → (쌍 열쇠, 기억 id, 못 찾은 이름)."""
    keys, ids, missing = set(), set(), []
    for name, rel in names.items():
        path = os.path.join(old_root, rel)
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids.update(_MEM_ID.findall(line))
                if path.endswith(".jsonl") and line.strip():
                    _walk_pairs(json.loads(line), keys, ids)
    return keys, ids, missing


def check(cands, reserve_doc, build, old_keys=frozenset(), old_ids=frozenset()):
    """학습 후보 × (예약 묶음 · 예약 글 · ood · 옛 셋) 겹침 수."""
    banned = reserved_groups(reserve_doc)
    held = held_keys(build, banned)
    counts = collections.Counter()
    for row in cands:
        key = pair_key(row)
        if set(row.get("groups") or [row.get("group")]) & banned:
            counts["group_reserved"] += 1
        if key in held:
            counts["text_reserved"] += 1
        if key in old_keys:
            counts["text_old"] += 1
        if set(_MEM_ID.findall(json.dumps(row.get("src_ids", [])))) & set(old_ids):
            counts["mem_id_old"] += 1
    return {k: counts.get(k, 0) for k in ("group_reserved", "text_reserved", "text_old", "mem_id_old")}


def arch_overlap(cands, arch_doc):
    """학습 후보 × 아카이브 예약(묶음 · 출처 id) 겹침 수. 예약에 든 줄 자체는 세지 않는다."""
    banned, srcs = set(arch_doc["reserved"]), set(arch_doc["reserved_src_ids"])
    own = {rid for per in arch_doc["sets"].values() for entry in per.values() for rid in entry["row_ids"]}
    counts = collections.Counter()
    for row in cands:
        if row.get("id") in own:
            continue
        if set(row.get("groups") or [row.get("group")]) & banned:
            counts["arch_group_reserved"] += 1
        if set(row.get("src_ids") or []) & srcs:
            counts["arch_src_reserved"] += 1
    return {k: counts.get(k, 0) for k in ("arch_group_reserved", "arch_src_reserved")}


# ---------------------------------------------------------------- eval-pool (공개 원천 평가 후보, 소단계 45-라)
#
# 예약 묶음(셋별) 안에서만 평가 후보를 고른다. 칸 비율은 설계 2-4 표의 셋별 (a · b · c · d) 를 hint 로 맞추고,
# 라벨 뒤 버림을 보고 EVAL_FACTOR 배를 고른다. c 는 같은 셋 · 같은 원천의 예약 묶음끼리 BM25 근접(10~60위)으로
# 다시 뽑는다(near-x) — 학습 쪽 묶음과 이어진 c 는 못 쓰기 때문이다. 셋 사이는 잇지 않는다.

EVAL_FACTOR = 1.6
EVAL_SOURCES = ("proposals", "vitaminc", "klue-mrc", "jira")
EVAL_HINT = {"dev-cal": {"a": 80, "b": 70, "c": 70, "d": 100},
             "dev-cmp": {"a": 220, "b": 180, "c": 200, "d": 400},
             "blind-g1": {"a": 170, "b": 140, "c": 150, "d": 240}}
NEAR_C = "near-x"
NEAR_C_MIN = 0.5                 # c 의 절반 이상은 근접 c (설계 3-2 「c 도 어렵게」)
KLUE_TRUE_FORMS = ("qa-answer", "removed")   # 근접 c 의 주장으로 쓸 KLUE 평서문 (지문에 맞는 말)
OOD_SIZE = 200
OOD_KINDS = ("wice", "sf", "cnli")
OOD_GROUP_CAP = 2

_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_EN_AUX = frozenset("is are was were be been being has have had do does did will would can could shall should "
                    "may might must said says".split())
_EN_STOP = frozenset("the a an of to in on at for with by and or but that this which who as from it its he she "
                     "they his her their not del di de la le van von der".split())
_EN_VERBISH = re.compile(r"^[a-z][a-z'\-]+(ed|es|ing)$")
_CLOSED = re.compile(r"[.!?。][\"'’”」)\]\s]*$")
_KO_PRED = re.compile(r"(다|요|음|함|됨|임)(?=[.!?\s\"'」)\]]|$)"
                      r"|(했|하는|되는|있는|없는|된다|한다|이다|였|었|았|하며|으며|이며|하고|되어|하여)")


def _sentence_like(text):
    """글 한 토막이 문장(서술어가 있는 글)인가. 한국어는 서술 끝말, 영어는 조동사 · 동사꼴 · 소문자 내용어 비율로 본다."""
    if re.search(r"[가-힣]", text):          # 서술 끝말이 있거나, 낱말 넷 이상이 마침표로 닫힌 명사 끝 문장(「…한 것.」)
        return bool(_KO_PRED.search(text) or (len(text.split()) >= 4 and _CLOSED.search(text)))
    words = _EN_WORD.findall(text)
    if len(words) < 3:
        return False
    lower = [w for w in words if w[0].islower()]
    content = sum(1 for w in lower if w not in _EN_STOP and len(w) > 2) / len(words)
    if content < 0.03:
        return False
    verb = any(w.lower() in _EN_AUX for w in words) or any(_EN_VERBISH.match(w) for w in lower)
    closed = len(words) >= 6 and content > 0 and _CLOSED.search(text)
    return bool(verb or closed or content >= 0.2)


def list_like(text):
    """근거가 목록 · 제목 · 메뉴뿐인가 — 줄이 셋 이상이면 문장 줄이 1/3 아래일 때, 한 덩이면 문장꼴이 아닐 때 참."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) >= 3:
        return sum(_sentence_like(line) for line in lines) / len(lines) < 1 / 3
    return not _sentence_like(" ".join(lines))


def _largest_remainder(total, weights):
    """weights 비율로 total 을 정수로 나눈다 (큰 나머지 차례)."""
    keys = sorted(weights)
    whole = sum(weights.values()) or 1
    raw = {k: total * weights[k] / whole for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    for k in sorted(keys, key=lambda k: -(raw[k] - out[k]))[:total - sum(out.values())]:
        out[k] += 1
    return out


def eval_hint_targets(set_name, sizes, allowed):
    """원천별 고를 수 sizes · 원천이 낼 수 있는 칸 allowed → {원천: {칸: 수}}.

    칸이 빠진 원천(KLUE 는 b 없음)은 제 칸끼리 비율을 다시 맞추고, 그 몫을 뺀 나머지를 다른 원천이 크기 비례로 나눈다.
    """
    ratio = EVAL_HINT[set_name]
    whole = sum(ratio.values())
    total = sum(sizes.values())
    goal = {h: total * n / whole for h, n in ratio.items()}
    out = {}
    narrow = [s for s in sizes if set(allowed[s]) != set(ratio)]
    for s in narrow:
        out[s] = _largest_remainder(sizes[s], {h: ratio[h] for h in allowed[s]})
    rest = {h: max(0.0, goal[h] - sum(out[s].get(h, 0) for s in narrow)) for h in ratio}
    for s in sizes:
        if s not in narrow:
            out[s] = _largest_remainder(sizes[s], rest)
    return out


def _spread_pick(rows, n, per_group, rng):
    """덜 쓴 묶음부터 n 줄 — 같은 묶음에 몰리지 않게 (per_group 은 원천 안 여러 칸이 같이 쓴다)."""
    pool_rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(pool_rows)
    picked = []
    while pool_rows and len(picked) < n:
        best = min(range(len(pool_rows)), key=lambda i: per_group[pool_rows[i]["group"]])
        row = pool_rows.pop(best)
        per_group[row["group"]] += 1
        picked.append(row)
    return picked


def near_c(rows, n, rng, per_group, prefix, claim_rows=None):
    """같은 셋 · 같은 원천 예약 묶음끼리 근접 c : 묶음 A 의 근거 × 다른 묶음 B 의 주장(BM25 10~60위). groups 에 둘 다."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:                     # python -I 는 스크립트 폴더를 import 길에 안 넣는다
        sys.path.insert(0, here)
    from bm25 import BM25   # build/ 안 모듈 — 이 판에서만 쓴다

    claims, seen = [], set()
    for row in sorted(claim_rows if claim_rows is not None else rows, key=lambda r: r["id"]):
        key = norm(row.get("claim"))
        if key and key not in seen:
            seen.add(key)
            claims.append(row)
    if not claims or n <= 0:
        return []
    index = BM25([c["claim"] for c in claims])
    evs, seen_ev = [], set()
    for row in sorted(rows, key=lambda r: r["id"]):
        key = norm(row.get("evidence"))
        if key and key not in seen_ev:
            seen_ev.add(key)
            evs.append(row)
    rng.shuffle(evs)
    evs.sort(key=lambda r: per_group[r["group"]])
    out, used = [], set()
    for ev in evs:
        if len(out) >= n:
            break
        mine = set(ev["groups"])
        skip = [i for i, c in enumerate(claims) if set(c["groups"]) & mine]
        ranked = index.rank(ev["evidence"], exclude=skip)
        band = [(rank, i) for rank, i in enumerate(ranked[9:60], start=10) if i not in used]
        if not band:
            continue
        rank, i = rng.choice(band)
        used.add(i)
        other = claims[i]
        per_group[ev["group"]] += 1
        lic = ev.get("lic", "") if ev.get("lic") == other.get("lic") else f"{ev.get('lic', '')}+{other.get('lic', '')}"
        out.append({"id": f"{prefix}{len(out) + 1:04d}", "evidence": ev["evidence"], "claim": other["claim"],
                    "want": "", "hint": "c", "form": NEAR_C, "domain": ev.get("domain", ""), "group": ev["group"],
                    "groups": sorted(mine | set(other["groups"])), "lang": ev.get("lang", ""), "lic": lic,
                    "split": "eval", "src_ids": list(ev.get("src_ids", [])) + list(other.get("src_ids", [])),
                    "origin": "g1-eval", "near_rank": rank})
    return out


def klue_with_claims(rows, rewrites):
    """KLUE 후보에 평서문 claim 을 붙인다 — skip · 빈 claim 은 빼고, needs_rewrite 가 남는 줄은 안 쓴다."""
    out = []
    for row in rows:
        rw = rewrites.get(row["id"])
        if not rw or rw.get("skip") or not (rw.get("claim") or "").strip():
            continue
        row = dict(row, claim=rw["claim"].strip(), needs_rewrite=False)
        if rw.get("fake_answer"):
            row["fake_answer"] = rw["fake_answer"]
        out.append(row)
    return out


def attach_rewrite(cands, rewrites):
    """학습 후보의 KLUE 줄에 평서문 claim 을 붙인다 → (붙인 klue 줄, 보고).

    다시 쓴 줄은 cand_idx(학습 후보 파일의 줄 번호)로 찾고, group · src_ids 가 맞는지 본다.
    cand_idx 가 없거나 안 맞으면 group+src_ids 로 찾는다. skip 줄은 뺀다.
    맞출 수 없는 줄 · claim 이 빈 줄 · 다시 쓴 것이 없는 klue 줄은 보고의 stop 에 센다(부르는 쪽이 멈춘다).
    """
    by_key = {(c.get("group"), tuple(c.get("src_ids") or ())): i for i, c in enumerate(cands)}
    attached, matched = {}, set()
    report = {"rewrite_rows": len(rewrites), "skip": 0, "unmatched": 0, "empty_claim": 0, "no_rewrite": 0,
              "fake_answer": 0}
    for rw in rewrites:
        key = (rw.get("group"), tuple(rw.get("src_ids") or ()))
        idx = rw.get("cand_idx")
        if not (isinstance(idx, int) and 0 <= idx < len(cands)
                and (cands[idx].get("group"), tuple(cands[idx].get("src_ids") or ())) == key):
            idx = by_key.get(key)
        if idx is None or idx in matched:
            report["unmatched"] += 1
            continue
        matched.add(idx)
        if rw.get("skip"):
            report["skip"] += 1
            continue
        claim = (rw.get("claim") or "").strip()
        if not claim:
            report["empty_claim"] += 1
            continue
        row = dict(cands[idx], claim=claim, needs_rewrite=False)
        if rw.get("fake_answer"):
            row["fake_answer"] = rw["fake_answer"]
            report["fake_answer"] += 1
        attached[idx] = row
    report["no_rewrite"] = sum(1 for i, c in enumerate(cands)
                               if c.get("domain") == "klue-mrc" and i not in matched)
    out = [attached[i] for i in sorted(attached)]
    report["attached"] = len(out)
    report["hint"] = dict(sorted(collections.Counter(r.get("hint") for r in out).items()))
    report["stop"] = report["unmatched"] + report["empty_claim"] + report["no_rewrite"]
    return out, report


def eval_source_pick(set_name, source, rows, n, hint_target, rng):
    """한 셋 · 한 원천 : 칸 몫(hint_target)대로 고르고 c 는 근접 c 를 절반 이상 섞는다 → (줄 목록, 보고)."""
    per_group = collections.Counter()
    c_want = hint_target.get("c", 0)
    c_rows = [r for r in rows if r["hint"] == "c"]
    near_want = max(math.ceil(c_want * NEAR_C_MIN), c_want - len(c_rows))
    claim_rows = None
    if source == "klue-mrc":
        claim_rows = [r for r in rows if r["form"] in KLUE_TRUE_FORMS]
    near = near_c(rows, near_want, rng, per_group, f"{set_name}-{source}-cx", claim_rows)
    c_plain = _spread_pick(c_rows, c_want - len(near), per_group, rng)
    buckets = collections.defaultdict(list)
    for row in rows:
        if row["hint"] != "c":
            buckets[(row["hint"], row["form"])].append(row)
    mix = {}
    for hint in ("a", "b", "d"):
        forms = {k: w for k, w in MIX.get(source, {}).items() if k[0] == hint}
        forms.update({k: 1e-6 for k in buckets if k[0] == hint and k not in forms})   # MIX 에 없는 꼴도 모자랄 때 쓴다
        whole = sum(forms.values()) or 1
        for key, w in forms.items():
            mix[key] = hint_target.get(hint, 0) * w / whole
    rest = n - len(near) - len(c_plain)
    scale = rest / (sum(mix.values()) or 1)          # c 가 모자라면 그 몫을 a · b · d 가 비율대로 메운다
    mix = _largest_remainder(rest, {k: v * scale for k, v in mix.items()}) if rest > 0 else {}
    quotas =_quotas(rest, {k: v / max(1, rest) for k, v in mix.items()}, {k: len(v) for k, v in buckets.items()})
    chosen = near + c_plain
    for key in sorted(quotas):
        chosen += _spread_pick(buckets.get(key, []), quotas[key], per_group, rng)
    for row in chosen:
        row["set"] = set_name
    report = {"target": n, "picked": len(chosen), "hint_target": hint_target,
              "hint": dict(sorted(collections.Counter(r["hint"] for r in chosen).items())),
              "form": dict(sorted(collections.Counter(f"{r['hint']}:{r['form']}" for r in chosen).items())),
              "groups": len({r["group"] for r in chosen}), "near_c": len(near)}
    return chosen, report


def _in_set_rows(rows, groups):
    return [r for r in rows if r.get("groups") and set(r["groups"]) <= groups]


def ruler4_ood_excludes(gold_rows, input_rows, raw_root):
    """자 ④ 에 쓴 WiCE · SufficientFacts 쌍 → (ood 후보 src_id 집합, 주장 열쇠 집합, 못 푼 ref 수).

    SF ref `이름:줄:…` → `sf-이름-줄`. WiCE ref `subclaim/갈래:줄` → 원자료 subclaim 줄의 meta id → claim 수준 id
    (끝 `-N` 뗌) → claim 파일 줄 번호 → `wice-갈래-줄`. 원자료가 없으면 주장 글로만 거른다.
    """
    ids, claims, unresolved = set(), set(), 0
    by_q = {r["q"]: r for r in input_rows}
    wice_refs = collections.defaultdict(set)
    for g in gold_rows:
        if g.get("src") == "audit":
            continue
        row = by_q.get(g["q"])
        if row:
            claims.add(norm(row.get("claim")))
        ref = g.get("ref", "")
        if g.get("src") == "sufficientfacts":
            name, line_no = ref.split(":")[:2]
            ids.add(f"sf-{name}-{line_no}")
        elif g.get("src") == "wice" and ref.startswith("subclaim/"):
            split, line_no = ref[len("subclaim/"):].split(":")
            wice_refs[split].add(int(line_no))
    base = os.path.join(raw_root or "", "wice", "wice", "data", "entailment_retrieval")
    for split, wanted in wice_refs.items():
        sub_path, claim_path = (os.path.join(base, kind, f"{split}.jsonl") for kind in ("subclaim", "claim"))
        if not raw_root or not os.path.exists(sub_path) or not os.path.exists(claim_path):
            unresolved += len(wanted)
            continue
        parents = set()
        with open(sub_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no in wanted:
                    parents.add(str(json.loads(line)["meta"]["id"]).rsplit("-", 1)[0])
        with open(claim_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if str(json.loads(line)["meta"]["id"]) in parents:
                    ids.add(f"wice-{split}-{line_no}")
        unresolved += len(wanted) - len(parents)
    return ids, claims, unresolved


def ood_pick(rows, size, rng, exclude_ids=frozenset(), exclude_claims=frozenset()):
    """ood 묶음에서 size 줄 — 세 원천 고르게 · 자 ④ 쌍 · 목록 근거 뺌 · 묶음당 OOD_GROUP_CAP. 원래 라벨은 지운다."""
    counts = collections.Counter()
    by_kind = collections.defaultdict(list)
    for row in rows:
        if set(row.get("src_ids", [])) & exclude_ids or norm(row.get("claim")) in exclude_claims:
            counts["ruler4_excluded"] += 1
            continue
        if list_like(row.get("evidence")):
            counts["list_dropped"] += 1
            continue
        by_kind[row["form"].split("-")[0]].append(row)
    kinds = [k for k in OOD_KINDS if by_kind.get(k)]
    quota = _largest_remainder(size, {k: 1 for k in kinds})
    picked = {}
    per_group = collections.Counter()
    for kind in kinds:
        usable = [r for r in by_kind[kind]]
        got = _spread_pick(usable, len(usable), collections.Counter(), rng)
        picked[kind] = []
        for row in got:
            if len(picked[kind]) >= quota[kind]:
                break
            if per_group[row["group"]] >= OOD_GROUP_CAP:
                continue
            per_group[row["group"]] += 1
            picked[kind].append(row)
    short = size - sum(len(v) for v in picked.values())
    for kind in kinds:                                     # 모자란 몫은 남는 원천이 채운다
        for row in by_kind[kind]:
            if short <= 0:
                break
            if row in picked[kind] or per_group[row["group"]] >= OOD_GROUP_CAP:
                continue
            per_group[row["group"]] += 1
            picked[kind].append(row)
            short -= 1
    out = []
    for kind in kinds:
        for row in picked[kind]:
            clean = {k: v for k, v in row.items() if k not in ("source_label",)}
            clean.update({"hint": "", "want": "", "form": kind, "set": "ood-g1"})
            out.append(clean)
    report = dict(counts, picked=len(out), by_kind={k: len(v) for k, v in picked.items()},
                  groups=len({r["group"] for r in out}))
    return out, report


def eval_pool(build, reserve_doc, seed=SEED, factor=EVAL_FACTOR, ood_excludes=(frozenset(), frozenset())):
    """예약 묶음 → {셋: 후보 줄}, 보고. 셋마다 원천 몫 = reserve.json 의 pairs × factor."""
    rng = random.Random(seed)
    loaded = {s: load_source(build, s)[0] or [] for s in EVAL_SOURCES}
    rewrites = {}
    rw_path = os.path.join(build, "klue-mrc", "rewrite-reserved.jsonl")
    if os.path.exists(rw_path):
        rewrites = {r["id"]: r for r in read_jsonl(rw_path)}
    sets, report = {}, {"seed": seed, "factor": factor, "sets": {}}
    taken = set()                                   # 앞 셋이 쓴 (근거, 주장) — 셋 사이 같은 글을 막는다
    for set_name in SET_SIZES:
        per = reserve_doc["sets"].get(set_name, {})
        sizes, pools, rep = {}, {}, {}
        for source in EVAL_SOURCES:
            if source not in per or not loaded[source]:
                continue
            rows = _in_set_rows(loaded[source], set(per[source]["groups"]))
            before = len(rows)
            if source == "klue-mrc":
                rows = klue_with_claims(rows, rewrites)
            rows = [r for r in rows if not r.get("needs_rewrite") or r.get("needs_rewrite") == "False"]
            kept = [r for r in rows if not list_like(r.get("evidence"))]
            dup = [r for r in kept if pair_key(r) in taken]     # 틀 글이 같은 다른 묶음(KEP 틀 등) — 앞 셋이 먼저 가져갔다
            kept = [r for r in kept if pair_key(r) not in taken]
            rep[source] = {"in_set": before, "with_claim": len(rows), "list_dropped": len(rows) - len(kept) - len(dup),
                           "dup_dropped": len(dup)}
            pools[source] = kept
            sizes[source] = math.ceil(round(per[source]["pairs"] * factor, 6))   # 210 × 1.6 이 336.00…01 로 올라가지 않게
        allowed = {s: sorted({r["hint"] for r in pools[s]} | {"c"}) for s in pools}
        targets = eval_hint_targets(set_name, sizes, allowed)
        chosen = []
        for source in pools:
            got, info = eval_source_pick(set_name, source, pools[source], sizes[source], targets[source], rng)
            rep[source].update(info)
            chosen.extend(got)
        for row in chosen:
            key = pair_key(row)
            taken.update((key, (key[1], key[0])))
        rep["total"] = len(chosen)
        rep["hint_total"] = dict(sorted(collections.Counter(r["hint"] for r in chosen).items()))
        sets[set_name] = chosen
        report["sets"][set_name] = rep
    ood_rows, _ = load_source(build, "ood")
    if ood_rows and "ood-g1" in reserve_doc["sets"]:
        ood_groups = set(reserve_doc["sets"]["ood-g1"]["ood"]["groups"])
        rows = _in_set_rows(ood_rows, ood_groups)
        rows = [r for r in rows if pair_key(r) not in taken]
        got, info = ood_pick(rows, math.ceil(round(OOD_SIZE * factor, 6)), rng, *ood_excludes)
        sets["ood-g1"] = got
        report["sets"]["ood-g1"] = info
    return sets, report


def eval_cross(eval_sets, train_rows=(), old_keys=frozenset(), old_ids=frozenset(), skip_relabel_old=False):
    """평가 셋 사이 · 평가 × 학습 후보 · 평가 × 옛 셋 겹침 수 (묶음 · (근거, 주장) 양쪽 차례).

    skip_relabel_old : dev-r2 다시 붙이기 줄(relabel)은 원래 옛 셋 쌍이라 옛 셋 겹침(글 · 기억 id)에서만 뺀다.
    """
    owner_group, owner_key = {}, {}
    counts = collections.Counter()
    for set_name, rows in eval_sets.items():
        for row in rows:
            for gid in row.get("groups") or [row.get("group")]:
                if owner_group.setdefault(gid, set_name) != set_name:
                    counts["eval_group_across_sets"] += 1
            key = pair_key(row)
            for k in (key, (key[1], key[0])):
                if owner_key.setdefault(k, set_name) != set_name:
                    counts["eval_text_across_sets"] += 1
                    break
            if skip_relabel_old and row.get("relabel"):
                continue
            if key in old_keys:
                counts["eval_text_old"] += 1
            if old_ids and set(_MEM_ID.findall(json.dumps(row.get("src_ids", [])))) & set(old_ids):
                counts["eval_mem_id_old"] += 1
    for row in train_rows:
        if set(row.get("groups") or [row.get("group")]) & set(owner_group):
            counts["eval_vs_train_group"] += 1
        if pair_key(row) in owner_key:
            counts["eval_vs_train_text"] += 1
    out = {k: counts.get(k, 0) for k in ("eval_group_across_sets", "eval_text_across_sets", "eval_vs_train_group",
                                         "eval_vs_train_text", "eval_text_old")}
    if old_ids:
        out["eval_mem_id_old"] = counts.get("eval_mem_id_old", 0)
    return out


def load_eval_dir(folder):
    sets = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith("-cands.jsonl"):
            sets[name[:-len("-cands.jsonl")]] = read_jsonl(os.path.join(folder, name))
    return sets


def load_eval_dirs(folders):
    """평가 폴더 여럿 → {셋: 줄}. 둘 이상이면 셋 이름 앞에 폴더 이름을 붙인다(공개 dev-cal 과 아카이브 dev-cal 을 따로 셈)."""
    if len(folders) == 1:
        return load_eval_dir(folders[0])
    sets = {}
    for folder in folders:
        tag = os.path.basename(os.path.normpath(folder))
        sets.update({f"{tag}/{name}": rows for name, rows in load_eval_dir(folder).items()})
    return sets


# ---------------------------------------------------------------- eval-pool --archive (아카이브 · mem 평가 후보, 소단계 45-마)
#
# reserve-archive.json 의 셋별 row_ids 안에서만 고른다. dev-r2 다시 붙이기 줄(relabel)은 전부 그대로 넣고 한 줄을
# 한 쌍으로 센다(1배). 나머지 몫(pairs − relabel 수)은 새 줄을 EVAL_FACTOR 배 고른다. 칸 몫은 셋 비율(설계 2-4)을
# 도메인 전체에 걸고 relabel 이 이미 채운 칸을 뺀 나머지로 맞춘다. b 후보가 모자라면 그 몫을 d · c 에 반씩 넘긴다.
# 근접 c 는 같은 셋 · 같은 도메인의 새 줄끼리(relabel 줄은 옛 기억 id 라 안 섞는다).

ARCH_EVAL_DOMAINS = ("mem", "archive")


def _first_of_each_pair(rows, before=()):
    """같은 (근거, 주장)(양쪽 차례)이 여러 줄이면 첫 줄만 — before 에 이미 있는 쌍도 뺀다 → (줄 목록, 뺀 수)."""
    seen = set()
    for row in before:
        key = pair_key(row)
        seen.update((key, (key[1], key[0])))
    out = []
    for row in rows:
        key = pair_key(row)
        if key in seen:
            continue
        seen.update((key, (key[1], key[0])))
        out.append(row)
    return out, len(rows) - len(out)


def arch_hint_target(set_name, n_total, relabel_rows, fresh_n, avail):
    """도메인 전체 n_total 에 셋 비율을 걸고 relabel 칸을 뺀 나머지를 fresh_n 으로 맞춘다 → (칸 몫, b 를 넘긴 수)."""
    ratio = EVAL_HINT[set_name]
    goal = _largest_remainder(n_total, ratio)
    have = collections.Counter(r["hint"] for r in relabel_rows)
    rest = {h: max(0, goal[h] - have.get(h, 0)) for h in ratio}
    target = _largest_remainder(fresh_n, rest if sum(rest.values()) else ratio) if fresh_n > 0 else {h: 0 for h in ratio}
    moved = max(0, target.get("b", 0) - avail.get("b", 0))
    if moved:
        target["b"] -= moved
        target["d"] = target.get("d", 0) + (moved + 1) // 2
        target["c"] = target.get("c", 0) + moved // 2
    return target, moved


def eval_pool_archive(build, arch_doc, seed=SEED, factor=EVAL_FACTOR, taken_keys=()):
    """아카이브 · mem 예약 → {셋: 후보 줄}, 보고. taken_keys 는 이미 쓴 (근거, 주장) — 공개 평가 후보."""
    rng = random.Random(seed)
    loaded = {d: _arch_rows(build, d)[0] for d in ARCH_EVAL_DOMAINS}
    sets, report = {}, {"seed": seed, "factor": factor, "sets": {}}
    taken = set(taken_keys)
    for set_name in SET_SIZES:
        per = arch_doc["sets"].get(set_name, {})
        chosen, rep = [], {}
        for domain in ARCH_EVAL_DOMAINS:
            if domain not in per:
                continue
            ids = set(per[domain]["row_ids"])
            rows = [r for r in loaded[domain] if r["id"] in ids]
            relabel = [dict(r, set=set_name) for r in rows if r.get("relabel")]
            fresh = [r for r in rows if not r.get("relabel")]
            kept = [r for r in fresh if not list_like(r.get("evidence"))]
            dup = [r for r in kept if pair_key(r) in taken]
            kept = [r for r in kept if pair_key(r) not in taken]
            kept, twins = _first_of_each_pair(kept)
            fresh_pairs = max(0, per[domain]["pairs"] - len(relabel))
            n = math.ceil(round(fresh_pairs * factor, 6))
            avail = collections.Counter(r["hint"] for r in kept)
            target, moved = arch_hint_target(set_name, len(relabel) + n, relabel, n, avail)
            got, info = eval_source_pick(set_name, domain, kept, n, target, rng) if n else ([], {"picked": 0})
            got, near_twins = _first_of_each_pair(got, relabel)   # 근접 c 가 원래 줄과 같은 쌍을 만들면 뺀다
            chosen += relabel + got
            info.update({"reserved_rows": len(rows), "relabel": len(relabel), "fresh_target": n,
                         "list_dropped": len(fresh) - len(kept) - len(dup) - twins, "dup_dropped": len(dup),
                         "twin_dropped": twins + near_twins, "picked": len(got),
                         "b_moved_to_dc": moved,
                         "relabel_hint": dict(sorted(collections.Counter(r["hint"] for r in relabel).items())),
                         "hint_with_relabel": dict(sorted(collections.Counter(r["hint"] for r in relabel + got).items()))})
            rep[domain] = info
        for row in chosen:
            key = pair_key(row)
            taken.update((key, (key[1], key[0])))
        rep["total"] = len(chosen)
        rep["hint_total"] = dict(sorted(collections.Counter(r["hint"] for r in chosen).items()))
        sets[set_name] = chosen
        report["sets"][set_name] = rep
    return sets, report


def traps_without(traps, rows):
    """함정 중 (근거, 주장)이 rows 에 든 것을 뺀다 — dev-r2 함정이 같은 셋의 relabel 실제 쌍으로 또 나오지 않게."""
    keys = set()
    for row in rows:
        key = pair_key(row)
        keys.update((key, (key[1], key[0])))
    return [t for t in traps if pair_key(t) not in keys]


def eval_pool_archive_main(args):
    reserve_path = args.reserve or os.path.join(args.build, "reserve-archive.json")
    with open(reserve_path, encoding="utf-8") as f:
        arch_doc = json.load(f)
    public = args.public_dir or os.path.join(args.build, "eval-public")
    taken = set()
    if os.path.isdir(public):
        for rows in load_eval_dir(public).values():
            for row in rows:
                key = pair_key(row)
                taken.update((key, (key[1], key[0])))
    sets, report = eval_pool_archive(args.build, arch_doc, args.seed, args.factor, taken)
    out = args.out or os.path.join(args.build, "eval-archive")
    os.makedirs(out, exist_ok=True)
    traps = read_jsonl(args.traps) if args.traps else []
    report["traps"] = {}
    for set_name, rows in sets.items():
        with open(os.path.join(out, f"{set_name}-cands.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if traps:
            kept = traps_without(traps, rows)
            with open(os.path.join(out, f"{set_name}-traps.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for row in kept:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            report["traps"][set_name] = {"kept": len(kept), "dropped": len(traps) - len(kept),
                                         "gold": dict(sorted(collections.Counter(t["gold"] for t in kept).items()))}
    report["public_keys"] = len(taken) // 2
    report["cross"] = eval_cross(sets, skip_relabel_old=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({s: (v["total"], v["hint_total"]) for s, v in report["sets"].items()}
                     | {"traps": report["traps"], "cross": report["cross"]}, ensure_ascii=False))
    return 1 if sum(report["cross"].values()) else 0


# ---------------------------------------------------------------- pool --archive (아카이브 · mem 학습 후보, 소단계 48)
#
# 세 원천(git · docs · mem)에서 학습 후보를 칸 몫(설계 4-1 : d 900 · a 450 · b 250 · c 400)대로 고른다.
# 빼는 것 : reserve-archive.json 의 예약(row_ids · 묶음 · 출처 id) · 기억 금지 목록(얼린 셋 · blind v4 본문에서 뽑은
# 기억 id) · 공개 학습 후보와 같은 (근거, 주장) · 목록 근거 · 원천 안 같은 쌍 · dev-r2 다시 붙이기 줄(평가 몫).
# 칸 안에서는 꼴마다 고르게 나눈다(물 채우기) — 꼴이 곧 원천이라 출처 말투가 한 칸에 몰리지 않는다.
# 모자란 칸 몫은 d · c 에 반씩 넘긴다(eval-pool --archive 의 b_moved_to_dc 와 같은 규칙).

ARCH_TRAIN_SOURCES = ("archive/git", "archive/docs", "mem")
ARCH_TRAIN_HINT = {"d": 900, "a": 450, "b": 250, "c": 400}
ARCH_TRAIN_SEED = 20261048
ARCH_TRAIN_GROUP_CAP = 8
FORBID_SETS = ("dev-cal", "dev-cmp", "blind-g1")


def forbidden_mem_ids(paths):
    """파일 본문에서 기억 id 를 직접 뽑는다(설계 4-1 「R3 4절 함정」 — 칸 이름에 기대지 않는다). 없는 파일은 멈춘다."""
    ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            ids.update(_MEM_ID.findall(f.read()))
    return ids


def _water_fill(total, avail):
    """total 을 칸마다 고르게 나누되 avail 을 넘지 않게 — 남는 몫은 여유 있는 칸이 고르게 나눠 진다."""
    out = {k: 0 for k in avail}
    left = total
    open_keys = sorted(k for k in avail if avail[k] > 0)
    while left > 0 and open_keys:
        share = _largest_remainder(left, {k: 1 for k in open_keys})
        for k in open_keys:
            out[k] += min(share[k], avail[k] - out[k])
        left = total - sum(out.values())
        open_keys = [k for k in open_keys if out[k] < avail[k]]
    return out


def _capped_pick(rows, n, per_group, cap, rng):
    """덜 쓴 묶음부터 n 줄, 한 묶음은 cap 줄까지 (per_group 은 칸 · 꼴을 넘어 같이 쓴다)."""
    pool_rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(pool_rows)
    picked = []
    while pool_rows and len(picked) < n:
        best = min(range(len(pool_rows)), key=lambda i: per_group[pool_rows[i]["group"]])
        row = pool_rows.pop(best)
        if per_group[row["group"]] >= cap:
            break
        per_group[row["group"]] += 1
        picked.append(row)
    return picked


def arch_train_targets(hint_goal, avail_by_hint):
    """칸 몫 → (칸 몫, 넘긴 표). 모자란 칸 몫은 d · c 에 반씩(d 먼저), 그래도 모자라면 남는 칸에."""
    target = {h: min(n, avail_by_hint.get(h, 0)) for h, n in hint_goal.items()}
    moved = {h: hint_goal[h] - target[h] for h in hint_goal if hint_goal[h] > target[h]}
    short = sum(moved.values())
    for i in range(short):
        order = ("d", "c") if i % 2 == 0 else ("c", "d")
        order += tuple(h for h in hint_goal if h not in order)
        for h in order:
            if target[h] < avail_by_hint.get(h, 0):
                target[h] += 1
                break
    return target, moved


def pool_archive(build, arch_doc, seed=ARCH_TRAIN_SEED, hint_goal=None, forbid_ids=frozenset(),
                 taken_keys=frozenset(), group_cap=ARCH_TRAIN_GROUP_CAP, eval_keys=frozenset(), used=None):
    """아카이브 · mem 예약 밖 → (학습 후보 줄, 보고).

    used 가 있으면({"groups", "src_ids"}) 예약 전체(row_ids · 넓힌 묶음) 대신 실제로 평가에 쓴 묶음 · 출처 id 만 막는다.

    taken_keys 는 공개 학습 후보, eval_keys 는 평가 후보 · 얼린 셋의 (근거, 주장) 양쪽 차례 — 묶음이 달라도
    글이 같은 쌍(틀 글을 나눠 쓰는 문서)이 있어 글로도 뺀다.
    """
    rng = random.Random(seed)
    hint_goal = dict(hint_goal or ARCH_TRAIN_HINT)
    if used is None:
        own = {rid for per in arch_doc["sets"].values() for entry in per.values() for rid in entry["row_ids"]}
        banned, srcs = set(arch_doc["reserved"]), set(arch_doc["reserved_src_ids"])
        tag = "reserved"
    else:
        own, banned, srcs, tag = set(), set(used["groups"]), set(used["src_ids"]), "used"
    drops = collections.Counter()
    free, by_source_in = [], {}
    for source in ARCH_TRAIN_SOURCES:
        rows, _ = load_source(build, source)
        by_source_in[source] = len(rows or [])
        for row in rows or []:
            ids_text = json.dumps([row.get("groups"), row.get("src_ids"), row.get("evidence"), row.get("claim")],
                                  ensure_ascii=False)
            if row.get("relabel") or row.get("split") not in ("train", "any"):
                drops["relabel_or_dev"] += 1
            elif row["id"] in own:
                drops["reserved_row"] += 1
            elif set(row.get("groups") or [row["group"]]) & banned:
                drops[f"{tag}_group"] += 1
            elif set(row.get("src_ids") or []) & srcs:
                drops[f"{tag}_src"] += 1
            elif set(_MEM_ID.findall(ids_text)) & forbid_ids:
                drops["forbidden_mem_id"] += 1
            elif pair_key(row) in eval_keys:
                drops["eval_pair"] += 1
            elif pair_key(row) in taken_keys:
                drops["public_train_pair"] += 1
            elif not (row.get("evidence") or "").strip() or not (row.get("claim") or "").strip():
                drops["empty"] += 1
            elif list_like(row.get("evidence")):
                drops["list_like"] += 1
            else:
                free.append(dict(row, source=row.get("source") or source.split("/")[-1]))
    free, twins = _first_of_each_pair(free)
    drops["twin"] = twins
    buckets = collections.defaultdict(list)
    for row in free:
        buckets[(row["hint"], row["form"])].append(row)
    avail_hint = collections.Counter(row["hint"] for row in free)
    target, moved = arch_train_targets(hint_goal, avail_hint)
    per_group, chosen = collections.Counter(), []
    for hint in sorted(target):
        forms = {key: len(rows) for key, rows in buckets.items() if key[0] == hint}
        quotas = _water_fill(target[hint], forms)
        got = []
        for key in sorted(quotas):
            got += _capped_pick(buckets[key], quotas[key], per_group, group_cap, rng)
        if len(got) < target[hint]:                     # 묶음 상한에 걸린 꼴 몫은 같은 칸 다른 꼴이 채운다
            ids = {r["id"] for r in got}
            rest = [r for key in sorted(quotas) for r in buckets[key] if r["id"] not in ids]
            got += _capped_pick(rest, target[hint] - len(got), per_group, group_cap, rng)
        chosen += got
    d_rows = [r for r in chosen if r["hint"] == "d"]
    by_source = collections.defaultdict(collections.Counter)
    for row in chosen:
        by_source[row["source"]][row["hint"]] += 1
    report = {
        "seed": seed, "goal": hint_goal, "target": target, "moved_to_dc": moved,
        "in": by_source_in, "dropped": dict(sorted(drops.items())), "free": len(free),
        "free_hint_form": {f"{h}:{f}": len(v) for (h, f), v in sorted(buckets.items())},
        "total": len(chosen),
        "hint_total": dict(sorted(collections.Counter(r["hint"] for r in chosen).items())),
        "hint_form": dict(sorted(collections.Counter(f"{r['hint']}:{r['form']}" for r in chosen).items())),
        "by_source": {s: dict(sorted(c.items())) for s, c in sorted(by_source.items())},
        "source_max_hint_share": {s: round(max(c.values()) / sum(c.values()), 3) for s, c in sorted(by_source.items())},
        "d_form_share": {f: round(n / max(1, len(d_rows)), 3)
                         for f, n in sorted(collections.Counter(r["form"] for r in d_rows).items())},
        "groups": len({r["group"] for r in chosen}), "group_cap": group_cap,
        "lang": dict(sorted(collections.Counter(r.get("lang", "") for r in chosen).items())),
    }
    return chosen, report


def used_eval_rows(build, eval_archive, frozen_paths):
    """실제로 평가에 쓴 쌍 → (줄 목록, 글만 있는 쌍 목록).

    얼린 셋 줄 전부 + eval-archive 의 셋마다 <셋>-cands.jsonl · <셋>-traps.jsonl · 묶음 폴더 map 의 id(원천 후보에서 찾음)
    · batches 의 쌍 글(함정 · 버린 쌍 포함). 묶음 폴더 map 에 원천에서도 후보 파일에서도 못 찾는 id 가 있으면 센다.
    """
    rows = [r for path in frozen_paths for r in read_jsonl(path)]
    texts = []
    by_id = {}
    for source in ARCH_TRAIN_SOURCES:
        for row in load_source(build, source)[0] or []:
            by_id[row["id"]] = row
    unknown = 0
    for name in sorted(os.listdir(eval_archive)):
        folder = os.path.join(eval_archive, name)
        if not os.path.isdir(os.path.join(folder, "map")):
            continue
        cands = read_jsonl(os.path.join(eval_archive, f"{name}-cands.jsonl"))
        traps_path = os.path.join(eval_archive, f"{name}-traps.jsonl")
        texts += read_jsonl(traps_path) if os.path.exists(traps_path) else []
        rows += cands
        known = {r["id"] for r in cands}
        for entry in read_jsonl(os.path.join(folder, "map", "map.jsonl")):
            if entry.get("trap") or entry["id"] in known:
                continue
            if entry["id"] in by_id:
                rows.append(by_id[entry["id"]])
            else:
                unknown += 1
        batches = os.path.join(folder, "batches")
        for fname in sorted(os.listdir(batches)) if os.path.isdir(batches) else []:
            if fname.endswith(".jsonl"):
                texts += read_jsonl(os.path.join(batches, fname))
    return rows, texts, unknown


def used_blocks(rows):
    """평가 줄 → 막을 {groups, src_ids, mem_ids}."""
    groups, srcs, mem_ids = set(), set(), set()
    for row in rows:
        groups.update(g for g in (row.get("groups") or [row.get("group")]) if g)
        srcs.update(row.get("src_ids") or [])
        mem_ids.update(_MEM_ID.findall(json.dumps([row.get("group"), row.get("groups"), row.get("src_ids")])))
    return {"groups": groups, "src_ids": srcs, "mem_ids": mem_ids}


def pool_archive_main(args):
    with open(args.reserve or os.path.join(args.build, "reserve-archive.json"), encoding="utf-8") as f:
        arch_doc = json.load(f)
    old_root = args.old_root or os.path.join(os.path.expanduser("~"), ".judgemodel")
    forbid_paths = args.forbid or ([os.path.join(old_root, "sets", "g1", f"{s}.jsonl") for s in FORBID_SETS]
                                   + [os.path.join(old_root, OLD_SETS["blind-v4"])])
    public = args.public_cands or [os.path.join(args.build, name) for name in
                                   ("train-cands.jsonl", "train-cands-klue.filtered.jsonl",
                                    "train-cands-jira.filtered.jsonl")]
    missing = [p for p in forbid_paths + public if not os.path.exists(p)]
    if missing:
        print(json.dumps({"stop": "금지 목록 · 공개 후보 파일이 없다", "missing": missing}, ensure_ascii=False),
              file=sys.stderr)
        return 2
    forbid = forbidden_mem_ids(forbid_paths)
    taken = set()
    for path in public:
        for row in read_jsonl(path):
            key = pair_key(row)
            taken.update((key, (key[1], key[0])))
    eval_dirs = args.eval_dir or [os.path.join(args.build, d) for d in ("eval-public", "eval-archive")]
    eval_keys = set()
    eval_rows = [r for path in forbid_paths for r in read_jsonl(path)]
    eval_rows += [r for rows in load_eval_dirs([d for d in eval_dirs if os.path.isdir(d)]).values() for r in rows]
    for row in eval_rows:
        key = pair_key(row)
        eval_keys.update((key, (key[1], key[0])))
    used, used_info = None, {}
    if args.reserve_used_only:
        eval_archive = os.path.join(args.build, "eval-archive")
        frozen = [os.path.join(old_root, "sets", "g1", f"{s}.jsonl") for s in FORBID_SETS]
        rows, texts, unknown = used_eval_rows(args.build, eval_archive, frozen)
        if unknown:
            print(json.dumps({"stop": "묶음 폴더 map 의 id 를 원천에서 못 찾았다", "unknown": unknown}), file=sys.stderr)
            return 2
        used = used_blocks(rows)
        forbid |= used["mem_ids"]
        for row in rows + texts:
            key = pair_key(row)
            eval_keys.update((key, (key[1], key[0])))
        used_info = {"rows": len(rows), "texts": len(texts), "groups": len(used["groups"]),
                     "src_ids": len(used["src_ids"]), "mem_ids": len(used["mem_ids"])}
    seed = args.seed if args.seed != SEED else ARCH_TRAIN_SEED
    chosen, report = pool_archive(args.build, arch_doc, seed, forbid_ids=forbid, taken_keys=taken,
                                  group_cap=args.group_cap, eval_keys=eval_keys, used=used)
    if used is not None:
        report["used_only"] = used_info
    report["forbidden_ids"] = len(forbid)
    report["forbid_files"] = [os.path.basename(p) for p in forbid_paths]
    report["public_keys"] = len(taken) // 2
    report["eval_keys"] = len(eval_keys) // 2
    out = args.out or os.path.join(args.build, "train-cands-archive.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out + ".summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def attach_rewrite_main(args):
    cands = read_jsonl(args.cands)
    rewrites = [row for path in args.rewrite for row in read_jsonl(path)]
    rows, report = attach_rewrite(cands, rewrites)
    print(json.dumps(report, ensure_ascii=False))
    if report["stop"]:
        print("멈춤 : 맞출 수 없는 줄 · 빈 claim · 다시 쓴 것이 없는 klue 줄이 있다 — 파일을 안 쓴다", file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


# ---------------------------------------------------------------- reserve-g2 · check-g2 (v1.2 B 판 b 자, 설계 v1.2-B판 3-1 · 5-1 · 7절)
#
# b 후보만 다룬다. 평가 몫(dev-cmp-g2 · dev-cal-g2 · blind-g2)을 묶음째 먼저 예약하고, 남은 묶음을 학습 층에 준다.
# 원천 몫(b 목표)을 남김률로 나눠 후보 수를 정한다. 한국어 · KLUE 가 모자라면 그 셋의 VitaminC 몫으로 한 번 넘긴다(shift).

G2_SETS = ("dev-cmp-g2", "dev-cal-g2", "blind-g2")      # 차례 : 관문에 쓰는 dev-cmp 먼저
G2_B_TARGET = {"dev-cmp-g2": {"ko": 60, "klue": 20, "vitaminc": 100},
               "dev-cal-g2": {"ko": 15, "klue": 5, "vitaminc": 25},
               "blind-g2": {"ko": 85, "klue": 35, "vitaminc": 130}}
G2_DOMAIN_SOURCES = {"ko": ("ko-supersede", "ko-revision"), "klue": ("klue",), "vitaminc": ("vitaminc-heldout",)}
G2_KEEP = {"ko-supersede": 0.30, "ko-revision": 0.16, "klue": 0.22, "vitaminc-heldout": 0.52}
G2_FLOOR = 1.6                   # 후보는 b 목표의 1.6배 이상
G2_GROUP_CAP = {"ko-supersede": 4, "ko-revision": 12, "klue": 3, "vitaminc-heldout": 4}   # 한국어 고침은 문서가 적어 넉넉히
G2_TRAIN_SOURCES = ("vitaminc-train", "ko-revision", "ko-supersede")
G2_SEED = 20261201
G1_SET_FILES = ("dev-cal", "dev-cmp", "blind-g1", "ood-g1")
G1_TRAIN_FILES = ("train-g1-II", "train-g1-arch", "train-g1-syn")


def _strip_tags():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
    from data_prep import strip_tags
    return strip_tags


def use1_key(row, strip_tags):
    """use1 쌍 열쇠 (protocol-use1 `pair_key` 와 같은 셈)."""
    text = strip_tags(row.get("evidence") or "") + "\x1f" + strip_tags(row.get("claim") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_keys_ids_groups(paths):
    keys, ids, groups = set(), set(), set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ids.update(_MEM_ID.findall(line))
                row = json.loads(line)
                _walk_pairs(row, keys, ids)
                groups.update(row.get("groups") or ([row["group"]] if row.get("group") else []))
    return keys, ids, groups


class G2Forbid:
    """g2 후보가 피해야 할 것 — 옛 셋 글 · 기억 id · 묶음 · use1 얼린 쌍 · 잣대 기억."""

    def __init__(self, old_root, g1_build, use1_root, ruler_ids_path):
        self.missing = []
        old_keys, old_ids, self.missing = load_old(old_root)
        sets_dir = os.path.join(old_root, "sets", "g1")
        g1_eval = [os.path.join(sets_dir, n + ".jsonl") for n in G1_SET_FILES]
        g1_train = [os.path.join(sets_dir, n + ".jsonl") for n in G1_TRAIN_FILES]
        for path in g1_eval + g1_train:
            if not os.path.exists(path):
                self.missing.append(os.path.basename(path))
        g1_eval = [p for p in g1_eval if os.path.exists(p)]
        g1_train = [p for p in g1_train if os.path.exists(p)]
        ek, ei, eg = _file_keys_ids_groups(g1_eval)
        tk, ti, tg = _file_keys_ids_groups(g1_train)
        self.text = old_keys | ek | tk                       # 어느 몫이든 옛 셋과 같은 (근거, 주장) 금지
        self.eval_ids = old_ids | ei | ti                    # 평가 몫 : 옛 셋 어디에든 쓴 기억 금지
        self.train_ids = old_ids | ei | ti                   # 학습 층도 옛 셋 어디에든 쓴 기억 금지 (check 의 mem_id_old 0)
        # g1 예약(reserve · reserve-archive)은 쓰지 않은 묶음까지 넓게 잡혀 있어(문서 묶음 거의 전부) 실제로 셋에 든 묶음만 막는다
        # — g1 아카이브 학습 후보의 --reserve-used-only 와 같은 생각이다.
        # 옛 학습과는 묶음(문서) 대신 문장으로 가른다 : 한국어 문서 거의 모두가 train-g1-arch 묶음이라 문서로 막으면
        # 한국어 평가 b 가 0 이 된다. 대신 평가 줄의 근거 · 주장 어느 한쪽이라도 옛 셋 문장과 같으면 뺀다(side_old).
        self.eval_groups = eg                                # 평가 몫 : g1 평가 셋에 든 묶음 금지
        self.train_groups = eg                               # 학습 층 : g1 평가 셋에 든 묶음 금지
        self.sides = {side for key in self.text for side in key if side}
        self.ruler_ids = set()
        if ruler_ids_path and os.path.exists(ruler_ids_path):
            with open(ruler_ids_path, encoding="utf-8") as f:
                self.ruler_ids = set(_MEM_ID.findall(f.read()))
        elif ruler_ids_path:
            self.missing.append(os.path.basename(ruler_ids_path))
        self.strip_tags = _strip_tags()
        self.use1_keys, self.use1_texts = set(), set()
        exclude = os.path.join(use1_root, "EXCLUDE.sha256list")
        if os.path.exists(exclude):
            with open(exclude, encoding="utf-8") as f:
                self.use1_keys = {line.strip() for line in f if line.strip()}
        else:
            self.missing.append("EXCLUDE.sha256list")
        for path in sorted(glob.glob(os.path.join(use1_root, "*", "set-use1*.jsonl"))):
            for row in read_jsonl(path):
                for side in ("evidence", "claim"):
                    self.use1_texts.add(norm(self.strip_tags(row.get(side) or "")))
        self.use1_texts.discard("")
        self.g1_klue_used = set()
        for path in glob.glob(os.path.join(g1_build, "klue-mrc", "rewrite-*.jsonl")):
            self.g1_klue_used.update(r.get("id") for r in read_jsonl(path))

    def reasons(self, row, role):
        """이 줄이 role(eval · train) 몫에 못 드는 까닭 목록 (빈 목록이면 됨)."""
        out = []
        key = pair_key(row)
        if key in self.text:
            out.append("text_old")
        elif role == "eval" and {norm(row.get("evidence")), norm(row.get("claim"))} & self.sides:
            out.append("side_old")
        ids = set(_MEM_ID.findall(json.dumps(row.get("src_ids", []))))
        if ids & self.ruler_ids:
            out.append("ruler_mem")
        if ids & (self.eval_ids if role == "eval" else self.train_ids):
            out.append("mem_id_old")
        if set(row.get("groups") or [row.get("group")]) & (self.eval_groups if role == "eval" else self.train_groups):
            out.append("group_g1")
        if use1_key(row, self.strip_tags) in self.use1_keys:
            out.append("use1_pair")
        sides = {norm(self.strip_tags(row.get("evidence") or "")), norm(self.strip_tags(row.get("claim") or ""))}
        if sides & self.use1_texts:
            out.append("use1_text")
        return out


def _g2_eval_ok(source, row):
    if source == "klue":
        return row.get("form") == "unanswerable" and row.get("split") == "heldout"
    return row.get("hint") == "b" or source.startswith("ko-")


def _both_keys(row):
    """(근거, 주장) 열쇠 양쪽 차례 — 출처 id(글)와 안 섞이게 튜플로 둔다."""
    key = pair_key(row)
    return {key, (key[1], key[0])}


def _reserve_domain(doc, domain, targets, keep, shift, orders, by_group, owner, owner_src):
    """한 도메인의 b 몫을 셋 셋에 묶음째 나눈다. 차례마다 채움률(기대 b / 목표)이 가장 낮은 셋이 다음 묶음을 가져간다 —
    원천이 모자라도 셋 사이 비율이 목표 비율대로 남는다. 모자란 b 는 같은 셋의 VitaminC 몫으로 넘긴다(shift)."""
    target = {s: targets[s][domain] + (shift[s] if domain == "vitaminc" else 0) for s in G2_SETS}
    covered = {s: 0.0 for s in G2_SETS}
    count = collections.Counter()
    for set_name in G2_SETS:
        for source in G2_DOMAIN_SOURCES[domain]:
            doc["sets"].setdefault(set_name, {})[source] = {"groups": [], "row_ids": [], "candidates": 0, "b_expected": 0.0}

    def hungry():
        open_sets = [s for s in G2_SETS if target[s] and (covered[s] < target[s] or count[s] < G2_FLOOR * target[s])]
        return min(open_sets, key=lambda s: (covered[s] / target[s], G2_SETS.index(s))) if open_sets else None

    for source in G2_DOMAIN_SOURCES[domain]:
        for gid in orders[source]:
            set_name = hungry()
            if set_name is None:
                break
            if gid in owner:
                continue
            rows = [r for r in by_group[source][gid] if not set(r["groups"]) & set(owner)
                    and not set(r.get("src_ids", [])) & set(owner_src) and _both_keys(r).isdisjoint(owner_src)]
            rows = rows[:G2_GROUP_CAP[source]]
            if not rows:
                continue
            for g in {g for r in rows for g in r["groups"]} | {gid}:
                owner[g] = set_name
            owner_src.update({src: set_name for r in rows for src in r.get("src_ids", [])})
            owner_src.update({key: set_name for r in rows for key in _both_keys(r)})
            entry = doc["sets"][set_name][source]
            entry["groups"].append(gid)
            entry["row_ids"].extend(r["id"] for r in rows)
            entry["candidates"] = len(entry["row_ids"])
            entry["b_expected"] = round(entry["b_expected"] + keep[source] * len(rows), 1)
            covered[set_name] += keep[source] * len(rows)
            count[set_name] += len(rows)
    for set_name in G2_SETS:
        short = max(0, math.ceil(target[set_name] - covered[set_name] - 1e-9))
        if domain != "vitaminc" and short:
            shift[set_name] += short
            doc["shift"].setdefault(set_name, {})[domain] = short
        doc["sets"][set_name].setdefault("domains", {})[domain] = {
            "b_target": target[set_name], "b_expected": round(covered[set_name], 1), "candidates": count[set_name],
            "short": short if domain == "vitaminc" else 0}


def reserve_g2(build, forbid, seed=G2_SEED, targets=None, keep=None):
    """평가 몫 b 후보를 묶음째 예약하고 남은 묶음을 학습 층으로 → (예약 문서, 셋별 줄, 학습 층 줄)."""
    targets = targets or G2_B_TARGET
    keep = keep or G2_KEEP
    rng = random.Random(seed)
    loaded = {}
    for source in set(G2_KEEP) | set(G2_TRAIN_SOURCES):
        rows, groups = load_source(build, source)
        loaded[source] = (rows or [], groups or {})
    train_pages = set(loaded["vitaminc-train"][1])
    doc = {"seed": seed, "keep": keep, "targets": targets, "floor": G2_FLOOR, "sets": {}, "shift": {},
           "dropped_eval": {}, "dropped_train": {}, "missing": list(forbid.missing)}
    owner, owner_src = {}, {}             # 묶음 · 출처 id(커밋 · 기억 · 판 고침) → 가져간 셋
    orders = {}
    by_group = {}
    for source in G2_KEEP:
        rows, groups = loaded[source]
        dropped = collections.Counter()
        good = collections.defaultdict(list)
        for row in rows:
            if not _g2_eval_ok(source, row):
                continue
            why = forbid.reasons(row, "eval")
            if source == "klue" and row["id"] in forbid.g1_klue_used:
                why.append("klue_g1_rewrite")
            if source == "vitaminc-heldout" and set(row["groups"]) & train_pages:
                why.append("page_in_train")
            if why:
                dropped.update(why)
                continue
            good[row["group"]].append(row)
        doc["dropped_eval"][source] = dict(dropped)
        order = sorted(good)
        rng.shuffle(order)
        if source == "ko-revision":
            order.sort(key=lambda g: -sum(1 for r in good[g] if r.get("num_change")))
        for gid in order:
            good[gid].sort(key=lambda r: (not r.get("num_change"), r["id"]))
        orders[source], by_group[source] = order, good
    shift = collections.Counter()
    for domain in ("ko", "klue", "vitaminc"):
        _reserve_domain(doc, domain, targets, keep, shift, orders, by_group, owner, owner_src)
    eval_rows = {s: [] for s in G2_SETS}
    for source in G2_KEEP:
        rows = {r["id"]: r for r in loaded[source][0]}
        for set_name in G2_SETS:
            for rid in doc["sets"][set_name][source]["row_ids"]:
                eval_rows[set_name].append(dict(rows[rid], set=set_name, g2_source=source))
    train_rows, dropped_pairs = [], set()
    owned, owned_src = set(owner), set(owner_src)
    for source in G2_TRAIN_SOURCES:
        dropped = collections.Counter()
        kept = []
        for row in loaded[source][0]:
            why = forbid.reasons(row, "train")
            if set(row["groups"]) & owned:
                why.append("group_eval_g2")
            if set(row.get("src_ids", [])) & owned_src:
                why.append("src_eval_g2")
            if pair_key(row) in owned_src:
                why.append("text_eval_g2")
            if why:
                dropped.update(why)
                if row.get("pair"):
                    dropped_pairs.add(row["pair"])
                continue
            kept.append(dict(row, g2_source=source))
        if source == "vitaminc-train":
            before = len(kept)
            kept = [r for r in kept if r.get("pair") not in dropped_pairs]
            dropped["pair_partner"] += before - len(kept)
        doc["dropped_train"][source] = dict(dropped)
        train_rows.extend(kept)
    doc["train"] = {s: {"rows": sum(1 for r in train_rows if r["g2_source"] == s),
                        "b_hint": sum(1 for r in train_rows if r["g2_source"] == s and r["hint"] == "b"),
                        "groups": len({r["group"] for r in train_rows if r["g2_source"] == s})}
                    for s in G2_TRAIN_SOURCES}
    doc["reserved"] = sorted(owner)
    return doc, eval_rows, train_rows


def check_g2(eval_rows, train_rows, forbid):
    """평가 셋 사이 · 평가 × 학습 층 · 둘 × 옛 셋(글 · 기억 id · 묶음 · use1 · 잣대) 겹침 수. 0 이어야 한다."""
    counts = collections.Counter()
    seen_group, seen_key, seen_src = {}, {}, {}
    for set_name, rows in eval_rows.items():
        for row in rows:
            key = pair_key(row)
            for gid in row["groups"]:
                if seen_group.setdefault(gid, set_name) != set_name:
                    counts["eval_group_across_sets"] += 1
            if seen_key.setdefault(key, set_name) != set_name or seen_key.setdefault((key[1], key[0]), set_name) != set_name:
                counts["eval_text_across_sets"] += 1
            for src in row.get("src_ids", []):
                if seen_src.setdefault(src, set_name) != set_name:
                    counts["eval_src_across_sets"] += 1
            for why in forbid.reasons(row, "eval"):
                counts["eval_" + why] += 1
    for row in train_rows:
        key = pair_key(row)
        if set(row["groups"]) & set(seen_group):
            counts["train_group_eval"] += 1
        if key in seen_key or (key[1], key[0]) in seen_key:
            counts["train_text_eval"] += 1
        if set(row.get("src_ids", [])) & set(seen_src):
            counts["train_src_eval"] += 1
        for why in forbid.reasons(row, "train"):
            counts["train_" + why] += 1
    names = ["eval_group_across_sets", "eval_text_across_sets", "eval_src_across_sets", "train_group_eval",
             "train_text_eval", "train_src_eval"]
    names += [f"{role}_{why}" for role in ("eval", "train")
              for why in ("text_old", "side_old", "mem_id_old", "ruler_mem", "group_g1", "use1_pair", "use1_text")]
    return {k: counts.get(k, 0) for k in names}


def _g2_forbid(args):
    root = args.old_root or os.path.join(os.path.expanduser("~"), ".judgemodel")
    return G2Forbid(root, args.g1_build or os.path.join(root, "build", "g1"),
                    args.use1 or os.path.join(root, "measure", "use1"),
                    args.ruler_ids or os.path.join(args.build, "forbid-ruler-ids.txt"))


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reserve_g2_main(args):
    forbid = _g2_forbid(args)
    if forbid.missing:
        print(json.dumps({"missing": forbid.missing}, ensure_ascii=False))
        return 2
    doc, eval_rows, train_rows = reserve_g2(args.build, forbid, args.seed)
    out = os.path.join(args.build, "reserve")
    os.makedirs(out, exist_ok=True)
    for set_name, rows in eval_rows.items():
        _write_jsonl(os.path.join(out, f"{set_name}-b-cands.jsonl"), rows)
    _write_jsonl(os.path.join(out, "train-layer-cands.jsonl"), train_rows)
    doc["check"] = check_g2(eval_rows, train_rows, forbid)
    with open(os.path.join(out, "reserve-g2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    brief = {s: per["domains"] for s, per in doc["sets"].items()}
    print(json.dumps({"sets": brief, "shift": doc["shift"], "train": doc["train"], "check": doc["check"]},
                     ensure_ascii=False, indent=1))
    return 1 if sum(doc["check"].values()) else 0


def check_g2_main(args):
    forbid = _g2_forbid(args)
    out = os.path.join(args.build, "reserve")
    eval_rows = {s: read_jsonl(os.path.join(out, f"{s}-b-cands.jsonl")) for s in G2_SETS}
    train_path = os.path.join(out, "train-layer-cands.jsonl")
    if args.train:
        train_path = args.train
    train_rows = read_jsonl(train_path)
    result = check_g2(eval_rows, train_rows, forbid)
    result["eval_rows"] = {s: len(r) for s, r in eval_rows.items()}
    result["train_rows"] = len(train_rows)
    result["old_missing"] = forbid.missing
    result["old_pairs"] = len(forbid.text) // 2
    result["use1_keys"] = len(forbid.use1_keys)
    result["ruler_ids"] = len(forbid.ruler_ids)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    bad = sum(v for k, v in result.items() if isinstance(v, int) and k.startswith(("eval_", "train_"))
              and k not in ("train_rows",))
    return 1 if bad or forbid.missing else 0


# ---------------------------------------------------------------- reserve-g3 (v1.2 K 판 평가 몫 예약, 설계 v1.2-한국어b보강판 2-3 · 3-1 · 4절 · 흐름 6)
#
# 라벨이 이미 확정된 원천(위키 · MIT · BoolQ · KLUE 판정 파일)과 VitaminC heldout 영어 후보에서 g3 셋 셋의 「새로 더할 쌍」을
# 묶음째 예약한다. g3 = g2 + 새 쌍이라 g2 셋 쌍을 이어 붙이는 건 다음 단계다. 원천 몫은 b 수로 센다(확정 원천 남김률 1.0,
# VitaminC 는 g2 남김률 .52). 모자란 b 는 G3_SHIFT 차례로 다음 원천의 같은 셋에 넘긴다. 없는 입력 파일은 몫 0 으로 보고한다.

G3_SETS = ("dev-cmp-g3", "dev-cal-g3", "blind-g3")      # 차례 : 관문에 쓰는 dev-cmp 먼저
G3_B_TARGET = {"dev-cmp-g3": {"mit": 30, "wiki": 80, "klue": 20, "boolq": 70},
               "dev-cal-g3": {"mit": 5, "wiki": 20, "klue": 10, "boolq": 15},
               "blind-g3": {"mit": 35, "wiki": 90, "klue": 35, "boolq": 90, "vitaminc": 150}}
G3_ORDER = ("mit", "wiki", "klue", "boolq", "vitaminc")   # 예약 차례 = 넘김 차례 (앞 원천의 예약은 뒤 원천 파일이 생겨도 그대로)
G3_SHIFT = {"mit": "wiki", "wiki": "boolq", "klue": "boolq"}   # 모자란 b 를 받는 원천 (설계 흐름 「원천 몫 넘김」)
G3_PRIORITY = ("mit",)                     # 양이 적어 채움률 대신 셋 차례대로 채운다 (dev-cmp-g3 먼저)
G3_REST_TO_EVAL = ("wiki", "klue", "boolq")   # 평가 몫으로 떼어 둔 파일 — b 없는 묶음(a · c · d)도 셋에 고르게 준다
G3_INPUTS = {"mit": "mit/judged.jsonl", "wiki": "kowiki/wiki-labeled-eval.jsonl",
             "klue": "klue/judged-eval.jsonl", "boolq": "boolq/judged-eval.jsonl"}
G3_TRAIN_INPUTS = {"wiki-train": "kowiki/wiki-labeled-train.jsonl"}
G3_KEEP = {"vitaminc": 0.52}               # 라벨 안 된 원천만. 나머지는 1.0
G3_FLOOR = 1.6                             # 라벨 안 된 원천은 후보를 b 목표의 1.6배 이상
G3_GROUP_CAP = {"vitaminc": 4}
G3_A_PER_B = {"mit": 1, "boolq": 1}       # b 묶음 안 a 는 b 수만큼만 (MIT 는 a 가 몰려 있다). 남은 줄은 버린다(묶음째 규칙)
G3_OWN_RNG = ("klue", "boolq")             # 따로 씨앗 — 이 둘의 파일이 뒤에 생겨도 VitaminC 차례(공용 씨앗)가 안 바뀐다
G3_SEED = 20261301
G3_PARA_SIM = 0.8                          # 문장 글자 3묶음 닮음 (설계 3-1)
G3_PARA_MIN = 15                           # 이보다 짧은 문장은 겹침 검사에서 뺀다
G2_SET_FILES = ("dev-cmp-g2", "dev-cal-g2", "blind-g2", "train-g2-b")
_OLD_EVAL_FILES = (("g1", "dev-cal"), ("g1", "dev-cmp"), ("g1", "blind-g1"),
                   ("g2", "dev-cmp-g2"), ("g2", "dev-cal-g2"), ("g2", "blind-g2"))
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_COMMIT = re.compile(r"^([^:]+):([0-9a-f]{12})(?::|$)")


def judge_pick(judge):
    """판정 결과의 확정 칸 : 사람이 되돌린 칸(override.to) → final → label 차례. 없으면 ""."""
    if not isinstance(judge, dict):
        return ""
    override = judge.get("override")
    if isinstance(override, dict) and override.get("to") in ("a", "b", "c", "d"):
        return override["to"]
    for name in ("final", "label"):
        if judge.get(name) in ("a", "b", "c", "d"):
            return judge[name]
    return ""


def g3_label(row):
    """확정 칸 : 판정 결과(override · final · label) → label → want → hint 차례."""
    picked = judge_pick(row.get("judge"))
    if picked:
        return picked
    for name in ("label", "want", "hint"):
        if row.get(name) in ("a", "b", "c", "d"):
            return row[name]
    return ""


def sha_key(row):
    """blind-g2.forbid.json 의 key_rule — sha256(evidence.strip() + '\\x1f' + claim.strip())."""
    text = (row.get("evidence") or "").strip() + "\x1f" + (row.get("claim") or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _para_sentences(row):
    """근거 문장 정규형 목록. 위키 근거 앞의 「제목 : 」은 뗀다."""
    text = row.get("evidence") or row.get("premise") or ""
    title = row.get("title")
    if title and text.startswith(title + " : "):
        text = text[len(title) + 3:]
    out = []
    for sent in _SENT_SPLIT.split(text):
        sent = norm(sent)
        if len(sent) >= G3_PARA_MIN:
            out.append(sent)
    return out


class ParaIndex:
    """문장 겹침 색인 — 정규화 일치 또는 글자 3묶음 Jaccard ≥ sim. 앞부분 거르기(prefix filter)로 후보만 맞춰 본다."""

    def __init__(self, sim=G3_PARA_SIM):
        self.sim = sim
        self.grams, self.tags = [], []
        self.post = collections.defaultdict(list)
        self.exact = collections.defaultdict(set)

    def _prefix(self, grams):
        keep = len(grams) - math.ceil(self.sim * len(grams)) + 1
        return sorted(grams)[:max(1, keep)]

    @staticmethod
    def _gram_set(sent):
        return frozenset(sent[i:i + 3] for i in range(max(1, len(sent) - 2)))

    def add(self, row, tag):
        for sent in _para_sentences(row):
            self.exact[sent].add(tag)
            grams = self._gram_set(sent)
            idx = len(self.grams)
            self.grams.append(grams)
            self.tags.append(tag)
            for gram in self._prefix(grams):
                self.post[gram].append(idx)

    def hits(self, row):
        """이 줄 근거와 겹치는 색인 줄의 꼬리표 모음."""
        out = set()
        for sent in _para_sentences(row):
            out |= self.exact.get(sent, set())
            grams = self._gram_set(sent)
            seen = set()
            for gram in self._prefix(grams):
                for idx in self.post.get(gram, ()):
                    if idx in seen or self.tags[idx] in out:
                        continue
                    seen.add(idx)
                    other = self.grams[idx]
                    if len(grams & other) >= self.sim * len(grams | other):
                        out.add(self.tags[idx])
        return out


class G3Forbid(G2Forbid):
    """g3 후보가 피해야 할 것 — G2Forbid(g1 · 옛 셋 · use1 · 잣대) + g2 셋 넷의 글 · 문장 · 기억 id · 묶음
    + blind-g2.forbid.json + 씨앗 forbid-g3.txt(평가 몫만) + g2 예약 묶음(VitaminC 몫)."""

    def __init__(self, old_root, g1_build, use1_root, ruler_ids_path, g2_build, seed_forbid_path):
        super().__init__(old_root, g1_build, use1_root, ruler_ids_path)
        g2_sets = [os.path.join(old_root, "sets", "g2", n + ".jsonl") for n in G2_SET_FILES]
        for path in g2_sets:
            if not os.path.exists(path):
                self.missing.append(os.path.basename(path))
        keys, ids, groups = _file_keys_ids_groups([p for p in g2_sets if os.path.exists(p)])
        self.text |= keys
        self.sides |= {side for key in keys for side in key if side}
        self.eval_ids |= ids
        self.train_ids |= ids
        self.eval_groups = set(self.eval_groups) | groups
        self.train_groups = set(self.train_groups) | groups
        self.g2_json = {}
        path = os.path.join(old_root, "sets", "g2", "blind-g2.forbid.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.g2_json = {k: set(v) for k, v in json.load(f).items() if isinstance(v, list)}
        else:
            self.missing.append("blind-g2.forbid.json")
        self.seed = collections.defaultdict(set)
        if seed_forbid_path and os.path.exists(seed_forbid_path):
            with open(seed_forbid_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#") and "\t" in line:
                        name, value = line.rstrip("\n").split("\t", 1)
                        self.seed[name].add(value)
        else:
            self.missing.append(os.path.basename(seed_forbid_path or "forbid-g3.txt"))
        self.g2_reserved, self.vitc_train_pages = set(), set()
        path = os.path.join(g2_build, "reserve", "reserve-g2.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.g2_reserved = set(json.load(f)["reserved"])
        else:
            self.missing.append("reserve-g2.json")
        _, pages = load_source(g2_build, "vitaminc-train")
        self.vitc_train_pages = set(pages or ())

    @staticmethod
    def _docs_commits(row):
        docs = {g[4:] for g in row.get("groups") or [row.get("group")] if g and g.startswith("doc:")}
        commits = set()
        for src in row.get("src_ids") or []:
            match = _COMMIT.match(src)
            if match:
                commits.add(f"{match.group(1)}:{match.group(2)}")
            elif "/" in src or src.endswith(".md"):
                docs.add(src)
        return docs, commits

    def reasons(self, row, role):
        out = super().reasons(row, role)
        groups = set(row.get("groups") or [row.get("group")])
        srcs = set(row.get("src_ids") or [])
        ids = set(_MEM_ID.findall(json.dumps(row.get("src_ids", []))))
        docs, commits = self._docs_commits(row)
        key = sha_key(row)
        flip = sha_key({"evidence": row.get("claim"), "claim": row.get("evidence")})
        j = self.g2_json
        if (groups & j.get("groups", set()) or srcs & j.get("src_ids", set()) or {key, flip} & j.get("pair_keys", set())
                or ids & j.get("mem_ids", set()) or docs & j.get("doc_paths", set()) or commits & j.get("commits", set())
                or row.get("id") in j.get("cand_ids", set())):
            out.append("blind_g2_forbid")
        if role == "eval":
            s = self.seed
            if (groups & s["groups"] or docs & s["doc_paths"] or ids & s["mem_ids"] or {key, flip} & s["pair_keys"]):
                out.append("seed_g3")
            if groups & self.g2_reserved:
                out.append("g2_reserved")
            if row.get("lang") == "en" and groups & self.vitc_train_pages:
                out.append("page_in_train")
        return out


def _old_eval_index(old_root):
    """g1 · g2 평가 셋의 한국어 지문(KLUE 등) 문장 색인 — 위키 · BoolQ 문단 겹침을 뺄 때 쓴다."""
    index = ParaIndex()
    for gen, name in _OLD_EVAL_FILES:
        path = os.path.join(old_root, "sets", gen, name + ".jsonl")
        if os.path.exists(path):
            for row in read_jsonl(path):
                if row.get("lang") == "ko" or row.get("domain") in ("klue-mrc", "klue"):
                    index.add(row, "old")
    return index


def _g3_eval_paragraphs(build):
    """BoolQ · KLUE 평가 몫 원문(판정 전 후보) — 학습 층이 그 문단과 겹치면 뺀다. 없으면 빈 목록."""
    rows = []
    for rel in ("boolq/dev.jsonl", "boolq/test.jsonl"):
        path = os.path.join(build, rel)
        if os.path.exists(path):
            rows.extend(read_jsonl(path))
    path = os.path.join(build, "klue", "answered.jsonl")
    if os.path.exists(path):
        rows.extend(r for r in read_jsonl(path) if r.get("split") == "heldout")
    return rows


def _group_rows(source, rows):
    """한 묶음에서 평가 셋에 넣을 줄 — b 와 c · d 는 다, a 는 G3_A_PER_B 만큼, VitaminC 는 앞 G3_GROUP_CAP 줄."""
    rows = sorted(rows, key=lambda r: (g3_label(r) != "b", r["id"]))
    if source in G3_GROUP_CAP:
        return rows[:G3_GROUP_CAP[source]]
    if source in G3_A_PER_B:
        n_b = sum(1 for r in rows if g3_label(r) == "b")
        if not n_b:
            return rows                    # b 없는 묶음(a · c · d)은 그대로 — 셋에 고르게 나눈다
        a_rows = [r for r in rows if g3_label(r) == "a"][:G3_A_PER_B[source] * n_b]
        return [r for r in rows if g3_label(r) != "a"] + a_rows
    return rows


def reserve_g3(build, forbid, old_root, seed=G3_SEED, targets=None, vitc_path=None, old_index=None):
    """평가 몫을 묶음째 예약하고 남은 묶음을 학습 층으로 → (예약 문서, 셋별 줄, 학습 층 줄)."""
    targets = targets or G3_B_TARGET
    rng = random.Random(seed)
    paths = {s: os.path.join(build, rel) for s, rel in G3_INPUTS.items()}
    paths["vitaminc"] = vitc_path or os.path.join(old_root, "build", "g2", "vitaminc-heldout", "candidates.jsonl")
    loaded = {s: read_jsonl(p) if os.path.exists(p) else None for s, p in paths.items()}
    old_index = old_index if old_index is not None else _old_eval_index(old_root)
    doc = {"seed": seed, "targets": targets, "keep": G3_KEEP, "floor": G3_FLOOR, "a_per_b": G3_A_PER_B,
           "inputs": {s: {"path": os.path.relpath(p, build) if p.startswith(build) else os.path.basename(p),
                          "rows": len(loaded[s]) if loaded[s] is not None else None} for s, p in paths.items()},
           "sets": {s: {} for s in G3_SETS}, "shift": {}, "pending": {}, "dropped_eval": {},
           "dropped_train": {}, "overlap": collections.Counter(), "missing": list(forbid.missing)}
    owner, owner_src, owner_key = {}, {}, {}
    owned_index = ParaIndex()
    eval_rows = {s: [] for s in G3_SETS}
    shift = {s: collections.Counter() for s in G3_SETS}
    for source in G3_ORDER:
        rows = loaded[source]
        keep = G3_KEEP.get(source, 1.0)
        tgt = {s: targets.get(s, {}).get(source, 0) + shift[s][source] for s in G3_SETS}
        dropped = collections.Counter()
        good = collections.defaultdict(list)
        for row in rows or []:
            if source == "vitaminc" and row.get("hint") != "b":
                continue
            if row.get("keep") is False:
                dropped["keep_false"] += 1
                continue
            why = forbid.reasons(row, "eval")
            if row.get("lang") == "ko" and old_index.hits(row):
                why.append("para_old")
            if why:
                dropped.update(why)
                continue
            good[row["group"]].append(row)
        order = sorted(good)
        (random.Random(f"{seed}:{source}") if source in G3_OWN_RNG else rng).shuffle(order)
        # b 가 든 묶음 먼저(몫 채우기), 그다음 b 없는 묶음(a · c · d 를 셋에 고르게)
        order.sort(key=lambda g: not any(g3_label(r) == "b" for r in good[g]))
        covered = {s: 0.0 for s in G3_SETS}
        count = collections.Counter()
        rows_in = collections.Counter()

        def hungry_b():
            open_sets = [s for s in G3_SETS if tgt[s] and (covered[s] < tgt[s] or (keep < 1 and count[s] < G3_FLOOR * tgt[s]))]
            if not open_sets:
                return None
            if source in G3_PRIORITY:
                return open_sets[0]
            return min(open_sets, key=lambda s: (covered[s] / tgt[s], G3_SETS.index(s)))

        def hungry_rest():
            sized = [s for s in G3_SETS if tgt[s]]
            return min(sized, key=lambda s: (rows_in[s] / tgt[s], G3_SETS.index(s))) if sized else None

        for gid in order:
            has_b = any(g3_label(r) == "b" for r in good[gid])
            set_name = hungry_b() if has_b else (hungry_rest() if source in G3_REST_TO_EVAL else None)
            if set_name is None:
                continue
            if gid in owner:
                continue
            picked = []
            for row in _group_rows(source, good[gid]):
                if {owner[g] for g in row.get("groups") or [row["group"]] if g in owner} - {set_name}:
                    dropped["group_other_set"] += 1
                    continue
                if set(row.get("src_ids", [])) & set(owner_src) or _both_keys(row) & set(owner_key):
                    dropped["src_or_text_owned"] += 1
                    continue
                if row.get("lang") == "ko" and owned_index.hits(row) - {set_name}:
                    dropped["para_cross_set"] += 1
                    doc["overlap"]["eval_para_cross_set_dropped"] += 1
                    continue
                picked.append(row)
            if not picked:
                continue
            for g in {g for r in picked for g in r.get("groups") or [r["group"]]} | {gid}:
                owner[g] = set_name
            for r in picked:
                owner_src.update({src: set_name for src in r.get("src_ids", [])})
                owner_key.update({k: set_name for k in _both_keys(r)})
                owned_index.add(r, set_name)
                eval_rows[set_name].append(dict(r, set=set_name, g3_source=source, g3_label=g3_label(r)))
            n_b = sum(1 for r in picked if g3_label(r) == "b")
            covered[set_name] += keep * n_b
            count[set_name] += n_b
            rows_in[set_name] += len(picked)
            entry = doc["sets"][set_name].setdefault(source, {"groups": [], "row_ids": []})
            entry["groups"].append(gid)
            entry["row_ids"].extend(r["id"] for r in picked)
        for set_name in G3_SETS:
            got = [r for r in eval_rows[set_name] if r["g3_source"] == source]
            entry = doc["sets"][set_name].setdefault(source, {"groups": [], "row_ids": []})
            short = max(0, math.ceil(tgt[set_name] - covered[set_name] - 1e-9))
            entry.update({"b_target": targets.get(set_name, {}).get(source, 0), "b_shift_in": shift[set_name][source],
                          "b_expected": round(covered[set_name], 1), "b_rows": count[set_name],
                          "labels": dict(sorted(collections.Counter(r["g3_label"] for r in got).items())),
                          "rows": len(got), "short": short, "input_missing": rows is None})
            to = G3_SHIFT.get(source)
            if rows is None:
                # 판정 파일이 아직 없는 원천은 넘기지 않고 제 몫(넘겨 받은 것 포함)을 대기로 둔다
                if tgt[set_name]:
                    doc["pending"].setdefault(set_name, {})[source] = tgt[set_name]
            elif short and to:
                shift[set_name][to] += short
                doc["shift"].setdefault(set_name, {})[f"{source}->{to}"] = short
        doc["dropped_eval"][source] = dict(dropped)
    # 학습 층 : 확정 원천의 예약 안 된 묶음 + 위키 학습 몫. VitaminC heldout · BoolQ · KLUE 평가 몫 나머지는 학습에 안 준다.
    eval_para = ParaIndex()
    for rows in eval_rows.values():
        for r in rows:
            eval_para.add(r, "g3")
    for r in _g3_eval_paragraphs(build):
        eval_para.add(r, "eval-split")
    train_rows = []
    train_inputs = {"mit": loaded["mit"], "wiki": loaded["wiki"]}
    for name, rel in G3_TRAIN_INPUTS.items():
        path = os.path.join(build, rel)
        train_inputs[name] = read_jsonl(path) if os.path.exists(path) else None
        doc["inputs"][name] = {"path": rel, "rows": len(train_inputs[name]) if train_inputs[name] is not None else None}
    owned_groups, owned_src = set(owner), set(owner_src)
    for name, rows in train_inputs.items():
        dropped = collections.Counter()
        for row in rows or []:
            why = forbid.reasons(row, "train")
            if set(row.get("groups") or [row["group"]]) & owned_groups:
                why.append("group_eval_g3")
            elif set(row.get("src_ids", [])) & owned_src:
                why.append("src_eval_g3")
            elif _both_keys(row) & set(owner_key):
                why.append("text_eval_g3")
            if not why:
                hits = eval_para.hits(row) | (old_index.hits(row) if row.get("lang") == "ko" else set())
                if hits:
                    why.append("para_eval")
                    doc["overlap"]["train_para_" + "_".join(sorted(hits))] += 1
            if why:
                dropped.update(why)
                continue
            train_rows.append(dict(row, g3_source=name, g3_label=g3_label(row)))
        doc["dropped_train"][name] = dict(dropped)
    doc["train"] = {n: {"rows": sum(1 for r in train_rows if r["g3_source"] == n),
                        "labels": dict(sorted(collections.Counter(r["g3_label"] for r in train_rows
                                                                  if r["g3_source"] == n).items()))}
                    for n in train_inputs}
    doc["overlap"].update({f"eval_para_old_dropped_{s}": doc["dropped_eval"][s].get("para_old", 0) for s in G3_ORDER})
    doc["overlap"] = dict(doc["overlap"])
    doc["reserved"] = sorted(owner)
    return doc, eval_rows, train_rows


def check_g3(eval_rows, train_rows, forbid, old_index):
    """평가 셋 사이 · 평가 × 학습 층 · 둘 × 금지 목록 겹침 수. 0 이어야 한다."""
    counts = collections.Counter()
    seen_group, seen_key, seen_src = {}, {}, {}
    para = ParaIndex()
    for set_name, rows in eval_rows.items():
        for row in rows:
            for gid in row.get("groups") or [row["group"]]:
                if seen_group.setdefault(gid, set_name) != set_name:
                    counts["eval_group_across_sets"] += 1
            for key in _both_keys(row):
                if seen_key.setdefault(key, set_name) != set_name:
                    counts["eval_text_across_sets"] += 1
            for src in row.get("src_ids", []):
                if seen_src.setdefault(src, set_name) != set_name:
                    counts["eval_src_across_sets"] += 1
            if row.get("lang") == "ko":
                if para.hits(row) - {set_name}:
                    counts["eval_para_across_sets"] += 1
                if old_index.hits(row):
                    counts["eval_para_old"] += 1
                para.add(row, set_name)
            for why in forbid.reasons(row, "eval"):
                counts["eval_" + why] += 1
    for row in train_rows:
        if set(row.get("groups") or [row["group"]]) & set(seen_group):
            counts["train_group_eval"] += 1
        if _both_keys(row) & set(seen_key):
            counts["train_text_eval"] += 1
        if set(row.get("src_ids", [])) & set(seen_src):
            counts["train_src_eval"] += 1
        if row.get("lang") == "ko" and para.hits(row):
            counts["train_para_eval"] += 1
        for why in forbid.reasons(row, "train"):
            counts["train_" + why] += 1
    names = ["eval_group_across_sets", "eval_text_across_sets", "eval_src_across_sets", "eval_para_across_sets",
             "eval_para_old", "train_group_eval", "train_text_eval", "train_src_eval", "train_para_eval"]
    names += [f"eval_{why}" for why in ("text_old", "side_old", "mem_id_old", "ruler_mem", "group_g1", "use1_pair",
                                         "use1_text", "blind_g2_forbid", "seed_g3", "g2_reserved", "page_in_train")]
    names += [f"train_{why}" for why in ("text_old", "mem_id_old", "ruler_mem", "group_g1", "use1_pair", "use1_text",
                                          "blind_g2_forbid")]
    return {k: counts.get(k, 0) for k in names}


def forbid_doc(set_name, rows, made_from):
    """blind-g2.forbid.json 과 같은 꼴의 금지 목록."""
    out = {"set": set_name, "made_from": made_from,
           "key_rule": "sha256(evidence.strip() + '\\x1f' + claim.strip())"}
    mem_ids, docs, commits, groups, srcs = set(), set(), set(), set(), set()
    for row in rows:
        mem_ids.update(_MEM_ID.findall(json.dumps(row.get("src_ids", []))))
        d, c = G3Forbid._docs_commits(row)
        docs |= d
        commits |= c
        groups.update(row.get("groups") or [row["group"]])
        srcs.update(row.get("src_ids", []))
    out.update({"mem_ids": sorted(mem_ids), "doc_paths": sorted(docs), "commits": sorted(commits),
                "groups": sorted(groups), "src_ids": sorted(srcs), "cand_ids": sorted(r["id"] for r in rows),
                "pair_keys": sorted({sha_key(r) for r in rows})})
    return out


def _g3_forbid(args):
    root = args.old_root or os.path.join(os.path.expanduser("~"), ".judgemodel")
    g2_build = args.g2_build or os.path.join(root, "build", "g2")
    return G3Forbid(root, args.g1_build or os.path.join(root, "build", "g1"),
                    args.use1 or os.path.join(root, "measure", "use1"),
                    args.ruler_ids or os.path.join(g2_build, "forbid-ruler-ids.txt"), g2_build,
                    args.seed_forbid or os.path.join(args.build, "seed", "forbid-g3.txt")), root


def reserve_g3_main(args):
    forbid, root = _g3_forbid(args)
    if forbid.missing:
        print(json.dumps({"missing": forbid.missing}, ensure_ascii=False))
        return 2
    targets = None
    if args.targets:
        with open(args.targets, encoding="utf-8") as f:
            targets = json.load(f)
    old_index = _old_eval_index(root)
    doc, eval_rows, train_rows = reserve_g3(args.build, forbid, root, args.seed, targets, args.vitaminc, old_index)
    out = os.path.join(args.build, "reserve")
    os.makedirs(out, exist_ok=True)
    for set_name, rows in eval_rows.items():
        _write_jsonl(os.path.join(out, f"{set_name}-cands.jsonl"), rows)
    _write_jsonl(os.path.join(out, "train-layer-cands.jsonl"), train_rows)
    doc["check"] = check_g3(eval_rows, train_rows, forbid, old_index)
    with open(os.path.join(out, "blind-g3.forbid.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(forbid_doc("blind-g3", eval_rows["blind-g3"], ["build/g3/reserve/blind-g3-cands.jsonl"]), f,
                  ensure_ascii=False, indent=1)
    with open(os.path.join(out, "reserve-g3.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    brief = {s: {src: (v["b_target"], v["b_shift_in"], v["b_expected"], v["rows"], v["short"])
                 for src, v in per.items()} for s, per in doc["sets"].items()}
    print(json.dumps({"sets(b_target,shift_in,b_expected,rows,short)": brief, "shift": doc["shift"],
                      "pending": doc["pending"], "train": doc["train"], "overlap": doc["overlap"],
                      "check": doc["check"]}, ensure_ascii=False, indent=1))
    return 1 if sum(doc["check"].values()) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="열쇠 묶음 예약 · 학습 후보 · 겹침 검사")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("reserve", "pool", "check"):
        p = sub.add_parser(name)
        p.add_argument("--build", required=True, help="후보 뿌리 (<원천>/candidates.jsonl 이 있는 곳)")
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--reserve", help="reserve.json (기본 <build>/reserve.json)")
    sub.choices["pool"].add_argument("--out", help="기본 <build>/train-cands.jsonl")
    sub.choices["pool"].add_argument("--total", type=int, default=TRAIN_TOTAL)
    sub.choices["pool"].add_argument("--archive", action="store_true",
                                     help="아카이브 · mem 학습 후보 (기본 --reserve <build>/reserve-archive.json · "
                                          "--out <build>/train-cands-archive.jsonl · 씨앗 20261048)")
    sub.choices["pool"].add_argument("--old-root", help="--archive 때 금지 목록 뿌리 (기본 ~/.judgemodel)")
    sub.choices["pool"].add_argument("--forbid", nargs="+",
                                     help="--archive 때 기억 id 를 뽑을 파일 (기본 sets/g1 의 dev-cal · dev-cmp · "
                                          "blind-g1 + blind v4)")
    sub.choices["pool"].add_argument("--public-cands", nargs="+",
                                     help="--archive 때 겹치면 안 되는 공개 학습 후보 (기본 train-cands · "
                                          "train-cands-klue.filtered · train-cands-jira.filtered)")
    sub.choices["pool"].add_argument("--eval-dir", nargs="+",
                                     help="--archive 때 글이 겹치면 안 되는 평가 후보 폴더 (기본 eval-public · eval-archive)")
    sub.choices["pool"].add_argument("--reserve-used-only", action="store_true",
                                     help="--archive 때 예약 전체 대신 얼린 셋 · eval-archive 묶음 폴더에 실제로 든 "
                                          "묶음 · 출처 id · 글만 막는다")
    sub.choices["pool"].add_argument("--group-cap", type=int, default=ARCH_TRAIN_GROUP_CAP,
                                     help="--archive 때 한 열쇠 묶음에서 고를 최대 줄 수 (기본 8)")
    sub.choices["reserve"].add_argument("--archive", action="store_true",
                                        help="아카이브 · mem 몫 예약 (기본 <build>/reserve-archive.json)")
    sub.choices["check"].add_argument("--cands", nargs="+", required=True)
    sub.choices["check"].add_argument("--exclude-relabel", action="store_true",
                                      help="dev-r2 다시 붙이기 줄(relabel)은 세기만 하고 검사에서 뺀다")
    sub.choices["check"].add_argument("--archive-reserve",
                                      help="reserve-archive.json — 학습 후보가 그 예약 묶음 · 출처 id 와 겹치는지도 본다")
    sub.choices["check"].add_argument("--old-root", help="옛 셋 뿌리 (~/.judgemodel). 없으면 옛 셋 검사를 건너뛴다")
    sub.choices["check"].add_argument("--eval-dir", nargs="+",
                                      help="eval-pool 출력 폴더(여럿 가능) — 평가 셋 사이 · 평가 × --cands(학습) · 평가 × 옛 셋 겹침도 본다")
    ev = sub.add_parser("eval-pool", help="예약 묶음 → 셋별 평가 후보 (공개 원천 · ood)")
    ev.add_argument("--build", required=True, help="후보 뿌리 (<원천>/candidates.jsonl 이 있는 곳)")
    ev.add_argument("--seed", type=int, default=SEED)
    ev.add_argument("--reserve", help="reserve.json (기본 <build>/reserve.json)")
    ev.add_argument("--out", help="기본 <build>/eval-public")
    ev.add_argument("--factor", type=float, default=EVAL_FACTOR)
    ev.add_argument("--ruler4", help="자 ④ 폴더 (ruler4-gold.jsonl · ruler4-input.jsonl) — 그 쌍을 ood-g1 에서 뺀다")
    ev.add_argument("--raw", help="원자료 뿌리 (WiCE subclaim ref 를 claim id 로 풀 때)")
    ev.add_argument("--archive", action="store_true",
                    help="아카이브 · mem 몫 (기본 --reserve <build>/reserve-archive.json · --out <build>/eval-archive)")
    ev.add_argument("--public-dir", help="--archive 때 이미 쓴 공개 평가 후보 폴더(읽기만, 기본 <build>/eval-public)")
    ev.add_argument("--traps", help="--archive 때 함정 풀 — 셋마다 그 셋 후보와 같은 쌍을 뺀 <셋>-traps.jsonl 을 쓴다")
    for name in ("reserve-g2", "check-g2"):
        g2 = sub.add_parser(name, help="v1.2 B 판 b 자 — 평가 몫 예약 · 학습 층 (reserve-g2) / 겹침 검사 (check-g2)")
        g2.add_argument("--build", required=True, help="g2 후보 뿌리 (<원천>/candidates.jsonl · reserve/ 가 생김)")
        g2.add_argument("--seed", type=int, default=G2_SEED)
        g2.add_argument("--old-root", help="옛 셋 뿌리 (기본 ~/.judgemodel)")
        g2.add_argument("--g1-build", help="g1 후보 뿌리 (reserve.json · reserve-archive.json · klue 다시쓰기, 기본 <old-root>/build/g1)")
        g2.add_argument("--use1", help="use1 뿌리 (EXCLUDE.sha256list · */set-use1*.jsonl, 기본 <old-root>/measure/use1)")
        g2.add_argument("--ruler-ids", help="잣대 기억 id 목록 (기본 <build>/forbid-ruler-ids.txt)")
        g2.add_argument("--train", help="check-g2 : 학습 층 대신 볼 학습 파일 (예 sets/g2/train-g2-b.jsonl)")
    g3 = sub.add_parser("reserve-g3", help="v1.2 K 판 — g3 평가 셋 새 쌍 묶음째 예약 · 학습 층 · blind-g3.forbid.json")
    g3.add_argument("--build", required=True, help="g3 뿌리 (kowiki · mit · boolq · klue · seed 아래 판정 파일, reserve/ 가 생김)")
    g3.add_argument("--seed", type=int, default=G3_SEED)
    g3.add_argument("--targets", help="셋별 원천 b 목표 json ({셋: {원천: 수}}, 기본 설계 2-3 표)")
    g3.add_argument("--vitaminc", help="VitaminC heldout 후보 (기본 <old-root>/build/g2/vitaminc-heldout/candidates.jsonl)")
    g3.add_argument("--old-root", help="옛 셋 뿌리 (기본 ~/.judgemodel)")
    g3.add_argument("--g1-build", help="g1 후보 뿌리 (기본 <old-root>/build/g1)")
    g3.add_argument("--g2-build", help="g2 후보 뿌리 (reserve/reserve-g2.json · vitaminc-train, 기본 <old-root>/build/g2)")
    g3.add_argument("--use1", help="use1 뿌리 (기본 <old-root>/measure/use1)")
    g3.add_argument("--ruler-ids", help="잣대 기억 id 목록 (기본 <g2-build>/forbid-ruler-ids.txt)")
    g3.add_argument("--seed-forbid", help="씨앗 금지 목록 (기본 <build>/seed/forbid-g3.txt)")
    at = sub.add_parser("attach-rewrite", help="학습 후보 KLUE 줄에 Opus 평서문 claim 붙이기 → klue 줄만")
    at.add_argument("--cands", required=True, help="학습 후보 (train-cands.jsonl)")
    at.add_argument("--rewrite", nargs="+", required=True, help="rewrite-train-*.jsonl (cand_idx · group · src_ids · claim)")
    at.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "reserve-g2":
        return reserve_g2_main(args)
    if args.cmd == "check-g2":
        return check_g2_main(args)
    if args.cmd == "reserve-g3":
        return reserve_g3_main(args)
    if args.cmd == "attach-rewrite":
        return attach_rewrite_main(args)
    if args.cmd == "eval-pool" and args.archive:
        return eval_pool_archive_main(args)
    if args.cmd == "pool" and args.archive:
        return pool_archive_main(args)
    reserve_path = args.reserve or os.path.join(args.build, "reserve.json")

    if args.cmd == "eval-pool":
        with open(reserve_path, encoding="utf-8") as f:
            reserve_doc = json.load(f)
        excludes, unresolved = (frozenset(), frozenset()), 0
        if args.ruler4:
            gold = read_jsonl(os.path.join(args.ruler4, "ruler4-gold.jsonl"))
            inputs = read_jsonl(os.path.join(args.ruler4, "ruler4-input.jsonl"))
            ids, claims, unresolved = ruler4_ood_excludes(gold, inputs, args.raw)
            excludes = (frozenset(ids), frozenset(claims))
        sets, report = eval_pool(args.build, reserve_doc, args.seed, args.factor, excludes)
        report["ruler4_unresolved"] = unresolved
        out = args.out or os.path.join(args.build, "eval-public")
        os.makedirs(out, exist_ok=True)
        for set_name, rows in sets.items():
            with open(os.path.join(out, f"{set_name}-cands.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["cross"] = eval_cross(sets)
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(json.dumps({s: (v.get("total", v.get("picked")), v.get("hint_total", v.get("by_kind")))
                          for s, v in report["sets"].items()} | {"cross": report["cross"]}, ensure_ascii=False))
        return 1 if sum(report["cross"].values()) else 0

    if args.cmd == "reserve" and args.archive:
        doc = reserve_archive(args.build, args.seed)
        doc["cross"] = arch_cross_check(doc)
        path = args.reserve or os.path.join(args.build, "reserve-archive.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        brief = {s: {d: (v["pairs"], v["pairs_covered"], len(v["groups"]), v["usable_candidates"]) for d, v in per.items()}
                 for s, per in doc["sets"].items()}
        print(json.dumps({"shift_to_archive": doc["shift"], "sets(pairs,covered,groups,usable)": brief,
                          "reserved_groups": len(doc["reserved"]), "cross": doc["cross"]}, ensure_ascii=False, indent=1))
        return 1 if sum(doc["cross"].values()) else 0
    if args.cmd == "reserve":
        doc = reserve(args.build, args.seed)
        with open(reserve_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        brief = {s: {src: (v.get("pairs"), len(v["groups"]), v.get("usable_candidates")) for src, v in per.items()}
                 for s, per in doc["sets"].items()}
        print(json.dumps({"jira": doc["jira"], "shares": doc["shares"], "sets(pairs,groups,usable)": brief,
                          "reserved_groups": len(doc["reserved"])}, ensure_ascii=False, indent=1))
        return 0

    with open(reserve_path, encoding="utf-8") as f:
        reserve_doc = json.load(f)
    if args.cmd == "pool":
        chosen, report = pool(args.build, reserve_doc, args.seed, args.total)
        out = args.out or os.path.join(args.build, "train-cands.jsonl")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for row in chosen:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out + ".summary.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0

    cands = [row for path in args.cands for row in read_jsonl(path)]
    relabel = [row for row in cands if row.get("relabel")] if args.exclude_relabel else []
    if relabel:
        cands = [row for row in cands if not row.get("relabel")]
    old_keys, old_ids, missing = (set(), set(), [])
    if args.old_root:
        old_keys, old_ids, missing = load_old(args.old_root)
    result = check(cands, reserve_doc, args.build, old_keys, old_ids)
    result["candidates"] = len(cands)
    result["old_sets_missing"] = missing
    result["old_pairs"] = len(old_keys) // 2
    result["old_mem_ids"] = len(old_ids)
    if args.exclude_relabel:
        result["relabel_skipped"] = len(relabel)
    if args.archive_reserve:
        with open(args.archive_reserve, encoding="utf-8") as f:
            result.update(arch_overlap(cands, json.load(f)))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    bad = sum(result[k] for k in ("group_reserved", "text_reserved", "text_old", "mem_id_old"))
    bad += result.get("arch_group_reserved", 0) + result.get("arch_src_reserved", 0)
    if args.eval_dir:
        cross = eval_cross(load_eval_dirs(args.eval_dir), cands, old_keys, old_ids if args.exclude_relabel else frozenset(),
                           skip_relabel_old=args.exclude_relabel)
        print(json.dumps(cross, ensure_ascii=False, indent=1))
        bad += sum(cross.values())
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
