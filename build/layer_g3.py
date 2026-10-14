"""layer_g3 — v1.2 K 판 학습 층 train-g3-ko 조립 (원천 다섯 → 상한 맞춘 한 파일).

설계 `../Docs/Design/2026-10-13-v1.2-한국어b보강판설계.md` 3-1 · 3-3 · 4-1 · 8-2. 쌍 본문은 화면에 찍지 않는다.

    python -I layer_g3.py build --g3 ~/.judgemodel/build/g3 [--targets <json>] [--seed 20261013] \
        [--eval <평가 셋.jsonl> …] [--forbid-json <셋.forbid.json> …] [--old-root ~/.judgemodel] [--no-old-forbid] [--out-dir <폴더>]
    → <g3>/train-g3-ko/cands.jsonl · summary.json

원천 (모두 확정 칸 v6) :
  위키 · MIT  `reserve/train-layer-cands.jsonl` (예약이 평가와 겹친 줄을 이미 뺀 학습 후보, 칸 = g3_label)
  BoolQ       `boolq/boolq-labeled-train.jsonl`
  씨앗        `seed/judged.jsonl` + `seed/judged-d3.jsonl` (d 는 셋째 판정이 남긴 줄만)
  KLUE        `klue/klue-labeled-train.jsonl` — 없으면 0 으로 돈다. 있으면 `klue/klue-labeled-eval.jsonl` 이 --eval 기본에 붙는다

규칙 (TARGETS, --targets 로 덮음) :
  - b 몫을 원천 목표로 잡고 아래 상한을 다 지킬 때까지 줄인다(줄이기만 하므로 끝난다).
    위키 b ≤ 층 ko b 의 40% · 위키 + BoolQ b ≤ 65%(넘으면 둘을 지금 비율대로 줄임) ·
    씨앗 + KLUE b ≤ (기준 처방 b 767 + 층 b) 의 25%(합성 상한, 넘으면 둘을 지금 비율대로 줄임).
  - 위키 b 는 문서당 2 · 되돌림 flag 붙은 b 는 고른 위키 b 의 20% 까지.
  - 씨앗은 씨앗 하나(b · a · d 줄)가 단위다. 있는 줄 중 하나라도 확정 칸 ≠ hint(d 는 셋째 판정 keep 아님)면 씨앗째 버린다.
    d 있는 씨앗 → a 있는 씨앗 → b 만 → 차례로, 같은 차례 안은 고정 씨앗으로 섞어 고른다.
  - BoolQ 는 같은 문단 짝이 없어 a 를 b 수만큼 따로 고른다. MIT 는 a 만 있어 a 가 b 에 모자란 만큼만 (상한 100, 문서당 2).
  - d 는 씨앗 d 먼저, 나머지를 위키 d 로 목표(450)까지. BoolQ d 는 목표 0.
  - 평가 셋과 글(근거 · 주장 양쪽 차례) · 묶음 · 출처 id · 문단(문장 정규화 일치 또는 글자 3묶음 닮음 ≥ .8) 겹침,
    옛 셋 금지(make_sets.G3Forbid 학습 몫), forbid.json 들, scrub 걸림 → 고르기 전에 뺀다(씨앗 · KLUE 는 짝째).
"""
import argparse
import collections
import copy
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scrub  # noqa: E402
from make_sets import (ParaIndex, _both_keys, _old_eval_index, G3Forbid, g3_label, read_jsonl,  # noqa: E402
                       sha_key)

SEED = 20261013
LAYER = "train-g3-ko"
SOURCES = ("wiki", "boolq", "seed", "mit", "klue")
TARGETS = {
    "b": {"wiki": 500, "boolq": 350, "seed": 280, "mit": 0, "klue": 150},   # 설계 4-1 표
    "d_total": 450,
    "boolq_d": 0,
    "boolq_a_per_b": 1.0,
    "mit_a_cap": 100,
    "per_doc_b": {"wiki": 2},
    "per_doc_a": {"mit": 2},
    "caps": {"wiki_share": 0.40, "wiki_boolq_share": 0.65, "synth_share": 0.25, "base_b": 767,
             "flag_share": 0.20},
}
LABELS = ("a", "b", "c", "d")
OUT_FIELDS = ("id", "evidence", "claim", "want", "label_guide", "domain", "group", "groups", "lang", "form", "hint",
              "origin", "lic", "src_ids", "status", "set", "pair", "pick", "flag")


def merge_targets(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_targets(out[key], value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------- 원천 읽기


def _label(row):
    return g3_label(row) if row.get("judge") else (row.get("want") or row.get("hint") or "")


def load_sources(g3):
    """→ {원천: 줄 목록}, {원천: 파일 있음}. 줄마다 `_src` · `_label` 을 단다(나갈 때 뗀다)."""
    out, present = {s: [] for s in SOURCES}, {}
    layer = os.path.join(g3, "reserve", "train-layer-cands.jsonl")
    present["wiki"] = present["mit"] = os.path.exists(layer)
    if present["wiki"]:
        for row in read_jsonl(layer):
            src = {"wiki-train": "wiki", "mit": "mit"}.get(row.get("g3_source"))
            if src:
                out[src].append(dict(row, _src=src, _label=g3_label(row)))
    path = os.path.join(g3, "boolq", "boolq-labeled-train.jsonl")
    present["boolq"] = os.path.exists(path)
    if present["boolq"]:
        out["boolq"] = [dict(r, _src="boolq", _label=_label(r)) for r in read_jsonl(path) if r.get("keep", True)]
    judged, d3 = os.path.join(g3, "seed", "judged.jsonl"), os.path.join(g3, "seed", "judged-d3.jsonl")
    present["seed"] = os.path.exists(judged)
    if present["seed"]:
        third = {r["id"]: r for r in read_jsonl(d3)} if os.path.exists(d3) else {}
        for row in read_jsonl(judged):
            row = dict(row, _src="seed")
            if row.get("hint") == "d":
                t = third.get(row["id"])
                row["_ok"] = bool(t and t.get("keep"))
                row["_label"] = "d" if row["_ok"] else ((t or {}).get("judge3") or {}).get("label", "")
            else:
                row["_label"] = row.get("want") or ""
                row["_ok"] = bool(row.get("keep")) and row["_label"] == row.get("hint")
            out["seed"].append(row)
    path = os.path.join(g3, "klue", "klue-labeled-train.jsonl")
    present["klue"] = os.path.exists(path)
    if present["klue"]:
        out["klue"] = [dict(r, _src="klue", _label=_label(r)) for r in read_jsonl(path) if r.get("keep", True)]
    return out, present


def seed_units(rows):
    """씨앗 줄 → (쓸 씨앗 단위 목록, 버린 까닭 셈). 단위 = 한 `pair` 의 줄들. 하나라도 어긋나면 씨앗째 버림."""
    by_pair = collections.OrderedDict()
    for row in rows:
        by_pair.setdefault(row["pair"], []).append(row)
    units, why = [], collections.Counter()
    for pid, members in by_pair.items():
        if "b" not in {r["hint"] for r in members}:
            why["no_b"] += 1
            continue
        bad = [r for r in members if not r.get("_ok")]
        if bad:
            why["off_hint:" + "".join(sorted(r["hint"] for r in bad))] += 1
            continue
        units.append(members)
    return units, why


def klue_units(rows):
    """KLUE 줄 → 짝(b · a) 단위. 둘 다 확정 칸 = hint 일 때만."""
    by_pair = collections.OrderedDict()
    for row in rows:
        by_pair.setdefault(row.get("pair") or row["id"], []).append(row)
    units, why = [], collections.Counter()
    for members in by_pair.values():
        if any(r["_label"] != r.get("hint") for r in members):
            why["off_hint"] += 1
            continue
        if "b" not in {r["_label"] for r in members}:
            why["no_b"] += 1
            continue
        units.append(members)
    return units, why


# ---------------------------------------------------------------- 겹침 · 금지


class Guard:
    """학습 줄이 못 드는 까닭 — 평가 셋 겹침(글 · 묶음 · 출처 id · 문단) · 옛 셋 금지 · forbid.json · scrub."""

    def __init__(self, eval_paths, forbid_json_paths=(), old_forbid=None, old_index=None, scrubber=None):
        self.keys, self.groups, self.srcs = set(), set(), set()
        self.para = ParaIndex()
        self.eval_rows = 0
        for path in eval_paths:
            for row in read_jsonl(path):
                self.eval_rows += 1
                self.keys |= _both_keys(row)
                self.groups.update(g for g in (row.get("groups") or [row.get("group")]) if g)
                self.srcs.update(row.get("src_ids") or [])
                self.para.add(row, "eval")
        self.json = collections.defaultdict(set)
        for path in forbid_json_paths:
            with open(path, encoding="utf-8") as f:
                for key, value in json.load(f).items():
                    if isinstance(value, list):
                        self.json[key].update(value)
        self.old_forbid, self.old_index, self.scrubber = old_forbid, old_index, scrubber

    def reasons(self, row):
        out = []
        if _both_keys(row) & self.keys:
            out.append("eval_text")
        groups = {g for g in (row.get("groups") or [row.get("group")]) if g}
        if groups & self.groups:
            out.append("eval_group")
        srcs = set(row.get("src_ids") or [])
        if srcs & self.srcs:
            out.append("eval_src")
        if self.para.hits(row):
            out.append("eval_para")
        j = self.json
        flip = sha_key({"evidence": row.get("claim"), "claim": row.get("evidence")})
        if groups & j["groups"] or srcs & j["src_ids"] or {sha_key(row), flip} & j["pair_keys"] \
                or row.get("id") in j["cand_ids"]:
            out.append("forbid_json")
        if self.old_index is not None and self.old_index.hits(row):
            out.append("old_para")
        if self.old_forbid is not None:
            out.extend("old_" + why for why in self.old_forbid.reasons(row, "train"))
        if self.scrubber is not None and sum(self.scrubber.hits(row).values()):
            out.append("scrub")
        return out


def screen(sources, guard):
    """→ (남은 원천 줄, 버린 까닭 셈). 씨앗 · KLUE 는 한 줄이 걸리면 짝째 뺀다."""
    kept, dropped = {}, collections.Counter()
    for src, rows in sources.items():
        bad_pairs, ok = set(), []
        for row in rows:
            why = guard.reasons(row)
            if why:
                for w in why:
                    dropped[f"{src}:{w}"] += 1
                if src in ("seed", "klue"):
                    bad_pairs.add(row.get("pair") or row["id"])
                continue
            ok.append(row)
        if bad_pairs:
            dropped[f"{src}:pair_dropped_rows"] += sum(1 for r in ok if (r.get("pair") or r["id"]) in bad_pairs)
            ok = [r for r in ok if (r.get("pair") or r["id"]) not in bad_pairs]
        kept[src] = ok
    return kept, dropped


# ---------------------------------------------------------------- 상한


def _shrink_pair(counts, names, cap):
    """names 원천의 합이 cap 을 넘으면 지금 비율대로 줄인다. 바꿨으면 True."""
    total = sum(counts[n] for n in names)
    if total <= cap:
        return False
    first, second = names
    counts[first] = int(round(cap * counts[first] / total))
    counts[second] = min(counts[second], cap - counts[first])
    return True


def cap_b(want, caps):
    """원천별 b 수(목표와 가진 수 중 작은 값) → 상한을 다 지키는 b 수. 줄이기만 해 반드시 끝난다."""
    n = dict(want)
    for _ in range(10000):
        total = sum(n.values())
        changed = _shrink_pair(n, ["seed", "klue"], math.floor(caps["synth_share"] * (caps["base_b"] + total)))
        total = sum(n.values())
        if n["wiki"] > math.floor(caps["wiki_share"] * total):
            n["wiki"] = math.floor(caps["wiki_share"] * total)
            changed = True
        total = sum(n.values())
        changed |= _shrink_pair(n, ["wiki", "boolq"], math.floor(caps["wiki_boolq_share"] * total))
        if not changed:
            return n
    raise RuntimeError("상한 맞추기가 안 끝났다")


def shares(n, caps):
    total = sum(n.values())
    return {"layer_b": total,
            "wiki_share": round(n["wiki"] / total, 4) if total else 0.0,
            "wiki_boolq_share": round((n["wiki"] + n["boolq"]) / total, 4) if total else 0.0,
            "synth_share": round((n["seed"] + n["klue"]) / (caps["base_b"] + total), 4),
            "train_b_total": caps["base_b"] + total}


# ---------------------------------------------------------------- 고르기


def pick_per_doc(rows, n, per_doc, rng, flag_cap=None):
    """섞은 차례에서 문서당 per_doc · flag 붙은 줄 flag_cap 까지 n 줄."""
    rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(rows)
    used, out, flagged = collections.Counter(), [], 0
    for row in rows:
        if len(out) >= n:
            break
        if per_doc and used[row.get("group")] >= per_doc:
            continue
        is_flag = bool(row.get("flag"))
        if flag_cap is not None and is_flag and flagged >= flag_cap:
            continue
        used[row.get("group")] += 1
        flagged += is_flag
        out.append(row)
    return out


def wiki_b_avail(rows, per_doc, flag_share):
    """문서당 · flag 규칙 아래 위키 b 를 최대 몇 줄 고를 수 있나."""
    used, plain, flag = collections.Counter(), 0, 0
    for row in sorted(rows, key=lambda r: (bool(r.get("flag")), r["id"])):
        if per_doc and used[row.get("group")] >= per_doc:
            continue
        used[row.get("group")] += 1
        if row.get("flag"):
            flag += 1
        else:
            plain += 1
    best = plain
    while best < plain + flag and (best + 1 - plain) <= math.floor(flag_share * (best + 1)):
        best += 1
    return best


def order_seed_units(units, rng):
    """d 있는 씨앗 → a 있는 씨앗 → b 만. 같은 차례 안은 고정 씨앗으로 섞는다."""
    tiers = {0: [], 1: [], 2: []}
    for unit in sorted(units, key=lambda u: u[0]["pair"]):
        hints = {r["hint"] for r in unit}
        tiers[0 if "d" in hints else 1 if "a" in hints else 2].append(unit)
    out = []
    for k in (0, 1, 2):
        rng.shuffle(tiers[k])
        out.extend(tiers[k])
    return out


def assemble(sources, targets, seed=SEED):
    """원천 줄(겹침 뺀 뒤) → (층 줄, 요약)."""
    caps = targets["caps"]
    by_label = {s: collections.defaultdict(list) for s in SOURCES}
    for src, rows in sources.items():
        if src in ("seed", "klue"):
            continue
        for row in rows:
            by_label[src][row["_label"]].append(row)
    units, unit_why = seed_units(sources.get("seed", []))
    k_units, k_why = klue_units(sources.get("klue", []))
    avail = {
        "wiki": wiki_b_avail(by_label["wiki"]["b"], targets["per_doc_b"].get("wiki"), caps["flag_share"]),
        "boolq": len(by_label["boolq"]["b"]),
        "seed": len(units),
        "mit": len(by_label["mit"]["b"]),
        "klue": len(k_units),
    }
    want = {s: min(targets["b"].get(s, 0), avail[s]) for s in SOURCES}
    n = cap_b(want, caps)
    rng = random.Random(seed)
    chosen = collections.defaultdict(list)

    # b
    wiki_b = pick_per_doc(by_label["wiki"]["b"], n["wiki"], targets["per_doc_b"].get("wiki"), random.Random(f"{seed}:wiki-b"),
                          flag_cap=math.floor(caps["flag_share"] * n["wiki"]))
    chosen["wiki"] += wiki_b
    chosen["boolq"] += pick_per_doc(by_label["boolq"]["b"], n["boolq"], 0, random.Random(f"{seed}:boolq-b"))
    chosen["mit"] += pick_per_doc(by_label["mit"]["b"], n["mit"], 0, random.Random(f"{seed}:mit-b"))
    seed_take = order_seed_units(units, random.Random(f"{seed}:seed"))[:n["seed"]]
    for unit in seed_take:
        chosen["seed"] += unit
    k_rng = random.Random(f"{seed}:klue")
    k_units = sorted(k_units, key=lambda u: u[0]["id"])
    k_rng.shuffle(k_units)
    for unit in k_units[:n["klue"]]:
        chosen["klue"] += unit

    # a
    n_boolq_a = min(len(by_label["boolq"]["a"]), int(round(targets["boolq_a_per_b"] * n["boolq"])))
    chosen["boolq"] += pick_per_doc(by_label["boolq"]["a"], n_boolq_a, 0, random.Random(f"{seed}:boolq-a"))
    chosen["wiki"] += sorted(by_label["wiki"]["a"] + by_label["wiki"]["c"], key=lambda r: r["id"])
    layer_b = sum(n.values())
    a_now = sum(1 for rows in chosen.values() for r in rows if r["_label"] == "a")
    n_mit_a = min(targets["mit_a_cap"], max(0, layer_b - a_now), len(by_label["mit"]["a"]))
    chosen["mit"] += pick_per_doc(by_label["mit"]["a"], n_mit_a, targets["per_doc_a"].get("mit"),
                                  random.Random(f"{seed}:mit-a"))

    # d
    d_now = sum(1 for rows in chosen.values() for r in rows if r["_label"] == "d")
    n_boolq_d = min(targets["boolq_d"], len(by_label["boolq"]["d"]))
    chosen["boolq"] += pick_per_doc(by_label["boolq"]["d"], n_boolq_d, 0, random.Random(f"{seed}:boolq-d"))
    n_wiki_d = min(len(by_label["wiki"]["d"]), max(0, targets["d_total"] - d_now - n_boolq_d))
    chosen["wiki"] += pick_per_doc(by_label["wiki"]["d"], n_wiki_d, 0, random.Random(f"{seed}:wiki-d"))

    out = []
    for src in SOURCES:
        for row in sorted(chosen[src], key=lambda r: r["id"]):
            out.append(to_layer_row(row, src))
    wiki_flag = sum(1 for r in wiki_b if r.get("flag"))
    pool = {s: {lab: len(by_label[s][lab]) for lab in LABELS} for s in ("wiki", "boolq", "mit")}
    pool["seed"] = dict(collections.Counter(r["hint"] for u in units for r in u))
    pool["klue"] = dict(collections.Counter(r["_label"] for u in k_units for r in u))
    final = {s: dict(collections.Counter(r["want"] for r in out if r["pick"] == s)) for s in SOURCES}
    summary = {
        "seed": seed, "targets": targets, "b_avail": avail, "b_want": want, "b_after_caps": n,
        "shares_before_caps": shares(want, caps), "shares_after_caps": shares(n, caps),
        "pool_after_screen": pool, "final_by_source": final,
        "final_total": dict(collections.Counter(r["want"] for r in out)), "rows": len(out),
        "wiki_flag_b": wiki_flag, "wiki_flag_share": round(wiki_flag / len(wiki_b), 4) if wiki_b else 0.0,
        "seed_units": {"kept": len(units), "used": len(seed_take), "dropped": dict(unit_why)},
        "klue_units": {"kept": len(k_units), "dropped": dict(k_why)},
        "short": {"a": max(0, layer_b - sum(1 for r in out if r["want"] == "a")),
                  "d": max(0, targets["d_total"] - sum(1 for r in out if r["want"] == "d")),
                  "b_vs_design": {s: targets["b"].get(s, 0) - n[s] for s in SOURCES}},
    }
    return out, summary


def to_layer_row(row, src):
    """학습기가 읽는 칸(id · evidence · claim · want) + train-g2-b 와 같은 곁 칸."""
    label = row["_label"]
    out = {k: row[k] for k in OUT_FIELDS if k in row}
    out.update(want=label, hint=row.get("hint") if src != "mit" else label, set=LAYER, pick=src,
               status=(row.get("judge") or {}).get("status") or "agree", label_guide=row.get("label_guide", "support-v6"))
    out.setdefault("pair", row.get("pair") or row["id"])
    out.setdefault("groups", [row.get("group")])
    if src == "mit":
        out["hint_pick"] = row.get("hint")
    return out


# ---------------------------------------------------------------- 명령


def default_eval_paths(g3, old_root):
    """겹침을 볼 평가 셋 기본값 — g3 예약 후보 셋 · g2 평가 셋 셋 · KLUE 원천이 있으면 그 평가 몫(klue-labeled-eval)."""
    paths = [os.path.join(g3, "reserve", f"{n}-cands.jsonl") for n in ("dev-cmp-g3", "dev-cal-g3", "blind-g3")]
    paths += [os.path.join(old_root, "sets", "g2", f"{n}.jsonl") for n in ("dev-cmp-g2", "dev-cal-g2", "blind-g2")]
    # KLUE 학습 몫을 쓰면 평가 몫과 겹치지 않는지도 본다 (학습 몫이 없으면 KLUE 는 0 으로 돌아 볼 것도 없다).
    # 학습 몫만 있고 평가 몫이 없으면 빠진 파일로 exit 2 — 조용히 거르지 않는다
    if os.path.exists(os.path.join(g3, "klue", "klue-labeled-train.jsonl")):
        paths.append(os.path.join(g3, "klue", "klue-labeled-eval.jsonl"))
    return paths


def recheck(rows, guard):
    """조립한 줄을 같은 잣대로 다시 본다 — 칸마다 0 이어야 한다. 줄 사이 같은 쌍도 센다."""
    counts = collections.Counter()
    seen = set()
    for row in rows:
        for why in guard.reasons(row):
            counts[why] += 1
        key = tuple(sorted(_both_keys(row)))
        if key in seen:
            counts["dup_in_layer"] += 1
        seen.add(key)
    return dict(counts)


def cmd_build(a):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    g3 = os.path.expanduser(a.g3)
    old_root = os.path.expanduser(a.old_root)
    targets = TARGETS
    if a.targets:
        with open(a.targets, encoding="utf-8") as f:
            targets = merge_targets(TARGETS, json.load(f))
    eval_paths = a.eval or default_eval_paths(g3, old_root)
    forbid_json = a.forbid_json if a.forbid_json is not None else [
        os.path.join(g3, "reserve", "blind-g3.forbid.json"), os.path.join(old_root, "sets", "g2", "blind-g2.forbid.json")]
    missing = [p for p in eval_paths + forbid_json if not os.path.exists(p)]
    old_forbid = old_index = None
    if not a.no_old_forbid:
        g2_build = os.path.join(old_root, "build", "g2")
        old_forbid = G3Forbid(old_root, os.path.join(old_root, "build", "g1"), os.path.join(old_root, "measure", "use1"),
                              os.path.join(g2_build, "forbid-ruler-ids.txt"), g2_build,
                              os.path.join(g3, "seed", "forbid-g3.txt"))
        missing += old_forbid.missing
        old_index = _old_eval_index(old_root)
    names = [os.path.join(old_root, "build", "g1", n) for n in ("names.txt", "game-names.txt")]
    missing += [p for p in names if not os.path.exists(p)]
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    scrubber = scrub.Scrubber(scrub.read_names([names[0]]), scrub.read_names([names[1]]))
    guard = Guard(eval_paths, forbid_json, old_forbid, old_index, scrubber)
    sources, present = load_sources(g3)
    raw = {s: dict(collections.Counter(r["_label"] for r in rows)) for s, rows in sources.items()}
    kept, dropped = screen(sources, guard)
    rows, summary = assemble(kept, targets, a.seed)
    summary.update(present=present, pool_raw=raw, dropped_screen=dict(sorted(dropped.items())),
                   eval_files=[os.path.basename(p) for p in eval_paths], eval_rows=guard.eval_rows,
                   forbid_json=[os.path.basename(p) for p in forbid_json], old_forbid=not a.no_old_forbid)
    summary["recheck"] = recheck(rows, guard)
    out_dir = os.path.expanduser(a.out_dir) if a.out_dir else os.path.join(g3, LAYER)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cands.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    brief = {k: summary[k] for k in ("present", "b_avail", "b_want", "b_after_caps", "shares_after_caps",
                                      "final_by_source", "final_total", "wiki_flag_share", "short", "recheck")}
    print(json.dumps(brief, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="train-g3-ko 후보 조립")
    s.add_argument("--g3", default="~/.judgemodel/build/g3")
    s.add_argument("--old-root", default="~/.judgemodel")
    s.add_argument("--targets", help="TARGETS 를 덮을 json (일부 칸만 적어도 됨)")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out-dir", help="기본 <g3>/train-g3-ko")
    s.add_argument("--eval", nargs="+", help="겹침을 볼 평가 셋 (기본 reserve 의 g3 셋 셋 후보 + g2 평가 셋 셋"
                   " + KLUE 학습 몫이 있으면 klue/klue-labeled-eval.jsonl)")
    s.add_argument("--forbid-json", nargs="*", help="forbid.json (기본 blind-g3 · blind-g2)")
    s.add_argument("--no-old-forbid", action="store_true", help="옛 셋 금지 · 옛 평가 문단 검사를 끈다 (시험용)")
    s.set_defaults(func=cmd_build)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
