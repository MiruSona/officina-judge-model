"""layer_g4bal — v1.2 K4 원천 균형 층 train-g4-bal (v5 표본 뽑기 → 새 판정 후보 뽑기 → 조립).

비공개 설계 1-1 · 1-2 · 2절 · 3절 · 5-5 ②. 쌍 본문은 화면에 찍지 않는다.

    python -I layer_g4bal.py sample-v5 --g4 ~/.judgemodel/build/g4 [--v5 <train-g1-II.jsonl> …]
    → <g4>/train-g4-bal/sample-v5.jsonl · sample-v5-summary.json   (2-1 층별 200줄 · label_batches.py make --mode train 에 넣는다)
    python -I layer_g4bal.py v5-rate   --g4 … [--dirs <v5 표본 판정 폴더> …]
    → <g4>/train-g4-bal/v5-rate.json   (2-2 칸별 r_h · R · Wilson 위끝 · 2-3 규칙의 결론)
    python -I layer_g4bal.py pick-new  --g4 … [--old-root ~/.judgemodel]
    → <g4>/train-g4-bal/pick.jsonl · pick-summary.json   (3-1 꼴 다섯)
    python -I layer_g4bal.py build     --g4 … [--v6 …] [--v5 …] [--v5-dirs …] [--dirs …] [--seed-pairs …]
    → <g4>/train-g4-bal/cands.jsonl · summary.json
    python -I layer_g4bal.py pick-klb  --g4 …   → <g4>/train-g4-klb/pick-ka.jsonl · pick-mrc.jsonl · pick-summary.json   (K5 4-3)
    python -I layer_g4bal.py build-klb --g4 … [--cands …] [--dirs <g4>/klb/judge]
    → <g4>/train-g4-klb/cands.jsonl · summary.json   (K5 4-2 조성 — compose_klb)
    python -I layer_g4bal.py thin-mrc-d [--layer <sets/g4/train-g4-bal.jsonl>] [--frozen …] [--klb …] [--keep 120]
    → <g4>/train-g4-bal2/cands.jsonl · summary.json   (K6 3-1 · 3-3 — klue-mrc d 줄이기, 남은 줄은 바이트 그대로)

조립 규칙 (1-1) : 재료 차례는 v6 → v5(실물만) → 새 판정. 원천 × 칸 목표(TARGETS build)를 넘지 않는다.
  짝 꼴(form contrast · seed-*)은 같은 `pair` 줄을 한 묶음으로 몫 안에 다 들 때만 통째로 넣고(깨진 짝 = 한 줄 묶음은 뺀다),
  그 차례 끝에 짝 일부 판으로 남은 몫을 채운다(VitaminC b 320 대 a 200).
  d 는 같은 원천에서 고른 c 의 근거 묶음 · 겹침 구간과 같은 줄을 먼저 쓴다(R2). 근거 묶음당 칸마다 2줄 · 쌍 열쇠 한 번 ·
  K 처방에 든 쌍은 안 넣는다. 금지 = layer_g4c.make_guard(+ 얼린 blind-g4 · dev-cmp-g4). 다른 칸으로 정해진 줄은 그 칸으로 옮긴다.
안 쓰는 것 : 합성 v5(origin · form synth · synth_from · set *-syn) · jira · minicheck · 목표 밖 원천.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))   # freeze.read_frozen (뒤에 붙여 build 이름이 이긴다)

import layer_g4d as ld  # noqa: E402
import public_pick  # noqa: E402
from assemble_g3 import Stop, judge_label, load_decisions, resolve  # noqa: E402
from assemble_g4 import CELL_OF, cell_key  # noqa: E402
from freeze import read_frozen  # noqa: E402
from label_batches import inside_git, load_map, read_jsonl, write_json, write_jsonl  # noqa: E402
from layer_g3 import merge_targets, recheck  # noqa: E402
from layer_g4c import _drop_boundary, _refuse_or_move, _targets, make_guard  # noqa: E402
from make_sets import _SENT_SPLIT, G1_SET_FILES, OLD_SETS, ParaIndex, load_old, near_c, norm, pair_key, sha_key  # noqa: E402

SEED = 20261404
KLB_SEED = 20261505            # build-klb 조성 씨앗
LAYER = "train-g4-bal"
KLB_LAYER = "train-g4-klb"     # K5 설계 4-3 — pick-klb 출력 폴더
THIN_SEED = 20261506           # thin-mrc-d 씨앗 (K6 설계 3-3)
BAL2_LAYER = "train-g4-bal2"   # K6 설계 3-2 — thin-mrc-d 출력 폴더
CELLS = ("a", "b", "c", "d")
SKIP_DOMAINS = ("jira", "minicheck")
UNIT_FORMS = ("contrast",)          # 그 밖에 form 이 seed- 로 시작하는 줄도 짝 묶음이다
BOOLQ, KLUE_ANS, MIT_REV, MIT_NEAR, MIT_OTHER = "boolq-removed", "klue-ans-removed", "mit-rev", "mit-near", "mit-other"
NEW_FORMS = {BOOLQ: ("kobest-boolq", "d"), KLUE_ANS: ("klue-answered", "d"), MIT_REV: ("mit-cs", "b"),
             MIT_NEAR: ("mit-cs", "c"), MIT_OTHER: ("mit-cs", "d")}
TARGETS = {
    "build": {   # 설계 1-2 「새 층」 열
        "vitaminc": {"a": 200, "b": 320, "c": 100, "d": 180},
        "klue-mrc": {"a": 90, "b": 37, "c": 132, "d": 243},
        "proposals": {"a": 50, "b": 13, "c": 310, "d": 130},
        "kobest-boolq": {"c": 300, "d": 67},
        "kowiki": {"c": 200, "d": 100},
        "klue-answered": {"c": 250, "d": 60},
        "mit-cs": {"b": 60, "c": 40, "d": 40},
        "archive": {"a": 100, "b": 100, "d": 100},
    },
    "sample": {  # 설계 2-1 표본 (모집단이 적으면 있는 만큼)
        "vitaminc": {"c": 20},
        "klue-mrc": {"a": 25, "b": 15, "c": 30},
        "proposals": {"a": 20, "b": 13, "c": 45, "d": 32},
    },
    "pick": {BOOLQ: 130, KLUE_ANS: 60, MIT_REV: 180, MIT_NEAR: 45, MIT_OTHER: 75},   # 설계 3-1 후보 열
    "per_group": 2,
    "klb": {     # K5 설계 4-3 후보 열 — klue-answered 질문(b · a 짝) · klue-mrc unanswerable 같은 묶음 질문
        "ka": {"b": 225, "a": 85}, "ka_per_group": 2,
        "mrc": 300, "mrc_min_q": 2, "mrc_per_group": 2,
        # K5 설계 4-2 「새 층」 열 — build-klb 조성 목표 (원천 × 칸)
        "build": {"klue-answered": {"a": 80, "b": 200}, "klue-mrc": {"b": 40, "d": 40}},
    },
    "rate_limit": 0.10,     # 설계 2-3 — 칸별 · 전체 변화율 문턱
    "edges": ld.TARGETS["edges"],
}
# 옛 평가 셋 (OLD_SETS 중 학습 · 합성 · twins 를 뺀 것) — ReuseGuard 가 이 글은 그대로 막는다
OLD_EVAL = {k: v for k, v in OLD_SETS.items() if k in ("dev-r1", "dev-r0", "blind-v1", "blind-v2", "blind-v3", "blind-v4", "fold40")}


# ---------------------------------------------------------------- 줄 판별


def is_synth(row):
    """합성 v5 줄(가5) — 안 쓴다(R8)."""
    return (row.get("origin") == "synth" or row.get("form") == "synth" or "synth_from" in row
            or str(row.get("set") or "").endswith("-syn"))


def _judge_cell(judge):
    """판정 칸 (override · final · label → 셋째 → 두 판이 같을 때). 없으면 ""."""
    label, _, _ = judge_label({"judge": judge})
    if label:
        return label
    if not isinstance(judge, dict):
        return ""
    votes = [judge.get(j).get("label") if isinstance(judge.get(j), dict) else judge.get(j) for j in ("j1", "j2", "j3")]
    if votes[2] in CELLS:
        return votes[2]
    return votes[0] if votes[0] in CELLS and votes[0] == votes[1] else ""


def row_cell(row):
    """학습 줄의 칸 글자 — want(낱말 · 글자) → 판정 칸. keep False · cd · 모름은 ""."""
    if row.get("keep") is False:
        return ""
    cell = CELL_OF.get(str(row.get("want") or ""))
    if cell:
        return cell
    return _judge_cell(row.get("judge"))


def is_unit(row):
    form = str(row.get("form") or "")
    return bool(row.get("pair")) and (form in UNIT_FORMS or form.startswith("seed-"))


def _usable(row, targets):
    """목표 안 원천 · 합성 · jira · minicheck 거름 → 뺀 까닭 (쓸 줄이면 "")."""
    domain = row.get("domain") or "mem"
    if is_synth(row):
        return "synth"
    if domain in SKIP_DOMAINS:
        return "skip_domain"
    if domain not in targets["build"]:
        return "no_target_domain"
    return ""


# ---------------------------------------------------------------- v5 표본 (2-1 · 2-2)


def v5_population(rows, targets):
    """v5 줄 → {(원천, 칸): 줄 목록} — 표본 목표가 있는 칸만 · 합성 · jira · minicheck 뺌 · id 한 번."""
    out, seen = collections.defaultdict(list), set()
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["id"] in seen or is_synth(row) or row.get("domain") in SKIP_DOMAINS:
            continue
        seen.add(row["id"])
        cell = row_cell(row)
        if cell and cell in (targets["sample"].get(row.get("domain")) or {}):
            out[(row["domain"], cell)].append(row)
    return out


def sample_v5(rows, targets=None, seed=SEED):
    """층별 표본 → (표본 줄, 요약). 판정자에게 v5 칸을 안 보이게 want 는 비우고 v5 칸은 v5_want 에 둔다(map 쪽에서만 쓴다)."""
    targets = merge_targets(TARGETS, targets)
    population = v5_population(rows, targets)
    out, summary = [], {"seed": seed, "targets": targets["sample"], "population": {}, "sample": {}}
    for domain in sorted(targets["sample"]):
        for cell in CELLS:
            if cell not in targets["sample"][domain]:
                continue
            pool = list(population.get((domain, cell)) or [])
            random.Random(f"{seed}:{domain}:{cell}:sample").shuffle(pool)
            got = pool[:targets["sample"][domain][cell]]
            for row in got:
                clean = {k: v for k, v in row.items() if k not in ("votes", "status", "relabel", "judge", "label_guide")}
                out.append(dict(clean, want="", hint=cell, v5_want=cell, set=f"{LAYER}-v5"))
            summary["population"][f"{domain} · {cell}"] = len(population.get((domain, cell)) or [])
            summary["sample"][f"{domain} · {cell}"] = len(got)
    summary["rows"] = len(out)
    return sorted(out, key=lambda r: r["id"]), summary


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre + spread) / (1 + z * z / n), 4)


def v5_rate(sample_rows, decisions, population, limit=TARGETS["rate_limit"]):
    """표본 판정 → 칸별 r_h · R · Wilson 위끝 · 결론. 바뀜 = 정함 칸 ≠ v5 칸, cd · drop · need-third 도 바뀜(2-2).
    판정 줄이 없거나 missing 이면 「판정 전」으로 센다."""
    decisions, _ = _drop_boundary(decisions)
    per = collections.defaultdict(lambda: collections.Counter())
    for row in sample_rows:
        key = f"{row['domain']} · {row['v5_want']}"
        dec = decisions.get(row["id"])
        if dec is None or dec.get("status") == "missing":
            per[key]["pending"] += 1
            continue
        if dec.get("status") == "need-third":
            per[key]["changed"] += 1
            per[key]["n"] += 1
            continue
        label, _, _, how = resolve(dict(row, want=""), decisions)
        per[key]["n"] += 1
        if how == "dropped" or label != row["v5_want"]:
            per[key]["changed"] += 1
    cells, total_n, weighted = {}, sum(population.values()), 0.0
    for key in sorted(per):
        n, k = per[key]["n"], per[key]["changed"]
        rate = round(k / n, 4) if n else None
        cells[key] = {"n": n, "changed": k, "pending": per[key]["pending"], "r_h": rate, "wilson_upper": wilson_upper(k, n),
                      "population": population.get(key, 0)}
        if rate is not None and total_n:
            weighted += population.get(key, 0) / total_n * rate
    over = sorted(k for k, v in cells.items() if v["r_h"] is not None and v["r_h"] >= limit)
    big = round(weighted, 4)
    if any(v["pending"] for v in cells.values()):
        verdict = "판정 전 줄이 남았다 — 결론 없음"
    elif big < limit and not over:
        verdict = "v5 줄을 그대로 쓴다 · 표본은 v6 칸으로 갈아 끼운다"
    elif len(over) >= 3 and big >= limit:
        verdict = "칸 셋 이상 재판정 — 메인에 알린다"
    else:
        verdict = "넘은 칸 종류만 모집단 전부 재판정"
    return {"limit": limit, "cells": cells, "R": big, "over": over, "verdict": verdict}


# ---------------------------------------------------------------- 새 판정 후보 (3-1)


def _sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def boolq_removed(layer_rows, pool_rows=()):
    """BoolQ 문단에서 주장과 가장 겹치는 문장을 뺀 문단 × 같은 주장 → (후보, 뺀 까닭).
    train-g3-ko 에 a · b 로 든 문단을 먼저 쓴다(R2 짝 — 제 문단에선 a · b, 그 문장을 뺀 문단에선 d)."""
    dropped = collections.Counter()
    first = [r for r in layer_rows if r.get("domain") == "kobest-boolq" and row_cell(r) in ("a", "b")]
    rest = [r for r in pool_rows if (r.get("domain") or "kobest-boolq") == "kobest-boolq"]
    out, seen = [], set()
    for rank, rows in enumerate((first, rest)):
        for row in sorted(rows, key=lambda r: r["id"]):
            sents = _sentences(row.get("evidence"))
            if len(sents) < 2:
                dropped["boolq_one_sentence"] += 1
                continue
            scores = [ld.overlap(s, row["claim"]) for s in sents]
            cut = scores.index(max(scores))
            evidence = " ".join(s for i, s in enumerate(sents) if i != cut)
            key = (norm(evidence), norm(row["claim"]))
            if key in seen:
                dropped["boolq_same_pair"] += 1
                continue
            seen.add(key)
            out.append(_new(BOOLQ, row, evidence, row["claim"], row.get("src_ids") or [row["id"]],
                            anchor=row["id"] if rank == 0 else None))
    return out, dropped


def _new(form, base, evidence, claim, src_ids, anchor=None):
    domain, hint = NEW_FORMS[form]
    row = {"id": None, "evidence": evidence, "claim": claim, "want": "", "hint": hint, "form": form,
           "domain": domain, "group": base["group"], "groups": list(base.get("groups") or [base["group"]]),
           "lang": base.get("lang") or "ko", "lic": base.get("lic", ""), "split": "train",
           "src_ids": list(dict.fromkeys(src_ids)), "origin": "g4-train", "set": LAYER, "pick": form}
    if anchor:
        row["anchor"] = anchor
    return row


def _retag(rows, form, layer_groups=()):
    """layer_g4d 꼴 줄 → 이 층 꼴 (domain · hint · form · set · pick). layer_groups 의 묶음 줄을 앞으로(anchor)."""
    out = []
    for row in rows:
        new = _new(form, row, row["evidence"], row["claim"], row.get("src_ids") or [])
        new["groups"] = list(row.get("groups") or [row["group"]])
        if row["group"] in layer_groups:
            new["anchor"] = row["group"]
        out.append(new)
    return out


def _repo(row):
    """doc:<저장소>:<경로> → 저장소."""
    parts = str(row.get("group") or "").split(":")
    return parts[1] if len(parts) > 2 else ""


def mit_near(mit_rows, n, seed):
    """MIT c — 같은 저장소 다른 문서의 주장 · BM25 10~60위 (make_sets.near_c 를 저장소마다)."""
    by_repo = collections.defaultdict(list)
    for row in mit_rows:
        by_repo[_repo(row)].append(dict(row, groups=list(row.get("groups") or [row["group"]])))
    out, per_group = [], collections.Counter()
    repos = sorted(by_repo)
    for k, repo in enumerate(repos):
        share = -(-(n - len(out)) // (len(repos) - k))       # 남은 몫을 남은 저장소에 고루 (올림)
        made = near_c(by_repo[repo], share, random.Random(f"{seed}:{repo}:mit-near"), per_group, f"mit-{k}-",
                      split="train", origin="g4-train")
        out += [dict(_new(MIT_NEAR, r, r["evidence"], r["claim"], r["src_ids"]), groups=r["groups"],
                     near_rank=r.get("near_rank")) for r in made]
    return out


def pick_new(src, guard, targets=None, seed=SEED, taken=frozenset()):
    """원천 줄(src) → (후보 줄, 요약). 꼴마다 앵커(K 에 든 문단 · 묶음) 먼저 · 묶음당 꼴(칸)마다 2 · 금지는 꺼낼 때 거른다.
    taken = 이미 학습 층(K 처방 · 앞 판 층)에 든 쌍 열쇠 — 같은 쌍은 안 짓는다."""
    targets = merge_targets(TARGETS, targets)
    dropped = collections.Counter()
    made = {}
    made[BOOLQ], why = boolq_removed(src.get("g3ko") or [], src.get("boolq") or [])
    dropped.update(why)
    klue, why = ld.klue_cands(src.get("klue_mrc") or [], src.get("klue_rw") or [], src.get("klue_check") or [])
    dropped.update(why)
    ka_groups = {r["group"] for r in src.get("g3ko") or [] if r.get("domain") == "klue-answered" and row_cell(r) in ("a", "b")}
    made[KLUE_ANS] = _retag(klue, KLUE_ANS, ka_groups)
    judged = {r["id"] for r in src.get("mit_judged") or []}
    made[MIT_REV] = [_new(MIT_REV, r, r["evidence"], r["claim"], r.get("src_ids") or [r["id"]])
                     for r in sorted(src.get("mit_rev") or [], key=lambda r: r["id"]) if r["id"] not in judged]
    dropped["mit_rev_judged"] += sum(1 for r in src.get("mit_rev") or [] if r["id"] in judged)
    made[MIT_NEAR] = mit_near(src.get("mit") or [], targets["pick"].get(MIT_NEAR, 0) * 2, seed)
    other, why = ld.wiki_cands(src.get("mit") or [])
    for key, n in why.items():
        dropped["mit_" + key] += n
    made[MIT_OTHER] = _retag(other, MIT_OTHER)
    out, summary = [], {"seed": seed, "targets": targets["pick"], "forms": {}}
    used, keys = collections.Counter(), set()
    for form in NEW_FORMS:
        rows = [r for r in made[form] if sha_key(r) not in taken]
        dropped[f"{form}:taken"] += len(made[form]) - len(rows)
        rows.sort(key=sha_key)
        random.Random(f"{seed}:{form}:pick").shuffle(rows)
        rows.sort(key=lambda r: 0 if r.get("anchor") else 1)
        got, limit = [], targets["pick"].get(form, 0)
        for row in rows:      # 금지는 꺼낼 때 본다 (MIT 같은 문서 꼴은 후보가 수만 줄이라 다 보면 느리다)
            if len(got) >= limit:
                break
            if used[(row["group"], form)] >= targets["per_group"]:
                continue
            key, flip = sha_key(row), sha_key({"evidence": row["claim"], "claim": row["evidence"]})
            if key in keys or flip in keys:
                dropped[f"{form}:dup_pair"] += 1
                continue
            why = guard.reasons(row)
            if why:
                dropped.update("made:" + w for w in why)
                continue
            used[(row["group"], form)] += 1
            keys |= {key, flip}
            got.append(row)
        for k, row in enumerate(sorted(got, key=sha_key), start=1):
            row["id"] = f"g4bal-{form}-{k:04d}-h{sha_key(row)[:8]}"
        out += got
        summary["forms"][form] = {"pool": len(made[form]), "picked": len(got), "anchor": sum(1 for r in got if r.get("anchor")),
                                  "short": targets["pick"].get(form, 0) - len(got)}
    summary["dropped"] = dict(sorted((k, v) for k, v in dropped.items() if v))
    summary["rows"] = len(out)
    return sorted(out, key=lambda r: r["id"]), summary


# ---------------------------------------------------------------- K5 train-g4-klb 후보 (K5 설계 4-3)


def _layer_marks(layer_rows):
    """학습 층 줄 → (이미 쓴 KLUE 질문 guid, K4 removed d 의 질문 guid → 그 d 줄 id, 문단 색인).
    removed d(KLUE_ANS 꼴) 줄은 세 줄 짝의 닻이라 「쓴 질문」 · 문단 색인에 안 넣는다."""
    used, anchors, para = set(), {}, ParaIndex()
    for row in layer_rows:
        if row.get("form") == KLUE_ANS:
            for src in row.get("src_ids") or []:
                anchors.setdefault(src, row["id"])
            continue
        if str(row.get("domain") or "").startswith("klue"):
            used.update(row.get("src_ids") or [])
            if row.get("pair"):
                used.add(row["pair"])
        para.add(row, "layer")
    return used, anchors, para


def _klb_id(kind, k, row):
    # 꼬리 -h 는 src_ids 해시라 assemble_g3.HASH_ID(쌍 해시) 뜻과 다르다 — 얼린 층이라 그대로 두고, 다음 층부터 꼬리 -s
    return f"klb-{kind}-{k:04d}-h{hashlib.sha256(json.dumps(row.get('src_ids'), ensure_ascii=False).encode('utf-8')).hexdigest()[:8]}"


def pick_klb(src, guard, layer_rows, targets=None, seed=SEED, taken=frozenset()):
    """klue-answered 세 줄 짝(qa-alt b · qa-true a, K4 removed d 닻 먼저) + klue-mrc 같은 문단(묶음) unanswerable 질문
    → (klue-answered 후보, klue-mrc 후보 — 평서문 전, 요약). 한 질문 줄이 하나라도 걸리면 그 질문을 통째 뺀다.
    금지 = guard(평가 · forbid.json · 옛 셋 · scrub) + 학습 층(K 처방 · train-g4-c · train-g4-bal) 문단 · 쓴 질문 · 쌍 열쇠."""
    t = merge_targets(TARGETS, targets)["klb"]
    used, anchors, para = _layer_marks(layer_rows)
    dropped = collections.Counter()
    prepared, why = public_pick.prepare([r for r in src.get("klue_rw") or [] if r.get("split") == "train"], "train",
                                        drop_ids=public_pick.failed_ids(src.get("klue_check") or []))
    for key, count in why.items():
        dropped["ka:" + key] += count
    by_q = collections.defaultdict(dict)
    for row in prepared:
        q = row.get("pair") or row["id"]
        if row["hint"] in by_q[q]:
            dropped["ka:dup_hint"] += 1          # 뒤 줄이 앞 줄을 덮는다 — cmd_pick_klb 가 경고 한 줄
        by_q[q][row["hint"]] = row

    def bad(rows, tag):
        for row in rows:
            reasons = list(guard.reasons(row))
            if para.hits(row):
                reasons.append("layer_para")
            if row.get("claim") and sha_key(row) in taken:
                reasons.append("taken")
            if reasons:
                dropped.update(f"{tag}:{w}" for w in reasons)
                return True
        return False

    questions = []
    for q in sorted(by_q):
        rows = [by_q[q][c] for c in ("b", "a") if c in by_q[q]]
        if "b" not in by_q[q]:
            dropped["ka:no_b"] += 1
            continue
        if q in used or any(s in used for r in rows for s in r.get("src_ids") or []):
            dropped["ka:used_question"] += 1
            continue
        if bad(rows, "ka"):
            continue
        questions.append(q)
    rng = random.Random(f"{seed}:klb:ka")
    rng.shuffle(questions)
    rank = {q: (0 if q in anchors else 1 if "a" in by_q[q] else 2) for q in questions}
    questions.sort(key=lambda q: rank[q])
    ka, per_group, n = [], collections.Counter(), {"a": 0, "b": 0}
    for q in questions:
        if n["b"] >= t["ka"]["b"]:
            break
        rows = by_q[q]
        group = rows["b"]["group"]
        if per_group[group] >= t["ka_per_group"]:
            continue
        per_group[group] += 1
        for cell in ("b", "a"):
            if cell not in rows or n[cell] >= t["ka"][cell]:
                continue
            row = {k: v for k, v in rows[cell].items() if k not in ("judge", "status", "votes")}
            row.update(want="", set=KLB_LAYER, origin="g4-train", pick=f"ka-{cell}", trio=rank[q] == 0)
            if q in anchors:
                row["anchor"] = anchors[q]
            n[cell] += 1
            ka.append(row)
    for k, row in enumerate(sorted(ka, key=lambda r: (r["pick"], r["id"])), start=1):
        row["src_id"], row["id"] = row["id"], _klb_id(row["pick"], k, row)

    done = set()
    for path_rows in src.get("klue_done") or []:
        done.update((r.get("group"), tuple(r.get("src_ids") or ())) for r in path_rows)
    by_group = collections.defaultdict(list)
    for row in sorted(src.get("klue_mrc") or [], key=lambda r: r["id"]):
        if row.get("split") != "train" or row.get("form") != "unanswerable":
            continue
        if (row.get("group"), tuple(row.get("src_ids") or ())) in done:
            dropped["mrc:rewritten_g1"] += 1
            continue
        if any(s in used for s in row.get("src_ids") or []):
            dropped["mrc:used_question"] += 1
            continue
        if bad([row], "mrc"):
            continue
        by_group[row["group"]].append(row)
    groups = [g for g in sorted(by_group) if len(by_group[g]) >= t["mrc_min_q"]]
    dropped["mrc:one_question_group"] = sum(len(v) for g, v in by_group.items() if len(v) < t["mrc_min_q"])
    random.Random(f"{seed}:klb:mrc").shuffle(groups)
    mrc = []
    for cap in (t["mrc_per_group"], max(len(v) for v in by_group.values()) if by_group else 0):
        for g in groups:
            have = sum(1 for r in mrc if r["group"] == g)
            for row in by_group[g][have:cap]:
                if len(mrc) >= t["mrc"]:
                    break
                mrc.append(dict(row, want="", claim="", needs_rewrite=True, set=KLB_LAYER, origin="g4-train", pick="mrc-u"))
        if len(mrc) >= t["mrc"]:
            break
    mrc.sort(key=lambda r: (r["group"], r["id"]))
    for k, row in enumerate(mrc, start=1):
        row["src_id"], row["id"] = row["id"], _klb_id("mrc-u", k, row)
    summary = {"seed": seed, "targets": t,
               "ka": {"pool_questions": len(by_q), "usable_questions": len(questions),
                      "usable_with_a": sum(1 for q in questions if "a" in by_q[q]),
                      "anchor_questions": sum(1 for q in questions if q in anchors), "anchors_in_k4": len(set(anchors.values())),
                      "picked": dict(n), "trio_rows": sum(1 for r in ka if r["trio"])},
               "mrc": {"usable_rows": sum(len(v) for v in by_group.values()), "multi_groups": len(groups),
                       "multi_rows": sum(len(by_group[g]) for g in groups), "picked": len(mrc),
                       "picked_groups": len({r["group"] for r in mrc}), "short": t["mrc"] - len(mrc)},
               "dropped": dict(sorted((k, v) for k, v in dropped.items() if v))}
    return ka, mrc, summary


def judged_keys(dirs):
    """판정 폴더들 → {후보 id: 판정자가 본 쌍 열쇠(sha_key) 모음}. map/map.jsonl(q → id) × batches/*.jsonl(q → 글).
    map 이나 batches 가 없거나, 같은 q 의 글이 묶음마다 다르거나, map 의 q 가 묶음에 없으면 Stop."""
    out = collections.defaultdict(set)
    for d in dirs:
        mapping, batch_dir = load_map(d), os.path.join(d, "batches")
        if not mapping or not os.path.isdir(batch_dir):
            raise Stop(f"판정 폴더에 map/map.jsonl · batches/ 가 없다 — 판정자가 본 글을 대조할 수 없다 ({d})")
        seen = {}
        for name in sorted(os.listdir(batch_dir)):
            if not name.endswith(".jsonl"):
                continue
            for item in read_jsonl(os.path.join(batch_dir, name)):
                key = sha_key(item)
                if seen.setdefault(item["q"], key) != key:
                    raise Stop(f"{name} : 같은 q 의 글이 묶음마다 다르다")
        for m in mapping:
            if m.get("trap"):
                continue
            if m["q"] not in seen:
                raise Stop(f"map 의 q 가 묶음에 없다 ({d})")
            out[m["id"]].add(seen[m["q"]])
    return dict(out)


def check_judged_text(cands, decisions, keys):
    """판정이 있는 후보마다 판정자가 본 글 = 후보 쌍인지 본다. 다르거나 본 글이 없으면 Stop.
    klb id 꼬리 해시는 src_ids 해시라 assemble_g3.resolve(id 꼬리 쌍 해시 대조)를 못 쓴다 — 그래서 따로 둔다."""
    for row in cands:
        if row["id"] not in decisions:
            continue
        if keys.get(row["id"]) != {sha_key(row)}:
            raise Stop(f"{row['id']} : 판정자가 본 쌍과 후보 쌍이 다르다 (또는 map 에 없다)")


def klb_labels(cands, decisions):
    """후보 id → 판정 칸 (agree · third 이고 칸이 a~d 일 때만, 그 밖 · 판정 없음은 "")."""
    out = {}
    for row in cands:
        dec = decisions.get(row["id"]) or {}
        ok = dec.get("status") in ("agree", "third") and dec.get("label") in CELLS
        out[row["id"]] = dec["label"] if ok else ""
    return out


class _KlbPlacer:
    """compose_klb 의 자리 셈 — 원천 × 칸 목표 안에서만 넣는다."""

    def __init__(self, build_t, dec):
        self.build_t, self.dec = build_t, dec
        self.got, self.placed = collections.defaultdict(list), set()

    def room(self, domain, cell):
        return len(self.got[(domain, cell)]) < self.build_t.get(domain, {}).get(cell, 0)

    def put(self, row, cell, how):
        if not self.room(row["domain"], cell):
            return False
        dec = self.dec[row["id"]]
        new = dict(row, want=cell, klb_how=how, label_guide="support-v6", status=dec.get("status"),
                   votes=dict(dec.get("votes") or {}))
        if cell != row.get("hint"):
            new["moved_from"] = row.get("hint")
        if cell in ("c", "d"):
            new["overlap"] = round(ld.overlap(new["evidence"], new["claim"]), 4)
        self.got[(row["domain"], cell)].append(new)
        self.placed.add(row["id"])
        return True


def _ka_tiers(by_q, label):
    """klue-answered 짝 차례 — (이름, 질문 → 그 차례에 드는가). 세 줄 짝(닻 질문) 먼저, 다음 b · a 짝."""
    def ok(row):
        return row is not None and label[row["id"]] == row["hint"]

    def pair(q):
        return ok(by_q[q].get("b")) and ok(by_q[q].get("a"))

    def trio(q):
        return bool((by_q[q].get("b") or {}).get("trio")) and pair(q)

    return ok, (("trio", trio), ("pair", pair))


def _compose_ka(cands, label, place, seed, stat):
    by_q = collections.defaultdict(dict)
    for row in cands:
        if row["domain"] == "klue-answered":
            by_q[row["pair"]][row["hint"]] = row
    qs = sorted(by_q)
    random.Random(f"{seed}:compose").shuffle(qs)
    ok, tiers = _ka_tiers(by_q, label)
    used = set()
    for how, test in tiers:
        for q in qs:
            if q in used or not test(q):
                continue
            if not (place.room("klue-answered", "a") and place.room("klue-answered", "b")):
                continue
            place.put(by_q[q]["b"], "b", how)
            place.put(by_q[q]["a"], "a", how)
            used.add(q)
    for q in qs:
        for c in ("b", "a"):
            row = by_q[q].get(c)
            if row is not None and row["id"] not in place.placed and ok(row):
                place.put(row, c, "single")
    for q in qs:
        for c in ("b", "a"):
            row = by_q[q].get(c)
            if row is None or not label[row["id"]] or ok(row):
                continue
            if place.put(row, label[row["id"]], "moved"):
                stat["ka_moved_in"] += 1
            else:
                stat[f"ka_moved_dropped:{label[row['id']]}"] += 1


def _compose_mrc(cands, label, place, stat):
    mrc = sorted((r for r in cands if r["domain"] == "klue-mrc"), key=lambda r: r["id"])
    by_g = collections.defaultdict(list)
    for row in mrc:
        by_g[row["group"]].append(row)

    def no_mate(row):
        return 0 if any(label[m["id"]] == "d" for m in by_g[row["group"]]) else 1

    bs = sorted((r for r in mrc if label[r["id"]] == "b"), key=no_mate)
    paired = 0
    for row in bs:
        if not place.put(row, "b", "mrc-b"):
            stat["mrc_b_over"] += 1
            continue
        mate = next((m for m in by_g[row["group"]] if label[m["id"]] == "d" and m["id"] not in place.placed), None)
        if mate is not None and place.put(dict(mate, mate_of=row["id"]), "d", "mrc-d-mate"):
            paired += 1
    stat["mrc_b_paired"] = paired
    for row in mrc:
        if label[row["id"]] in ("a", "c"):
            stat[f"mrc_dropped:{label[row['id']]}"] += 1


def _stat_key(key):
    return "|".join(key) if isinstance(key, tuple) else key


def compose_klb(cands, decisions, targets=None, seed=KLB_SEED):
    """판정 끝난 train-g4-klb 후보 → (층 줄, 요약). K5 설계 4-2 · 4-3 · README 36-2.
    klue-answered : 세 줄 짝 → b · a 짝 → 홀로 → 다른 칸은 자리가 있을 때만.
    klue-mrc : 같은 묶음에 d 가 있는 b 먼저, b 마다 그 d 하나 짝 (짝 없는 d 는 버림)."""
    build_t = merge_targets(TARGETS, targets)["klb"]["build"]
    label = klb_labels(cands, decisions)
    dec = {k: decisions.get(k) or {} for k in label}
    stat = collections.Counter()
    for row in cands:
        stat[(row["domain"], row["hint"], label[row["id"]] or dec[row["id"]].get("status", "pending"))] += 1
    place = _KlbPlacer(build_t, dec)
    _compose_ka(cands, label, place, seed, stat)
    _compose_mrc(cands, label, place, stat)
    got = place.got
    rows = sorted((r for v in got.values() for r in v), key=lambda r: r["id"])
    by_cell = {f"{d}:{c}": len(got[(d, c)]) for d in build_t for c in CELLS}
    summary = {"seed": seed, "targets": build_t, "rows": len(rows), "by_cell": by_cell,
               "short": {f"{d}:{c}": n - len(got[(d, c)]) for d in build_t for c, n in build_t[d].items() if len(got[(d, c)]) < n},
               "how": dict(sorted(collections.Counter(r["klb_how"] for r in rows).items())),
               "moved": sum(1 for r in rows if r.get("moved_from")),
               "judged": {_stat_key(k): v for k, v in sorted(stat.items(), key=str)}}
    return rows, summary


# ---------------------------------------------------------------- K6 train-g4-bal2 (K6 설계 3-1 · 3-3)


THIN_RULE = "thin-mrc-d: keep mate_b → pair → band largest-remainder (K6 설계 3-1)"


def _line_rows(data, name="입력"):
    """bytes → [(원래 줄 bytes, 읽은 dict 또는 None)]. 빈 줄은 dict 없이 그대로 둔다. 꼴이 틀리면 Stop(name 은 오류 문구의 파일 이름)."""
    out = []
    for k, raw in enumerate(data.splitlines(keepends=True), start=1):
        text = raw.strip()
        if not text:
            out.append((raw, None))
            continue
        try:
            row = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Stop(f"{name} {k} 번째 줄이 JSON 이 아니다")
        if not isinstance(row, dict):
            raise Stop(f"{name} {k} 번째 줄이 JSON 객체가 아니다")
        out.append((raw, row))
    return out


def _is_mrc(row, cell):
    """klue-mrc 의 그 칸 줄인가 — 원천 · 칸 견주기는 assemble_g4.cell_key 하나로."""
    return cell_key(row) == ("klue-mrc", cell)


def _thin_target(row):
    return row is not None and "_meta" not in row and row.get("origin") == "g4-train" and _is_mrc(row, "d")


def _row_groups(row):
    return {g for g in (row.get("groups") or [row.get("group")]) if g}


def _largest_remainder(total, weights):
    """total 을 weights 비로 나눈다 — 바닥 몫 뒤 남은 수를 소수 부분 큰 차례(같으면 앞 번호)로 하나씩."""
    whole = sum(weights.values())
    if not whole:
        return {k: 0 for k in weights}
    exact = {k: total * w / whole for k, w in weights.items()}
    got = {k: int(math.floor(v)) for k, v in exact.items()}
    for k in sorted(weights, key=lambda k: (-(exact[k] - got[k]), k))[:total - sum(got.values())]:
        got[k] += 1
    return got


def thin_mrc_d(lines, klb_rows, keep, seed=None, edges=None):
    """train-g4-bal 줄([(bytes, dict)]) → (남길 줄 bytes 목록, 요약). 대상 = klue-mrc d (origin g4-train).
    남김 차례 : ① 같은 묶음에 klue-mrc b(이 층 + klb)가 있는 d ② `pair` 칸이 있는 d ③ 나머지를 겹침 구간별로
    keep − (①+②) 을 대상 전체의 구간 수 비로 큰 나머지 몫 나눔 · 구간 안은 id 차례 정렬 뒤 씨앗으로 섞어 앞에서.
    대상 밖 줄은 바이트 그대로 차례 그대로. 대상이 없거나 · keep 범위 밖 · id 겹침 · 구간 몫이 모자라면 Stop."""
    seed = THIN_SEED if seed is None else seed
    edges = list(TARGETS["edges"] if edges is None else edges)
    rows = [r for _, r in lines if r is not None and "_meta" not in r]
    ids = [r.get("id") for r in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise Stop("id 가 비었거나 겹친다")
    target = sorted((r for _, r in lines if _thin_target(r)), key=lambda r: r["id"])
    if not target:
        raise Stop("대상 줄(klue-mrc d · origin g4-train)이 없다")
    if any(not (r.get("evidence") and r.get("claim")) for r in target):
        raise Stop("대상 줄에 빈 근거 · 주장이 있다")
    b_groups = set()
    for row in rows + list(klb_rows):
        if _is_mrc(row, "b"):
            b_groups |= _row_groups(row)
    stage, band = {}, {}
    for row in target:
        band[row["id"]] = ld.bin_of(ld.overlap(row["evidence"], row["claim"]), edges)
        stage[row["id"]] = "mate_b" if _row_groups(row) & b_groups else "pair" if row.get("pair") else "band"
    must = [r["id"] for r in target if stage[r["id"]] != "band"]
    if not len(must) <= keep < len(target):
        raise Stop(f"keep {keep} 가 범위 밖이다 — 짝 {len(must)} 이상 · 대상 {len(target)} 미만이어야 한다")
    bands = sorted(set(band.values()) | set(range(len(edges) + 1)))
    before = {b: sum(1 for r in target if band[r["id"]] == b) for b in bands}
    quota = _largest_remainder(keep - len(must), before)
    kept = set(must)
    for b in bands:
        pool = [r["id"] for r in target if stage[r["id"]] == "band" and band[r["id"]] == b]
        if quota[b] > len(pool):
            raise Stop(f"구간 {b} 몫 {quota[b]} 이 남은 줄 {len(pool)} 보다 많다 — keep 을 줄이거나 몫을 넘겨야 함")
        random.Random(f"{seed}:thin-mrc-d:{b}").shuffle(pool)
        kept.update(pool[:quota[b]])
    out = [raw for raw, row in lines if not _thin_target(row) or row["id"] in kept]
    removed = [r["id"] for r in target if r["id"] not in kept]
    summary = {"rule": THIN_RULE, "seed": seed, "keep": keep, "edges": edges,
               "target": len(target), "kept": len(kept), "removed_n": len(removed),
               "stages": {s: sum(1 for r in target if stage[r["id"]] == s and r["id"] in kept) for s in ("mate_b", "pair", "band")},
               "band_quota": {str(b): quota[b] for b in bands},
               "bands_before": {str(b): before[b] for b in bands},
               "bands_after": {str(b): sum(1 for r in target if band[r["id"]] == b and r["id"] in kept) for b in bands},
               "rows_before": len(rows), "rows_after": len(rows) - len(removed), "removed": removed}
    return out, summary


# ---------------------------------------------------------------- 재료 준비 (v6 · v5 · 새 판정)


def judged_rows(cands, decisions, draft=False):
    """판정 후보 + decided → (칸이 정해진 줄, 셈). 경계 칸 · 버림 · 판정 전은 뺀다. 다른 칸으로 정한 줄은 그 칸(moved)."""
    decisions, boundary = _drop_boundary(decisions)
    state, out = collections.Counter(boundary=boundary), []
    for cand in sorted(cands, key=lambda r: r["id"]):
        label, votes, status, how = resolve(dict(cand, want=""), decisions)
        state[how if how in ("pending", "dropped") else "decided"] += 1
        if label not in CELLS:
            continue
        row = {k: v for k, v in cand.items() if k != "judge"}
        row.update(want=label, label_guide="support-v6", status=status, votes=votes or {})
        if cand.get("hint") and cand["hint"] != label:
            row["moved_from"] = cand["hint"]
            state["moved"] += 1
        out.append(row)
    if state["pending"] and not draft:
        raise Stop(f"판정 전 줄 {state['pending']} — 판정을 끝내거나 --draft")
    return out, dict(state)


def v6_rows(rows):
    """얼린 v6 층 · 판정 풀 줄 → 칸 글자를 want 에 둔 줄 (cd · keep False · 모름은 뺀다)."""
    out = []
    for row in rows:
        cell = row_cell(row)
        if cell:
            out.append(dict({k: v for k, v in row.items() if k != "judge"}, want=cell))
    return out


def v5_rows(rows, decisions, block=(), cells=None):
    """v5 실물 줄 → (쓸 줄, 셈). cells(원천 → 칸 목록, 기본 TARGETS sample = 설계 2-1 의 여덟 칸)의 v5 칸 줄만 쓴다.
    v6 다시 판정한 줄은 그 칸을 따르고(cd · drop · need-third 는 뺌 · 다른 칸이면 그 칸으로 옮김),
    block(원천, 칸)의 줄은 다시 판정한 것만 쓴다(칸 종류 전부 재판정 때 남은 v5 칸을 안 섞게)."""
    decisions, _ = _drop_boundary(decisions)
    cells = TARGETS["sample"] if cells is None else cells
    count, out = collections.Counter(), []
    for row in rows:
        cell = row_cell(row)
        if not cell or is_synth(row):
            continue
        if cell not in (cells.get(row.get("domain")) or {}):
            count["not_v5_cell"] += 1
            continue
        dec = decisions.get(row["id"])
        if dec is not None:
            label, votes, status, how = (("", None, None, "dropped") if dec.get("status") in ("need-third", "missing")
                                         else resolve(dict(row, want=""), decisions))
            if label not in CELLS:
                count["rejudged_dropped"] += 1
                continue
            new = dict(row, want=label, label_guide="support-v6", status=status, votes=votes or {}, v5_want=cell)
            count["rejudged"] += 1
            if label != cell:
                new["moved_from"] = cell
                count["moved"] += 1
            out.append(new)
        elif (row.get("domain"), cell) in block:
            count["blocked"] += 1
        else:
            out.append(dict(row, want=cell))
    return out, dict(count)


# ---------------------------------------------------------------- 조립 (1-1 · 1-2)


class ReuseGuard:
    """재료 자신(이미 판정해 얼린 학습 줄, pair_key ∈ reuse)의 「자기 차단」 두 가지만 봐준다.
    G2Forbid · G3Forbid 는 옛 학습 셋(g1 학습 셋 · train-g2-b)의 글과 묶음까지 금지 집합(text · train_groups)에 넣는다.
    그래서 그 셋 줄을 다시 쓰면 자기 글 · 자기 묶음에 걸린다.
      old_text_old  : 글이 옛 평가 셋 글(eval_keys)에 없을 때만 뺀다
      old_group_g1  : 줄의 묶음 ∩ 옛 평가 셋 묶음(eval_groups) = ∅ 일 때만 뺀다 (이름은 g1 이지만 g2 셋 넷 묶음까지 가리킨다)
    그 밖 까닭(eval_* · old_*_g3 · blind_*_forbid · old_para* · mem_id_old · scrub)은 그대로다. 재료가 아닌 줄은 하나도 안 봐준다."""

    def __init__(self, guard, reuse_rows, eval_keys, eval_groups=frozenset()):
        self.guard, self.eval_keys, self.eval_groups = guard, eval_keys, eval_groups
        self.reuse = {pair_key(r) for r in reuse_rows}

    def reasons(self, row):
        out = list(self.guard.reasons(row))
        key = pair_key(row)
        if key not in self.reuse:
            return out
        if "old_text_old" in out and key not in self.eval_keys:
            out.remove("old_text_old")
        groups = {g for g in (row.get("groups") or [row.get("group")]) if g}
        if "old_group_g1" in out and not groups & self.eval_groups:
            out.remove("old_group_g1")
        return out


G2_EVAL_FILES = ("dev-cmp-g2", "dev-cal-g2", "blind-g2")


def old_eval_marks(old_root):
    """옛 평가 셋 표시 → (글 열쇠, 묶음, 못 찾은 이름). 글 = g1 평가 넷 + g2 평가 셋 + OLD_EVAL, 묶음 = g1 평가 넷 + g2 평가 셋."""
    files = {n: os.path.join("sets", "g1", n + ".jsonl") for n in G1_SET_FILES}
    files.update({n: os.path.join("sets", "g2", n + ".jsonl") for n in G2_EVAL_FILES})
    keys, _, missing = load_old(old_root, OLD_EVAL)
    more, _, missing_sets = load_old(old_root, files)
    groups = set()
    for rel in files.values():
        path = os.path.join(old_root, rel)
        if os.path.exists(path):
            for row in read_jsonl(path):
                groups.update(g for g in (row.get("groups") or [row.get("group")]) if g)
    return keys | more, groups, missing + missing_sets


class _Layer:
    """고른 줄과 몫 — 원천 × 칸 목표 · 묶음 × 칸 per_group · 쌍 열쇠 한 번 · K 처방 쌍 빼기 · 금지."""

    def __init__(self, targets, guard, taken, dropped):
        self.targets, self.guard, self.taken, self.dropped = targets, guard, taken, dropped
        self.rows, self.keys = [], set()
        self.count = collections.Counter()          # (원천, 칸)
        self.group = collections.Counter()          # (묶음, 칸)
        self.c_groups = collections.defaultdict(set)
        self.c_bins = collections.defaultdict(collections.Counter)

    def room(self, domain, cell):
        return self.targets["build"].get(domain, {}).get(cell, 0) - self.count[(domain, cell)]

    def why_not(self, row):
        key = sha_key(row)
        flip = sha_key({"evidence": row.get("claim"), "claim": row.get("evidence")})
        if key in self.taken or flip in self.taken:
            return "in_k"
        if key in self.keys or flip in self.keys:
            return "dup_pair"
        why = self.guard.reasons(row)
        return "guard:" + why[0] if why else ""

    def fits(self, unit):
        need = collections.Counter((r["domain"], r["want"]) for r in unit)
        if any(self.room(d, c) < n for (d, c), n in need.items()):
            return "no_room"
        groups = collections.Counter((r["group"], r["want"]) for r in unit)
        if any(self.group[g] + n > self.targets["per_group"] for g, n in groups.items()):
            return "group_cap"
        keys = [sha_key(r) for r in unit]
        if len(set(keys)) < len(keys):
            return "dup_pair"
        for row in unit:
            why = self.why_not(row)
            if why:
                return why
        return ""

    def add(self, row, stage, how):
        row = dict(row, set=LAYER, bal_stage=stage, bal_how=how, src_set=row.get("set"))
        if row["want"] in ("c", "d"):
            row["overlap"] = round(ld.overlap(row["evidence"], row["claim"]), 4)
        self.rows.append(row)
        self.keys.add(sha_key(row))
        self.count[(row["domain"], row["want"])] += 1
        self.group[(row["group"], row["want"])] += 1
        if row["want"] == "c":
            self.c_groups[row["domain"]].add(row["group"])
            self.c_bins[row["domain"]][ld.bin_of(row["overlap"], self.targets["edges"])] += 1

    def try_one(self, row, stage, how):
        why = self.fits([row])
        if why:
            if why not in ("no_room", "group_cap"):
                self.dropped[why] += 1
            return False
        self.add(row, stage, how)
        return True


def _d_first(layer, domain):
    """d 줄 차례 열쇠 — 고른 c 와 같은 묶음 먼저, 그다음 고른 c 가 많은 겹침 구간 먼저 (R2)."""
    def key(row):
        b = ld.bin_of(ld.overlap(row["evidence"], row["claim"]), layer.targets["edges"])
        return (0 if row["group"] in layer.c_groups[domain] else 1, -layer.c_bins[domain][b])
    return key


def _fill_singles(layer, rows, stage, seed, how):
    """원천마다 c 먼저 · d 는 c 를 따라 · 나머지 칸 — 몫까지 한 줄씩."""
    by = collections.defaultdict(list)
    for row in rows:
        by[(row["domain"], row["want"])].append(row)
    for domain in layer.targets["build"]:
        for cell in ("a", "b", "c", "d"):
            queue = sorted(by.get((domain, cell)) or [], key=sha_key)
            random.Random(f"{seed}:{domain}:{cell}:{stage}:{how}").shuffle(queue)
            if cell == "d":
                queue.sort(key=_d_first(layer, domain))
            for row in queue:
                if layer.room(domain, cell) <= 0:
                    break
                layer.try_one(row, stage, how)


def build(stages, guard, targets=None, seed=SEED, taken=frozenset()):
    """stages = [(차례 이름, 칸이 정해진 줄)] (v6 → v5 → new 차례) → (train-g4-bal 줄, 요약)."""
    targets = merge_targets(TARGETS, targets)
    dropped = collections.Counter()
    layer = _Layer(targets, guard, taken, dropped)
    seen_ids, by_stage = set(), {}
    units_taken = units_broken = 0
    for stage, rows in stages:
        usable = []
        for row in sorted(rows, key=lambda r: r["id"]):
            if row["id"] in seen_ids:
                dropped["same_id_earlier"] += 1
                continue
            why = _usable(row, targets)
            if why or row.get("want") not in CELLS:
                dropped[why or "no_cell"] += 1
                continue
            seen_ids.add(row["id"])
            usable.append(row)
        before = len(layer.rows)
        units = collections.defaultdict(list)
        singles = []
        for row in usable:
            (units[(row["domain"], row["pair"])] if is_unit(row) else singles).append(row)
        unit_list = [units[k] for k in sorted(units)]
        random.Random(f"{seed}:{stage}:units").shuffle(unit_list)
        left = []
        for unit in unit_list:
            if len(unit) < 2:
                units_broken += 1
                continue
            why = layer.fits(unit)
            if why:
                if why not in ("no_room", "group_cap"):
                    dropped["unit:" + why] += 1
                    continue
                left += unit
                continue
            for row in unit:
                layer.add(row, stage, "unit")
            units_taken += 1
        _fill_singles(layer, singles, stage, seed, "single")
        _fill_singles(layer, left, stage, seed, "partial")
        by_stage[stage] = collections.Counter(f"{r['domain']} · {r['want']}" for r in layer.rows[before:])
    rows = sorted(layer.rows, key=lambda r: r["id"])
    got = {f"{d} · {c}": layer.count[(d, c)] for d in targets["build"] for c in targets["build"][d]}
    summary = {"seed": seed, "targets": targets["build"], "got": got,
               "short": {k: targets["build"][k.split(" · ")[0]][k.split(" · ")[1]] - v for k, v in got.items()
                         if v < targets["build"][k.split(" · ")[0]][k.split(" · ")[1]]},
               "by_stage": {s: dict(sorted(c.items())) for s, c in by_stage.items()},
               "how": dict(collections.Counter(r["bal_how"] for r in rows)),
               "units": {"taken": units_taken, "broken": units_broken},
               "moved": sum(1 for r in rows if r.get("moved_from")),
               "dropped": dict(sorted(dropped.items())), "rows": len(rows)}
    return rows, summary


# ---------------------------------------------------------------- 명령


def _stop(message, **extra):
    print(json.dumps(dict({"stop": message}, **extra), ensure_ascii=False))
    return 2


def _guard(g4, old_root, no_old_forbid):
    """→ (Guard, 빠진 파일). 얼린 blind-g4 · dev-cmp-g4 가 없으면 빠진 파일로 센다(make_guard 는 있을 때만 읽는다)."""
    frozen = [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("blind-g4", "dev-cmp-g4")]
    missing = [p for p in frozen if not os.path.exists(p)]
    if missing:
        return None, missing
    return make_guard(g4, old_root, no_old_forbid)


def _out(a, names, layer=LAYER):
    """→ (출력 폴더, 출력 경로들, 멈춤 exit 또는 None). git 안 · 이미 있음(--force 없이)은 거부.
    검사만 한다 — 옛 출력 옮기기(--force)는 입력 검사가 다 끝난 뒤 _write 가 한다."""
    out_dir = os.path.join(os.path.expanduser(a.g4), layer)
    if inside_git(out_dir):
        return out_dir, [], _stop("출력 폴더가 git 안이다 — 쌍 본문을 git 안에 안 쓴다", out=out_dir)
    outs = [os.path.join(out_dir, n) for n in names]
    clash = [os.path.basename(p) for p in outs if os.path.exists(p)]
    if clash and not a.force:
        return out_dir, outs, _stop("출력이 이미 있다 — 덮지 않는다 (--force 면 날짜 꼬리로 돌려 둔다)", files=clash)
    return out_dir, outs, None


def _write(outs, force, *payloads):
    """옛 출력을 (--force 면) 날짜 꼬리로 돌려 두고 쓴다. payload 는 출력 차례대로 줄 목록(jsonl) 또는 dict(json)."""
    _refuse_or_move(outs, force)
    for path, data in zip(outs, payloads):
        if isinstance(data, list):
            write_jsonl(path, data)
        else:
            write_json(path, data)


def _v5_paths(a, old_root):
    return a.v5 if a.v5 else [os.path.join(old_root, "sets", "g1", "train-g1-II.jsonl")]


def _read_all(paths):
    rows = []
    for path in paths:
        rows += read_jsonl(os.path.expanduser(path))
    return rows


def _k_paths(old_root):
    """K 처방 학습 셋 (taken 쌍 열쇠)."""
    return [os.path.join(old_root, "sets", "2026-10-10", "build", "train-r2.jsonl"),
            os.path.join(old_root, "sets", "g1", "train-g1-arch.jsonl"),
            os.path.join(old_root, "sets", "g3", "train-g3-ko.jsonl")]


def _keys(rows):
    out = set()
    for row in rows:
        out.add(sha_key(row))
    return out


def cmd_sample(a):
    old_root = os.path.expanduser(a.old_root)
    _, outs, code = _out(a, ("sample-v5.jsonl", "sample-v5-summary.json"))
    if code:
        return code
    paths = _v5_paths(a, old_root)
    missing = [p for p in paths if not os.path.exists(os.path.expanduser(p))]
    if missing:
        return _stop("v5 파일이 없다", missing=missing)
    rows, summary = sample_v5(_read_all(paths), _targets(a.targets), a.seed)
    summary["inputs"] = [os.path.basename(p) for p in paths]
    _write(outs, a.force, rows, summary)
    print(json.dumps({k: summary[k] for k in ("population", "sample", "rows")}, ensure_ascii=False))
    return 0


def cmd_rate(a):
    g4 = os.path.expanduser(a.g4)
    out_dir, outs, code = _out(a, ("v5-rate.json",))
    if code:
        return code
    names = [os.path.join(out_dir, n) for n in ("sample-v5.jsonl", "sample-v5-summary.json")]
    missing = [p for p in names if not os.path.exists(p)]
    if missing:
        return _stop("표본 파일이 없다 — sample-v5 를 먼저", missing=missing)
    with open(names[1], encoding="utf-8") as f:
        population = json.load(f)["population"]
    dirs = [os.path.expanduser(d) for d in (a.dirs if a.dirs is not None else [os.path.join(g4, "judge", f"{LAYER}-v5")])]
    try:
        report = v5_rate(read_jsonl(names[0]), load_decisions(dirs), population,
                         merge_targets(TARGETS, _targets(a.targets))["rate_limit"])
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    report["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    _write(outs, a.force, report)
    print(json.dumps({k: report[k] for k in ("R", "over", "verdict")}, ensure_ascii=False))
    return 0


def _pick_inputs(a, old_root, keys=None):
    """원천 경로 dict. keys 를 주면 그 열쇠만. --input 의 모르는 열쇠는 Stop(exit 2)."""
    b = os.path.join(old_root, "build")
    paths = {"g3ko": os.path.join(old_root, "sets", "g3", "train-g3-ko.jsonl"),
             "boolq": os.path.join(b, "g3", "boolq", "boolq-labeled-train.jsonl"),
             "klue_mrc": os.path.join(b, "g1", "klue-mrc", "candidates.jsonl"),
             "klue_rw": os.path.join(b, "g3", "klue", "answered-rw.jsonl"),
             "klue_check": os.path.join(b, "g3", "klue", "answered-rw.check.jsonl"),
             "mit": os.path.join(b, "g3", "mit", "pairs.jsonl"),
             "mit_rev": os.path.join(b, "g3", "mit", "pairs-f.jsonl"),
             "mit_judged": os.path.join(b, "g3", "mit", "judged.jsonl")}
    if keys is not None:
        paths = {k: paths[k] for k in keys}
    for spec in a.input or []:
        key, _, path = spec.partition("=")
        if key not in paths:
            raise Stop(f"모르는 입력 {key} — {sorted(paths)}")
        paths[key] = os.path.expanduser(path)
    return paths


def _taken_paths(old_root):
    """이미 학습 층에 든 쌍 — K 처방 셋 + 앞 판 층(train-g2-b · train-g4-c · train-g4-d)."""
    return _k_paths(old_root) + [os.path.join(old_root, "sets", "g2", "train-g2-b.jsonl")] + \
        [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("train-g4-c", "train-g4-d")]


def cmd_pick(a):
    old_root = os.path.expanduser(a.old_root)
    _, outs, code = _out(a, ("pick.jsonl", "pick-summary.json"))
    if code:
        return code
    try:
        paths = _pick_inputs(a, old_root)
    except Stop as e:
        return _stop(str(e))
    taken_paths = _taken_paths(old_root)
    guard, missing = _guard(os.path.expanduser(a.g4), old_root, a.no_old_forbid)
    missing += [p for p in list(paths.values()) + taken_paths if not os.path.exists(p)]
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    src = {k: read_jsonl(p) for k, p in paths.items()}
    rows, summary = pick_new(src, guard, _targets(a.targets), a.seed, _keys(_read_all(taken_paths)))
    summary["inputs"] = {k: os.path.basename(p) for k, p in paths.items()}
    summary["recheck"] = recheck(rows, guard)
    _write(outs, a.force, rows, summary)
    print(json.dumps({k: summary[k] for k in ("rows", "forms", "recheck")}, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def cmd_pick_klb(a):
    """K5 train-g4-klb 후보 → <g4>/train-g4-klb/pick-ka.jsonl · pick-mrc.jsonl(평서문 전) · pick-summary.json."""
    old_root = os.path.expanduser(a.old_root)
    _, outs, code = _out(a, ("pick-ka.jsonl", "pick-mrc.jsonl", "pick-summary.json"), KLB_LAYER)
    if code:
        return code
    try:
        paths = _pick_inputs(a, old_root, ("klue_mrc", "klue_rw", "klue_check"))
    except Stop as e:
        return _stop(str(e))
    rw_dir = os.path.dirname(paths["klue_mrc"])
    done_paths = [os.path.join(rw_dir, f"{n}.jsonl") for n in ("rewrite-train-1", "rewrite-train-2", "rewrite-reserved")]
    layer_paths = _k_paths(old_root) + [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("train-g4-c", "train-g4-bal")]
    guard, missing = _guard(os.path.expanduser(a.g4), old_root, a.no_old_forbid)
    missing += [p for p in list(paths.values()) + done_paths + layer_paths if not os.path.exists(p)]
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    src = {k: read_jsonl(p) for k, p in paths.items()}
    src["klue_done"] = [read_jsonl(p) for p in done_paths]
    layer_rows = _read_all(layer_paths)
    ka, mrc, summary = pick_klb(src, guard, layer_rows, _targets(a.targets), a.seed, _keys(layer_rows))
    summary["inputs"] = {k: os.path.basename(p) for k, p in paths.items()}
    summary["layers"] = [os.path.basename(p) for p in layer_paths]
    summary["recheck_ka"] = recheck(ka, guard)
    if summary["dropped"].get("ka:dup_hint"):
        print(f"경고 : 같은 질문 · 같은 hint 줄 {summary['dropped']['ka:dup_hint']} — 뒤 줄이 앞 줄을 덮었다", file=sys.stderr)
    _write(outs, a.force, ka, mrc, summary)
    print(json.dumps({k: summary[k] for k in ("ka", "mrc", "recheck_ka")}, ensure_ascii=False))
    return 1 if sum(summary["recheck_ka"].values()) else 0


def cmd_build_klb(a):
    """K5 train-g4-klb 조성 — 판정 후보(pick-ka + 평서문 바꾼 pick-mrc) + decided → <g4>/train-g4-klb/cands.jsonl · summary.json.
    exit 2 = 파일 없음 · 후보 꼴 · 판정자가 본 쌍(map × batches)과 다름, exit 1 = 고른 줄이 금지 · 학습 층 쌍에 걸림."""
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    _, outs, code = _out(a, ("cands.jsonl", "summary.json"), KLB_LAYER)
    if code:
        return code
    cand_paths = [os.path.expanduser(p) for p in (a.cands or [os.path.join(g4, KLB_LAYER, "pick-ka.jsonl"),
                                                              os.path.join(g4, "klb", "pick-mrc-rw.jsonl")])]
    dirs = [os.path.expanduser(d) for d in (a.dirs or [os.path.join(g4, "klb", "judge")])]
    layer_paths = _k_paths(old_root) + [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("train-g4-c", "train-g4-bal")]
    guard, missing = _guard(g4, old_root, a.no_old_forbid)
    missing += [p for p in cand_paths + layer_paths if not os.path.exists(p)]
    missing += [d for d in dirs if not os.path.exists(os.path.join(d, "decided.jsonl"))]
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    cands = _read_all(cand_paths)
    ids = [r.get("id") for r in cands]
    domains = merge_targets(TARGETS, _targets(a.targets))["klb"]["build"]
    need = {"klue-answered": "pair", "klue-mrc": "group"}       # 짝을 묶는 칸
    bad = [r.get("id") for r in cands if not (r.get("evidence") and r.get("claim") and r.get("hint") in CELLS
                                              and r.get("domain") in domains and r.get(need.get(r.get("domain"), "id")))]
    if bad or len(ids) != len(set(ids)):
        return _stop("후보 꼴이 틀렸다 — 빈 쌍 · 칸 · 목표 밖 원천 · 짝 칸 없음 · id 겹침",
                     bad=len(bad), dup_ids=len(ids) - len(set(ids)))
    try:
        decisions = load_decisions(dirs)
        check_judged_text(cands, decisions, judged_keys(dirs))
        rows, summary = compose_klb(cands, decisions, _targets(a.targets), a.seed)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    taken = _keys(_read_all(layer_paths))
    summary["inputs"] = [os.path.basename(p) for p in cand_paths]
    summary["dirs"] = dirs
    summary["recheck"] = recheck(rows, guard)
    summary["taken"] = sum(1 for r in rows if sha_key(r) in taken)
    _write(outs, a.force, rows, summary)
    print(json.dumps({k: summary[k] for k in ("rows", "by_cell", "short", "how", "recheck", "taken")}, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) or summary["taken"] else 0


def _frozen_mismatch(path, frozen, digest):
    """FROZEN-*.sha256(train/freeze.py read_frozen 으로 읽음)과 입력을 견준다 → 맞으면 "", 아니면 멈춤 문구.
    파일 이름 칸이 입력 basename 과 같고 sha 가 같아야 맞다."""
    try:
        want, name = read_frozen(frozen)
    except (SystemExit, ValueError):
        return f"{os.path.basename(frozen)} 에 「sha 파일이름」 줄이 없다"
    if name.strip() != os.path.basename(path):
        return f"{os.path.basename(frozen)} 의 파일 이름({name.strip()})이 입력({os.path.basename(path)})과 다르다"
    if want != digest:
        return f"{os.path.basename(path)} sha 가 FROZEN 과 다르다"
    return ""


def cmd_thin(a):
    """K6 train-g4-bal2 — 얼린 train-g4-bal 에서 klue-mrc d 를 keep 줄로 줄임 → <g4>/train-g4-bal2/cands.jsonl · summary.json.
    klb 층도 FROZEN-<klb>.sha256 이 있으면 견준다(없으면 경고 한 줄 · summary klb_frozen null).
    exit 2 = 파일 없음 · 입력(또는 klb) sha · 이름 ≠ FROZEN · 입력 꼴 · 대상 없음 · keep 범위 밖 · 출력이 이미 있음(--force 없이) · git 안."""
    sets = os.path.join(os.path.expanduser(a.old_root), "sets", "g4")
    layer = os.path.expanduser(a.layer or os.path.join(sets, f"{LAYER}.jsonl"))
    frozen = os.path.expanduser(a.frozen or os.path.join(sets, f"FROZEN-{LAYER}.sha256"))
    klb = os.path.expanduser(a.klb or os.path.join(sets, f"{KLB_LAYER}.jsonl"))
    _, outs, code = _out(a, ("cands.jsonl", "summary.json"), BAL2_LAYER)
    if code:
        return code
    missing = [p for p in (layer, frozen, klb) if not os.path.exists(p)]
    if missing:
        return _stop("입력 파일이 없다", missing=missing)
    with open(layer, "rb") as f:
        data = f.read()
    with open(klb, "rb") as f:
        klb_data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    klb_digest = hashlib.sha256(klb_data).hexdigest()
    bad = _frozen_mismatch(layer, frozen, digest)
    if bad:
        return _stop(f"{bad} — 얼린 층만 줄인다", sha256=digest)
    klb_frozen = os.path.join(os.path.dirname(klb), f"FROZEN-{os.path.splitext(os.path.basename(klb))[0]}.sha256")
    if os.path.exists(klb_frozen):
        bad = _frozen_mismatch(klb, klb_frozen, klb_digest)
        if bad:
            return _stop(f"klb : {bad}", sha256=klb_digest)
    else:
        print(f"경고 : {os.path.basename(klb_frozen)} 이 없어 klb 층 sha 를 대조하지 않았다", file=sys.stderr)
        klb_frozen = None
    try:
        lines = _line_rows(data, os.path.basename(layer))
        klb_rows = [r for _, r in _line_rows(klb_data, os.path.basename(klb)) if r is not None]
        kept, summary = thin_mrc_d(lines, klb_rows, a.keep, a.seed, merge_targets(TARGETS, _targets(a.targets))["edges"])
    except Stop as e:
        return _stop(str(e))
    body = b"".join(kept)
    summary.update(input=os.path.basename(layer), input_sha256=digest, frozen=os.path.basename(frozen),
                   klb=os.path.basename(klb), klb_sha256=klb_digest,
                   klb_frozen=klb_frozen and os.path.basename(klb_frozen), out_sha256=hashlib.sha256(body).hexdigest())
    _refuse_or_move(outs, a.force)
    os.makedirs(os.path.dirname(outs[0]), exist_ok=True)
    with open(outs[0], "wb") as f:
        f.write(body)
    write_json(outs[1], summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "removed"}, ensure_ascii=False))
    return 0


def _split_pair(spec):
    """「decided.jsonl@pick.jsonl」 → (판정 폴더 또는 decided 파일, 후보 파일). @ 가 없으면 (None, 파일)."""
    if "@" in spec:
        dec, _, cands = spec.partition("@")
        return os.path.expanduser(dec), os.path.expanduser(cands)
    return None, os.path.expanduser(spec)


def _decisions_of(path):
    """decided.jsonl 파일 또는 그 폴더 → {id: 판정}."""
    if os.path.isdir(path):
        return load_decisions([path])
    return {d["id"]: d for d in read_jsonl(path)}


def _refuse_klb(rows, name):
    """train-g4-klb 줄(set · klb_how)이 v6 재료로 들어오면 Stop. klb id 꼬리 -h 는 src_ids 해시라
    resolve 가 쌍 해시로 잘못 읽는다 (README 36항)."""
    if any(r.get("set") == KLB_LAYER or "klb_how" in r for r in rows):
        raise Stop(f"{name} : train-g4-klb 줄은 build --v6 재료가 아니다 — id 꼬리 -h 가 쌍 해시가 아니다")


def _stage_inputs(a, old_root, g4):
    """→ (차례 목록, 셈, 다시 쓰는 얼린 줄) · 파일이 빠졌으면 (None, None, 빠진 파일)."""
    v6 = a.v6 if a.v6 is not None else [os.path.join(old_root, "sets", "g2", "train-g2-b.jsonl")] + \
        [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("train-g4-c", "train-g4-d")]
    files = [p for spec in v6 for p in _split_pair(spec) if p] + [os.path.expanduser(p) for p in _v5_paths(a, old_root)]
    new_pick = os.path.join(g4, LAYER, "pick.jsonl")
    if a.new is not False:
        files.append(new_pick)
    if a.v5_rate:
        files.append(os.path.join(g4, LAYER, "v5-rate.json"))
    files += [os.path.expanduser(p) for p in a.seed_pairs or []]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        return None, None, missing
    count, reuse = {}, []
    v6_all = []
    for spec in v6:
        dec, cands = _split_pair(spec)
        if dec:
            rows = read_jsonl(cands)
            _refuse_klb(rows, os.path.basename(cands))
            got, state = judged_rows(rows, _decisions_of(dec), draft=True)
            count[f"v6:{os.path.basename(cands)}"] = state
            v6_all += got
        else:
            rows = read_jsonl(cands)
            _refuse_klb(rows, os.path.basename(cands))
            reuse += rows
            v6_all += v6_rows(rows)
    v5_dirs = [os.path.expanduser(d) for d in a.v5_dirs or []]
    block = {tuple(s.split(":", 1)) for s in a.v5_block or []}
    if a.v5_rate:
        with open(os.path.join(g4, LAYER, "v5-rate.json"), encoding="utf-8") as f:
            over = json.load(f).get("over") or []
        block |= {tuple(k.split(" · ", 1)) for k in over}      # 넘은 칸 종류는 다시 판정한 줄만 쓴다 (설계 2-3)
    count["v5_block"] = sorted(":".join(b) for b in block)
    v5_raw = _read_all(_v5_paths(a, old_root))
    reuse += v5_raw
    v5_all, count["v5"] = v5_rows(v5_raw, load_decisions(v5_dirs), block,
                                  merge_targets(TARGETS, _targets(a.targets))["sample"])
    new_all = []
    if a.new is not False:
        dirs = [os.path.expanduser(d) for d in (a.dirs if a.dirs is not None else [os.path.join(g4, "judge", LAYER)])]
        new_all, count["new"] = judged_rows(read_jsonl(new_pick), load_decisions(dirs), a.draft)
    for path in a.seed_pairs or []:
        seeds = [r for r in read_jsonl(os.path.expanduser(path)) if r.get("keep") is True]
        new_all += v6_rows(seeds)
        count[f"seed:{os.path.basename(path)}"] = len(seeds)
    return [("v6", v6_all), ("v5", v5_all), ("new", new_all)], count, reuse


def cmd_build(a):
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    _, outs, code = _out(a, ("cands.jsonl", "summary.json"))
    if code:
        return code
    guard, missing = _guard(g4, old_root, a.no_old_forbid)
    k_paths = _k_paths(old_root)
    missing += [p for p in k_paths if not os.path.exists(p)]
    if not missing:
        try:
            stages, count, reuse = _stage_inputs(a, old_root, g4)
        except Stop as e:
            print(f"멈춤 : {e}", file=sys.stderr)
            return 2
        if stages is None:
            missing += reuse
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    eval_keys, eval_groups, eval_missing = old_eval_marks(old_root)
    guard = ReuseGuard(guard, reuse, eval_keys, eval_groups)
    rows, summary = build(stages, guard, _targets(a.targets), a.seed, _keys(_read_all(k_paths)))
    summary["old_eval_missing"] = eval_missing
    summary["inputs"] = count
    summary["recheck"] = recheck(rows, guard)
    _write(outs, a.force, rows, summary)
    print(json.dumps({k: summary[k] for k in ("got", "short", "units", "moved", "rows", "recheck")}, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, func in (("sample-v5", cmd_sample), ("v5-rate", cmd_rate), ("pick-new", cmd_pick), ("build", cmd_build),
                       ("pick-klb", cmd_pick_klb), ("build-klb", cmd_build_klb), ("thin-mrc-d", cmd_thin)):
        s = sub.add_parser(name)
        s.add_argument("--g4", default="~/.judgemodel/build/g4")
        s.add_argument("--old-root", default="~/.judgemodel")
        s.add_argument("--targets", help="TARGETS 를 덮을 json (일부 칸만 적어도 됨)")
        s.add_argument("--seed", type=int, default=SEED)
        s.add_argument("--force", action="store_true", help="출력이 있으면 옛 파일을 날짜 꼬리로 돌려 두고 쓴다")
        s.set_defaults(func=func)
    for name in ("sample-v5", "build"):
        sub.choices[name].add_argument("--v5", nargs="*", help="v5 실물 줄 파일. 기본 <old-root>/sets/g1/train-g1-II.jsonl")
    th = sub.choices["thin-mrc-d"]
    th.set_defaults(seed=THIN_SEED)
    th.add_argument("--layer", default="", help="얼린 train-g4-bal. 기본 <old-root>/sets/g4/train-g4-bal.jsonl")
    th.add_argument("--frozen", default="", help="그 FROZEN. 기본 <old-root>/sets/g4/FROZEN-train-g4-bal.sha256")
    th.add_argument("--klb", default="", help="klue-mrc b 묶음을 볼 층. 기본 <old-root>/sets/g4/train-g4-klb.jsonl")
    th.add_argument("--keep", type=int, default=120, help="남길 klue-mrc d 줄 수 (K6 설계 3-1)")
    kb = sub.choices["build-klb"]
    kb.set_defaults(seed=KLB_SEED)
    kb.add_argument("--cands", nargs="*", default=None,
                    help="판정 후보 파일. 기본 <g4>/train-g4-klb/pick-ka.jsonl · <g4>/klb/pick-mrc-rw.jsonl (평서문 바꾼 pick-mrc)")
    kb.add_argument("--dirs", nargs="*", default=None, help="판정 폴더 (decided.jsonl). 기본 <g4>/klb/judge")
    for name in ("pick-new", "build", "pick-klb", "build-klb"):
        sub.choices[name].add_argument("--no-old-forbid", action="store_true", help="G4Forbid · 옛 평가 문단 검사를 끈다 (시험용)")
    sub.choices["v5-rate"].add_argument("--dirs", nargs="*", default=None,
                                        help="표본 판정 폴더 (decided.jsonl). 기본 <g4>/judge/train-g4-bal-v5")
    for name in ("pick-new", "pick-klb"):
        sub.choices[name].add_argument("--input", nargs="*", help="원천 줄 바꾸기 (열쇠=경로 · g3ko · boolq · klue_mrc · "
                                         "klue_rw · klue_check · mit · mit_rev · mit_judged)")
    bd = sub.choices["build"]
    bd.add_argument("--v6", nargs="*", default=None, help="v6 재료 — 얼린 층 파일 또는 <decided.jsonl 또는 폴더>@<후보 파일>. "
                    "기본 sets/g2/train-g2-b · sets/g4/train-g4-c · train-g4-d")
    bd.add_argument("--v5-dirs", nargs="*", default=None, help="v5 다시 판정 폴더 (표본 · 칸 종류 재판정). 그 칸을 따른다")
    bd.add_argument("--v5-block", nargs="*", default=None,
                    help="원천:칸 — 다시 판정한 줄만 쓴다 (예 proposals:c). v5-rate.json 의 over 칸에 더한다")
    bd.add_argument("--no-v5-rate", dest="v5_rate", action="store_false",
                    help="<g4>/train-g4-bal/v5-rate.json 을 안 읽는다 (시험 · dry 용 — 실물 조립에서는 쓰지 않는다)")
    bd.add_argument("--dirs", nargs="*", default=None, help="새 판정 폴더 (pick-new 후보). 기본 <g4>/judge/train-g4-bal")
    bd.add_argument("--no-new", dest="new", action="store_false", default=None, help="새 판정 후보(pick.jsonl)를 안 쓴다")
    bd.add_argument("--seed-pairs", nargs="*", default=None, help="archive 씨앗 판정 줄 (seed_to_pairs judge 출력, keep True 만)")
    bd.add_argument("--draft", action="store_true", help="새 판정 전 줄을 세어서 빼고 조립")
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
