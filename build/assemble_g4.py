"""assemble_g4 — v1.2 K2 판 g4 평가 셋 둘 조립 · 얼림 · 검사(min_groups) · 규약 해시 채우기.

설계 `../Docs/Design/2026-10-14-v1.2-K2-한국어c보강판설계.md` 3-2 · 3-4 · 3-5 · 5절 ⑤. 쌍 본문은 화면에 찍지 않는다.

    python -I assemble_g4.py assemble --set dev-cmp-g4|blind-g4 [--dest ~/.judgemodel/sets/g4] [--draft] \
        [--build ~/.judgemodel/build/g4] [--g3-dir ~/.judgemodel/sets/g3] [--dirs <판정 폴더> …]
    python -I assemble_g4.py freeze --set <셋> [--dest …]
    python -I assemble_g4.py check [--dest …] [--build …] [--g3-dir …] [--protocol ../measure/protocol-g4.draft.json] [--draft] \
        [--recipe <학습 파일 …>] [--thin 10]   (K4 원천 × 칸 빈칸 검사 · 새 층 겹침)
    python -I assemble_g4.py cells --recipe <학습 파일 …> --dev <dev-cmp-g4.jsonl> [--thin 10] [--draft]
    check · cells 공통 점검표(K5 설계 3절) : --base-recipe <K 처방 …> [--prev-recipe <앞 판 …>] [--ack ../measure/cell-ack-g7.json] [--json]
    python -I assemble_g4.py protocol --draft ../measure/protocol-g4.draft.json --dest … --out ../measure/protocol-g4.json
    python -I assemble_g4.py protocol --draft ../measure/protocol-g5.draft.json --dest … --out ../measure/protocol-g5.json   (K3)
    python -I assemble_g4.py protocol --draft ../measure/protocol-g7.draft.json --out ../measure/protocol-g7.json   (K5)

셋 = 이을 g3 셋(dev-cmp-g4 ← dev-cmp-g3 · blind-g4 는 안 잇는다) + 예약 후보 `<build>/reserve/<셋>-cands.jsonl` 중 확정 칸이 있는 줄.
확정 칸 · 새 줄 꼴 · 중복 규칙은 assemble_g3 와 같다. 꼬리표만 g4 다 — 새 줄 `g4_label: v6-new`,
이은 g3 줄은 칸 · 라벨 기준을 그대로 두고 `set` 만 새 이름으로, 옛 이름은 `g3_set`, `g4_label: g3-carry`.
check 는 규약 초안 release_gate.min_groups 를 읽어 그 셋(blind-g4)의 칸별 묶음 수가 모자라면 exit 1 — 얼리기를 막는다.
check --recipe 와 cells 는 학습 전체(K 처방 + 새 층)와 dev 를 (원천, 칸)으로 세어, 학습 0 · dev ≥ 1 칸(zero)이 있으면
exit 1 — 얼리기를 막는다(원천 균형 판 설계 4절). 10줄 미만(thin)은 stderr 경고만, dev 0 칸(dev_missing)은 보고만.
check --recipe 는 recipe 파일 중 --train 에 없는 층(새 층 train-g4-bal 같은 것)도 겹침 셈(overlap)에 넣는다.
단 K 처방 기본 층(BASE_RECIPE_LAYERS : train-r2 · train-g1-arch · train-g3-ko)은 빼고 overlap_skipped_base 에 적는다.
recipe · train 경로는 <build> 기준 상대 경로 또는 절대 경로다.
--base-recipe 를 주면 T1(원천 안 칸 몫을 K · 없으면 dev 와) · T2(언어 × 겹침 구간 칸 몫을 dev 와, K 부터 울린 칸은 기준선) ·
T3(dev 30쌍 미만 칸 · 막지 않음)을 돌려 표(--json 이면 JSON)를 찍는다. 허용 안 된 T1 · 새 T2 는 exit 1, 금지 허용(T1 b 내림) · 파일 없음은 exit 2.
protocol 은 초안의 빈 "" 해시 자리(sets 의 g4 셋 둘 · train_sets 의 TRAIN_SETS 중 초안에 있는 것)만 채운다.
이미 해시가 적힌 자리(protocol-g5 초안의 g4 셋 · train-g4-c 같은 것)는 얼린 파일과 같은지만 본다.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemble_g3 import (Gen, Stop, assemble, cmd_freeze, freeze_tool, load_decisions, overlap_counts,  # noqa: E402
                         scrub_hits, sha256_file, table)
from label_batches import inside_git, read_jsonl, write_json, write_jsonl  # noqa: E402
from layer_g4d import bin_of, overlap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.join(os.path.expanduser("~"), ".judgemodel")
G4_SETS = ("dev-cmp-g4", "blind-g4")
G4 = Gen("g4_label", "g3_set", "g3-carry", {"dev-cmp-g4": "dev-cmp-g3", "blind-g4": None}, "g3")
G3_SIDE_SETS = ("dev-cal-g3", "blind-g3")         # 겹침만 같이 본다 (dev-cal-g3 는 g4 규약에서도 쓴다)
TRAIN_SETS = ("train-g4-c", "train-g4-c-h", "train-g4-d", "train-g4-bal", "train-g4-klb")   # train/freeze.py 로 얼린 학습 층 — 규약 train_sets 자리
DRAFT = os.path.join(HERE, "..", "measure", "protocol-g4.draft.json")
CELL_OF = {"support": "a", "contradict": "b", "unrelated": "c", "related": "d", "a": "a", "b": "b", "c": "c", "d": "d"}
THIN = 10
MAX_SHIFT = 0.20            # T1 · T2 칸 몫 차 문턱 (K5 설계 3-2 · 3-3)
BAND_EDGES = (0.1, 0.2, 0.35)
BAND_MIN = 100              # T2 — 학습 쪽 구간이 이 줄 수 미만이면 안 울린다
WEAK_MIN = 30               # T3 — dev (원천, 칸) 쌍이 이 수 미만이면 판정 약함
# K 처방 기본 층 — check --recipe 겹침 셈에서 뺀다. g1 학습 묶음은 평가에서 문서째 막지 않는다는
# g2 때 결정(measure/sets-g2.md)이라, 이 층과 평가 셋의 묶음 겹침은 새로 막을 것이 아니다.
BASE_RECIPE_LAYERS = ("train-r2", "train-g1-arch", "train-g3-ko")


def cell_key(row):
    """줄 → (원천, 칸 글자). 원천은 domain(없으면 mem), 칸은 want 의 낱말 · 글자를 a~d 로 맞춘다. cd 와 그 밖은 None."""
    cell = CELL_OF.get(str(row.get("want") or ""))
    if cell is None:
        return None
    return str(row.get("domain") or "mem"), cell


def _cell_counts(rows):
    counts = collections.Counter()
    for row in rows:
        key = cell_key(row)
        if key:
            counts[key] += 1
    return counts


def cell_gaps(train_rows, dev_rows, thin=THIN):
    """학습 줄 · dev 줄을 (원천, 칸)으로 센다 → {zero, thin_cells, dev_missing, table}.
    zero = 학습 0 · dev ≥ 1 (얼리기 막음) · thin_cells = 1 ≤ 학습 < thin · dev ≥ 1 (경고) · dev_missing = 학습 ≥ 1 · dev 0 (보고).
    table 은 원천마다 칸별 [학습, dev] — 결과 문서에 옮길 표다. 쌍 본문은 안 담는다."""
    train, dev = _cell_counts(train_rows), _cell_counts(dev_rows)
    out = {"zero": [], "thin_cells": [], "dev_missing": [], "table": {}}
    for domain, cell in sorted(set(train) | set(dev)):
        n_train, n_dev = train[(domain, cell)], dev[(domain, cell)]
        if n_dev and not n_train:
            out["zero"].append({"domain": domain, "cell": cell, "train": 0, "dev": n_dev})
        elif n_dev and n_train < thin:
            out["thin_cells"].append({"domain": domain, "cell": cell, "train": n_train, "dev": n_dev})
        elif n_train and not n_dev:
            out["dev_missing"].append({"domain": domain, "cell": cell, "train": n_train})
    for domain in sorted({d for d, _ in set(train) | set(dev)}):
        out["table"][domain] = {c: [train[(domain, c)], dev[(domain, c)]] for c in "abcd"}
    return out


def read_recipe(build, paths):
    """학습 파일 목록(<build> 기준 상대 또는 절대) → (줄, 빠진 파일)."""
    rows, missing = [], []
    for rel in paths:
        path = os.path.join(build, rel)
        if not os.path.exists(path):
            missing.append(path)
            continue
        rows.extend(read_jsonl(path))
    return rows, missing


def cells_report(recipe, dev_name, train_rows, dev_rows, thin):
    return {"recipe": [os.path.basename(p) for p in recipe], "dev": dev_name, "thin": thin,
            **cell_gaps(train_rows, dev_rows, thin)}


# ---------------------------------------------------------------- 학습 전 점검표 T1~T3 (K5 설계 3절)


def cell_shares(rows):
    """줄 → ({원천: {칸: 몫}}, {원천: 줄 수}). 몫 = 그 칸 줄 ÷ 그 원천 줄. 원천에 없는 칸도 0 으로 채운다."""
    counts = _cell_counts(rows)
    totals = collections.Counter()
    for (domain, _), n in counts.items():
        totals[domain] += n
    shares = {d: {c: counts[(d, c)] / totals[d] for c in "abcd"} for d in totals}
    return shares, totals


def _dir(diff):
    if diff > 0:
        return "up"
    return "down"


def cell_drift(recipe_rows, base_rows, dev_rows, prev_rows=None, max_shift=MAX_SHIFT):
    """T1 — 원천마다 칸 몫을 기준 처방(K) 몫과, K 에 없는 원천은 dev 몫과 견준다. |차| > max_shift 면 flags.
    prev_rows(앞 판) 몫과의 차는 보고만 한다. 기준 · dev 어디에도 없는 원천은 ref 없음으로 적기만."""
    shares, _ = cell_shares(recipe_rows)
    counts = _cell_counts(recipe_rows)
    base, _ = cell_shares(base_rows)
    dev, _ = cell_shares(dev_rows)
    prev = cell_shares(prev_rows)[0] if prev_rows is not None else {}
    rows, flags = [], []
    for domain in sorted(shares):
        if domain in base:
            ref, kind = base[domain], "K"
        elif domain in dev:
            ref, kind = dev[domain], "dev"
        else:
            ref, kind = None, None
        for cell in "abcd":
            share = shares[domain][cell]
            line = {"domain": domain, "cell": cell, "n": counts[(domain, cell)], "share": round(share, 4),
                    "ref_kind": kind, "ref": None, "diff": None, "prev": None, "prev_diff": None}
            if ref is not None:
                line["ref"] = round(ref[cell], 4)
                line["diff"] = round(share - ref[cell], 4)
            if domain in prev:
                line["prev"] = round(prev[domain][cell], 4)
                line["prev_diff"] = round(share - prev[domain][cell], 4)
            rows.append(line)
            if line["diff"] is not None and abs(line["diff"]) > max_shift:
                flags.append({"domain": domain, "cell": cell, "dir": _dir(line["diff"]), "diff": line["diff"],
                              "share": line["share"], "ref": line["ref"], "ref_kind": kind})
    return {"max_shift": max_shift, "rows": rows, "flags": flags}


def _band_counts(rows, edges):
    """줄 → (Counter[(언어, 구간, 칸)], 칸은 있으나 언어가 ko · en 이 아니라 버린 줄 수). 겹침은 layer_g4d.overlap."""
    counts, dropped = collections.Counter(), 0
    for row in rows:
        key = cell_key(row)
        lang = str(row.get("lang") or "")
        if key is None:
            continue
        if lang not in ("ko", "en"):
            dropped += 1
            continue
        counts[(lang, bin_of(overlap(row.get("evidence"), row.get("claim")), edges), key[1])] += 1
    return counts, dropped


def _band_rows(train, dev, edges, max_shift, min_n):
    """(언어, 구간)마다 학습 칸 몫 − dev 칸 몫. 학습 구간 min_n 줄 이상 · |차| > max_shift 면 flag."""
    rows = []
    for lang in ("ko", "en"):
        for band in range(len(edges) + 1):
            n_train = sum(train[(lang, band, c)] for c in "abcd")
            n_dev = sum(dev[(lang, band, c)] for c in "abcd")
            if not n_train:
                continue
            if not n_dev:            # dev 에 그 구간이 없으면 견줄 몫이 없다 — 왜 안 울렸는지 줄로 남긴다
                rows.append({"lang": lang, "band": band, "cell": None, "n_band": n_train, "dev_band": 0,
                             "share": None, "dev": None, "diff": None, "flag": False, "note": "dev 없음"})
                continue
            for cell in "abcd":
                share, ref = train[(lang, band, cell)] / n_train, dev[(lang, band, cell)] / n_dev
                diff = round(share - ref, 4)
                rows.append({"lang": lang, "band": band, "cell": cell, "n_band": n_train, "dev_band": n_dev,
                             "share": round(share, 4), "dev": round(ref, 4), "diff": diff,
                             "flag": n_train >= min_n and abs(diff) > max_shift})
    return rows


def band_drift(recipe_rows, dev_rows, base_rows, edges=BAND_EDGES, max_shift=MAX_SHIFT, min_n=BAND_MIN):
    """T2 — 언어 × 겹침 구간 안 칸 몫을 학습과 dev 에서 따로 세어 견준다.
    기준 처방(K)에서도 같은 칸 · 같은 방향으로 울리면 baseline(「기준선부터」 · 보고만), 아니면 flags(새 경고)."""
    dev, dev_dropped = _band_counts(dev_rows, edges)
    train, train_dropped = _band_counts(recipe_rows, edges)
    rows = _band_rows(train, dev, edges, max_shift, min_n)
    base_hits = {(r["lang"], r["band"], r["cell"], _dir(r["diff"]))
                 for r in _band_rows(_band_counts(base_rows, edges)[0], dev, edges, max_shift, min_n) if r["flag"]}
    flags, baseline = [], []
    for r in rows:
        if not r["flag"]:
            continue
        hit = {"lang": r["lang"], "band": r["band"], "cell": r["cell"], "dir": _dir(r["diff"]), "diff": r["diff"],
               "share": r["share"], "dev": r["dev"], "n_band": r["n_band"]}
        if (r["lang"], r["band"], r["cell"], hit["dir"]) in base_hits:
            baseline.append(hit)
        else:
            flags.append(hit)
    return {"edges": list(edges), "max_shift": max_shift, "min_n": min_n, "rows": rows, "flags": flags,
            "baseline": baseline, "dropped_lang": {"train": train_dropped, "dev": dev_dropped}}


def protocol_guards(path):
    """규약 pair_gate.guards 중 언어로 나눈 칸 재현 지킴 → [(칸 글자, 언어)]. b_recall@ko → ("b", "ko").
    규약이 없거나 JSON 이 깨졌거나 꼴(dict · guards 는 dict 목록)이 틀리면 Stop."""
    doc = _read_json(path, "규약")
    gate = doc.get("pair_gate", {}) if isinstance(doc, dict) else None
    guards = gate.get("guards", []) if isinstance(gate, dict) else None
    if not isinstance(guards, list) or not all(isinstance(g, dict) for g in guards):
        raise Stop(f"규약 꼴이 틀렸다 {os.path.basename(path)} — dict 이고 pair_gate.guards 는 dict 목록")
    out = []
    for guard in guards:
        metric, lang = str(guard.get("metric") or ""), guard.get("lang")
        if lang and metric.endswith("_recall") and metric[0] in "abcd" and len(metric) == len("a_recall"):
            out.append((metric[0], str(lang)))
    return out


def weak_cells(dev_rows, guards, min_dev=WEAK_MIN):
    """T3 — dev (원천, 칸) 쌍이 min_dev 미만이면 「판정 약함」. 지킴(칸, 언어)마다 약한 칸에 든 쌍 몫과
    「쌍 하나 = 지킴 몇 %p」를 적는다. 막지 않는다."""
    counts = _cell_counts(dev_rows)
    cells = [{"domain": d, "cell": c, "dev": n} for (d, c), n in sorted(counts.items()) if n < min_dev]
    weak = {(c["domain"], c["cell"]) for c in cells}
    out = []
    for letter, lang in guards:
        keys = [cell_key(r) for r in dev_rows if str(r.get("lang") or "") == lang]
        mine = [k for k in keys if k is not None and k[1] == letter]
        in_weak = sum(1 for k in mine if k in weak)
        out.append({"guard": f"{letter}@{lang}", "pairs": len(mine), "weak_pairs": in_weak,
                    "weak_share": round(in_weak / len(mine), 4) if mine else None,
                    "pp_per_pair": round(100 / len(mine), 3) if mine else None})
    return {"min_dev": min_dev, "cells": cells, "guards": out}


DEFAULT_GUARDS = (("b", "ko"), ("b", "en"))   # 규약에 언어 지킴이 없을 때 T3 지킴 — protocol-g6 과 같다
ACK_FIELDS = {"T1": ("domain", "cell", "dir"), "T2": ("lang", "band", "cell", "dir")}   # 허용 줄을 맞출 칸
ACK_COMMON = ("tool", "why")      # 늘 있어야 하는 칸
ACK_OPTIONAL = ("design",)


def _read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        raise Stop(f"{what} 파일을 못 읽는다 {path}") from None
    except (ValueError, UnicodeDecodeError) as e:
        raise Stop(f"{what} JSON 이 깨졌다 {os.path.basename(path)} ({type(e).__name__})") from None


def _ack_value_error(entry):
    """허용 줄 값 검사 — 틀린 까닭 글 하나, 맞으면 None. 모르는 값은 조용히 넘기지 않는다."""
    if entry["cell"] not in ("a", "b", "c", "d"):
        return "cell 은 a · b · c · d"
    if entry["dir"] not in ("up", "down"):
        return "dir 은 up · down"
    if not isinstance(entry["why"], str):
        return "why 는 글"
    if entry["tool"] == "T1" and not isinstance(entry["domain"], str):
        return "domain 은 글"
    if entry["tool"] == "T2" and entry["lang"] not in ("ko", "en"):
        return "lang 은 ko · en"
    band = entry.get("band")
    if entry["tool"] == "T2" and (type(band) is not int or not 0 <= band <= len(BAND_EDGES)):
        return f"band 는 0 ~ {len(BAND_EDGES)} 정수"
    return None


def load_ack(path):
    """허용 목록 json → 줄 목록. 꼴 · 값이 틀리거나 모르는 칸이 있거나 T1 b 내림을 허용하려 하면 Stop."""
    doc = _read_json(path, "허용 목록")
    if not isinstance(doc, list):
        raise Stop("허용 목록은 줄 목록이어야 한다")
    for i, entry in enumerate(doc):
        tool = entry.get("tool") if isinstance(entry, dict) else None
        if not isinstance(tool, str) or tool not in ACK_FIELDS:
            raise Stop(f"허용 목록 {i} 번 줄 : tool 은 T1 · T2 중 하나")
        need = ACK_FIELDS[tool] + ACK_COMMON
        unknown = set(entry) - set(need) - set(ACK_OPTIONAL)
        missing = [k for k in need if entry.get(k) in (None, "")]
        if unknown or missing:
            raise Stop(f"허용 목록 {i} 번 줄 : 모르는 칸 {sorted(unknown)} · 빈 칸 {missing}")
        bad = _ack_value_error(entry)
        if bad:
            raise Stop(f"허용 목록 {i} 번 줄 : {bad}")
        if tool == "T1" and entry["cell"] == "b" and entry["dir"] == "down":
            raise Stop(f"허용 목록 {i} 번 줄 : T1 b 칸 내림은 허용할 수 없다 (b 지킴 · K5 설계 3-1)")
    return doc


def split_ack(flags, acks, tool):
    """경고 → (허용됨, 허용 안 됨). 허용 줄과 칸 · 방향이 다 같아야 허용이다."""
    fields = ACK_FIELDS[tool]
    keys = {tuple(str(a[k]) for k in fields) for a in acks if a.get("tool") == tool}
    acked = [f for f in flags if tuple(str(f[k]) for k in fields) in keys]
    unacked = [f for f in flags if tuple(str(f[k]) for k in fields) not in keys]
    return acked, unacked


def cell_checks(recipe_rows, dev_rows, inputs, max_shift=MAX_SHIFT, band_min=BAND_MIN, weak_min=WEAK_MIN):
    """T1 · T2 · T3 을 다 돌려 결과 칸 셋(drift · bands · weak)을 낸다. 각 칸에 acked · unacked 를 붙인다.
    inputs 는 checks_inputs 가 낸 dict (base_rows · prev_rows · acks · guards)."""
    drift = cell_drift(recipe_rows, inputs["base_rows"], dev_rows, inputs["prev_rows"], max_shift)
    drift["acked"], drift["unacked"] = split_ack(drift["flags"], inputs["acks"], "T1")
    bands = band_drift(recipe_rows, dev_rows, inputs["base_rows"], BAND_EDGES, max_shift, band_min)
    bands["acked"], bands["unacked"] = split_ack(bands["flags"], inputs["acks"], "T2")
    weak = weak_cells(dev_rows, inputs["guards"], weak_min)
    weak["acked"], weak["unacked"] = [], []
    return {"drift": drift, "bands": bands, "weak": weak,
            "unused_ack": unused_ack(inputs["acks"], drift["flags"], bands["flags"])}


def unused_ack(acks, t1_flags, t2_flags):
    """허용 목록 중 어느 경고와도 안 맞은 줄 (막지 않음 · 보고만). 낡은 허용이 쌓이지 않게 보인다."""
    flags = {"T1": t1_flags, "T2": t2_flags}
    out = []
    for entry in acks:
        fields = ACK_FIELDS[entry["tool"]]
        want = tuple(str(entry[k]) for k in fields)
        if not any(tuple(str(f[k]) for k in fields) == want for f in flags[entry["tool"]]):
            out.append({k: entry[k] for k in ("tool",) + fields})
    return out


def apply_checks(result, checks):
    """점검표 결과를 result 에 붙인다 — drift · bands · weak 는 맨 위, unused_ack 는 cell_check 안."""
    result.update({k: checks[k] for k in ("drift", "bands", "weak")})
    result["cell_check"]["unused_ack"] = checks["unused_ack"]


def checks_table(checks):
    """T1~T3 결과 → 사람이 읽는 마크다운 표 (경고 · 허용 · 기준선 줄만, 쌍 본문 없음)."""
    drift, bands, weak = checks["drift"], checks["bands"], checks["weak"]
    lines = [f"## T1 원천 안 칸 몫 이동 (|차| > {drift['max_shift']})", "",
             "| 원천 | 칸 | 몫 | 기준 | 기준 몫 | 차 | 앞 판 차 | 판정 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    acked = {(f["domain"], f["cell"]) for f in drift["acked"]}
    for f in drift["flags"]:
        row = next(r for r in drift["rows"] if (r["domain"], r["cell"]) == (f["domain"], f["cell"]))
        word = "허용" if (f["domain"], f["cell"]) in acked else "**경고**"
        lines.append(f"| {f['domain']} | {f['cell']} | {f['share']:.3f} | {f['ref_kind']} | {f['ref']:.3f} "
                     f"| {f['diff']:+.3f} | {_signed(row['prev_diff'])} | {word} |")
    flagged = {(f["domain"], f["cell"]) for f in drift["flags"]}
    for r in drift["rows"]:
        big_prev = r["prev_diff"] is not None and abs(r["prev_diff"]) > drift["max_shift"]
        if big_prev and (r["domain"], r["cell"]) not in flagged:
            lines.append(f"| {r['domain']} | {r['cell']} | {r['share']:.3f} | {r['ref_kind'] or '—'} "
                         f"| {_signed(r['ref'], sign=False)} | {_signed(r['diff'])} | {r['prev_diff']:+.3f} | 앞 판 차 큼 (보고) |")
    for domain in sorted({r["domain"] for r in drift["rows"] if r["ref_kind"] is None}):
        lines.append(f"| {domain} | | | — | | | | 기준 없음 (K · dev 에 없는 원천) |")
    if len(lines) == 4:
        lines.append("| — | | | | | | | 없음 |")
    lines += ["", f"## T2 겹침 구간별 칸 몫 (구간 {bands['edges']} · |차| > {bands['max_shift']} · 구간 {bands['min_n']}줄 이상)",
              "", "| 언어 | 구간 | 칸 | 학습 몫 | dev 몫 | 차 | 구간 줄 | 판정 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    marks = [(f, "**새 경고**") for f in bands["unacked"]] + [(f, "허용") for f in bands["acked"]] + \
            [(f, "기준선부터") for f in bands["baseline"]]
    for f, word in marks:
        lines.append(f"| {f['lang']} | {f['band']} | {f['cell']} | {f['share']:.3f} | {f['dev']:.3f} | {f['diff']:+.3f} "
                     f"| {f['n_band']} | {word} |")
    if not marks:
        lines.append("| — | | | | | | | 없음 |")
    lines += ["", f"## T3 작은 지킴 칸 (dev {weak['min_dev']}쌍 미만 = 판정 약함 · 막지 않음)", "",
              "| 원천 | 칸 | dev 쌍 |", "| --- | --- | --- |"]
    lines += [f"| {c['domain']} | {c['cell']} | {c['dev']} |" for c in weak["cells"]] or ["| — | | 없음 |"]
    lines += ["", "| 지킴 | 쌍 | 약한 칸 쌍 | 몫 | 쌍 하나 = %p |", "| --- | --- | --- | --- | --- |"]
    for g in weak["guards"]:
        lines.append(f"| {g['guard']} | {g['pairs']} | {g['weak_pairs']} | {_signed(g['weak_share'], sign=False)} "
                     f"| {_signed(g['pp_per_pair'], sign=False)} |")
    lines += ["", "## 안 쓴 허용 줄 (어느 경고와도 안 맞음 · 막지 않음)", "", "| 도구 | 칸 |", "| --- | --- |"]
    lines += [f"| {u['tool']} | {_ack_text(u)} |" for u in checks.get("unused_ack", [])] or ["| — | 없음 |"]
    return "\n".join(lines) + "\n"


def _ack_text(entry):
    return " ".join(str(entry[k]) for k in ACK_FIELDS[entry["tool"]])


def _signed(value, sign=True):
    if value is None:
        return "—"
    if sign:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def warn_lines(checks):
    """stderr 에 찍을 경고 한 줄씩 (허용 안 된 T1 · 새 T2 · 허용 · 기준선은 머리말로 갈라 적는다)."""
    out = []
    for f in checks["drift"]["unacked"]:
        out.append(f"경고 T1 : {f['domain']} {f['cell']} {f['diff']:+.3f} ({f['ref_kind']} {f['ref']:.3f} → {f['share']:.3f})")
    for f in checks["drift"]["acked"]:
        out.append(f"허용 T1 : {f['domain']} {f['cell']} {f['diff']:+.3f}")
    for f in checks["bands"]["unacked"]:
        out.append(f"경고 T2 : {f['lang']} 구간 {f['band']} {f['cell']} {f['diff']:+.3f} (학습 {f['n_band']}줄)")
    for f in checks["bands"]["acked"]:
        out.append(f"허용 T2 : {f['lang']} 구간 {f['band']} {f['cell']} {f['diff']:+.3f}")
    for f in checks["bands"]["baseline"]:
        out.append(f"기준선부터 T2 : {f['lang']} 구간 {f['band']} {f['cell']} {f['diff']:+.3f}")
    for domain in sorted({r["domain"] for r in checks["drift"]["rows"] if r["ref_kind"] is None}):
        out.append(f"기준 없음 T1 : {domain} (K · dev 에 없는 원천 — 몫 이동을 못 잰다)")
    for u in checks.get("unused_ack", []):
        out.append(f"안 쓴 허용 : {u['tool']} {_ack_text(u)}")
    for g in checks["weak"]["guards"]:
        if g["weak_pairs"]:
            out.append(f"약함 T3 : {g['guard']} {g['pairs']}쌍 중 {g['weak_pairs']} 이 dev {checks['weak']['min_dev']}쌍 미만 칸")
    return out


def checks_inputs(a, build):
    """--base-recipe · --prev-recipe · --ack · 규약 지킴을 읽는다 → dict(base_rows · prev_rows · acks · ack_info · guards).
    파일이 없거나 허용 목록 · 규약이 틀리면 Stop."""
    base_rows, missing = read_recipe(build, a.base_recipe)
    prev_rows = None
    if a.prev_recipe:
        prev_rows, more = read_recipe(build, a.prev_recipe)
        missing += more
    if a.ack and not os.path.exists(a.ack):
        missing.append(a.ack)
    if missing:
        raise Stop(f"점검표 파일이 없다 {missing}")
    acks, ack_info = [], None
    if a.ack:
        acks = load_ack(a.ack)
        ack_info = {"file": os.path.basename(a.ack), "sha256": sha256_file(a.ack), "entries": len(acks)}
    guards = []
    if a.protocol:
        guards = protocol_guards(a.protocol)
    if not guards:
        print("(T3 : 규약에 언어 지킴이 없어 b@ko · b@en 으로 본다)", file=sys.stderr)
        guards = list(DEFAULT_GUARDS)
    return {"base_rows": base_rows, "prev_rows": prev_rows, "acks": acks, "ack_info": ack_info, "guards": guards}


def checks_args_error(a):
    """점검표 문턱이 양수가 아니거나, 점검표 인자만 주고 --base-recipe 를 빠뜨렸으면 그 까닭 글, 아니면 None."""
    bad = [name for name, value in (("--max-shift", a.max_shift), ("--band-min", a.band_min), ("--weak-min", a.weak_min))
           if not value > 0]
    if bad:
        return f"{' · '.join(bad)} 는 양수여야 한다"
    if a.base_recipe is not None:
        return None
    given = [name for name, value in (("--ack", a.ack), ("--prev-recipe", a.prev_recipe), ("--json", a.json)) if value]
    if given:
        return f"{' · '.join(given)} 는 --base-recipe 와 같이 준다"
    return None


def checks_meta(a, inputs):
    return {"base_recipe": [os.path.basename(p) for p in a.base_recipe],
            "prev_recipe": [os.path.basename(p) for p in a.prev_recipe or []], "ack": inputs["ack_info"],
            "guards": [f"{c}@{lang}" for c, lang in inputs["guards"]],
            "max_shift": a.max_shift, "band_edges": list(BAND_EDGES), "band_min": a.band_min, "weak": a.weak_min}


def checks_bad(checks):
    """얼리기를 막는 수 = 허용 안 된 T1 + 새(기준선 아님 · 허용 안 됨) T2. T3 은 안 막는다."""
    return len(checks["drift"]["unacked"]) + len(checks["bands"]["unacked"])


def group_counts(min_groups, rows, group_key="group"):
    """min_groups 자 이름(칸 글자 · 꼬리 @lang) → 그 칸(want) 줄의 묶음 수. compare.group_counts 와 같은 셈."""
    out = {}
    for name in min_groups:
        letter, _, lang = name.partition("@")
        out[name] = len({str(r.get(group_key) or r["id"]) for r in rows
                         if r.get("want") == letter and (not lang or str(r.get("lang")) == lang)})
    return out


def read_gate(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc["release_gate"], doc.get("group_key", "group")


def pending_quota(build, set_name):
    """reserve-g4.json 의 pending — 판정 파일이 아직 없어 줄도 없는 원천 몫(blind-g4 ko b)."""
    path = os.path.join(build, "reserve", "reserve-g4.json")
    if set_name != "blind-g4" or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("pending", {})


def cmd_assemble(a):
    if inside_git(a.dest):
        sys.exit("--dest 가 git 안이다 — 셋 본문은 git 밖에 둔다")
    dirs = a.dirs if a.dirs is not None else [os.path.join(a.build, "eval-public", a.set)]
    cands = read_jsonl(os.path.join(a.build, "reserve", f"{a.set}-cands.jsonl"))
    base = G4.base[a.set]
    g3_rows = read_jsonl(os.path.join(a.g3_dir, f"{base}.jsonl")) if base else []
    try:
        rows, summary = assemble(a.set, cands, g3_rows, load_decisions(dirs), a.draft, G4)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    if base:
        summary["g3_sha256"] = sha256_file(os.path.join(a.g3_dir, f"{base}.jsonl"))
    summary["pending_quota"] = pending_quota(a.build, a.set)
    summary["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    os.makedirs(a.dest, exist_ok=True)
    path = os.path.join(a.dest, f"{a.set}.jsonl")
    write_jsonl(path, rows)
    summary["sha256"] = sha256_file(path)
    write_json(os.path.join(a.dest, f"{a.set}.summary.json"), summary)
    keys = ("set", "draft", "pairs", "g3_pairs", "new_pairs", "want", "ko_b", "pending", "pending_quota",
            "dropped", "deduped", "sha256")
    print(json.dumps({k: (len(summary[k]) if k == "deduped" else summary[k]) for k in keys}, ensure_ascii=False))
    return 0


def _norm_path(build, rel):
    return os.path.normcase(os.path.normpath(os.path.join(build, rel)))


def _stem(rel):
    """경로 → 확장자 뺀 파일 이름 (train-r2.jsonl → train-r2)."""
    return os.path.splitext(os.path.basename(rel))[0]


def cmd_check(a):
    sets = {}
    paths = {n: os.path.join(a.dest, f"{n}.jsonl") for n in G4_SETS}
    paths.update({n: os.path.join(a.g3_dir, f"{n}.jsonl") for n in G3_SIDE_SETS})
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"멈춤 : 셋이 없다 {path}", file=sys.stderr)
            return 2
        sets[name] = read_jsonl(path)
    train_rows = []
    for rel in a.train:
        path = os.path.join(a.build, rel)
        if os.path.exists(path):
            train_rows.extend(read_jsonl(path))
            continue
        if not a.draft:
            print(f"멈춤 : 학습 층이 없어 겹침을 못 잰다 {path} (초안 검사는 --draft)", file=sys.stderr)
            return 2
        print(f"(학습 층 없음 : {rel})", file=sys.stderr)
    recipe_rows, extra, skipped, checks_in = None, [], [], None
    if a.base_recipe is not None and a.recipe is None:
        print("멈춤 : --base-recipe 는 --recipe 와 같이 준다", file=sys.stderr)
        return 2
    if checks_args_error(a):
        print(f"멈춤 : {checks_args_error(a)}", file=sys.stderr)
        return 2
    if a.base_recipe is not None:
        try:
            checks_in = checks_inputs(a, a.build)
        except Stop as e:
            print(f"멈춤 : {e}", file=sys.stderr)
            return 2
    if a.recipe is not None:
        recipe_rows, missing = read_recipe(a.build, a.recipe)
        if missing:
            print(f"멈춤 : 빈칸 검사 학습 파일이 없다 {missing}", file=sys.stderr)
            return 2
        known = {_norm_path(a.build, rel) for rel in a.train}
        outside = [rel for rel in a.recipe if _norm_path(a.build, rel) not in known]
        skipped = [rel for rel in outside if _stem(rel) in BASE_RECIPE_LAYERS]
        extra = [rel for rel in outside if _stem(rel) not in BASE_RECIPE_LAYERS]
        for rel in extra:                      # --train 밖의 새 층(train-g4-bal 같은 것)은 평가 셋과 겹치지 않아야 한다
            train_rows.extend(read_jsonl(os.path.join(a.build, rel)))
    gate, group_key = read_gate(a.protocol)
    want = gate.get("min_groups") or {}
    have = group_counts(want, sets.get(gate.get("set"), []), group_key)
    short = {name: [have[name], need] for name, need in want.items() if have[name] < need}
    overlap = overlap_counts(sets, train_rows)
    result = {"overlap": overlap, "train_rows": len(train_rows), "sets": {},
              "min_groups": {"set": gate.get("set"), "have": have, "need": want, "short": short}}
    if recipe_rows is not None:
        result["overlap_extra_recipe"] = [os.path.basename(r) for r in extra]
        result["overlap_skipped_base"] = [os.path.basename(r) for r in skipped]
    bad = sum(overlap.values())
    for name in G4_SETS:
        rows = sets[name]
        hits = scrub_hits(rows, a.names, a.game_names)
        bad += hits
        result["sets"][name] = {"pairs": len(rows), "scrub_hits": hits,
                                "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
                                "lang_want": table(rows, "lang"), "domain_want": table(rows, "domain")}
    if recipe_rows is not None:
        result["cells"] = cells_report(a.recipe, "dev-cmp-g4.jsonl", recipe_rows, sets["dev-cmp-g4"], a.thin)
        if not a.draft:
            bad += len(result["cells"]["zero"])
    if checks_in is not None:
        result["cell_check"] = checks_meta(a, checks_in)
        checks = cell_checks(recipe_rows, sets["dev-cmp-g4"], checks_in, a.max_shift, a.band_min, a.weak_min)
        apply_checks(result, checks)
        if not a.draft:
            bad += checks_bad(checks)
        print_checks(result, checks, a.json, head=check_head(result))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    if recipe_rows is not None and result["cells"]["thin_cells"]:
        print(f"경고 : 학습 {a.thin}줄 미만 칸 {len(result['cells']['thin_cells'])}", file=sys.stderr)
    if bad:
        return 1
    if short and not a.draft:
        return 1
    return 0


def cmd_cells(a):
    """빈칸 검사만 홀로 — 학습 파일 목록 × dev. zero 가 있으면 exit 1 (--draft 면 적기만)."""
    train_rows, missing = read_recipe(a.build, a.recipe)
    if not os.path.exists(a.dev):
        missing.append(a.dev)
    if missing:
        print(f"멈춤 : 파일이 없다 {missing}", file=sys.stderr)
        return 2
    checks_in = None
    if checks_args_error(a):
        print(f"멈춤 : {checks_args_error(a)}", file=sys.stderr)
        return 2
    if a.base_recipe is not None:
        try:
            checks_in = checks_inputs(a, a.build)
        except Stop as e:
            print(f"멈춤 : {e}", file=sys.stderr)
            return 2
    dev_rows = read_jsonl(a.dev)
    report = cells_report(a.recipe, os.path.basename(a.dev), train_rows, dev_rows, a.thin)
    bad = len(report["zero"])
    if checks_in is None:
        print(json.dumps({"cells": report}, ensure_ascii=False, indent=1))
    else:
        result = {"cells": report, "cell_check": checks_meta(a, checks_in)}
        checks = cell_checks(train_rows, dev_rows, checks_in, a.max_shift, a.band_min, a.weak_min)
        apply_checks(result, checks)
        bad += checks_bad(checks)
        print_checks(result, checks, a.json, head=cells_head(report))
    if report["thin_cells"]:
        print(f"경고 : 학습 {a.thin}줄 미만 칸 {len(report['thin_cells'])}", file=sys.stderr)
    return 1 if bad and not a.draft else 0


def print_checks(result, checks, as_json, head):
    """점검표 결과를 찍는다 — --json 이면 결과 JSON 통째, 아니면 머리말 + T1~T3 표. 경고는 stderr 에 한 줄씩."""
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        print(head + "\n" + checks_table(checks), end="")
    for line in warn_lines(checks):
        print(line, file=sys.stderr)


def cells_head(report):
    return (f"# 점검표 — 학습 {', '.join(report['recipe'])} · dev {report['dev']}\n\n"
            f"- 2-1 빈칸 zero {len(report['zero'])} · thin {len(report['thin_cells'])} · dev 0 칸 {len(report['dev_missing'])}\n")


def check_head(result):
    short = result["min_groups"]["short"]
    hits = sum(s["scrub_hits"] for s in result["sets"].values())
    return (cells_head(result["cells"]) + f"- 겹침 {sum(result['overlap'].values())} · scrub {hits}"
            f" · min_groups 모자람 {len(short)}\n")


def fill_protocol(draft_text, hashes):
    """초안 바이트에서 `"<이름>": ""` 자리만 sha256 으로 바꾼다(그 밖 글자는 그대로). 자리가 한 번이 아니거나 빈 칸이 남으면 Stop."""
    out = draft_text
    for name, digest in hashes.items():
        token = f'"{name}": ""'
        if out.count(token) != 1:
            raise Stop(f"초안에 {token} 가 한 번 있어야 한다 ({out.count(token)})")
        out = out.replace(token, f'"{name}": "{digest}"')
    doc = json.loads(out)
    left = [k for part in ("sets", "train_sets") for k, v in doc.get(part, {}).items() if not v or "TODO" in str(v)]
    if left:
        raise Stop(f"빈 해시가 남았다 {left}")
    filled = {**doc.get("sets", {}), **doc.get("train_sets", {})}
    if {k: filled.get(k) for k in hashes} != hashes:
        raise Stop("채운 해시가 얼린 파일과 다르다")
    return out


def frozen_hash(dest, name):
    """FROZEN-<이름>.sha256 과 맞는 파일의 sha256. 안 얼렸거나 다르면 Stop."""
    pairs = os.path.join(dest, f"{name}.jsonl")
    frozen = os.path.join(dest, f"FROZEN-{name}.sha256")
    if not os.path.exists(frozen):
        raise Stop(f"얼리지 않은 파일 {name}")
    code, out = freeze_tool("check", "--pairs", pairs, "--frozen", frozen)
    if code:
        raise Stop(f"{name} 해시가 FROZEN 과 다르다 {out}")
    return sha256_file(pairs)


def slots_to_fill(doc, dest):
    """초안의 g4 셋 · 학습 층 자리 중 빈 "" 자리 → {이름: 얼린 sha256}. 해시가 적힌 자리는 얼린 파일과 다르면 Stop, 초안에 없는 이름은 건너뛴다."""
    written = {**doc.get("sets", {}), **doc.get("train_sets", {})}
    hashes = {}
    for name in G4_SETS + TRAIN_SETS:
        if name not in written:
            continue
        digest = frozen_hash(dest, name)
        if written[name] == "":
            hashes[name] = digest
        elif written[name] != digest:
            raise Stop(f"초안의 {name} 해시가 얼린 파일과 다르다")
    return hashes


def cmd_protocol(a):
    if os.path.exists(a.out):
        print(f"멈춤 : {a.out} 이 이미 있다 — 규약은 첫 비교 뒤 안 고친다(hash_rule)", file=sys.stderr)
        return 2
    with open(a.draft_protocol, encoding="utf-8", newline="") as f:
        text = f.read()
    try:
        hashes = slots_to_fill(json.loads(text), a.dest)
        cal = json.loads(text)["sets"].get("dev-cal-g3")
        if cal != frozen_hash(a.g3_dir, "dev-cal-g3"):
            raise Stop("초안의 dev-cal-g3 해시가 얼린 dev-cal-g3 와 다르다")
        out = fill_protocol(text, hashes)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(json.dumps({"out": a.out, "hashes": hashes, "protocol_sha256": sha256_file(a.out)}, ensure_ascii=False))
    return 0


def add_check_args(s):
    """T1~T3 점검표 인자 — --base-recipe 를 줄 때만 돈다 (안 주면 옛 출력 그대로)."""
    s.add_argument("--base-recipe", nargs="+", default=None,
                   help="T1 · T2 기준 처방 (K = train-r2 · train-g1-arch · train-g3-ko). 주면 T1~T3 을 돈다")
    s.add_argument("--prev-recipe", nargs="+", default=None, help="앞 판 처방 — T1 몫 차를 보고만")
    s.add_argument("--ack", default="", help="허용 목록 json (measure/cell-ack-g7.json)")
    s.add_argument("--max-shift", type=float, default=MAX_SHIFT)
    s.add_argument("--band-min", type=int, default=BAND_MIN)
    s.add_argument("--weak-min", type=int, default=WEAK_MIN)
    s.add_argument("--json", action="store_true", help="점검표를 표 대신 결과 JSON 으로")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="g4 평가 셋 조립 · 얼림 · 검사")
    sub = p.add_subparsers(dest="cmd", required=True)
    dest = os.path.join(HOME, "sets", "g4")
    build = os.path.join(HOME, "build", "g4")
    g3_dir = os.path.join(HOME, "sets", "g3")
    s = sub.add_parser("assemble")
    s.add_argument("--set", required=True, choices=G4_SETS)
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--g3-dir", default=g3_dir)
    s.add_argument("--dirs", nargs="*", default=None, help="판정 폴더 (decided.jsonl). 기본 <build>/eval-public/<셋>")
    s.add_argument("--draft", action="store_true", help="판정 전 줄을 세어서 빼고 조립 (얼리지 않는다)")
    s = sub.add_parser("freeze")
    s.add_argument("--set", required=True, choices=G4_SETS)
    s.add_argument("--dest", default=dest)
    s = sub.add_parser("check")
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--g3-dir", default=g3_dir)
    s.add_argument("--protocol", default=DRAFT, help="min_groups 를 읽을 규약 (기본 measure/protocol-g4.draft.json)")
    s.add_argument("--train", nargs="*", default=["train-g4-c/cands.jsonl", "train-g4-c/cands-h.jsonl",
                                                  "train-g4-d/cands.jsonl", os.path.join(g3_dir, "train-g3-ko.jsonl")],
                   help="학습 층 (<build> 기준 상대 경로 또는 절대 경로)")
    s.add_argument("--names", nargs="*", default=[os.path.join(HOME, "build", "g1", "names.txt")])
    s.add_argument("--game-names", nargs="*", default=[os.path.join(HOME, "build", "g1", "game-names.txt")])
    s.add_argument("--draft", action="store_true", help="min_groups 모자람 · 학습 층 없음 · 빈칸을 실패로 안 본다")
    s.add_argument("--recipe", nargs="*", default=None,
                   help="빈칸 검사할 학습 전체 (K 처방 + 새 층, <build> 기준 상대 또는 절대). 안 주면 옛 동작 그대로")
    s.add_argument("--thin", type=int, default=THIN, help="이 수 미만 학습 칸은 경고 (기본 10)")
    add_check_args(s)
    s = sub.add_parser("cells")
    s.add_argument("--build", default=build)
    s.add_argument("--recipe", nargs="+", required=True, help="학습 파일 (<build> 기준 상대 또는 절대)")
    s.add_argument("--dev", default=os.path.join(dest, "dev-cmp-g4.jsonl"))
    s.add_argument("--thin", type=int, default=THIN)
    s.add_argument("--draft", action="store_true", help="빈칸 · T1 · T2 경고를 적기만 하고 exit 0")
    s.add_argument("--protocol", default="", help="T3 지킴(pair_gate.guards 의 lang 줄)을 읽을 규약. 비면 b@ko · b@en")
    add_check_args(s)
    s = sub.add_parser("protocol")
    s.add_argument("--draft", dest="draft_protocol", required=True, help="protocol-g4.draft.json · protocol-g5.draft.json")
    s.add_argument("--dest", default=dest)
    s.add_argument("--g3-dir", default=g3_dir, help="dev-cal-g3 · FROZEN-dev-cal-g3.sha256 이 있는 곳")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"assemble": cmd_assemble, "freeze": cmd_freeze, "check": cmd_check, "cells": cmd_cells,
            "protocol": cmd_protocol}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
