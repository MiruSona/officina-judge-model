"""관문 — 릴리스 관문(설계 6-2)과 짝 비교 관문(2-3)을 한 규약 파일에서 읽는다 · 옛 고리 관문(legacy).

    python -I gates.py release --protocol measure/protocol-g1.json --set <blind-g1.jsonl> \
        --pred runs/g1-B1-s2/eval-blind/pred.jsonl --calibration runs/g1-B1-s2/calibration.json \
        [--ref-pred <기준 모델 pred.jsonl>] [--ref-calibration <기준 모델 calibration.json>] [--p95-ms 414] \
        [--out gates.json] [--md gates.md]
    python -I gates.py pair <compare.py 인자 그대로>
    python -I gates.py legacy r2-02/diag.json r2-03/diag.json      # 옛 dev 관문 8 + 곁 3 (eval-side.txt)

calibration 이 잴 셋과 같은 셋으로 골라졌으면(calib_set_sha256 · 옛 fitted_on.sha256) 거부한다 — exit 2.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare as cmp  # noqa: E402
from compare import Refuse, cal  # noqa: E402

# 옛 넷칸 보강 고리의 관문 (넷칸보강설계 2절 · 곁 자 표). 옛 판을 다시 셀 때만 쓴다
LEGACY_DEV = [("acc3", ">=", 0.870), ("b_recall", ">=", 0.78), ("false_support_no_threshold", "<=", 14),
              ("sure_false_support", "<=", 1), ("unsure_rate", "<=", 0.28), ("d_recall", ">=", 0.68),
              ("sure_related_precision", ">=", 0.80), ("ab_mix", "<=", 12)]
LEGACY_SIDE = [("k48", "hits", 44), ("mnli-devm2000", "acc3", 0.856), ("klue-dev", "acc3", 0.868)]


def check_release_hashes(protocol, protocol_sha, set_sha, run, calibration):
    want = protocol.get("sets", {}).get(protocol["release_gate"].get("set", ""), "")
    if want and want != set_sha:
        raise Refuse(f"--set 해시 {set_sha[:12]} 가 규약의 {protocol['release_gate']['set']} 해시 {want[:12]} 와 다르다")
    s = run["summary"]
    if s is None:
        raise Refuse(f"{run['path']} 옆에 summary.json 이 없다")
    if s.get("set_sha256") != set_sha:
        raise Refuse(f"{run['path']} 의 set_sha256 ≠ 잴 셋 {set_sha[:12]}")
    if s.get("protocol_sha256") != protocol_sha:
        raise Refuse(f"{run['path']} 의 protocol_sha256 ≠ 규약 {protocol_sha[:12]}")
    if cmp.calib_sha(calibration) == set_sha:
        raise Refuse("calibration 이 잴 셋으로 골랐다 — 문턱·온도는 dev-cal 로만 고른다")
    if not calibration.get("thresholds"):
        raise Refuse("calibration 에 문턱(thresholds)이 없다 — --temperature-only 판으로는 릴리스 관문을 못 잰다")


def release_gate(protocol, set_rows, run, calibration, ref_run=None, p95_ms=None, ref_calibration=None):
    """릴리스 자 → 줄마다 (자, 값, 기준, 통과). 잴 수 없는 자는 통과 None.

    규약에 있는 칸만 잰다 — g1 은 절대 자(acc3_min · unsure_max · d_*), g2 는 vs_ref 목록 · unsure_vs_ref_max_diff.
    g3 는 b_recall_min · vs_ref 항목에 거름 칸(lang · domain)을 달 수 있다 — 자 이름 b_recall@ko · vs_ref_b_recall@ko.
    """
    gate = protocol["release_gate"]
    table = cmp.pair_hits([run], list(set_rows))
    th = calibration["thresholds"]
    scored = scored_rows(run, table["ids"], calibration)
    non_support = sum(1 for _, label in scored if label != cmp.SUPPORT)
    sure_fs = cal.false_support(scored, th["nli_sure_support"])
    rows = []
    if "acc3_min" in gate:
        rows.append(min_row(table, "acc3", gate["acc3_min"]))
    rows.extend(recall_min_rows(set_rows, table, gate["b_recall_min"]))
    rows.append(("sure_false_support", sure_fs, f"<= {non_support} × {gate['sure_false_support_share_max']}",
                 sure_fs <= non_support * gate["sure_false_support_share_max"] + 1e-9))
    rows.append(unsure_row(gate, unsure_of(scored, th), ref_run, ref_calibration, table["ids"]))
    if "d_recall_min" in gate:
        rows.append(min_row(table, "d_recall", gate["d_recall_min"]))
    if "d_sure_precision_min" in gate:
        d_prec = None
        if th.get("nli_sure_related") is not None:
            d_prec = cal.sure_related_precision(scored, th["nli_sure_related"])[0]
        rows.append(("d_sure_precision", d_prec, f">= {gate['d_sure_precision_min']}",
                     ge(d_prec, gate["d_sure_precision_min"])))
    rows.append(("p95_ms", p95_ms, f"<= {gate['p95_ms_max']}", None if p95_ms is None else p95_ms <= gate["p95_ms_max"]))
    rows.extend(vs_ref_rows(protocol, set_rows, table, ref_run))
    return [{"ruler": r[0], "value": round(r[1], 4) if isinstance(r[1], float) else r[1], "need": r[2], "ok": r[3]}
            for r in rows]


def scored_rows(run, ids, calibration):
    return [(cal.softmax(run["preds"][i]["logits"], calibration["temperature"]), run["preds"][i]["label"])
            for i in ids]


def unsure_of(scored, th):
    return cal.unsure_rate(scored, th["nli_sure"], th["nli_sure_support"], th.get("nli_sure_related"))


def min_row(table, metric, limit):
    value = cmp.rate(table, metric)
    return (metric, value, f">= {limit}", ge(value, limit))


def recall_min_specs(value):
    """b_recall_min 칸 → 절대 자 목록. 옛 꼴은 숫자 하나(쌍 전체 b 재현),
    g3 꼴은 {"metric": "b_recall", "lang": "ko", "min": 0.70} 하나나 그 목록 (숫자를 섞으면 쌍 전체 b 재현)."""
    items = value if isinstance(value, list) else [value]
    return [{"metric": "b_recall", "min": v} if isinstance(v, (int, float)) else v for v in items]


def recall_min_rows(set_rows, table, value):
    """절대 재현 자 줄 — 거름 칸이 있으면 걸러진 쌍만 센다. 자 이름은 b_recall · b_recall@ko.
    걸러진 쌍 중 셀 쌍이 0 이면 Refuse (compare.slice_compare 와 같다 — 규약 · 셋 칸이 어긋남)."""
    rows = []
    for spec in recall_min_specs(value):
        sub = table
        if cmp.is_slice(spec):
            sub = cmp.sub_table(table, cmp.slice_positions(set_rows, table["ids"], spec))
        _, rate, need, ok = min_row(sub, spec["metric"], spec["min"])
        if rate is None and cmp.is_slice(spec):
            raise Refuse(f"자 {cmp.ruler_name(spec)} 에 셀 쌍이 없다 — 셋의 {' · '.join(cmp.SLICE_KEYS)} 칸과 규약을 본다")
        rows.append((cmp.ruler_name(spec), rate, need, ok))
    return rows


def unsure_row(gate, unsure, ref_run, ref_calibration, ids):
    """보류 몫 — g1 은 절대 한도(unsure_max), g2 는 기준 모델 보류 + 차(unsure_vs_ref_max_diff, 기준 모델 제 문턱으로)."""
    if "unsure_max" in gate:
        return ("unsure", round(unsure, 4), f"<= {gate['unsure_max']}", unsure <= gate["unsure_max"] + 1e-9)
    diff = gate["unsure_vs_ref_max_diff"]
    if ref_run is None or ref_calibration is None or not ref_calibration.get("thresholds"):
        return ("unsure", round(unsure, 4), f"<= ref 보류 + {diff} (ref calibration 없음)", None)
    ref_unsure = unsure_of(scored_rows(ref_run, ids, ref_calibration), ref_calibration["thresholds"])
    return ("unsure", round(unsure, 4), f"<= {ref_unsure:.4f} + {diff}", unsure <= ref_unsure + diff + 1e-9)


def vs_ref_specs(gate):
    """견줌 자 목록 — g2 는 vs_ref 칸 그대로, g1 은 옛 칸 vs_ref_acc3_ci_low_ge 하나를 같은 꼴로."""
    if "vs_ref" in gate:
        return gate["vs_ref"]
    if "vs_ref_acc3_ci_low_ge" in gate:
        return [{"metric": "acc3", "ci_low_ge": gate["vs_ref_acc3_ci_low_ge"]}]
    return []


def vs_ref_rows(protocol, set_rows, table, ref_run):
    """기준 모델 견줌 — 지표마다 같은 쌍 차의 묶음 붓스트랩 아래끝 · McNemar(보고)."""
    specs = vs_ref_specs(protocol["release_gate"])
    ref = None
    groups = None
    if ref_run is not None and specs:
        ref = cmp.pair_hits([ref_run], table["ids"])
        cmp.check_same_pairs(ref, table, "기준 모델 pred 와 잴 판")
        key = protocol.get("group_key", "group")
        groups = [str(set_rows[pid].get(key) or pid) for pid in table["ids"]]
    boot = protocol.get("bootstrap", {})
    rows = []
    for spec in specs:
        metric = spec["metric"]
        name = f"vs_ref_{cmp.ruler_name(spec)}"
        if "ci_low_gt" in spec:
            need = f"차 95% 아래끝 > {spec['ci_low_gt']}"
        else:
            need = f"차 95% 아래끝 >= {spec['ci_low_ge']}"
        if ref is None:
            rows.append((name, None, need, None))
            continue
        if cmp.is_slice(spec):
            # g3 꼴 {"metric": "b_recall", "lang": "ko", …} — 걸러진 쌍만으로 같은 셈 (짝 비교와 같은 함수)
            s = cmp.slice_compare(protocol, set_rows, ref, table, spec)
        else:
            s = cmp.cluster_bootstrap(groups, ref, table, metric, boot.get("n", 10000), boot.get("ci", 0.95),
                                      boot.get("rng_seed", 20261011))
            s["mcnemar"] = cmp.mcnemar_exact(cmp.majority(ref, metric), cmp.majority(table, metric))
        if s["low"] is None:
            ok = None
        elif "ci_low_gt" in spec:
            ok = s["low"] > spec["ci_low_gt"]
        else:
            ok = s["low"] >= spec["ci_low_ge"] - 1e-9
        rows.append((name, s, need, ok))
    return rows


def ge(value, limit):
    if value is None:
        return None
    return value >= limit - 1e-9


def release_word(rows):
    if any(r["ok"] is False for r in rows):
        return "미달"
    if any(r["ok"] is None for r in rows):
        return "잴 수 없는 자 있음"
    return "통과"


def legacy_counts(diag, side):
    """옛 고리 관문 — dev 8 (diag.json) + 곁 3 (eval-side.txt, 없으면 None)."""
    flat = dict(diag)
    flat.update(diag["calibrated"])
    dev = [cmpop(flat[k], op, v) for k, op, v in LEGACY_DEV]
    side_ok = None
    if side is not None:
        side_ok = []
        for name, kind, limit in LEGACY_SIDE:
            entry = side.get(name)
            value = None
            if entry is not None:
                value = round(entry["acc3"] * entry["n"]) if kind == "hits" else entry["acc3"]
            side_ok.append(value is not None and value >= limit)
    return dev, side_ok


def cmpop(value, op, limit):
    if op == ">=":
        return value >= limit
    return value <= limit


def run_legacy(paths):
    for path in paths:
        diag = cmp.read_json(path)
        side_path = cmp.find_beside(path, ["eval-side.txt"])
        dev, side = legacy_counts(diag, cmp.read_side(side_path) if side_path else None)
        line = f"{path} {diag['score']} {sum(dev)}/8 {''.join('O' if x else 'X' for x in dev)}"
        if side is not None:
            line += f" + 곁 {sum(side)}/3 {''.join('O' if x else 'X' for x in side)}"
        print(line)


def run_release(argv):
    p = argparse.ArgumentParser(description="릴리스 관문 (설계 6-2)")
    p.add_argument("--protocol", required=True)
    p.add_argument("--set", required=True, help="잴 셋 jsonl (blind-g1)")
    p.add_argument("--pred", required=True, help="릴리스 판의 eval_nli pred.jsonl (또는 그 폴더)")
    p.add_argument("--calibration", required=True, help="dev-cal 로 고른 calibration.json")
    p.add_argument("--ref-pred", default="", help="기준 모델(규약 ref_model)의 같은 셋 pred.jsonl (견줌 자)")
    p.add_argument("--ref-calibration", default="",
                   help="기준 모델의 calibration.json (g2 보류 자). 비면 --ref-pred 옆 · 윗 폴더의 것")
    p.add_argument("--p95-ms", type=float, default=None)
    p.add_argument("--out", default="")
    p.add_argument("--md", default="")
    a = p.parse_args(argv)
    protocol, protocol_sha = cmp.read_protocol(a.protocol)
    set_sha = cmp.sha256_file(a.set)
    set_rows = cmp.read_set(a.set)
    run = cmp.load_run(a.pred)
    calibration = cmp.read_json(a.calibration)
    check_release_hashes(protocol, protocol_sha, set_sha, run, calibration)
    ref_run = None
    ref_calibration = None
    if a.ref_pred:
        ref_run = cmp.load_run(a.ref_pred)
        ref_calibration = ref_run["calibration"]
    if a.ref_calibration:
        ref_calibration = cmp.read_json(a.ref_calibration)
    if ref_calibration is not None and cmp.calib_sha(ref_calibration) == set_sha:
        raise Refuse("기준 모델 calibration 이 잴 셋으로 골랐다 — 문턱 · 온도는 dev-cal 로만 고른다")
    rows = release_gate(protocol, set_rows, run, calibration, ref_run, a.p95_ms, ref_calibration)
    result = {"protocol_sha256": protocol_sha, "set_sha256": set_sha, "pred": run["path"],
              "calibration": os.path.abspath(a.calibration), "verdict": release_word(rows), "rulers": rows}
    lines = [f"## 릴리스 관문 : **{result['verdict']}**", "", "| 자 | 값 | 기준 | 통과 |", "| --- | --- | --- | --- |"]
    for r in rows:
        value = r["value"]
        if isinstance(value, dict):
            value = f"{value['diff']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"
        mark = "?" if r["ok"] is None else ("O" if r["ok"] else "X")
        lines.append(f"| {r['ruler']} | {value} | {r['need']} | {mark} |")
    md = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
            f.write("\n")
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
    print(md)
    return result


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("release", "pair", "legacy"):
        print("사용: gates.py release|pair|legacy …", file=sys.stderr)
        return 1
    try:
        if argv[0] == "pair":
            cmp.run(argv[1:])
        elif argv[0] == "legacy":
            run_legacy(argv[1:])
        else:
            run_release(argv[1:])
    except Refuse as e:
        print(f"거부: {e}", file=sys.stderr)
        return cmp.REFUSE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
