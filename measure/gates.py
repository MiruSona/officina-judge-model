"""관문 — 릴리스 관문(설계 6-2)과 짝 비교 관문(2-3)을 한 규약 파일에서 읽는다 · 옛 고리 관문(legacy).

    python -I gates.py release --protocol measure/protocol-g1.json --set <blind-g1.jsonl> \
        --pred runs/g1-B1-s2/eval-blind/pred.jsonl --calibration runs/g1-B1-s2/calibration.json \
        [--ref-pred <v1.0.0 pred.jsonl>] [--p95-ms 414] [--out gates.json] [--md gates.md]
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


def release_gate(protocol, set_rows, run, calibration, ref_run=None, p95_ms=None):
    """릴리스 자 여덟 → 줄마다 (자, 값, 기준, 통과). 잴 수 없는 자는 통과 None."""
    gate = protocol["release_gate"]
    ids = list(set_rows)
    table = cmp.pair_hits([run], ids)
    th = calibration["thresholds"]
    scored = [(cal.softmax(run["preds"][i]["logits"], calibration["temperature"]), run["preds"][i]["label"])
              for i in table["ids"]]
    non_support = sum(1 for _, label in scored if label != cmp.SUPPORT)
    sure_fs = cal.false_support(scored, th["nli_sure_support"])
    unsure = cal.unsure_rate(scored, th["nli_sure"], th["nli_sure_support"], th.get("nli_sure_related"))
    d_prec = None
    if th.get("nli_sure_related") is not None:
        d_prec = cal.sure_related_precision(scored, th["nli_sure_related"])[0]
    rows = [
        ("acc3", cmp.rate(table, "acc3"), f">= {gate['acc3_min']}", ge(cmp.rate(table, "acc3"), gate["acc3_min"])),
        ("b_recall", cmp.rate(table, "b_recall"), f">= {gate['b_recall_min']}", ge(cmp.rate(table, "b_recall"), gate["b_recall_min"])),
        ("sure_false_support", sure_fs, f"<= {non_support} × {gate['sure_false_support_share_max']}",
         sure_fs <= non_support * gate["sure_false_support_share_max"] + 1e-9),
        ("unsure", round(unsure, 4), f"<= {gate['unsure_max']}", unsure <= gate["unsure_max"] + 1e-9),
        ("d_recall", cmp.rate(table, "d_recall"), f">= {gate['d_recall_min']}", ge(cmp.rate(table, "d_recall"), gate["d_recall_min"])),
        ("d_sure_precision", d_prec, f">= {gate['d_sure_precision_min']}", ge(d_prec, gate["d_sure_precision_min"])),
        ("p95_ms", p95_ms, f"<= {gate['p95_ms_max']}", None if p95_ms is None else p95_ms <= gate["p95_ms_max"]),
    ]
    rows.append(vs_ref_row(protocol, set_rows, table, run, ref_run))
    return [{"ruler": r[0], "value": round(r[1], 4) if isinstance(r[1], float) else r[1], "need": r[2], "ok": r[3]}
            for r in rows]


def vs_ref_row(protocol, set_rows, table, run, ref_run):
    """v1.0.0 견줌 — 같은 쌍 3칸 정확도 차의 묶음 붓스트랩 아래끝 · McNemar(보고)."""
    need = f"차 95% 아래끝 >= {protocol['release_gate']['vs_ref_acc3_ci_low_ge']}"
    if ref_run is None:
        return ("vs_ref_acc3", None, need, None)
    ref = cmp.pair_hits([ref_run], table["ids"])
    cmp.check_same_pairs(ref, table, "기준 모델 pred 와 잴 판")
    key = protocol.get("group_key", "group")
    groups = [str(set_rows[pid].get(key) or pid) for pid in table["ids"]]
    boot = protocol.get("bootstrap", {})
    s = cmp.cluster_bootstrap(groups, ref, table, "acc3", boot.get("n", 10000), boot.get("ci", 0.95),
                              boot.get("rng_seed", 20261011))
    s["mcnemar"] = cmp.mcnemar_exact(cmp.majority(ref, "acc3"), cmp.majority(table, "acc3"))
    return ("vs_ref_acc3", s, need, s["low"] >= protocol["release_gate"]["vs_ref_acc3_ci_low_ge"] - 1e-9)


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
    p.add_argument("--ref-pred", default="", help="v1.0.0 의 같은 셋 pred.jsonl (견줌 자)")
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
    ref_run = cmp.load_run(a.ref_pred) if a.ref_pred else None
    rows = release_gate(protocol, set_rows, run, calibration, ref_run, a.p95_ms)
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
