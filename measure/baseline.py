"""baseline.py — 기준선 한 조건(씨앗 여럿 + 같은 씨앗 재실행)의 자 상태를 잰다 (d 자료 층 설계 2-3 · 8절 위험 1).

    python -I measure/baseline.py --protocol measure/protocol-g1.json --set <dev-cmp-g1.jsonl> \
        --runs B0=<판1>/eval-dev-cmp,<판2>/…,<판3>/… --rerun <판1r>/eval-dev-cmp --out b.json --md b.md

짝 비교(compare.py)와 달리 「새 조건」이 없다. 대신 아래 다섯을 낸다.
  ① 판마다 지표 — 3칸 정확도 · 칸별 재현 · 문턱 없는 거짓 지지 · 확신 거짓 지지 · 보류율 · d 확신 정밀 (문턱은 판 옆 calibration.json)
  ② 자가 3%p 를 가리나 — 씨앗 짝(1v1) 붓스트랩 반폭과, 그걸 씨앗 평균으로 줄인 어림(÷√씨앗 수), 수준 반폭(보수)
  ③ 같은 씨앗 재실행 차 — 3칸 차가 씨앗 사이 폭의 절반을 넘으면 결정 모드를 따진다 (설계 2-3)
  ④ 곁 자 — 판 옆 eval-side.txt
  ⑤ 언어별 b · c 재현 — 셋 줄에 lang 칸이 있으면 ko · en 마다 씨앗별 값 · 씨앗 평균 [95%] · 짝 반폭 어림 (c 는 g4 판별 문턱).
     ko · en 밖 언어(mix 등)는 빠지고, 빠진 언어와 쌍 수를 요약(lang_b_skipped)에 적는다
릴리스 pick 은 규약 release_pick 을 통째로 compare.pick_release 에 넘긴다 — lang 거름(g3 의 ko b)도 compare 와 같게 센다.
규약 어긋남(셋 · 규약 해시 · calibration 이 잴 셋으로 골라짐)은 compare.py 와 같이 exit 2.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare as cmp  # noqa: E402

cal = cmp.cal
TARGET_HALF_WIDTH = 0.03  # 설계 8절 위험 1 — 짝 구간 반폭 ≤ 3%p
LANGS = ("ko", "en")  # ⑤ 언어별 b 줄을 낼 언어 (이 차례로)


def zero_table(table):
    """같은 정답 · 맞힘 0 표 — cluster_bootstrap(…, zero, table) 의 차가 곧 수준(level)이라 수준 구간을 그대로 얻는다."""
    return {"ids": table["ids"], "gold": table["gold"], "counts": [[0, 0, 0, 0] for _ in table["gold"]],
            "seeds": table["seeds"]}


def run_metrics(run, ids):
    """판 하나의 지표 — 칸별 재현 · 거짓 지지 둘 · 보류율 · d 확신 정밀 · 곁 자."""
    table = cmp.pair_hits([run], ids)
    out = {"path": run["path"], "seed": run["seed"]}
    for metric in cmp.METRICS:
        value = cmp.rate(table, metric)
        out[metric] = None if value is None else round(value, 4)
    out["false_support"] = cmp.false_support_mean(table)
    out["non_support"] = sum(1 for g in table["gold"] if g != cmp.SUPPORT)
    c = run["calibration"]
    th = (c or {}).get("thresholds")
    if th:
        scored = [(cal.softmax(run["preds"][i]["logits"], c["temperature"]), run["preds"][i]["label"])
                  for i in table["ids"]]
        out["temperature"] = c["temperature"]
        out["sure_false_support"] = cal.false_support(scored, th["nli_sure_support"])
        out["unsure"] = round(cal.unsure_rate(scored, th["nli_sure"], th["nli_sure_support"],
                                              th.get("nli_sure_related")), 4)
        prec, n = (None, 0)
        if th.get("nli_sure_related") is not None:
            prec, n = cal.sure_related_precision(scored, th["nli_sure_related"])
        out["d_sure_precision"] = None if prec is None else round(prec, 4)
        out["d_sure_n"] = n
    out["side"] = run["side"]
    return out


def resolve(protocol, set_rows, runs, rerun):
    """② · ③ 의 셈. 돌려줌 {seed_pairs, level, estimate, rerun}."""
    ids = list(set_rows)
    key = protocol.get("group_key", "group")
    boot = protocol.get("bootstrap", {})
    n_boot, ci, rng = boot.get("n", 10000), boot.get("ci", 0.95), boot.get("rng_seed", 20261011)
    tables = [cmp.pair_hits([r], ids) for r in runs]
    groups = [str(set_rows[pid].get(key) or pid) for pid in tables[0]["ids"]]
    metrics = ("d_recall", "acc3", "b_recall")
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            line = {"pair": f"s{runs[i]['seed']} vs s{runs[j]['seed']}"}
            for m in metrics:
                s = cmp.cluster_bootstrap(groups, tables[i], tables[j], m, n_boot, ci, rng)
                line[m] = {"diff": s["diff"], "low": s["low"], "high": s["high"],
                           "half": round((s["high"] - s["low"]) / 2, 4)}
            pairs.append(line)
    all_table = cmp.pair_hits(runs, ids)
    level, estimate = {}, {}
    for m in metrics:
        s = cmp.cluster_bootstrap(groups, zero_table(all_table), all_table, m, n_boot, ci, rng)
        level[m] = {"mean": s["new"], "low": s["low"], "high": s["high"], "half": round((s["high"] - s["low"]) / 2, 4),
                    "pairs": s["pairs"], "groups": s["groups"]}
        one = sum(p[m]["half"] for p in pairs) / len(pairs) if pairs else None
        estimate[m] = None if one is None else {"one_vs_one": round(one, 4),
                                                 "seed_mean": round(one / math.sqrt(len(runs)), 4)}
    out = {"seed_pairs": pairs, "level": level, "estimate": estimate, "groups": len(set(groups)),
           "seeds": len(runs)}
    d = estimate["d_recall"]
    if d is not None:
        half = d["seed_mean"]
        n_d = level["d_recall"]["pairs"]
        out["verdict"] = {"metric": "d_recall", "half": half, "limit": TARGET_HALF_WIDTH, "ok": half <= TARGET_HALF_WIDTH,
                          "d_pairs": n_d,
                          # 반폭은 √쌍 수에 반비례 — 3%p 로 낮추려면 d 쌍이 (반폭/0.03)² 배 있어야 한다
                          "d_pairs_needed": math.ceil(n_d * (half / TARGET_HALF_WIDTH) ** 2)}
    if rerun is not None:
        same_seed = [r for r in runs if r["seed"] is not None and r["seed"] == rerun["seed"]]
        if not same_seed:
            raise cmp.Refuse(f"재실행 판 {rerun['path']} 의 씨앗 {rerun['seed']} 이 --runs 에 없다 — 폴더 이름의 -s숫자를 본다")
        base = same_seed[0]
        t_base, t_re = cmp.pair_hits([base], ids), cmp.pair_hits([rerun], ids)
        cmp.check_same_pairs(t_base, t_re, "재실행 판과 기준 판")
        acc3 = [cmp.rate(t, "acc3") for t in tables]
        spread = max(acc3) - min(acc3)
        line = {"pair": f"s{base['seed']} vs s{rerun['seed']}r", "seed_acc3_spread": round(spread, 4)}
        for m in metrics:
            s = cmp.cluster_bootstrap(groups, t_base, t_re, m, n_boot, ci, rng)
            line[m] = {"diff": s["diff"], "low": s["low"], "high": s["high"]}
            line[m]["mcnemar"] = cmp.mcnemar_exact(cmp.majority(t_base, m), cmp.majority(t_re, m))
        flips = sum(1 for a, b in zip(t_base["counts"], t_re["counts"]) if a != b)
        line["pred_changed"] = flips
        line["acc3_diff_over_half_spread"] = abs(line["acc3"]["diff"]) > spread / 2 + 1e-9
        out["rerun"] = line
    return out


def lang_b(protocol, set_rows, runs, metric="b_recall"):
    """⑤ 언어별 재현(기본 b, g4 부터 c 도). 셋 줄에 lang 칸이 하나도 없으면 None, 그 언어 그 칸 쌍이 없으면 그 줄은 뺀다.

    resolve 를 lang 으로 거른 쌍에 돌린 것과 같은 셈이다 (g3 기준선 때 쓴 스크래치 lang_b_g3.py 를 옮김) —
    씨앗별 값 · 수준 구간 · 씨앗 짝(1v1) 붓스트랩 반폭 평균 · 그걸 ÷√씨앗 수 한 어림.
    """
    if not any(row.get("lang") for row in set_rows.values()):
        return None
    ids = list(set_rows)
    key = protocol.get("group_key", "group")
    boot = protocol.get("bootstrap", {})
    n_boot, ci, rng = boot.get("n", 10000), boot.get("ci", 0.95), boot.get("rng_seed", 20261011)
    tables = [cmp.pair_hits([r], ids) for r in runs]
    all_table = cmp.pair_hits(runs, ids)
    out = {}
    x = metric[0]
    for lang in LANGS:
        positions = cmp.slice_positions(set_rows, all_table["ids"], {"metric": metric, "lang": lang})
        subs = [cmp.sub_table(t, positions) for t in tables]
        sub_all = cmp.sub_table(all_table, positions)
        groups = [str(set_rows[pid].get(key) or pid) for pid in sub_all["ids"]]
        level = cmp.cluster_bootstrap(groups, zero_table(sub_all), sub_all, metric, n_boot, ci, rng)
        if level["pairs"] == 0:
            continue
        halves = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                s = cmp.cluster_bootstrap(groups, subs[i], subs[j], metric, n_boot, ci, rng)
                halves.append(round((s["high"] - s["low"]) / 2, 4))
        one = sum(halves) / len(halves) if halves else None
        out[lang] = {"pairs": len(positions), f"{x}_pairs": level["pairs"], "groups": level["groups"],
                     f"{x}_per_seed": [round(cmp.rate(t, metric), 4) for t in subs],
                     f"{x}_level": {"mean": level["new"], "low": level["low"], "high": level["high"],
                                    "half": round((level["high"] - level["low"]) / 2, 4)},
                     f"{x}_estimate": None if one is None else {"one_vs_one": round(one, 4),
                                                                "seed_mean": round(one / math.sqrt(len(runs)), 4)}}
    return out


def lang_b_skipped(set_rows):
    """⑤ 에서 빠지는 언어(LANGS 밖) → 쌍 수. lang 칸이 하나도 없으면 None. lang 이 빈 줄은 「-」로 센다."""
    if not any(row.get("lang") for row in set_rows.values()):
        return None
    out = {}
    for row in set_rows.values():
        lang = row.get("lang") or "-"
        if lang not in LANGS:
            out[lang] = out.get(lang, 0) + 1
    return dict(sorted(out.items()))


def to_markdown(result):
    lines = [f"## 기준선 {result['name']} — 셋 {result['set_sha256'][:12]} · 규약 {result['protocol_sha256'][:12]}", "",
             "| 판 | acc3 | a | b | c | d | d→a | a→d | 거짓 지지 | 확신 거짓 지지 | 보류율 | d 확신 정밀 (수) | T |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["runs"]:
        lines.append(f"| s{r['seed']}{'r' if r.get('rerun') else ''} | {r['acc3']} | {r['a_recall']} | {r['b_recall']} "
                     f"| {r['c_recall']} | {r['d_recall']} | {r['d_to_a']} | {r['a_to_d']} | {r['false_support']:g} "
                     f"| {r.get('sure_false_support', '—')} / {r['non_support']} | {r.get('unsure', '—')} "
                     f"| {r.get('d_sure_precision', '—')} ({r.get('d_sure_n', 0)}) | {r.get('temperature', '—')} |")
    s = result["resolve"]
    lines += ["", "| 씨앗 짝 | d 차 [95%] 반폭 | acc3 차 [95%] 반폭 | b 차 [95%] 반폭 |", "| --- | --- | --- | --- |"]
    for p in s["seed_pairs"]:
        cells = [f"{p[m]['diff']:+.4f} [{p[m]['low']:+.4f}, {p[m]['high']:+.4f}] {p[m]['half']:.4f}"
                 for m in ("d_recall", "acc3", "b_recall")]
        lines.append(f"| {p['pair']} | " + " | ".join(cells) + " |")
    lines += ["", "| 지표 | 씨앗 평균 [95%] | 수준 반폭 | 1v1 반폭 평균 | 씨앗 평균 짝 반폭 어림 |", "| --- | --- | --- | --- | --- |"]
    for m, lv in s["level"].items():
        e = s["estimate"][m] or {"one_vs_one": "—", "seed_mean": "—"}
        lines.append(f"| {m} | {lv['mean']} [{lv['low']}, {lv['high']}] | {lv['half']} | {e['one_vs_one']} | {e['seed_mean']} |")
    v = s.get("verdict")
    if v:
        lines += ["", f"- {v['metric']} 씨앗 평균 짝 반폭 어림 {v['half']:.4f} {'≤' if v['ok'] else '>'} {v['limit']} → "
                  f"{'가린다' if v['ok'] else '못 가린다'} · 그 칸 쌍 {v['d_pairs']} (3%p 에 필요 어림 {v['d_pairs_needed']})"]
    r = s.get("rerun")
    if r:
        lines += ["", f"- 같은 씨앗 재실행 {r['pair']} : acc3 차 {r['acc3']['diff']:+.4f} [{r['acc3']['low']:+.4f}, "
                  f"{r['acc3']['high']:+.4f}] · d 차 {r['d_recall']['diff']:+.4f} · 예측 바뀐 쌍 {r['pred_changed']} · "
                  f"씨앗 사이 acc3 폭 {r['seed_acc3_spread']:.4f} → 절반 넘나 {r['acc3_diff_over_half_spread']}"]
    for c, langs in (("b", result.get("lang_b")), ("c", result.get("lang_c"))):
        if not langs:
            continue
        lines += ["", f"| 언어 {c} | {c} 쌍 | 씨앗별 | 씨앗 평균 [95%] | 수준 반폭 | 1v1 반폭 평균 | 씨앗 평균 짝 반폭 어림 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for lang, x in langs.items():
            lv, e = x[f"{c}_level"], x[f"{c}_estimate"] or {"one_vs_one": "—", "seed_mean": "—"}
            lines.append(f"| {lang} | {x[f'{c}_pairs']} | {' · '.join(f'{v:.4f}' for v in x[f'{c}_per_seed'])} "
                         f"| {lv['mean']} [{lv['low']}, {lv['high']}] | {lv['half']} | {e['one_vs_one']} | {e['seed_mean']} |")
    skipped = result.get("lang_b_skipped")
    if skipped:
        lines += ["", "- 언어별 b 에서 빠진 언어 : " + " · ".join(f"{k} {n} 쌍" for k, n in skipped.items())]
    if result.get("release_pick"):
        p = result["release_pick"]
        lines += ["", f"- 릴리스 pick (씨앗 평균 {p['mean']} 에 가장 가까운 {p['metric']}) : {p['pick']} ({p['value']})"]
    side_lines = [(f"s{r['seed']}{'r' if r.get('rerun') else ''}", r["side"]) for r in result["runs"] if r["side"]]
    if side_lines:
        names = sorted({n for _, side in side_lines for n in side})
        lines += ["", "| 곁 자 | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        for tag, side in side_lines:
            cells = [f"acc3 {side[n]['acc3']} · 거짓 지지 {side[n]['false_support']}" if n in side else "—" for n in names]
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="기준선 한 조건의 자 상태 (씨앗 짝 반폭 · 재실행 차 · 곁 자)")
    p.add_argument("--protocol", required=True)
    p.add_argument("--set", required=True, help="dev-cmp jsonl")
    p.add_argument("--runs", required=True, help="이름=판1,판2,판3 (씨앗 차례)")
    p.add_argument("--rerun", default="", help="같은 씨앗 재실행 판 (비면 ③ 없음)")
    p.add_argument("--out", default="")
    p.add_argument("--md", default="")
    return p.parse_args(argv)


def run(argv=None):
    a = parse_args(argv)
    protocol, protocol_sha = cmp.read_protocol(a.protocol)
    set_sha = cmp.sha256_file(a.set)
    set_rows = cmp.read_set(a.set)
    name, runs = cmp.load_runs(a.runs)
    rerun = cmp.load_run(a.rerun) if a.rerun else None
    cmp.check_hashes(protocol, protocol_sha, set_sha, [(name, runs)] + ([("rerun", [rerun])] if rerun else []))
    ids = list(set_rows)
    rows = [run_metrics(r, ids) for r in runs]
    if rerun is not None:
        rows.append(dict(run_metrics(rerun, ids), rerun=True))
    result = {"name": name, "protocol_sha256": protocol_sha, "set_sha256": set_sha, "set": os.path.basename(a.set),
              "runs": rows, "resolve": resolve(protocol, set_rows, runs, rerun),
              "lang_b": lang_b(protocol, set_rows, runs), "lang_b_skipped": lang_b_skipped(set_rows),
              "lang_c": lang_b(protocol, set_rows, runs, "c_recall")}
    # 규약 release_pick 을 통째로 넘긴다 — lang · domain 거름이 있으면(g3 ko b) compare.py 와 같은 쌍으로 센다
    pick = protocol.get("release_pick", {})
    result["release_pick"] = cmp.pick_release(runs, ids, pick.get("metric", "d_recall"), set_rows, pick)
    all_table = cmp.pair_hits(runs, ids)
    result["by"] = {k: [{"key": line["key"], "pairs": line["pairs"], "b_pairs": line["b_pairs"],
                         "b_recall": line["b_recall"]["base"], "d_pairs": line["d_pairs"],
                         "d_recall": line["d_recall"]["base"], "acc3": line["acc3"]["base"]}
                        for line in cmp.by_table(set_rows, all_table["ids"], all_table, all_table, [k])]
                    for k in ("lang", "domain")}
    md = to_markdown(result)
    md += "".join(["", *[f"\n| {k} | 쌍 | b 쌍 | b 재현 | d 쌍 | d 재현 | acc3 (씨앗 평균) |\n"
                         "| --- | --- | --- | --- | --- | --- | --- |\n" +
                         "".join(f"| {r['key']} | {r['pairs']} | {r['b_pairs']} | {cmp.fmt(r['b_recall'])} "
                                 f"| {r['d_pairs']} | {cmp.fmt(r['d_recall'])} | {cmp.fmt(r['acc3'])} |\n"
                                 for r in rows_)
                         for k, rows_ in result["by"].items()]])
    for path, text in ((a.out, json.dumps(result, ensure_ascii=False, indent=1) + "\n"), (a.md, md)):
        if path:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    print(md)
    return result


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        run(argv)
    except cmp.Refuse as e:
        print(f"거부: {e}", file=sys.stderr)
        return cmp.REFUSE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
