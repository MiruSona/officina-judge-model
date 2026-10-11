"""report — 얼린 use1 셋(①~③ · ④ · ⑤)의 수치 · 95% 구간 · 짝 차이 · McNemar · result.md (설계 2절 · 8절).
구간 · McNemar 는 compare.py 를 그대로 부르고, 지표마다 「셀 쌍(mask) · 나쁜 일(event)」 로 compare 쌍 표 꼴을 만든다.
"""
import collections
import os
import platform

import compare
from use_sites import common

BASE, NEW = "v100", "v110"
TEXT_MAX = 80
PARTS = (None, 4, 5)
PART_PLACES = {None: (1, 2, 3), 4: (4,), 5: (5,)}
PART_NAMES = {None: "①~③", 4: "④ 보고 ↔ 결정", 5: "⑤ 하네스"}
TELL_MIN_PP_DEFAULT = 10


def not_a(g, o):
    return g != "a"


def contradict_metrics():
    """② · ④ 는 같은 C19 꼴 (큐에 올림 flag · 안 올림 none)."""
    return [("false_contradict", False, lambda g, o: o == "flag", lambda g, o: g != "b"),
            ("d_to_b", True, lambda g, o: g == "d", lambda g, o: o == "flag"),
            ("b_recall", True, lambda g, o: g == "b", lambda g, o: o == "flag"),
            ("a_to_b", True, lambda g, o: g == "a", lambda g, o: o == "flag"),
            ("c_to_b", True, lambda g, o: g == "c", lambda g, o: o == "flag"),
            ("miss", True, lambda g, o: True, lambda g, o: (o == "flag") != (g == "b"))]


# 자리 → [(지표, 짝 지표인가, 셀 쌍, 나쁜 일)]. 짝 지표는 셀 쌍이 정답에만 달려 두 모델이 같은 쌍을 센다.
METRICS = {
    1: [("sure_false_support", True, not_a, lambda g, o: o == "pass"),
        ("false_reject", True, lambda g, o: g == "a", lambda g, o: o == "reject"),
        ("unsure", True, lambda g, o: True, lambda g, o: o == "unsure"),
        ("filter_rate", True, not_a, lambda g, o: o == "reject"),
        ("miss", True, lambda g, o: True, lambda g, o: (o == "reject") == (g == "a"))],
    2: contradict_metrics(),
    3: [("hidden_miss", True, lambda g, o: True, lambda g, o: o == "hidden" and g != "b"),
        ("hidden", True, lambda g, o: True, lambda g, o: o == "hidden")],
    4: contradict_metrics(),
    5: [("sure_false_support", True, not_a, lambda g, o: o == "pass"),
        ("false_reject", True, lambda g, o: g == "a", lambda g, o: o == "reject"),
        ("unsure", True, lambda g, o: True, lambda g, o: o == "unsure"),
        ("miss", True, lambda g, o: True, lambda g, o: (o == "pass" and g != "a") or (o == "reject" and g == "a"))],
}
# 높을수록 좋은 지표 (나머지는 낮을수록 좋다)
GOOD_HIGH = {"b_recall", "filter_rate"}
# ⑤ 결과 ↔ 하네스 SemIf 판정 (g5.go : 통과 · 애매 · 검토)
SEMIF_OF = {"pass": "pass", "unsure": "unsure", "reject": "review"}


def table(mask, events):
    """compare 쌍 표 꼴 — 셀 쌍은 정답 칸 0(SUPPORT), 나쁜 일은 counts[0]. 지표 a_recall 로 센다."""
    return {"gold": [0 if m else 1 for m in mask], "counts": [[1 if e else 0, 0, 0, 0] for e in events], "seeds": 1}


def interval(groups, mask, base_events, new_events, boot):
    return compare.cluster_bootstrap(groups, table(mask, base_events), table(mask, new_events), "a_recall",
                                     boot["n"], boot["ci"], boot["rng_seed"])


def one_rate(groups, mask, events, boot):
    """한 모델 비율과 구간 — 0 표와의 차로 낸다."""
    s = interval(groups, mask, [False] * len(mask), events, boot)
    return {"value": s["new"], "low": s["low"], "high": s["high"], "n": s["pairs"]}


def tells(s, tell_min_pp):
    """가림 = 구간이 0 을 안 걸치고 짝 차이가 tell_min_pp(%p) 이상 (설계 2-2 : 10%p 아래는 못 가림)."""
    if s["low"] is None or s.get("diff") is None:
        return False
    outside = s["low"] > 0 or s["high"] < 0
    return outside and abs(s["diff"]) >= tell_min_pp / 100


def metric_cells(rows, models, thresholds, place, boot, rule=None, tell_min_pp=TELL_MIN_PP_DEFAULT):
    """한 층(줄 묶음)의 지표들 → {지표: {모델: 칸, "pair": 짝 차이}}."""
    out = {}
    outs = {m: [common.place_outcome(r, m, thresholds, rule) for r in rows] for m in models}
    gold = [r["label"] for r in rows]
    groups = [r["group"] for r in rows]
    for name, paired, mask_fn, event_fn in METRICS[place]:
        cell = {}
        for m in models:
            known = [o is not None for o in outs[m]]
            if not any(known):
                cell[m] = None
                continue
            mask = [k and mask_fn(g, o) for k, g, o in zip(known, gold, outs[m])]
            events = [k and event_fn(g, o) for k, g, o in zip(known, gold, outs[m])]
            cell[m] = one_rate(groups, mask, events, boot)
        if paired and BASE in models and NEW in models and cell.get(BASE) and cell.get(NEW):
            both = [b is not None and n is not None for b, n in zip(outs[BASE], outs[NEW])]
            mask = [k and mask_fn(g, None) for k, g in zip(both, gold)]
            ev_b = [k and event_fn(g, o) for k, g, o in zip(both, gold, outs[BASE])]
            ev_n = [k and event_fn(g, o) for k, g, o in zip(both, gold, outs[NEW])]
            s = interval(groups, mask, ev_b, ev_n, boot)
            good = name in GOOD_HIGH
            hits_b = [e == good for m, e in zip(mask, ev_b) if m]
            hits_n = [e == good for m, e in zip(mask, ev_n) if m]
            s["mcnemar"] = compare.mcnemar_exact(hits_b, hits_n)
            s["tell"] = tells(s, tell_min_pp)
            cell["pair"] = s
        elif cell.get(BASE) and cell.get(NEW) and None not in (cell[BASE]["value"], cell[NEW]["value"]):
            cell["pair"] = {"diff": round(cell[NEW]["value"] - cell[BASE]["value"], 4), "low": None, "high": None,
                            "note": "분모가 모델마다 다르다 — 점추정만"}
        out[name] = cell
    return out


def weighted(rows, models, thresholds, metric, rule=None):
    """② flagged+rest 를 뽑힘 확률의 역수로 가중한 점추정 (구간 없음)."""
    spec = [s for s in METRICS[2] if s[0] == metric][0]
    out = {}
    for m in models:
        num = den = 0.0
        for r in rows:
            o = common.place_outcome(r, m, thresholds, rule)
            if o is None or not spec[2](r["label"], o):
                continue
            den += r.get("weight", 1.0)
            num += r.get("weight", 1.0) if spec[3](r["label"], o) else 0.0
        out[m] = round(num / den, 4) if den else None
    return out


def strata_of(rows, place):
    by = collections.OrderedDict()
    for r in rows:
        if r["place"] == place:
            by.setdefault(r["stratum"], []).append(r)
    return by


def confusion(rows, models, thresholds, rule=None):
    out = {}
    for m in models:
        c = collections.Counter()
        for r in rows:
            c[(r["place"], r["label"], common.place_outcome(r, m, thresholds, rule) or "없음")] += 1
        out[m] = {f"{p}|{g}|{o}": n for (p, g, o), n in sorted(c.items())}
    return out


def collected(rows, models, thresholds, rule):
    """T6 — ④ ⑤ 의 모델마다 보류율 · ⑤ SemIf 판정과 일치율."""
    out = {}
    for m in models:
        answered = unsure = agree = semif_n = 0
        for r in rows:
            letter, sure = common.judge_answer((r.get("answers") or {}).get(m), thresholds, rule)
            if letter is None:
                continue
            answered += 1
            unsure += 0 if sure else 1
            semif = r.get("semif") or {}
            if r["place"] == 5 and semif.get("verdict"):
                semif_n += 1
                o = common.place_outcome(r, m, thresholds, rule)
                agree += 1 if SEMIF_OF.get(o) == semif["verdict"] else 0
        out[m] = {"answered": answered, "unsure": round(unsure / answered, 4) if answered else None,
                  "semif_agree": round(agree / semif_n, 4) if semif_n else None, "semif_n": semif_n}
    return out


def analyse(alias, part, protocol, protocol_sha):
    paths = common.part_paths(alias, part)
    meta = common.read_json(paths["frozen"])
    if common.sha256_file(paths["set"]) != meta["set_sha256"]:
        raise common.Refuse(f"{alias} 셋 해시가 얼린 값과 다르다 — 얼린 뒤 손댔다")
    if meta["protocol_sha256"] != protocol_sha:
        raise common.Refuse(f"{alias} 를 얼린 규약과 --protocol 해시가 다르다")
    rows = common.read_jsonl(paths["set"])
    thresholds = meta["thresholds"]["repo"]
    rule = common.unsure_rule_of(protocol)
    tell_min = protocol.get("tell_min_pp", TELL_MIN_PP_DEFAULT)
    models = [m for m in (BASE, NEW) if m in meta["models"]]
    boot = protocol["bootstrap"]
    result = {"alias": alias, "part": part, "meta": meta, "models": models, "places": {}}
    for place in PART_PLACES[part]:
        strata = strata_of(rows, place)
        if not strata:
            continue
        cells = {}
        for name, srows in strata.items():
            unseen = [r for r in srows if not r.get("seen")]
            cells[name] = {"all": metric_cells(srows, models, thresholds, place, boot, rule, tell_min),
                           "unseen": metric_cells(unseen, models, thresholds, place, boot, rule, tell_min) if unseen else None,
                           "n": len(srows), "n_unseen": len(unseen)}
        result["places"][place] = cells
        if place == 2:
            mixed = [r for r in rows if r["place"] == 2 and r["stratum"] in ("flagged", "rest")]
            result["weighted_2"] = {k: weighted(mixed, models, thresholds, k, rule)
                                    for k in ("b_recall", "d_to_b", "false_contradict")}
    if part is not None:
        result["collected"] = collected(rows, models, thresholds, rule)
    result["confusion"] = confusion(rows, models, thresholds, rule)
    result["errors"] = error_forms(rows, models, thresholds, rule)
    return result


def error_forms(rows, models, thresholds, rule=None):
    """오판 꼴 = 「자리 · 정답 → 결과」. 맞힌 꼴은 뺀다."""
    out = {}
    for m in models:
        forms = collections.defaultdict(list)
        for r in rows:
            o = common.place_outcome(r, m, thresholds, rule)
            if o is None:
                continue
            bad = [spec for spec in METRICS[r["place"]] if spec[0] == "miss" or spec[0] == "hidden_miss"][0]
            if bad[3](r["label"], o):
                forms[f"{r['place']} · {r['label']}→{o}"].append(r)
        out[m] = forms
    return out


# ---------------------------------------------------------------- md

def safe(text):
    """믿지 않는 글 — 자르고 표 · 코드 꾸밈을 막는다."""
    text = " ".join(str(text).split())
    if len(text) > TEXT_MAX:
        text = text[:TEXT_MAX] + "…"
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("<", "&lt;")


def cell_text(c):
    if c is None:
        return "unavailable"
    if c["value"] is None:
        return "— (0)"
    return f"{c['value']:.3f} [{c['low']:.3f}, {c['high']:.3f}] ({c['n']})"


def pair_text(p):
    if not p or p.get("diff") is None:
        return "— | — | —"
    if p.get("low") is None:
        return f"{p['diff']:+.3f} | {p.get('note', '')} | —"
    m = p["mcnemar"]
    return f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}] | {m['b']}/{m['c']} · p {m['p']} | {'가림' if p['tell'] else '못 가림'}"


def to_markdown(results, protocol_sha, examples, tell_min_pp=TELL_MIN_PP_DEFAULT):
    lines = ["# 실사용 측정 use1 결과", "", f"- 규약 protocol-use1 · {protocol_sha[:12]} · 만든 때 {common.now_iso()}"
             f" · {platform.system()} {platform.release()}", f"- 가림 = 짝 차이 구간이 0 밖이고 차이가 {tell_min_pp}%p 이상", ""]
    for res in results:
        lines += alias_markdown(res, examples)
    lines += t7()
    return "\n".join(lines) + "\n"


def t0_lines(res):
    meta = res["meta"]
    th = meta["thresholds"]["repo"]
    model_ids = " · ".join(f"{k} {v.get('name', '')} {v.get('model', '')}" if v.get("ok") else f"{k} unavailable"
                           for k, v in meta["models"].items())
    seen = meta.get("seen_pairs", 0)
    return ["### T0 측정 조건", "", "| 칸 | 값 |", "| --- | --- |",
            f"| 기억 수 | {meta.get('memories')} |", f"| 모델 | {safe(model_ids)} |",
            f"| 문턱 repo | nli_sure {th['nli_sure']} · nli_sure_support {th['nli_sure_support']} |",
            f"| 셋 sha256 | {meta['set_sha256'][:16]} ({meta['pairs']}쌍) |",
            f"| 규약 sha256 | {meta['protocol_sha256'][:16]} |", f"| 얼린 때 | {meta['frozen_at']} |",
            f"| 본 쌍 | {seen} / {meta['pairs']} |",
            f"| 라벨 | 사람 {meta['labels']['human']} · 사람이 바꾼 것 {meta['labels']['human_changed']} · κ {(meta.get('kappa') or {}).get('all')} |",
            ""]


def alias_markdown(res, examples):
    models = res["models"]
    lines = [f"## 별명 {safe(res['alias'])} · {PART_NAMES[res['part']]}", ""] + t0_lines(res)
    lines += ["### T1 자리 × 층 × 모델 (값 [95% 구간] (n)) · T2 짝 차이 (v110 − v100)", "",
              "| 자리 · 층 | 지표 | 줄 | " + " | ".join(models) + " | 차 [구간] | McNemar b/c · p | 가림 |",
              "| --- | --- | --- | " + " | ".join("---" for _ in models) + " | --- | --- | --- |"]
    for place, strata in res["places"].items():
        for name, block in strata.items():
            for which in ("all", "unseen"):
                cells = block[which]
                if cells is None:
                    continue
                label = "전부" if which == "all" else "본 쌍 뺌"
                for metric, cell in cells.items():
                    vals = " | ".join(cell_text(cell.get(m)) for m in models)
                    lines.append(f"| {place} · {name} | {metric} | {label} | {vals} | {pair_text(cell.get('pair'))} |")
    if res.get("weighted_2"):
        lines += ["", "- ② flagged+rest 가중 점추정 : " + " · ".join(
            f"{k} " + " / ".join(f"{m} {v}" for m, v in w.items()) for k, w in res["weighted_2"].items())]
    lines += t3_lines(res) + t4_lines(res) + t5_lines(res, examples) + t6_lines(res)
    return lines


def t3_lines(res):
    queues = res["meta"].get("queues")
    if not queues:
        return []
    lines = ["", "### T3 실제 큐 (mem review)", "", "| 판 | contradict | C19 | C19(NLI) | merge | 숨긴 merge | NLI 물음 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, q in queues.items():
        nli = q.get("nli") or {}
        lines.append(f"| {name} | {q['contradict']} | {q['c19']} | {q['c19_nli']} | {q['merge']} | {q.get('hidden_merge', '—')} "
                     f"| {nli.get('calls', '—')} (못 받음 {nli.get('failed', '—')}) |")
    return lines


def t4_lines(res):
    lines = ["", "### T4 혼동표 (자리 · 정답 · 결과)", "", "| 모델 | 자리 | 정답 | 결과 | 수 |", "| --- | --- | --- | --- | --- |"]
    for m, table_ in res["confusion"].items():
        for key, n in table_.items():
            p, g, o = key.split("|")
            lines.append(f"| {m} | {p} | {g} | {o} | {n} |")
    return lines


def t5_lines(res, examples):
    lines = ["", "### T5 오판 꼴", "", "| 모델 | 꼴 | 수 | 보기 |", "| --- | --- | --- | --- |"]
    for m, forms in res["errors"].items():
        for form, rows in sorted(forms.items(), key=lambda kv: -len(kv[1])):
            sample = "—"
            if examples and rows:
                sample = f"「{safe(rows[0]['evidence'])}」 → 「{safe(rows[0]['claim'])}」"
            lines.append(f"| {m} | {form} | {len(rows)} | {sample} |")
    return lines


def t6_lines(res):
    lines = ["", "### T6 수집 자리", ""]
    if res["part"] is None:
        return lines + ["- ①~③ 은 수집 자리가 아니다 — ④ ⑤ 는 같은 별명의 `--place 4|5` 판 블록을 본다", ""]
    meta = res["meta"]
    span = meta.get("collect") or {}
    lines += [f"- 모은 기간 {span.get('first')} ~ {span.get('last')} · 수집 줄 {span.get('rows')} (비밀 꼴 refused "
              f"{span.get('refused')}) · 물은 풀 {meta.get('pool')} · 뽑은 쌍 {meta['pairs']}",
              "- 축약 손실률 : Q0 라벨(labels extra)이 아직 없어 못 낸다", "",
              "| 모델 | 답 받은 쌍 | 보류율 | SemIf 일치율 (n) |", "| --- | --- | --- | --- |"]
    for m, c in res.get("collected", {}).items():
        lines.append(f"| {m} | {c['answered']} | {c['unsure']} | {c['semif_agree']} ({c['semif_n']}) |")
    return lines + [""]


def t7():
    return ["## T7 읽는 법 (B · C 가늠)", "", "| 숫자가 이러면 | 기운다 |", "| --- | --- |",
            "| ① 확신 거짓 지지가 두 모델 다 낮고(위 끝 ≤ 6%) ② 거짓 반대 · d → b 가 큐의 큰 몫 | **C** — 규칙 · 임베딩 문턱 · SemIf 넘김 |",
            "| ② 진짜 반대 재현이 v110 에서 0.7 아래이고 놓친 b 가 수 · 한정어 꼴 | **B** 의 b 묶음이 mem 에도 값이 있다 |",
            "| v110 이 v100 보다 mem 자리에서 나쁨(구간 0 밖) | 8092 를 v1.0.0 으로 되돌릴지 · 문턱(model 벌)을 llm.toml 에 넣을지 사용자 판단 |",
            "| 짝 차이 10%p 아래 · 구간이 0 을 걸침 | 「못 가림」 — 합격선을 맞추려고 층을 바꾸지 않는다 |",
            "| ⑤ 보류 > 50% | ⑤ 자리는 접는다 (SemIf 그대로) |",
            "| ③-2 · v110d 줄 | 줄인 판(11-1)에서 뺐다 |", ""]


def frozen_parts(alias, place):
    """report 할 판들. --place 가 있으면 그 판만(안 얼렸으면 멈춤), 없으면 얼린 판 전부."""
    if place is not None:
        if not os.path.isfile(common.part_paths(alias, place)["frozen"]):
            raise common.Refuse(f"{alias} 의 자리 {place} 판은 아직 안 얼렸다")
        return [place]
    parts = [p for p in PARTS if os.path.isfile(common.part_paths(alias, p)["frozen"])]
    if not parts:
        raise common.Refuse(f"{alias} 에 얼린 판이 없다 — freeze 를 먼저")
    return parts


def cmd_report(a):
    protocol, protocol_sha = common.load_protocol(a.protocol)
    aliases = [x for x in a.alias.split(",") if x]
    results = [analyse(alias, part, protocol, protocol_sha) for alias in aliases for part in frozen_parts(alias, a.place)]
    md = to_markdown(results, protocol_sha, a.examples, protocol.get("tell_min_pp", TELL_MIN_PP_DEFAULT))
    base = common.part_paths(aliases[0], a.place)["result"]
    out = base + ".md"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    slim = [{k: v for k, v in r.items() if k != "errors"} for r in results]
    for r, full in zip(slim, results):
        r["errors"] = {m: {form: len(rows) for form, rows in forms.items()} for m, forms in full["errors"].items()}
    common.write_json(base + ".json", {"protocol_sha256": protocol_sha, "results": slim})
    print(f"{os.path.basename(out)} → {out}")
    return md


def register(sub):
    s = sub.add_parser("report", help="수치 · 구간 · McNemar · result.md (얼린 판 전부를 한 파일에)")
    s.add_argument("--protocol", default="")
    s.add_argument("--alias", required=True, help="A[,B] — 결과는 첫 별명 폴더에")
    s.add_argument("--place", type=int, choices=(4, 5), default=None,
                   help="그 판만 result-N.md 로 (없으면 얼린 판 전부를 result.md 로)")
    s.add_argument("--examples", action="store_true", help="T5 에 원문 보기 (스튜디오만)")
    s.set_defaults(func=cmd_report)
