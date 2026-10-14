"""짝 비교 — 두 조건(씨앗 여럿)을 같은 쌍에서 견줘 「먹혔나」를 가린다 (d 자료 층 설계 2-3).

    python -I compare.py --protocol measure/protocol-g1.json --set <dev-cmp-g1.jsonl> \
        --base B0=runs/g1-B0-s1/eval-dev-cmp,runs/g1-B0-s2/eval-dev-cmp,runs/g1-B0-s3/eval-dev-cmp \
        --new B1=runs/g1-B1-s1/eval-dev-cmp,... --by domain,lang --out compare-B1.json --md compare-B1.md

판 자리는 eval_nli.py 의 pred.jsonl 이나 그 폴더다. 같은 폴더의 summary.json 에서 set_sha256 · protocol_sha256 을,
그 폴더나 윗 폴더에서 eval-side.txt(곁 자) · calibration.json(보고만 하는 확신 자)을 읽는다.
해시가 하나라도 어긋나거나 없으면 · calibration 이 잴 셋으로 골라졌으면 exit 2 (거부).
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import calibrate as cal  # noqa: E402

LETTERS = "abcd"
SUPPORT, CONTRA, UNRELATED, RELATED = 0, 1, 2, 3
CD = -2
REFUSE_EXIT = 2
# 지표 이름 → (어느 정답 쌍을 세나, 무엇을 맞힘으로 치나). 재현은 정답 칸 기준, x_to_y 는 정답 x 중 y 로 본 몫
METRICS = {
    "acc3": ("all", "fold"),
    "a_recall": (SUPPORT, SUPPORT), "b_recall": (CONTRA, CONTRA),
    "c_recall": (UNRELATED, UNRELATED), "d_recall": (RELATED, RELATED),
    "d_to_a": (RELATED, SUPPORT), "a_to_d": (SUPPORT, RELATED),
}
HIGHER_IS_BETTER = {"d_to_a": False, "a_to_d": False}


class Refuse(Exception):
    """규약이 어긋나 셈을 거부한다 — main 이 exit 2 로 바꾼다."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_protocol(path):
    """(규약 dict, 파일 sha256). 결과에는 이 sha 를 protocol_sha256 으로 적는다."""
    return read_json(path), sha256_file(path)


def read_set(path):
    """셋 jsonl → {id: 줄}. group · domain · lang 칸을 여기서 읽는다."""
    rows = {}
    for row in read_jsonl(path):
        rows[str(row.get("id", ""))] = row
    return rows


def read_side(path):
    """eval-side.txt → {이름: {sha8, n, acc, acc3, false_support}}. 줄 꼴 : 이름 sha8 수 acc acc3 거짓지지 {재현}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            out[parts[0]] = {"sha8": parts[1], "n": int(parts[2]), "acc": float(parts[3]),
                             "acc3": float(parts[4]), "false_support": int(parts[5])}
    return out


def find_beside(pred_path, names):
    """pred 폴더 → 윗 폴더 차례로 names 중 처음 있는 파일. 없으면 빈 문자열."""
    here = os.path.dirname(os.path.abspath(pred_path))
    for folder in (here, os.path.dirname(here)):
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return ""


def load_run(path):
    """판 하나 — pred 줄 · summary · 곁 자 · calibration (없으면 None)."""
    pred_path = path
    if os.path.isdir(path):
        pred_path = os.path.join(path, "pred.jsonl")
    if not os.path.isfile(pred_path):
        raise Refuse(f"pred 파일이 없다: {path}")
    summary_path = os.path.join(os.path.dirname(os.path.abspath(pred_path)), "summary.json")
    side_path = find_beside(pred_path, ["eval-side.txt"])
    cal_path = find_beside(pred_path, ["calibration.json"])
    # 씨앗은 판 폴더 이름(g1-B0-s2)에서 — 윗 경로에 「-s숫자」가 또 있을 수 있어 마지막 것을 쓴다
    seeds = re.findall(r"-s(\d+)", os.path.abspath(pred_path).replace("\\", "/"))
    return {
        "path": pred_path,
        "seed": int(seeds[-1]) if seeds else None,
        "preds": {str(r.get("id", "")): r for r in read_jsonl(pred_path)},
        "summary": read_json(summary_path) if os.path.isfile(summary_path) else None,
        "side": read_side(side_path) if side_path else None,
        "calibration": read_json(cal_path) if cal_path else None,
    }


def load_runs(spec):
    """「이름=판1,판2,판3」 → (이름, [판…]). 판 차례가 씨앗 차례다."""
    if "=" not in spec:
        raise Refuse(f"조건 꼴은 이름=판1,판2,… 이다: {spec}")
    name, paths = spec.split("=", 1)
    runs = [load_run(p) for p in paths.split(",") if p]
    if not runs:
        raise Refuse(f"판이 없다: {spec}")
    return name, runs


def pred_index(row):
    letter = row.get("pred")
    if letter in LETTERS:
        return LETTERS.index(letter)
    logits = row["logits"]
    return max(range(len(logits)), key=lambda i: logits[i])


def fold3(label):
    if label in (UNRELATED, RELATED, CD):
        return UNRELATED
    return label


def pair_hits(runs, ids):
    """쌍마다 정답과 「씨앗 몇이 어느 칸을 골랐나」. 재현 · 접은 정확도는 이 표에서 정답 칸 기준으로 센다.

    돌려줌 {"ids", "gold", "counts"(쌍마다 칸별 고른 씨앗 수), "seeds"}. 라벨 없는 쌍은 뺀다.
    """
    gold, counts, kept = [], [], []
    for pid in ids:
        labels = set()
        row_counts = [0, 0, 0, 0]
        for run in runs:
            row = run["preds"].get(pid)
            if row is None:
                raise Refuse(f"{run['path']} 에 쌍 {pid} 가 없다 — 같은 셋을 잰 판인지 본다")
            labels.add(row.get("label"))
            row_counts[pred_index(row)] += 1
        if len(labels) != 1:
            raise Refuse(f"쌍 {pid} 의 정답이 판마다 다르다 {sorted(map(str, labels))} — --public-neutral 을 맞춘다")
        label = labels.pop()
        if label is None:
            continue
        gold.append(label)
        counts.append(row_counts)
        kept.append(pid)
    return {"ids": kept, "gold": gold, "counts": counts, "seeds": len(runs)}


def check_same_pairs(base, new, what="두 조건"):
    """두 쌍 표의 쌍 · 정답이 같은 차례인가. 다르면 Refuse — 붓스트랩 · McNemar 가 차례로 짝지어 조용히 틀린다.

    (한 조건 안 씨앗끼리는 pair_hits 가 본다. 조건 사이는 --public-neutral · 칸 수가 달라 정답이 갈릴 수 있다)
    """
    if base["ids"] != new["ids"]:
        raise Refuse(f"{what}의 라벨 있는 쌍이 다르다 — 같은 셋 · 같은 --public-neutral 로 쟀는지 본다")
    if base["gold"] != new["gold"]:
        diff = sum(1 for x, y in zip(base["gold"], new["gold"]) if x != y)
        raise Refuse(f"{what}의 정답이 {diff} 쌍에서 다르다 — 칸 수 · --public-neutral 을 맞춘다")


def hit_counts(table, metric):
    """지표 하나 → (쌍마다 셀지, 쌍마다 맞힌 씨앗 수)."""
    scope, target = METRICS[metric]
    mask, hits = [], []
    for g, c in zip(table["gold"], table["counts"]):
        if scope == "all":
            mask.append(True)
            hits.append(c[UNRELATED] + c[RELATED] if fold3(g) == UNRELATED else c[g])
            continue
        mask.append(g == scope)
        hits.append(c[target])
    return mask, hits


# 자를 언어 · 도메인으로 걸러 쓰는 칸 (protocol-g3 부터). 자 이름 꼬리는 이 차례로 붙는다 — b_recall@mem·ko
SLICE_KEYS = ("domain", "lang")


def is_slice(spec):
    """{"metric": "b_recall", "lang": "ko"} 처럼 거름 칸이 있는 자인가. 옛 꼴(metric 만)은 False."""
    return any(spec.get(k) for k in SLICE_KEYS)


def ruler_name(spec):
    """자 이름 — 거름이 없으면 지표 이름 그대로(g1 · g2 와 같다), 있으면 b_recall@ko · b_recall@mem·ko."""
    parts = [str(spec[k]) for k in SLICE_KEYS if spec.get(k)]
    return spec["metric"] + ("@" + "·".join(parts) if parts else "")


def slice_positions(set_rows, ids, spec):
    """쌍 표 차례(ids)에서 거름 칸이 다 맞는 자리 번호."""
    want = {k: str(spec[k]) for k in SLICE_KEYS if spec.get(k)}
    return [pos for pos, pid in enumerate(ids)
            if all(str(set_rows.get(pid, {}).get(k)) == v for k, v in want.items())]


def sub_table(table, positions):
    """쌍 표에서 몇 자리만 떼어 같은 꼴의 표로."""
    return {"ids": [table["ids"][i] for i in positions], "gold": [table["gold"][i] for i in positions],
            "counts": [table["counts"][i] for i in positions], "seeds": table["seeds"]}


def slice_compare(protocol, set_rows, base, new, spec):
    """자 하나를 걸러진 쌍만으로 — 묶음 붓스트랩(new − base) · McNemar. 셀 쌍이 0 이면 Refuse(규약 · 셋 칸이 어긋남).

    base · new 는 같은 차례의 쌍 표다 (짝 비교의 두 조건, 또는 릴리스의 ref · 잴 판).
    """
    positions = slice_positions(set_rows, base["ids"], spec)
    b, n = sub_table(base, positions), sub_table(new, positions)
    key = protocol.get("group_key", "group")
    groups = [str(set_rows[pid].get(key) or pid) for pid in b["ids"]]
    boot = protocol.get("bootstrap", {})
    s = cluster_bootstrap(groups, b, n, spec["metric"], boot.get("n", 10000), boot.get("ci", 0.95),
                          boot.get("rng_seed", 20261011))
    if s["pairs"] == 0:
        raise Refuse(f"자 {ruler_name(spec)} 에 셀 쌍이 없다 — 셋의 {' · '.join(SLICE_KEYS)} 칸과 규약을 본다")
    s["metric"] = ruler_name(spec)
    s["mcnemar"] = mcnemar_exact(majority(b, spec["metric"]), majority(n, spec["metric"]))
    return s


def gate_slice_specs(protocol):
    """pair_gate 의 목표 · 지킴 · 경보 중 걸러 쓴 자 (나온 차례, 같은 이름은 하나)."""
    gate = protocol.get("pair_gate", {})
    specs = {}
    for spec in [gate.get("target", {})] + gate.get("guards", []) + gate.get("alarms", []):
        if spec.get("metric") in METRICS and is_slice(spec):
            specs.setdefault(ruler_name(spec), spec)
    return list(specs.values())


def rate(table, metric):
    mask, hits = hit_counts(table, metric)
    n = sum(mask)
    if n == 0:
        return None
    return sum(h for m, h in zip(mask, hits) if m) / (n * table["seeds"])


def false_support_mean(table):
    """문턱 없는 거짓 지지 — 정답이 지지가 아닌데 최고 답이 지지인 쌍 수의 씨앗 평균."""
    total = sum(c[SUPPORT] for g, c in zip(table["gold"], table["counts"]) if g != SUPPORT)
    return total / table["seeds"]


def percentile(sorted_values, q):
    """numpy 기본(선형 보간)과 같은 백분위."""
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cluster_bootstrap(groups, base, new, metric, n=10000, ci=0.95, rng_seed=20261011):
    """묶음(group) 단위로 다시 뽑아 (new − base) 차의 구간. 씨앗은 고정 — 쌍마다 씨앗 평균 맞힘을 그대로 쓴다.

    groups 는 쌍마다 묶음 열쇠(base · new 표와 같은 차례). 돌려줌 base · new · diff · low · high.
    """
    mask_b, hits_b = hit_counts(base, metric)
    _, hits_n = hit_counts(new, metric)
    keys = sorted(set(g for g, m in zip(groups, mask_b) if m))
    index = {k: i for i, k in enumerate(keys)}
    sum_b, sum_n, cnt = [0.0] * len(keys), [0.0] * len(keys), [0] * len(keys)
    for g, m, hb, hn in zip(groups, mask_b, hits_b, hits_n):
        if not m:
            continue
        i = index[g]
        sum_b[i] += hb / base["seeds"]
        sum_n[i] += hn / new["seeds"]
        cnt[i] += 1
    total = sum(cnt)
    if total == 0:
        return {"metric": metric, "pairs": 0, "groups": 0, "base": None, "new": None, "diff": None,
                "low": None, "high": None}
    rng = random.Random(rng_seed)
    picks = range(len(keys))
    diffs = []
    for _ in range(n):
        idx = rng.choices(picks, k=len(keys))
        c = sum(cnt[i] for i in idx)
        if c:
            diffs.append((sum(sum_n[i] for i in idx) - sum(sum_b[i] for i in idx)) / c)
    diffs.sort()
    tail = (1 - ci) / 2
    return {"metric": metric, "pairs": total, "groups": len(keys), "base": round(sum(sum_b) / total, 4),
            "new": round(sum(sum_n) / total, 4), "diff": round((sum(sum_n) - sum(sum_b)) / total, 4),
            "low": round(percentile(diffs, tail), 4), "high": round(percentile(diffs, 1 - tail), 4)}


def majority(table, metric):
    """세는 쌍마다 「씨앗 과반이 맞혔나」 (맞힌 수 × 2 > 씨앗 수)."""
    mask, hits = hit_counts(table, metric)
    return [h * 2 > table["seeds"] for m, h in zip(mask, hits) if m]


def mcnemar_exact(base_hits, new_hits):
    """짝 맞힘 둘 → McNemar 정확 검정(양쪽). b = base 만 맞힘 · c = new 만 맞힘."""
    b = sum(1 for x, y in zip(base_hits, new_hits) if x and not y)
    c = sum(1 for x, y in zip(base_hits, new_hits) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"b": b, "c": c, "p": round(min(1.0, 2 * tail), 6)}


def side_means(runs, side_names):
    """곁 자 씨앗 평균 — {mnli: acc3, klue: acc3, k48_hits: 맞힌 수}. 한 판이라도 없으면 None."""
    if any(r["side"] is None for r in runs):
        return None
    out = {}
    for key, name in side_names.items():
        values = [r["side"].get(name) for r in runs]
        if any(v is None for v in values):
            out[key] = None
            continue
        if key == "k48":
            out["k48_hits"] = sum(round(v["acc3"] * v["n"]) for v in values) / len(values)
            continue
        out[key] = sum(v["acc3"] for v in values) / len(values)
    return out


def sure_report(runs, ids):
    """보고만 — calibration.json 의 온도·문턱을 건 보류 몫 · d 확신 정밀(씨앗 평균). calibration 없는 판이 있으면 None."""
    if any(r["calibration"] is None for r in runs):
        return None
    unsure, precision = [], []
    for run in runs:
        c = run["calibration"]
        th = c.get("thresholds")
        if not th:
            return None
        scored = [(cal.softmax(run["preds"][i]["logits"], c["temperature"]), run["preds"][i]["label"])
                  for i in ids]
        unsure.append(cal.unsure_rate(scored, th["nli_sure"], th["nli_sure_support"], th.get("nli_sure_related")))
        if th.get("nli_sure_related") is not None:
            precision.append(cal.sure_related_precision(scored, th["nli_sure_related"])[0])
    known = [p for p in precision if p is not None]
    return {"unsure": round(sum(unsure) / len(unsure), 4),
            "d_sure_precision": round(sum(known) / len(known), 4) if known else None}


def pick_release(runs, ids, metric="d_recall", set_rows=None, spec=None):
    """씨앗 판 중 지표가 평균에 가장 가까운 판 (최고 판 아님 — 승자 저주). 같으면 acc3 높은 판 · 씨앗 작은 판.

    spec(규약 release_pick)에 거름 칸(lang · domain)이 있으면 지표만 걸러진 쌍으로 센다. acc3 동률 깨기는 쌍 전체로.
    """
    sliced = bool(spec) and is_slice(spec)
    name = ruler_name(dict(spec, metric=metric)) if sliced else metric
    scores = []
    for i, run in enumerate(runs):
        table = pair_hits([run], ids)
        value = rate(table, metric)
        if sliced:
            value = rate(sub_table(table, slice_positions(set_rows or {}, table["ids"], spec)), metric)
        scores.append((value, rate(table, "acc3"), i))
    if any(s[0] is None for s in scores):
        return None
    mean = sum(s[0] for s in scores) / len(scores)
    best = min(scores, key=lambda s: (abs(s[0] - mean), -s[1], s[2]))
    return {"metric": name, "mean": round(mean, 4), "pick": runs[best[2]]["path"], "value": round(best[0], 4)}


def calib_sha(calibration):
    """calibration 이 고른 셋의 해시 — 새 칸 calib_set_sha256, 없으면 옛 칸 fitted_on.sha256."""
    if calibration.get("calib_set_sha256"):
        return calibration["calib_set_sha256"]
    return (calibration.get("fitted_on") or {}).get("sha256", "")


def check_hashes(protocol, protocol_sha, set_sha, conditions):
    """두 조건의 모든 판이 같은 셋 · 같은 규약을 쟀나, calibration 이 잴 셋으로 고르지 않았나. 어긋나면 Refuse."""
    want = protocol.get("sets", {}).get(protocol.get("compare_on", ""), "")
    if want and want != set_sha:
        raise Refuse(f"--set 해시 {set_sha[:12]} 가 규약의 {protocol['compare_on']} 해시 {want[:12]} 와 다르다")
    for name, runs in conditions:
        for run in runs:
            s = run["summary"]
            if s is None:
                raise Refuse(f"{run['path']} 옆에 summary.json 이 없다 — eval_nli.py 로 다시 잰다")
            if s.get("set_sha256") != set_sha:
                raise Refuse(f"[{name}] {run['path']} 의 set_sha256 {str(s.get('set_sha256'))[:12]} ≠ 잴 셋 {set_sha[:12]}")
            if s.get("protocol_sha256") != protocol_sha:
                raise Refuse(f"[{name}] {run['path']} 의 protocol_sha256 {str(s.get('protocol_sha256'))[:12]} "
                             f"≠ 규약 {protocol_sha[:12]} — eval_nli.py --protocol 로 다시 잰다")
            if run["calibration"] is not None and calib_sha(run["calibration"]) == set_sha:
                raise Refuse(f"[{name}] {run['path']} 의 calibration 이 잴 셋으로 골랐다 — 문턱은 dev-cal 로만")


BY_METRICS = ("d_recall", "b_recall", "acc3")


def by_table(set_rows, ids, base, new, keys):
    """--by 표 — 칸 조합마다 d · b 재현 · 3칸 정확도 (base · new · 차). 점 추정만, 구간 없음."""
    buckets = {}
    for pos, pid in enumerate(ids):
        row = set_rows.get(pid, {})
        key = " · ".join(str(row.get(k, "-")) for k in keys)
        buckets.setdefault(key, []).append(pos)
    out = []
    for key in sorted(buckets):
        sub = buckets[key]
        b = {"gold": [base["gold"][i] for i in sub], "counts": [base["counts"][i] for i in sub], "seeds": base["seeds"]}
        n = {"gold": [new["gold"][i] for i in sub], "counts": [new["counts"][i] for i in sub], "seeds": new["seeds"]}
        line = {"key": key, "pairs": len(sub), "d_pairs": sum(1 for g in b["gold"] if g == RELATED),
                "b_pairs": sum(1 for g in b["gold"] if g == CONTRA)}
        for metric in BY_METRICS:
            rb, rn = rate(b, metric), rate(n, metric)
            line[metric] = {"base": rb, "new": rn, "diff": None if rb is None or rn is None else round(rn - rb, 4)}
        out.append(line)
    return out


def compare(protocol, set_rows, base_runs, new_runs):
    """두 조건의 쌍 표 · 지표별 구간 · McNemar · 거짓 지지 · 곁 자 · 보고만 칸."""
    ids = [pid for pid in set_rows]
    base = pair_hits(base_runs, ids)
    new = pair_hits(new_runs, ids)
    check_same_pairs(base, new)
    groups = [str(set_rows[pid].get(protocol.get("group_key", "group")) or pid) for pid in base["ids"]]
    boot = protocol.get("bootstrap", {})
    stats = {"pairs": len(base["ids"]), "groups": len(set(groups)), "seeds": [base["seeds"], new["seeds"]],
             "missing_group": sum(1 for pid in base["ids"] if not set_rows[pid].get(protocol.get("group_key", "group"))),
             "metrics": {}}
    for metric in METRICS:
        s = cluster_bootstrap(groups, base, new, metric, boot.get("n", 10000), boot.get("ci", 0.95),
                              boot.get("rng_seed", 20261011))
        s["mcnemar"] = mcnemar_exact(majority(base, metric), majority(new, metric))
        stats["metrics"][metric] = s
    for spec in gate_slice_specs(protocol):
        stats["metrics"][ruler_name(spec)] = slice_compare(protocol, set_rows, base, new, spec)
    stats["false_support"] = {"base": false_support_mean(base), "new": false_support_mean(new)}
    side_names = protocol.get("side", {})
    stats["side"] = {"base": side_means(base_runs, side_names), "new": side_means(new_runs, side_names)}
    stats["by_lang"] = by_table(set_rows, base["ids"], base, new, ["lang"])
    stats["sure"] = {"base": sure_report(base_runs, base["ids"]), "new": sure_report(new_runs, new["ids"])}
    pick = protocol.get("release_pick", {})
    stats["release_pick"] = pick_release(new_runs, base["ids"], pick.get("metric", "d_recall"), set_rows, pick)
    stats["_tables"] = (base, new)
    return stats


def guard_result(guard, stats):
    """지킴 하나 → (통과 True/False/None(잴 수 없음), 한 줄 까닭)."""
    metric = guard["metric"]
    if metric == "false_support":
        fs = stats["false_support"]
        limit = fs["base"] * guard["max_ratio"]
        return fs["new"] <= limit + 1e-9, f"거짓 지지 {fs['base']:.2f} → {fs['new']:.2f} (한도 {limit:.2f})"
    if metric == "side_acc3":
        sb, sn = stats["side"]["base"], stats["side"]["new"]
        if sb is None or sn is None:
            return None, "곁 자 없음 (eval-side.txt)"
        notes, ok = [], True
        for key in guard["sides"]:
            if sb.get(key) is None or sn.get(key) is None:
                return None, f"곁 자 {key} 없음"
            diff = sn[key] - sb[key]
            ok = ok and diff >= guard["min_mean_diff"] - 1e-9
            notes.append(f"{key} {diff:+.4f}")
        return ok, " · ".join(notes)
    name = ruler_name(guard)
    s = stats["metrics"][name]
    return s["low"] >= guard["ci_low_ge"] - 1e-9, f"{name} 차 {s['diff']:+.4f} [{s['low']:+.4f}, {s['high']:+.4f}]"


def verdict(protocol, stats):
    """관문 표 → 먹힘 · 같다 · 지킴 깨짐 · 보류 (뜻은 protocol pair_gate.verdicts)."""
    gate = protocol["pair_gate"]
    target = gate["target"]
    target_name = ruler_name(target)
    t = stats["metrics"][target_name]
    target_ok = t["diff"] >= target["min_mean_diff"] - 1e-9 and t["low"] > target["ci_low_gt"]
    guards = []
    for guard in gate["guards"]:
        ok, note = guard_result(guard, stats)
        guards.append({"metric": ruler_name(guard), "ok": ok, "note": note})
    alarms = []
    for alarm in gate.get("alarms", []):
        if alarm["metric"] == "k48_hits":
            sn = stats["side"]["new"] or {}
            if sn.get("k48_hits") is not None and sn["k48_hits"] < alarm["min"]:
                alarms.append(f"K48 맞힌 수 평균 {sn['k48_hits']:.1f} < {alarm['min']}")
            continue
        if alarm["metric"] in METRICS:
            # 새 꼴 (g3) — {"metric": "b_recall", "domain": "mem", "lang": "ko", "min_mean_diff": …} 를 걸러 쓴 자로
            name = ruler_name(alarm)
            diff = stats["metrics"][name]["diff"]
            if diff is not None and diff < alarm["min_mean_diff"]:
                alarms.append(f"{name} 평균 차 {diff:+.4f} < {alarm['min_mean_diff']}")
            continue
        # 옛 꼴 lang_d_recall · lang_b_recall (g1 · g2) — 「lang_」 뒤가 by_lang 줄에서 볼 지표다
        metric = alarm["metric"][len("lang_"):]
        for line in stats["by_lang"]:
            diff = line[metric]["diff"]
            if line["key"] == alarm["lang"] and diff is not None and diff < alarm["min_mean_diff"]:
                alarms.append(f"{alarm['lang']} {metric[0]} 재현 {diff:+.4f} < {alarm['min_mean_diff']}")
    if any(g["ok"] is False for g in guards):
        word = "지킴 깨짐"
    elif not target_ok:
        word = "같다"
    elif any(g["ok"] is None for g in guards):
        word = "보류"
    else:
        word = "먹힘"
    return {"verdict": word, "target": {"metric": target_name, "ok": target_ok, "diff": t["diff"],
                                         "low": t["low"], "high": t["high"]},
            "guards": guards, "alarms": alarms}


def to_markdown(result):
    lines = [f"## 짝 비교 {result['base']} → {result['new']} : **{result['verdict']['verdict']}**", "",
             f"- 쌍 {result['stats']['pairs']} · 묶음 {result['stats']['groups']} · 씨앗 {result['stats']['seeds']}"
             f" · 셋 {result['set_sha256'][:12]} · 규약 {result['protocol_sha256'][:12]}", "",
             "| 지표 | base | new | 차 | 95% 구간 | McNemar b/c · p |", "| --- | --- | --- | --- | --- | --- |"]
    for metric, s in result["stats"]["metrics"].items():
        m = s["mcnemar"]
        lines.append(f"| {metric} | {s['base']} | {s['new']} | {s['diff']} | [{s['low']}, {s['high']}] "
                     f"| {m['b']}/{m['c']} · {m['p']} |")
    lines += ["", "| 관문 | 결과 | 값 |", "| --- | --- | --- |"]
    t = result["verdict"]["target"]
    lines.append(f"| 목표 {t['metric']} | {'O' if t['ok'] else 'X'} | {t['diff']:+.4f} [{t['low']:+.4f}, {t['high']:+.4f}] |")
    for g in result["verdict"]["guards"]:
        mark = "?" if g["ok"] is None else ("O" if g["ok"] else "X")
        lines.append(f"| 지킴 {g['metric']} | {mark} | {g['note']} |")
    for alarm in result["verdict"]["alarms"]:
        lines.append(f"| 경보 | ! | {alarm} |")
    for title, rows in (("언어", result["stats"]["by_lang"]), ("나눔", result.get("by", []))):
        if not rows:
            continue
        lines += ["", f"| {title} | 쌍 | d 쌍 | d 재현 base → new (차) | b 쌍 | b 재현 base → new (차) "
                      "| 3칸 base → new (차) |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            d, b, a = r["d_recall"], r["b_recall"], r["acc3"]
            lines.append(f"| {r['key']} | {r['pairs']} | {r['d_pairs']} | {fmt(d['base'])} → {fmt(d['new'])} ({fmt(d['diff'])}) "
                         f"| {r['b_pairs']} | {fmt(b['base'])} → {fmt(b['new'])} ({fmt(b['diff'])}) "
                         f"| {fmt(a['base'])} → {fmt(a['new'])} ({fmt(a['diff'])}) |")
    return "\n".join(lines) + "\n"


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.4f}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="짝 비교 (씨앗 여럿 · 같은 쌍 · 묶음 붓스트랩)")
    p.add_argument("--protocol", required=True, help="measure/protocol-g1.json")
    p.add_argument("--set", required=True, help="잰 셋 jsonl (dev-cmp) — group · domain · lang 을 읽고 해시를 견준다")
    p.add_argument("--base", required=True, help="이름=판1,판2,판3")
    p.add_argument("--new", required=True, help="이름=판1,판2,판3")
    p.add_argument("--by", default="", help="쉼표로 칸 이름 (domain,lang) — 칸 조합별 표")
    p.add_argument("--out", default="", help="결과 json")
    p.add_argument("--md", default="", help="결과 마크다운 (비면 표준 출력만)")
    return p.parse_args(argv)


def run(argv=None):
    a = parse_args(argv)
    protocol, protocol_sha = read_protocol(a.protocol)
    set_sha = sha256_file(a.set)
    set_rows = read_set(a.set)
    base_name, base_runs = load_runs(a.base)
    new_name, new_runs = load_runs(a.new)
    check_hashes(protocol, protocol_sha, set_sha, [(base_name, base_runs), (new_name, new_runs)])
    stats = compare(protocol, set_rows, base_runs, new_runs)
    base, new = stats.pop("_tables")
    result = {"base": base_name, "new": new_name, "protocol_sha256": protocol_sha, "set_sha256": set_sha,
              "set": os.path.basename(a.set), "stats": stats, "verdict": verdict(protocol, stats)}
    if a.by:
        result["by"] = by_table(set_rows, base["ids"], base, new, [k.strip() for k in a.by.split(",") if k.strip()])
    md = to_markdown(result)
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
    try:
        run(argv)
    except Refuse as e:
        print(f"거부: {e}", file=sys.stderr)
        return REFUSE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
