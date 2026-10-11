"""dev 로짓으로 온도 T 와 문턱을 고른다 → calibration.json (설계 4-4 · 7절 · 판정 넷 확장 설계 4-3).

    python -I calibrate.py --pred runs/<id>/dev-pred.jsonl --out runs/<id>/calibration.json
    python -I calibrate.py --pred … --out … --temperature-only      # 기본 모델 (KLUE dev)

입력은 eval_nli.py 가 쓰는 쌍별 줄 {"id", "label": 0|1|2|3|-2, "logits": [a, b, c(, d)]} 이다. 칸 수는 로짓 길이다.
label -2 (CD, 「c 나 d」)는 NLL 에서 p_c + p_d 로 세고, 예측이 c·d 면 맞힘으로 친다. 문턱은 온도 적용 뒤 확률로 고른다.
- nli_sure_support : 0.50~0.99 중 dev 거짓 지지 ≤ floor(지지 아닌 쌍 수 × 몫) 를 만족하는 가장 작은 값.
  몫은 3칸 2% · 4칸 1% (4칸 자가 1.5% 라 여유를 둔다, `--false-support-share` 로 바꿈)
- nli_sure : 같은 간격에서 dev 확신 B·C 답 정확도 ≥ 0.90 을 만족하는 가장 작은 값
- nli_sure_related (4칸만) : 같은 간격에서 dev 확신 D 답 정밀(정답 d 몫) ≥ 0.80 을 만족하는 가장 작은 값
- 고른 값들의 dev 보류가 30% 를 넘으면 nli_sure 만 내려 맞춘다 (nli_sure_support · nli_sure_related 는 안 내린다)
「담김」 문장 판정은 따로 문턱이 없다 — nli_sure_support 를 쓴다 (설계 4-3 의 3).
"""
import argparse
import hashlib
import json
import math
import os
import sys

T_GRID = [round(0.5 + 0.05 * i, 2) for i in range(51)]          # 0.50 ~ 3.00
SURE_GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]      # 0.50 ~ 0.99
FALSE_SUPPORT_SHARE = {3: 0.02, 4: 0.01}
SURE_ACC = 0.90
RELATED_PRECISION = 0.80
UNSURE_MAX = 0.30
# Go 판정기와 같은 「모른다」 선 (AIMemoryTool judge.go 의 UnsureBelow · tieGap 과 같은 뜻)
UNSURE_BELOW = 0.40
TIE_GAP = 0.01
SUPPORT = 0
RELATED = 3
CD = -2              # data_prep.CD 와 같다 (이 파일은 torch·data_prep 없이도 돌게 따로 둔다)
CD_COLS = (2, 3)
ECE_BINS = 10


def softmax(logits, temperature):
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def gold_prob(probs, label):
    """정답 칸의 확률. CD 는 c·d 합 (있는 칸만)."""
    if label == CD:
        return sum(probs[i] for i in CD_COLS if i < len(probs))
    return probs[label]


def is_hit(top, label):
    """최고 칸이 정답인가. CD 정답은 c 나 d 면 맞힘."""
    if label == CD:
        return top in CD_COLS
    return top == label


def nll(rows, temperature):
    total = 0.0
    for row in rows:
        p = gold_prob(softmax(row["logits"], temperature), row["label"])
        total -= math.log(max(p, 1e-12))
    return total / len(rows)


def ece(rows, temperature, bins=ECE_BINS):
    """기대 눈금 오차 — 최고 확률 구간별 |정확도 − 평균 확신| 의 가중 합."""
    buckets = [[0, 0.0, 0] for _ in range(bins)]
    for row in rows:
        probs = softmax(row["logits"], temperature)
        top = max(range(len(probs)), key=lambda i: probs[i])
        k = min(bins - 1, int(probs[top] * bins))
        buckets[k][0] += 1
        buckets[k][1] += probs[top]
        buckets[k][2] += 1 if is_hit(top, row["label"]) else 0
    total = 0.0
    for count, conf, hit in buckets:
        if count:
            total += abs(hit / count - conf / count) * count
    return total / len(rows)


def fit_temperature(rows):
    return min(T_GRID, key=lambda t: (nll(rows, t), t))


def answer(probs):
    """(최고 칸, 그 확률, Go 의 기본 「모른다」인가). 칸 수는 probs 길이."""
    top = max(range(len(probs)), key=lambda i: probs[i])
    second = max(probs[i] for i in range(len(probs)) if i != top)
    base_unsure = probs[top] < UNSURE_BELOW or probs[top] - second < TIE_GAP
    return top, probs[top], base_unsure


def is_sure(probs, sure, sure_support, sure_related=None):
    """확신 답인가. 지지는 sure_support, d 는 sure_related(없으면 sure), 나머지는 sure 와 견준다."""
    top, p, base_unsure = answer(probs)
    if base_unsure:
        return False
    if top == SUPPORT:
        return p >= sure_support
    if top == RELATED and sure_related is not None:
        return p >= sure_related
    return p >= sure


def false_support(scored, sure_support):
    """정답이 지지가 아닌데(B·C·D·CD) 확신 있게 A 로 본 쌍 수."""
    n = 0
    for probs, label in scored:
        top, p, base_unsure = answer(probs)
        if top == SUPPORT and label != SUPPORT and not base_unsure and p >= sure_support:
            n += 1
    return n


def sure_bc_accuracy(scored, sure):
    """확신 B·C 답의 (정확도, 수). D 답은 nli_sure_related 몫이라 뺀다. 확신 답이 없으면 정확도 None."""
    hit = total = 0
    for probs, label in scored:
        top, p, base_unsure = answer(probs)
        if top in (SUPPORT, RELATED) or base_unsure or p < sure:
            continue
        total += 1
        hit += 1 if is_hit(top, label) else 0
    if total == 0:
        return None, 0
    return hit / total, total


def sure_related_precision(scored, sure_related):
    """확신 D 답의 (정밀, 수) — 확신 D 중 정답이 d 인 몫. CD 정답은 맞힘으로도 틀림으로도 안 센다."""
    hit = total = 0
    for probs, label in scored:
        top, p, base_unsure = answer(probs)
        if top != RELATED or base_unsure or p < sure_related or label == CD:
            continue
        total += 1
        hit += 1 if label == RELATED else 0
    if total == 0:
        return None, 0
    return hit / total, total


def unsure_rate(scored, sure, sure_support, sure_related=None):
    return sum(1 for probs, _ in scored if not is_sure(probs, sure, sure_support, sure_related)) / len(scored)


def pick_thresholds(scored, false_support_share=None):
    """설계 7절 1~3 (+ 넷 확장 4-3). 고른 값과 근거 수를 dict 로. 칸 수는 확률 길이에서 읽는다."""
    n_labels = len(scored[0][0])
    if false_support_share is None:
        false_support_share = FALSE_SUPPORT_SHARE.get(n_labels, FALSE_SUPPORT_SHARE[3])
    non_support = sum(1 for _, label in scored if label != SUPPORT)
    allowed = math.floor(non_support * false_support_share)
    sure_support = None
    for s in SURE_GRID:
        if false_support(scored, s) <= allowed:
            sure_support = s
            break
    sure = None
    for t in SURE_GRID:
        acc, _ = sure_bc_accuracy(scored, t)
        if acc is not None and acc >= SURE_ACC:
            sure = t
            break
    notes = []
    sure_related = None
    if n_labels > RELATED:
        for t in SURE_GRID:
            prec, _ = sure_related_precision(scored, t)
            if prec is not None and prec >= RELATED_PRECISION:
                sure_related = t
                break
        if sure_related is None:
            sure_related = SURE_GRID[-1]
            notes.append("nli_sure_related: 자를 만족하는 값이 없어 0.99")
    if sure_support is None:
        sure_support = SURE_GRID[-1]
        notes.append("nli_sure_support: 자를 만족하는 값이 없어 0.99")
    if sure is None:
        sure = SURE_GRID[-1]
        notes.append("nli_sure: 자를 만족하는 값이 없어 0.99")
    lowered_from = None
    rate = unsure_rate(scored, sure, sure_support, sure_related)
    if rate > UNSURE_MAX:
        lowered_from = sure
        for t in reversed([g for g in SURE_GRID if g < sure]):
            sure = t
            rate = unsure_rate(scored, sure, sure_support, sure_related)
            if rate <= UNSURE_MAX:
                break
        if rate > UNSURE_MAX:
            notes.append("보류가 30% 를 넘는다 — 그대로 두고 측정에서 판정받는다")
    acc, n_sure = sure_bc_accuracy(scored, sure)
    out = {"nli_sure_support": sure_support, "nli_sure": sure,
           "dev_false_support": false_support(scored, sure_support), "dev_false_support_allowed": allowed,
           "dev_false_support_share": false_support_share,
           "dev_non_support": non_support, "dev_sure_bc_accuracy": acc, "dev_sure_bc": n_sure,
           "dev_unsure_rate": round(rate, 4), "nli_sure_lowered_from": lowered_from, "notes": notes}
    if n_labels > RELATED:
        prec, n_rel = sure_related_precision(scored, sure_related)
        out.update({"nli_sure_related": sure_related, "dev_sure_related_precision": prec, "dev_sure_related": n_rel})
    return out


def calibrate(rows, temperature_only=False, false_support_share=None):
    t = fit_temperature(rows)
    out = {"temperature": t, "pairs": len(rows), "labels": len(rows[0]["logits"]),
           "nll_before": round(nll(rows, 1.0), 5), "nll_after": round(nll(rows, t), 5),
           "ece_before": round(ece(rows, 1.0), 5), "ece_after": round(ece(rows, t), 5)}
    if not temperature_only:
        scored = [(softmax(r["logits"], t), r["label"]) for r in rows]
        out["thresholds"] = pick_thresholds(scored, false_support_share)
    return out


def read_pred(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") is None:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"라벨 있는 줄이 없다: {path}")
    widths = {len(r["logits"]) for r in rows}
    if len(widths) != 1:
        raise SystemExit(f"로짓 칸 수가 줄마다 다르다: {sorted(widths)} — 한 모델의 출력만 넣는다")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="온도 · 문턱 고르기 (dev 로만)")
    p.add_argument("--pred", required=True, help="eval_nli.py 쌍별 출력 jsonl (logits · label)")
    p.add_argument("--out", required=True, help="calibration.json")
    p.add_argument("--fitted-on", default="", help="dev 셋 파일 (sha 를 적는다). 비면 --pred 의 sha")
    p.add_argument("--temperature-only", action="store_true", help="기본 모델 — 온도만")
    p.add_argument("--false-support-share", type=float, default=None,
                   help="거짓 지지 허용 몫 (기본 3칸 0.02 · 4칸 0.01)")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    rows = read_pred(a.pred)
    out = calibrate(rows, a.temperature_only, a.false_support_share)
    fitted = a.fitted_on or a.pred
    with open(fitted, "rb") as f:
        out["fitted_on"] = {"file": os.path.basename(fitted), "sha256": hashlib.sha256(f.read()).hexdigest()}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
