"""칸 수에 매이지 않는 셈 — 갈래 가중치 · 평가 수치 · 합친 칸 손실의 기준값 (판정 넷 확장 설계 3-3 · 5절).

torch 없이 돈다. train_nli · eval_nli 가 가져다 쓰고, torch 가 없는 PC 에서도 시험한다.
칸 수는 로짓 길이(= 모델 머리 칸 수)에서 읽는다. 3칸 모델과 4칸 모델이 같은 함수를 쓴다.
"""
import math

from data_prep import CD, CD_COLS, LETTERS, is_label, train_target


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def fold3(label):
    """넷 칸 → 옛 세 칸 (d 와 CD 를 c 로). 「3칸으로 접은 정확도」에 쓴다."""
    return 2 if label in (3, CD) else label


def class_weights(rows, num_labels=3):
    """mem 쌍 갈래 수의 역수로 만든 [wa, wb, wc(, wd)].

    「갈래 평균 1」 = mem 쌍이 하나라도 있는 갈래들의 가중치를 단순 평균하면 1 이다 (쌍 수로 가중한 평균이 아니다).
    mem 쌍이 없는 갈래는 0 이고 평균에서 뺀다. mem 쌍이 아예 없으면 모두 1.
    `label` 은 train_target 을 거친 목표다 — CD 쌍은 어느 갈래에도 안 세고 가중치 1 을 쓴다.
    """
    counts = [0] * num_labels
    for r in rows:
        if r["src"] == "mem" and 0 <= r["label"] < num_labels:
            counts[r["label"]] += 1
    present = [c for c in counts if c]
    if not present:
        return [1.0] * num_labels
    inv = [1.0 / c if c else 0.0 for c in counts]
    mean = sum(inv) / len(present)
    return [w / mean for w in inv]


def parse_weight_scale(text):
    """`a=1.0,b=1.2` → {칸 번호: 배수}. 빈 글이면 {}. 모르는 글자·0 이하 값이면 ValueError."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        letter, _, value = part.partition("=")
        letter = letter.strip().lower()
        if letter not in LETTERS:
            raise ValueError(f"모르는 칸 글자 {letter!r} (쓸 수 있는 것 {LETTERS})")
        factor = float(value)
        if factor <= 0:
            raise ValueError(f"{letter} 배수는 0 보다 커야 한다: {factor}")
        out[LETTERS.index(letter)] = factor
    return out


def scale_weights(weights, scale):
    """갈래 가중에 칸별 배수를 곱하고, 쌍이 있는 갈래(가중 > 0)의 단순 평균이 다시 1 이 되게 나눈다.

    설계 8절 「반대 재현이 떨어지면 b 를 1.2 로 한 번만 올림」에 쓴다. scale 이 비면 그대로 돌려준다.
    """
    if not scale:
        return list(weights)
    out = [w * scale.get(i, 1.0) for i, w in enumerate(weights)]
    present = [w for w in out if w > 0]
    if not present:
        return out
    mean = sum(present) / len(present)
    return [w / mean for w in out]


def row_weight(weights, row):
    """한 줄의 손실 가중. mem 쌍만 갈래 가중을 받고, CD 쌍은 1."""
    if weights is None or row["src"] != "mem" or row["label"] == CD:
        return 1.0
    return weights[row["label"]]


def nll_ref(logits, label):
    """한 쌍의 손실 기준값 (torch 판 forward_loss 와 같은 셈).

    보통 라벨은 −log p_label, CD 는 합친 칸 손실 −log(p_c + p_d) 이다. 3칸 로짓이면 CD 는 −log p_c 와 같다.
    """
    top = max(logits)
    log_total = top + math.log(sum(math.exp(x - top) for x in logits))
    if label == CD:
        cols = [i for i in CD_COLS if i < len(logits)]
        sub_top = max(logits[i] for i in cols)
        log_sub = sub_top + math.log(sum(math.exp(logits[i] - sub_top) for i in cols))
        return log_total - log_sub
    return log_total - logits[label]


def metrics_from(logits, labels, kinds):
    """정확도 · 갈래별 재현 · 거짓 지지 · 혼동표[정답][예측] · 3칸으로 접은 정확도 · 지지 kind 별 재현.

    - acc : 정답이 한 칸인 쌍만 센다 (CD 쌍 제외). 3칸 모델이면 옛 정확도와 같다.
    - acc3 : c·d 를 한 칸으로 합친 정확도. CD 쌍은 예측이 c 나 d 면 맞힘이다 (설계 4-2 「3칸으로 접어서」).
    - false_support : 정답이 지지가 아닌데(CD 포함) 최고 확률 답이 지지인 쌍 수.
    """
    n_cols = len(logits[0]) if logits else 3
    confusion = [[0] * n_cols for _ in range(n_cols)]
    exact = folded = definite = cd_n = cd_hit = false_support = 0
    for lg, gold in zip(logits, labels):
        pred = argmax(lg)
        if pred == 0 and gold != 0:
            false_support += 1
        if fold3(pred) == fold3(gold):
            folded += 1
        if gold == CD:
            cd_n += 1
            cd_hit += 1 if pred in CD_COLS else 0
            continue
        definite += 1
        confusion[gold][pred] += 1
        exact += 1 if pred == gold else 0
    recall = {}
    for i in range(n_cols):
        total = sum(confusion[i])
        recall[LETTERS[i]] = round(confusion[i][i] / total, 4) if total else None
    by_kind = {}
    for lg, gold, kind in zip(logits, labels, kinds):
        if gold != 0 or not kind:
            continue
        hit, total = by_kind.get(kind, (0, 0))
        by_kind[kind] = (hit + (1 if argmax(lg) == 0 else 0), total + 1)
    out = {"n": len(labels), "acc": round(exact / max(1, definite), 4), "acc3": round(folded / max(1, len(labels)), 4),
           "recall": recall, "false_support": false_support, "confusion": confusion,
           "support_recall_by_kind": {k: round(h / t, 4) for k, (h, t) in sorted(by_kind.items())}}
    if cd_n:
        out["cd"] = {"n": cd_n, "hit": cd_hit}
    return out


def min_recall(recall):
    """갈래별 재현 중 가장 낮은 값. 쌍이 없는 갈래(None)는 빼고, 다 None 이면 0 이다 (studio 고르기 · 설계 3-2)."""
    return min((v for v in recall.values() if v is not None), default=0)


def eval_targets(rows, num_labels, public_neutral="cd"):
    """평가 줄마다 이 모델의 목표 라벨. 라벨 없는 줄은 None.

    public_neutral 은 학습 때 쓴 --replay-neutral 과 맞춘다 — c 로 배운 모델을 cd 로 채점하면 c·d 를 헷갈린 셈이 흐려진다.
    """
    return [train_target(r, num_labels, public_neutral) if is_label(r.get("label")) else None for r in rows]


def epoch_end_metrics(metrics, history):
    """바퀴 끝 검사점(--save-every-epoch)에 적을 dev 수치.

    run_eval 이 같은 스텝에서 이미 재서 None 을 돌려주면 그때 잰 dev_history 마지막 줄을 쓴다 (epoch · step 칸은 뺀다).
    dev 셋이 없어 기록도 없으면 None.
    """
    if metrics is not None or not history:
        return metrics
    return {k: v for k, v in history[-1].items() if k not in ("epoch", "step")}
