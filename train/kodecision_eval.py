"""0단계 — ko-decision 을 학습 없이 쌍 jsonl 에 한 번 잰다 (설계 1절). 서빙에 안 들어가는 한 번 쓰는 평가.

    python -I kodecision_eval.py --model /work/hf/ko-decision --pairs /work/data/mem/k48.jsonl \
        --out-dir /work/runs/kodecision-k48 --device cuda

모델 카드 꼴 그대로 : 선택지마다 (지시 + 선택지, 상태) 쌍 하나 → 점수 하나 → 한 질문의 선택지끼리 softmax.
상태는 「전제: 근거\\n가설: 주장」, 선택지는 카드 예시의 NLI 세 줄(함의 · 중립 · 모순)이다.
출력 칸은 a·b·c(지지 · 반대 · 무관) 차례로 바꿔 적는다 — 함의 → a · 모순 → b · 중립 → c.
쌍 하나 = 전진 한 번(선택지 3개 묶음)이라 쌍당 ms 는 서빙 꼴(3배 계산)의 값이다.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from calibrate import softmax  # noqa: E402
from data_prep import LETTERS, read_jsonl, read_pairs, sha256_file  # noqa: E402
from train_nli import metrics_from  # noqa: E402

INSTRUCTION = "전제에 대해 가설이 갖는 논리적 관계를 판정하세요."
# 카드 예시 차례 (함의 · 중립 · 모순)
OPTIONS = ["entailment: 가설이 전제로부터 반드시 참이다 (함의)",
           "neutral: 가설이 전제로부터 참인지 거짓인지 알 수 없다 (중립)",
           "contradiction: 가설이 전제와 모순된다 (모순)"]
# 카드 차례 → 출력 칸 (a 지지 · b 반대 · c 무관)
OPTION_TO_LABEL = [0, 2, 1]
CARD_MAX_LEN = 512
CARD_TEMPERATURE_KEY = "klue_nli"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ko-decision 무학습 평가 (0단계)")
    p.add_argument("--model", required=True, help="ko-decision 폴더")
    p.add_argument("--pairs", required=True, help="쌍 jsonl (mem 쌍 꼴 또는 공통 꼴)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--max-len", type=int, default=CARD_MAX_LEN)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


def state_text(row):
    return f"전제: {row['premise']}\n가설: {row['hypothesis']}"


def to_abc(card_scores):
    """카드 차례 점수 [함의, 중립, 모순] → a·b·c 차례."""
    out = [0.0, 0.0, 0.0]
    for i, score in enumerate(card_scores):
        out[OPTION_TO_LABEL[i]] = score
    return out


def card_temperature(model_dir):
    path = os.path.join(model_dir, "calibration.json")
    if not os.path.exists(path):
        return 1.0
    with open(path, encoding="utf-8") as f:
        return json.load(f)["temperatures"].get(CARD_TEMPERATURE_KEY, 1.0)


def score_pair(model, tokenizer, row, max_len, device):
    """쌍 하나의 카드 차례 점수 셋 · 잘렸나."""
    texts = [f"{INSTRUCTION} {o}" for o in OPTIONS]
    state = state_text(row)
    batch = tokenizer(texts, [state] * len(OPTIONS), truncation="only_second", max_length=max_len,
                      padding=True, return_tensors="pt", return_overflowing_tokens=False)
    full = len(tokenizer(texts[0], state)["input_ids"])
    batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    scores = model(**batch).logits[:, 0].float().tolist()
    return scores, full > max_len


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    a = parse_args(argv)
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda 를 못 잡았다")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    if model.config.num_labels != 1:
        raise SystemExit(f"카드 꼴은 점수 하나다 — num_labels {model.config.num_labels}")
    temperature = card_temperature(a.model)
    rows = read_pairs(a.pairs)
    raw = [r for r in read_jsonl(a.pairs) if "_meta" not in r]
    kinds = [r.get("kind") or r.get("rule", "") for r in raw]

    with torch.inference_mode():
        score_pair(model, tokenizer, rows[0], a.max_len, a.device)  # 첫 판 커널 준비는 시간에서 뺀다
        results, times, truncated = [], [], 0
        for row in rows:
            t0 = time.perf_counter()
            scores, cut = score_pair(model, tokenizer, row, a.max_len, a.device)
            if a.device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            truncated += 1 if cut else 0
            results.append(to_abc(scores))

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "pred.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for row, kind, abc in zip(rows, kinds, results):
            label = row.get("label") if row.get("label") in (0, 1, 2) else None
            f.write(json.dumps({"id": row.get("id", ""), "label": label, "logits": [round(x, 6) for x in abc],
                                "probs": [round(x, 6) for x in softmax(abc, 1.0)],
                                "probs_card_t": [round(x, 6) for x in softmax(abc, temperature)],
                                "pred": LETTERS[max(range(3), key=lambda i: abc[i])], "kind": kind},
                               ensure_ascii=False) + "\n")
    lab = [i for i, r in enumerate(rows) if r.get("label") in (0, 1, 2)]
    metrics = metrics_from([results[i] for i in lab], [rows[i]["label"] for i in lab], [kinds[i] for i in lab])
    ordered = sorted(times)
    summary = {"model": a.model, "pairs": os.path.basename(a.pairs), "pairs_sha256": sha256_file(a.pairs),
               "rows": len(rows), "labeled": len(lab), "device": a.device, "max_len": a.max_len,
               "truncated": truncated, "card_temperature": temperature, "options": OPTIONS,
               "instruction": INSTRUCTION, "metrics": metrics,
               "ms_per_pair": {"mean": round(sum(times) / len(times), 1),
                               "p50": round(ordered[len(ordered) // 2], 1),
                               "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1)}}
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
