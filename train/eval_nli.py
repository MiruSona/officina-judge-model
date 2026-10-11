"""학습한 모델(HF 폴더)로 쌍 jsonl 을 평가한다 → summary.json · 쌍별 로짓·확률 jsonl.

    python -I eval_nli.py --model runs/studio-1/ckpt-best --pairs mem/dev.jsonl --out-dir runs/studio-1/eval-dev \
        --max-len 256 --device cpu --dtype fp32

쌍별 줄 {"id", "label", "logits", "probs", "pred", "kind", "src"} 은 calibrate.py 의 입력이다.
`--calibration` 을 주면 probs 에 그 온도를 건다 (logits 는 늘 온도 전 값).
`--public-neutral` 은 학습 때 준 `--replay-neutral` 과 맞춘다 — 기본 cd(옛 동작). c 로 배운 모델은 c 로 채점한다.
칸 수는 모델 머리에서 읽는다. label 은 그 모델이 배우는 목표(data_prep.train_target)다 — 4칸 모델에서
공개 NLI 중립 · want `cd` 쌍은 -2(CD, 「c 나 d」)로 적는다. 라벨 없는 쌍은 null.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from calibrate import softmax  # noqa: E402
from data_prep import LETTERS, PUBLIC_NEUTRAL_MODES, read_pairs, sha256_file  # noqa: E402
from heads import argmax, eval_targets, metrics_from  # noqa: E402
from train_nli import build_model, evaluate, load_tokenizer, tokenize  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NLI 모델 평가")
    p.add_argument("--model", required=True, help="HF 모델 폴더 (ckpt-best)")
    p.add_argument("--pairs", required=True, help="쌍 jsonl (공통 꼴 또는 mem 쌍 꼴)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--calibration", default="", help="calibration.json — probs 에 온도를 건다")
    p.add_argument("--public-neutral", choices=list(PUBLIC_NEUTRAL_MODES), default="cd",
                   help="4칸 모델에서 공개 NLI 중립을 채점할 칸 — 학습의 --replay-neutral 과 같게 (기본 cd)")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    temperature = 1.0
    if a.calibration:
        with open(a.calibration, encoding="utf-8") as f:
            temperature = json.load(f)["temperature"]
    tokenizer = load_tokenizer(a.model)
    model = build_model(a.model, a.device)
    n_labels = model.config.num_labels
    rows = read_pairs(a.pairs)
    # 라벨을 이 모델의 목표로 바꾼다 (3칸이면 d·CD → c, 4칸이면 공개 중립 → CD 또는 c). 라벨 없는 줄은 None
    targets = eval_targets(rows, n_labels, a.public_neutral)
    labeled = [t for t in targets if t is not None]
    # 라벨 없는 쌍(길이만 재는 셋)도 확률은 낸다 — 평가 수치는 라벨 있는 쌍만
    for_items = [dict(r, label=t if t is not None else 0) for r, t in zip(rows, targets)]
    items, truncated = tokenize(for_items, tokenizer.backend_tokenizer, a.max_len, None, n_labels)
    _, logits = evaluate(model, items, a.batch, tokenizer.pad_token_id, a.device, a.dtype == "bf16", want_logits=True)

    os.makedirs(a.out_dir, exist_ok=True)
    pred_path = os.path.join(a.out_dir, "pred.jsonl")
    with open(pred_path, "w", encoding="utf-8", newline="\n") as f:
        for r, t, lg in zip(rows, targets, logits):
            probs = softmax(lg, temperature)
            f.write(json.dumps({"id": r.get("id", ""), "label": t, "logits": [round(x, 6) for x in lg],
                                "probs": [round(x, 6) for x in probs], "pred": LETTERS[argmax(lg)],
                                "kind": r.get("kind", ""), "src": r.get("src", "")}, ensure_ascii=False) + "\n")
    lab_idx = [i for i, t in enumerate(targets) if t is not None]
    metrics = metrics_from([logits[i] for i in lab_idx], [targets[i] for i in lab_idx],
                           [rows[i].get("kind", "") for i in lab_idx])
    summary = {"model": os.path.abspath(a.model), "labels": n_labels, "pairs": os.path.basename(a.pairs),
               "pairs_sha256": sha256_file(a.pairs), "rows": len(rows), "labeled": len(labeled),
               "max_len": a.max_len, "truncated": truncated, "temperature": temperature,
               "public_neutral": a.public_neutral, "metrics": metrics}
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
