"""학습한 모델(HF 폴더)로 쌍 jsonl 을 평가한다 → summary.json · 쌍별 로짓·확률 jsonl.

    python -I eval_nli.py --model runs/studio-1/ckpt-best --pairs mem/dev.jsonl --out-dir runs/studio-1/eval-dev \
        --max-len 256 --device cpu --dtype fp32

쌍별 줄 {"id", "label", "logits", "probs", "pred", "kind", "src"} 은 calibrate.py 의 입력이다.
`--calibration` 을 주면 probs 에 그 온도를 건다 (logits 는 늘 온도 전 값).
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
from data_prep import LETTERS, read_pairs, sha256_file  # noqa: E402
from train_nli import build_model, evaluate, load_tokenizer, metrics_from, tokenize  # noqa: E402


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
    model = build_model(a.model, a.device, False)
    rows = read_pairs(a.pairs)
    labeled = [r for r in rows if r.get("label") in (0, 1, 2)]
    # 라벨 없는 쌍(길이만 재는 셋)도 확률은 낸다 — 평가 수치는 라벨 있는 쌍만
    for_items = [dict(r, label=r["label"] if r.get("label") in (0, 1, 2) else 0) for r in rows]
    items, truncated = tokenize(for_items, tokenizer.backend_tokenizer, a.max_len, None)
    _, logits = evaluate(model, items, a.batch, tokenizer.pad_token_id, a.device, a.dtype == "bf16", want_logits=True)

    os.makedirs(a.out_dir, exist_ok=True)
    pred_path = os.path.join(a.out_dir, "pred.jsonl")
    with open(pred_path, "w", encoding="utf-8", newline="\n") as f:
        for r, lg in zip(rows, logits):
            probs = softmax(lg, temperature)
            label = r.get("label") if r.get("label") in (0, 1, 2) else None
            f.write(json.dumps({"id": r.get("id", ""), "label": label, "logits": [round(x, 6) for x in lg],
                                "probs": [round(x, 6) for x in probs],
                                "pred": LETTERS[max(range(3), key=lambda i: lg[i])],
                                "kind": r.get("kind", ""), "src": r.get("src", "")}, ensure_ascii=False) + "\n")
    lab_idx = [i for i, r in enumerate(rows) if r.get("label") in (0, 1, 2)]
    metrics = metrics_from([logits[i] for i in lab_idx], [rows[i]["label"] for i in lab_idx],
                           [rows[i].get("kind", "") for i in lab_idx])
    summary = {"model": os.path.abspath(a.model), "pairs": os.path.basename(a.pairs),
               "pairs_sha256": sha256_file(a.pairs), "rows": len(rows), "labeled": len(labeled),
               "max_len": a.max_len, "truncated": truncated, "temperature": temperature, "metrics": metrics}
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
