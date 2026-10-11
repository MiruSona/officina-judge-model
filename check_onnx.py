"""서빙 일치 검사 (설계 5-3) — 다 통과해야 모델을 꽂는다.

    python -I check_onnx.py --model export/nli-studio-v1 --torch-src runs/studio-1/ckpt-best \
        --pairs k48.jsonl --pairs mem/dev.jsonl --out check.txt [--url http://127.0.0.1:8092] [--int8]

torch 와 onnxruntime 이 한 venv 에 없으면 두 번에 나눠 돈다 :

    (torch 쪽)  python -I check_onnx.py --model <폴더> --torch-src <ckpt> --pairs … --write-ref ref.jsonl
    (ONNX 쪽)   python -I check_onnx.py --model <폴더> --ref ref.jsonl --pairs … --out check.txt [--url …]

- 토큰 일치 : transformers 토크나이저와 tokenizers(서버가 쓰는 것) ids 가 쌍마다 완전히 같다
- 확률 일치 : torch 판과 ONNX 판(serve.OnnxNLI) 확률 차 최대 ≤ 1e-3 · 최고 답 100% 같다
- p95 (--url) : 순차 HTTP 왕복 p95 ≤ 1.0초 (서버는 따로 띄워 둔다)
- int8 (--int8) : fp32 와 최고 답 일치 ≥ 99% · 라벨 있는 쌍 정확도 하락 ≤ 1점
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "train"))

from data_prep import read_pairs  # noqa: E402
from pairenc import encode_pair  # noqa: E402
from serve import OnnxNLI, load_spec, softmax  # noqa: E402

LIMIT_DIFF = 1e-3
LIMIT_P95_MS = 1000.0
LIMIT_INT8_AGREE = 0.99
LIMIT_INT8_ACC_DROP = 0.01


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def hf_ids(tokenizer, evidence, claim, max_len):
    """transformers 쪽 셈 — pairenc 와 같은 물러서기 규칙을 transformers 로 따로 셈한다."""
    claim_len = len(tokenizer(claim, add_special_tokens=False)["input_ids"])
    strategy = "only_first"
    if claim_len + tokenizer.num_special_tokens_to_add(pair=True) >= max_len:
        strategy = "longest_first"
    return tokenizer(evidence, claim, truncation=strategy, max_length=max_len)["input_ids"]


def ask(url, evidence, claim):
    data = json.dumps({"evidence": evidence, "claim": claim}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url + "/judge", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def torch_refs(model_dir, torch_src, rows, max_len, threads):
    """torch 쪽 기준값 — 쌍마다 transformers ids 와, 서버와 같은 ids 로 돌린 torch logits."""
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    hf_tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), use_fast=True)
    fast = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
    model = AutoModelForSequenceClassification.from_pretrained(torch_src).float().eval()
    refs = []
    for r in rows:
        ids, _ = encode_pair(fast, r["premise"], r["hypothesis"], max_len)
        with torch.no_grad():
            t = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids=t, attention_mask=torch.ones_like(t)).logits[0].tolist()
        refs.append({"id": r.get("id", ""), "hf_ids": hf_ids(hf_tok, r["premise"], r["hypothesis"], max_len),
                     "logits": [float(x) for x in logits]})
    return refs


def write_ref(path, spec, refs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        meta = {"weights_sha256": spec["weights_sha256"], "max_len": spec["input"]["max_len"], "pairs": len(refs)}
        f.write(json.dumps({"_meta": meta}) + "\n")
        for ref in refs:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")


def read_ref(path, spec, rows):
    """기준값 파일을 읽는다. 모델 · max_len · 쌍 차례가 이번 검사와 다르면 SystemExit."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    meta = lines[0].get("_meta") or {}
    if meta.get("weights_sha256") != spec["weights_sha256"] or meta.get("max_len") != spec["input"]["max_len"]:
        raise SystemExit(f"기준값의 모델·max_len 이 다르다: {meta}")
    refs = lines[1:]
    if [x["id"] for x in refs] != [r.get("id", "") for r in rows]:
        raise SystemExit("기준값의 쌍 차례가 --pairs 와 다르다")
    return refs


def compare(onnx, rows, refs, max_len, temperature):
    """토큰 · 확률 일치. (결과 칸, fp32 확률들, 통과) 를 돌려준다."""
    if not rows:
        raise SystemExit("견줄 쌍이 없다 — --pairs 를 본다")
    token_bad, diffs, agree, probs = [], [], 0, []
    for r, ref in zip(rows, refs):
        got_ids, _ = encode_pair(onnx.tokenizer, r["premise"], r["hypothesis"], max_len)
        if got_ids != ref["hf_ids"]:
            token_bad.append(r.get("id", ""))
        want = softmax(ref["logits"], temperature)
        got = onnx.probs(r["premise"], r["hypothesis"])
        probs.append(got)
        diffs.append(max(abs(x - y) for x, y in zip(want, got)))
        agree += 1 if argmax(want) == argmax(got) else 0
    fields = {"token_mismatch": len(token_bad), "token_mismatch_ids": token_bad[:20],
              "prob_diff_max": max(diffs), "top_agree": f"{agree}/{len(rows)}"}
    passed = not token_bad and max(diffs) <= LIMIT_DIFF and agree == len(rows)
    return fields, probs, passed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ONNX 서빙 일치 검사")
    p.add_argument("--model", required=True, help="서빙 모델 폴더 (export_onnx.py 결과)")
    p.add_argument("--torch-src", default="", help="HF 모델 폴더 — 비면 <model>/torch")
    p.add_argument("--pairs", action="append", required=True, help="쌍 jsonl (여러 번)")
    p.add_argument("--write-ref", default="", help="torch 기준값만 이 파일에 쓰고 끝낸다 (onnxruntime 불필요)")
    p.add_argument("--ref", default="", help="--write-ref 로 쓴 기준값 — 주면 torch 없이 견준다")
    p.add_argument("--url", default="", help="서버 주소 — 주면 HTTP p95 를 잰다")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--out", default="", help="결과를 덧붙일 txt (--write-ref 가 아니면 꼭)")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    if not a.out and not a.write_ref:
        raise SystemExit("--out 이 있어야 한다")
    model_dir = os.path.abspath(os.path.expanduser(a.model))
    spec = load_spec(model_dir)
    max_len = spec["input"]["max_len"]
    temperature = spec["_temperature"]
    rows = []
    for path in a.pairs:
        rows.extend(read_pairs(path))

    if a.ref:
        refs = read_ref(a.ref, spec, rows)
    else:
        refs = torch_refs(model_dir, a.torch_src or os.path.join(model_dir, "torch"), rows, max_len, a.threads)
    if a.write_ref:
        write_ref(a.write_ref, spec, refs)
        print(f"기준값 {len(refs)}쌍 씀 {a.write_ref}", flush=True)
        return 0

    onnx = OnnxNLI(model_dir, spec, a.threads)
    fields, probs_fp32, passed = compare(onnx, rows, refs, max_len, temperature)
    result = {"model": spec["name"], "weights": spec["weights_sha256"][:8], "pairs": len(rows),
              "files": [os.path.basename(p) for p in a.pairs], "max_len": max_len, "temperature": temperature,
              **fields}

    if a.int8:
        spec8 = dict(spec, onnx_file="model.int8.onnx")
        onnx8 = OnnxNLI(model_dir, spec8, a.threads)
        same, hit32, hit8, graded = 0, 0, 0, 0
        for r, p32 in zip(rows, probs_fp32):
            p8 = onnx8.probs(r["premise"], r["hypothesis"])
            same += 1 if argmax(p8) == argmax(p32) else 0
            if r.get("label") in range(len(p32)):
                graded += 1
                hit32 += 1 if argmax(p32) == r["label"] else 0
                hit8 += 1 if argmax(p8) == r["label"] else 0
        drop = (hit32 - hit8) / graded if graded else 0.0
        result["int8"] = {"top_agree": round(same / len(rows), 4), "acc_drop": round(drop, 4), "graded": graded}
        passed = passed and same / len(rows) >= LIMIT_INT8_AGREE and drop <= LIMIT_INT8_ACC_DROP

    if a.url:
        ask(a.url, "데우기", "데우기")
        lat = []
        for r in rows:
            start = time.perf_counter()
            ask(a.url, r["premise"], r["hypothesis"])
            lat.append((time.perf_counter() - start) * 1000)
        result["http_ms"] = {"p50": round(pct(lat, 0.5), 1), "p95": round(pct(lat, 0.95), 1), "max": round(max(lat), 1)}
        passed = passed and pct(lat, 0.95) <= LIMIT_P95_MS

    result["limits"] = {"prob_diff": LIMIT_DIFF, "p95_ms": LIMIT_P95_MS, "int8_agree": LIMIT_INT8_AGREE,
                        "int8_acc_drop": LIMIT_INT8_ACC_DROP}
    result["passed"] = passed
    text = json.dumps(result, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M')} check_onnx\n{text}\n")
    print(text, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
