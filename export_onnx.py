"""학습한 모델(HF 폴더)을 서빙 모델 폴더로 내보낸다 — model.onnx · tokenizer/ · model.json · calibration.json (설계 5-1 · 5-2).

    python -I export_onnx.py --src runs/studio-1/ckpt-best --out export/nli-studio-v1 --name nli-studio-v1 \
        --train-info runs/studio-1/train_info.json --calibration runs/studio-1/calibration.json \
        --parent export/nli-base-v2/model.json      # 몸통(body) 칸은 parent 에서 물려받는다
    # 기반 판은 parent 가 없으니 몸통을 직접 준다
    python -I export_onnx.py --src runs/base-2/ckpt-best --out export/nli-base-v2 --name nli-base-v2 --max-len 320 \
        --body-name FacebookAI/xlm-roberta-large --body-weights hf/xlm-roberta-large/model.safetensors

torch · transformers · onnx · onnxruntime 이 든 venv 에서 돈다. 입력은 input_ids · attention_mask(int64),
출력은 logits [batch, 칸 수] (온도 전). 칸 수는 모델 머리에서 읽는다 — 3칸(`nli-v1` 꼴) · 4칸(`nli-v2` 꼴) 둘 다 내보내고,
model.json 의 labels · letters · protocol · label_guide · label_set(nli3/nli4) 을 그 칸 수에 맞춰 적는다 (판정 넷 확장 설계 2-2).
`--model-version 1.1.0` 은 model.json 의 model_version 칸이 된다 — 서버의 opt-in 응답이 싣는 판 번호다 (d 자료층 재구축 설계 6-1).
몸통(klue roberta · XLM-R)은 type_vocab_size 1 이라 token_type_ids 를 안 넣는다 —
1 보다 크면 내보내지 않고 멈춘다 (pairenc.check_no_token_types).
가중치가 2GB 를 넘으면(XLM-R large fp32 2.24GB) protobuf 한도 때문에 torch 가 가중치를 낱개 파일 수백 개로 흩뜨린다 —
그때는 `model.onnx`(그래프) + `model.onnx.data`(가중치 한 파일) 둘로 모으고 model.json 에 둘의 SHA 를 다 적는다
(onnx_sha256 은 그래프만, onnx_data_sha256 은 가중치 파일). 한 파일 판이면 out 의 낡은 model.onnx.data 를 지운다.
torch 는 내보내는 함수 안에서만 부른다 — model.json 셈(build_spec)은 torch 없는 곳에서도 시험한다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "train"))

from data_prep import LABELS, LETTERS, SOURCE_LABELS  # noqa: E402
from pairenc import check_no_token_types, pair_template  # noqa: E402

OPSET = 17
FORMAT = 1
# 칸 수 → (응답 꼴 판 이름, 라벨 기준 글 판). 서버 · 손님이 이 이름으로 옛 판과 새 판을 가른다
PROTOCOLS = {3: ("nli-v1", "support-v3"), 4: ("nli-v2", "support-v4")}
# 칸 수 → 손님이 opt-in 으로 고르는 칸 세트 이름 (serve.LABEL_SETS 와 같은 이름, d 자료층 재구축 설계 6-1)
LABEL_SET_NAMES = {3: "nli3", 4: "nli4"}
ONNX_DATA = "model.onnx.data"
TORCH_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "sentencepiece.bpe.model")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wrap(model):
    """HF 모델을 입력 둘 · 출력 logits 하나로 감싼다."""
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    return Wrapper(model)


def gather_external(tmp_dir, onnx_path):
    """임시 폴더에 내보낸 그래프를 onnx_path 로 옮긴다. 가중치가 흩어졌으면 ONNX_DATA 한 파일로 모은다.

    돌려주는 값 : 모은 가중치 파일 경로 (흩어지지 않았으면 None — 그때 out 의 낡은 ONNX_DATA 는 지운다).
    """
    names = sorted(os.listdir(tmp_dir))
    tmp_onnx = os.path.join(tmp_dir, "model.onnx")
    if names == ["model.onnx"]:
        shutil.move(tmp_onnx, onnx_path)
        stale = os.path.join(os.path.dirname(onnx_path), ONNX_DATA)
        if os.path.exists(stale):
            os.remove(stale)
            print(f"낡은 {ONNX_DATA} 지움 (이번 판은 한 파일)", flush=True)
        return None
    import onnx

    data_path = os.path.join(os.path.dirname(onnx_path), ONNX_DATA)
    for path in (onnx_path, data_path):
        if os.path.exists(path):
            os.remove(path)
    proto = onnx.load(tmp_onnx, load_external_data=True)
    onnx.save_model(proto, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=ONNX_DATA, size_threshold=1024)
    print(f"가중치 {len(names) - 1}개 파일 → {ONNX_DATA} 하나로 모음 ({os.path.getsize(data_path) / 1e6:.0f}MB)", flush=True)
    return data_path


def export_graph(model, onnx_path, opset, pad_id):
    import torch

    # 둘째 줄 끝을 가려 마스크 길을 그래프에 남긴다 — 모두 1 이면 transformers 가 마스크를 빼고 그려 attention_mask 입력이 사라진다
    ids = torch.full((2, 16), pad_id + 5, dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[1, 12:] = 0
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}}
    t0 = time.time()
    tmp_dir = os.path.join(os.path.dirname(onnx_path), ".onnx-export")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    try:
        with torch.no_grad():
            torch.onnx.export(wrap(model).eval(), (ids, mask), os.path.join(tmp_dir, "model.onnx"),
                              input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes=axes, opset_version=opset, dynamo=False, do_constant_folding=True)
        data_path = gather_external(tmp_dir, onnx_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"ONNX 내보냄 {onnx_path} ({time.time() - t0:.0f}s, {os.path.getsize(onnx_path) / 1e6:.0f}MB)", flush=True)
    check_inputs(onnx_path)
    return data_path


def check_inputs(onnx_path):
    """그래프 입력이 input_ids · attention_mask 둘인지 본다. 서버가 둘 다 넣으므로 하나라도 빠지면 못 쓴다."""
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    names = [i.name for i in graph.input]
    if names != ["input_ids", "attention_mask"]:
        raise SystemExit(f"그래프 입력이 다르다: {names}")


def smoke_compare(model, onnx_path, vocab_size):
    """길이가 다른 열 둘 · 배치 2 로 torch 와 ONNX logits 를 견준다 — 동적 축이 굳지 않았는지 본다.

    onnxruntime 이 없는 venv 면 건너뛰고 None 이다 — 그때는 check_onnx.py 의 확률 일치가 같은 몫을 한다.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime 없음 — 연기 견줌 건너뜀 (check_onnx.py 로 잰다)", flush=True)
        return None
    import numpy as np
    import torch

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for batch, seq in ((1, 7), (2, 200)):
        ids = torch.randint(5, vocab_size, (batch, seq), generator=gen)
        mask = torch.ones_like(ids)
        got = sess.run(["logits"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        with torch.no_grad():
            want = model(input_ids=ids, attention_mask=mask).logits.numpy()
        worst = max(worst, float(np.abs(got - want).max()))
    print(f"연기 견줌 logits 차 최대 {worst:.2e}", flush=True)
    return worst


def make_int8(onnx_path, out_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    print(f"int8 만듦 {out_path} ({os.path.getsize(out_path) / 1e6:.0f}MB)", flush=True)


def resolve_body(body_name, body_sha, parent):
    """model.json body 칸. --body-name 이 있으면 그것, 없으면 parent 의 body 를 물려받는다. 둘 다 없으면 멈춘다.

    물려받을 때 parent body 의 다른 칸(sha256 · license 등)도 같이 오고, --body-weights 를 줬으면 sha256 만 새 값.
    --body-name 을 줬고 parent 몸통 이름이 같으면 parent 의 sha256 · license 를 빈 자리에 채운다.
    기본값을 두지 않는 까닭 : 옛 klue 체크포인트에 XLM-R 이름·라이선스가 조용히 적힐 수 있었다.
    """
    inherited = dict((parent or {}).get("body") or {})
    if body_name:
        body = {"name": body_name, "sha256": body_sha}
        if inherited.get("name") == body_name:
            for key, value in inherited.items():
                if body.get(key) is None:
                    body[key] = value
        return body
    if not inherited.get("name"):
        raise SystemExit("몸통을 모른다 — --body-name 을 주거나, body 칸이 있는 --parent model.json 을 준다")
    if body_sha:
        inherited["sha256"] = body_sha
    return inherited


def data_section(train_info):
    """train_info.json 의 data 칸 → model.json data 칸. nc 가 하나라도 참이면 공개 금지."""
    rows = []
    for d in (train_info or {}).get("data", []):
        rows.append({"name": d["file"], "sha256": d["sha256"], "rows": d["rows"], "license": d["license"],
                     "nc": bool(d["nc"]), "private": "mem" in d.get("by_src", {})})
    return rows


def build_spec(a, weights_sha, onnx_sha, tok_sha, calibration, train_info, parent, body_sha, smoke_diff, pair="",
               onnx_data_sha=None, num_labels=3):
    """model.json 내용. pair 는 쌍 꼴 글(pairenc.pair_template) — 비면 klue 꼴로 적는다(옛 판 호환).

    onnx_data_sha 는 가중치를 model.onnx.data 로 뗀 판에서만 준다 — 없으면 두 칸 다 null.
    num_labels 는 모델 머리 칸 수다 (3 이나 4). labels · letters · source_labels 는 data_prep 표의 앞 그만큼이다.
    """
    if num_labels not in PROTOCOLS:
        raise SystemExit(f"머리 칸이 3 이나 4 가 아니다: {num_labels}")
    protocol, guide = PROTOCOLS[num_labels]
    guide = getattr(a, "label_guide", "") or guide
    data = data_section(train_info)
    nc = any(d["nc"] for d in data)
    private = any(d["private"] for d in data)
    if parent:
        nc = nc or bool(parent.get("nc"))
        private = private or bool(parent.get("private"))
    thresholds = (calibration or {}).get("thresholds") or {}
    max_len = a.max_len or (train_info or {}).get("max_len")
    if not max_len:
        raise SystemExit("max_len 을 모른다 — --max-len 이나 --train-info 를 준다")
    return {
        "format": FORMAT,
        "name": a.name, "version": a.version or a.name,
        "body": resolve_body(a.body_name, body_sha, parent),
        "weights_sha256": weights_sha, "onnx_sha256": onnx_sha, "tokenizer_sha256": tok_sha,
        "onnx_file": "model.onnx", "opset": a.opset, "onnx_smoke_logit_diff": smoke_diff,
        "onnx_data_file": ONNX_DATA if onnx_data_sha else None, "onnx_data_sha256": onnx_data_sha,
        "input": {"pair": pair or "[CLS] evidence [SEP] claim [SEP]", "names": ["input_ids", "attention_mask"],
                  "max_len": max_len, "truncation": "only_first", "keep": "evidence head"},
        "protocol": protocol, "label_guide": guide,
        # 1.1.0 메타 — 서버가 opt-in 응답의 model_version · label_set 으로 싣는다. 비면 서버가 가중치 SHA 앞 8자를 쓴다
        "model_version": getattr(a, "model_version", "") or None, "label_set": LABEL_SET_NAMES[num_labels],
        "labels": LABELS[:num_labels], "letters": LETTERS[:num_labels], "source_labels": SOURCE_LABELS[:num_labels],
        "calibration": {"temperature": (calibration or {}).get("temperature", 1.0),
                        "file": "calibration.json" if calibration else None},
        "data": data, "nc": nc, "private": private, "publishable": not nc and not private,
        "parent": {"name": parent["name"], "weights_sha256": parent["weights_sha256"]} if parent else None,
        "scores": {"thresholds": {"nli_sure": thresholds.get("nli_sure"),
                                  "nli_sure_support": thresholds.get("nli_sure_support"),
                                  **({"nli_sure_related": thresholds.get("nli_sure_related")} if num_labels > 3 else {})},
                   **(read_json(a.scores) or {})},
        "trained_at": time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(a.src_weights))),
        "run_id": (train_info or {}).get("run_id"),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NLI 모델 → 서빙 폴더")
    p.add_argument("--src", required=True, help="HF 모델 폴더 (ckpt-best)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True, help="nli-base-v1 · nli-studio-v1")
    p.add_argument("--version", default="")
    p.add_argument("--train-info", default="")
    p.add_argument("--calibration", default="")
    p.add_argument("--body-name", default="",
                   help="몸통 HF id (model.json body 칸). 비면 --parent 의 body 를 물려받고, 둘 다 없으면 멈춘다")
    p.add_argument("--body-weights", default="", help="몸통 원본 가중치 파일 (SHA-256 을 적는다)")
    p.add_argument("--parent", default="", help="덧학습이면 시작 모델의 model.json")
    p.add_argument("--scores", default="", help="model.json scores 칸에 덧붙일 JSON")
    p.add_argument("--max-len", type=int, default=0, help="비면 train_info 의 max_len")
    p.add_argument("--opset", type=int, default=OPSET)
    p.add_argument("--int8", action="store_true", help="동적 양자화 model.int8.onnx 도 만든다 (p95 가 넘을 때만)")
    p.add_argument("--with-torch", action="store_true", help="torch/ 에 HF 가중치를 같이 둔다 (serve --backend torch)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--label-guide", default="", help="라벨 기준 글 판 (비면 3칸 support-v3 · 4칸 support-v4)")
    p.add_argument("--model-version", default="",
                   help="릴리스 판 번호 (예: 1.1.0). opt-in 응답 · /info 의 model_version 이 된다")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    a = parse_args(argv)
    parent = read_json(a.parent)
    resolve_body(a.body_name, None, parent)   # 몸통을 모르면 오래 걸리는 내보내기 전에 멈춘다
    torch.set_num_threads(a.threads)
    src = os.path.abspath(os.path.expanduser(a.src))
    out = os.path.abspath(os.path.expanduser(a.out))
    a.src_weights = os.path.join(src, "model.safetensors")
    os.makedirs(out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(src).float().eval()
    check_no_token_types(model.config)
    num_labels = model.config.num_labels
    if num_labels not in PROTOCOLS or num_labels > len(LABELS):
        raise SystemExit(f"머리 칸이 3 이나 4 가 아니다: {num_labels}")

    onnx_path = os.path.join(out, "model.onnx")
    data_path = export_graph(model, onnx_path, a.opset, tokenizer.pad_token_id)
    diff = smoke_compare(model, onnx_path, model.config.vocab_size)
    if a.int8:
        make_int8(onnx_path, os.path.join(out, "model.int8.onnx"))
    if a.with_torch:
        dest = os.path.join(out, "torch")
        os.makedirs(dest, exist_ok=True)
        for name in TORCH_FILES:
            if os.path.exists(os.path.join(src, name)):
                shutil.copy2(os.path.join(src, name), os.path.join(dest, name))

    calibration = read_json(a.calibration)
    if calibration:
        shutil.copy2(a.calibration, os.path.join(out, "calibration.json"))
    body_sha = sha256_of(a.body_weights) if a.body_weights else None
    spec = build_spec(a, sha256_of(a.src_weights), sha256_of(onnx_path),
                      sha256_of(os.path.join(out, "tokenizer", "tokenizer.json")),
                      calibration, read_json(a.train_info), parent, body_sha, diff,
                      pair=pair_template(tokenizer.backend_tokenizer),
                      onnx_data_sha=sha256_of(data_path) if data_path else None, num_labels=num_labels)
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"model.json 씀 · {a.name} · {spec['protocol']} {''.join(spec['letters'])} · "
          f"가중치 {spec['weights_sha256'][:8]} · T {spec['calibration']['temperature']} · "
          f"publishable {spec['publishable']}", flush=True)


if __name__ == "__main__":
    main()
