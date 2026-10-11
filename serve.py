"""JudgeModel NLI 판정 서버 — POST /judge {"evidence","claim"} → {"a","b","c"(,"d"),"model","ms"} · GET /health · GET /info.

127.0.0.1 에만 붙는다. a 지지 · b 반대 · c 무관 · d 비슷한 계열 확률(온도 적용, 합 1)이고 model 은 학습 가중치
SHA-256 앞 8자다. 이 응답 꼴이 다른 툴(예: AIMemoryTool 의 NLI 손님)과의 약속이다. 기본은 ONNX Runtime CPU 판이다.

응답 칸은 model.json 의 `letters` 가 정한다 — 3칸 모델(옛 `nli-v1` 꼴)도 4칸 모델(`nli-v2` 꼴)도 같은 서버로 띄운다.
`--compat3` 은 4칸 모델을 띄우되 d 를 c 에 더해 옛 3칸 꼴(`nli-v1`)로 답한다 — 판올림 전 손님용 물러설 길이다.

1.1.0 opt-in (d 자료층 재구축 설계 6-1) : 요청에 `"label_set": "nli3" | "nli4"` 를 실으면 응답에 `labels` · `model_version` ·
`label_set` 세 칸이 더 붙는다. 안 실으면 응답은 옛 꼴 그대로다(바이트까지). 3칸으로만 답할 수 있는 서버(3칸 모델 ·
`--compat3`)는 `nli4` 요청에 3칸으로 답하고 `label_set: "nli3"` 을 싣는다 — 손님은 응답의 `label_set` 을 읽는다.
모르는 `label_set` 은 400. `GET /info` 는 판 번호 · 칸 세트 · 라벨 이름을 알린다.

    python -I serve.py --model ~/.judgemodel/models/nli-studio-v3 --port 8092 --threads 4
    python -I serve.py --model ~/.judgemodel/models/nli-studio-v3 --port 8092 --compat3     # 옛 손님용
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pairenc import encode_pair  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 8092
MAX_BODY = 1 << 20          # 1MB — 근거 한 덩이로 넉넉하다. 더 크면 413
READ_TIMEOUT = 10           # 초 — 몸통을 덜 보내고 멈춘 손님이 스레드를 붙잡지 않게
DEFAULT_DIR = os.path.join("~", ".judgemodel", "models", "nli-studio-v1")
ALL_LETTERS = ["a", "b", "c", "d"]
LETTERS_V1 = ALL_LETTERS[:3]
# 응답 꼴 판 이름 — 칸 수로 정한다. model.json 의 protocol 이 있으면 그 값을 쓴다
PROTOCOLS = {3: "nli-v1", 4: "nli-v2"}
# 응답 꼴의 판 번호 — label_set opt-in 이 붙은 판. 기본 응답을 4칸으로 바꾸는 날이 2.0.0 이다
API_VERSION = "1.1.0"
# 손님이 고르는 칸 세트 → 칸 수. 칸 수 → 세트 이름은 그 거꾸로다
LABEL_SETS = {"nli3": 3, "nli4": 4}
SET_OF = {n: name for name, n in LABEL_SETS.items()}
# model.json 에 labels 가 없는 옛 판의 라벨 이름 (train/data_prep.LABELS 와 같은 차례)
DEFAULT_LABELS = ["support", "contradict", "unrelated", "related"]


def load_spec(model_dir):
    """model.json 을 읽고 온도를 정한다 — model.json 의 calibration.temperature, 없으면 calibration.json, 없으면 1."""
    path = os.path.join(model_dir, "model.json")
    if not os.path.exists(path):
        raise SystemExit(f"model.json 없음: {path} — export_onnx.py 로 먼저 내보낸다")
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    temperature = (spec.get("calibration") or {}).get("temperature")
    cal_path = os.path.join(model_dir, "calibration.json")
    if temperature is None and os.path.exists(cal_path):
        with open(cal_path, encoding="utf-8") as f:
            temperature = json.load(f).get("temperature")
    if temperature is None:
        temperature = 1.0
    temperature = float(temperature)
    # 0 은 나눗셈이 터지고 음수는 확률 차례를 뒤집는다
    if not math.isfinite(temperature) or temperature <= 0:
        raise SystemExit(f"온도가 양수가 아니다: {temperature} ({model_dir})")
    spec["_temperature"] = temperature
    spec["_letters"] = spec_letters(spec, model_dir)
    spec["_protocol"] = spec.get("protocol") or PROTOCOLS[len(spec["_letters"])]
    check_release_meta(spec, model_dir)
    return spec


def check_release_meta(spec, model_dir=""):
    """1.1.0 메타(model_version · label_set)를 확인한다. 옛 model.json 에는 둘 다 없고 그대로 읽힌다.

    label_set 이 칸 수와 어긋나면 멈춘다 — 손님에게 「nli4」 라 해 놓고 3칸을 내면 안 된다.
    """
    version = spec.get("model_version")
    if version is not None and not (isinstance(version, str) and version):
        raise SystemExit(f"model.json model_version 이 비었거나 글이 아니다: {version!r} ({model_dir})")
    label_set = spec.get("label_set")
    want = SET_OF[len(spec["_letters"])]
    if label_set is not None and label_set != want:
        raise SystemExit(f"model.json label_set {label_set!r} 이 칸 수 {len(spec['_letters'])} 과 다르다 "
                         f"(기대 {want!r}, {model_dir})")


def spec_letters(spec, model_dir=""):
    """model.json 의 letters (없으면 옛 3칸). a·b·c(·d) 차례가 아니면 멈춘다 — 칸 차례는 손님과의 약속이다."""
    letters = spec.get("letters") or LETTERS_V1
    if letters not in (LETTERS_V1, ALL_LETTERS):
        raise SystemExit(f"model.json letters 가 a·b·c 나 a·b·c·d 가 아니다: {letters} ({model_dir})")
    return list(letters)


class Compat3:
    """4칸 판정기를 감싸 d 를 c 에 더한 3칸 확률을 낸다 (--compat3). 합은 그대로 1 이다."""

    def __init__(self, inner):
        self.inner = inner

    def probs(self, evidence, claim):
        p = self.inner.probs(evidence, claim)
        return [p[0], p[1], p[2] + p[3]]


def fit_label_set(p, want_n):
    """확률을 요청 세트 칸 수에 맞춘다. 4칸을 3칸으로 줄일 때만 d 를 c 에 더한다 — 늘릴 수는 없으니 그대로 둔다."""
    if want_n == 3 and len(p) == 4:
        return [p[0], p[1], p[2] + p[3]]
    return list(p)


def softmax(logits, temperature):
    scaled = [float(x) / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


class OnnxNLI:
    """쌍 인코딩 → ONNX 한 번 전진 → logits / T → softmax."""

    def __init__(self, model_dir, spec, threads):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.max_len = spec["input"]["max_len"]
        self.temperature = spec["_temperature"]
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        onnx_file = spec.get("onnx_file", "model.onnx")
        self.session = ort.InferenceSession(os.path.join(model_dir, onnx_file), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def logits(self, evidence, claim):
        ids, _ = encode_pair(self.tokenizer, evidence, claim, self.max_len)
        feeds = {"input_ids": self.np.array([ids], dtype=self.np.int64),
                 "attention_mask": self.np.ones((1, len(ids)), dtype=self.np.int64)}
        return [float(x) for x in self.session.run(["logits"], feeds)[0][0]]

    def probs(self, evidence, claim):
        return softmax(self.logits(evidence, claim), self.temperature)


class TorchNLI:
    """물러설 길 — model_dir/torch/ 의 HF 가중치를 torch CPU 로 돈다 (export_onnx.py --with-torch)."""

    def __init__(self, model_dir, spec, threads):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.max_len = spec["input"]["max_len"]
        self.temperature = spec["_temperature"]
        src = os.path.join(model_dir, "torch")
        self.tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True).backend_tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(src).float().eval()

    def logits(self, evidence, claim):
        ids, _ = encode_pair(self.tokenizer, evidence, claim, self.max_len)
        with self.torch.no_grad():
            tensor = self.torch.tensor([ids], dtype=self.torch.long)
            out = self.model(input_ids=tensor, attention_mask=self.torch.ones_like(tensor)).logits[0]
        return [float(x) for x in out]

    def probs(self, evidence, claim):
        return softmax(self.logits(evidence, claim), self.temperature)


def make_info(model_id, name, letters, protocol, labels, model_version, label_guide):
    """GET /info 몸통 — 판 번호 · 이 서버가 답할 수 있는 칸 세트 · 라벨 이름."""
    served = len(letters)
    return {"api_version": API_VERSION, "model_version": model_version, "model": model_id, "name": name,
            "protocol": protocol, "label_guide": label_guide, "default_label_set": SET_OF[served],
            "label_sets": {s: {"letters": ALL_LETTERS[:n], "labels": labels[:n]}
                           for s, n in LABEL_SETS.items() if n <= served},
            "id2label": {str(i): label for i, label in enumerate(labels[:served])}}


def make_handler(judge, model_id, backend, name, letters=None, protocol=None, labels=None, model_version=None,
                 label_guide=None):
    """letters 가 None 이면 옛 3칸 꼴이고 /health 에 letters · protocol 칸을 안 붙인다 (옛 시험·손님 그대로).

    labels 는 model.json 의 라벨 이름(비면 DEFAULT_LABELS), model_version 은 비면 model_id 다 — opt-in 응답과 /info 에만 쓴다.
    """
    lock = threading.Lock()  # 한 번에 한 판 — 스레드 수는 --threads 가 정한다
    reply_letters = letters or LETTERS_V1
    health = {"ok": True, "model": model_id, "backend": backend, "name": name}
    if letters is not None:
        health["letters"] = list(letters)
        health["protocol"] = protocol or PROTOCOLS[len(letters)]
    all_labels = list(labels or DEFAULT_LABELS)
    version = model_version or model_id
    info = make_info(model_id, name, reply_letters, protocol or PROTOCOLS[len(reply_letters)], all_labels, version,
                     label_guide)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = READ_TIMEOUT

        def log_message(self, fmt, *args):
            return

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if code != 200:
                # 안 읽은 몸통이 남았을 수 있다 — 연결을 이어 쓰면 그 몸통이 다음 요청으로 읽힌다
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, health)
            elif self.path == "/info":
                self.reply(200, info)
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/judge":
                self.reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                self.reply(400, {"error": "need Content-Length"})
                return
            if length > MAX_BODY:
                self.reply(413, {"error": "body size"})
                return
            try:
                req = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.reply(400, {"error": "bad json"})
                return
            if not isinstance(req, dict) or not isinstance(req.get("evidence"), str) \
                    or not isinstance(req.get("claim"), str):
                self.reply(400, {"error": "need evidence and claim strings"})
                return
            want = req.get("label_set")
            if want is not None and not (isinstance(want, str) and want in LABEL_SETS):
                self.reply(400, {"error": "unknown label_set"})
                return
            start = time.perf_counter()
            try:
                with lock:
                    p = judge.probs(req["evidence"], req["claim"])
            except Exception as err:  # 경계 — 판정 하나가 죽어도 서버는 산다. 까닭은 stderr 로
                print(f"judge 실패: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                self.reply(500, {"error": "judge failed"})
                return
            ms = (time.perf_counter() - start) * 1000
            if len(p) != len(reply_letters) or not all(math.isfinite(float(v)) for v in p):
                self.reply(500, {"error": "bad probabilities"})
                return
            if want is not None:
                p = fit_label_set(p, LABEL_SETS[want])
            body = {letter: round(float(v), 6) for letter, v in zip(ALL_LETTERS, p)}
            body["model"] = model_id
            body["ms"] = round(ms, 1)
            if want is not None:
                # opt-in 손님에게만 세 칸을 더 붙인다 — 실제로 답한 칸 수가 label_set 을 정한다
                body["labels"] = all_labels[:len(p)]
                body["model_version"] = version
                body["label_set"] = SET_OF[len(p)]
            self.reply(200, body)

    return Handler


def make_server(judge, model_id, backend, name, port, letters=None, protocol=None, **meta):
    """meta 는 make_handler 의 labels · model_version · label_guide 다."""
    return ThreadingHTTPServer((HOST, port), make_handler(judge, model_id, backend, name, letters, protocol, **meta))


def serving_shape(spec, compat3):
    """(응답 칸, 판 이름). --compat3 이면 4칸 모델이라도 옛 3칸 · 옛 판 이름이다."""
    letters, protocol = spec["_letters"], spec["_protocol"]
    if compat3 and len(letters) == 4:
        return list(LETTERS_V1), PROTOCOLS[3]
    return letters, protocol


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="JudgeModel NLI 판정 서버 (127.0.0.1)")
    ap.add_argument("--model", default=os.environ.get("JUDGEMODEL_MODEL_DIR", DEFAULT_DIR),
                    help="모델 폴더 (model.onnx · tokenizer/ · model.json · calibration.json)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--compat3", action="store_true",
                    help="4칸 모델의 d 를 c 에 더해 옛 3칸 꼴(nli-v1)로 답한다 — 판올림 전 손님용")
    a = ap.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(a.model))
    spec = load_spec(model_dir)
    model_id = spec["weights_sha256"][:8]
    t0 = time.time()
    if a.backend == "onnx":
        judge = OnnxNLI(model_dir, spec, a.threads)
    else:
        judge = TorchNLI(model_dir, spec, a.threads)
    warm = judge.probs("데우기", "데우기")
    if len(warm) != len(spec["_letters"]):
        raise SystemExit(f"모델 출력 {len(warm)} 칸이 model.json letters {spec['_letters']} 와 다르다")
    if a.compat3 and len(warm) == 3:
        print("--compat3 : 3칸 모델이라 할 일이 없다 — 그대로 띄운다", flush=True)
    letters, protocol = serving_shape(spec, a.compat3)
    if len(letters) < len(warm):
        judge = Compat3(judge)
    print(f"{spec.get('name', '?')} {a.backend} 판 · 모델 {model_id} · T {spec['_temperature']} · "
          f"응답 {protocol} {''.join(letters)} · 올림 {time.time() - t0:.1f}s · http://{HOST}:{a.port}", flush=True)
    server = make_server(judge, model_id, a.backend, spec.get("name", ""), a.port, letters, protocol,
                         labels=spec.get("labels"), model_version=spec.get("model_version"),
                         label_guide=spec.get("label_guide"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
