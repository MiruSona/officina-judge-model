"""JudgeModel NLI 판정 서버 — POST /judge {"evidence","claim"} → {"a","b","c","model","ms"} · GET /health.

127.0.0.1 에만 붙는다. a 지지 · b 반대 · c 무관 확률(온도 적용)이고 model 은 학습 가중치 SHA-256 앞 8자다.
이 응답 꼴이 다른 툴(예: AIMemoryTool 의 NLI 손님)과의 약속이다. 기본은 ONNX Runtime CPU 판이다.

    python -I serve.py --model ~/.judgemodel/models/nli-studio-v1 --port 8092 --threads 4
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
LETTERS = ["a", "b", "c"]


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
    return spec


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


def make_handler(judge, model_id, backend, name):
    lock = threading.Lock()  # 한 번에 한 판 — 스레드 수는 --threads 가 정한다

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
            if self.path != "/health":
                self.reply(404, {"error": "not found"})
                return
            self.reply(200, {"ok": True, "model": model_id, "backend": backend, "name": name})

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
            start = time.perf_counter()
            try:
                with lock:
                    p = judge.probs(req["evidence"], req["claim"])
            except Exception as err:  # 경계 — 판정 하나가 죽어도 서버는 산다. 까닭은 stderr 로
                print(f"judge 실패: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                self.reply(500, {"error": "judge failed"})
                return
            ms = (time.perf_counter() - start) * 1000
            if len(p) != len(LETTERS) or not all(math.isfinite(float(v)) for v in p):
                self.reply(500, {"error": "bad probabilities"})
                return
            body = {letter: round(float(v), 6) for letter, v in zip(LETTERS, p)}
            body["model"] = model_id
            body["ms"] = round(ms, 1)
            self.reply(200, body)

    return Handler


def make_server(judge, model_id, backend, name, port):
    return ThreadingHTTPServer((HOST, port), make_handler(judge, model_id, backend, name))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="JudgeModel NLI 판정 서버 (127.0.0.1)")
    ap.add_argument("--model", default=os.environ.get("JUDGEMODEL_MODEL_DIR", DEFAULT_DIR),
                    help="모델 폴더 (model.onnx · tokenizer/ · model.json · calibration.json)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    a = ap.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(a.model))
    spec = load_spec(model_dir)
    model_id = spec["weights_sha256"][:8]
    t0 = time.time()
    if a.backend == "onnx":
        judge = OnnxNLI(model_dir, spec, a.threads)
    else:
        judge = TorchNLI(model_dir, spec, a.threads)
    judge.probs("데우기", "데우기")
    print(f"{spec.get('name', '?')} {a.backend} 판 · 모델 {model_id} · T {spec['_temperature']} · "
          f"올림 {time.time() - t0:.1f}s · http://{HOST}:{a.port}", flush=True)
    server = make_server(judge, model_id, a.backend, spec.get("name", ""), a.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
