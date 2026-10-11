"""기록 대리 서버 — mem 이 보내는 /judge 를 두 모델에 다 묻고 기록 jsonl 에 덧붙인다 (설계 3-1).
mem 에는 reply 모델의 답을 바이트 그대로 돌려준다. queries.jsonl 의 frozen_at 뒤 · EXCLUDE 밖 줄만 「학습 풀」 후보다.
"""
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from use_sites import common

HOST = "127.0.0.1"
BODY_MAX = 1 << 20
ASK_TIMEOUT_S = 2.0
HEALTH_TIMEOUT_S = 3.0
PROB_SLACK = 0.02
UNAVAILABLE = "unavailable"


def http_get_json(url, timeout=HEALTH_TIMEOUT_S):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read(BODY_MAX).decode("utf-8"))


def health(url):
    """/health → dict 또는 None (안 닿음)."""
    try:
        return http_get_json(url + "/health")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def check_answer(data):
    """서버 답 바이트 → (dict, 오류 까닭). mem readNLI 와 같은 검사 — 칸 셋 · 0~1 · 합 1±0.02."""
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, "bad-json"
    if not isinstance(parsed, dict):
        return None, "bad-json"
    probs = {}
    for letter in common.LETTERS3:
        value = parsed.get(letter)
        if not isinstance(value, (int, float)) or math.isnan(value) or not 0 <= value <= 1:
            return None, "bad-answer"
        probs[letter] = float(value)
    if abs(sum(probs.values()) - 1) > PROB_SLACK:
        return None, "bad-sum"
    probs["model"] = str(parsed.get("model", ""))[:64]
    return probs, ""


class Proxy:
    """대리 서버 하나. models = {이름: 주소}. 이름 → 기대 모델 이름(/health 의 name)은 expect 로 받는다."""

    def __init__(self, models, reply, log_path, alias, expect=None, ask_timeout=ASK_TIMEOUT_S):
        if reply not in models:
            raise common.Refuse(f"reply 모델 {reply} 가 --models 에 없다")
        self.models = dict(models)
        self.reply = reply
        self.log_path = log_path
        self.alias = alias
        self.expect = expect or {}
        self.ask_timeout = ask_timeout
        self.place = ""
        self.run = ""
        self.status = {}
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    # ---------------------------------------------------------------- 살아 있음

    def probe(self):
        """모델마다 /health 를 보고 쓸 수 있나 가른다. 이름이 기대와 다르면 못 쓴다 (다른 모델이 그 포트에 떠 있음)."""
        for name, url in self.models.items():
            h = health(url)
            if h is None:
                self.status[name] = {"ok": False, "why": "unreachable", "url": url}
                continue
            want = self.expect.get(name)
            letters = "".join(h.get("letters", []))
            if want and (h.get("name") != want["name"] or letters != want.get("letters", letters)):
                self.status[name] = {"ok": False, "why": f"다른 모델 {h.get('name')} {letters}", "url": url}
                continue
            self.status[name] = {"ok": True, "url": url, "name": h.get("name"), "model": h.get("model"),
                                 "letters": letters}
        if not self.status[self.reply]["ok"]:
            raise common.Refuse(f"reply 모델 {self.reply} 를 못 쓴다 : {self.status[self.reply]['why']}")
        return self.status

    def usable(self, name):
        return self.status.get(name, {}).get("ok", False)

    # ---------------------------------------------------------------- 묻기

    def ask_one(self, name, body):
        """한 모델에 몸통 그대로 → (기록 칸 dict, 날 바이트 또는 None)."""
        if not self.usable(name):
            return {UNAVAILABLE: True}, None
        started = time.time()
        request = urllib.request.Request(self.models[name] + "/judge", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.ask_timeout) as response:
                data = response.read(BODY_MAX)
        except urllib.error.HTTPError as e:
            return {"error": f"http-{e.code}"}, None
        except (urllib.error.URLError, OSError) as e:
            return {"error": "timeout" if "timed out" in str(e) else "unreachable"}, None
        answer, why = check_answer(data)
        ms = round((time.time() - started) * 1000, 1)
        if answer is None:
            return {"error": why, "ms": ms}, data
        answer["ms"] = ms
        return answer, data

    def ask_all(self, evidence, claim, body=None):
        """모든 모델에 묻고 한 줄 기록. (답 dict, reply 날 바이트)."""
        if body is None:
            body = json.dumps({"evidence": evidence, "claim": claim}, ensure_ascii=False).encode("utf-8")
        answers, reply_bytes = {}, None
        for name in self.models:
            answers[name], data = self.ask_one(name, body)
            if name == self.reply and "error" not in answers[name] and UNAVAILABLE not in answers[name]:
                reply_bytes = data
        row = {"at": common.now_iso(), "alias": self.alias, "place": self.place, "run": self.run,
               "reply": self.reply, "pair": common.pair_key(evidence, claim), "group": common.group_key(evidence),
               "evidence": evidence, "claim": claim, "answers": answers}
        with self._lock:
            common.append_jsonl(self.log_path, [row])
        return answers, reply_bytes

    # ---------------------------------------------------------------- 서버

    def start(self, port=0):
        self._server = ThreadingHTTPServer((HOST, port), ProxyHandler)
        self._server.proxy = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url()

    def url(self):
        return f"http://{HOST}:{self._server.server_address[1]}"

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


class ProxyHandler(BaseHTTPRequestHandler):
    """대리 서버의 요청 하나. self.server.proxy 가 그 Proxy 다."""

    def log_message(self, *args):
        return

    def send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health":
            self.send(404, b'{"error":"not-found"}')
            return
        owner = self.server.proxy
        h = health(owner.models[owner.reply])
        if h is None:
            self.send(502, b'{"error":"reply-down"}')
            return
        self.send(200, json.dumps(h, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        if self.path != "/judge":
            self.send(404, b'{"error":"not-found"}')
            return
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > BODY_MAX:
            self.send(413, b'{"error":"size"}')
            return
        body = self.rfile.read(size)
        try:
            asked = json.loads(body.decode("utf-8"))
            evidence, claim = str(asked["evidence"]), str(asked["claim"])
        except (UnicodeDecodeError, ValueError, KeyError, TypeError):
            self.send(400, b'{"error":"bad-request"}')
            return
        _, reply_bytes = self.server.proxy.ask_all(evidence, claim, body)
        if reply_bytes is None:
            # 꼴 검사에 걸린 답도 502 로 막는다 — mem 은 어느 쪽이든 실패로 세고, 502 가 원인(기록의 error 칸) 추적에 낫다
            self.send(502, b'{"error":"reply-failed"}')
            return
        self.send(200, reply_bytes)


def expected_models(protocol):
    return {name: {"name": spec["name"], "letters": spec.get("letters", "abc")}
            for name, spec in protocol["models"].items()}
