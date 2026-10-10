"""serve — 응답 꼴 {"a","b","c","model","ms"} · /health · 400/404/413 · 판정 실패 500 (가짜 판정기)."""
import http.client
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import helpers  # noqa: F401

import serve


class FakeJudge:
    def __init__(self, probs=None, fail=False):
        self.probs_value = probs or [0.7, 0.2, 0.1]
        self.fail = fail
        self.calls = []

    def probs(self, evidence, claim):
        self.calls.append((evidence, claim))
        if self.fail:
            raise RuntimeError("boom")
        return self.probs_value


class ServeTest(unittest.TestCase):
    def start(self, judge):
        server = serve.make_server(judge, "abcd1234", "onnx", "nli-test", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def post(self, url, body, raw=None):
        data = raw if raw is not None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url + "/judge", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status, json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            with err:
                return err.code, json.loads(err.read().decode("utf-8"))

    def too_big(self, url):
        # 몸통은 안 보내고 길이만 크게 적는다 — 서버는 몸통을 읽기 전에 413 으로 끊는다
        host, port = url.split("//")[1].split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=5)
        conn.putrequest("POST", "/judge")
        conn.putheader("Content-Length", str(serve.MAX_BODY + 1))
        conn.endheaders()
        code = conn.getresponse().status
        conn.close()
        return code

    def test_judge_reply_shape(self):
        judge = FakeJudge()
        url = self.start(judge)
        code, body = self.post(url, {"evidence": "근거", "claim": "주장"})
        self.assertEqual(code, 200)
        self.assertEqual(set(body), {"a", "b", "c", "model", "ms"})
        self.assertEqual((body["a"], body["b"], body["c"]), (0.7, 0.2, 0.1))
        self.assertEqual(body["model"], "abcd1234")
        self.assertIsInstance(body["ms"], float)
        self.assertEqual(judge.calls, [("근거", "주장")])

    def test_health(self):
        url = self.start(FakeJudge())
        with urllib.request.urlopen(url + "/health", timeout=5) as res:
            body = json.loads(res.read().decode("utf-8"))
        self.assertEqual(body, {"ok": True, "model": "abcd1234", "backend": "onnx", "name": "nli-test"})

    def test_bad_requests(self):
        url = self.start(FakeJudge())
        self.assertEqual(self.post(url, None, raw=b"{not json")[0], 400)
        self.assertEqual(self.post(url, {"evidence": "only"})[0], 400)
        self.assertEqual(self.post(url, None, raw=b"")[0], 400)   # 몸통 없음은 크기 초과가 아니다
        self.assertEqual(self.too_big(url), 413)
        req = urllib.request.Request(url + "/nope")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 404)
        ctx.exception.close()

    def test_judge_failure_is_500(self):
        url = self.start(FakeJudge(fail=True))
        self.assertEqual(self.post(url, {"evidence": "e", "claim": "c"})[0], 500)

    def test_non_finite_is_500(self):
        url = self.start(FakeJudge(probs=[float("nan"), 0.5, 0.5]))
        self.assertEqual(self.post(url, {"evidence": "e", "claim": "c"})[0], 500)


class LoadSpecTest(unittest.TestCase):
    def spec_with(self, calibration):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "model.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "m", "weights_sha256": "0" * 64, "calibration": calibration}, f)
        return tmp.name

    def test_missing_temperature_is_one(self):
        self.assertEqual(serve.load_spec(self.spec_with({}))["_temperature"], 1.0)

    def test_zero_or_negative_temperature_is_refused(self):
        # 0 이 조용히 1 로 바뀌거나 음수로 확률 차례가 뒤집히면 안 된다
        for bad in (0, -1.5, "nan"):
            with self.assertRaises(SystemExit):
                serve.load_spec(self.spec_with({"temperature": bad}))


class SoftmaxTest(unittest.TestCase):
    def test_temperature_flattens(self):
        sharp = serve.softmax([2.0, 0.0, 0.0], 1.0)
        flat = serve.softmax([2.0, 0.0, 0.0], 2.0)
        self.assertAlmostEqual(sum(flat), 1.0)
        self.assertLess(flat[0], sharp[0])


if __name__ == "__main__":
    unittest.main()
