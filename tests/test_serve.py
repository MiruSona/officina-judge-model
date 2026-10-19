"""serve — 응답 꼴 {"a","b","c","model","ms"} · /health · 400/404/413 · 판정 실패 500 (가짜 판정기)."""
import hashlib
import http.client
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

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


class BlockingJudge:
    """첫 판정을 release 가 set 될 때까지 붙잡는다 — 자물쇠를 쥔 채 멈춘 판정 흉내."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def probs(self, evidence, claim):
        self.entered.set()
        self.release.wait(10)
        return [0.7, 0.2, 0.1]


class AuditFixTest(unittest.TestCase):
    start, post = ServeTest.start, ServeTest.post

    def test_busy_is_503_quickly(self):
        judge = BlockingJudge()
        url = self.start(judge)
        self.addCleanup(judge.release.set)
        first = threading.Thread(target=self.post, args=(url, {"evidence": "e", "claim": "c"}), daemon=True)
        first.start()
        self.assertTrue(judge.entered.wait(5))
        with mock.patch.object(serve, "BUSY_WAIT", 0.2):
            t0 = time.perf_counter()
            code, body = self.post(url, {"evidence": "e", "claim": "c"})
        self.assertEqual((code, body), (503, {"error": "busy"}))
        self.assertLess(time.perf_counter() - t0, 3)
        judge.release.set()
        first.join(5)
        # 자물쇠가 풀렸으니 다음 판정은 다시 된다
        self.assertEqual(self.post(url, {"evidence": "e", "claim": "c"})[0], 200)

    def test_deep_json_is_400(self):
        url = self.start(FakeJudge())
        self.assertEqual(self.post(url, None, raw=b"[" * 100000)[0], 400)

    def test_truncated_only_in_opt_in(self):
        judge = FakeJudge()
        judge.last_truncated = True
        url = self.start(judge)
        self.assertNotIn("truncated", self.post(url, {"evidence": "e", "claim": "c"})[1])
        code, body = self.post(url, {"evidence": "e", "claim": "c", "label_set": "nli3"})
        self.assertEqual((code, body["truncated"]), (200, True))
        judge.last_truncated = False
        self.assertIs(self.post(url, {"evidence": "e", "claim": "c", "label_set": "nli3"})[1]["truncated"], False)

    def test_compat3_passes_truncated(self):
        inner = FakeJudge([0.1, 0.2, 0.3, 0.4])
        inner.last_truncated = True
        wrapped = serve.Compat3(inner)
        wrapped.probs("e", "c")
        self.assertTrue(wrapped.last_truncated)


class OnnxHashTest(unittest.TestCase):
    def test_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.onnx"), "wb") as f:
                f.write(b"graph")
            good = hashlib.sha256(b"graph").hexdigest()
            serve.check_onnx_files({"onnx_sha256": good}, tmp)
            serve.check_onnx_files({}, tmp)   # 옛 model.json (해시 없음) 은 그대로 띄운다
            with self.assertRaises(SystemExit):
                serve.check_onnx_files({"onnx_sha256": "0" * 64}, tmp)
            with open(os.path.join(tmp, "model.onnx.data"), "wb") as f:
                f.write(b"weights")
            with self.assertRaises(SystemExit):
                serve.check_onnx_files({"onnx_sha256": good, "onnx_data_file": "model.onnx.data",
                                        "onnx_data_sha256": good}, tmp)


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
