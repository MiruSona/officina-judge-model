"""serve 1.1.0 호환 규칙 — label_set opt-in (d 자료층 재구축 설계 6-1).

opt-in 없음 → 옛 꼴 그대로(바이트 비교) · opt-in → labels · model_version · label_set 세 칸 더함 ·
3칸 서버의 nli4 → 3칸 + label_set nli3 · 모르는 세트 400 · /info · model.json 메타 저장/로드.
"""
import http.client
import json
import os
import tempfile
import threading
import unittest

import helpers  # noqa: F401

import serve

FOUR = ["a", "b", "c", "d"]
LABELS4 = ["support", "contradict", "unrelated", "related"]


class FakeJudge:
    def __init__(self, probs):
        self.value = probs

    def probs(self, evidence, claim):
        return self.value


def old_reply_bytes(letters, probs, model_id, ms):
    """1.0.0 서버가 쓰던 그대로의 몸통 바이트 — 칸 차례 · 반올림 · json.dumps 인자까지 옛 코드와 같다."""
    body = {letter: round(float(v), 6) for letter, v in zip(letters, probs)}
    body["model"] = model_id
    body["ms"] = ms
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class CompatServeTest(unittest.TestCase):
    def start(self, judge, letters=None, protocol=None, **meta):
        server = serve.make_server(judge, "abcd1234", "onnx", "nli-test", 0, letters, protocol, **meta)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def call(self, port, method, path, body=None):
        """(상태, 날 몸통 바이트)."""
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        res = conn.getresponse()
        return res.status, res.read()

    def judge(self, port, extra=None):
        code, raw = self.call(port, "POST", "/judge", dict({"evidence": "근거", "claim": "주장"}, **(extra or {})))
        return code, raw, json.loads(raw.decode("utf-8"))

    def test_no_opt_in_is_old_bytes_three(self):
        probs = [0.70000012, 0.2, 0.09999988]
        port = self.start(FakeJudge(probs), labels=LABELS4[:3], model_version="1.1.0")
        code, raw, body = self.judge(port)
        self.assertEqual(code, 200)
        self.assertEqual(raw, old_reply_bytes(["a", "b", "c"], probs, "abcd1234", body["ms"]))

    def test_no_opt_in_is_old_bytes_four(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        port = self.start(FakeJudge(probs), FOUR, "nli-v2", labels=LABELS4, model_version="1.1.0")
        code, raw, body = self.judge(port)
        self.assertEqual(code, 200)
        self.assertEqual(raw, old_reply_bytes(FOUR, probs, "abcd1234", body["ms"]))

    def test_health_unchanged_by_meta(self):
        port = self.start(FakeJudge([0.7, 0.2, 0.1]), labels=LABELS4[:3], model_version="1.1.0")
        _, raw = self.call(port, "GET", "/health")
        self.assertEqual(json.loads(raw), {"ok": True, "model": "abcd1234", "backend": "onnx", "name": "nli-test"})

    def test_opt_in_nli4_on_four_label_server(self):
        port = self.start(FakeJudge([0.1, 0.2, 0.3, 0.4]), FOUR, "nli-v2", labels=LABELS4, model_version="1.1.0")
        code, _, body = self.judge(port, {"label_set": "nli4"})
        self.assertEqual(code, 200)
        self.assertEqual(list(body), ["a", "b", "c", "d", "model", "ms", "labels", "model_version", "label_set"])
        self.assertEqual((body["labels"], body["model_version"], body["label_set"]), (LABELS4, "1.1.0", "nli4"))
        self.assertAlmostEqual(body["d"], 0.4)

    def test_opt_in_nli3_on_four_label_server_folds_d(self):
        port = self.start(FakeJudge([0.1, 0.2, 0.3, 0.4]), FOUR, "nli-v2", labels=LABELS4, model_version="1.1.0")
        _, _, body = self.judge(port, {"label_set": "nli3"})
        self.assertEqual(list(body), ["a", "b", "c", "model", "ms", "labels", "model_version", "label_set"])
        self.assertAlmostEqual(body["c"], 0.7)
        self.assertAlmostEqual(body["a"] + body["b"] + body["c"], 1.0)
        self.assertEqual((body["labels"], body["label_set"]), (LABELS4[:3], "nli3"))

    def test_three_label_model_answers_nli4_as_nli3(self):
        # 3칸 모델(nli-studio-v2 꼴 — model.json 에 메타 없음)은 nli4 요청에 3칸 + label_set nli3. 에러 아님
        port = self.start(FakeJudge([0.7, 0.2, 0.1]), labels=LABELS4[:3])
        code, _, body = self.judge(port, {"label_set": "nli4"})
        self.assertEqual(code, 200)
        self.assertEqual(set(body), {"a", "b", "c", "model", "ms", "labels", "model_version", "label_set"})
        self.assertEqual(body["label_set"], "nli3")
        self.assertEqual(body["labels"], LABELS4[:3])
        self.assertEqual(body["model_version"], "abcd1234")   # model.json 에 판 번호가 없으면 가중치 SHA 앞 8자

    def test_compat3_answers_nli4_as_nli3(self):
        port = self.start(serve.Compat3(FakeJudge([0.1, 0.2, 0.3, 0.4])), ["a", "b", "c"], "nli-v1",
                          labels=LABELS4, model_version="1.1.0")
        _, _, body = self.judge(port, {"label_set": "nli4"})
        self.assertEqual((body["label_set"], body["labels"]), ("nli3", LABELS4[:3]))
        self.assertAlmostEqual(body["c"], 0.7)

    def test_unknown_label_set_is_400(self):
        port = self.start(FakeJudge([0.7, 0.2, 0.1]))
        for bad in ("nli5", "", 4, ["nli4"], {"x": 1}):
            code, raw, body = self.judge(port, {"label_set": bad})
            self.assertEqual((code, body), (400, {"error": "unknown label_set"}), bad)

    def test_null_label_set_is_no_opt_in(self):
        probs = [0.7, 0.2, 0.1]
        port = self.start(FakeJudge(probs))
        _, raw, body = self.judge(port, {"label_set": None})
        self.assertEqual(raw, old_reply_bytes(["a", "b", "c"], probs, "abcd1234", body["ms"]))

    def test_info_four(self):
        port = self.start(FakeJudge([0.1, 0.2, 0.3, 0.4]), FOUR, "nli-v2", labels=LABELS4, model_version="1.1.0",
                          label_guide="support-v5")
        code, raw = self.call(port, "GET", "/info")
        info = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertEqual(info["api_version"], "1.1.0")
        self.assertEqual((info["model_version"], info["default_label_set"]), ("1.1.0", "nli4"))
        self.assertEqual(info["label_sets"], {"nli3": {"letters": ["a", "b", "c"], "labels": LABELS4[:3]},
                                              "nli4": {"letters": FOUR, "labels": LABELS4}})
        self.assertEqual(info["id2label"], {"0": "support", "1": "contradict", "2": "unrelated", "3": "related"})
        self.assertEqual((info["label_guide"], info["protocol"]), ("support-v5", "nli-v2"))

    def test_info_three_offers_only_nli3(self):
        port = self.start(FakeJudge([0.7, 0.2, 0.1]))
        info = json.loads(self.call(port, "GET", "/info")[1])
        self.assertEqual(list(info["label_sets"]), ["nli3"])
        self.assertEqual((info["default_label_set"], info["model_version"]), ("nli3", "abcd1234"))


class ReleaseMetaTest(unittest.TestCase):
    def spec_dir(self, extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "model.json"), "w", encoding="utf-8") as f:
            json.dump(dict({"name": "m", "weights_sha256": "0" * 64, "calibration": {"temperature": 1.0}}, **extra), f)
        return tmp.name

    def test_old_model_json_without_meta_loads(self):
        spec = serve.load_spec(self.spec_dir({}))
        self.assertNotIn("model_version", spec)
        self.assertNotIn("label_set", spec)

    def test_meta_loads(self):
        spec = serve.load_spec(self.spec_dir({"letters": FOUR, "model_version": "1.1.0", "label_set": "nli4"}))
        self.assertEqual((spec["model_version"], spec["label_set"]), ("1.1.0", "nli4"))

    def test_label_set_must_match_letters(self):
        with self.assertRaises(SystemExit):
            serve.load_spec(self.spec_dir({"letters": FOUR, "label_set": "nli3"}))
        with self.assertRaises(SystemExit):
            serve.load_spec(self.spec_dir({"label_set": "nli4"}))

    def test_bad_model_version_refused(self):
        for bad in ("", 110, ["1.1.0"]):
            with self.assertRaises(SystemExit):
                serve.load_spec(self.spec_dir({"model_version": bad}))

    def test_export_writes_meta_and_serve_reads_it(self):
        import export_onnx

        tmp = self.spec_dir({})
        weights = os.path.join(tmp, "model.safetensors")
        with open(weights, "wb") as f:
            f.write(b"x")
        for num_labels, version, want in ((4, "1.1.0", "nli4"), (3, "", "nli3")):
            argv = ["--src", tmp, "--out", tmp, "--name", "nli-test", "--max-len", "320",
                    "--body-name", "FacebookAI/xlm-roberta-large"]
            if version:
                argv += ["--model-version", version]
            a = export_onnx.parse_args(argv)
            a.src_weights = weights
            spec = export_onnx.build_spec(a, "w" * 64, "o" * 64, "t" * 64, {"temperature": 2.0, "thresholds": {}},
                                          None, None, None, None, num_labels=num_labels)
            self.assertEqual((spec["model_version"], spec["label_set"]), (version or None, want))
            with open(os.path.join(tmp, "model.json"), "w", encoding="utf-8") as f:
                json.dump(spec, f)
            loaded = serve.load_spec(tmp)
            self.assertEqual((loaded.get("model_version"), loaded["label_set"], len(loaded["_letters"])),
                             (version or None, want, num_labels))


if __name__ == "__main__":
    unittest.main()
