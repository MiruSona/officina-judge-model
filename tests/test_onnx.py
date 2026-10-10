"""ONNX 길 — 기준값 파일 · 일치 셈 · 내보내기 그래프 입력 · model.json 공개 칸 · 실제 모델 폴더.

패키지가 없으면 그 묶음만 건너뛴다. 실제 모델 시험은 JUDGEMODEL_TEST_MODEL 에 모델 폴더를 줄 때만 돈다.
"""
import json
import os
import tempfile
import unittest

import helpers

import check_onnx
from serve import softmax

HAS_TOKENIZERS = helpers.has("tokenizers")
HAS_EXPORT = helpers.has("torch", "transformers", "onnx")
HAS_ORT = helpers.has("onnxruntime", "tokenizers", "numpy")
REAL_MODEL = os.environ.get("JUDGEMODEL_TEST_MODEL", "")

SPEC = {"weights_sha256": "ab" * 32, "input": {"max_len": 32}}


class RefFileTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "k1", "premise": "가", "hypothesis": "나"}, {"id": "k2", "premise": "다", "hypothesis": "라"}]
        self.refs = [{"id": "k1", "hf_ids": [2, 4, 3, 5, 3], "logits": [1.0, 0.0, -1.0]},
                     {"id": "k2", "hf_ids": [2, 6, 3, 7, 3], "logits": [0.0, 2.0, 0.0]}]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "ref.jsonl")
        check_onnx.write_ref(self.path, SPEC, self.refs)

    def test_round_trip(self):
        self.assertEqual(check_onnx.read_ref(self.path, SPEC, self.rows), self.refs)

    def test_other_model_is_refused(self):
        other = {"weights_sha256": "cd" * 32, "input": {"max_len": 32}}
        with self.assertRaises(SystemExit):
            check_onnx.read_ref(self.path, other, self.rows)

    def test_other_max_len_is_refused(self):
        other = {"weights_sha256": SPEC["weights_sha256"], "input": {"max_len": 320}}
        with self.assertRaises(SystemExit):
            check_onnx.read_ref(self.path, other, self.rows)

    def test_other_pair_order_is_refused(self):
        with self.assertRaises(SystemExit):
            check_onnx.read_ref(self.path, SPEC, list(reversed(self.rows)))


class FakeOnnx:
    """serve.OnnxNLI 대역 — 토크나이저와 정해 둔 logits."""

    def __init__(self, tokenizer, logits, temperature):
        self.tokenizer = tokenizer
        self.logits = logits
        self.temperature = temperature

    def probs(self, evidence, claim):
        return softmax(self.logits[(evidence, claim)], self.temperature)


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers 없음")
class CompareTest(unittest.TestCase):
    def setUp(self):
        from pairenc import encode_pair
        from test_pairenc import char_tokenizer

        self.tok = char_tokenizer("가나다라마바사")
        self.rows = [{"id": "p1", "premise": "가나다", "hypothesis": "라"}, {"id": "p2", "premise": "마바", "hypothesis": "사"}]
        self.logits = {("가나다", "라"): [2.0, 0.5, 0.1], ("마바", "사"): [0.1, 0.2, 3.0]}
        self.refs = [{"id": r["id"], "hf_ids": encode_pair(self.tok, r["premise"], r["hypothesis"], 32)[0],
                      "logits": self.logits[(r["premise"], r["hypothesis"])]} for r in self.rows]

    def test_same_ids_and_logits_pass(self):
        fields, probs, passed = check_onnx.compare(FakeOnnx(self.tok, self.logits, 2.0), self.rows, self.refs, 32, 2.0)
        self.assertTrue(passed)
        self.assertEqual(fields["token_mismatch"], 0)
        self.assertEqual(fields["top_agree"], "2/2")
        self.assertAlmostEqual(sum(probs[0]), 1.0)

    def test_token_mismatch_fails(self):
        self.refs[1]["hf_ids"] = self.refs[1]["hf_ids"][:-1]
        fields, _, passed = check_onnx.compare(FakeOnnx(self.tok, self.logits, 2.0), self.rows, self.refs, 32, 2.0)
        self.assertFalse(passed)
        self.assertEqual(fields["token_mismatch_ids"], ["p2"])

    def test_no_pairs_is_refused(self):
        with self.assertRaises(SystemExit):
            check_onnx.compare(FakeOnnx(self.tok, self.logits, 2.0), [], [], 32, 2.0)

    def test_probability_gap_fails(self):
        shifted = dict(self.logits)
        shifted[("가나다", "라")] = [2.1, 0.5, 0.1]
        fields, _, passed = check_onnx.compare(FakeOnnx(self.tok, shifted, 2.0), self.rows, self.refs, 32, 2.0)
        self.assertFalse(passed)
        self.assertGreater(fields["prob_diff_max"], check_onnx.LIMIT_DIFF)


def tiny_roberta():
    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=80, num_labels=3, type_vocab_size=1,
                           pad_token_id=1)
    return RobertaForSequenceClassification(config).float().eval()


@unittest.skipUnless(HAS_EXPORT, "torch · transformers · onnx 없음")
class ExportGraphTest(unittest.TestCase):
    def setUp(self):
        import export_onnx

        self.e = export_onnx
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_graph_keeps_both_inputs_and_dynamic_seq(self):
        import torch

        model = tiny_roberta()
        path = os.path.join(self.tmp, "model.onnx")
        self.e.export_graph(model, path, self.e.OPSET, 1)   # 입력이 빠지면 check_inputs 가 SystemExit
        if not helpers.has("onnxruntime"):
            return
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for seq in (5, 40):
            ids = torch.randint(5, 64, (1, seq), generator=torch.Generator().manual_seed(seq))
            mask = torch.ones_like(ids)
            got = sess.run(["logits"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            with torch.no_grad():
                want = model(input_ids=ids, attention_mask=mask).logits.numpy()
            self.assertLess(float(abs(got - want).max()), 1e-4)

    def test_small_model_keeps_one_file(self):
        path = os.path.join(self.tmp, "model.onnx")
        with open(os.path.join(self.tmp, self.e.ONNX_DATA), "wb") as f:
            f.write(b"stale")   # 앞 판(외부 가중치)이 남긴 낡은 파일 — 한 파일 판이면 지워져야 한다
        self.assertIsNone(self.e.export_graph(tiny_roberta(), path, self.e.OPSET, 1))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["model.onnx"])

    def test_scattered_weights_gathered_into_one_file(self):
        # 2GB 를 넘는 몸통에서 torch 가 하는 꼴(가중치 낱개 파일)을 작은 모델로 흉내 낸다
        import onnx

        out = os.path.join(self.tmp, "out")
        scattered = os.path.join(self.tmp, "scattered")
        os.makedirs(out)
        os.makedirs(scattered)
        self.e.export_graph(tiny_roberta(), os.path.join(out, "model.onnx"), self.e.OPSET, 1)
        proto = onnx.load(os.path.join(out, "model.onnx"))
        onnx.save_model(proto, os.path.join(scattered, "model.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=False, size_threshold=0)
        self.assertGreater(len(os.listdir(scattered)), 3)
        os.remove(os.path.join(out, "model.onnx"))
        data = self.e.gather_external(scattered, os.path.join(out, "model.onnx"))
        self.assertEqual(os.path.basename(data), self.e.ONNX_DATA)
        self.assertEqual(sorted(os.listdir(out)), ["model.onnx", self.e.ONNX_DATA])
        onnx.checker.check_model(os.path.join(out, "model.onnx"))
        if helpers.has("onnxruntime"):
            import onnxruntime as ort

            ort.InferenceSession(os.path.join(out, "model.onnx"), providers=["CPUExecutionProvider"])


BODY = "FacebookAI/xlm-roberta-large"


class SpecTest(unittest.TestCase):
    """model.json 셈 — torch 없이 돈다 (export_onnx 는 torch 를 함수 안에서만 부른다)."""

    def setUp(self):
        import export_onnx

        self.e = export_onnx
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def args(self, max_len, body=BODY):
        argv = ["--src", self.tmp, "--out", self.tmp, "--name", "nli-test", "--max-len", max_len]
        if body:
            argv += ["--body-name", body]
        return self.e.parse_args(argv)

    def test_onnx_data_sha_written(self):
        weights = os.path.join(self.tmp, "model.safetensors")
        with open(weights, "wb") as f:
            f.write(b"x")
        a = self.args("320")
        a.src_weights = weights
        spec = self.e.build_spec(a, "w" * 64, "o" * 64, "t" * 64, None, self.data("klue"), None, None, None,
                                 onnx_data_sha="d" * 64)
        self.assertEqual((spec["onnx_data_file"], spec["onnx_data_sha256"]), (self.e.ONNX_DATA, "d" * 64))
        self.assertIsNone(self.spec(self.data("klue"))["onnx_data_sha256"])

    def spec(self, train_info, parent=None):
        weights = os.path.join(self.tmp, "model.safetensors")
        with open(weights, "wb") as f:
            f.write(b"x")
        a = self.args("320", body="" if parent else BODY)
        a.src_weights = weights
        cal = {"temperature": 2.0, "thresholds": {"nli_sure": 0.7, "nli_sure_support": 0.8}}
        return self.e.build_spec(a, "w" * 64, "o" * 64, "t" * 64, cal, train_info, parent, None, None)

    def data(self, src, nc=False):
        return {"data": [{"file": f"{src}.jsonl", "sha256": "s", "rows": 1, "license": "l", "nc": nc,
                          "by_src": {src: 1}}]}

    def test_public_data_is_publishable(self):
        spec = self.spec(self.data("klue"))
        self.assertFalse(spec["private"])
        self.assertTrue(spec["publishable"])
        self.assertEqual(spec["input"]["max_len"], 320)
        self.assertEqual(spec["scores"]["thresholds"], {"nli_sure": 0.7, "nli_sure_support": 0.8})

    def test_pair_text_written(self):
        weights = os.path.join(self.tmp, "model.safetensors")
        with open(weights, "wb") as f:
            f.write(b"x")
        a = self.args("128")
        a.src_weights = weights
        pair = "<s> evidence </s> </s> claim </s>"
        spec = self.e.build_spec(a, "w" * 64, "o" * 64, "t" * 64, None, self.data("mnli"), None, None, None, pair=pair)
        self.assertEqual(spec["input"]["pair"], pair)
        self.assertEqual(spec["body"], {"name": BODY, "sha256": None})

    def test_mem_data_is_private(self):
        spec = self.spec(self.data("mem"))
        self.assertTrue(spec["private"])
        self.assertFalse(spec["publishable"])

    def test_nc_blocks_publish(self):
        self.assertFalse(self.spec(self.data("klue", nc=True))["publishable"])

    def test_private_parent_is_inherited(self):
        parent = {"name": "p", "weights_sha256": "p" * 64, "nc": False, "private": True, "body": {"name": BODY}}
        spec = self.spec(self.data("klue"), parent)
        self.assertFalse(spec["publishable"])
        self.assertEqual(spec["parent"], {"name": "p", "weights_sha256": "p" * 64})


    def test_body_inherited_from_parent(self):
        parent = {"name": "nli-base-v2", "weights_sha256": "p" * 64,
                  "body": {"name": BODY, "sha256": "b" * 64, "license": "MIT"}}
        spec = self.spec(self.data("mem"), parent)
        self.assertEqual(spec["body"], {"name": BODY, "sha256": "b" * 64, "license": "MIT"})

    def test_body_name_fills_from_same_parent_body(self):
        parent = {"body": {"name": BODY, "sha256": "b" * 64, "license": "MIT"}}
        self.assertEqual(self.e.resolve_body(BODY, None, parent), {"name": BODY, "sha256": "b" * 64, "license": "MIT"})
        other = self.e.resolve_body("klue/roberta-large", "k" * 64, parent)
        self.assertEqual(other, {"name": "klue/roberta-large", "sha256": "k" * 64})

    def test_body_weights_override_inherited_sha(self):
        parent = {"body": {"name": BODY, "sha256": "b" * 64}}
        self.assertEqual(self.e.resolve_body("", "n" * 64, parent)["sha256"], "n" * 64)

    def test_no_body_and_no_parent_stops(self):
        a = self.args("320", body="")
        self.assertEqual(a.body_name, "")
        a.src_weights = os.path.join(self.tmp, "model.safetensors")
        with open(a.src_weights, "wb") as f:
            f.write(b"x")
        with self.assertRaises(SystemExit):
            self.e.build_spec(a, "w" * 64, "o" * 64, "t" * 64, None, self.data("klue"), None, None, None)
        with self.assertRaises(SystemExit):
            self.e.resolve_body("", None, {"name": "p", "weights_sha256": "p" * 64})


@unittest.skipUnless(HAS_ORT and REAL_MODEL, "onnxruntime · tokenizers 또는 JUDGEMODEL_TEST_MODEL 없음")
class RealModelTest(unittest.TestCase):
    def test_model_folder_serves(self):
        from serve import OnnxNLI, load_spec

        model_dir = os.path.expanduser(REAL_MODEL)
        spec = load_spec(model_dir)
        judge = OnnxNLI(model_dir, spec, 2)
        names = [i.name for i in judge.session.get_inputs()]
        self.assertEqual(names, ["input_ids", "attention_mask"])
        for evidence in ("짧은 근거", "긴 근거 문장이다. " * 200):   # 짧은 쌍 · max_len 에서 잘리는 쌍
            p = judge.probs(evidence, "주장")
            self.assertEqual(len(p), 3)
            self.assertAlmostEqual(sum(p), 1.0, places=5)
        with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["opset"], 17)


if __name__ == "__main__":
    unittest.main()
