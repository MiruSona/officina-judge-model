"""판정 넷 확장 — 넷 칸(d 비슷한 계열) · CD 합친 칸 · grow-head · 응답 nli-v2 · compat3 · 문턱 셋 · 새 자 · fold40.

torch 가 없으면 grow-head · 손실 텐서 · smoke 넷 칸 판만 건너뛴다. 나머지는 torch 없이 돈다.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import helpers

import calibrate as cal
import data_prep as dp
import heads
import prereg
import run_fold as rf
import run_measure as rm
import serve
import split_pairs as sp

HAS_TORCH = helpers.has("torch", "transformers", "tokenizers")


def protocol_gate():
    """정본 규약(measure/protocol-g1.json)의 release_gate — 시험도 숫자를 따로 박지 않는다."""
    with open(prereg.DEFAULT_PROTOCOL, encoding="utf-8") as f:
        return json.load(f)["release_gate"]


CRIT4 = prereg.criteria_from_gate(protocol_gate())


def logits_for(probs):
    return [math.log(p) for p in probs]


class TargetTest(unittest.TestCase):
    def test_want_names_and_letters(self):
        self.assertEqual(dp.want_label("related"), 3)
        self.assertEqual(dp.want_label("D"), 3)
        self.assertEqual(dp.want_label("cd"), dp.CD)
        self.assertTrue(dp.is_label(3) and dp.is_label(dp.CD))
        self.assertFalse(dp.is_label(4) or dp.is_label(None) or dp.is_label(-1))

    def test_three_label_model_folds_d_and_cd_into_c(self):
        for label in (3, dp.CD):
            self.assertEqual(dp.train_target({"label": label, "src": "mem"}, 3), 2)
        self.assertEqual(dp.train_target({"label": 2, "src": "klue"}, 3), 2)

    def test_four_label_model_public_neutral_is_cd(self):
        self.assertEqual(dp.train_target({"label": 2, "src": "klue"}, 4), dp.CD)
        self.assertEqual(dp.train_target({"label": 2, "src": "mnli"}, 4), dp.CD)
        # 우리 무관(mem)은 c 그대로 · 공개 자료의 지지·반대도 그대로
        self.assertEqual(dp.train_target({"label": 2, "src": "mem"}, 4), 2)
        self.assertEqual(dp.train_target({"label": 1, "src": "klue"}, 4), 1)
        self.assertEqual(dp.train_target({"label": 3, "src": "mem"}, 4), 3)

    def test_public_neutral_c_mode_keeps_c(self):
        """--replay-neutral c (넷칸 보강 R1) — 공개 중립은 c 로 고정, 다른 라벨은 그대로."""
        self.assertEqual(dp.train_target({"label": 2, "src": "klue"}, 4, "c"), 2)
        self.assertEqual(dp.train_target({"label": 2, "src": "mnli"}, 4, "c"), 2)
        self.assertEqual(dp.train_target({"label": 1, "src": "klue"}, 4, "c"), 1)
        self.assertEqual(dp.train_target({"label": 3, "src": "mem"}, 4, "c"), 3)
        self.assertEqual(dp.train_target({"label": 2, "src": "klue"}, 4, "cd"), dp.CD)

    def test_eval_targets_follow_public_neutral(self):
        """eval_nli --public-neutral c — c 로 배운 모델은 공개 중립을 c 로 채점한다. 기본 cd 는 옛 동작, 라벨 없는 줄은 None."""
        rows = [{"label": 2, "src": "klue"}, {"label": 2, "src": "mem"}, {"label": None, "src": "klue"}]
        self.assertEqual(heads.eval_targets(rows, 4), [dp.CD, 2, None])
        self.assertEqual(heads.eval_targets(rows, 4, "c"), [2, 2, None])
        self.assertEqual(heads.eval_targets(rows, 3, "c"), [2, 2, None])

    def test_public_neutral_c_mode_leaves_want_cd_and_three_labels(self):
        """want `cd` 줄은 c 모드에서도 합친 칸이고, 3칸 모델은 모드와 상관없이 c 로 접는다. 모르는 모드는 막는다."""
        self.assertEqual(dp.train_target({"label": dp.CD, "src": "mem"}, 4, "c"), dp.CD)
        self.assertEqual(dp.train_target({"label": dp.CD, "src": "mem"}, 3, "c"), 2)
        self.assertEqual(dp.train_target({"label": 2, "src": "klue"}, 3, "c"), 2)
        with self.assertRaises(ValueError):
            dp.train_target({"label": 2, "src": "klue"}, 4, "d")

    def test_mem_cd_pair_normalizes(self):
        row = dp.normalize({"id": "t", "evidence": "근거", "claim": "주장", "want": "cd"})
        self.assertEqual(row["label"], dp.CD)

    def test_public_sampling_still_three_classes(self):
        self.assertEqual(dp.SOURCE_CLASSES, [0, 1, 2])


class HeadsTest(unittest.TestCase):
    def test_weight_scale_raises_b_share_mean_one(self):
        rows = [{"src": "mem", "label": 0}] * 6 + [{"src": "mem", "label": 1}] * 3 +                [{"src": "mem", "label": 2}] * 4 + [{"src": "mem", "label": 3}] * 2
        w = heads.class_weights(rows, 4)
        scale = heads.parse_weight_scale("a=1.0,b=1.2")
        self.assertEqual(scale, {0: 1.0, 1: 1.2})
        s = heads.scale_weights(w, scale)
        self.assertAlmostEqual(sum(s) / 4, 1.0)                  # 다시 평균 1
        self.assertAlmostEqual(s[1] / s[0], 1.2 * w[1] / w[0])   # b 만 1.2 배 (비율)
        self.assertGreater(s[1] / sum(s), w[1] / sum(w))         # b 몫이 커진다
        self.assertEqual(heads.scale_weights(w, {}), w)
        for bad in ("e=1.2", "b=0", "b=x"):
            with self.assertRaises(ValueError):
                heads.parse_weight_scale(bad)

    def test_class_weights_four_mean_one(self):
        rows = [{"src": "mem", "label": 0}] * 6 + [{"src": "mem", "label": 1}] * 3 + \
               [{"src": "mem", "label": 2}] * 4 + [{"src": "mem", "label": 3}] * 2 + \
               [{"src": "mem", "label": dp.CD}] * 9 + [{"src": "klue", "label": 1}] * 50
        w = heads.class_weights(rows, 4)
        self.assertEqual(len(w), 4)
        self.assertAlmostEqual(sum(w) / 4, 1.0)
        self.assertEqual(max(w), w[3])          # d 가 가장 적어 가장 무겁다
        self.assertEqual(heads.row_weight(w, {"src": "mem", "label": dp.CD}), 1.0)
        self.assertEqual(heads.row_weight(w, {"src": "klue", "label": 1}), 1.0)
        self.assertEqual(heads.row_weight(w, {"src": "mem", "label": 3}), w[3])

    def test_merged_loss_is_minus_log_c_plus_d(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        got = heads.nll_ref(logits_for(probs), dp.CD)
        self.assertAlmostEqual(got, -math.log(0.7), places=9)
        self.assertAlmostEqual(heads.nll_ref(logits_for(probs), 3), -math.log(0.4), places=9)
        # 3칸 로짓이면 합친 칸은 c 하나다
        self.assertAlmostEqual(heads.nll_ref(logits_for([0.2, 0.3, 0.5]), dp.CD), -math.log(0.5), places=9)

    def test_merged_loss_does_not_care_how_c_d_split(self):
        a = heads.nll_ref(logits_for([0.1, 0.2, 0.6, 0.1]), dp.CD)
        b = heads.nll_ref(logits_for([0.1, 0.2, 0.1, 0.6]), dp.CD)
        self.assertAlmostEqual(a, b, places=9)

    def test_metrics_four_columns(self):
        logits = [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 3], [0, 0, 3, 0], [3, 0, 0, 0], [0, 0, 3, 0]]
        golds = [0, 1, 3, 3, 3, dp.CD]
        m = heads.metrics_from(logits, golds, ["long", "", "", "", "", ""])
        self.assertEqual(len(m["confusion"]), 4)
        self.assertEqual(m["recall"]["d"], round(1 / 3, 4))
        self.assertEqual(m["acc"], 0.6)                     # CD 뺀 다섯 중 셋
        self.assertEqual(m["acc3"], round(5 / 6, 4))        # c·d 를 합치면 d→c 도 맞힘 · CD→c 도 맞힘
        self.assertEqual(m["false_support"], 1)             # 정답 d 를 a 로
        self.assertEqual(m["cd"], {"n": 1, "hit": 1})

    def test_metrics_three_columns_same_as_before(self):
        m = heads.metrics_from([[3, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], [0, 1, 1, 2], ["long", "", "", ""])
        self.assertEqual((m["acc"], m["acc3"], m["false_support"]), (0.75, 0.75, 1))
        self.assertNotIn("cd", m)


class CalibrateFourTest(unittest.TestCase):
    def test_related_threshold_smallest_meeting_precision(self):
        scored = []
        scored += [([0.05, 0.05, 0.05, 0.85], 3)] * 8       # 맞은 확신 d 0.85
        scored += [([0.05, 0.05, 0.25, 0.65], 2)] * 4       # 틀린 d 0.65
        scored += [([0.9, 0.04, 0.03, 0.03], 0)] * 10
        scored += [([0.03, 0.9, 0.04, 0.03], 1)] * 10
        got = cal.pick_thresholds(scored)
        # 0.65 이하면 8/12 · 0.66 부터 8/8
        self.assertEqual(got["nli_sure_related"], 0.66)
        self.assertEqual(got["dev_sure_related_precision"], 1.0)
        self.assertEqual(got["dev_false_support_share"], 0.01)

    def test_three_columns_have_no_related(self):
        scored = [([0.9, 0.05, 0.05], 0)] * 5 + [([0.05, 0.9, 0.05], 1)] * 5
        got = cal.pick_thresholds(scored)
        self.assertNotIn("nli_sure_related", got)
        self.assertEqual(got["dev_false_support_share"], 0.02)

    def test_cd_gold_counts_c_plus_d(self):
        rows = [{"label": dp.CD, "logits": logits_for([0.1, 0.1, 0.4, 0.4])}]
        self.assertAlmostEqual(cal.nll(rows, 1.0), -math.log(0.8), places=6)
        self.assertTrue(cal.is_hit(3, dp.CD) and cal.is_hit(2, dp.CD))
        self.assertFalse(cal.is_hit(0, dp.CD))

    def test_related_answer_uses_own_threshold(self):
        probs = [0.05, 0.05, 0.1, 0.8]
        self.assertTrue(cal.is_sure(probs, 0.9, 0.9, 0.75))
        self.assertFalse(cal.is_sure(probs, 0.9, 0.9, 0.85))
        self.assertFalse(cal.is_sure(probs, 0.9, 0.9))      # 문턱이 없으면 nli_sure


class SplitFourTest(unittest.TestCase):
    def rows(self):
        wants = ["support", "contradict", "unrelated", "d"]
        rows = []
        for g in range(80):
            for k in range(2):
                want = wants[(g + k) % 4]
                if g % 10 == 0:
                    want = "cd"
                rows.append({"id": f"g{g}-{k}", "evidence": f"근거 {g}-{k}", "claim": f"주장 {g}-{k}", "want": want,
                             "src_ids": [f"m{g}-{k}", f"m{g}-x"]})
        return rows

    def test_d_counted_and_cd_stays_in_train(self):
        train, dev, _, _ = sp.split(self.rows(), seed=1, dev_frac=0.15, band=(0.08, 0.25), min_contradict=3)
        self.assertIn("related", sp.stats(dev)["want"])
        self.assertFalse([r for r in dev if r["want"] == "cd"])
        self.assertEqual(sum(1 for r in train if r["want"] == "cd"), 16)


class FakeJudge:
    def __init__(self, probs):
        self.value = probs

    def probs(self, evidence, claim):
        return self.value


class ServeFourTest(unittest.TestCase):
    def start(self, judge, letters=None, protocol=None):
        server = serve.make_server(judge, "abcd1234", "onnx", "nli-test", 0, letters, protocol)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def ask(self, url):
        data = json.dumps({"evidence": "근거", "claim": "주장"}).encode("utf-8")
        req = urllib.request.Request(url + "/judge", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as res:
            return json.loads(res.read().decode("utf-8"))

    def health(self, url):
        with urllib.request.urlopen(url + "/health", timeout=5) as res:
            return json.loads(res.read().decode("utf-8"))

    def spec_dir(self, extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "model.json"), "w", encoding="utf-8") as f:
            json.dump(dict({"name": "m", "weights_sha256": "0" * 64, "calibration": {"temperature": 1.0}}, **extra), f)
        return tmp.name

    def test_four_letters_reply(self):
        spec = serve.load_spec(self.spec_dir({"letters": ["a", "b", "c", "d"], "protocol": "nli-v2"}))
        letters, protocol = serve.serving_shape(spec, compat3=False)
        url = self.start(FakeJudge([0.1, 0.2, 0.3, 0.4]), letters, protocol)
        body = self.ask(url)
        self.assertEqual(set(body), {"a", "b", "c", "d", "model", "ms"})
        self.assertAlmostEqual(body["a"] + body["b"] + body["c"] + body["d"], 1.0)
        h = self.health(url)
        self.assertEqual((h["letters"], h["protocol"]), (["a", "b", "c", "d"], "nli-v2"))

    def test_compat3_folds_d_into_c(self):
        spec = serve.load_spec(self.spec_dir({"letters": ["a", "b", "c", "d"]}))
        letters, protocol = serve.serving_shape(spec, compat3=True)
        self.assertEqual((letters, protocol), (["a", "b", "c"], "nli-v1"))
        url = self.start(serve.Compat3(FakeJudge([0.1, 0.2, 0.3, 0.4])), letters, protocol)
        body = self.ask(url)
        self.assertEqual(set(body), {"a", "b", "c", "model", "ms"})
        self.assertAlmostEqual(body["c"], 0.7)
        # 옛 손님의 합 검사(a+b+c = 1 ± 0.02)를 통과한다
        self.assertAlmostEqual(body["a"] + body["b"] + body["c"], 1.0)
        self.assertEqual(self.health(url)["protocol"], "nli-v1")

    def test_old_three_label_model_json_serves_old_shape(self):
        spec = serve.load_spec(self.spec_dir({}))
        self.assertEqual((spec["_letters"], spec["_protocol"]), (["a", "b", "c"], "nli-v1"))
        letters, protocol = serve.serving_shape(spec, compat3=True)    # 3칸이면 compat3 는 할 일이 없다
        url = self.start(FakeJudge([0.7, 0.2, 0.1]), letters, protocol)
        self.assertEqual(set(self.ask(url)), {"a", "b", "c", "model", "ms"})

    def test_four_letter_server_refuses_three_probs(self):
        url = self.start(FakeJudge([0.7, 0.2, 0.1]), ["a", "b", "c", "d"], "nli-v2")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.ask(url)
        self.assertEqual(ctx.exception.code, 500)
        ctx.exception.close()

    def test_bad_letters_refused(self):
        with self.assertRaises(SystemExit):
            serve.load_spec(self.spec_dir({"letters": ["a", "c", "b"]}))


class ExportSpecFourTest(unittest.TestCase):
    def setUp(self):
        import export_onnx

        self.e = export_onnx
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def spec(self, num_labels, thresholds):
        weights = os.path.join(self.tmp, "model.safetensors")
        with open(weights, "wb") as f:
            f.write(b"x")
        a = self.e.parse_args(["--src", self.tmp, "--out", self.tmp, "--name", "nli-test", "--max-len", "320",
                               "--body-name", "FacebookAI/xlm-roberta-large"])
        a.src_weights = weights
        cal_ = {"temperature": 2.0, "thresholds": thresholds}
        train_info = {"data": [{"file": "mem.jsonl", "sha256": "s", "rows": 1, "license": "l", "nc": False,
                                "by_src": {"mem": 1}}]}
        return self.e.build_spec(a, "w" * 64, "o" * 64, "t" * 64, cal_, train_info, None, None, None,
                                 num_labels=num_labels)

    def test_four_labels_spec(self):
        spec = self.spec(4, {"nli_sure": 0.8, "nli_sure_support": 0.85, "nli_sure_related": 0.7})
        self.assertEqual(spec["labels"], ["support", "contradict", "unrelated", "related"])
        self.assertEqual(spec["letters"], ["a", "b", "c", "d"])
        self.assertEqual(spec["source_labels"], ["entailment", "contradiction", "neutral", None])
        self.assertEqual((spec["protocol"], spec["label_guide"]), ("nli-v2", "support-v4"))
        self.assertEqual(spec["scores"]["thresholds"]["nli_sure_related"], 0.7)

    def test_three_labels_spec_keeps_old_shape(self):
        spec = self.spec(3, {"nli_sure": 0.8, "nli_sure_support": 0.85})
        self.assertEqual(spec["letters"], ["a", "b", "c"])
        self.assertEqual((spec["protocol"], spec["label_guide"]), ("nli-v1", "support-v3"))
        self.assertNotIn("nli_sure_related", spec["scores"]["thresholds"])

    def test_other_label_count_stops(self):
        with self.assertRaises(SystemExit):
            self.spec(5, {})


class MeasureFourTest(unittest.TestCase):
    golds = [0, 1, 2, 3, 3, 3]
    probs = [[0.9, 0.04, 0.03, 0.03], [0.05, 0.9, 0.03, 0.02], [0.05, 0.05, 0.1, 0.8],
             [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.6, 0.3], [0.86, 0.05, 0.05, 0.04]]

    def test_model_table_four(self):
        t = rm.model_table(self.golds, self.probs)
        self.assertEqual(t["labels"], 4)
        self.assertEqual(t["accuracy"], 0.5)               # a · b · d 하나
        self.assertEqual(t["accuracy3"], round(5 / 6, 4))  # c↔d 섞임은 맞힘, d→a 만 틀림
        self.assertEqual(t["recall"]["d"], round(1 / 3, 4))
        self.assertEqual(t["false_support_no_threshold"], 1)

    def test_sure_table(self):
        t = rm.sure_table(self.golds, self.probs, sure=0.7, sure_support=0.85, sure_related=0.75)
        self.assertEqual(t["false_support"], 1)             # 정답 d 를 0.86 지지로
        self.assertEqual(t["non_support"], 5)
        self.assertEqual(t["unsure"], 1)                    # c 0.6 < 0.7
        self.assertEqual((t["related_sure"], t["related_sure_hit"]), (2, 1))
        self.assertEqual(t["related_sure_precision"], 0.5)

    def verdict(self, **over):
        model = {"accuracy3": 0.9, "recall": {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.75}}
        sure = {"false_support_share": 0.01, "unsure_rate": 0.2, "related_sure_precision": 0.85}
        model["recall"].update(over.pop("recall", {}))
        sure.update(over)
        return rm.verdict(CRIT4, model, None, 500.0, sure)

    def test_v2_pass(self):
        v = self.verdict()
        self.assertTrue(v["passed"])
        self.assertIn("run_fold", v["also_needs"])

    def test_v2_related_only_fail_goes_compat3(self):
        self.assertEqual(self.verdict(recall={"d": 0.5})["next"], "fail_related")
        self.assertEqual(self.verdict(related_sure_precision=0.6)["next"], "fail_related")

    def test_v2_old_job_fail(self):
        self.assertEqual(self.verdict(false_support_share=0.02)["next"], "fail_old")
        self.assertEqual(self.verdict(recall={"b": 0.7})["next"], "fail_old")

    def test_v2_missing_sure_table_is_not_pass(self):
        model = {"accuracy3": 0.9, "recall": {"b": 0.8, "d": 0.75}}
        self.assertFalse(rm.verdict(CRIT4, model, None, 500.0, None)["passed"])

    def test_letters_must_match_prereg(self):
        with self.assertRaises(SystemExit):
            rm.server_letters({"letters": ["a", "b", "c"]}, {"letters": ["a", "b", "c", "d"]})
        self.assertEqual(rm.server_letters({}, {}), ["a", "b", "c"])

    def test_label_set_follows_prereg_letters(self):
        # prereg 칸 수로 청할 칸을 고른다 — letters 없는 옛 prereg 는 3칸
        self.assertEqual(rm.prereg_label_set({}), (["a", "b", "c"], "nli3"))
        self.assertEqual(rm.prereg_label_set({"letters": ["a", "b", "c", "d"]}), (["a", "b", "c", "d"], "nli4"))

    def test_ask_judge_sends_label_set_and_checks_reply(self):
        sent = []

        def fake(url, body=None):
            sent.append((url, body))
            return {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "label_set": "nli4"}
        with mock.patch.object(rm, "http_json", fake):
            body = rm.ask_judge("http://x/", "e", "c", "nli4")
        self.assertEqual(sent, [("http://x/judge", {"evidence": "e", "claim": "c", "label_set": "nli4"})])
        self.assertEqual(body["d"], 0.4)
        # --compat3 서버는 nli4 요청에 nli3 으로 답한다 — 청한 칸과 다르면 멈춘다
        with mock.patch.object(rm, "http_json", lambda url, body=None: {"a": 0.5, "b": 0.3, "c": 0.2,
                                                                         "label_set": "nli3"}):
            with self.assertRaises(SystemExit):
                rm.ask_judge("http://x", "e", "c", "nli4")

    def test_cd_not_allowed_in_measure_set(self):
        with self.assertRaises(SystemExit):
            rm.gold_labels([{"id": "x", "want": "cd"}])
        self.assertEqual(rm.gold_labels([{"want": "related"}, {"want": "D"}]), [3, 3])


class PreregFourTest(unittest.TestCase):
    spec = {"name": "nli-studio-v3", "weights_sha256": "0" * 64, "calibration": {"temperature": 1.5},
            "letters": ["a", "b", "c", "d"], "protocol": "nli-v2",
            "scores": {"thresholds": {"nli_sure": 0.8, "nli_sure_support": 0.85}}}

    def test_four_needs_related_threshold(self):
        with self.assertRaises(SystemExit):
            prereg.build(self.spec, {}, set(), set())

    def four_spec(self):
        spec = json.loads(json.dumps(self.spec))
        spec["scores"]["thresholds"]["nli_sure_related"] = 0.7
        return spec

    def test_four_needs_protocol(self):
        with self.assertRaises(SystemExit):
            prereg.build(self.four_spec(), {}, set(), set())

    def test_four_reads_criteria_from_protocol(self):
        gate = protocol_gate()
        got = prereg.build(self.four_spec(), {}, set(), set(), {"release_gate": gate}, "/x/protocol-g1.json", "f" * 64)
        self.assertEqual((got["nli_sure_related"], got["letters"]), (0.7, ["a", "b", "c", "d"]))
        c = got["criteria"]
        self.assertEqual(c["accuracy3_min"], gate["acc3_min"])
        self.assertEqual(c["contradict_recall_min"], gate["b_recall_min"])
        self.assertEqual(c["false_support_share_max"], gate["sure_false_support_share_max"])
        self.assertEqual(c["unsure_max"], gate["unsure_max"])
        self.assertEqual(c["related_recall_min"], gate["d_recall_min"])
        self.assertEqual(c["related_sure_precision_min"], gate["d_sure_precision_min"])
        self.assertEqual(c["p95_ms_max"], gate["p95_ms_max"])
        self.assertNotIn("fold_wrong_max", c)  # 규약에 없으면 옮기지 않는다
        self.assertEqual(got["release_gate"], gate)
        self.assertEqual((got["protocol_file"], got["protocol_sha256"]), ("protocol-g1.json", "f" * 64))

    def test_gate_values_follow_protocol_not_code(self):
        # 규약 값을 바꾸면 prereg 도 따라간다 — 코드에 숫자가 숨어 있지 않다
        gate = dict(protocol_gate(), d_recall_min=0.123, d_sure_precision_min=0.456, fold_wrong_max=2)
        c = prereg.criteria_from_gate(gate)
        self.assertEqual((c["related_recall_min"], c["related_sure_precision_min"], c["fold_wrong_max"]),
                         (0.123, 0.456, 2))
        del gate["d_recall_min"]
        with self.assertRaises(SystemExit):
            prereg.criteria_from_gate(gate)

    def test_main_writes_default_protocol_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.json"), "w", encoding="utf-8") as f:
                json.dump(self.four_spec(), f)
            out = os.path.join(tmp, "prereg.json")
            with mock.patch("sys.stdout"):
                prereg.main(["--model-dir", tmp, "--set", f"m={os.path.join(tmp, 'model.json')}", "--out", out])
            with open(out, encoding="utf-8") as f:
                got = json.load(f)
        gate = protocol_gate()
        self.assertEqual(got["criteria"]["related_recall_min"], gate["d_recall_min"])
        self.assertEqual(got["criteria"]["related_sure_precision_min"], gate["d_sure_precision_min"])
        self.assertEqual(got["protocol_sha256"], prereg.sha256_file(prereg.DEFAULT_PROTOCOL))


class FoldTest(unittest.TestCase):
    """양방향 「담김」 — 가짜 판정기 : 근거 안에 주장 글이 그대로 있으면 확신 a, 「아니」가 있으면 확신 b."""

    SURE, SURE_SUPPORT = 0.8, 0.85

    def ask_factory(self):
        calls = []

        def ask(evidence, claim):
            calls.append((evidence, claim))
            if "아니" in claim:
                return [0.03, 0.9, 0.04, 0.03]
            if claim in evidence:
                return [0.9, 0.04, 0.03, 0.03]
            return [0.05, 0.05, 0.1, 0.8]
        return ask, calls

    def mem(self, summary, *sentences):
        # evidence 는 손님 FoldEvidence 자리 — 시험에선 요약 + 문장을 이어 붙인 것으로 둔다
        return {"summary": summary, "sentences": list(sentences), "evidence": " ".join((summary,) + sentences)}

    def test_inside_folds_small_side(self):
        ask, _ = self.ask_factory()
        row = {"id": "f1", "x": self.mem("문턱은 0.3", "기본은 켬"),
               "y": self.mem("문턱은 0.3", "기본은 켬", "상한은 100"), "foldable": "x", "group": "inside"}
        got = rf.judge_pair(row, ask, self.SURE, self.SURE_SUPPORT)
        self.assertEqual((got["x_in_y"], got["y_in_x"]), (True, False))
        self.assertEqual((got["fold"], got["wrong"], got["missed"]), (["x"], [], []))

    def test_same_content_both_sides(self):
        ask, _ = self.ask_factory()
        row = {"id": "f2", "x": self.mem("문턱은 0.3"), "y": self.mem("문턱은 0.3"), "foldable": "both"}
        got = rf.judge_pair(row, ask, self.SURE, self.SURE_SUPPORT)
        self.assertEqual((got["fold"], got["wrong"]), (["x", "y"], []))

    def test_human_says_loss_is_wrong(self):
        ask, _ = self.ask_factory()
        row = {"id": "f3", "x": self.mem("문턱은 0.3"), "y": self.mem("문턱은 0.3", "기본은 켬"), "foldable": "none"}
        got = rf.judge_pair(row, ask, self.SURE, self.SURE_SUPPORT)
        self.assertEqual(got["wrong"], ["x"])

    def test_summary_miss_stops_after_one_call_each_way(self):
        ask, calls = self.ask_factory()
        row = {"id": "f4", "x": self.mem("요약 갑", "문장 하나", "문장 둘"), "y": self.mem("요약 을", "문장 셋"),
               "foldable": "none"}
        got = rf.judge_pair(row, ask, self.SURE, self.SURE_SUPPORT)
        self.assertEqual(got["calls"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(got["fold"], [])

    def test_sure_b_sentence_blocks_containment(self):
        ask, _ = self.ask_factory()
        row = {"id": "f5", "x": self.mem("문턱은 0.3", "기본은 아니 켬"), "y": self.mem("문턱은 0.3 기본은 아니 켬"),
               "foldable": "none"}
        got = rf.judge_pair(row, ask, self.SURE, self.SURE_SUPPORT)
        self.assertFalse(got["x_in_y"])
        self.assertIn("확신 b", got["why"]["x_in_y"])

    def test_claims_capped_at_summary_plus_five(self):
        m = self.mem("요약", *[f"문장 {i}" for i in range(9)])
        self.assertEqual(len(rf.claims_of(m)), 6)
        self.assertEqual(rf.evidence_of({"summary": "요약", "sentences": ["가"], "evidence": "왜 : 나 가"}), "나 가")

    def test_rows_and_set_checked_before_http(self):
        """evidence 없는 줄 · 모르는 foldable · `=` 없는 --set 은 HTTP 전에 거절한다 (손님과 다른 글로 재지 않게)."""
        good = {"id": "f1", "x": self.mem("가"), "y": self.mem("나"), "foldable": "x"}
        rf.check_rows([good])
        no_evidence = dict(good, y={"summary": "나", "sentences": []})
        with self.assertRaises(SystemExit) as cm:
            rf.check_rows([good, no_evidence])
        self.assertIn("evidence", str(cm.exception))
        with self.assertRaises(SystemExit):
            rf.check_rows([dict(good, foldable="maybe")])
        self.assertEqual(rf.parse_set("fold40=a=b.jsonl"), ("fold40", "a=b.jsonl"))
        with self.assertRaises(SystemExit):
            rf.parse_set("fold40.jsonl")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fold40.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(no_evidence, ensure_ascii=False) + "\n")
            pre = os.path.join(tmp, "prereg.json")
            with open(pre, "w", encoding="utf-8") as f:
                json.dump({"criteria": {"fold_wrong_max": 1}, "sets": {}}, f)
            # 닿지 않는 주소 — 줄 검사가 먼저 막으면 HTTP 까지 안 간다
            with self.assertRaises(SystemExit) as cm:
                rf.main(["--prereg", pre, "--set", f"fold40={path}", "--url", "http://127.0.0.1:9",
                         "--out-dir", tmp])
            self.assertIn("evidence", str(cm.exception))

    def test_table_and_verdict(self):
        results = [{"id": "a", "group": "same", "fold": ["x"], "wrong": [], "missed": [], "calls": 2},
                   {"id": "b", "group": "overlap", "fold": ["y"], "wrong": ["y"], "missed": [], "calls": 5},
                   {"id": "c", "group": "inside", "fold": [], "wrong": [], "missed": ["x"], "calls": 2}]
        t = rf.fold_table(results)
        self.assertEqual((t["fold_wrong"], t["fold_missed"], t["pairs_done_in_two"]), (1, 1, 2))
        self.assertTrue(rf.fold_verdict({"fold_wrong_max": 1}, t)["passed"])
        t["fold_wrong"] = 2
        self.assertFalse(rf.fold_verdict({"fold_wrong_max": 1}, t)["passed"])


@unittest.skipUnless(HAS_TORCH, "torch · transformers · tokenizers 없음")
class TorchFourTest(unittest.TestCase):
    def setUp(self):
        import train_nli

        self.t = train_nli

    def small_model(self, num_labels=3):
        from transformers import RobertaConfig, RobertaForSequenceClassification

        config = RobertaConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32, max_position_embeddings=40, type_vocab_size=1,
                               pad_token_id=0, num_labels=num_labels)
        return RobertaForSequenceClassification(config)

    def test_grow_head_copies_abc_and_starts_d_near_c(self):
        import torch

        model = self.small_model(3)
        old_w = model.classifier.out_proj.weight.detach().clone()
        old_b = model.classifier.out_proj.bias.detach().clone()
        self.t.grow_head(model, 4, seed=1)
        new = model.classifier.out_proj
        self.assertEqual(tuple(new.weight.shape), (4, 16))
        self.assertTrue(torch.equal(new.weight[:3], old_w))
        self.assertTrue(torch.equal(new.bias[:3], old_b))
        self.assertFalse(torch.equal(new.weight[3], old_w[2]))
        self.assertLess(float((new.weight[3] - old_w[2]).abs().max()), 0.1)
        self.assertEqual(model.config.num_labels, 4)
        self.assertEqual(model.config.id2label[3], "related")
        ids = torch.tensor([[1, 5, 6, 2]])
        self.assertEqual(tuple(model(input_ids=ids).logits.shape), (1, 4))

    def test_grow_head_refuses_shrink(self):
        with self.assertRaises(SystemExit):
            self.t.grow_head(self.small_model(4), 4)

    def test_tensor_loss_matches_reference(self):
        import torch

        logits = torch.tensor([[0.1, 0.4, -0.2, 0.3], [1.0, -1.0, 0.5, 0.2], [0.0, 0.0, 2.0, 1.0]])
        labels = torch.tensor([3, 0, dp.CD])
        got = self.t.nli_losses(logits, labels).tolist()
        for row, label, value in zip(logits.tolist(), labels.tolist(), got):
            self.assertAlmostEqual(value, heads.nll_ref(row, label), places=5)

    def test_smoke_grow_head_four(self):
        script = os.path.join(helpers.ROOT, "train", "train_nli.py")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", script, "--smoke", "--grow-head", "4", "--out", tmp,
                                   "--threads", "2"], capture_output=True, text=True, encoding="utf-8", timeout=600)
            tail = (proc.stdout + proc.stderr)[-2000:]
            self.assertEqual(proc.returncode, 0, tail)
            self.assertIn("SMOKE OK", proc.stdout, tail)
            with open(os.path.join(tmp, "train_info.json"), encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(len(info["labels"]), 4)
            self.assertEqual(info["hparams"]["grow_head"], 4)


if __name__ == "__main__":
    unittest.main()
