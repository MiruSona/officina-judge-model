"""measure — 모델 표 · 사다리 표 · 자 판정 셈."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

import prereg
import run_measure as rm


class TablesTest(unittest.TestCase):
    def test_model_table_counts(self):
        golds = [0, 1, 1, 2]
        probs = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        t = rm.model_table(golds, probs)
        self.assertEqual(t["accuracy"], 0.75)
        self.assertEqual(t["recall"]["b"], 0.5)
        self.assertEqual(t["false_support_no_threshold"], 1)

    def test_ladder_counts_only_sure_false_support(self):
        golds = [1, 2, 1, 0]
        ladder = [{"letter": "A", "stage": "nli"}, {"letter": "A", "stage": "nli", "unsure": True},
                  {"letter": "B", "stage": "rules"}, {"letter": "A", "stage": "nli"}]
        probs = [[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]
        t = rm.ladder_table(golds, ladder, probs)
        self.assertEqual(t["false_support"], 1)
        self.assertEqual(t["unsure"], 1)
        self.assertEqual(t["stages"], {"nli": 3, "rules": 1})
        self.assertEqual(t["contradict_not_by_rules"], 1)
        self.assertEqual(t["contradict_not_by_rules_model_recall"], 0.0)


class LadderRowsTest(unittest.TestCase):
    def test_short_ladder_output_is_refused(self):
        # mem 이 중간에 죽어 줄이 모자라면 zip 이 조용히 잘라 거짓 지지를 덜 센다 — 재지 않는다
        with self.assertRaises(SystemExit):
            rm.check_ladder_rows([{"letter": "A"}], 2, "boom")
        rm.check_ladder_rows([{"letter": "A"}, {"letter": "B"}], 2, "")

    def test_unknown_want_is_refused(self):
        with self.assertRaises(SystemExit):
            rm.gold_labels([{"id": "x1", "want": "support"}, {"id": "x2", "want": "maybe"}])
        self.assertEqual(rm.gold_labels([{"want": "B"}, {"want": "unrelated"}]), [1, 2])


class VerdictTest(unittest.TestCase):
    model = {"accuracy": 0.9, "recall": {"a": 0.9, "b": 0.8, "c": 0.9}}
    ladder = {"false_support": 2, "unsure_rate": 0.2}

    def test_all_pass(self):
        v = rm.verdict(prereg.CRITERIA, self.model, self.ladder, 500.0)
        self.assertTrue(v["passed"])
        self.assertEqual(v["next"], "pass")

    def test_p95_only(self):
        v = rm.verdict(prereg.CRITERIA, self.model, self.ladder, 1500.0)
        self.assertEqual(v["next"], "p95_only")

    def test_missing_ladder_is_not_pass(self):
        v = rm.verdict(prereg.CRITERIA, self.model, None, 500.0)
        self.assertFalse(v["passed"])
        self.assertIsNone(v["checks"]["false_support"])

    def test_false_support_fails(self):
        v = rm.verdict(prereg.CRITERIA, self.model, {"false_support": 4, "unsure_rate": 0.1}, 500.0)
        self.assertEqual(v["next"], "fail")


class ReleaseGatePreregTest(unittest.TestCase):
    def test_vs_ref_prereg_refused_before_server(self):
        # g2 · g3 규약 prereg 는 옛 판정 길을 안 탄다 — 서버를 부르기 전에 exit 2
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prereg.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"criteria": {"vs_ref": [{"metric": "b_recall", "ci_low_gt": 0.0}]}, "sets": {}}, f)
            with mock.patch.object(rm, "measure_set") as measure, contextlib.redirect_stderr(io.StringIO()) as err:
                code = rm.main(["--prereg", path, "--set", "blind=x.jsonl", "--out-dir", tmp])
        self.assertEqual(code, 2)
        measure.assert_not_called()
        self.assertIn("gates.py release", err.getvalue())


class PreregTest(unittest.TestCase):
    def test_needs_thresholds(self):
        with self.assertRaises(SystemExit):
            prereg.build({"name": "m", "weights_sha256": "0" * 64, "calibration": {"temperature": 1.0}}, {}, set(), set())


if __name__ == "__main__":
    unittest.main()
