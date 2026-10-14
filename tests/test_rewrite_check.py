"""rewrite_check — BoolQ · KLUE 평서문 바꾸기의 기계 검사 · 조각 · 거두기. 입력은 모두 지어낸 짧은 글이다."""
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import rewrite_check as rc  # noqa: E402


class BoolqCheckTest(unittest.TestCase):
    def test_declarative_passes_and_failures_are_named(self):
        claim = "그 다리는 1990년에 완공되었는가?"
        self.assertEqual(rc.check_boolq(claim, "그 다리는 1990년에 완공되었다."), [])
        self.assertEqual(rc.check_boolq("그 섬에는 공항이 없나요?", "그 섬에는 공항이 없다."), [])
        self.assertIn("question", rc.check_boolq(claim, "그 다리는 1990년에 완공되었는가"))
        self.assertIn("negation_added", rc.check_boolq(claim, "그 다리는 1990년에 완공되지 않았다."))
        self.assertIn("negation_dropped", rc.check_boolq("그 섬에는 공항이 없나요?", "그 섬에는 공항이 있다."))
        self.assertIn("chunks", rc.check_boolq(claim, "어떤 큰 다리가 오래전 강 위에 처음으로 세워졌다."))
        self.assertEqual(rc.check_boolq(claim, None), ["empty"])


class KlueCheckTest(unittest.TestCase):
    def test_pair_differs_only_at_answer(self):
        q, ans, alt = "박람회가 열린 도시는?", "대구", "인천"
        self.assertEqual(rc.check_klue(q, ans, alt, "박람회가 열린 도시는 대구이다.", "박람회가 열린 도시는 인천이다."), {})
        why = rc.check_klue(q, ans, alt, "박람회가 열린 도시는 대구이다.", "박람회가 처음 열린 큰 도시는 인천이었다.")
        self.assertIn("chunks", why["pair"])
        why = rc.check_klue(q, ans, alt, "박람회가 열린 도시는 대구이다.", "박람회가 열린 도시는 대구와 인천이다.")
        self.assertIn("b_has_answer", why["b"])
        why = rc.check_klue(q, ans, alt, "박람회가 열린 도시는 어디인가?", "박람회가 열린 도시는 인천이다.")
        self.assertIn("value_missing", why["a"])
        self.assertIn("question", why["a"])


class PrepMergeTest(unittest.TestCase):
    def test_klue_skips_person_and_merges_by_pair_role(self):
        rows = [
            {"id": "k1", "pair": "p1", "hint": "b", "ans_type": "place", "question": "열린 도시는?", "answer": "대구",
             "alt_answer": "인천", "claim": "열린 도시는?"},
            {"id": "k2", "pair": "p1", "hint": "a", "ans_type": "place", "question": "열린 도시는?", "answer": "대구",
             "alt_answer": "인천", "claim": "열린 도시는?"},
            {"id": "k3", "pair": "p2", "hint": "b", "ans_type": "person", "question": "누가 이겼나?", "answer": "갑",
             "alt_answer": "을", "claim": "누가 이겼나?"},
        ]
        units = rc.prep_units("klue", rows)
        self.assertEqual([u["pair"] for u in units], ["p1"])
        merged, count = rc.merge_rows("klue", rows, [{"pair": "p1", "a": "열린 도시는 대구이다.", "b": "열린 도시는 인천이다."}, None])
        self.assertEqual({r["id"]: r["claim_rewrite"] for r in merged}, {"k1": "열린 도시는 인천이다.", "k2": "열린 도시는 대구이다."})
        self.assertEqual(count, {"bad": 1, "merged": 2, "skip_type": 1})
        _, c = rc.check_rows("klue", merged)
        self.assertEqual((c["pairs"], c["pass"]), (1, 1))

    def test_boolq_only_rewrite_rows_and_missing_counted(self):
        rows = [{"id": "b1", "claim": "그 강은 길까?", "needs_rewrite": True},
                {"id": "b2", "claim": "그 강은 길다.", "needs_rewrite": False},
                {"id": "b3", "claim": "그 산은 높은가?", "needs_rewrite": True}]
        self.assertEqual([u["id"] for u in rc.prep_units("boolq", rows)], ["b1", "b3"])
        merged, count = rc.merge_rows("boolq", rows, [{"id": "b1", "claim_rewrite": "그 강은 길다."}])
        self.assertEqual(count, {"merged": 1, "missing": 1})
        self.assertNotIn("claim_rewrite", merged[1])
        results, c = rc.check_rows("boolq", merged)
        self.assertEqual([r["pass"] for r in results], [True, False])
        self.assertEqual(c["empty"], 1)


if __name__ == "__main__":
    unittest.main()
