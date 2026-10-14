"""rule_filter — v6 자리 규칙 ①~④ 거르기. 입력은 모두 지어낸 짧은 글이다."""
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import rule_filter as rf  # noqa: E402


class RuleOfTest(unittest.TestCase):
    def rule(self, evidence, claim):
        return rf.rule_of(evidence, claim)[0]

    def test_qualifier_and_number(self):
        self.assertEqual(self.rule("Northtown has a population of 47,000 and a river.",
                                   "Northtown has a population of more than 50,000 ."), 1)
        self.assertEqual(self.rule("그 도시 인구는 4만 7천이다.", "그 도시 인구는 5만이 넘는다."), 1)

    def test_ordinal(self):
        self.assertEqual(self.rule("The team finished 4th in the league this year.",
                                   "The team finished 2nd in the league this year."), 2)

    def test_marker_and_word_swap(self):
        self.assertEqual(self.rule("근거 : 2026-09-18 리뷰 R4", "근거 : 2026-09-18 리뷰 R3"), 4)
        self.assertEqual(self.rule("The game was released by Alpha Studio in spring.",
                                   "The game was released by Beta Studio in spring."), 3)

    def test_unrelated_not_picked(self):
        self.assertEqual(self.rule("The bridge was built of stone in 1950.",
                                   "The museum opens every Monday for school groups."), 0)
        self.assertEqual(self.rule("같은 문장이다.", "같은 문장이다."), 0)

    def test_pick_cap_is_seeded(self):
        rows = [{"id": f"p{i:02d}", "evidence": f"Model {i} scored 4th in the test.",
                 "claim": f"Model {i} scored 2nd in the test."} for i in range(10)]
        first = rf.pick(rows, cap=4, seed=7)
        self.assertEqual(len(first), 4)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in rf.pick(rows, cap=4, seed=7)])
        self.assertTrue(all(r["rule"] == 2 for r in first))
        # 드문 자리(서수 3쌍)는 자를 때도 통째로 남는다
        mixed = rows[:3] + [{"id": f"w{i:02d}", "evidence": f"Tool {i} was made by Alpha Studio.",
                             "claim": f"Tool {i} was made by Beta Studio."} for i in range(40)]
        kept = rf.pick(mixed, cap=10, seed=7)
        self.assertEqual(len(kept), 10)
        self.assertEqual(sum(1 for r in kept if r["rule"] == 2), 3)


if __name__ == "__main__":
    unittest.main()
