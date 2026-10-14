"""pick_contrast — 대조쌍 짝 고르기 · 판정 뒤 짝 규칙. 입력은 모두 지어낸 짧은 글이다."""
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import pick_contrast as pc  # noqa: E402


def pair_rows(pid, group, claim, ev_a, ev_b):
    base = {"claim": claim, "group": group, "groups": [group], "g2_source": "vitaminc-train", "pair": pid}
    return [dict(base, id=pid + "-a", hint="a", evidence=ev_a, src_ids=[pid + "_1"]),
            dict(base, id=pid + "-b", hint="b", evidence=ev_b, src_ids=[pid + "_2"])]


class EmptyForbid:
    def reasons(self, row):
        return []


class PickTest(unittest.TestCase):
    def test_qual_first_one_pair_per_group_and_forbid(self):
        rows = []
        rows += pair_rows("p1", "vitc:A", "The band sold more than 5 million copies .",
                          "The band sold 6 million copies .", "The band sold 4 million copies .")
        rows += pair_rows("p2", "vitc:A", "The film earned 30 awards .",
                          "The film earned 30 awards .", "The film earned 12 awards .")
        rows += pair_rows("p3", "vitc:B", "The river is the longest in the country .",
                          "The river is the longest in the country .", "The river is the shortest in the country .")
        rows += pair_rows("p4", "vitc:C", "The tower was built after the war .",
                          "The tower was built after the war .", "The tower was built before the war .")
        rows.append(dict(rows[0], id="half", pair="p5", hint="b"))       # 짝 없는 줄
        forbid = pc.Forbid([], [])
        forbid.groups.add("vitc:C")
        out, summary = pc.pick(rows, forbid, n=2, qual_share=0.5, seed=1)
        self.assertEqual(summary["dropped"], {"group": 1})              # 짝 단위로 센다 (p4)
        self.assertEqual(summary["chosen"]["pairs"], 2)
        self.assertEqual(summary["chosen"]["qual"], 1)                  # vitc:A 에서 한 짝만
        self.assertEqual(summary["chosen"]["swap"], 1)                  # p3 은 낱말 하나 바꿈
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted({r["pair"] for r in out}), sorted({out[0]["pair"], "p3"}))

    def test_claim_kind(self):
        self.assertEqual(pc.claim_kind("It has 3 floors .", "It has fewer floors ."), "qual")
        self.assertEqual(pc.claim_kind("It came second .", "It came third ."), "qual")
        self.assertEqual(pc.claim_kind("The cat is black .", "The cat is white ."), "swap")


class KeepPairsTest(unittest.TestCase):
    def test_keeps_only_ab_pairs_and_counts_reasons(self):
        cands = []
        for pid in ("p1", "p2", "p3", "p4"):
            cands += pair_rows(pid, "vitc:" + pid, "claim " + pid, "ev a " + pid, "ev b " + pid)
        judged = [{"id": "p1-a", "want": "a"}, {"id": "p1-b", "want": "b"},
                  {"id": "p2-a", "want": "a"}, {"id": "p2-b", "want": "d"},
                  {"id": "p3-a", "want": "a"},
                  {"id": "p4-a", "want": "a"}, {"id": "p4-b", "want": "a"}]
        kept, summary = pc.keep_pairs(judged, cands)
        self.assertEqual([r["id"] for r in kept], ["p1-a", "p1-b"])
        self.assertEqual(summary["dropped"], {"c_or_d": 1, "judged_dropped": 1, "cell_mismatch:aa": 1})
        self.assertEqual(summary["want"], {"a": 1, "b": 1})


class HalveTest(unittest.TestCase):
    def test_halves_each_stratum_by_pair(self):
        rows = []
        for i in range(5):
            for r in pair_rows(f"q{i}", f"vitc:q{i}", "c", "ea", "eb"):
                rows.append(dict(r, pick="qual", want=r["hint"]))
        for i in range(3):
            for r in pair_rows(f"s{i}", f"vitc:s{i}", "c", "ea", "eb"):
                rows.append(dict(r, pick="swap", want=r["hint"]))
        out, summary = pc.halve(rows, seed=7)
        self.assertEqual(summary["strata"], {"qual": {"pairs": 5, "kept": 2}, "swap": {"pairs": 3, "kept": 1}})
        self.assertEqual(summary["want"], {"a": 3, "b": 3})
        self.assertEqual([r["id"] for r in out], [r["id"] for r in rows if r["pair"] in {o["pair"] for o in out}])
        self.assertEqual(pc.halve(rows, seed=7)[0], out)                 # 같은 씨앗 같은 결과

    def test_refuses_broken_pair(self):
        rows = [dict(r, want=r["hint"]) for r in pair_rows("p1", "vitc:A", "c", "ea", "eb")[:1]]
        with self.assertRaises(ValueError):
            pc.halve(rows)


if __name__ == "__main__":
    unittest.main()
