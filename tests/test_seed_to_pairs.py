"""seed_to_pairs — 씨앗 → 쌍 펴기 · 두 판정 합치기(짝째 버림) · 금지 목록. 입력은 모두 지어낸 짧은 글이다."""
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import seed_to_pairs as sp  # noqa: E402


def seed(sid, a=None, d=None, group="doc:studio:X/a.md"):
    return {"seed_id": sid, "group": group, "seed": "캐시는 3개를 둔다.", "slot": {"kind": "num"},
            "lines": {"b": "캐시는 5개를 둔다.", "a": a, "d": d}}


class PairsTest(unittest.TestCase):
    def test_spreads_lines_and_skips_missing(self):
        rows = sp.seed_pairs(seed("k1", d="5월에 캐시는 4개를 둔다."))
        self.assertEqual([r["hint"] for r in rows], ["b", "d"])
        b = rows[0]
        self.assertEqual((b["id"], b["evidence"], b["claim"], b["pair"]), ("seed-k1-b", "캐시는 3개를 둔다.", "캐시는 5개를 둔다.", "k1"))
        self.assertEqual((b["origin"], b["domain"], b["lic"], b["lang"], b["groups"]), ("g3-seed", "seed", "MIT", "ko", ["doc:studio:X/a.md"]))


class JudgeTest(unittest.TestCase):
    def test_agree_disagree_and_drop_whole_seed(self):
        pairs = sp.seed_pairs(seed("k1", a="캐시는 셋을 둔다.")) + sp.seed_pairs(seed("k2", a="캐시는 셋을 둔다."))
        j1 = [{"id": "seed-k1-b", "label": "b"}, {"id": "seed-k1-a", "label": "a"},
              {"id": "seed-k2-b", "label": "b"}, {"id": "seed-k2-a", "label": "d"}]
        j2 = [{"id": "seed-k1-b", "label": "b"}, {"id": "seed-k1-a", "label": "a"},
              {"id": "seed-k2-b", "label": "b"}, {"id": "seed-k2-a", "label": "a"}]
        rows, summary = sp.judge_rows(pairs, j1, j2)
        keep = {r["id"]: r["keep"] for r in rows}
        self.assertEqual(keep, {"seed-k1-b": True, "seed-k1-a": True, "seed-k2-b": False, "seed-k2-a": False})
        self.assertEqual(rows[3]["judge"]["final"], "disagree")
        self.assertEqual(summary["by_hint"]["a"]["disagree"], 1)
        self.assertEqual(summary["seeds_kept"], 1)
        with self.assertRaises(ValueError):
            sp.judge_rows(pairs, j1[:3], j2)

    def test_forbid_lines_split_doc_and_mem(self):
        seeds = [seed("k1"), seed("k2", group="mem:20261001-abcd1234")]
        lines = sp.forbid_lines(seeds, ["doc:studio:Y/b.md"], sp.seed_pairs(seeds[0]))
        self.assertIn("doc_paths\tstudio:X/a.md", lines)
        self.assertIn("mem_ids\t20261001-abcd1234", lines)
        self.assertEqual(sum(1 for line in lines if line.startswith("pair_keys\t")), 1)


if __name__ == "__main__":
    unittest.main()
