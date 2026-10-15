"""seed_to_pairs — 씨앗 → 쌍 펴기 · 두 판정 합치기(짝째 버림) · 금지 목록. 입력은 모두 지어낸 짧은 글이다."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

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
        self.assertEqual(b["form"], "contrast")

    def test_paragraph_seed_uses_paragraph_and_domain(self):
        # K4 archive 씨앗 (seed_contrast --evidence paragraph) — 근거 = 문단 · domain archive · form seed-<칸>
        row = dict(seed("ka0001", a="캐시는 셋을 둔다."), evidence="앞 문장이다. 캐시는 3개를 둔다. 뒤 문장이다.",
                   domain="archive", evidence_mode="paragraph")
        rows = sp.seed_pairs(row)
        self.assertEqual([(r["id"], r["form"]) for r in rows], [("seed-ka0001-b", "seed-b"), ("seed-ka0001-a", "seed-a")])
        for r in rows:
            self.assertEqual((r["evidence"], r["domain"], r["origin"]), (row["evidence"], "archive", "g4-seed"))

    def test_drafts_used_when_no_lines(self):
        # seed_contrast make --evidence paragraph 출력 — lines 없이 drafts 만 있다 (None 칸은 건너뛴다)
        row = dict(seed("ka0002", a="캐시는 셋을 둔다."), evidence="캐시는 3개를 둔다.", domain="archive",
                   evidence_mode="paragraph")
        row["drafts"] = row.pop("lines")
        rows = sp.seed_pairs(row)
        self.assertEqual([(r["id"], r["lines_from"]) for r in rows],
                         [("seed-ka0002-b", "rule-draft"), ("seed-ka0002-a", "rule-draft")])
        self.assertNotIn("lines_from", sp.seed_pairs(seed("k1"))[0])         # lines 가 있으면 옛 꼴 그대로

    def test_drafts_and_lines_inputs_give_same_pairs(self):
        # 같은 씨앗을 drafts 꼴 · lines 꼴로 줘도 lines_from 칸만 빼면 출력 파일 sha 가 같다
        base = dict(seed("ka0003", a=None, d="5월에 캐시는 4개를 둔다."), evidence="캐시는 3개를 둔다.",
                    domain="archive", evidence_mode="paragraph")
        drafts = {k: v for k, v in base.items() if k != "lines"}
        drafts["drafts"] = base["lines"]
        with tempfile.TemporaryDirectory() as tmp:
            shas = []
            for name, row in (("lines", base), ("drafts", drafts)):
                src, out = os.path.join(tmp, f"{name}.jsonl"), os.path.join(tmp, f"{name}-pairs.jsonl")
                sp.write_jsonl(src, [row])
                with unittest.mock.patch("sys.stdout"):
                    self.assertEqual(sp.main(["pairs", "--in", src, "--out", out]), 0)
                rows = [{k: v for k, v in r.items() if k != "lines_from"} for r in sp.read_jsonl(out)]
                shas.append(hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest())
            self.assertEqual(shas[0], shas[1])
            self.assertEqual(len(sp.read_jsonl(os.path.join(tmp, "drafts-pairs.jsonl"))), 2)


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
