"""dev_g9 trim · check · blind (K7 설계 1-2 · 4-1 2단계). 글은 모두 지어낸 것이다."""
import hashlib
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import dev_g9  # noqa: E402


def row(rid, group, guide="support-v6", want="a", ev="근거 글", cl="주장 글", lang="ko", src=None):
    return {"id": rid, "group": group, "groups": [group], "label_guide": guide, "want": want, "evidence": ev,
            "claim": cl, "lang": lang, "domain": "kowiki", "src_ids": src or [rid]}


class TrimTest(unittest.TestCase):
    def test_drop_layer_group_and_v5_list(self):
        dev = [row("d1", "g:1", "support-v5"), row("d2", "g:2", "support-v5"), row("d3", "g:3"),
               row("d4", "g:4", "support-v6", src=["s:9"])]
        layers = {"train-g1-arch": [row("t1", "g:2")], "train-r2": [row("t2", "g:x", src=["s:9"])]}
        drop, v5, why = dev_g9.trim(dev, layers)
        self.assertEqual(drop, ["d2", "d4"])           # 묶음 겹침 · 출처 id 겹침
        self.assertEqual(v5, ["d1"])                   # 뺀 쌍은 재라벨 대상에서도 빠진다
        self.assertEqual(why["train-g1-arch"], 1)


class CheckTest(unittest.TestCase):
    def test_counts_overlap_and_within(self):
        marks = {"train": dev_g9.Marks(), "blind-g4": dev_g9.Marks()}
        marks["train"].add_rows([row("t1", "g:1", ev="학습 근거 하나", cl="학습 주장 하나")], "train")
        hit = row("x", "g:9", ev="학습 근거 하나", cl="학습 주장 하나")
        sha = dev_g9.ms.sha_key({"evidence": "블라인드 근거", "claim": "블라인드 주장"})
        marks["blind-g4"].add_forbid({"groups": ["g:7"], "src_ids": [], "pair_keys": [sha]})
        cands = [{"id": "c1", "src": "kowiki", "form": "wiki-edit", "evidence": "새 근거", "claim": "새 주장"},
                 {"id": "c2", "src": "kowiki", "form": "wiki-edit", "evidence": hit["evidence"], "claim": hit["claim"]},
                 {"id": "c3", "src": "kowiki", "form": "wiki-edit", "evidence": "블라인드 근거", "claim": "블라인드 주장"}]
        hints = [{"id": "c1", "group": "g:5", "groups": ["g:5"], "src_ids": ["n1"]},
                 {"id": "c2", "group": "g:6", "groups": ["g:6"], "src_ids": ["n2"]},
                 {"id": "c3", "group": "g:7", "groups": ["g:7"], "src_ids": ["n3"]}]
        out = dev_g9.check(cands, hints, marks)
        self.assertEqual(out["text_train"], 1)
        self.assertEqual(out["group_blind-g4"], 1)
        self.assertEqual(out["sha_blind-g4"], 1)
        self.assertEqual(out["group_train"], 0)
        clean = dev_g9.check(cands[:1], hints[:1], marks)
        self.assertEqual(sum(clean.values()), 0)


class BlindTest(unittest.TestCase):
    def test_batches_hide_labels_and_map_back(self):
        new = [{"id": f"g9n-{i:012x}", "src": "kowiki", "form": "wiki-edit", "evidence": f"근거 {i}",
                "claim": f"주장 {i}"} for i in range(150)]
        old = [row(f"boolq-dev-{i:05d}", f"g:{i}", "support-v5", want="b") for i in range(60)]
        batches, table = dev_g9.blind(new, old, seed=7, size=100)
        self.assertEqual([len(b) for b in batches], [100, 100, 10])
        for b in batches:
            for r in b:
                self.assertEqual(set(r), {"bid", "evidence", "claim"})
        self.assertEqual(len(table), 210)
        first = next(m for m in table if m["id"] == "boolq-dev-00000")
        self.assertEqual(first["bid"], hashlib.sha256(b"g9-blind-v1:boolq-dev-00000").hexdigest()[:12])
        self.assertEqual((first["from"], first["old_want"]), ("old-dev", "b"))
        again, _ = dev_g9.blind(new, old, seed=7, size=100)
        self.assertEqual(batches, again)                 # 씨앗 고정 → 같은 차례
        self.assertNotEqual([r["bid"] for r in batches[0]][:5], sorted(m["bid"] for m in table)[:5])


class ExtraTest(unittest.TestCase):
    """보충 판 — 묶음 번호는 start 부터 · 앞 판 후보(avoid)와 겹치면 걸린다 · near-x 는 허용할 때만."""

    def test_blind_start_numbers_batches(self):
        new = [{"id": f"g9n-{i:012x}", "src": "kowiki", "form": "near-x", "evidence": f"근거 {i}",
                "claim": f"주장 {i}"} for i in range(120)]
        batches, table = dev_g9.blind(new, [], seed=7, size=100, start=24)
        self.assertEqual([len(b) for b in batches], [100, 20])
        self.assertEqual(sorted({m["batch"] for m in table}), ["g9-b024", "g9-b025"])
        self.assertTrue(all(m["from"] == "new-ko" for m in table))

    def test_avoid_marks_and_near_x_gate(self):
        old = row("o1", "g:1", ev="앞 판 근거", cl="앞 판 주장")
        marks, _ = dev_g9.load_marks(os.path.join(helpers.ROOT, "no-such-root"), avoid={"cands-g9": [old]})
        self.assertIn("group", marks["cands-g9"].reasons(row("n1", "g:1", ev="다른 근거", cl="다른 주장")))
        rows = [dict(row("r1", "g:5", ev="근접 근거 하나", cl="근접 주장 하나"), hint="c", form="near-x"),
                dict(row("r2", "g:6", ev="근접 근거 둘", cl="근접 주장 둘"), hint="c", form="near-x")]
        plain = {"train": dev_g9.Marks()}
        got, dropped = dev_g9.pick_source("kowiki", rows, {"total": 5, "b": 0}, None, plain, dev_g9.random.Random(1))
        self.assertEqual((len(got), dropped["generator_form"]), (0, 2))
        got, _ = dev_g9.pick_source("kowiki", rows, {"total": 5, "b": 0}, None, plain, dev_g9.random.Random(1),
                                    allow_forms={"near-x"})
        self.assertEqual(sorted(r["id"] for r in got), ["r1", "r2"])


if __name__ == "__main__":
    unittest.main()
