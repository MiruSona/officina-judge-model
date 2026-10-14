"""wiki_pick — 위키 후보 층화 뽑기 · 두 판정 합치기. 입력은 모두 지어낸 짧은 글이다."""
import collections
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import wiki_pick as wp  # noqa: E402


def row(i, hint, kind, group, flag=()):
    return {"evidence": f"문서 : 값은 {i}이다.", "claim": f"값은 {i + 1}이다.", "want": "", "hint": hint, "kind": kind,
            "flag": list(flag), "group": f"kowiki:{group}", "src_ids": [f"kowiki:{i}", f"kowiki:{i + 100000}"]}


class PickTest(unittest.TestCase):
    def test_strata_doc_cap_flag_cap_and_all_d(self):
        rows = [row(i, "b", "date", i // 3, flag=("reverted",) if i % 2 else ()) for i in range(60)]
        rows += [row(100 + i, "b", "num", 100 + i) for i in range(30)]
        rows += [row(200 + i, "b", "ordinal", 200 + i) for i in range(10)]
        rows += [row(300 + i, "d", "date", 300) for i in range(4)]
        out, quota = wp.pick_rows(rows, 20, "eval", seed=7)
        b = [r for r in out if r["hint"] == "b"]
        self.assertEqual(quota, {"date": 12, "num": 6, "ordinal": 2})
        self.assertEqual(collections.Counter(r["kind"] for r in b), collections.Counter(quota))
        self.assertLessEqual(max(collections.Counter(r["group"] for r in b).values()), 2)
        self.assertLessEqual(sum(1 for r in b if r["flag"]), int(12 * 0.2))
        self.assertEqual(sum(1 for r in out if r["hint"] == "d"), 4)  # d 는 문서가 같아도 다 넣는다
        self.assertTrue(all(r["pick_set"] == "eval" and r["id"].startswith("wiki-") for r in out))
        again, _ = wp.pick_rows(rows, 20, "eval", seed=7)
        self.assertEqual([r["id"] for r in out], [r["id"] for r in again])  # 씨앗 고정이면 같은 결과
        with self.assertRaises(ValueError):
            wp.pick_rows(rows[:6], 5, "eval", seed=7)  # 두 문서 · 문서당 2 라 4 까지만


class JudgeTest(unittest.TestCase):
    def test_agree_disagree_noise_keep(self):
        pairs = [dict(row(i, h, "num", i), id=f"w{i}") for i, h in enumerate("bbdd")]
        j1 = [{"id": "w0", "label": "b"}, {"id": "w1", "label": "a"}, {"id": "w2", "label": "noise"}, {"id": "w3", "label": "b"}]
        j2 = [{"id": "w0", "label": "b"}, {"id": "w1", "label": "b"}, {"id": "w2", "label": "noise"}, {"id": "w3", "label": "b"}]
        rows, summary = wp.judge_rows(pairs, j1, j2)
        self.assertEqual([r["keep"] for r in rows], [True, False, False, True])
        self.assertEqual([r["want"] for r in rows], ["b", "", "", "b"])
        self.assertEqual(rows[1]["judge"]["final"], "disagree")
        self.assertEqual(rows[0]["label_guide"], "support-v6")
        self.assertEqual(summary["by_hint"]["d"], {"n": 2, "labels": {"b": 1, "noise": 1}, "disagree": 0, "noise": 1,
                                                    "kept": 1, "kept_as_hint": 0})
        with self.assertRaises(ValueError):
            wp.judge_rows(pairs, j1[:3], j2)
        with self.assertRaises(ValueError):
            wp.judge_rows(pairs, j1[:3] + [{"id": "w3", "label": "x"}], j2)


def judged(i, hint, final, pick_set, group, keep=None):
    r = dict(row(i, hint, "num", group, flag=("revert",) if i % 2 else ()), id=f"w{i}", pick_set=pick_set,
             judge={"label1": final, "label2": final, "final": final, "reason": ""})
    r["keep"] = final not in ("disagree", "noise") if keep is None else keep
    return r


class FinalizeTest(unittest.TestCase):
    def test_keep_only_hint_to_final_and_override(self):
        rows = [judged(0, "b", "b", "eval", 1), judged(1, "b", "a", "eval", 2), judged(2, "b", "noise", "eval", 3),
                judged(3, "d", "d", "train", 10), judged(4, "d", "disagree", "train", 11),
                judged(5, "d", "disagree", "train", 12)]
        ev, tr, summary = wp.finalize_rows(rows, wp.parse_overrides(["w4=a"]))
        self.assertEqual([r["id"] for r in ev], ["w0", "w1"])  # noise 는 빠진다
        self.assertEqual([r["id"] for r in tr], ["w3", "w4"])  # override 없는 disagree 는 빠진다
        self.assertEqual([(r["hint_pick"], r["hint"], r["want"]) for r in ev + tr],
                         [("b", "b", "b"), ("b", "a", "a"), ("d", "d", "d"), ("d", "a", "a")])
        self.assertEqual(tr[1]["judge"]["override"], {"from": "disagree", "to": "a"})
        self.assertTrue(all(r["label_guide"] == "support-v6" and r["keep"] for r in ev + tr))
        self.assertEqual(summary["sets"]["eval"]["labels"], {"a": 1, "b": 1})
        self.assertEqual(summary["sets"]["train"]["by_hint_pick"], {"d": {"a": 1, "d": 1}})
        self.assertEqual(summary["sets"]["eval"]["flag_rate"], 0.5)
        self.assertEqual(summary["doc_overlap_eval_train"], 0)

    def test_rejects_overlap_dup_and_bad_override(self):
        rows = [judged(0, "b", "b", "eval", 1), judged(1, "b", "b", "train", 1)]
        with self.assertRaises(ValueError):
            wp.finalize_rows(rows)  # 평가 · 학습이 같은 문서
        ok = [judged(0, "b", "b", "eval", 1), judged(1, "b", "b", "train", 2)]
        with self.assertRaises(ValueError):
            wp.finalize_rows(ok + [ok[0]])  # 같은 id 두 번
        with self.assertRaises(ValueError):
            wp.finalize_rows(ok, {"w9": "a"})  # 없는 id
        for bad in ("w0", "w0=noise", "=a"):
            with self.assertRaises(ValueError):
                wp.parse_overrides([bad])


if __name__ == "__main__":
    unittest.main()
