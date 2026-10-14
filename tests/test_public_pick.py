"""public_pick — BoolQ · KLUE 후보 뽑기 · 판정 입력 숨기기. 입력은 모두 지어낸 짧은 글이다."""
import collections
import contextlib
import io
import os
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import public_pick as pp  # noqa: E402


def row(i, hint, split, group=None, rewrite=None, near=None):
    r = {"id": f"boolq-{split}-{i:05d}", "evidence": f"그 다리는 {i}년에 지었다.", "claim": f"그 다리는 {i + 1}년에 지었나?",
         "hint": hint, "split": split, "group": f"boolq:{group if group is not None else i}"}
    if rewrite:
        r["claim_rewrite"] = rewrite
    if near:
        r["near_train"] = near
    return r


class PrepareTest(unittest.TestCase):
    def test_claim_choice_and_drops(self):
        rows = [row(1, "b", "dev", rewrite="그 다리는 2년에 지었다."),
                row(2, "b", "dev"),  # 물음꼴이 남음
                row(3, "b", "dev", rewrite="그 다리는 4년에 지었다.", near="near"),
                row(4, "a", "test", rewrite="그 다리는 5년에 지었다.", group="t"),
                row(5, "a", "test", rewrite="그 다리는 6년에 지었다.")]
        out, dropped = pp.prepare(rows, "eval", drop_ids={"boolq-test-00005"}, forbid_groups={"boolq:t"})
        self.assertEqual([r["id"] for r in out], ["boolq-dev-00001"])
        self.assertEqual(out[0]["claim"], "그 다리는 2년에 지었다.")
        self.assertEqual(out[0]["claim_question"], "그 다리는 2년에 지었나?")
        self.assertEqual(out[0]["kind"], "boolq-dev")
        self.assertEqual(dropped, {"check_fail": 1, "forbid_group": 1, "near_train": 1, "question": 1})
        out, _ = pp.prepare(rows[2:3], "train")  # 학습은 near_train 을 안 본다
        self.assertEqual(len(out), 1)


class PickTest(unittest.TestCase):
    def test_quota_group_caps_and_a_mate(self):
        rows = [dict(row(i, "b", "dev", group=i // 3), claim="주장.", kind="boolq-dev") for i in range(30)]
        rows += [dict(row(100 + i, "b", "test", group=100 + i), claim="주장.", kind="boolq-test") for i in range(30)]
        rows += [dict(row(200, "a", "dev", group=0), claim="주장.", kind="boolq-dev")]
        rows += [dict(row(300 + i, "a", "test", group=300 + i), claim="주장.", kind="boolq-test") for i in range(10)]
        out, quota = pp.pick_rows(rows, "eval", n_b=10, n_a=4, seed=7)
        b = [r for r in out if r["hint"] == "b"]
        self.assertEqual(quota["b"], {"boolq-dev": 5, "boolq-test": 5})
        self.assertLessEqual(max(collections.Counter(r["group"] for r in b).values()), 2)
        a = [r for r in out if r["hint"] == "a"]
        self.assertEqual(len(a), 4)
        self.assertEqual(len({r["group"] for r in a}), 4)
        if any(r["group"] == "boolq:0" for r in b):
            self.assertEqual([r["a_mate"] for r in a if r["group"] == "boolq:0"], ["same"])
        self.assertTrue(all(r["pick_set"] == "eval" for r in out))
        again, _ = pp.pick_rows(rows, "eval", n_b=10, n_a=4, seed=7)
        self.assertEqual([r["id"] for r in out], [r["id"] for r in again])
        with self.assertRaises(ValueError):
            pp.pick_rows(rows, "eval", n_b=100, n_a=4, seed=7)

    def test_blind_hides_fields_and_shuffles(self):
        rows = [dict(row(i, "b", "dev"), near_train="near") for i in range(20)]
        one, two = pp.blind_rows(rows, 101), pp.blind_rows(rows, 202)
        self.assertEqual(set(one[0]), {"id", "evidence", "claim"})
        self.assertEqual(sorted(r["id"] for r in one), sorted(r["id"] for r in two))
        self.assertNotEqual([r["id"] for r in one], [r["id"] for r in two])


def klue(i, hint, split, pair, group, rewrite=True):
    r = {"id": f"klue-ans-{i:06d}", "evidence": "그 다리는 1990년에 지었고 2001년에 고쳤다.", "claim": "그 다리는 언제 지었는가?",
         "hint": hint, "split": split, "group": f"klue:{group}", "pair": pair, "domain": "klue-answered"}
    if rewrite:
        r["claim_rewrite"] = "그 다리는 2001년에 지었다." if hint == "b" else "그 다리는 1990년에 지었다."
    return r


class KlueTest(unittest.TestCase):
    def test_pair_check_no_rewrite_and_kind(self):
        rows = [klue(1, "b", "heldout", "p1", "g1"), klue(2, "a", "heldout", "p1", "g1"),
                klue(3, "b", "heldout", "p2", "g2"), klue(4, "a", "heldout", "p2", "g2"),
                klue(5, "b", "heldout", "p3", "g3", rewrite=False)]
        drop = pp.failed_ids([{"pair": "p2", "pass": False, "why": {}}, {"pair": "p1", "pass": True}])
        self.assertEqual(drop, {"p2"})
        out, dropped = pp.prepare(rows, "eval", drop_ids=drop)
        self.assertEqual([r["id"] for r in out], ["klue-ans-000001", "klue-ans-000002"])
        self.assertEqual({r["kind"] for r in out}, {"klue-heldout"})
        self.assertEqual(out[0]["claim"], "그 다리는 2001년에 지었다.")
        self.assertEqual(dropped, {"check_fail": 2, "no_rewrite": 1})

    def test_forbid_uses_other_split(self):
        rows = [klue(1, "b", "heldout", "p1", "g1"), klue(2, "b", "train", "p2", "g1"), klue(3, "b", "train", "p3", "g3")]
        self.assertEqual(pp.forbid_groups_for(rows, "eval"), {"klue:g1", "klue:g3"})
        self.assertEqual(pp.forbid_groups_for(rows, "train"), {"klue:g1"})
        out, dropped = pp.prepare(rows, "train", forbid_groups=pp.forbid_groups_for(rows, "train"))
        self.assertEqual([r["id"] for r in out], ["klue-ans-000003"])
        self.assertEqual(dropped, {"forbid_group": 2})


class FinalizeTest(unittest.TestCase):
    """finalize — git 안 출력 거부 · 없는 폴더 만들기 · 줄끝 LF."""

    def judged(self, tmp):
        rows = [{"id": "boolq-x-00001", "evidence": "근거", "claim": "주장", "hint": "b", "group": "boolq:1",
                 "pick_set": "eval", "keep": True, "kind": "boolq", "judge": {"final": "b"}},
                {"id": "boolq-x-00002", "evidence": "근거2", "claim": "주장2", "hint": "a", "group": "boolq:2",
                 "pick_set": "train", "keep": True, "kind": "boolq", "judge": {"final": "a"}}]
        path = os.path.join(tmp, "judged.jsonl")
        pp.wp.write_jsonl(path, rows)
        return path

    def test_refuses_out_dir_inside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(helpers.ROOT, "no-such-finalize-out")
            with contextlib.redirect_stderr(io.StringIO()):
                code = pp.main(["finalize", "--in", self.judged(tmp), "--out-dir", out_dir])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_dir))

    def test_makes_out_dir_and_writes_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "new", "boolq")
            with contextlib.redirect_stdout(io.StringIO()):
                code = pp.main(["finalize", "--in", self.judged(tmp), "--out-dir", out_dir])
            self.assertEqual(code, 0)
            for name in ("boolq-labeled-eval.jsonl", "boolq-labeled-train.jsonl", "boolq-labeled.summary.json"):
                with open(os.path.join(out_dir, name), "rb") as f:
                    data = f.read()
                self.assertTrue(data, name)
                self.assertNotIn(b"\r\n", data, name)


if __name__ == "__main__":
    unittest.main()
