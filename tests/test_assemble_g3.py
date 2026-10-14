"""assemble_g3 — g3 평가 셋 조립 (g2 잇기 · 확정 칸 통일 · 판정 전 줄 · 중복 · 얼림 · 규약 해시). 글은 모두 지어낸 것이다."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import assemble_g3 as ag  # noqa: E402


def g2(rid, want, lang="ko"):
    return {"id": rid, "evidence": f"옛 근거 {rid}", "claim": f"옛 주장 {rid}", "want": want, "label_guide": "support-v5",
            "domain": "mem", "group": f"g2:{rid}", "groups": [f"g2:{rid}"], "lang": lang, "hint": want,
            "set": "dev-cmp-g2", "g2_label": "v5-carry"}


def cand(rid, hint, judge=None, want="", **extra):
    row = {"id": rid, "evidence": f"새 근거 {rid}", "claim": f"새 주장 {rid}", "want": want, "hint": hint,
           "domain": "kowiki", "group": f"new:{rid}", "groups": [f"new:{rid}"], "lang": "ko", "src_ids": [rid],
           "set": "dev-cmp-g3", "g3_label": hint}
    if judge is not None:
        row["judge"] = judge
    row.update(extra)
    return row


def run(cands, g2_rows=(), decisions=None, draft=False, set_name="dev-cmp-g3"):
    return ag.assemble(set_name, list(cands), list(g2_rows), decisions or {}, draft)


class MergeTest(unittest.TestCase):
    def test_g2_carried_and_new_added(self):
        rows, s = run([cand("wiki-2", "b", {"label1": "b", "label2": "b", "final": "b"}, want="b")],
                      [g2("mem-1", "c"), g2("mem-3", "b")])
        self.assertEqual([r["id"] for r in rows], ["mem-1", "mem-3", "wiki-2"])   # id 차례
        old = rows[0]
        self.assertEqual((old["set"], old["g2_set"], old["g3_label"]), ("dev-cmp-g3", "dev-cmp-g2", "g2-carry"))
        self.assertEqual((old["want"], old["label_guide"]), ("c", "support-v5"))   # g2 칸 · 기준 그대로
        self.assertEqual((s["g2_pairs"], s["new_pairs"], s["ko_b"]), (2, 1, 2))
        self.assertEqual(s["domain_want"]["kowiki"], {"b": 1})

    def test_new_id_clashing_with_g2_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("mem-1", "b", {"final": "b"})], [g2("mem-1", "c")])


class LabelTest(unittest.TestCase):
    def test_mit_judge_label_wins_over_hint(self):
        rows, _ = run([cand("mit-1", "b", {"j1": {"label": "a"}, "j2": {"label": "a"}, "label": "a", "status": "agree"})])
        r = rows[0]
        self.assertEqual((r["want"], r["hint"], r["hint_pick"]), ("a", "a", "b"))
        self.assertEqual((r["votes"], r["status"], r["label_guide"], r["g3_label"]),
                         ({"j1": "a", "j2": "a"}, "agree", "support-v6", "v6-new"))
        self.assertNotIn("judge", r)

    def test_boolq_want_used_hint_kept_as_pick(self):
        rows, _ = run([cand("boolq-1", "b", {"label1": "d", "label2": "d", "final": "d"}, want="d")])
        self.assertEqual((rows[0]["want"], rows[0]["hint"], rows[0]["hint_pick"]), ("d", "d", "b"))

    def test_existing_hint_pick_kept(self):
        rows, _ = run([cand("wiki-1", "a", {"final": "a"}, want="a", hint_pick="b")])
        self.assertEqual(rows[0]["hint_pick"], "b")

    def test_want_disagreeing_with_judge_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("boolq-1", "b", {"final": "a"}, want="b")])

    def test_wrong_set_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("x-1", "b", {"final": "b"}, set="blind-g3")])

    def test_override_wins_over_final(self):
        judge = {"label1": "a", "label2": "a", "final": "a", "override": {"from": "a", "to": "c"}}
        rows, _ = run([cand("wiki-9", "b", judge, want="c")])
        self.assertEqual((rows[0]["want"], rows[0]["hint"]), ("c", "c"))

    def test_keys_shared_with_make_sets(self):
        import make_sets
        self.assertIs(ag.text_keys, make_sets.text_keys)
        self.assertIs(ag.sha_key, make_sets.sha_key)


class PendingTest(unittest.TestCase):
    def cands(self):
        return [cand("vit-1", "b", domain="vitaminc", lang="en"), cand("vit-2", "a", domain="vitaminc", lang="en"),
                cand("vit-3", "b", domain="vitaminc", lang="en"), cand("vit-4", "c", domain="vitaminc", lang="en")]

    def decisions(self):
        return {"vit-1": {"id": "vit-1", "label": "b", "status": "agree", "votes": {"j1": "b", "j2": "b"}},
                "vit-2": {"id": "vit-2", "label": None, "status": "drop", "votes": {}},
                "vit-3": {"id": "vit-3", "label": None, "status": "need-third", "votes": {}}}

    def test_draft_counts_and_drops_pending(self):
        rows, s = run(self.cands(), decisions=self.decisions(), draft=True)
        self.assertEqual([r["id"] for r in rows], ["vit-1"])
        self.assertEqual(rows[0]["votes"], {"j1": "b", "j2": "b"})
        self.assertEqual((s["pending"], s["dropped"], s["cand_state"]),
                         (2, {"vitaminc": 1}, {"decided": 1, "dropped": 1, "pending": 2}))
        self.assertEqual(s["pending_by"], {"vitaminc · hint b": 1, "vitaminc · hint c": 1})

    def test_not_draft_stops_on_pending(self):
        with self.assertRaises(ag.Stop):
            run(self.cands(), decisions=self.decisions())

    def test_decided_folder_read_and_double_id_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("one", "two"):
                os.makedirs(os.path.join(tmp, name))
                with open(os.path.join(tmp, name, "decided.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"q": "q1", "id": "vit-1", "label": "b", "status": "agree"}) + "\n")
            self.assertIn("vit-1", ag.load_decisions([os.path.join(tmp, "one"), os.path.join(tmp, "missing")]))
            with self.assertRaises(ag.Stop):
                ag.load_decisions([os.path.join(tmp, "one"), os.path.join(tmp, "two")])


class DedupeTest(unittest.TestCase):
    def test_same_norm_text_new_rows_keeps_small_id(self):
        a = cand("mit-2", "b", {"label": "b"})
        b = dict(cand("mit-1", "b", {"label": "b"}), evidence=" " + a["evidence"].upper() + "  ", claim=a["claim"])
        rows, s = run([a, b])
        self.assertEqual([r["id"] for r in rows], ["mit-1"])
        self.assertEqual(s["deduped"], ["mit-2"])

    def test_new_row_same_as_g2_dropped(self):
        old = g2("mem-1", "b")
        new = dict(cand("wiki-1", "b", {"final": "b"}), evidence=old["claim"], claim=old["evidence"])   # 차례 바뀜
        rows, s = run([new], [old])
        self.assertEqual([r["id"] for r in rows], ["mem-1"])
        self.assertEqual(s["deduped"], ["wiki-1"])

    def test_same_text_other_label_stops(self):
        a = cand("mit-1", "b", {"label": "b"})
        b = dict(cand("mit-2", "b", {"label": "a"}), evidence=a["evidence"], claim=a["claim"])
        with self.assertRaises(ag.Stop):
            run([a, b])


class OverlapTest(unittest.TestCase):
    def test_counts_across_sets_and_train(self):
        x, y = cand("a-1", "b"), cand("a-2", "b")
        y["groups"] = ["new:a-1"]
        train = [dict(cand("t-1", "a"), evidence=x["evidence"], claim=x["claim"])]
        got = ag.overlap_counts({"dev-cmp-g3": [x], "blind-g3": [y]}, train)
        self.assertEqual((got["sets_group"], got["sets_text"], got["train_text"], got["train_key"]), (1, 0, 1, 1))


DRAFT = """{
 "name": "protocol-g3",
 "sets": {
  "dev-cal-g3": "TODO-dev-cal-g3",
  "dev-cmp-g3": "TODO-dev-cmp-g3",
  "blind-g3": "TODO-blind-g3"
 },
 "note": "그대로 둔다"
}
"""


class CheckTest(unittest.TestCase):
    def _sets(self, tmp):
        for name in ag.G3_SETS:
            rows, _ = run([cand(f"{name}-1", "b", {"final": "b"}, set=name)], set_name=name)
            ag.write_jsonl(os.path.join(tmp, f"{name}.jsonl"), rows)
        names = os.path.join(tmp, "names.txt")
        with open(names, "w", encoding="utf-8") as f:
            f.write("")
        return ["--dest", tmp, "--build", os.path.join(tmp, "build"), "--names", names, "--game-names", names]

    def test_missing_train_layer_stops_unless_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._sets(tmp)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(["check"] + args), 2)
                self.assertEqual(ag.main(["check"] + args + ["--draft"]), 0)


class ProtocolTest(unittest.TestCase):
    def test_fill_only_todo(self):
        hashes = {n: n[0] * 64 for n in ag.G3_SETS}
        out = ag.fill_protocol(DRAFT, hashes)
        self.assertEqual(json.loads(out)["sets"], hashes)
        self.assertEqual(out.replace("d" * 64, "X").replace("b" * 64, "X").count("\n"), DRAFT.count("\n"))
        self.assertIn('"note": "그대로 둔다"', out)

    def test_missing_todo_stops(self):
        with self.assertRaises(ag.Stop):
            ag.fill_protocol(DRAFT.replace('"TODO-blind-g3"', '"x"'), {n: "0" * 64 for n in ag.G3_SETS})

    def test_freeze_then_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ag.G3_SETS:
                rows, s = run([cand(f"{name}-1", "b", {"final": "b"}, set=name)], set_name=name)
                path = os.path.join(tmp, f"{name}.jsonl")
                ag.write_jsonl(path, rows)
                s["sha256"] = ag.sha256_file(path)
                ag.write_json(os.path.join(tmp, f"{name}.summary.json"), s)
                self.assertEqual(ag.main(["freeze", "--set", name, "--dest", tmp]), 0)
                self.assertEqual(ag.main(["freeze", "--set", name, "--dest", tmp]), 2)   # 두 번 얼리지 않는다
            draft = os.path.join(tmp, "protocol-g3.draft.json")
            with open(draft, "w", encoding="utf-8", newline="") as f:
                f.write(DRAFT)
            out = os.path.join(tmp, "protocol-g3.json")
            self.assertEqual(ag.main(["protocol", "--draft", draft, "--dest", tmp, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                sets = json.load(f)["sets"]
            self.assertEqual(sets["blind-g3"], ag.sha256_file(os.path.join(tmp, "blind-g3.jsonl")))
            self.assertEqual(ag.main(["protocol", "--draft", draft, "--dest", tmp, "--out", out]), 2)   # 덮지 않는다

    def test_freeze_refuses_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, s = run([cand("x-1", "b", {"final": "b"})], draft=True)
            path = os.path.join(tmp, "dev-cmp-g3.jsonl")
            ag.write_jsonl(path, rows)
            s["sha256"] = ag.sha256_file(path)
            ag.write_json(os.path.join(tmp, "dev-cmp-g3.summary.json"), s)
            self.assertEqual(ag.main(["freeze", "--set", "dev-cmp-g3", "--dest", tmp]), 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "FROZEN-dev-cmp-g3.sha256")))


if __name__ == "__main__":
    unittest.main()
