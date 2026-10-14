"""assemble_g4 — g4 평가 셋 조립 (g3 잇기 · 확정 칸 · 판정 전 줄 · 중복 · min_groups · 얼림 · 규약 해시).

test_assemble_g3 의 시험을 g4 이름으로 옮기고 min_groups · 이은 줄 바이트를 더했다. 글은 모두 지어낸 것이다.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import assemble_g4 as ag  # noqa: E402


def g3(rid, want, lang="ko"):
    return {"id": rid, "evidence": f"옛 근거 {rid}", "claim": f"옛 주장 {rid}", "want": want, "label_guide": "support-v6",
            "domain": "mem", "group": f"g3:{rid}", "groups": [f"g3:{rid}"], "lang": lang, "hint": want,
            "set": "dev-cmp-g3", "g2_set": "dev-cmp-g2", "g3_label": "g2-carry"}


def cand(rid, hint, judge=None, want="", **extra):
    row = {"id": rid, "evidence": f"새 근거 {rid}", "claim": f"새 주장 {rid}", "want": want, "hint": hint,
           "domain": "kowiki", "group": f"new:{rid}", "groups": [f"new:{rid}"], "lang": "ko", "src_ids": [rid],
           "set": "dev-cmp-g4", "g4_source": "wiki", "g4_label": hint}
    if judge is not None:
        row["judge"] = judge
    row.update(extra)
    return row


def run(cands, g3_rows=(), decisions=None, draft=False, set_name="dev-cmp-g4"):
    return ag.assemble(set_name, list(cands), list(g3_rows), decisions or {}, draft, ag.G4)


class MergeTest(unittest.TestCase):
    def test_g3_carried_and_new_added(self):
        rows, s = run([cand("wiki-2", "c", {"label1": "c", "label2": "c", "final": "c"}, want="c")],
                      [g3("mem-1", "c"), g3("mem-3", "b")])
        self.assertEqual([r["id"] for r in rows], ["mem-1", "mem-3", "wiki-2"])
        old = rows[0]
        self.assertEqual((old["set"], old["g3_set"], old["g4_label"], old["g3_label"]),
                         ("dev-cmp-g4", "dev-cmp-g3", "g3-carry", "g2-carry"))
        self.assertEqual((s["g3_base"], s["g3_pairs"], s["new_pairs"], s["ko_b"]), ("dev-cmp-g3", 2, 1, 1))
        self.assertEqual(s["g4_label"], {"g3-carry": 2, "v6-new": 1})
        self.assertEqual(s["new_domain_want"]["kowiki"], {"c": 1})
        self.assertNotIn("g2_base", s)

    def test_carried_rows_bytes_same(self):
        # 이은 g3 줄은 set 만 새 이름 · g3_set · g4_label 두 칸이 붙을 뿐, 나머지 칸 · 차례 · 값은 그대로
        olds = [g3("mem-1", "c"), g3("mem-3", "b", lang="en")]
        rows, _ = run([cand("wiki-2", "c", {"final": "c"})], olds)
        by_id = {r["id"]: r for r in rows}
        for old in olds:
            got = dict(by_id[old["id"]])
            self.assertEqual((got.pop("g3_set"), got.pop("g4_label")), (old["set"], "g3-carry"))
            got["set"] = old["set"]
            self.assertEqual(json.dumps(got, ensure_ascii=False), json.dumps(old, ensure_ascii=False))

    def test_blind_g4_has_no_base(self):
        rows, s = run([cand("vit-1", "b", {"final": "b"}, set="blind-g4", lang="en")], set_name="blind-g4")
        self.assertEqual((s["g3_base"], s["g3_pairs"], [r["set"] for r in rows]), (None, 0, ["blind-g4"]))

    def test_new_id_clashing_with_g3_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("mem-1", "b", {"final": "b"})], [g3("mem-1", "c")])


class LabelTest(unittest.TestCase):
    def test_judge_label_wins_over_hint(self):
        rows, _ = run([cand("w-1", "c", {"j1": {"label": "d"}, "j2": {"label": "d"}, "label": "d", "status": "agree"})])
        r = rows[0]
        self.assertEqual((r["want"], r["hint"], r["hint_pick"]), ("d", "d", "c"))
        self.assertEqual((r["votes"], r["status"], r["label_guide"], r["g4_label"]),
                         ({"j1": "d", "j2": "d"}, "agree", "support-v6", "v6-new"))
        self.assertNotIn("judge", r)
        self.assertNotIn("g3_label", r)

    def test_existing_hint_pick_kept(self):
        rows, _ = run([cand("wiki-1", "a", {"final": "a"}, want="a", hint_pick="b")])
        self.assertEqual(rows[0]["hint_pick"], "b")

    def test_want_disagreeing_with_judge_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("boolq-1", "b", {"final": "a"}, want="b")])

    def test_wrong_set_stops(self):
        with self.assertRaises(ag.Stop):
            run([cand("x-1", "b", {"final": "b"}, set="blind-g4")])

    def test_override_wins_over_final(self):
        judge = {"label1": "a", "label2": "a", "final": "a", "override": {"from": "a", "to": "c"}}
        rows, _ = run([cand("wiki-9", "b", judge, want="c")])
        self.assertEqual((rows[0]["want"], rows[0]["hint"]), ("c", "c"))


class PendingTest(unittest.TestCase):
    def cands(self):
        return [cand(f"vit-{i}", h, domain="vitaminc", lang="en", set="blind-g4") for i, h in enumerate("babc", 1)]

    def decisions(self):
        return {"vit-1": {"id": "vit-1", "label": "b", "status": "agree", "votes": {"j1": "b", "j2": "b"}},
                "vit-2": {"id": "vit-2", "label": None, "status": "drop", "votes": {}},
                "vit-3": {"id": "vit-3", "label": None, "status": "need-third", "votes": {}}}

    def test_draft_counts_and_drops_pending(self):
        rows, s = run(self.cands(), decisions=self.decisions(), draft=True, set_name="blind-g4")
        self.assertEqual([r["id"] for r in rows], ["vit-1"])
        self.assertEqual((s["pending"], s["dropped"], s["cand_state"]),
                         (2, {"vitaminc": 1}, {"decided": 1, "dropped": 1, "pending": 2}))

    def test_not_draft_stops_on_pending(self):
        with self.assertRaises(ag.Stop):
            run(self.cands(), decisions=self.decisions(), set_name="blind-g4")

    def test_decided_folder_double_id_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("one", "two"):
                os.makedirs(os.path.join(tmp, name))
                with open(os.path.join(tmp, name, "decided.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"q": "q1", "id": "vit-1", "label": "b", "status": "agree"}) + "\n")
            self.assertIn("vit-1", ag.load_decisions([os.path.join(tmp, "one")]))
            with self.assertRaises(ag.Stop):
                ag.load_decisions([os.path.join(tmp, "one"), os.path.join(tmp, "two")])


class DedupeTest(unittest.TestCase):
    def test_same_norm_text_new_rows_keeps_small_id(self):
        a = cand("w-2", "c", {"label": "c"})
        b = dict(cand("w-1", "c", {"label": "c"}), evidence=" " + a["evidence"].upper() + "  ", claim=a["claim"])
        rows, s = run([a, b])
        self.assertEqual(([r["id"] for r in rows], s["deduped"]), (["w-1"], ["w-2"]))

    def test_new_row_same_as_g3_dropped(self):
        old = g3("mem-1", "c")
        new = dict(cand("wiki-1", "c", {"final": "c"}), evidence=old["claim"], claim=old["evidence"])
        rows, s = run([new], [old])
        self.assertEqual(([r["id"] for r in rows], s["deduped"]), (["mem-1"], ["wiki-1"]))

    def test_same_text_other_label_stops(self):
        a = cand("w-1", "c", {"label": "c"})
        b = dict(cand("w-2", "c", {"label": "d"}), evidence=a["evidence"], claim=a["claim"])
        with self.assertRaises(ag.Stop):
            run([a, b])


def blind_rows(n_groups):
    """blind-g4 꼴 — 묶음마다 en b · ko b · a · c 한 줄씩."""
    rows = []
    for g in range(n_groups):
        for k, (want, lang) in enumerate((("b", "en"), ("b", "ko"), ("a", "ko"), ("c", "ko"))):
            rid = f"bl-{g:03d}-{k}"
            rows.append({"id": rid, "evidence": f"근거 {rid}", "claim": f"주장 {rid}", "want": want, "lang": lang,
                         "group": f"bg:{g}:{lang}", "groups": [f"bg:{g}:{lang}"], "src_ids": [rid], "set": "blind-g4"})
    return rows


class CheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        rows, _ = run([cand("dc-1", "c", {"final": "c"})])
        ag.write_jsonl(os.path.join(self.tmp, "dev-cmp-g4.jsonl"), rows)
        ag.write_jsonl(os.path.join(self.tmp, "blind-g4.jsonl"), blind_rows(3))
        for name in ag.G3_SIDE_SETS:
            ag.write_jsonl(os.path.join(self.tmp, f"{name}.jsonl"), [dict(g3(f"{name}-1", "b"), set=name)])
        self.names = os.path.join(self.tmp, "names.txt")
        with open(self.names, "w", encoding="utf-8") as f:
            f.write("")
        self.protocol = os.path.join(self.tmp, "protocol.json")
        self.write_gate({"b@en": 3, "b@ko": 3, "a": 3, "c": 3})

    def tearDown(self):
        self._tmp.cleanup()

    def write_gate(self, min_groups):
        with open(self.protocol, "w", encoding="utf-8") as f:
            json.dump({"group_key": "group", "release_gate": {"set": "blind-g4", "min_groups": min_groups}}, f)

    def check(self, *extra):
        args = ["check", "--dest", self.tmp, "--g3-dir", self.tmp, "--build", os.path.join(self.tmp, "build"),
                "--protocol", self.protocol, "--names", self.names, "--game-names", self.names, "--train"]
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return ag.main(args + list(extra))

    def test_enough_groups_pass(self):
        self.assertEqual(self.check(), 0)

    def test_min_groups_short_exit1(self):
        self.write_gate({"b@en": 4, "b@ko": 3, "a": 3, "c": 3})
        self.assertEqual(self.check(), 1)
        self.assertEqual(self.check("--draft"), 0)          # 초안 검사는 모자람을 보고만

    def test_group_counts_by_letter_and_lang(self):
        rows = blind_rows(2) + [dict(blind_rows(1)[0], id="x", group="bg:0:en")]   # 같은 묶음 한 번 더
        self.assertEqual(ag.group_counts({"b@en": 0, "b": 0, "a": 0, "d": 0}, rows),
                         {"b@en": 2, "b": 4, "a": 2, "d": 0})

    def test_draft_protocol_min_groups_read(self):
        gate, key = ag.read_gate(ag.DRAFT)
        self.assertEqual((gate["set"], key), ("blind-g4", "group"))
        self.assertEqual(gate["min_groups"], {"b@en": 200, "a": 380, "c": 260, "b@ko": 220})

    def test_overlap_with_g3_side_sets_fails(self):
        rows = blind_rows(3)
        side = dict(g3("dev-cal-g3-1", "b"), set="dev-cal-g3")
        rows[0] = dict(rows[0], evidence=side["evidence"], claim=side["claim"])
        ag.write_jsonl(os.path.join(self.tmp, "blind-g4.jsonl"), rows)
        self.assertEqual(self.check(), 1)

    def test_missing_train_stops_unless_draft(self):
        args = ["check", "--dest", self.tmp, "--g3-dir", self.tmp, "--build", os.path.join(self.tmp, "build"),
                "--protocol", self.protocol, "--names", self.names, "--game-names", self.names]
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            self.assertEqual(ag.main(args), 2)
            self.assertEqual(ag.main(args + ["--draft"]), 0)


DRAFT = """{
 "name": "protocol-g4",
 "sets": {
  "dev-cal-g3": "CAL",
  "dev-cmp-g4": "",
  "blind-g4": ""
 },
 "train_sets": {
  "train-g3-ko": "4a10",
  "train-g4-c": "",
  "train-g4-c-h": ""
 },
 "note": "그대로 둔다"
}
"""


class ProtocolTest(unittest.TestCase):
    NAMES = ag.G4_SETS + ("train-g4-c", "train-g4-c-h")      # g4 초안의 빈 자리 (train-g4-d 는 g5 초안부터)

    def test_fill_only_empty(self):
        hashes = {n: str(i) * 64 for i, n in enumerate(self.NAMES)}
        out = ag.fill_protocol(DRAFT, hashes)
        doc = json.loads(out)
        self.assertEqual(doc["sets"]["blind-g4"], hashes["blind-g4"])
        self.assertEqual(doc["train_sets"]["train-g4-c-h"], hashes["train-g4-c-h"])
        self.assertEqual((doc["sets"]["dev-cal-g3"], doc["train_sets"]["train-g3-ko"]), ("CAL", "4a10"))
        self.assertEqual(out.count("\n"), DRAFT.count("\n"))
        self.assertIn('"note": "그대로 둔다"', out)

    def test_missing_slot_stops(self):
        with self.assertRaises(ag.Stop):
            ag.fill_protocol(DRAFT.replace('"blind-g4": ""', '"blind-g4": "x"'), {n: "0" * 64 for n in self.NAMES})
        with self.assertRaises(ag.Stop):
            ag.fill_protocol(DRAFT, {n: "0" * 64 for n in ag.G4_SETS})       # train 빈 자리가 남음

    def test_real_draft_has_slots(self):
        with open(ag.DRAFT, encoding="utf-8", newline="") as f:
            text = f.read()
        out = ag.fill_protocol(text, {n: "0" * 64 for n in self.NAMES})
        self.assertEqual(json.loads(out)["name"], "protocol-g4")

    def test_freeze_then_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            g3_dir = os.path.join(tmp, "g3")
            os.makedirs(g3_dir)
            cal = os.path.join(g3_dir, "dev-cal-g3.jsonl")
            ag.write_jsonl(cal, [g3("c-1", "b")])
            self.assertEqual(ag.freeze_tool("write", "--pairs", cal, "--out",
                                            os.path.join(g3_dir, "FROZEN-dev-cal-g3.sha256"))[0], 0)
            for name in ag.G4_SETS:
                rows, s = run([cand(f"{name}-1", "b", {"final": "b"}, set=name)], set_name=name)
                path = os.path.join(tmp, f"{name}.jsonl")
                ag.write_jsonl(path, rows)
                s["sha256"] = ag.sha256_file(path)
                ag.write_json(os.path.join(tmp, f"{name}.summary.json"), s)
                with mock.patch("sys.stdout"):
                    self.assertEqual(ag.main(["freeze", "--set", name, "--dest", tmp]), 0)
            for name in ag.TRAIN_SETS:
                path = os.path.join(tmp, f"{name}.jsonl")
                ag.write_jsonl(path, [cand(f"{name}-1", "c", want="c", set=name)])
                self.assertEqual(ag.freeze_tool("write", "--pairs", path, "--out",
                                                os.path.join(tmp, f"FROZEN-{name}.sha256"))[0], 0)
            draft = os.path.join(tmp, "protocol-g4.draft.json")
            with open(draft, "w", encoding="utf-8", newline="") as f:
                f.write(DRAFT.replace('"CAL"', f'"{ag.sha256_file(cal)}"'))
            out = os.path.join(tmp, "protocol-g4.json")
            args = ["protocol", "--draft", draft, "--dest", tmp, "--g3-dir", g3_dir, "--out", out]
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(args), 0)
                with open(out, encoding="utf-8") as f:
                    doc = json.load(f)
                self.assertEqual(doc["sets"]["blind-g4"], ag.sha256_file(os.path.join(tmp, "blind-g4.jsonl")))
                self.assertEqual(doc["train_sets"]["train-g4-c"], ag.sha256_file(os.path.join(tmp, "train-g4-c.jsonl")))
                self.assertEqual(ag.main(args), 2)                                 # 덮지 않는다
                os.remove(out)
                with open(draft, "w", encoding="utf-8", newline="") as f:
                    f.write(DRAFT.replace('"CAL"', '"' + "0" * 64 + '"'))
                self.assertEqual(ag.main(args), 2)                                 # dev-cal-g3 해시 어긋남
                self.assertFalse(os.path.exists(out))

    def freeze_all(self, tmp):
        """g3 dev-cal · g4 셋 둘 · 학습 층 셋을 얼린다 → (g3 폴더, dev-cal sha256)."""
        g3_dir = os.path.join(tmp, "g3")
        os.makedirs(g3_dir)
        cal = os.path.join(g3_dir, "dev-cal-g3.jsonl")
        ag.write_jsonl(cal, [g3("c-1", "b")])
        self.assertEqual(ag.freeze_tool("write", "--pairs", cal, "--out",
                                        os.path.join(g3_dir, "FROZEN-dev-cal-g3.sha256"))[0], 0)
        for name in ag.G4_SETS + ag.TRAIN_SETS:
            path = os.path.join(tmp, f"{name}.jsonl")
            ag.write_jsonl(path, [cand(f"{name}-1", "c", want="c", set=name)])
            self.assertEqual(ag.freeze_tool("write", "--pairs", path, "--out",
                                            os.path.join(tmp, f"FROZEN-{name}.sha256"))[0], 0)
        return g3_dir, ag.sha256_file(cal)

    def test_g5_draft_fills_only_train_g4_d(self):
        # K3 설계 5절 ② : g5 초안은 g4 해시가 이미 적혀 있고 train-g4-d 만 빈 자리 — 그 하나만 채우고 나머지는 대조만
        with tempfile.TemporaryDirectory() as tmp:
            g3_dir, cal_sha = self.freeze_all(tmp)
            sha = {n: ag.sha256_file(os.path.join(tmp, f"{n}.jsonl")) for n in ag.G4_SETS + ag.TRAIN_SETS}
            draft_text = (DRAFT.replace('"protocol-g4"', '"protocol-g5"').replace('"CAL"', f'"{cal_sha}"')
                          .replace('"train-g4-c-h": ""', '"train-g4-c-h": "",\n  "train-g4-d": ""'))
            for name in ag.G4_SETS + ("train-g4-c", "train-g4-c-h"):
                draft_text = draft_text.replace(f'"{name}": ""', f'"{name}": "{sha[name]}"')
            draft = os.path.join(tmp, "protocol-g5.draft.json")
            with open(draft, "w", encoding="utf-8", newline="") as f:
                f.write(draft_text)
            out = os.path.join(tmp, "protocol-g5.json")
            args = ["protocol", "--draft", draft, "--dest", tmp, "--g3-dir", g3_dir, "--out", out]
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(args), 0)
            with open(out, encoding="utf-8", newline="") as f:
                text = f.read()
            self.assertEqual(text, draft_text.replace('"train-g4-d": ""', f'"train-g4-d": "{sha["train-g4-d"]}"'))
            os.remove(out)
            with open(draft, "w", encoding="utf-8", newline="") as f:
                f.write(draft_text.replace(sha["train-g4-c"], "0" * 64))
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(args), 2)                                 # 적힌 해시가 얼린 파일과 다름
            self.assertFalse(os.path.exists(out))

    def test_g4_draft_passes_without_frozen_train_g4_d(self):
        # g4 초안에는 train-g4-d 자리가 없다 — FROZEN-train-g4-d 가 없어도 옛 동작 그대로 채운다
        with tempfile.TemporaryDirectory() as tmp:
            g3_dir, cal_sha = self.freeze_all(tmp)
            os.remove(os.path.join(tmp, "FROZEN-train-g4-d.sha256"))
            draft = os.path.join(tmp, "protocol-g4.draft.json")
            with open(draft, "w", encoding="utf-8", newline="") as f:
                f.write(DRAFT.replace('"CAL"', f'"{cal_sha}"'))
            out = os.path.join(tmp, "protocol-g4.json")
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(["protocol", "--draft", draft, "--dest", tmp, "--g3-dir", g3_dir, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                self.assertNotIn("train-g4-d", f.read())

    def test_real_g5_draft_slot_is_train_g4_d_only(self):
        with open(os.path.join(helpers.ROOT, "measure", "protocol-g5.draft.json"), encoding="utf-8") as f:
            doc = json.load(f)
        empty = [k for part in ("sets", "train_sets") for k, v in doc[part].items() if v == ""]
        self.assertEqual(empty, ["train-g4-d"])

    def test_check_default_train_has_h_and_d(self):
        with mock.patch.object(ag, "cmd_check", lambda a: a.train):
            got = ag.main(["check"])
        self.assertEqual(got[:3], ["train-g4-c/cands.jsonl", "train-g4-c/cands-h.jsonl", "train-g4-d/cands.jsonl"])
        self.assertTrue(got[3].endswith("train-g3-ko.jsonl"))

    def test_freeze_refuses_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, s = run([cand("x-1", "b", {"final": "b"})], draft=True)
            path = os.path.join(tmp, "dev-cmp-g4.jsonl")
            ag.write_jsonl(path, rows)
            s["sha256"] = ag.sha256_file(path)
            ag.write_json(os.path.join(tmp, "dev-cmp-g4.summary.json"), s)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(["freeze", "--set", "dev-cmp-g4", "--dest", tmp]), 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "FROZEN-dev-cmp-g4.sha256")))


if __name__ == "__main__":
    unittest.main()
