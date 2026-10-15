"""assemble_g4 — g4 평가 셋 조립 (g3 잇기 · 확정 칸 · 판정 전 줄 · 중복 · min_groups · 얼림 · 규약 해시).

test_assemble_g3 의 시험을 g4 이름으로 옮기고 min_groups · 이은 줄 바이트를 더했다. 글은 모두 지어낸 것이다.
"""
import importlib.util
import io
import json
import os
import subprocess
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


class Out(io.StringIO):
    """화면 받이 — main 이 부르는 reconfigure 를 받아 넘긴다."""

    def reconfigure(self, **kwargs):
        pass


def cell_row(rid, domain, want):
    row = {"id": rid, "evidence": f"근거 {rid}", "claim": f"주장 {rid}", "want": want, "lang": "ko", "group": f"cg:{rid}"}
    if domain is not None:
        row["domain"] = domain
    return row


class CellGapsTest(unittest.TestCase):
    """원천 × 칸 빈칸 검사 (원천 균형 판 설계 4절)."""

    def test_word_and_letter_cells_count_same(self):
        train = [cell_row("t1", "kowiki", "unrelated"), cell_row("t2", "kowiki", "c"), cell_row("t3", "kowiki", "related")]
        dev = [cell_row("d1", "kowiki", "c"), cell_row("d2", "kowiki", "d")]
        got = ag.cell_gaps(train, dev, thin=1)
        self.assertEqual(got["table"]["kowiki"], {"a": [0, 0], "b": [0, 0], "c": [2, 1], "d": [1, 1]})
        self.assertEqual((got["zero"], got["thin_cells"], got["dev_missing"]), ([], [], []))

    def test_no_domain_is_mem_and_cd_not_counted(self):
        train = [cell_row("t1", None, "support"), cell_row("t2", "", "contradict"), cell_row("t3", "mem", "cd"),
                 cell_row("t4", "archive", "cd")]
        dev = [cell_row("d1", "mem", "a"), cell_row("d2", "archive", "c")]
        got = ag.cell_gaps(train, dev)
        self.assertEqual(got["table"]["mem"], {"a": [1, 1], "b": [1, 0], "c": [0, 0], "d": [0, 0]})
        self.assertEqual(got["zero"], [{"domain": "archive", "cell": "c", "train": 0, "dev": 1}])  # cd 줄은 c 로 안 센다
        self.assertEqual(got["thin_cells"], [{"domain": "mem", "cell": "a", "train": 1, "dev": 1}])
        self.assertEqual(got["dev_missing"], [{"domain": "mem", "cell": "b", "train": 1}])

    def test_thin_edge(self):
        train = [cell_row(f"t{i}", "vitaminc", "b") for i in range(10)] + [cell_row(f"u{i}", "vitaminc", "a") for i in range(9)]
        dev = [cell_row("d1", "vitaminc", "b"), cell_row("d2", "vitaminc", "a")]
        got = ag.cell_gaps(train, dev)
        self.assertEqual([(c["cell"], c["train"]) for c in got["thin_cells"]], [("a", 9)])   # 10 은 경고 아님

    def write(self, tmp, name, rows):
        path = os.path.join(tmp, name)
        ag.write_jsonl(path, rows)
        return path

    def cells(self, *argv):
        out = Out()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr"):
            code = ag.main(["cells"] + list(argv))
        return code, json.loads(out.getvalue()) if out.getvalue() else None

    def test_cells_cli_zero_exit1_draft_exit0(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = self.write(tmp, "train-a.jsonl", [cell_row("t1", "kowiki", "c")])
            two = self.write(tmp, "train-b.jsonl", [cell_row("t2", "vitaminc", "a")])
            dev = self.write(tmp, "dev.jsonl", [cell_row("d1", "kowiki", "c"), cell_row("d2", "proposals", "d")])
            code, got = self.cells("--build", tmp, "--recipe", "train-a.jsonl", two, "--dev", dev)
            self.assertEqual(code, 1)
            self.assertEqual(got["cells"]["zero"], [{"domain": "proposals", "cell": "d", "train": 0, "dev": 1}])
            self.assertEqual((got["cells"]["recipe"], got["cells"]["dev"], got["cells"]["thin"]),
                             (["train-a.jsonl", "train-b.jsonl"], "dev.jsonl", 10))
            self.assertEqual(self.cells("--build", tmp, "--recipe", one, two, "--dev", dev, "--draft")[0], 0)

    def test_cells_cli_thin_only_exit0(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = self.write(tmp, "train.jsonl", [cell_row("t1", "kowiki", "c")])
            dev = self.write(tmp, "dev.jsonl", [cell_row("d1", "kowiki", "c")])
            code, got = self.cells("--recipe", one, "--dev", dev)
            self.assertEqual(code, 0)
            self.assertEqual(len(got["cells"]["thin_cells"]), 1)

    def test_cells_cli_missing_file_exit2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = self.write(tmp, "dev.jsonl", [cell_row("d1", "kowiki", "c")])
            self.assertEqual(self.cells("--recipe", os.path.join(tmp, "없음.jsonl"), "--dev", dev)[0], 2)


class CheckRecipeTest(CheckTest):
    """check --recipe — 빈칸이면 exit 1 (--draft 면 적기만), 안 주면 옛 출력 그대로."""

    def check_out(self, *extra, module=ag):
        args = ["check", "--dest", self.tmp, "--g3-dir", self.tmp, "--build", os.path.join(self.tmp, "build"),
                "--protocol", self.protocol, "--names", self.names, "--game-names", self.names, "--train"]
        out, err = Out(), Out()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            code = module.main(args + list(extra))
        self.err = err.getvalue()
        return code, out.getvalue()

    def head_module(self):
        """HEAD 의 assemble_g4.py 를 임시 폴더에 꺼내 읽는다 (git stash 안 씀). git 이 없으면 None."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            src = subprocess.run(["git", "show", "HEAD:build/assemble_g4.py"], cwd=root, capture_output=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        path = os.path.join(self.tmp, "head", "assemble_g4_head.py")
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("assemble_g4_head", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_recipe_zero_fails_unless_draft(self):
        # dev-cmp-g4 에는 kowiki c 한 줄 — 학습 파일이 kowiki c 를 안 가지면 빈칸
        path = os.path.join(self.tmp, "train-x.jsonl")
        ag.write_jsonl(path, [cell_row("t1", "kowiki", "a")])
        code, out = self.check_out("--recipe", path)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["cells"]["zero"], [{"domain": "kowiki", "cell": "c", "train": 0, "dev": 1}])
        self.assertEqual(self.check_out("--recipe", path, "--draft")[0], 0)
        ag.write_jsonl(path, [cell_row("t1", "kowiki", "unrelated")])
        self.assertEqual(self.check_out("--recipe", path)[0], 0)          # 1줄이라 thin 경고만

    def test_no_recipe_output_same_as_before(self):
        head = self.head_module()
        if head is None:
            self.skipTest("git show HEAD 를 못 읽는다")
        path = os.path.join(self.tmp, "train-x.jsonl")
        ag.write_jsonl(path, [cell_row("t1", "kowiki", "c")])
        code_head, before = self.check_out(module=head)            # 고치기 전(HEAD) 코드의 실제 출력
        code_old, old = self.check_out()
        self.assertEqual((code_head, code_old), (0, 0))
        self.assertEqual(old, before)
        self.assertNotIn('"cells"', old)
        code_new, new = self.check_out("--recipe", path)
        self.assertEqual(code_new, 0)
        doc = json.loads(new)
        self.assertEqual((doc.pop("cells")["recipe"], doc.pop("overlap_extra_recipe"), doc.pop("overlap_skipped_base"),
                          doc.pop("train_rows")),
                         (["train-x.jsonl"], ["train-x.jsonl"], [], 1))
        want = json.loads(old)
        want.pop("train_rows")
        self.assertEqual(doc, want)

    def test_recipe_layer_outside_train_counted_in_overlap(self):
        # --train 에 없는 새 층(recipe 만)에 blind-g4 줄과 같은 글이 있으면 겹침 → exit 1
        path = os.path.join(self.tmp, "train-g4-bal.jsonl")
        same = dict(blind_rows(1)[2], id="t-same", group="tg:1", groups=["tg:1"], src_ids=["t-src"], domain="kowiki")
        ag.write_jsonl(path, [same, cell_row("t1", "kowiki", "c")])
        code, out = self.check_out("--recipe", path, "--draft")
        doc = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual((doc["overlap"]["train_text"], doc["overlap"]["train_key"]), (1, 1))
        self.assertIn("경고 : 학습 10줄 미만 칸", self.err)                    # thin 은 stderr 경고
        listed = os.path.join(self.tmp, "build", "train-g4-bal.jsonl")      # --train 에 이미 있으면 두 번 안 센다
        os.makedirs(os.path.dirname(listed))
        ag.write_jsonl(listed, [same])
        code, out = self.check_out("train-g4-bal.jsonl", "--recipe", "train-g4-bal.jsonl", "--draft")
        doc = json.loads(out)
        self.assertEqual((doc["train_rows"], doc["overlap"]["train_text"]), (1, 1))
        self.assertEqual((doc["overlap_extra_recipe"], doc["overlap_skipped_base"]), ([], []))

    def test_base_recipe_layer_not_counted_in_overlap(self):
        # K 처방 기본 층(train-g1-arch 같은 것)은 평가 셋과 겹쳐도 겹침 셈에서 빠진다 → exit 0, 두 칸에 적힌다
        same = dict(blind_rows(1)[2], id="t-same", group="tg:1", groups=["tg:1"], src_ids=["t-src"], domain="kowiki")
        base = os.path.join(self.tmp, "train-g1-arch.jsonl")
        ag.write_jsonl(base, [same, cell_row("t1", "kowiki", "c")])
        code, out = self.check_out("--recipe", base, "--draft")
        doc = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(sum(doc["overlap"].values()), 0)
        self.assertEqual((doc["overlap_extra_recipe"], doc["overlap_skipped_base"]), ([], ["train-g1-arch.jsonl"]))
        newer = os.path.join(self.tmp, "train-g4-bal.jsonl")              # 기본 층이 아닌 층의 겹침은 그대로 센다
        ag.write_jsonl(newer, [same])
        code, out = self.check_out("--recipe", base, newer, "--draft")
        doc = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(doc["overlap"]["train_text"], 1)
        self.assertEqual((doc["overlap_extra_recipe"], doc["overlap_skipped_base"]),
                         (["train-g4-bal.jsonl"], ["train-g1-arch.jsonl"]))

    def test_recipe_missing_file_stops(self):
        self.assertEqual(self.check_out("--recipe", os.path.join(self.tmp, "없음.jsonl"))[0], 2)


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


def rows_of(domain, cells, lang="ko", high=False, tag="r"):
    """칸별 줄 수 {칸: n} → 가짜 층 줄. high 면 근거가 주장 낱말을 다 가져 겹침 1(구간 3), 아니면 겹침 0(구간 0)."""
    out = []
    for cell, n in cells.items():
        for i in range(n):
            rid = f"{tag}-{domain}-{lang}-{cell}-{i}"
            claim = f"주장 {rid}"
            row = {"id": rid, "evidence": claim if high else "다른 글", "claim": claim, "want": cell, "lang": lang,
                   "group": f"g:{rid}"}
            if domain is not None:
                row["domain"] = domain
            out.append(row)
    return out


class CellDriftTest(unittest.TestCase):
    """T1 — 원천 안 칸 몫 이동 (K5 설계 3-2)."""

    def test_ring_vs_base(self):
        base = rows_of("klue", {"a": 4, "b": 6}, tag="k")
        recipe = rows_of("klue", {"a": 7, "b": 3})
        got = ag.cell_drift(recipe, base, [])
        self.assertEqual([(f["cell"], f["dir"], f["diff"], f["ref_kind"]) for f in got["flags"]],
                         [("a", "up", 0.3, "K"), ("b", "down", -0.3, "K")])
        line = next(r for r in got["rows"] if r["cell"] == "b")
        self.assertEqual((line["n"], line["share"], line["ref"]), (3, 0.3, 0.6))

    def test_quiet_and_edge(self):
        base = rows_of("klue", {"a": 4, "b": 6}, tag="k")
        self.assertEqual(ag.cell_drift(rows_of("klue", {"a": 5, "b": 5}), base, [])["flags"], [])   # 차 .10
        self.assertEqual(ag.cell_drift(rows_of("klue", {"a": 6, "b": 4}), base, [])["flags"], [])   # 차 .20 은 안 울림
        five = rows_of("klue", {"a": 3, "b": 2}, tag="f")                  # 4/5 − 3/5 = .20000000000000007 도 안 울림
        self.assertEqual(ag.cell_drift(rows_of("klue", {"a": 4, "b": 1}), five, [])["flags"], [])

    def test_missing_cell_is_zero_share(self):
        # 기준에 원천이 있는데 칸이 없으면 기준 몫 0 — 새 칸 c 는 +.50
        got = ag.cell_drift(rows_of("klue", {"b": 5, "c": 5}), rows_of("klue", {"b": 4}, tag="k"), [])
        self.assertEqual([(f["cell"], f["diff"]) for f in got["flags"]], [("b", -0.5), ("c", 0.5)])

    def test_dev_fallback_and_no_ref(self):
        dev = rows_of("vitc", {"a": 1, "d": 3}, tag="d")
        recipe = rows_of("vitc", {"d": 4}) + rows_of("nowhere", {"a": 2})
        got = ag.cell_drift(recipe, rows_of("klue", {"a": 1}, tag="k"), dev)
        self.assertEqual([(f["domain"], f["cell"], f["diff"], f["ref_kind"]) for f in got["flags"]],
                         [("vitc", "a", -0.25, "dev"), ("vitc", "d", 0.25, "dev")])
        nowhere = [r for r in got["rows"] if r["domain"] == "nowhere"]
        self.assertTrue(all(r["ref"] is None and r["ref_kind"] is None for r in nowhere))

    def test_prev_diff_report_only(self):
        base = rows_of("klue", {"a": 4, "b": 6}, tag="k")
        got = ag.cell_drift(rows_of("klue", {"a": 5, "b": 5}), base, [], prev_rows=rows_of("klue", {"b": 10}, tag="p"))
        line = next(r for r in got["rows"] if r["cell"] == "b")
        self.assertEqual((line["prev"], line["prev_diff"]), (1.0, -0.5))
        self.assertEqual(got["flags"], [])                               # 앞 판 차는 안 울린다


class BandDriftTest(unittest.TestCase):
    """T2 — 언어 × 겹침 구간 칸 몫을 dev 와 견줌 (K5 설계 3-3)."""

    dev = rows_of("klue", {"c": 5, "d": 5}, tag="d")                   # ko 구간 0 · c .5 · d .5

    def test_ring_new_warning(self):
        got = ag.band_drift(rows_of("klue", {"c": 100}), self.dev, [])
        self.assertEqual([(f["lang"], f["band"], f["cell"], f["diff"]) for f in got["flags"]],
                         [("ko", 0, "c", 0.5), ("ko", 0, "d", -0.5)])
        self.assertEqual(got["baseline"], [])

    def test_floor_quiet(self):
        self.assertEqual(ag.band_drift(rows_of("klue", {"c": 99}), self.dev, [])["flags"], [])     # 구간 100줄 바닥
        self.assertEqual(ag.band_drift(rows_of("klue", {"c": 60, "d": 40}), self.dev, [])["flags"], [])

    def test_baseline_report_only(self):
        recipe = rows_of("klue", {"c": 100})
        got = ag.band_drift(recipe, self.dev, rows_of("klue", {"c": 120}, tag="k"))
        self.assertEqual(got["flags"], [])
        self.assertEqual(len(got["baseline"]), 2)
        flipped = ag.band_drift(recipe, self.dev, rows_of("klue", {"d": 120}, tag="k"))   # 방향이 다르면 새 경고
        self.assertEqual(len(flipped["flags"]), 2)

    def test_band_and_lang_split(self):
        recipe = rows_of("klue", {"c": 100}, high=True) + rows_of("w", {"a": 100}, lang="en")
        got = ag.band_drift(recipe, self.dev + rows_of("w", {"a": 3}, lang="en", tag="d"), [])
        self.assertEqual(got["flags"], [])                               # ko 구간 3 은 dev 에 없고 en 구간 0 은 같은 몫
        self.assertEqual({(r["lang"], r["band"]) for r in got["rows"]}, {("en", 0), ("ko", 3)})
        gap = next(r for r in got["rows"] if r["lang"] == "ko")
        self.assertEqual((gap["note"], gap["flag"], gap["dev_band"]), ("dev 없음", False, 0))   # 왜 안 울렸는지 남긴다


class WeakCellsTest(unittest.TestCase):
    """T3 — dev 30쌍 미만 칸 「판정 약함」 (K5 설계 3-4)."""

    def test_weak_edge_and_guard_share(self):
        dev = rows_of("klue", {"b": 29, "c": 40}) + rows_of("wiki", {"b": 30, "d": 1}) + rows_of("vit", {"b": 20}, lang="en")
        got = ag.weak_cells(dev, [("b", "ko"), ("b", "en")])
        self.assertEqual(got["cells"], [{"domain": "klue", "cell": "b", "dev": 29}, {"domain": "vit", "cell": "b", "dev": 20},
                                        {"domain": "wiki", "cell": "d", "dev": 1}])
        ko, en = got["guards"]
        self.assertEqual((ko["guard"], ko["pairs"], ko["weak_pairs"], ko["weak_share"], ko["pp_per_pair"]),
                         ("b@ko", 59, 29, round(29 / 59, 4), round(100 / 59, 3)))
        self.assertEqual((en["pairs"], en["weak_pairs"]), (20, 20))

    def test_protocol_guards_read(self):
        self.assertEqual(ag.protocol_guards(os.path.join(helpers.ROOT, "measure", "protocol-g6.json")),
                         [("b", "en"), ("b", "ko")])


class CellCheckCliTest(unittest.TestCase):
    """cells · check --base-recipe — exit 0 · 1(미허용 T1 · 새 T2) · 2(파일 없음 · 금지 칸 허용) · --draft 는 0."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.base = self.write("base.jsonl", rows_of("klue", {"a": 4, "b": 6}, tag="k"))
        self.dev = self.write("dev.jsonl", rows_of("klue", {"a": 4, "b": 6}, tag="d"))
        self.down = self.write("down.jsonl", rows_of("klue", {"a": 7, "b": 3}))       # b −.30 · a +.30
        self.ack = os.path.join(self.tmp, "ack.json")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, rows):
        path = os.path.join(self.tmp, name)
        ag.write_jsonl(path, rows)
        return path

    def write_ack(self, entries):
        with open(self.ack, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)

    def cells(self, recipe, *extra):
        out, err = Out(), Out()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            code = ag.main(["cells", "--recipe", recipe, "--dev", self.dev, "--base-recipe", self.base] + list(extra))
        self.out, self.err = out.getvalue(), err.getvalue()
        return code

    def test_quiet_exit0_table_out(self):
        self.assertEqual(self.cells(self.base), 0)
        self.assertIn("## T1 원천 안 칸 몫 이동", self.out)
        self.assertIn("## T3 작은 지킴 칸", self.out)
        self.assertNotIn("경고 T1", self.err)

    def test_ring_exit1_draft_exit0_json(self):
        self.assertEqual(self.cells(self.down), 1)
        self.assertIn("경고 T1 : klue b -0.300", self.err)
        self.assertEqual(self.cells(self.down, "--draft", "--json"), 0)
        doc = json.loads(self.out)
        self.assertEqual([(f["cell"], f["dir"]) for f in doc["drift"]["unacked"]], [("a", "up"), ("b", "down")])
        self.assertEqual(set(doc), {"cells", "cell_check", "drift", "bands", "weak"})
        self.assertEqual(doc["cell_check"]["base_recipe"], ["base.jsonl"])
        self.assertNotIn("주장", self.out)                                # 쌍 본문은 안 찍는다

    def test_ack_allows_but_b_down_still_blocks(self):
        self.write_ack([{"tool": "T1", "domain": "klue", "cell": "a", "dir": "up", "why": "시험"}])
        self.assertEqual(self.cells(self.down, "--ack", self.ack, "--json"), 1)          # b 내림은 남는다
        doc = json.loads(self.out)
        self.assertEqual([f["cell"] for f in doc["drift"]["acked"]], ["a"])
        self.assertEqual([f["cell"] for f in doc["drift"]["unacked"]], ["b"])
        self.assertEqual(doc["cell_check"]["ack"]["sha256"], ag.sha256_file(self.ack))
        up = self.write("up.jsonl", rows_of("klue", {"a": 1, "b": 9}))                 # b +.30 · a −.30
        self.write_ack([{"tool": "T1", "domain": "klue", "cell": "a", "dir": "down", "why": "시험"},
                        {"tool": "T1", "domain": "klue", "cell": "b", "dir": "up", "why": "시험"}])
        self.assertEqual(self.cells(up, "--ack", self.ack), 0)
        self.assertIn("허용 T1 : klue b +0.300", self.err)
        self.write_ack([{"tool": "T1", "domain": "klue", "cell": "a", "dir": "up", "why": "시험"}])
        self.assertEqual(self.cells(up, "--ack", self.ack), 1)                         # 방향이 다르면 허용 아님

    def test_forbidden_ack_exit2_even_draft(self):
        self.write_ack([{"tool": "T1", "domain": "klue", "cell": "b", "dir": "down", "why": "시험"}])
        self.assertEqual(self.cells(self.down, "--ack", self.ack, "--draft"), 2)
        self.assertIn("b 칸 내림은 허용할 수 없다", self.err)

    def test_bad_ack_and_missing_files_exit2(self):
        for entries in ([{"tool": "T1", "domain": "klue", "cell": "a", "dir": "up"}],                  # why 없음
                        [{"tool": "T1", "domain": "klue", "cell": "a", "dir": "up", "why": "x", "evil": 1}],
                        [{"tool": "T9", "cell": "a", "dir": "up", "why": "x"}],
                        [{"tool": "T1", "domain": "klue", "cell": "a", "dir": "sideways", "why": "x"}],
                        [{"tool": "T1", "domain": "klue", "cell": "B", "dir": "down", "why": "x"}],   # 큰 글자로 금지 비켜 가기
                        [{"tool": "T1", "domain": "klue", "cell": "contradict", "dir": "up", "why": "x"}],
                        [{"tool": ["T1"], "domain": "klue", "cell": "a", "dir": "up", "why": "x"}],
                        [{"tool": "T2", "lang": "jp", "band": 0, "cell": "a", "dir": "up", "why": "x"}],
                        [{"tool": "T2", "lang": "ko", "band": "0", "cell": "a", "dir": "up", "why": "x"}],
                        [{"tool": "T2", "lang": "ko", "band": 7, "cell": "a", "dir": "up", "why": "x"}],
                        {"tool": "T1"}):
            self.write_ack(entries)
            self.assertEqual(self.cells(self.down, "--ack", self.ack, "--draft"), 2, entries)
        self.assertEqual(self.cells(self.down, "--ack", os.path.join(self.tmp, "없음.json"), "--draft"), 2)
        with open(self.ack, "w", encoding="utf-8") as f:
            f.write("[{깨진")
        self.assertEqual(self.cells(self.down, "--ack", self.ack, "--draft"), 2)            # 깨진 JSON 도 exit 2
        self.assertEqual(self.cells(self.down, "--protocol", os.path.join(self.tmp, "없음.json"), "--draft"), 2)
        self.assertEqual(self.cells(self.down, "--prev-recipe", os.path.join(self.tmp, "없음.jsonl"), "--draft"), 2)
        out = Out()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr"):
            self.assertEqual(ag.main(["cells", "--recipe", self.down, "--dev", self.dev, "--base-recipe",
                                      os.path.join(self.tmp, "없음.jsonl")]), 2)

    def test_t2_new_warning_exit1_baseline_exit0_ack(self):
        dev = self.write("dev2.jsonl", rows_of("klue", {"c": 5, "d": 5}, tag="d"))
        self.dev = dev
        # d 한 줄은 구간 3(dev 에 없는 구간)에 둬 빈칸(zero)만 막는다
        recipe = self.write("c100.jsonl", rows_of("klue", {"c": 100}) + rows_of("klue", {"d": 1}, high=True, tag="h"))
        self.base = self.write("base2.jsonl", rows_of("klue", {"c": 5, "d": 5}, tag="k"))
        self.assertEqual(self.cells(recipe, "--draft"), 0)
        self.assertEqual(self.cells(recipe), 1)
        self.assertIn("경고 T2 : ko 구간 0 c +0.500", self.err)
        self.write_ack([{"tool": "T2", "lang": "ko", "band": 0, "cell": "c", "dir": "up", "why": "시험"},
                        {"tool": "T2", "lang": "ko", "band": 0, "cell": "d", "dir": "down", "why": "시험"},
                        {"tool": "T1", "domain": "klue", "cell": "c", "dir": "up", "why": "시험"},
                        {"tool": "T1", "domain": "klue", "cell": "d", "dir": "down", "why": "시험"}])
        self.assertEqual(self.cells(recipe, "--ack", self.ack), 0)
        self.base = self.write("base3.jsonl", rows_of("klue", {"c": 100}, tag="k"))   # 기준에서도 울림 → 기준선부터
        self.assertEqual(self.cells(recipe), 0)
        self.assertIn("기준선부터 T2 : ko 구간 0 c +0.500", self.err)

    def test_check_args_without_base_exit2(self):
        out = Out()
        for extra in (["--ack", self.ack], ["--json"], ["--prev-recipe", self.base]):
            with mock.patch("sys.stdout", out), mock.patch("sys.stderr"):
                self.assertEqual(ag.main(["cells", "--recipe", self.base, "--dev", self.dev] + extra), 2, extra)

    def test_prev_recipe_big_shift_shown_in_table(self):
        prev = self.write("prev.jsonl", rows_of("klue", {"b": 10}, tag="p"))
        self.assertEqual(self.cells(self.base, "--prev-recipe", prev), 0)                  # 앞 판 차는 안 막는다
        self.assertIn("| klue | b | 0.600 | K | 0.600 | +0.000 | -0.400 | 앞 판 차 큼 (보고) |", self.out)
        self.assertEqual(self.cells(self.base, "--prev-recipe", prev, "--json"), 0)
        doc = json.loads(self.out)
        self.assertEqual(doc["cell_check"]["prev_recipe"], ["prev.jsonl"])
        self.assertEqual(doc["cell_check"]["guards"], ["b@ko", "b@en"])

    def test_weak_never_blocks(self):
        self.assertEqual(self.cells(self.base), 0)                        # dev 칸이 다 30 미만 — 표시만
        self.assertIn("약함 T3 : b@ko 6쌍 중 6", self.err)

    def test_protocol_not_dict_exit2(self):
        for doc in ([1, 2], {"pair_gate": [1]}, {"pair_gate": {"guards": "b"}}, {"pair_gate": {"guards": [3]}}):
            path = os.path.join(self.tmp, "proto.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            self.assertEqual(self.cells(self.base, "--protocol", path, "--draft"), 2, doc)
            self.assertIn("규약 꼴이 틀렸다", self.err)

    def test_no_ref_domain_warned(self):
        recipe = self.write("nowhere.jsonl", rows_of("klue", {"a": 4, "b": 6}) + rows_of("nowhere", {"a": 2}, tag="n"))
        self.assertEqual(self.cells(recipe, "--draft"), 0)
        self.assertIn("기준 없음 T1 : nowhere", self.err)
        self.assertNotIn("기준 없음 T1 : klue", self.err)

    def test_unused_ack_reported_not_blocking(self):
        self.write_ack([{"tool": "T1", "domain": "klue", "cell": "a", "dir": "up", "why": "시험"},
                        {"tool": "T1", "domain": "wiki", "cell": "c", "dir": "up", "why": "낡은 허용"},
                        {"tool": "T2", "lang": "en", "band": 1, "cell": "d", "dir": "down", "why": "낡은 허용"}])
        self.assertEqual(self.cells(self.down, "--ack", self.ack, "--json"), 1)         # 막는 건 b 내림뿐
        doc = json.loads(self.out)
        self.assertEqual(doc["cell_check"]["unused_ack"],
                         [{"tool": "T1", "domain": "wiki", "cell": "c", "dir": "up"},
                          {"tool": "T2", "lang": "en", "band": 1, "cell": "d", "dir": "down"}])
        self.assertIn("안 쓴 허용 : T1 wiki c up", self.err)
        self.assertEqual(self.cells(self.base, "--ack", self.ack), 0)                  # 안 쓴 허용은 안 막는다
        self.assertIn("| T1 | klue a up |", self.out)                                  # 표에도 적는다

    def test_dropped_lang_counted(self):
        recipe = self.write("mix.jsonl", rows_of("klue", {"a": 4, "b": 6}) + rows_of("mixed", {"a": 3}, lang="mix", tag="m"))
        self.assertEqual(self.cells(recipe, "--draft", "--json"), 0)
        self.assertEqual(json.loads(self.out)["bands"]["dropped_lang"], {"train": 3, "dev": 0})

    def test_thresholds_must_be_positive(self):
        for extra in (["--max-shift", "0"], ["--max-shift", "-0.1"], ["--band-min", "0"], ["--weak-min", "-1"]):
            self.assertEqual(self.cells(self.base, "--draft", *extra), 2, extra)
            self.assertIn("양수여야 한다", self.err)


class CheckBaseRecipeTest(CheckTest):
    """check --recipe --base-recipe — 점검표가 겹침 · 빈칸 검사와 같이 돈다."""

    check_out = CheckRecipeTest.check_out

    def test_check_runs_t1_t3_and_blocks(self):
        base = os.path.join(self.tmp, "k.jsonl")
        ag.write_jsonl(base, rows_of("kowiki", {"c": 6, "a": 4}, tag="k"))
        recipe = os.path.join(self.tmp, "new.jsonl")
        ag.write_jsonl(recipe, rows_of("kowiki", {"c": 3, "a": 7}))
        code, out = self.check_out("--recipe", recipe, "--base-recipe", base, "--json")
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual([f["cell"] for f in doc["drift"]["unacked"]], ["a", "c"])
        self.assertEqual(doc["cell_check"]["max_shift"], 0.2)
        self.assertEqual(self.check_out("--recipe", recipe, "--base-recipe", base, "--draft")[0], 0)
        self.assertEqual(self.check_out("--recipe", base, "--base-recipe", base)[0], 0)
        self.assertEqual(self.check_out("--base-recipe", base)[0], 2)                  # --recipe 없이 못 준다
        code, out = self.check_out("--recipe", base, "--base-recipe", base)
        self.assertIn("## T1 원천 안 칸 몫 이동", out)                                   # check 도 기본은 표
        self.assertIn("- 겹침 0 · scrub 0 · min_groups 모자람 0", out)
        self.assertEqual(self.check_out("--recipe", base, "--ack", base)[0], 2)          # --base-recipe 없이 --ack


if __name__ == "__main__":
    unittest.main()
