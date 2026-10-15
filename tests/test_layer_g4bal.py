"""layer_g4bal — K4 원천 균형 층 (원천 균형 판 설계 1-1 · 1-2 · 2절 · 3-1 · 5-5 ②). 글은 모두 지어낸 것이다."""
import collections
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import layer_g4bal as lb  # noqa: E402


def row(rid, domain, cell, group=None, **extra):
    out = {"id": rid, "evidence": f"근거 {rid} 문장이다. 둘째 {rid} 문장이다.", "claim": f"주장 {rid} 이다.",
           "want": cell, "hint": cell, "domain": domain, "group": group or f"{domain}:G-{rid}", "lang": "ko",
           "src_ids": [f"src-{rid}"]}
    out["groups"] = [out["group"]]
    out.update(extra)
    return out


def many(prefix, domain, cell, n, **extra):
    return [row(f"{prefix}{i:03d}", domain, cell, **extra) for i in range(n)]


class NoGuard:
    def reasons(self, row):
        return []


class WordGuard:
    """주장에 「금지」 가 든 줄을 막는다."""

    def reasons(self, row):
        return ["eval_text"] if "금지" in row["claim"] else []


SMALL = {"build": {"vitaminc": {"a": 3, "b": 5, "c": 4, "d": 4}, "proposals": {"c": 6, "d": 3},
                   "kowiki": {"c": 2, "d": 2}, "archive": {"a": 2, "b": 2, "d": 2}}}


def build(stages, targets=SMALL, guard=None, seed=7, taken=frozenset()):
    return lb.build(stages, guard or NoGuard(), targets, seed, taken)


class FilterTest(unittest.TestCase):
    def test_synth_and_jira_minicheck_never_used(self):
        synth = many("s", "proposals", "c", 5, origin="synth", form="synth", synth_from="x", set="train-g1-syn")
        jira = many("j", "jira", "c", 5) + many("m", "minicheck", "a", 5)
        real = many("p", "proposals", "c", 10)
        rows, summary = build([("v5", synth + jira + real)])
        self.assertEqual(len(rows), 6)
        self.assertFalse(any(lb.is_synth(r) or r["domain"] in ("jira", "minicheck") for r in rows))
        self.assertEqual((summary["dropped"]["synth"], summary["dropped"]["skip_domain"]), (5, 10))

    def test_targets_never_exceeded_and_two_per_group(self):
        rows = many("c", "proposals", "c", 20, group="proposals:G1") + many("d", "proposals", "c", 20)
        got, summary = build([("v6", rows)])
        per_cell = collections.Counter((r["domain"], r["want"]) for r in got)
        for (domain, cell), n in per_cell.items():
            self.assertLessEqual(n, SMALL["build"][domain][cell])
        self.assertLessEqual(sum(1 for r in got if r["group"] == "proposals:G1"), 2)
        self.assertEqual(summary["short"].get("proposals · d"), 3)

    def test_forbidden_rows_never_in(self):
        rows = [row(f"x{i}", "proposals", "c", claim=f"금지 주장 {i} 이다.") for i in range(10)] + many("p", "proposals", "c", 3)
        got, summary = build([("v6", rows)], guard=WordGuard())
        self.assertEqual(len(got), 3)
        self.assertEqual(lb.recheck(got, WordGuard()), {})
        self.assertEqual(summary["dropped"]["guard:eval_text"], 10)

    def test_k_pairs_not_added(self):
        rows = many("p", "proposals", "c", 4)
        got, summary = build([("v6", rows)], taken={lb.sha_key(rows[0])})
        self.assertNotIn(rows[0]["id"], {r["id"] for r in got})
        self.assertEqual(summary["dropped"]["in_k"], 1)

    def test_stage_order_v6_before_v5(self):
        v6 = many("six", "proposals", "c", 4)
        v5 = many("five", "proposals", "c", 10)
        got, summary = build([("v6", v6), ("v5", v5)])
        self.assertEqual(summary["by_stage"]["v6"], {"proposals · c": 4})
        self.assertEqual(summary["by_stage"]["v5"], {"proposals · c": 2})


class PairTest(unittest.TestCase):
    def contrast(self, n, broken=0):
        rows = []
        for i in range(n):
            group = f"vitaminc:P{i}"
            rows.append(row(f"a{i}", "vitaminc", "a", group=group, form="contrast", pair=f"k{i}"))
            rows.append(row(f"b{i}", "vitaminc", "b", group=group, form="contrast", pair=f"k{i}"))
        for i in range(broken):
            rows.append(row(f"lone{i}", "vitaminc", "b", form="contrast", pair=f"lone{i}"))
        return rows

    def test_whole_pairs_then_partial_b(self):
        got, summary = build([("v6", self.contrast(6, broken=2))])
        cells = collections.Counter(r["want"] for r in got)
        self.assertEqual((cells["a"], cells["b"]), (3, 5))
        self.assertEqual(summary["units"], {"taken": 3, "broken": 2})
        self.assertFalse(any(r["id"].startswith("lone") for r in got))          # 깨진 짝은 안 든다
        whole = [r for r in got if r["bal_how"] == "unit"]
        pairs = collections.Counter(r["pair"] for r in whole)
        self.assertTrue(all(n == 2 for n in pairs.values()))                    # 짝째
        self.assertEqual(sum(1 for r in got if r["bal_how"] == "partial"), 2)
        a_pairs = {r["pair"] for r in got if r["want"] == "a"}
        self.assertTrue(a_pairs <= {r["pair"] for r in got if r["want"] == "b"})  # a 는 늘 b 짝과 같이

    def test_seed_three_rows_together(self):
        rows = []
        for i in range(4):
            for cell in ("b", "a", "d"):
                rows.append(row(f"seed-ka{i:04d}-{cell}", "archive", cell, group=f"doc:studio:A{i}.md",
                                form=f"seed-{cell}", pair=f"ka{i:04d}"))
        got, summary = build([("new", rows)])
        self.assertEqual(summary["units"]["taken"], 2)
        by_pair = collections.Counter(r["pair"] for r in got)
        self.assertEqual(sorted(by_pair.values()), [3, 3])

    def test_unit_with_forbidden_row_dropped_whole(self):
        rows = self.contrast(2)
        rows[0]["claim"] = "금지 주장 이다."
        got, summary = build([("v6", rows)], guard=WordGuard())
        self.assertEqual({r["pair"] for r in got}, {"k1"})
        self.assertEqual(summary["dropped"]["unit:guard:eval_text"], 1)


class DFollowsCTest(unittest.TestCase):
    def test_d_prefers_group_of_chosen_c(self):
        c_rows = many("c", "kowiki", "c", 2, group="kowiki:ANCHOR")
        d_rows = many("d", "kowiki", "d", 30) + many("e", "kowiki", "d", 2, group="kowiki:ANCHOR")
        got, _ = build([("v6", c_rows + d_rows)])
        d_groups = [r["group"] for r in got if r["want"] == "d"]
        self.assertEqual(d_groups, ["kowiki:ANCHOR", "kowiki:ANCHOR"])


class SameSeedTest(unittest.TestCase):
    def test_same_seed_same_sha(self):
        stages = [("v6", many("p", "proposals", "c", 30) + many("q", "proposals", "d", 30)),
                  ("v5", many("v", "vitaminc", "c", 30))]
        one, _ = build(stages, seed=3)
        two, _ = build(stages, seed=3)
        self.assertEqual(json.dumps(one, ensure_ascii=False), json.dumps(two, ensure_ascii=False))
        other, _ = build(stages, seed=4)
        self.assertNotEqual({r["id"] for r in one}, {r["id"] for r in other})


class JudgedTest(unittest.TestCase):
    def test_moved_to_decided_cell_and_boundary_dropped(self):
        cands = [dict(row(f"n{i}", "mit-cs", "b"), want="") for i in range(4)]
        decisions = {"n0": {"id": "n0", "label": "b", "status": "agree"}, "n1": {"id": "n1", "label": "a", "status": "agree"},
                     "n2": {"id": "n2", "label": "cd", "status": "agree"}, "n3": {"id": "n3", "label": "c", "status": "drop"}}
        rows, state = lb.judged_rows(cands, decisions)
        self.assertEqual([(r["id"], r["want"], r.get("moved_from")) for r in rows], [("n0", "b", None), ("n1", "a", "b")])
        self.assertEqual((state["moved"], state["boundary"], state["dropped"]), (1, 1, 2))

    def test_pending_stops_unless_draft(self):
        cands = [dict(row("n0", "mit-cs", "b"), want="")]
        with self.assertRaises(lb.Stop):
            lb.judged_rows(cands, {})
        rows, state = lb.judged_rows(cands, {}, draft=True)
        self.assertEqual((rows, state["pending"]), ([], 1))

    def test_v5_rejudge_moves_drops_and_blocks(self):
        rows = many("p", "proposals", "c", 4) + many("v", "vitaminc", "a", 2) + many("s", "proposals", "c", 1, origin="synth")
        decisions = {"p000": {"id": "p000", "label": "d", "status": "agree"},
                     "p001": {"id": "p001", "label": "c", "status": "need-third"}}
        out, count = lb.v5_rows(rows, decisions)
        got = {r["id"]: r["want"] for r in out}
        self.assertEqual(got, {"p000": "d", "p002": "c", "p003": "c"})       # vitaminc a 는 v5 칸 밖 · 합성은 뺌
        self.assertEqual((count["moved"], count["rejudged_dropped"], count["not_v5_cell"]), (1, 1, 2))
        out, count = lb.v5_rows(rows, decisions, block={("proposals", "c")})
        self.assertEqual({r["id"] for r in out}, {"p000"})
        self.assertEqual(count["blocked"], 2)


class FakeForbid:
    """G2Forbid · G4Forbid 꼴 가짜 guard — text 에 든 글이면 old_text_old, train_groups 에 걸리면 old_group_g1.
    실물처럼 train_groups 에 학습 셋 묶음과 평가 셋 묶음이 섞여 있다. extra 는 id 별 다른 까닭."""

    def __init__(self, text=(), train_groups=(), extra=None):
        self.text, self.train_groups, self.extra = set(text), set(train_groups), extra or {}

    def reasons(self, row):
        out = []
        if lb.pair_key(row) in self.text:
            out.append("old_text_old")
        if set(row.get("groups") or [row["group"]]) & self.train_groups:
            out.append("old_group_g1")
        return out + list(self.extra.get(row["id"], []))


class ReuseGuardTest(unittest.TestCase):
    def setUp(self):
        self.mine = row("r1", "vitaminc", "a", group="g2b:T1")                  # 재료 · 묶음은 학습 셋에만
        self.on_g2 = row("r2", "vitaminc", "a", group="g2dev:E1")               # 재료 · g2 평가 묶음
        self.on_g1 = row("r3", "vitaminc", "a", group="g1blind:E2")             # 재료 · g1 평가 묶음
        self.eval_text = row("r4", "vitaminc", "a", group="g2b:T2")             # 재료 · 글이 g2 평가 글
        self.para = row("r5", "vitaminc", "a", group="g2b:T3")                  # 재료 · eval_para · eval_group
        self.other = row("r6", "vitaminc", "a", group="g2b:T1")                 # 재료 아닌 줄
        reuse = [self.mine, self.on_g2, self.on_g1, self.eval_text, self.para]
        every = reuse + [self.other]
        base = FakeForbid(text={lb.pair_key(r) for r in every},
                          train_groups={"g2b:T1", "g2b:T2", "g2b:T3", "g2dev:E1", "g1blind:E2"},
                          extra={"r5": ["eval_para", "eval_group"]})
        self.guard = lb.ReuseGuard(base, reuse, {lb.pair_key(self.eval_text)}, {"g2dev:E1", "g1blind:E2"})

    def test_1_reuse_row_on_train_group_passes(self):
        self.assertEqual(self.guard.reasons(self.mine), [])

    def test_2_g2_eval_group_stays(self):
        self.assertEqual(self.guard.reasons(self.on_g2), ["old_group_g1"])

    def test_3_g1_eval_group_stays(self):
        self.assertEqual(self.guard.reasons(self.on_g1), ["old_group_g1"])

    def test_4_not_reuse_forgives_nothing(self):
        self.assertEqual(self.guard.reasons(self.other), ["old_text_old", "old_group_g1"])

    def test_5_eval_text_keeps_old_text_old(self):
        self.assertEqual(self.guard.reasons(self.eval_text), ["old_text_old"])

    def test_6_eval_para_and_group_always_stay(self):
        self.assertEqual(self.guard.reasons(self.para), ["eval_para", "eval_group"])

    def test_7_old_eval_marks_reads_g2_eval_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            def put(rel, rows):
                os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
                lb.write_jsonl(os.path.join(tmp, rel), rows)
            put("sets/g1/blind-g1.jsonl", [row("e1", "vitaminc", "a", group="g1:E")])
            for k, name in enumerate(lb.G2_EVAL_FILES):
                put(f"sets/g2/{name}.jsonl", [row(f"f{k}", "vitaminc", "a", group=f"g2:E{k}")])
            put("sets/g2/train-g2-b.jsonl", [row("t1", "vitaminc", "a", group="g2b:T")])     # 학습 셋은 안 읽는다
            keys, groups, missing = lb.old_eval_marks(tmp)
        self.assertEqual(groups, {"g1:E", "g2:E0", "g2:E1", "g2:E2"})
        self.assertIn(lb.pair_key(row("f2", "vitaminc", "a")), keys)
        self.assertNotIn(lb.pair_key(row("t1", "vitaminc", "a")), keys)
        self.assertNotIn("blind-g2", missing)
        self.assertIn("dev-cal", missing)

    def test_g4forbid_like_guard_train_b_passes_eval_group_blocked(self):
        train_b = [row(f"b{i}", "vitaminc", "a", group=f"g2b:T{i}") for i in range(3)]
        on_eval = row("b9", "vitaminc", "a", group="g2dev:E1")
        base = FakeForbid(text={lb.pair_key(r) for r in train_b + [on_eval]},
                          train_groups={r["group"] for r in train_b} | {"g2dev:E1"})
        guard = lb.ReuseGuard(base, train_b + [on_eval], set(), {"g2dev:E1"})
        self.assertEqual([guard.reasons(r) for r in train_b], [[], [], []])
        self.assertEqual(guard.reasons(on_eval), ["old_group_g1"])


class SampleTest(unittest.TestCase):
    def v5(self):
        return (many("pc", "proposals", "c", 60) + many("pb", "proposals", "b", 5) + many("ka", "klue-mrc", "a", 40)
                + many("syn", "proposals", "c", 30, origin="synth", form="synth")
                + many("jr", "jira", "c", 20) + many("va", "vitaminc", "a", 20))

    def test_strata_and_hidden_label(self):
        rows, summary = lb.sample_v5(self.v5(), seed=5)
        cells = collections.Counter((r["domain"], r["v5_want"]) for r in rows)
        self.assertEqual(dict(cells), {("proposals", "c"): 45, ("proposals", "b"): 5, ("klue-mrc", "a"): 25})
        self.assertEqual(summary["population"]["proposals · c"], 60)
        self.assertFalse(any(r["id"].startswith(("syn", "jr", "va")) for r in rows))
        self.assertTrue(all(r["want"] == "" and "votes" not in r for r in rows))
        again, _ = lb.sample_v5(self.v5(), seed=5)
        self.assertEqual(json.dumps(rows, ensure_ascii=False), json.dumps(again, ensure_ascii=False))

    def test_rate_rule(self):
        rows, summary = lb.sample_v5(self.v5(), seed=5)
        pc = [r for r in rows if r["v5_want"] == "c"]
        decisions = {r["id"]: {"id": r["id"], "label": r["v5_want"], "status": "agree"} for r in rows}
        for r in pc[:9]:                                                      # proposals c 9/45 = .20 바뀜
            decisions[r["id"]] = {"id": r["id"], "label": "d", "status": "agree"}
        report = lb.v5_rate(rows, decisions, summary["population"])
        cell = report["cells"]["proposals · c"]
        self.assertEqual((cell["n"], cell["changed"], cell["r_h"]), (45, 9, 0.2))
        self.assertGreater(cell["wilson_upper"], 0.2)
        self.assertEqual(report["over"], ["proposals · c"])
        self.assertEqual(report["R"], round(60 / 105 * 0.2, 4))
        self.assertEqual(report["verdict"], "넘은 칸 종류만 모집단 전부 재판정")
        decisions[pc[9]["id"]] = {"id": pc[9]["id"], "status": "missing"}
        self.assertEqual(lb.v5_rate(rows, decisions, summary["population"])["verdict"], "판정 전 줄이 남았다 — 결론 없음")
        clean = {r["id"]: {"id": r["id"], "label": r["v5_want"], "status": "agree"} for r in rows}
        self.assertEqual(lb.v5_rate(rows, clean, summary["population"])["verdict"], "v5 줄을 그대로 쓴다 · 표본은 v6 칸으로 갈아 끼운다")


class PickTest(unittest.TestCase):
    def test_boolq_removes_most_overlapping_sentence(self):
        base = row("q1", "kobest-boolq", "a")
        base.update(evidence="고양이는 낮에 잔다. 개는 밤에 짖는다. 새는 아침에 운다.", claim="개는 밤에 짖는다")
        out, _ = lb.boolq_removed([base])
        self.assertEqual(out[0]["evidence"], "고양이는 낮에 잔다. 새는 아침에 운다.")
        self.assertEqual((out[0]["hint"], out[0]["domain"], out[0]["anchor"]), ("d", "kobest-boolq", "q1"))

    def src(self):
        g3ko = []
        for i in range(12):
            r = row(f"bq{i}", "kobest-boolq", "a" if i % 2 else "b")
            r.update(evidence=f"첫 문장 {i} 이다. 둘째 문장 {i} 이다. 셋째 문장 {i} 이다.", claim=f"둘째 문장 {i} 이다")
            g3ko.append(r)
        mit_rev = [dict(row(f"mr{i}", "mit-cs", "", group=f"doc:repo:f{i % 4}.md"), want="") for i in range(10)]
        judged = mit_rev[:3]
        return {"g3ko": g3ko, "mit_rev": mit_rev, "mit_judged": judged}

    def test_pick_targets_ids_no_judged_no_taken(self):
        src = self.src()
        taken = set()
        targets = {"pick": {lb.BOOLQ: 5, lb.MIT_REV: 20}}
        rows, summary = lb.pick_new(src, NoGuard(), targets, 9, taken)
        forms = collections.Counter(r["form"] for r in rows)
        self.assertEqual(forms[lb.BOOLQ], 5)
        self.assertEqual(forms[lb.MIT_REV], 7)       # 판정한 3줄 빼고 7 (묶음 넷 × 2 = 8 안)
        self.assertFalse({r["src_ids"][0] for r in rows} & {"src-mr000", "src-mr001", "src-mr002"})
        for r in rows:
            self.assertRegex(r["id"], rf"^g4bal-{r['form']}-\d{{4}}-h{lb.sha_key(r)[:8]}$")
            self.assertEqual(r["set"], lb.LAYER)
        again, _ = lb.pick_new(src, NoGuard(), targets, 9, taken)
        self.assertEqual(json.dumps(rows, ensure_ascii=False), json.dumps(again, ensure_ascii=False))
        boolq_keys = {lb.sha_key(r) for r in rows if r["form"] == lb.BOOLQ}
        fewer, summary = lb.pick_new(src, NoGuard(), targets, 9, boolq_keys)
        self.assertFalse(boolq_keys & {lb.sha_key(r) for r in fewer})
        self.assertEqual(summary["dropped"][f"{lb.BOOLQ}:taken"], 5)

    def test_pick_two_per_group_counted_per_form(self):
        g3ko, mit_rev = [], []
        for i in range(4):
            r = row(f"bq{i}", "kobest-boolq", "a", group="shared:G")
            r.update(evidence=f"첫 문장 {i} 이다. 둘째 문장 {i} 이다. 셋째 문장 {i} 이다.", claim=f"둘째 문장 {i} 이다")
            g3ko.append(r)
            mit_rev.append(dict(row(f"mr{i}", "mit-cs", "", group="shared:G"), want=""))
        rows, _ = lb.pick_new({"g3ko": g3ko, "mit_rev": mit_rev}, NoGuard(), {"pick": {lb.BOOLQ: 4, lb.MIT_REV: 4}}, 9)
        forms = collections.Counter(r["form"] for r in rows)
        self.assertEqual((forms[lb.BOOLQ], forms[lb.MIT_REV]), (2, 2))      # 같은 묶음이라도 꼴마다 2

    def test_pick_guard_rows_never_in(self):
        src = self.src()
        src["mit_rev"][5]["claim"] = "금지 주장 이다."
        rows, summary = lb.pick_new(src, WordGuard(), {"pick": {lb.MIT_REV: 20}}, 9)
        self.assertFalse(any("금지" in r["claim"] for r in rows))
        self.assertEqual(summary["dropped"]["made:eval_text"], 1)


class CliTest(unittest.TestCase):
    def test_refuses_output_inside_git(self):
        for cmd in ("sample-v5", "pick-new", "build"):
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(lb, "inside_git", lambda path: True), mock.patch("sys.stdout"):
                    code = lb.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", lb.LAYER)))

    def test_refuses_overwrite(self):
        for cmd, name in (("sample-v5", "sample-v5.jsonl"), ("pick-new", "pick.jsonl"), ("build", "cands.jsonl")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "g4", lb.LAYER)
                os.makedirs(out)
                with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                    f.write("old\n")
                with mock.patch("sys.stdout"):
                    code = lb.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                with open(os.path.join(out, name), encoding="utf-8") as f:
                    self.assertEqual(f.read(), "old\n")

    def end_to_end_root(self, tmp, over=None):
        """build 시험 뿌리. over 가 None 이 아니면 <g4>/train-g4-bal/v5-rate.json 을 그 over 로 둔다."""
        def put(rel, rows):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            lb.write_jsonl(os.path.join(tmp, rel), rows)
        put("sets/2026-10-10/build/train-r2.jsonl", [])
        put("sets/g1/train-g1-arch.jsonl", [])
        put("sets/g3/train-g3-ko.jsonl", [])
        put("sets/g2/train-g2-b.jsonl", [])
        put("sets/g4/train-g4-c.jsonl", many("kc", "kowiki", "c", 5))
        put("sets/g4/train-g4-d.jsonl", many("kd", "kowiki", "d", 5))
        put("sets/g1/train-g1-II.jsonl", many("pc", "proposals", "c", 8) + many("jr", "jira", "c", 4)
            + many("sy", "proposals", "c", 4, origin="synth"))
        pick = [dict(row(f"g4bal-mit-rev-{i:04d}", "mit-cs", "b"), want="") for i in range(3)]
        put("g4/train-g4-bal/pick.jsonl", pick)
        put("g4/judge/train-g4-bal/decided.jsonl", [{"id": r["id"], "label": "b", "status": "agree"} for r in pick])
        if over is not None:
            lb.write_json(os.path.join(tmp, "g4", lb.LAYER, "v5-rate.json"), {"over": over})
        targets = os.path.join(tmp, "targets.json")
        with open(targets, "w", encoding="utf-8") as f:
            json.dump(SMALL | {"build": dict(SMALL["build"], **{"mit-cs": {"b": 2}})}, f)
        return ["build", "--g4", os.path.join(tmp, "g4"), "--old-root", tmp, "--targets", targets, "--seed", "3"]

    def run_build(self, argv):
        with mock.patch.object(lb, "_guard", lambda *args: (NoGuard(), [])), \
                mock.patch.object(lb, "old_eval_marks", lambda root: (set(), set(), [])), mock.patch("sys.stdout"):
            return lb.main(argv)

    def built(self, tmp):
        g4 = os.path.join(tmp, "g4", lb.LAYER)
        rows = lb.read_jsonl(os.path.join(g4, "cands.jsonl"))
        with open(os.path.join(g4, "summary.json"), encoding="utf-8") as f:
            return rows, json.load(f)

    def test_build_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self.end_to_end_root(tmp) + ["--no-v5-rate"]
            self.assertEqual(self.run_build(argv), 0)
            self.assertEqual(self.run_build(argv), 2)                                  # 덮어쓰기 거부
            rows, summary = self.built(tmp)
        got = collections.Counter(f"{r['domain']} · {r['want']}" for r in rows)
        self.assertEqual(dict(got), {"kowiki · c": 2, "kowiki · d": 2, "proposals · c": 6, "mit-cs · b": 2})
        self.assertFalse(any(r["domain"] == "jira" or r.get("origin") == "synth" for r in rows))
        self.assertEqual(summary["recheck"], {})
        self.assertTrue(all(r["set"] == lb.LAYER for r in rows))

    def test_build_blocks_v5_rate_over_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_build(self.end_to_end_root(tmp, over=["proposals · c"])), 0)
            rows, summary = self.built(tmp)
        self.assertFalse(any(r["domain"] == "proposals" for r in rows))              # 넘은 칸의 v5 줄은 안 든다
        self.assertEqual(summary["inputs"]["v5_block"], ["proposals:c"])
        self.assertEqual(summary["inputs"]["v5"]["blocked"], 8)

    def test_build_stops_without_v5_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_build(self.end_to_end_root(tmp)), 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "g4", lb.LAYER, "cands.jsonl")))

    def test_rate_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "g4", lb.LAYER)
            os.makedirs(out)
            lb.write_json(os.path.join(out, "v5-rate.json"), {"over": []})
            with open(os.path.join(out, "v5-rate.json"), encoding="utf-8") as f:
                before = f.read()
            with mock.patch("sys.stdout"):
                self.assertEqual(lb.main(["v5-rate", "--g4", os.path.join(tmp, "g4"), "--old-root", tmp]), 2)
            with open(os.path.join(out, "v5-rate.json"), encoding="utf-8") as f:
                self.assertEqual(f.read(), before)

    def test_force_moves_old_output_only_after_input_check(self):
        for cmd, name in (("sample-v5", "sample-v5.jsonl"), ("pick-new", "pick.jsonl"), ("build", "cands.jsonl")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "g4", lb.LAYER)
                os.makedirs(out)
                with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                    f.write("old\n")
                with mock.patch("sys.stdout"):                                     # 입력이 없다 → 멈춤
                    code = lb.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp, "--force"])
                self.assertEqual(code, 2, cmd)
                self.assertEqual(sorted(os.listdir(out)), [name], cmd)               # 옛 출력이 제자리에 그대로


if __name__ == "__main__":
    unittest.main()
