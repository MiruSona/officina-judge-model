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


class KlbPickTest(unittest.TestCase):
    """K5 train-g4-klb 후보 (K5 설계 4-3)."""

    def ka(self, q, group):
        out = []
        for cell, form in (("b", "qa-alt"), ("a", "qa-true")):
            r = row(f"{q}-{cell}", "klue-answered", cell, group=group, form=form, split="train", pair=q, src_ids=[q],
                    claim_rewrite=f"평서문 {q} {cell} 이다.", want="")
            r["claim"] = f"질문 {q} 인가?"
            r["evidence"] = f"근거 {q} 하나뿐인 문장이다."
            out.append(r)
        return out

    def mrc(self, rid, group):
        return dict(row(rid, "klue-mrc", "d", group=group, form="unanswerable", split="train", src_ids=[f"g-{rid}"],
                        want="", question=f"물음 {rid} 인가?"), claim="", evidence=f"{rid * 8} 은 서로 다른 글이다.")

    def src(self):
        rw = []
        for i in range(8):
            rw += self.ka(f"Q{i}", f"klue:D{i // 2}")      # 묶음 하나에 질문 둘
        mrc = [self.mrc(f"u{i}", f"klue:M{i // 3}") for i in range(9)] + [self.mrc("solo", "klue:S")]
        return {"klue_rw": rw, "klue_mrc": mrc, "klue_check": [{"pair": "Q7", "pass": False}],
                "klue_done": [[{"group": "klue:M0", "src_ids": ["g-u0"]}]]}

    def layer(self):
        anchor = row("bal-d", "klue-answered", "d", group="klue:D3", form=lb.KLUE_ANS, src_ids=["Q6"])
        used = row("k-b", "klue-answered", "b", group="klue:D0", form="qa-alt", src_ids=["Q0"], pair="Q0")
        return [anchor, used]

    def test_ka_order_targets_and_exclusions(self):
        targets = {"klb": {"ka": {"b": 4, "a": 2}, "mrc": 0}}
        ka, _, summary = lb.pick_klb(self.src(), NoGuard(), self.layer(), targets, 9)
        qs = [r["pair"] for r in ka]
        self.assertNotIn("Q0", qs)          # 층에 쓴 질문
        self.assertNotIn("Q7", qs)          # 평서문 검사 떨어짐
        self.assertEqual(collections.Counter(r["hint"] for r in ka), {"b": 4, "a": 2})
        trio = [r for r in ka if r["trio"]]
        self.assertEqual({(r["pair"], r["anchor"]) for r in trio}, {("Q6", "bal-d")})   # 닻 질문 먼저 · b · a 둘 다
        self.assertEqual(len(trio), 2)
        self.assertTrue(all(r["claim"].startswith("평서문") and r["want"] == "" and r["id"].startswith("klb-ka-") for r in ka))
        self.assertEqual(summary["ka"]["anchor_questions"], 1)
        per_group = collections.Counter(r["group"] for r in ka if r["hint"] == "b")
        self.assertLessEqual(max(per_group.values()), 2)

    def test_mrc_same_group_rewritten_and_layer_para_dropped(self):
        layer = self.layer() + [dict(row("bal-m", "klue-mrc", "d", group="klue:M2"), evidence="u8" * 8 + " 은 서로 다른 글이다.")]
        targets = {"klb": {"ka": {"b": 0, "a": 0}, "mrc": 10}}
        _, mrc, summary = lb.pick_klb(self.src(), NoGuard(), layer, targets, 9)
        groups = collections.Counter(r["group"] for r in mrc)
        self.assertNotIn("klue:S", groups)                       # 질문 하나뿐인 묶음
        self.assertNotIn("g-u0", {s for r in mrc for s in r["src_ids"]})    # g1 이 이미 바꾼 줄
        self.assertNotIn("g-u8", {s for r in mrc for s in r["src_ids"]})    # 층 문단과 겹침
        self.assertEqual(dict(groups), {"klue:M0": 2, "klue:M1": 3, "klue:M2": 2})   # 둘째 판에 셋째 질문까지 채움
        self.assertTrue(all(r["needs_rewrite"] and r["claim"] == "" and r["id"].startswith("klb-mrc-u-") for r in mrc))
        self.assertEqual(summary["dropped"]["mrc:layer_para"], 1)
        again = lb.pick_klb(self.src(), NoGuard(), layer, targets, 9)[1]
        self.assertEqual(json.dumps(mrc, ensure_ascii=False), json.dumps(again, ensure_ascii=False))

    def test_dup_hint_counted_and_unknown_input_exit2(self):
        src = self.src()
        src["klue_rw"].append(dict(src["klue_rw"][0]))                  # Q0 b 가 두 줄
        _, _, summary = lb.pick_klb(src, NoGuard(), [], {"klb": {"ka": {"b": 1, "a": 1}, "mrc": 0}}, 9)
        self.assertEqual(summary["dropped"]["ka:dup_hint"], 1)
        for cmd, key in (("pick-klb", "mit"), ("pick-klb", "bogus"), ("pick-new", "bogus")):
            with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stdout"):
                code = lb.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp, "--input", f"{key}=x.jsonl"])
            self.assertEqual(code, 2, (cmd, key))                        # pick-klb 는 klue 셋 열쇠만 안다

    def test_guard_drops_whole_question(self):
        src = self.src()
        src["klue_rw"][3]["claim_rewrite"] = "금지 평서문 이다."      # Q1 의 a 줄
        ka, _, summary = lb.pick_klb(src, WordGuard(), [], {"klb": {"ka": {"b": 10, "a": 10}, "mrc": 0}}, 9)
        self.assertNotIn("Q1", {r["pair"] for r in ka})
        self.assertEqual(summary["dropped"]["ka:eval_text"], 1)


def ka_rows(q, trio=False):
    """klue-answered 한 질문의 b · a 후보 둘."""
    return [row(f"{q}-{c}", "klue-answered", c, group=f"klue:{q}", pair=q, trio=trio) for c in ("b", "a")]


class KlbComposeTest(unittest.TestCase):
    """K5 train-g4-klb 조성 (K5 설계 4-2 · 4-3) — 세 줄 짝 먼저 · 같은 묶음 b · d 짝 · 다른 칸은 자리가 있을 때만."""

    def ka(self, q, trio=False):
        return ka_rows(q, trio)

    def cands(self):
        rows = self.ka("T0", trio=True) + self.ka("P0") + self.ka("P1") + self.ka("S0")
        rows += [row(f"m{i}", "klue-mrc", "d", group=f"klue:M{i // 2}") for i in range(6)]
        return rows

    def decisions(self, cands, change=None):
        out = {r["id"]: {"id": r["id"], "label": r["hint"], "status": "agree", "votes": {"j1": r["hint"], "j2": r["hint"]}}
               for r in cands}
        for rid, label in (change or {}).items():
            out[rid] = dict(out[rid], label=label) if label != "drop" else dict(out[rid], label="", status="drop")
        return out

    def test_trio_first_then_pairs_and_mrc_mates(self):
        cands = self.cands()
        change = {"S0-a": "d", "m0": "b", "m2": "b", "m4": "b", "m5": "a", "P1-b": "drop"}
        targets = {"klb": {"build": {"klue-answered": {"a": 2, "b": 2}, "klue-mrc": {"b": 2, "d": 3}}}}
        rows, summary = lb.compose_klb(cands, self.decisions(cands, change), targets, seed=7)
        got = {r["id"]: r for r in rows}
        self.assertEqual(got["T0-b"]["klb_how"], "trio")                 # 닻 질문이 늘 먼저
        self.assertEqual({r["pair"] for r in rows if r["domain"] == "klue-answered"}, {"T0", "P0"})
        self.assertNotIn("S0-a", got)                                    # d 로 정함 · 목표 0 칸 → 버림
        self.assertEqual(summary["judged"]["ka_moved_dropped:d"], 1)
        # klue-mrc : 짝 d 가 있는 b(m0 · m2) 먼저 b 2 를 채우고, 짝 없는 b(m4)는 넘침 · m5(a)는 버림
        self.assertEqual({r["id"] for r in rows if r["want"] == "b" and r["domain"] == "klue-mrc"}, {"m0", "m2"})
        self.assertEqual({r["id"]: r["mate_of"] for r in rows if r["domain"] == "klue-mrc" and r["want"] == "d"},
                         {"m1": "m0", "m3": "m2"})
        self.assertEqual(summary["judged"]["mrc_b_over"], 1)
        self.assertEqual(summary["judged"]["mrc_dropped:a"], 1)
        self.assertEqual(summary["short"], {"klue-mrc:d": 1})
        self.assertEqual(got["m0"]["moved_from"], "d")                   # hint d → b
        self.assertNotIn("moved_from", got["m1"])
        self.assertTrue(all(r["label_guide"] == "support-v6" and r["status"] == "agree" for r in rows))
        self.assertIn("overlap", got["m1"])
        self.assertNotIn("overlap", got["m0"])
        again = lb.compose_klb(cands, self.decisions(cands, change), targets, seed=7)[0]
        self.assertEqual(json.dumps(rows, ensure_ascii=False), json.dumps(again, ensure_ascii=False))

    def test_priority_beats_id_order(self):
        """쓸 짝이 목표보다 많고 닻 질문 · 짝 있는 b 가 id 뒤쪽 — 차례가 없으면 씨앗 따라 떨어진다."""
        ka = self.ka("Z9", trio=True) + self.ka("Y7") + self.ka("Y8")
        ka += [r for q in ("A0", "A1", "A2") for r in self.ka(q)]
        mrc = [row(f"a{i}", "klue-mrc", "d", group="klue:MA") for i in range(3)]   # b 셋 · d 없음
        mrc += [row("z0", "klue-mrc", "d", group="klue:MZ"), row("z1", "klue-mrc", "d", group="klue:MZ")]
        cands = ka + mrc
        change = {f"{q}-a": "drop" for q in ("A0", "A1", "A2")}
        change.update({"a0": "b", "a1": "b", "a2": "b", "z0": "b"})
        targets = {"klb": {"build": {"klue-answered": {"a": 2, "b": 2}, "klue-mrc": {"b": 1, "d": 1}}}}
        for seed in range(20):
            rows, _ = lb.compose_klb(cands, self.decisions(cands, change), targets, seed=seed)
            got = {r["id"]: r["klb_how"] for r in rows}
            self.assertEqual((got.get("Z9-b"), got.get("Z9-a")), ("trio", "trio"), seed)    # 닻 질문은 늘 든다
            self.assertEqual(sum(1 for k, how in got.items() if how == "pair"), 2, seed)    # 남은 자리는 짝 (홀로 아님)
            self.assertFalse(any(k.startswith("A") for k in got), seed)
            self.assertEqual({k for k in got if k[0] in "az"}, {"z0", "z1"}, seed)        # 짝 d 가 있는 b 먼저

    def test_cli_refuses_bad_cands_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            g4 = os.path.join(tmp, "g4")
            with mock.patch("sys.stdout"):
                self.assertEqual(lb.main(["build-klb", "--g4", g4, "--old-root", tmp, "--no-old-forbid"]), 2)
            self.assertFalse(os.path.exists(os.path.join(g4, lb.KLB_LAYER, "cands.jsonl")))


class KlbCliTest(unittest.TestCase):
    """build-klb 명령 — 판정자가 본 글(map × batches) 대조 · 후보 꼴 거부는 exit 2, 출력 없음."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        for rel in ("sets/2026-10-10/build/train-r2.jsonl", "sets/g1/train-g1-arch.jsonl", "sets/g3/train-g3-ko.jsonl",
                    "sets/g4/train-g4-c.jsonl", "sets/g4/train-g4-bal.jsonl"):
            self.put(rel, [])
        self.cands = ka_rows("T0", trio=True) + ka_rows("P0")
        self.cands += [row(f"m{i}", "klue-mrc", "d", group="klue:M0") for i in range(2)]
        self.targets = os.path.join(self.tmp, "targets.json")
        lb.write_json(self.targets, {"klb": {"build": {"klue-answered": {"a": 2, "b": 2}, "klue-mrc": {"b": 1, "d": 1}}}})

    def tearDown(self):
        self._tmp.cleanup()

    def put(self, rel, rows):
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lb.write_jsonl(path, rows)
        return path

    def judge(self, cands, seen=None):
        """판정 폴더 — decided · map(q → id) · batches(q → 판정자가 본 글). seen 은 글을 바꿔 보일 {id: 주장}."""
        seen = seen or {}
        self.put("judge/decided.jsonl", [{"q": f"q{i}", "id": r["id"], "label": r["hint"], "status": "agree",
                                          "votes": {"j1": r["hint"], "j2": r["hint"]}} for i, r in enumerate(cands)])
        self.put("judge/map/map.jsonl", [{"q": f"q{i}", "batch": "b01", "id": r["id"], "trap": False}
                                         for i, r in enumerate(cands)] + [{"q": "t1", "batch": "b01", "id": "trap-1", "trap": True}])
        items = [{"q": f"q{i}", "evidence": r["evidence"], "claim": seen.get(r["id"], r["claim"])} for i, r in enumerate(cands)]
        for j in (1, 2):
            self.put(f"judge/batches/b01-j{j}.jsonl", items + [{"q": "t1", "evidence": "함정", "claim": "함정 주장"}])

    def run_klb(self, cands):
        path = self.put("in/cands.jsonl", cands)
        argv = ["build-klb", "--g4", os.path.join(self.tmp, "g4"), "--old-root", self.tmp, "--targets", self.targets,
                "--cands", path, "--dirs", os.path.join(self.tmp, "judge")]
        with mock.patch.object(lb, "_guard", lambda *args: (NoGuard(), [])), mock.patch("sys.stdout"), \
                mock.patch("sys.stderr"):
            return lb.main(argv)

    def out_exists(self):
        return os.path.exists(os.path.join(self.tmp, "g4", lb.KLB_LAYER, "cands.jsonl"))

    def test_good_run_writes(self):
        self.judge(self.cands)
        self.assertEqual(self.run_klb(self.cands), 0)
        self.assertTrue(self.out_exists())

    def test_pair_sha_mismatch_stops(self):
        self.judge(self.cands, seen={"m0": "판정자가 본 다른 주장 이다."})
        self.assertEqual(self.run_klb(self.cands), 2)
        self.assertFalse(self.out_exists())

    def test_batches_disagree_or_map_missing_stops(self):
        self.judge(self.cands)
        self.put("judge/batches/b01-j2.jsonl", [{"q": "q0", "evidence": "다른 근거", "claim": "다른 주장"}])
        self.assertEqual(self.run_klb(self.cands), 2)
        self.judge(self.cands)
        os.remove(os.path.join(self.tmp, "judge", "map", "map.jsonl"))
        self.assertEqual(self.run_klb(self.cands), 2)
        self.judge(self.cands[:-1])                       # 판정은 있는데 판정자가 본 글이 없다
        self.put("judge/decided.jsonl", lb.read_jsonl(os.path.join(self.tmp, "judge", "decided.jsonl"))
                 + [{"q": "qx", "id": self.cands[-1]["id"], "label": "d", "status": "agree"}])
        self.assertEqual(self.run_klb(self.cands), 2)
        self.assertFalse(self.out_exists())

    def test_bad_cand_shapes_stop(self):
        self.judge(self.cands)
        bad = {"empty_claim": [dict(self.cands[0], claim="")] + self.cands[1:],
               "dup_id": self.cands + [dict(self.cands[0])],
               "outside_domain": self.cands + [row("w0", "kowiki", "c")],
               "no_pair": [{k: v for k, v in self.cands[0].items() if k != "pair"}] + self.cands[1:],
               "no_group": self.cands[:-1] + [{k: v for k, v in self.cands[-1].items() if k != "group"}]}
        for name, cands in bad.items():
            self.assertEqual(self.run_klb(cands), 2, name)
            self.assertFalse(self.out_exists(), name)


class CliTest(unittest.TestCase):
    def test_refuses_output_inside_git(self):
        for cmd in ("sample-v5", "pick-new", "build", "pick-klb", "build-klb"):
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(lb, "inside_git", lambda path: True), mock.patch("sys.stdout"):
                    code = lb.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", lb.LAYER)))
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", lb.KLB_LAYER)))

    def test_refuses_overwrite(self):
        for cmd, name, layer in (("sample-v5", "sample-v5.jsonl", lb.LAYER), ("pick-new", "pick.jsonl", lb.LAYER),
                                 ("build", "cands.jsonl", lb.LAYER), ("pick-klb", "pick-ka.jsonl", lb.KLB_LAYER),
                                 ("build-klb", "cands.jsonl", lb.KLB_LAYER)):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "g4", layer)
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

    def test_build_refuses_klb_rows_as_v6(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self.end_to_end_root(tmp, over=[])
            klb = os.path.join(tmp, "klb.jsonl")
            lb.write_jsonl(klb, [dict(row("klb-ka-b-0001-h00000000", "klue-answered", "b"), set=lb.KLB_LAYER, klb_how="pair")])
            pick = os.path.join(tmp, "klb-pick.jsonl")
            lb.write_jsonl(pick, [dict(row("klb-ka-b-0002-h00000000", "klue-answered", "b"), set=lb.KLB_LAYER, want="")])
            for v6 in ([klb], [f"{os.path.join(tmp, 'g4', 'judge', 'train-g4-bal')}@{pick}"]):
                with mock.patch("sys.stderr"):
                    self.assertEqual(self.run_build(argv + ["--v6"] + v6), 2, v6)
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", lb.LAYER, "cands.jsonl")))

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
