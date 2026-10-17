"""measure/compare.py · gates.py — protocol-g9 F1 지킴 (칸별 F1 · macro-F1 · 씨앗 5 필수).

F1 은 뽑기마다 혼동표(정답 × 예측, 씨앗 합)를 다시 세서 새로 센다. 기대값은 각 시험 주석의 손셈.
손 셋은 test_compare.make_set (쌍 40 = 묶음 5 × (d 4 · b 2 · a 1 · c 1)).
"""
import json
import os
import tempfile
import unittest

import helpers  # noqa: F401  (import 길)

import compare as cmp
import gates
import test_compare as tc

PROTOCOL_G8 = os.path.join(helpers.ROOT, "measure", "protocol-g8.json")
PROTOCOL_G9 = os.path.join(helpers.ROOT, "measure", "protocol-g9.json")
SEEDS5 = (1, 2, 3, 4, 5)


def table(gold, preds_by_seed):
    """손 쌍 표 — gold 는 쌍마다 정답 번호, preds_by_seed 는 씨앗마다 쌍별 예측 번호."""
    counts = []
    for i in range(len(gold)):
        row = [0, 0, 0, 0]
        for preds in preds_by_seed:
            row[preds[i]] += 1
        counts.append(row)
    return {"ids": [f"p{i}" for i in range(len(gold))], "gold": list(gold), "counts": counts,
            "seeds": len(preds_by_seed)}


class F1ValueTest(unittest.TestCase):
    """작은 손 혼동표 — 정답 [a a b c d] · 예측 [a b b c c] (씨앗 하나)."""

    GOLD = [0, 0, 1, 2, 3]
    PRED = [0, 1, 1, 2, 2]

    def test_cells_and_macro(self):
        # tp a1 b1 c1 d0 · 예측 수 a1 b2 c2 d0 · 정답 수 a2 b1 c1 d1
        # F1 = 2tp / (정답 + 예측) → a 2/3 · b 2/3 · c 2/3 · d 0 · macro (2/3 × 3 + 0) / 4 = .5
        gold, tp, pred = [2, 1, 1, 1], [1, 1, 1, 0], [1, 2, 2, 0]
        for metric, want in (("a_f1", 2 / 3), ("b_f1", 2 / 3), ("c_f1", 2 / 3), ("d_f1", 0.0), ("macro_f1", 0.5)):
            self.assertAlmostEqual(cmp.f1_value(metric, gold, tp, pred, 1), want, msg=metric)

    def test_seed_sum_is_seed_mean(self):
        # 씨앗 둘이 같은 답이면 혼동표가 두 배 — F1 은 그대로
        self.assertAlmostEqual(cmp.f1_value("b_f1", [2, 1, 1, 1], [2, 2, 2, 0], [2, 4, 4, 0], 2), 2 / 3)

    def test_undefined_cell(self):
        # 정답도 예측도 없는 칸은 셀 수 없다 — macro 도 None
        self.assertIsNone(cmp.f1_value("d_f1", [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], 1))
        self.assertIsNone(cmp.f1_value("macro_f1", [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], 1))

    def test_bootstrap_one_group(self):
        # 묶음 하나뿐이면 뽑기마다 같은 혼동표 — 구간 = 점값 차. base 는 다 맞힘(F1 다 1), new 는 위 손표
        base = table(self.GOLD, [self.GOLD])
        new = table(self.GOLD, [self.PRED])
        out = cmp.f1_bootstrap(["g"] * 5, base, new, ["b_f1", "d_f1", "macro_f1"], n=50)
        self.assertEqual((out["b_f1"]["base"], out["b_f1"]["new"], out["b_f1"]["diff"]), (1.0, 0.6667, -0.3333))
        self.assertEqual((out["b_f1"]["low"], out["b_f1"]["high"]), (-0.3333, -0.3333))
        self.assertEqual((out["d_f1"]["diff"], out["macro_f1"]["diff"]), (-1.0, -0.5))
        self.assertEqual((out["macro_f1"]["pairs"], out["macro_f1"]["groups"]), (5, 1))
        self.assertIsNone(out["b_f1"]["mcnemar"])

    def test_bootstrap_draws_all_groups(self):
        # 정밀은 모든 정답 칸에서 온다 — 정답 b 가 없는 묶음도 뽑는다 (재현 자와 다른 점)
        gold = [1, 0, 0]
        base = table(gold, [[1, 0, 0]])
        new = table(gold, [[1, 1, 1]])     # 묶음 h · i 의 a 를 b 로 — b 정밀만 떨어진다
        out = cmp.f1_bootstrap(["g", "h", "i"], base, new, ["b_f1"], n=200)
        self.assertEqual(out["b_f1"]["groups"], 3)
        self.assertEqual(out["b_f1"]["diff"], -0.5)          # 1 → 2 × 1 / (1 + 3) = .5
        self.assertLess(out["b_f1"]["low"], out["b_f1"]["high"])

    def test_public_neutral_gold_left_out(self):
        # 정답 CD(공개 중립)는 F1 셈에서 뺀다
        base = table([0, cmp.CD], [[0, 2]])
        out = cmp.f1_bootstrap(["g", "h"], base, base, ["a_f1"], n=10)
        self.assertEqual((out["a_f1"]["pairs"], out["a_f1"]["groups"], out["a_f1"]["base"]), (1, 1, 1.0))


class G9PairTest(unittest.TestCase):
    """protocol-g9 짝 비교 — 손 셋 · 씨앗 5."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = tc.Fixture(self._tmp.name, PROTOCOL_G9)

    def tearDown(self):
        self._tmp.cleanup()

    def test_f1_rulers(self):
        # base_pred : 묶음마다 d 넷 중 둘을 c 로 → 씨앗 5 합 d tp 50 · 예측 d 50 · 정답 d 100 → F1_d 100/150 = .6667
        #             c tp 25 · 예측 c 25 + 50 = 75 · 정답 c 25 → F1_c 50/100 = .5 · a · b 는 1
        # new_pred : g1~g4 d tp 16 (셋째까지 5 · 넷째 씨앗 1) · g5 10 → tp 74 · 예측 d 74 → F1_d 148/174 = .8506
        #            예측 c 25 + 26 = 51 → F1_c 50/76 = .6579 · macro (1 + 1 + .6579 + .8506) / 4 = .8771
        protocol, stats = self.fx.stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        m = stats["metrics"]
        self.assertEqual((m["d_f1"]["base"], m["d_f1"]["new"], m["d_f1"]["diff"]), (0.6667, 0.8506, 0.1839))
        self.assertEqual((m["c_f1"]["base"], m["c_f1"]["new"]), (0.5, 0.6579))
        self.assertEqual((m["macro_f1"]["base"], m["macro_f1"]["new"]), (0.7917, 0.8771))
        self.assertEqual((m["b_f1@en"]["diff"], m["b_f1@ko"]["diff"]), (0.0, 0.0))
        self.assertEqual(m["b_f1@ko"]["groups"], 3)
        self.assertLessEqual(m["d_f1"]["low"], m["d_f1"]["diff"])
        v = cmp.verdict(protocol, stats)
        names = [g["metric"] for g in v["guards"]]
        self.assertEqual(names, ["b_f1@en", "b_f1", "macro_f1", "a_f1", "c_f1", "d_f1", "false_support", "side_acc3",
                                 "b_f1@ko", "b_recall@ko", "b_recall@en"])
        self.assertTrue(all(g["ok"] for g in v["guards"]))
        self.assertEqual(v["verdict"], "같다")             # 목표 c_recall@ko 는 1 → 1

    def test_b_broken(self):
        # broken_pred : g1 · g2(ko) 의 b 넷을 a 로 → b F1@ko 와 재현 바닥이 깨진다
        protocol, stats = self.fx.stats(tc.base_pred, tc.broken_pred, seeds=SEEDS5)
        v = cmp.verdict(protocol, stats)
        bad = {g["metric"] for g in v["guards"] if g["ok"] is False}
        self.assertTrue({"b_f1@ko", "b_f1", "b_recall@ko"} <= bad, bad)
        self.assertNotIn("b_f1@en", bad)
        self.assertEqual(v["verdict"], "지킴 깨짐")

    def test_three_seeds_refused(self):
        with self.assertRaises(cmp.Refuse):
            self.fx.stats(tc.base_pred, tc.new_pred)          # 씨앗 3 < min_seeds 5

    def test_markdown_f1_rows(self):
        protocol, stats = self.fx.stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        stats.pop("_tables")
        result = {"base": "B0", "new": "B1", "set_sha256": "0" * 64, "protocol_sha256": "1" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats)}
        md = cmp.to_markdown(result)
        self.assertIn("| d_f1 | 0.6667 | 0.8506 | 0.1839 |", md)
        self.assertIn("| 지킴 macro_f1 | O |", md)


class G8RegressionTest(unittest.TestCase):
    """F1 을 더해도 g8 셈은 그대로 — g8 규약엔 F1 자가 없고, F1 줄을 더한 규약에서도 옛 칸 값은 한 자리도 안 바뀐다.
    (실자료 회귀 : K → K2 · K2-h · K3 · K4 · K5 · K6 g8 compare json · md 가 바꾸기 전과 바이트 같음 — 조사 E6 1절)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = tc.Fixture(self._tmp.name, PROTOCOL_G8)

    def tearDown(self):
        self._tmp.cleanup()

    def test_g8_has_no_f1(self):
        g8, _ = cmp.read_protocol(PROTOCOL_G8)
        self.assertEqual(cmp.gate_f1_specs(g8), [])
        _, stats = self.fx.stats(tc.base_pred, tc.new_pred)          # 씨앗 3 — g8 은 min_seeds 가 없다
        self.assertFalse([k for k in stats["metrics"] if "_f1" in k])
        self.assertTrue(all(s["mcnemar"] is not None for s in stats["metrics"].values()))

    def test_old_cells_same_with_f1_rulers(self):
        protocol, plain = self.fx.stats(tc.base_pred, tc.broken_pred)
        with_f1 = dict(protocol, pair_gate=dict(protocol["pair_gate"],
                                                guards=protocol["pair_gate"]["guards"] + [{"metric": "b_f1", "lang": "ko",
                                                                                           "ci_low_ge": -0.03}]))
        _, base_runs = cmp.load_runs(self.fx.runs("B0", tc.base_pred))
        _, new_runs = cmp.load_runs(self.fx.runs("B1", tc.broken_pred))
        again = cmp.compare(with_f1, cmp.read_set(self.fx.set_path), base_runs, new_runs)
        for key, s in plain["metrics"].items():
            self.assertEqual(again["metrics"][key], s, key)
        self.assertIn("b_f1@ko", again["metrics"])
        self.assertEqual(list(again["metrics"])[:len(plain["metrics"])], list(plain["metrics"]))


class G9ProtocolTest(unittest.TestCase):
    """protocol-g9 — g8 에서 바뀌는 칸은 지킴 · min_seeds · 문구뿐 (E5 5절 · 사용자 확정 2026-10-17)."""

    def test_same_as_g8_but_guards(self):
        g8, _ = cmp.read_protocol(PROTOCOL_G8)
        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        for key in ("sets", "compare_on", "calibrate_on", "release_pick", "bootstrap", "seeds", "majority", "release_gate",
                    "side", "group_key", "label_guide", "report_only_sets", "seed_rerun", "ckpt", "train_sets",
                    "cell_check", "sets_note", "train_sets_note"):
            self.assertEqual(g9[key], g8[key], key)
        for key in ("target", "alarms", "alarms_note", "slice_note", "base"):
            self.assertEqual(g9["pair_gate"][key], g8["pair_gate"][key], key)
        self.assertNotIn("draft_note", g9)
        self.assertEqual((g9["name"], g9["pair_gate"]["min_seeds"]), ("protocol-g9", 5))
        self.assertEqual(g9["bootstrap"], {"n": 10000, "ci": 0.95, "rng_seed": 20261011, "resample": "group",
                                           "seeds": "고정 — 씨앗은 다시 뽑지 않는다"})

    def test_guards_e5_draft(self):
        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        got = {cmp.ruler_name(g): g.get("ci_low_ge", g.get("max_ratio", g.get("min_mean_diff")))
               for g in g9["pair_gate"]["guards"]}
        self.assertEqual(got, {"b_f1@en": -0.03, "b_f1": -0.02, "macro_f1": -0.02, "a_f1": -0.03, "c_f1": -0.03,
                               "d_f1": -0.03, "false_support": 1.2, "side_acc3": -0.010, "b_f1@ko": -0.03,
                               "b_recall@ko": -0.05, "b_recall@en": -0.05})
        g8, _ = cmp.read_protocol(PROTOCOL_G8)
        side8 = [g for g in g8["pair_gate"]["guards"] if g["metric"] in ("side_acc3", "false_support")]
        side9 = [g for g in g9["pair_gate"]["guards"] if g["metric"] in ("side_acc3", "false_support")]
        self.assertEqual(side9, side8)


class G9ReleaseF1Test(unittest.TestCase):
    """gates.py release 의 vs_ref 도 F1 자를 읽는다 (g9 릴리스 관문은 g8 그대로라 지금은 안 쓰지만, 꼴만 맞춰 둔다)."""

    def test_vs_ref_f1_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = tc.Fixture(tmp, PROTOCOL_G9)
            protocol, _ = cmp.read_protocol(fx.protocol)
            protocol["release_gate"] = dict(protocol["release_gate"], vs_ref=[{"metric": "d_f1", "ci_low_ge": -0.03}])
            ref_run = cmp.load_run(fx.runs("R", tc.base_pred, seeds=(1,)).split("=", 1)[1])
            run = cmp.load_run(fx.runs("N", tc.new_pred, seeds=(1,)).split("=", 1)[1])
            table_new = cmp.pair_hits([run], list(cmp.read_set(fx.set_path)))
            rows = gates.vs_ref_rows(protocol, cmp.read_set(fx.set_path), table_new, ref_run)
            self.assertEqual(rows[0][0], "vs_ref_d_f1")
            # 씨앗 1 : base d tp 10 / 예측 10 / 정답 20 → .6667 · new g1~g4 tp 4 · g5 2 → 18 → 36/38 = .9474
            self.assertEqual(rows[0][1]["diff"], round(0.9474 - 0.6667, 4))
            self.assertIs(rows[0][3], True)


if __name__ == "__main__":
    unittest.main()
