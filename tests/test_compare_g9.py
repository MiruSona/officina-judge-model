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
        # 목표 macro_f1@ko/nogen (손 셋엔 form 칸이 없어 뺄 쌍 0) — ko 묶음 g1~g3 셋 다 같은 꼴이라 구간 = 점값 차
        # base ko : d tp 30 · 예측 d 30 · 정답 d 60 → .6667 · c tp 15 · 예측 c 45 · 정답 c 15 → .5 → macro .7917
        # new ko : d tp 48 · 예측 d 48 → 96/108 = .8889 · c 예측 27 → 30/42 = .7143 → macro .9008 · 차 +.1091
        self.assertEqual(v["target"]["metric"], "macro_f1@ko/nogen")
        self.assertEqual((v["target"]["diff"], v["target"]["low"]), (0.1091, 0.1091))
        self.assertEqual(m["macro_f1@ko"]["diff"], 0.1091)  # 모든 쌍 짝 (보고만) — 뺄 쌍이 없으니 같다
        self.assertEqual(v["verdict"], "먹힘")

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
        # K7 설계 1-5 : g8 에서 바뀌는 칸은 지킴 · min_seeds · 비교 셋(dev-cmp-g9) · 목표 · 보고만 · 생성기 꼴 · 학습 고정값
        g8, _ = cmp.read_protocol(PROTOCOL_G8)
        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        for key in ("calibrate_on", "bootstrap", "seeds", "majority", "release_gate", "side", "group_key", "label_guide",
                    "seed_rerun", "ckpt", "train_sets"):
            self.assertEqual(g9[key], g8[key], key)
        for key in ("alarms", "alarms_note", "slice_note", "base"):
            self.assertEqual(g9["pair_gate"][key], g8["pair_gate"][key], key)
        self.assertEqual(g9["release_pick"], dict(g8["release_pick"], set="dev-cmp-g9"))
        self.assertEqual(g9["cell_check"]["dev"], "dev-cmp-g9")
        self.assertEqual(g9["cell_check"]["recipe"], g9["train"]["recipe"])
        self.assertEqual({k: v for k, v in g9["cell_check"].items() if k not in ("dev", "recipe", "note")},
                         {k: v for k, v in g8["cell_check"].items() if k not in ("dev", "recipe", "note")})
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



def form_fixture(tmp, protocol_path=PROTOCOL_G9, forms=None):
    """손 셋에 form 칸을 단 짝 — ko 묶음 g1~g3 의 넷째 d(k 3)를 near-x 로. 셋 해시 · 규약 해시를 다시 맞춘다."""
    fx = tc.Fixture(tmp, protocol_path)
    for row in fx.set_rows:
        row["form"] = "near-x" if row["lang"] == "ko" and row["k"] == 3 else "labeled"
    tc.write_jsonl(fx.set_path, fx.set_rows)
    fx.set_sha = cmp.sha256_file(fx.set_path)
    with open(fx.protocol, encoding="utf-8") as f:
        protocol = json.load(f)
    protocol["sets"][protocol["compare_on"]] = fx.set_sha
    if forms is not None:
        protocol["generator_forms"] = forms
    with open(fx.protocol, "w", encoding="utf-8", newline="\n") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=1)
    fx.protocol_sha = cmp.sha256_file(fx.protocol)
    return fx


class GeneratorFormsTest(unittest.TestCase):
    """목표 자는 generator_forms 꼴을 뺀 쌍으로, 모든 쌍 값은 곁에 보고만, 지킴은 모든 쌍 그대로 (K7 설계 1-3)."""

    def test_target_drops_generator_forms(self):
        # 뺀 ko 쌍 21 (묶음마다 d 셋 · b 둘 · a · c) · 씨앗 5
        # base : d k0 · k1 만 d → d tp 30 · 예측 30 · 정답 45 → .8 · c tp 15 · 예측 30 → 30/45 = .6667 → macro .8667
        # new  : d k0~k2 다 d → d · c F1 1 → macro 1 · 차 +.1333 (모든 쌍 짝은 .7917 → .9008, +.1091)
        with tempfile.TemporaryDirectory() as tmp:
            fx = form_fixture(tmp)
            protocol, stats = fx.stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        m = stats["metrics"]
        self.assertEqual((m["macro_f1@ko/nogen"]["base"], m["macro_f1@ko/nogen"]["new"], m["macro_f1@ko/nogen"]["diff"]),
                         (0.8667, 1.0, 0.1333))
        self.assertEqual(m["macro_f1@ko/nogen"]["pairs"], 21)
        self.assertEqual((m["macro_f1@ko"]["pairs"], m["macro_f1@ko"]["diff"]), (24, 0.1091))
        v = cmp.verdict(protocol, stats)
        self.assertEqual((v["target"]["metric"], v["target"]["diff"]), ("macro_f1@ko/nogen", 0.1333))
        self.assertNotIn("macro_f1@ko", [g["metric"] for g in v["guards"]])     # 모든 쌍 짝은 판정에 안 쓴다
        self.assertEqual(stats["generator_forms"]["rulers"],
                         [{"ruler": "macro_f1@ko/nogen", "all_pairs_ruler": "macro_f1@ko", "all_pairs": 24, "pairs": 21,
                           "dropped": 3, "by_form": {"near-x": 3}}])

    def test_guards_keep_all_pairs(self):
        # 지킴은 꼴을 안 뺀다 — b_f1@ko · d_f1 의 쌍 수 · 값이 form 칸 없는 셋과 같다
        with tempfile.TemporaryDirectory() as tmp:
            _, plain = tc.Fixture(tmp, PROTOCOL_G9).stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        with tempfile.TemporaryDirectory() as tmp:
            _, marked = form_fixture(tmp).stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        for name in ("b_f1@ko", "d_f1", "macro_f1", "b_recall@ko", "c_recall@ko"):
            self.assertEqual(marked["metrics"][name], plain["metrics"][name], name)
        self.assertEqual(marked["metrics"]["b_f1@ko"]["pairs"], 24)

    def test_missing_list_refused(self):
        # 목표가 drop_forms 를 달았는데 규약에 목록이 비면 조용히 다 세지 않고 거부
        with tempfile.TemporaryDirectory() as tmp:
            fx = form_fixture(tmp, forms=[])
            with self.assertRaises(cmp.Refuse):
                fx.stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        with self.assertRaises(cmp.Refuse):
            cmp.drop_forms({}, {"metric": "macro_f1", "drop_forms": "other_forms"})

    def test_markdown_line_and_g8_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, stats = form_fixture(tmp).stats(tc.base_pred, tc.new_pred, seeds=SEEDS5)
        stats.pop("_tables")
        result = {"base": "B0", "new": "B1", "set_sha256": "0" * 64, "protocol_sha256": "1" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats)}
        md = cmp.to_markdown(result)
        self.assertIn("- 생성기 꼴 뺌 macro_f1@ko/nogen : 쌍 24 중 3 뺌 (near-x 3) — 모든 쌍 값 macro_f1@ko 은 보고만", md)
        self.assertIn("| 목표 macro_f1@ko/nogen | O |", md)
        # g8 은 drop_forms 자가 없어 출력 칸 · 자 이름이 그대로다 (G8RegressionTest 와 같은 셈)
        with tempfile.TemporaryDirectory() as tmp:
            _, g8 = form_fixture(tmp, PROTOCOL_G8).stats(tc.base_pred, tc.new_pred)
        self.assertNotIn("generator_forms", g8)
        self.assertFalse([k for k in g8["metrics"] if k.endswith(cmp.DROP_TAIL)])


class G9SetsTest(unittest.TestCase):
    """protocol-g9 의 K7 칸 — dev-cmp-g9 (해시는 얼린 뒤) · 옛 셋 보고만 · 목표 · 학습 고정값 · prereg 다음 걸음."""

    def test_k7_fields(self):
        g8, _ = cmp.read_protocol(PROTOCOL_G8)
        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        self.assertEqual(g9["compare_on"], "dev-cmp-g9")
        self.assertIsNone(g9["sets"]["dev-cmp-g9"])
        self.assertNotIn("dev-cmp-g4", g9["sets"])
        self.assertEqual(g9["report_only_sets"]["dev-cmp-g4"], g8["sets"]["dev-cmp-g4"])
        self.assertEqual({k: g9["sets"][k] for k in ("dev-cal-g3", "blind-g4")},
                         {k: g8["sets"][k] for k in ("dev-cal-g3", "blind-g4")})
        self.assertEqual(g9["pair_gate"]["target"], {"metric": "macro_f1", "lang": "ko", "drop_forms": "generator_forms",
                                                     "min_mean_diff": 0.03, "ci_low_gt": 0.0})
        self.assertEqual(g9["pair_gate"]["new"], "K7")
        self.assertIn("near-x", g9["generator_forms"])
        self.assertEqual(g9["train"]["class_weight_fixed"],
                         [0.8646582405778421, 0.9994239425532274, 0.8340563960214578, 1.3018614208474724])
        self.assertTrue(g9["pair_gate"]["report_only"])

    def test_null_set_hash_refused(self):
        # 셋을 얼리기 전(null)에는 compare 가 대조를 건너뛰지 않고 거부한다. 옛 꼴 빈 글자는 예전처럼 건너뛴다
        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        with self.assertRaisesRegex(cmp.Refuse, "아직 비었다"):
            cmp.check_hashes(g9, "p" * 64, "s" * 64, [])
        cmp.check_hashes(dict(g9, sets=dict(g9["sets"], **{"dev-cmp-g9": ""})), "p" * 64, "s" * 64, [])

    def test_prereg_next_g9(self):
        import prereg

        g9, _ = cmp.read_protocol(PROTOCOL_G9)
        got = prereg.next_for_gate("/x/protocol-g9.json")
        self.assertIs(got, prereg.NEXT_G9)
        self.assertEqual(set(got), {"pass", "p95_only", "fail", "먹힘", "지킴 깨짐", "두 판 연속", "같다"})
        self.assertIn("E6", got["먹힘"])
        self.assertIn("v1.2 접음", got["두 판 연속"])
        self.assertIn("8092 v1.1.0 유지", got["두 판 연속"])
        self.assertIn("대체 판은 적지 않는다", got["지킴 깨짐"])
        self.assertTrue(set(g9["pair_gate"]["verdicts"]) >= {"먹힘", "지킴 깨짐", "같다"})

if __name__ == "__main__":
    unittest.main()
