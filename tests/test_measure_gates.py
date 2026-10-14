"""measure/gates.py — 문턱 고른 셋(dev-cal)으로 관문을 재면 거부 · 릴리스 관문 셈 · calibrate 해시 칸 · --replay-n."""
import argparse
import ast
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import helpers

import baseline
import calibrate
import compare as cmp
import gates

PROTOCOL = os.path.join(helpers.ROOT, "measure", "protocol-g1.json")
PROTOCOL_G2 = os.path.join(helpers.ROOT, "measure", "protocol-g2.json")
R2 = os.path.join(os.path.expanduser("~"), ".judgemodel", "measure", "2026-10-11-r2", "loop")
DEV_R2 = os.path.join(os.path.expanduser("~"), ".judgemodel", "sets", "2026-10-10", "dev-r2.jsonl")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quiet(fn, *args):
    """표준 출력·오류를 잡는다. calibrate.main 이 stdout.reconfigure 를 불러 바이트 감싸개를 쓴다."""
    out, err = io.TextIOWrapper(io.BytesIO(), encoding="utf-8"), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args)
    out.flush()
    return code, out.buffer.getvalue().decode("utf-8"), err.getvalue()


def make_pred(n=80):
    """칸 넷이 고르게 · 대부분 맞힘(확신 높음) · 일부는 틀림."""
    rows = []
    for i in range(n):
        label = i % 4
        guess = label if i % 10 else (label + 1) % 4
        logits = [0.0, 0.0, 0.0, 0.0]
        logits[guess] = 4.0
        rows.append({"id": f"q{i:03d}", "label": label, "logits": logits, "probs": [0.25] * 4,
                     "pred": "abcd"[guess], "kind": "", "src": ""})
    return rows


class GateFixture:
    """dev-cal 셋 · blind 셋(같은 쌍, 다른 파일) · 각 pred 와 summary."""

    def __init__(self, tmp, protocol_path=PROTOCOL):
        self.tmp = tmp
        rows = make_pred()
        self.cal_set = os.path.join(tmp, "dev-cal-g1.jsonl")
        self.blind_set = os.path.join(tmp, "blind-g1.jsonl")
        write_jsonl(self.cal_set, [{"id": r["id"], "label": "abcd"[r["label"]], "group": f"g{i % 8}"}
                                   for i, r in enumerate(rows)])
        write_jsonl(self.blind_set, [{"id": r["id"], "label": "abcd"[r["label"]], "group": f"g{i % 8}", "blind": 1}
                                     for i, r in enumerate(rows)])
        # 실제 규약의 셋 해시 칸은 진짜 셋 해시다 — 시험은 그 칸을 비운 복사본을 쓴다 (빈 칸이면 셋 해시를 안 견줌)
        with open(protocol_path, encoding="utf-8") as f:
            protocol = json.load(f)
        protocol["sets"] = {name: "" for name in protocol.get("sets", {})}
        self.protocol = os.path.join(tmp, os.path.basename(protocol_path))
        with open(self.protocol, "w", encoding="utf-8", newline="\n") as f:
            json.dump(protocol, f, ensure_ascii=False, indent=1)
        self.protocol_sha = cmp.sha256_file(self.protocol)
        self.cal_pred = self.eval_dir("eval-dev-cal", rows, self.cal_set)
        self.blind_pred = self.eval_dir("eval-blind", rows, self.blind_set)
        self.calibration = os.path.join(tmp, "calibration.json")

    def eval_dir(self, name, rows, set_path):
        folder = os.path.join(self.tmp, name)
        os.makedirs(folder)
        write_jsonl(os.path.join(folder, "pred.jsonl"), rows)
        with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"set_sha256": cmp.sha256_file(set_path), "protocol_sha256": self.protocol_sha}, f)
        return os.path.join(folder, "pred.jsonl")

    def calibrate(self, *extra):
        quiet(calibrate.main, ["--pred", self.cal_pred, "--out", self.calibration, *extra])
        return cmp.read_json(self.calibration)

    def release(self, set_path, pred, *extra):
        return quiet(gates.main, ["release", "--protocol", self.protocol, "--set", set_path, "--pred", pred,
                                  "--calibration", self.calibration, "--out", os.path.join(self.tmp, "gates.json"),
                                  *extra])

    def ref_dir(self, rows):
        folder = os.path.join(self.tmp, "ref")
        os.makedirs(folder)
        write_jsonl(os.path.join(folder, "pred.jsonl"), rows)
        return folder

    def rulers(self):
        result = cmp.read_json(os.path.join(self.tmp, "gates.json"))
        return result, {r["ruler"]: r for r in result["rulers"]}


class CalibrateHashTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = GateFixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fitted_on_writes_set_hash(self):
        c = self.fx.calibrate("--fitted-on", self.fx.cal_set)
        self.assertEqual(c["calib_set_sha256"], cmp.sha256_file(self.fx.cal_set))
        self.assertEqual(c["fitted_on"]["sha256"], c["calib_set_sha256"])

    def test_without_fitted_on_reads_summary(self):
        c = self.fx.calibrate()
        self.assertEqual(c["calib_set_sha256"], cmp.sha256_file(self.fx.cal_set))

    def test_refuses_without_set_hash(self):
        # --fitted-on 도 없고 pred 옆 summary 에 set_sha256 도 없으면 조용히 None 을 쓰지 않고 거부
        os.remove(os.path.join(os.path.dirname(self.fx.cal_pred), "summary.json"))
        code, _, err = quiet(calibrate.main, ["--pred", self.fx.cal_pred, "--out", self.fx.calibration])
        self.assertEqual(code, 2)
        self.assertIn("거부", err)
        self.assertFalse(os.path.exists(self.fx.calibration))

    def test_old_keys_unchanged(self):
        c = self.fx.calibrate("--fitted-on", self.fx.cal_set)
        for key in ("temperature", "pairs", "labels", "thresholds", "fitted_on"):
            self.assertIn(key, c)


class ReleaseGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = GateFixture(self._tmp.name)
        self.fx.calibrate("--fitted-on", self.fx.cal_set)

    def tearDown(self):
        self._tmp.cleanup()

    def test_refuses_calibrate_dev(self):
        # 문턱을 dev-cal 로 골랐는데 관문도 dev-cal 로 재면 거부
        code, _, err = self.fx.release(self.fx.cal_set, self.fx.cal_pred)
        self.assertEqual(code, 2)
        self.assertIn("calibration", err)

    def test_runs_on_other_set(self):
        # 틀린 8 (i = 0,10,…,70) 은 정답 a→b 넷 · c→d 넷. c→d 는 접으면 맞힘 → 3칸 = 76/80 · d 정답은 다 맞힘
        code, out, _ = self.fx.release(self.fx.blind_set, self.fx.blind_pred)
        self.assertEqual(code, 0, out)
        result = cmp.read_json(os.path.join(self.fx.tmp, "gates.json"))
        rows = {r["ruler"]: r for r in result["rulers"]}
        self.assertAlmostEqual(rows["acc3"]["value"], 76 / 80)
        self.assertAlmostEqual(rows["d_recall"]["value"], 1.0)
        self.assertIsNone(rows["p95_ms"]["ok"])
        self.assertIsNone(rows["vs_ref_acc3"]["ok"])
        self.assertEqual(result["verdict"], "잴 수 없는 자 있음")
        self.assertEqual(result["protocol_sha256"], self.fx.protocol_sha)

    def test_refuses_set_hash_mismatch(self):
        # blind pred 의 summary 는 blind 해시인데 다른 셋 파일을 주면 거부
        other = os.path.join(self.fx.tmp, "other.jsonl")
        write_jsonl(other, [{"id": "x"}])
        code, _, err = self.fx.release(other, self.fx.blind_pred)
        self.assertEqual(code, 2)
        self.assertIn("set_sha256", err)

    def test_refuses_temperature_only_calibration(self):
        # 문턱 없는 calibration 은 KeyError 로 죽지 않고 거부
        self.fx.calibrate("--fitted-on", self.fx.cal_set, "--temperature-only")
        code, _, err = self.fx.release(self.fx.blind_set, self.fx.blind_pred)
        self.assertEqual(code, 2)
        self.assertIn("thresholds", err)

    def test_refuses_ref_with_other_gold(self):
        # 기준 모델 pred 의 쌍은 같은데 정답이 다르면(칸 수 · --public-neutral 다름) 거부 — 붓스트랩이 조용히 틀린다
        rows = make_pred()
        rows[1]["label"] = (rows[1]["label"] + 1) % 4
        ref_dir = os.path.join(self.fx.tmp, "ref")
        os.makedirs(ref_dir)
        write_jsonl(os.path.join(ref_dir, "pred.jsonl"), rows)
        code, _, err = quiet(gates.main, ["release", "--protocol", self.fx.protocol, "--set", self.fx.blind_set,
                                          "--pred", self.fx.blind_pred, "--calibration", self.fx.calibration,
                                          "--ref-pred", ref_dir, "--out", os.path.join(self.fx.tmp, "gates.json")])
        self.assertEqual(code, 2)
        self.assertIn("정답", err)

    def test_vs_ref_same_model_is_zero(self):
        code, _, _ = quiet(gates.main, ["release", "--protocol", self.fx.protocol, "--set", self.fx.blind_set,
                                        "--pred", self.fx.blind_pred, "--calibration", self.fx.calibration,
                                        "--ref-pred", self.fx.blind_pred, "--p95-ms", "400",
                                        "--out", os.path.join(self.fx.tmp, "gates.json")])
        self.assertEqual(code, 0)
        rows = {r["ruler"]: r for r in cmp.read_json(os.path.join(self.fx.tmp, "gates.json"))["rulers"]}
        self.assertEqual(rows["vs_ref_acc3"]["value"]["diff"], 0.0)
        self.assertTrue(rows["vs_ref_acc3"]["ok"])
        self.assertTrue(rows["p95_ms"]["ok"])


class G1ReleaseRegressionTest(unittest.TestCase):
    """protocol-g1 규약으로 돌리면 자 차례 · 기준 글이 g2 손질 전과 같다."""

    def test_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = GateFixture(tmp)
            fx.calibrate("--fitted-on", fx.cal_set)
            code, out, _ = fx.release(fx.blind_set, fx.blind_pred, "--ref-pred", fx.blind_pred, "--p95-ms", "400")
            self.assertEqual(code, 0, out)
            result, rows = fx.rulers()
        self.assertEqual([r["ruler"] for r in result["rulers"]],
                         ["acc3", "b_recall", "sure_false_support", "unsure", "d_recall", "d_sure_precision",
                          "p95_ms", "vs_ref_acc3"])
        self.assertEqual(rows["acc3"]["need"], ">= 0.85")
        self.assertEqual(rows["unsure"]["need"], "<= 0.3")
        self.assertEqual(rows["vs_ref_acc3"]["need"], "차 95% 아래끝 >= -0.02")
        self.assertEqual(result["verdict"], "통과")


def b_blind_ref():
    """v1.1.0 흉내 : 정답 b 를 다 c 로 놓친다 (확신 높음). 나머지는 make_pred 그대로."""
    rows = make_pred()
    for r in rows:
        if r["label"] == 1:
            r["logits"] = [0.0, 0.0, 4.0, 0.0]
            r["pred"] = "c"
    return rows


class G2ReleaseTest(unittest.TestCase):
    """protocol-g2 release_gate — b 재현 ≥ .80 · vs_ref 목록 · 보류 ≤ ref + 차. KeyError 없이 돈다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = GateFixture(self._tmp.name, PROTOCOL_G2)
        self.fx.calibrate("--fitted-on", self.fx.cal_set)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pass_against_weaker_ref(self):
        # 잴 판 b 재현 1.0 · ref 0.0 → b 차 +1.0 (아래끝 > 0). a·c·d 재현은 같아 차 0 · acc3 은 잴 판이 낫다
        ref = self.fx.ref_dir(b_blind_ref())
        code, out, _ = self.fx.release(self.fx.blind_set, self.fx.blind_pred, "--ref-pred", ref,
                                       "--ref-calibration", self.fx.calibration, "--p95-ms", "400")
        self.assertEqual(code, 0, out)
        result, rows = self.fx.rulers()
        self.assertEqual([r["ruler"] for r in result["rulers"]],
                         ["b_recall", "sure_false_support", "unsure", "p95_ms", "vs_ref_b_recall", "vs_ref_acc3",
                          "vs_ref_a_recall", "vs_ref_c_recall", "vs_ref_d_recall"])
        self.assertAlmostEqual(rows["b_recall"]["value"], 1.0)
        self.assertEqual(rows["b_recall"]["need"], ">= 0.8")
        self.assertAlmostEqual(rows["vs_ref_b_recall"]["value"]["diff"], 1.0)
        self.assertEqual(rows["vs_ref_b_recall"]["need"], "차 95% 아래끝 > 0.0")
        self.assertEqual(rows["vs_ref_a_recall"]["value"]["diff"], 0.0)
        self.assertTrue(all(r["ok"] for r in result["rulers"]), result["rulers"])
        self.assertEqual(result["verdict"], "통과")

    def test_same_model_ref_fails_b(self):
        # ref 가 잴 판과 같으면 b 차 0 → 「아래끝 > 0」 미달
        code, _, _ = self.fx.release(self.fx.blind_set, self.fx.blind_pred, "--ref-pred", self.fx.blind_pred,
                                     "--ref-calibration", self.fx.calibration, "--p95-ms", "400")
        self.assertEqual(code, 0)
        result, rows = self.fx.rulers()
        self.assertIs(rows["vs_ref_b_recall"]["ok"], False)
        self.assertIs(rows["vs_ref_acc3"]["ok"], True)
        self.assertIs(rows["unsure"]["ok"], True)
        self.assertEqual(result["verdict"], "미달")

    def test_b_recall_below_min_fails(self):
        # 잴 판이 b 를 다 놓치면 b 재현 0 < .80
        weak = self.fx.eval_dir("eval-blind-weak", b_blind_ref(), self.fx.blind_set)
        code, _, _ = self.fx.release(self.fx.blind_set, weak)
        self.assertEqual(code, 0)
        result, rows = self.fx.rulers()
        self.assertIs(rows["b_recall"]["ok"], False)
        self.assertEqual(result["verdict"], "미달")

    def test_unsure_over_ref_fails(self):
        # ref 보류 0 (확신 높음) · 잴 판은 로짓을 낮춰 다 보류 → 보류 차 > .03
        ref = self.fx.ref_dir(make_pred())
        flat = make_pred()
        for r in flat:
            r["logits"] = [x / 40 for x in r["logits"]]
        soft = self.fx.eval_dir("eval-blind-soft", flat, self.fx.blind_set)
        code, _, _ = self.fx.release(self.fx.blind_set, soft, "--ref-pred", ref,
                                     "--ref-calibration", self.fx.calibration)
        self.assertEqual(code, 0)
        _, rows = self.fx.rulers()
        self.assertIs(rows["unsure"]["ok"], False)
        self.assertIn("+ 0.03", rows["unsure"]["need"])

    def test_without_ref_is_unmeasured(self):
        # ref 없이 돌려도 KeyError 없이 「잴 수 없는 자 있음」 — vs_ref · 보류 자는 None
        code, out, _ = self.fx.release(self.fx.blind_set, self.fx.blind_pred)
        self.assertEqual(code, 0, out)
        result, rows = self.fx.rulers()
        self.assertIsNone(rows["vs_ref_b_recall"]["ok"])
        self.assertIsNone(rows["unsure"]["ok"])
        self.assertEqual(result["verdict"], "잴 수 없는 자 있음")

    def test_ref_calibration_on_measured_set_refuses(self):
        bad = os.path.join(self.fx.tmp, "ref-cal.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"temperature": 1.0, "thresholds": {"nli_sure": 0.9, "nli_sure_support": 0.9},
                       "calib_set_sha256": cmp.sha256_file(self.fx.blind_set)}, f)
        code, _, err = self.fx.release(self.fx.blind_set, self.fx.blind_pred, "--ref-pred", self.fx.blind_pred,
                                       "--ref-calibration", bad)
        self.assertEqual(code, 2)
        self.assertIn("calibration", err)


PROTOCOL_G3 = os.path.join(helpers.ROOT, "measure", "protocol-g3.draft.json")


def lang_of(i):
    """쌍 i 의 언어 — 넷씩 ko · en 번갈아. b(i%4==1) 는 ko 가 묶음 g1, en 이 묶음 g5 에 모인다."""
    return "ko" if (i // 4) % 2 == 0 else "en"


def ko_b_missed():
    """ko b 만 다 c 로 놓친다 (확신 높음). 나머지는 make_pred 그대로."""
    rows = make_pred()
    for i, r in enumerate(rows):
        if r["label"] == 1 and lang_of(i) == "ko":
            r["logits"] = [0.0, 0.0, 4.0, 0.0]
            r["pred"] = "c"
    return rows


class G3ReleaseTest(unittest.TestCase):
    """protocol-g3 초안 release_gate — 절대 자 · vs_ref 에 언어 걸러 쓰기 (설계 2-4)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = GateFixture(self._tmp.name, PROTOCOL_G3)
        self.fx.calibrate("--fitted-on", self.fx.cal_set)
        # blind 셋에 lang 칸을 달고 판을 새로 만든다 (summary 의 셋 해시가 새 파일을 가리키게)
        rows = make_pred()
        write_jsonl(self.fx.blind_set, [{"id": r["id"], "label": "abcd"[r["label"]], "group": f"g{i % 8}",
                                         "lang": lang_of(i), "blind": 1} for i, r in enumerate(rows)])
        self.pred = self.fx.eval_dir("eval-blind-lang", rows, self.fx.blind_set)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ko_b_pass_against_weaker_ref(self):
        # 잴 판 b 다 맞힘 · ref 는 b 를 다 놓침 → ko · en b 차 +1.0 · 절대 자 b ≥ .80 · ko b ≥ .70 통과
        ref = self.fx.ref_dir(b_blind_ref())
        code, out, _ = self.fx.release(self.fx.blind_set, self.pred, "--ref-pred", ref,
                                       "--ref-calibration", self.fx.calibration, "--p95-ms", "400")
        self.assertEqual(code, 0, out)
        result, rows = self.fx.rulers()
        self.assertEqual([r["ruler"] for r in result["rulers"]],
                         ["b_recall", "b_recall@ko", "sure_false_support", "unsure", "p95_ms", "vs_ref_b_recall@ko",
                          "vs_ref_b_recall", "vs_ref_b_recall@en", "vs_ref_acc3", "vs_ref_a_recall",
                          "vs_ref_c_recall", "vs_ref_d_recall"])
        self.assertEqual(rows["b_recall@ko"]["need"], ">= 0.7")
        self.assertEqual(rows["vs_ref_b_recall@ko"]["value"]["pairs"], 10)
        self.assertAlmostEqual(rows["vs_ref_b_recall@ko"]["value"]["diff"], 1.0)
        self.assertEqual(rows["vs_ref_b_recall@ko"]["need"], "차 95% 아래끝 > 0.0")
        self.assertEqual(rows["vs_ref_b_recall@en"]["need"], "차 95% 아래끝 >= -0.03")
        self.assertEqual(result["verdict"], "통과")

    def test_ko_b_missed_fails_ko_only(self):
        # 잴 판이 ko b 만 놓침 → b_recall@ko 0 · vs_ref ko 차 −1 미달 · en b 차 0 은 통과
        weak = self.fx.eval_dir("eval-blind-weak", ko_b_missed(), self.fx.blind_set)
        ref = self.fx.ref_dir(make_pred())
        code, _, _ = self.fx.release(self.fx.blind_set, weak, "--ref-pred", ref,
                                     "--ref-calibration", self.fx.calibration)
        self.assertEqual(code, 0)
        result, rows = self.fx.rulers()
        self.assertEqual(rows["b_recall@ko"]["value"], 0.0)
        self.assertIs(rows["b_recall@ko"]["ok"], False)
        self.assertIs(rows["vs_ref_b_recall@ko"]["ok"], False)
        self.assertIs(rows["vs_ref_b_recall@en"]["ok"], True)
        self.assertEqual(result["verdict"], "미달")

    def test_without_ref_names_slices(self):
        code, _, _ = self.fx.release(self.fx.blind_set, self.pred)
        self.assertEqual(code, 0)
        _, rows = self.fx.rulers()
        self.assertIsNone(rows["vs_ref_b_recall@ko"]["ok"])
        self.assertIs(rows["b_recall@ko"]["ok"], True)

    def test_recall_min_slice_without_pairs_refuses(self):
        # ko 쌍이 하나도 없는 셋에 b_recall@ko 자 → 조용히 None 이 아니라 거부
        set_rows = {"p1": {"lang": "en"}, "p2": {"lang": "en"}}
        table = {"ids": ["p1", "p2"], "gold": [cmp.CONTRA, cmp.SUPPORT], "counts": [[0, 1, 0, 0], [1, 0, 0, 0]], "seeds": 1}
        with self.assertRaises(cmp.Refuse):
            gates.recall_min_rows(set_rows, table, [0.8, {"metric": "b_recall", "lang": "ko", "min": 0.7}])
        rows = gates.recall_min_rows(set_rows, table, [0.8, {"metric": "b_recall", "lang": "en", "min": 0.7}])
        self.assertEqual([(r[0], r[1]) for r in rows], [("b_recall", 1.0), ("b_recall@en", 1.0)])

    def test_recall_min_old_number_form(self):
        self.assertEqual(gates.recall_min_specs(0.8), [{"metric": "b_recall", "min": 0.8}])
        self.assertEqual(gates.recall_min_specs({"metric": "b_recall", "lang": "ko", "min": 0.7}),
                         [{"metric": "b_recall", "lang": "ko", "min": 0.7}])


class PreregSliceTest(unittest.TestCase):
    """prereg.criteria_from_gate — g3 걸러 쓴 자를 이름째 풀어 적고, g2 박제는 그대로."""

    def gate(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["release_gate"]

    def test_g3_unrolled(self):
        import prereg
        c = prereg.criteria_from_gate(self.gate(PROTOCOL_G3))
        self.assertEqual(c["contradict_recall_min"], 0.80)
        self.assertEqual(c["recall_min_by_slice"], {"b_recall@ko": 0.70})
        self.assertEqual(c["vs_ref_rulers"][0], {"ruler": "vs_ref_b_recall@ko", "need": "차 95% 아래끝 > 0.0"})
        self.assertIn({"ruler": "vs_ref_b_recall@en", "need": "차 95% 아래끝 >= -0.03"}, c["vs_ref_rulers"])
        self.assertEqual(c["ref_model"], "v1.1.0")

    def test_g2_unchanged(self):
        import prereg
        c = prereg.criteria_from_gate(self.gate(PROTOCOL_G2))
        self.assertEqual(sorted(c), sorted(prereg.GATE_TO_CRITERIA_G2.values()))
        self.assertEqual(c["contradict_recall_min"], 0.80)

    def test_number_list_without_slice_unrolled(self):
        import prereg
        for value in ([0.80], {"metric": "b_recall", "min": 0.80}, [{"metric": "b_recall", "min": 0.80}]):
            gate = dict(self.gate(PROTOCOL_G2), b_recall_min=value)
            c = prereg.criteria_from_gate(gate)
            self.assertEqual(c["contradict_recall_min"], 0.80, value)


@unittest.skipUnless(os.path.isfile(os.path.join(R2, "r2-02", "calibration-devr2.json")) and os.path.isfile(DEV_R2),
                     "옛 r2 판 calibration · dev-r2 가 없다 (git 밖)")
class OldCalibrationTest(unittest.TestCase):
    def test_old_fitted_on_counts_as_calib_set(self):
        # 옛 calibration 은 calib_set_sha256 이 없고 fitted_on.sha256 만 있다 — 그것도 dev-r2 해시로 읽어 거부해야 한다
        c = cmp.read_json(os.path.join(R2, "r2-02", "calibration-devr2.json"))
        dev_sha = cmp.sha256_file(DEV_R2)
        if cmp.calib_sha(c) != dev_sha:
            self.skipTest("옛 calibration 이 다른 dev-r2 판으로 골라졌다")
        protocol, protocol_sha = cmp.read_protocol(PROTOCOL)
        run = {"path": "r2-02", "summary": {"set_sha256": dev_sha, "protocol_sha256": protocol_sha}}
        with self.assertRaises(cmp.Refuse):
            gates.check_release_hashes(protocol, protocol_sha, dev_sha, run, c)


def train_funcs(names, ns=None):
    """train_nli.py 에서 함수 몇 개만 떼어 온다 — torch 없이 돈다. ns 는 그 함수들이 쓰는 이름."""
    path = os.path.join(helpers.ROOT, "train", "train_nli.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(ns or {})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


class ReplayCountTest(unittest.TestCase):
    """train_nli.replay_count — torch 없이 그 함수 둘만 떼어 돌린다."""

    @classmethod
    def setUpClass(cls):
        cls.replay_count = staticmethod(train_funcs(("replay_count", "parse_ratio"))["replay_count"])

    def args(self, **kw):
        base = {"replay": "pool.jsonl", "mix_klue": "1:1", "replay_n": None}
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_ratio_unchanged_without_replay_n(self):
        self.assertEqual(self.replay_count(self.args(), 1000), 1000)
        self.assertEqual(self.replay_count(self.args(mix_klue="2:1"), 1000), 500)

    def test_replay_n_fixes_count(self):
        self.assertEqual(self.replay_count(self.args(replay_n=3433), 1000), 3433)
        self.assertEqual(self.replay_count(self.args(replay_n=3433), 5000), 3433)

    def test_no_replay_pool(self):
        self.assertEqual(self.replay_count(self.args(replay="", replay_n=10), 1000), 0)

    def test_old_namespace_without_field(self):
        ns = types.SimpleNamespace(replay="p", mix_klue="1:1")
        self.assertEqual(self.replay_count(ns, 10), 10)


class SeedAndRerunTest(unittest.TestCase):
    """compare.load_run 씨앗 읽기 · baseline 재실행 판 짝 찾기."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, *parts):
        folder = os.path.join(self.tmp, *parts)
        os.makedirs(folder)
        write_jsonl(os.path.join(folder, "pred.jsonl"), make_pred(8))
        return folder

    def test_seed_from_last_match(self):
        # 윗 경로에 「-s숫자」가 있어도 판 폴더 이름의 씨앗을 쓴다
        folder = self.run_dir("old-s9", "g1-B0-s2", "eval-dev-cmp")
        self.assertEqual(cmp.load_run(folder)["seed"], 2)

    def test_rerun_without_matching_seed_refuses(self):
        runs = [cmp.load_run(self.run_dir(f"g1-B0-s{s}", "eval")) for s in (1, 2)]
        rerun = cmp.load_run(self.run_dir("g1-B0-s7r", "eval"))
        set_rows = {f"q{i:03d}": {"id": f"q{i:03d}", "group": f"g{i}"} for i in range(8)}
        protocol = {"bootstrap": {"n": 20}}
        with self.assertRaises(cmp.Refuse):
            baseline.resolve(protocol, set_rows, runs, rerun)
        rerun_ok = cmp.load_run(self.run_dir("g1-B0-s2r", "eval"))
        self.assertIn("rerun", baseline.resolve(protocol, set_rows, runs, rerun_ok))


class BaselineLangTest(unittest.TestCase):
    """baseline ⑤ 언어별 b 줄 · 릴리스 pick 의 lang 거름 (g3) — lang 없는 g2 꼴도 그대로 돈다."""

    N = 48

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def set_rows(self, with_lang, mix=False):
        rows = {}
        for i in range(self.N):
            row = {"id": f"q{i:03d}", "label": "abcd"[i % 4], "group": f"g{i // 2}"}
            if with_lang:
                row["lang"] = "ko" if (i // 4) % 2 == 0 else "en"
            if mix and i % 12 == 11:
                row["lang"] = "mix"
            rows[row["id"]] = row
        return rows

    def make_runs(self, set_path, protocol_path, seeds=(1, 2, 3)):
        paths = []
        for seed in seeds:
            preds = []
            for i in range(self.N):
                label = i % 4
                guess = (label + 1) % 4 if (i * 3 + seed) % 5 == 0 else label
                logits = [0.0] * 4
                logits[guess] = 4.0
                preds.append({"id": f"q{i:03d}", "label": label, "logits": logits, "pred": "abcd"[guess]})
            folder = os.path.join(self.tmp, f"g3-B2-s{seed}", "eval-dev-cmp")
            os.makedirs(folder)
            write_jsonl(os.path.join(folder, "pred.jsonl"), preds)
            with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"set_sha256": cmp.sha256_file(set_path),
                           "protocol_sha256": cmp.sha256_file(protocol_path)}, f)
            paths.append(folder)
        return "B2=" + ",".join(paths)

    def write_case(self, with_lang, release_pick, mix=False):
        set_path = os.path.join(self.tmp, "dev-cmp.jsonl")
        write_jsonl(set_path, list(self.set_rows(with_lang, mix).values()))
        protocol_path = os.path.join(self.tmp, "protocol.json")
        with open(protocol_path, "w", encoding="utf-8") as f:
            json.dump({"bootstrap": {"n": 200}, "release_pick": release_pick}, f)
        return set_path, protocol_path, self.make_runs(set_path, protocol_path)

    def test_lang_b_matches_resolve_on_lang_subset(self):
        # 스크래치 lang_b_g3.py 의 셈(resolve 를 lang 으로 거른 쌍에) 과 같은 값이어야 한다
        set_path, protocol_path, spec = self.write_case(True, {"metric": "b_recall", "lang": "ko"})
        protocol = {"bootstrap": {"n": 200}}
        set_rows = cmp.read_set(set_path)
        _, runs = cmp.load_runs(spec)
        got = baseline.lang_b(protocol, set_rows, runs)
        self.assertEqual(list(got), ["ko", "en"])
        for lang in ("ko", "en"):
            sub = {pid: row for pid, row in set_rows.items() if row.get("lang") == lang}
            want = baseline.resolve(protocol, sub, runs, None)
            lv = want["level"]["b_recall"]
            self.assertEqual(got[lang]["b_level"], {k: lv[k] for k in ("mean", "low", "high", "half")})
            self.assertEqual(got[lang]["b_estimate"], want["estimate"]["b_recall"])
            self.assertEqual(got[lang]["b_pairs"], lv["pairs"])
            per_seed = [round(cmp.rate(cmp.pair_hits([r], list(sub)), "b_recall"), 4) for r in runs]
            self.assertEqual(got[lang]["b_per_seed"], per_seed)

    def test_lang_b_none_without_lang(self):
        set_path, _, spec = self.write_case(False, {"metric": "d_recall"})
        _, runs = cmp.load_runs(spec)
        self.assertIsNone(baseline.lang_b({"bootstrap": {"n": 50}}, cmp.read_set(set_path), runs))

    def test_run_g3_release_pick_uses_lang(self):
        set_path, protocol_path, spec = self.write_case(True, {"metric": "b_recall", "lang": "ko"})
        code, out, err = quiet(baseline.main, ["--protocol", protocol_path, "--set", set_path, "--runs", spec,
                                               "--out", os.path.join(self.tmp, "b.json")])
        self.assertEqual(code, 0, err)
        result = cmp.read_json(os.path.join(self.tmp, "b.json"))
        self.assertEqual(result["release_pick"]["metric"], "b_recall@ko")
        _, runs = cmp.load_runs(spec)
        want = cmp.pick_release(runs, list(cmp.read_set(set_path)), "b_recall", cmp.read_set(set_path),
                                {"metric": "b_recall", "lang": "ko"})
        self.assertEqual(result["release_pick"], want)
        self.assertEqual(set(result["lang_b"]), {"ko", "en"})
        self.assertIn("| 언어 b |", out)
        self.assertIn("가장 가까운 b_recall@ko", out)
        self.assertNotIn("d 재현 짝 반폭", out)

    def test_run_reports_skipped_mix_lang(self):
        set_path, protocol_path, spec = self.write_case(True, {"metric": "b_recall", "lang": "ko"}, mix=True)
        code, out, err = quiet(baseline.main, ["--protocol", protocol_path, "--set", set_path, "--runs", spec,
                                               "--out", os.path.join(self.tmp, "b.json")])
        self.assertEqual(code, 0, err)
        result = cmp.read_json(os.path.join(self.tmp, "b.json"))
        self.assertEqual(result["lang_b_skipped"], {"mix": self.N // 12})
        self.assertEqual(set(result["lang_b"]), {"ko", "en"})
        self.assertIn(f"- 언어별 b 에서 빠진 언어 : mix {self.N // 12} 쌍", out)

    def test_run_g2_shape_without_lang(self):
        set_path, protocol_path, spec = self.write_case(False, {"metric": "d_recall"})
        code, out, err = quiet(baseline.main, ["--protocol", protocol_path, "--set", set_path, "--runs", spec,
                                               "--out", os.path.join(self.tmp, "b.json")])
        self.assertEqual(code, 0, err)
        result = cmp.read_json(os.path.join(self.tmp, "b.json"))
        self.assertIsNone(result["lang_b"])
        self.assertEqual(result["release_pick"]["metric"], "d_recall")
        self.assertNotIn("| 언어 b |", out)
        self.assertIn("d_recall 씨앗 평균 짝 반폭 어림", out)


class ReplayNArgTest(unittest.TestCase):
    """--replay-n 의 type(positive_int) — 0 이하·정수 아님은 인자 단계에서 exit 2."""

    def setUp(self):
        positive_int = train_funcs(("positive_int",), {"argparse": argparse})["positive_int"]
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--replay-n", type=positive_int, default=None)

    def test_refuses_zero_and_negative(self):
        for bad in ("0", "-5", "x"):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    self.parser.parse_args(["--replay-n", bad])
                self.assertEqual(cm.exception.code, 2)

    def test_accepts_positive(self):
        self.assertEqual(self.parser.parse_args(["--replay-n", "3433"]).replay_n, 3433)


class DeterministicEnvTest(unittest.TestCase):
    """set_deterministic — 이미 있는 CUBLAS_WORKSPACE_CONFIG 는 안 덮는다 (torch 는 가짜)."""

    def run_it(self):
        fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=types.SimpleNamespace()),
                                           use_deterministic_algorithms=lambda *a, **k: None)
        ns = train_funcs(("set_deterministic",), {"os": os, "torch": fake_torch, "log": lambda msg: None})
        ns["set_deterministic"]()
        return fake_torch

    def test_keeps_existing_value(self):
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self.run_it()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_sets_default_when_missing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            fake_torch = self.run_it()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertTrue(fake_torch.backends.cudnn.deterministic)


if __name__ == "__main__":
    unittest.main()
