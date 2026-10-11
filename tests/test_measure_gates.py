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

    def __init__(self, tmp):
        self.tmp = tmp
        rows = make_pred()
        self.cal_set = os.path.join(tmp, "dev-cal-g1.jsonl")
        self.blind_set = os.path.join(tmp, "blind-g1.jsonl")
        write_jsonl(self.cal_set, [{"id": r["id"], "label": "abcd"[r["label"]], "group": f"g{i % 8}"}
                                   for i, r in enumerate(rows)])
        write_jsonl(self.blind_set, [{"id": r["id"], "label": "abcd"[r["label"]], "group": f"g{i % 8}", "blind": 1}
                                     for i, r in enumerate(rows)])
        # 실제 규약의 셋 해시 칸은 진짜 셋 해시다 — 시험은 그 칸을 비운 복사본을 쓴다 (빈 칸이면 셋 해시를 안 견줌)
        with open(PROTOCOL, encoding="utf-8") as f:
            protocol = json.load(f)
        protocol["sets"] = {name: "" for name in protocol.get("sets", {})}
        self.protocol = os.path.join(tmp, "protocol-g1.json")
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

    def release(self, set_path, pred):
        return quiet(gates.main, ["release", "--protocol", self.protocol, "--set", set_path, "--pred", pred,
                                  "--calibration", self.calibration, "--out", os.path.join(self.tmp, "gates.json")])


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
