"""measure/compare.py — 쌍 맞힘 · 묶음 붓스트랩 · McNemar · 관문 판정 · 해시 거부 (d 자료 층 설계 2-3).

손으로 만든 셋 : 쌍 40 = 묶음 5 × (d 4 · b 2 · a 1 · c 1). 두 조건 × 씨앗 3. 기대값은 각 시험 주석의 손셈.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest

import helpers

import compare as cmp
import gates

PROTOCOL = os.path.join(helpers.ROOT, "measure", "protocol-g1.json")
GROUPS = ["g1", "g2", "g3", "g4", "g5"]
R2 = os.path.join(os.path.expanduser("~"), ".judgemodel", "measure", "2026-10-11-r2", "loop")


def make_set():
    """묶음마다 d 넷 · b 둘 · a 하나 · c 하나. g1~g3 = mem·ko, g4~g5 = jira·en."""
    rows = []
    for gi, g in enumerate(GROUPS):
        domain, lang = ("mem", "ko") if gi < 3 else ("jira", "en")
        for k, label in enumerate([3, 3, 3, 3, 1, 1, 0, 2]):
            rows.append({"id": f"q{gi * 8 + k + 1:02d}", "label": "abcd"[label], "group": g,
                         "domain": domain, "lang": lang, "k": k})
    return rows


def base_pred(row, seed):
    """기준 : 묶음마다 d 앞 둘만 d, 뒤 둘은 c. 나머지는 다 맞힘."""
    label = "abcd".index(row["label"])
    if label == 3 and row["k"] >= 2:
        return "c"
    return row["label"]


def new_pred(row, seed):
    """새 판 : g1~g4 는 d 셋째까지 d, 넷째는 씨앗 1 만 d. g5 는 기준과 같다."""
    if row["group"] == "g5" or row["label"] != "d":
        return base_pred(row, seed)
    if row["k"] <= 2 or seed == 1:
        return "d"
    return "c"


def broken_pred(row, seed):
    """지킴 깨짐 : new_pred 에 더해 g1·g2 의 b 넷을 씨앗 셋 다 a 로 본다."""
    if row["label"] == "b" and row["group"] in ("g1", "g2"):
        return "a"
    return new_pred(row, seed)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pred_rows(set_rows, fn, seed):
    out = []
    for r in set_rows:
        letter = fn(r, seed)
        logits = [0.0, 0.0, 0.0, 0.0]
        logits["abcd".index(letter)] = 3.0
        out.append({"id": r["id"], "label": "abcd".index(r["label"]), "logits": logits,
                    "probs": [0.25] * 4, "pred": letter, "kind": "", "src": ""})
    return out


class Fixture:
    """임시 폴더에 셋 · 판(runs/g1-<조건>-s<씨앗>/eval-dev-cmp) · summary · eval-side 를 만든다."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.set_rows = make_set()
        self.set_path = os.path.join(tmp, "dev-cmp-g1.jsonl")
        write_jsonl(self.set_path, self.set_rows)
        self.set_sha = cmp.sha256_file(self.set_path)
        # 실제 규약의 셋 해시 칸은 진짜 셋 해시다 — 시험은 그 칸만 가짜 셋 해시로 바꾼 복사본을 쓴다
        with open(PROTOCOL, encoding="utf-8") as f:
            protocol = json.load(f)
        protocol["sets"] = {name: "" for name in protocol.get("sets", {})}
        protocol["sets"][protocol["compare_on"]] = self.set_sha
        self.protocol = os.path.join(tmp, "protocol-g1.json")
        with open(self.protocol, "w", encoding="utf-8", newline="\n") as f:
            json.dump(protocol, f, ensure_ascii=False, indent=1)
        self.protocol_sha = cmp.sha256_file(self.protocol)

    def runs(self, cond, fn, side=True, summary=None):
        paths = []
        for seed in (1, 2, 3):
            run_dir = os.path.join(self.tmp, "runs", f"g1-{cond}-s{seed}")
            eval_dir = os.path.join(run_dir, "eval-dev-cmp")
            os.makedirs(eval_dir, exist_ok=True)
            write_jsonl(os.path.join(eval_dir, "pred.jsonl"), pred_rows(self.set_rows, fn, seed))
            s = {"set_sha256": self.set_sha, "protocol_sha256": self.protocol_sha}
            s.update((summary or {}).get(seed, {}))
            with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(s, f)
            if side:
                with open(os.path.join(run_dir, "eval-side.txt"), "w", encoding="utf-8") as f:
                    f.write("k48 63fd061c 48 0.9167 0.9167 1 {}\n")
                    f.write("mnli-devm2000 0fbdd879 2000 0.8902 0.8675 98 {}\n")
                    f.write("klue-dev cdce9d34 3000 0.8895 0.874 153 {}\n")
            paths.append(eval_dir)
        return f"{cond}=" + ",".join(paths)

    def stats(self, base_fn, new_fn, side=True):
        protocol, _ = cmp.read_protocol(self.protocol)
        _, base = cmp.load_runs(self.runs("B0", base_fn, side))
        _, new = cmp.load_runs(self.runs("B1", new_fn, side))
        return protocol, cmp.compare(protocol, cmp.read_set(self.set_path), base, new)


def quiet(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args)
    return code, out.getvalue(), err.getvalue()


class PairHitsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rates_by_hand(self):
        # 기준 d 재현 = 묶음마다 2/4 → 0.5. 새 판 = (g1~g4 마다 3 + 1/3, g5 는 2) / 20 = 15.333/20 = 0.7667
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        _, new = cmp.load_runs(self.fx.runs("B1", new_pred))
        ids = [r["id"] for r in self.fx.set_rows]
        tb, tn = cmp.pair_hits(base, ids), cmp.pair_hits(new, ids)
        self.assertEqual(tb["seeds"], 3)
        self.assertAlmostEqual(cmp.rate(tb, "d_recall"), 0.5)
        self.assertAlmostEqual(cmp.rate(tn, "d_recall"), 46 / 60)
        # d → c 는 3칸으로 접으면 맞힘이라 둘 다 3칸 정확도 1.0
        self.assertAlmostEqual(cmp.rate(tb, "acc3"), 1.0)
        self.assertAlmostEqual(cmp.rate(tn, "acc3"), 1.0)
        self.assertEqual(cmp.false_support_mean(tn), 0)
        # 다수결 : g1~g4 넷째 d 는 씨앗 1/3 만 맞혀 틀림 → 새 판 다수결 맞힘 = 3×4 + 2 = 14
        self.assertEqual(sum(cmp.majority(tn, "d_recall")), 14)
        self.assertEqual(sum(cmp.majority(tb, "d_recall")), 10)

    def test_gold_differs_between_conditions_refuses(self):
        # 조건 사이 정답이 갈리면(예: 3칸 판은 d 를 c 로 접어 채점) 차례 짝짓기가 조용히 틀린다 — 거부
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        _, new = cmp.load_runs(self.fx.runs("B1", new_pred))
        for run in new:
            for row in run["preds"].values():
                if row["label"] == 3:
                    row["label"] = 2
        protocol, _ = cmp.read_protocol(self.fx.protocol)
        with self.assertRaises(cmp.Refuse):
            cmp.compare(protocol, cmp.read_set(self.fx.set_path), base, new)

    def test_missing_pair_refuses(self):
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        with self.assertRaises(cmp.Refuse):
            cmp.pair_hits(base, ["q01", "없는쌍"])


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_interval_matches_exact_enumeration(self):
        # 묶음마다 d 쌍이 4 라 차 = (g1~g4 를 뽑은 수 k × 4/3) / 20. k ~ 이항(5, 0.8)
        # 누적 : k≤1 0.0067 · k≤2 0.0579 → 2.5% 점은 k=2 (0.1333). k=5 가 0.672 부터 → 97.5% 점은 k=5 (0.3333)
        _, stats = self.fx.stats(base_pred, new_pred)
        d = stats["metrics"]["d_recall"]
        self.assertEqual(d["groups"], 5)
        self.assertEqual(d["pairs"], 20)
        self.assertAlmostEqual(d["diff"], 0.2667, places=4)
        self.assertAlmostEqual(d["low"], 0.1333, places=4)
        self.assertAlmostEqual(d["high"], 0.3333, places=4)

    def test_identical_groups_collapse(self):
        # 3칸 정확도는 두 조건 다 1.0 → 어느 재추출이든 차 0
        _, stats = self.fx.stats(base_pred, new_pred)
        a = stats["metrics"]["acc3"]
        self.assertEqual((a["diff"], a["low"], a["high"]), (0.0, 0.0, 0.0))

    def test_fixed_rng_is_repeatable(self):
        _, s1 = self.fx.stats(base_pred, new_pred)
        _, s2 = self.fx.stats(base_pred, new_pred)
        self.assertEqual(s1["metrics"]["d_recall"], s2["metrics"]["d_recall"])


class McNemarTest(unittest.TestCase):
    def test_exact_by_hand(self):
        # b=1, c=6 → p = 2 × (C(7,0) + C(7,1)) / 2^7 = 16/128 = 0.125
        base = [True] + [False] * 6 + [True] * 3
        new = [False] + [True] * 6 + [True] * 3
        self.assertEqual(cmp.mcnemar_exact(base, new), {"b": 1, "c": 6, "p": 0.125})

    def test_no_discordant(self):
        self.assertEqual(cmp.mcnemar_exact([True, False], [True, False])["p"], 1.0)

    def test_fixture_d(self):
        # 다수결 맞힘 : 기준 10 · 새 판 14, 기준만 맞힘 0 → b=0, c=4 → p = 2/16 = 0.125
        with tempfile.TemporaryDirectory() as tmp:
            _, stats = Fixture(tmp).stats(base_pred, new_pred)
        self.assertEqual(stats["metrics"]["d_recall"]["mcnemar"], {"b": 0, "c": 4, "p": 0.125})


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_gains(self):
        protocol, stats = self.fx.stats(base_pred, new_pred)
        v = cmp.verdict(protocol, stats)
        self.assertEqual(v["verdict"], "먹힘")
        self.assertTrue(all(g["ok"] for g in v["guards"]))
        self.assertEqual(v["alarms"], [])

    def test_same(self):
        protocol, stats = self.fx.stats(base_pred, base_pred)
        self.assertEqual(cmp.verdict(protocol, stats)["verdict"], "같다")

    def test_guard_broken(self):
        # b 재현 10/10 → 6/10 (차 −0.4) · 거짓 지지 0 → 4 · 3칸 1.0 → 0.9
        protocol, stats = self.fx.stats(base_pred, broken_pred)
        self.assertAlmostEqual(stats["metrics"]["b_recall"]["diff"], -0.4)
        self.assertAlmostEqual(stats["metrics"]["acc3"]["new"], 0.9)
        self.assertEqual(stats["false_support"], {"base": 0.0, "new": 4.0})
        v = cmp.verdict(protocol, stats)
        self.assertEqual(v["verdict"], "지킴 깨짐")
        broken = {g["metric"] for g in v["guards"] if g["ok"] is False}
        self.assertEqual(broken, {"acc3", "b_recall", "false_support"})

    def test_hold_without_side(self):
        protocol, stats = self.fx.stats(base_pred, new_pred, side=False)
        self.assertEqual(cmp.verdict(protocol, stats)["verdict"], "보류")

    def test_by_domain_lang(self):
        # mem·ko (g1~g3) : 0.5 → 10/12 · jira·en (g4~g5) : 0.5 → (10/3 + 2)/8
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        _, new = cmp.load_runs(self.fx.runs("B1", new_pred))
        ids = [r["id"] for r in self.fx.set_rows]
        rows = cmp.by_table(cmp.read_set(self.fx.set_path), ids, cmp.pair_hits(base, ids), cmp.pair_hits(new, ids),
                            ["domain", "lang"])
        got = {r["key"]: r["d_recall"]["new"] for r in rows}
        self.assertAlmostEqual(got["mem · ko"], 10 / 12)
        self.assertAlmostEqual(got["jira · en"], (10 / 3 + 2) / 8)

    def test_release_pick_closest_to_mean(self):
        # 새 판 씨앗별 d 재현 0.9 · 0.7 · 0.7 (평균 0.7667) → 가장 가까운 씨앗 2
        _, stats = self.fx.stats(base_pred, new_pred)
        self.assertIn("g1-B1-s2", stats["release_pick"]["pick"].replace("\\", "/"))


class CliHashTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)
        self.base = self.fx.runs("B0", base_pred)

    def tearDown(self):
        self._tmp.cleanup()

    def args(self, new):
        return ["--protocol", self.fx.protocol, "--set", self.fx.set_path, "--base", self.base, "--new", new,
                "--by", "domain,lang", "--out", os.path.join(self.fx.tmp, "out.json")]

    def test_ok_writes_json(self):
        code, out, _ = quiet(cmp.main, self.args(self.fx.runs("B1", new_pred)))
        self.assertEqual(code, 0)
        self.assertIn("먹힘", out)
        result = cmp.read_json(os.path.join(self.fx.tmp, "out.json"))
        self.assertEqual(result["protocol_sha256"], self.fx.protocol_sha)
        self.assertEqual(result["set_sha256"], self.fx.set_sha)
        self.assertEqual(len(result["by"]), 2)

    def test_set_hash_mismatch_exit2(self):
        new = self.fx.runs("B1", new_pred, summary={2: {"set_sha256": "0" * 64}})
        code, _, err = quiet(cmp.main, self.args(new))
        self.assertEqual(code, 2)
        self.assertIn("set_sha256", err)

    def test_protocol_hash_mismatch_exit2(self):
        new = self.fx.runs("B1", new_pred, summary={3: {"protocol_sha256": None}})
        code, _, err = quiet(cmp.main, self.args(new))
        self.assertEqual(code, 2)
        self.assertIn("protocol_sha256", err)

    def test_calibration_on_measured_set_exit2(self):
        new = self.fx.runs("B1", new_pred)
        run_dir = os.path.join(self.fx.tmp, "runs", "g1-B1-s1")
        with open(os.path.join(run_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"temperature": 1.0, "calib_set_sha256": self.fx.set_sha}, f)
        code, _, err = quiet(cmp.main, self.args(new))
        self.assertEqual(code, 2)
        self.assertIn("calibration", err)

    def test_gates_pair_same_protocol(self):
        code, out, _ = quiet(gates.main, ["pair"] + self.args(self.fx.runs("B1", new_pred)))
        self.assertEqual(code, 0)
        self.assertIn("먹힘", out)


@unittest.skipUnless(os.path.isfile(os.path.join(R2, "r2-02", "diag.json")), "옛 r2 판 pred 가 없다 (git 밖)")
class OldRunRecountTest(unittest.TestCase):
    """옛 r2-02 판 pred 를 compare 로 다시 세면 loopdiag 가 낸 diag.json 값과 같다."""

    def test_recount_equals_diag(self):
        diag = cmp.read_json(os.path.join(R2, "r2-02", "diag.json"))
        run = cmp.load_run(os.path.join(R2, "r2-02", "pred-devr2.jsonl"))
        table = cmp.pair_hits([run], list(run["preds"]))
        self.assertEqual(len(table["ids"]), diag["pairs"])
        self.assertAlmostEqual(cmp.rate(table, "acc3"), diag["acc3_hits"] / diag["pairs"])
        self.assertEqual(round(cmp.rate(table, "acc3"), 4), diag["acc3"])
        self.assertEqual(round(cmp.rate(table, "b_recall"), 4), diag["b_recall"])
        self.assertEqual(round(cmp.rate(table, "d_recall"), 4), diag["d_recall"])
        self.assertEqual(cmp.false_support_mean(table), diag["false_support_no_threshold"])

    def test_legacy_gates_count(self):
        # 고리 기록(loop.md) : r2-02 는 dev 관문 3/8
        diag = cmp.read_json(os.path.join(R2, "r2-02", "diag.json"))
        dev, side = gates.legacy_counts(diag, None)
        self.assertEqual(sum(dev), 3)
        self.assertIsNone(side)


if __name__ == "__main__":
    unittest.main()
