"""measure/compare.py — 쌍 맞힘 · 묶음 붓스트랩 · McNemar · 관문 판정 · 해시 거부 (d 자료 층 설계 2-3).

손으로 만든 셋 : 쌍 40 = 묶음 5 × (d 4 · b 2 · a 1 · c 1). 두 조건 × 씨앗 3. 기대값은 각 시험 주석의 손셈.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

import helpers

import compare as cmp
import gates

PROTOCOL = os.path.join(helpers.ROOT, "measure", "protocol-g1.json")
PROTOCOL_G2 = os.path.join(helpers.ROOT, "measure", "protocol-g2.json")
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

    def __init__(self, tmp, protocol_path=PROTOCOL):
        self.tmp = tmp
        self.set_rows = make_set()
        self.set_path = os.path.join(tmp, "dev-cmp-g1.jsonl")
        write_jsonl(self.set_path, self.set_rows)
        self.set_sha = cmp.sha256_file(self.set_path)
        # 실제 규약의 셋 해시 칸은 진짜 셋 해시다 — 시험은 그 칸만 가짜 셋 해시로 바꾼 복사본을 쓴다
        with open(protocol_path, encoding="utf-8") as f:
            protocol = json.load(f)
        protocol["sets"] = {name: "" for name in protocol.get("sets", {})}
        protocol["sets"][protocol["compare_on"]] = self.set_sha
        self.protocol = os.path.join(tmp, os.path.basename(protocol_path))
        with open(self.protocol, "w", encoding="utf-8", newline="\n") as f:
            json.dump(protocol, f, ensure_ascii=False, indent=1)
        self.protocol_sha = cmp.sha256_file(self.protocol)

    def runs(self, cond, fn, side=True, summary=None, seeds=(1, 2, 3)):
        paths = []
        for seed in seeds:
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

    def stats(self, base_fn, new_fn, side=True, seeds=(1, 2, 3)):
        protocol, _ = cmp.read_protocol(self.protocol)
        _, base = cmp.load_runs(self.runs("B0", base_fn, side, seeds=seeds))
        _, new = cmp.load_runs(self.runs("B1", new_fn, side, seeds=seeds))
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

    def test_by_table_has_a_c_old_values_same(self):
        # K2 설계 ⑧ : --by 표에 a · c 재현 칸이 붙고, 옛 d · b · 3칸 값은 rate 로 따로 센 값과 같다
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        _, new = cmp.load_runs(self.fx.runs("B1", new_pred))
        ids = [r["id"] for r in self.fx.set_rows]
        set_rows = cmp.read_set(self.fx.set_path)
        rows = cmp.by_table(set_rows, ids, cmp.pair_hits(base, ids), cmp.pair_hits(new, ids), ["domain", "lang"])
        by = {r["key"]: r for r in rows}
        self.assertAlmostEqual(by["mem · ko"]["d_recall"]["new"], 10 / 12)
        new_table = cmp.pair_hits(new, ids)
        for key, line in by.items():
            pos = [i for i, pid in enumerate(ids) if f"{set_rows[pid]['domain']} · {set_rows[pid]['lang']}" == key]
            sub = cmp.sub_table(new_table, pos)
            for metric in ("d_recall", "b_recall", "acc3", "a_recall", "c_recall"):
                self.assertEqual(line[metric]["new"], cmp.rate(sub, metric), (key, metric))
            self.assertEqual(line["c_pairs"], sum(1 for g in sub["gold"] if g == cmp.UNRELATED))
            self.assertEqual(line["a_pairs"], sum(1 for g in sub["gold"] if g == cmp.SUPPORT))
        protocol, stats = self.fx.stats(base_pred, new_pred)
        stats.pop("_tables")
        result = {"base": "B0", "new": "B1", "protocol_sha256": "0" * 64, "set_sha256": "0" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats), "by": rows}
        md = cmp.to_markdown(result)
        self.assertIn("c 재현", md)
        self.assertIn("a 재현", md)

    def test_by_table_weak_cells_marked_verdict_same(self):
        # K5 설계 3-4 : 정답 칸 쌍이 weak_min 미만이면 weak 에 적고 표에 「약함」 칸 — 판정은 안 바뀐다
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred))
        _, new = cmp.load_runs(self.fx.runs("B1", new_pred))
        ids = [r["id"] for r in self.fx.set_rows]
        set_rows = cmp.read_set(self.fx.set_path)
        hb, hn = cmp.pair_hits(base, ids), cmp.pair_hits(new, ids)
        rows = cmp.by_table(set_rows, ids, hb, hn, ["domain"])
        for line in rows:
            want = {c: line[f"{c}_pairs"] for c in "abcd" if 0 < line[f"{c}_pairs"] < 30}
            self.assertEqual(line["weak"], want)
        self.assertTrue(any(line["weak"] for line in rows))
        self.assertTrue(all(line["weak"] == {} for line in cmp.by_table(set_rows, ids, hb, hn, ["domain"], weak_min=1)))
        protocol, stats = self.fx.stats(base_pred, new_pred)
        stats.pop("_tables")
        result = {"base": "B0", "new": "B1", "protocol_sha256": "0" * 64, "set_sha256": "0" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats), "by": rows}
        md = cmp.to_markdown(result)
        self.assertIn("| 약함 |", md)
        self.assertIn(cmp.weak_mark(rows[0]["weak"]), md)
        self.assertEqual(cmp.weak_mark({"d": 4, "b": 19}), "b 19 · d 4")
        self.assertEqual(cmp.weak_mark(None), "—")                    # 옛 결과 파일 줄

    def test_release_pick_closest_to_mean(self):
        # 새 판 씨앗별 d 재현 0.9 · 0.7 · 0.7 (평균 0.7667) → 가장 가까운 씨앗 2
        _, stats = self.fx.stats(base_pred, new_pred)
        self.assertIn("g1-B1-s2", stats["release_pick"]["pick"].replace("\\", "/"))


def base_pred_all_right(row, seed):
    return row["label"]


def b_miss_pred(row, seed):
    """b 기준 판 : 묶음마다 b 둘째(k=5)를 c 로 놓친다. 나머지는 다 맞힘."""
    if row["label"] == "b" and row["k"] == 5:
        return "c"
    return row["label"]


class G2PairTest(unittest.TestCase):
    """protocol-g2 pair_gate — 목표 b 재현 · 지킴 a·c·d · ko b 경보 · 릴리스 판은 b 재현으로 고른다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name, PROTOCOL_G2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_b_gain_passes(self):
        # b 재현 0.5 → 1.0 (묶음마다 b 둘 중 하나 → 둘 다) · 아래끝 0.5 > 0 · a·c·d 그대로
        protocol, stats = self.fx.stats(b_miss_pred, base_pred_all_right)
        v = cmp.verdict(protocol, stats)
        self.assertEqual(v["target"]["metric"], "b_recall")
        self.assertAlmostEqual(v["target"]["diff"], 0.5)
        self.assertEqual({g["metric"] for g in v["guards"]},
                         {"acc3", "a_recall", "c_recall", "d_recall", "false_support", "side_acc3"})
        self.assertEqual(v["verdict"], "먹힘")
        self.assertEqual(v["alarms"], [])

    def test_b_same_is_same(self):
        protocol, stats = self.fx.stats(b_miss_pred, b_miss_pred)
        self.assertEqual(cmp.verdict(protocol, stats)["verdict"], "같다")

    def test_ko_b_alarm(self):
        # broken_pred : g1·g2(ko) 의 b 넷을 a 로 → ko b 재현 1.0 → 2/6 (차 −0.6667) · en b 는 그대로
        protocol, stats = self.fx.stats(base_pred, broken_pred)
        by_lang = {r["key"]: r for r in stats["by_lang"]}
        self.assertAlmostEqual(by_lang["ko"]["b_recall"]["diff"], -0.6667, places=4)
        self.assertEqual(by_lang["en"]["b_recall"]["diff"], 0.0)
        self.assertEqual(by_lang["ko"]["b_pairs"], 6)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertEqual(len(alarms), 1)
        self.assertIn("ko b 재현 -0.6667", alarms[0])

    def test_release_pick_uses_b_recall(self):
        protocol, stats = self.fx.stats(base_pred, new_pred)
        self.assertEqual(protocol["release_pick"]["metric"], "b_recall")
        self.assertEqual(stats["release_pick"]["metric"], "b_recall")

    def test_markdown_has_b_column(self):
        protocol, stats = self.fx.stats(base_pred, broken_pred)
        stats.pop("_tables")
        result = {"base": "B2", "new": "B4", "protocol_sha256": "0" * 64, "set_sha256": "0" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats)}
        self.assertIn("b 재현", cmp.to_markdown(result))


class G1AlarmRegressionTest(unittest.TestCase):
    """protocol-g1 의 lang_d_recall 경보는 그대로 d 재현을 본다."""

    def test_ko_d_alarm(self):
        # 거꾸로 견줌 : ko d 재현 10/12 → 0.5 (차 −0.3333) · b 는 그대로라 b 경보는 없다
        with tempfile.TemporaryDirectory() as tmp:
            protocol, stats = Fixture(tmp).stats(new_pred, base_pred)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertEqual(len(alarms), 1)
        self.assertIn("ko d 재현 -0.3333", alarms[0])
        self.assertEqual(stats["release_pick"]["metric"], "d_recall")


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


PROTOCOL_G3 = os.path.join(helpers.ROOT, "measure", "protocol-g3.draft.json")


def ko_b_fix_pred(row, seed):
    """K 흉내 : b_miss_pred 에서 ko(g1~g3) 의 b 만 다 맞힌다. en b 는 그대로 반만."""
    if row["label"] == "b" and row["lang"] == "ko":
        return "b"
    return b_miss_pred(row, seed)


def en_b_broken_pred(row, seed):
    """en(g4·g5) 의 b 를 다 a 로 놓친다. 나머지는 다 맞힘."""
    if row["label"] == "b" and row["lang"] == "en":
        return "a"
    return row["label"]


class G3SliceTest(unittest.TestCase):
    """protocol-g3 초안 — 목표 · 지킴 · 경보 · release_pick 을 언어 · 도메인으로 걸러 쓴다 (설계 2-2 · 5-4)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name, PROTOCOL_G3)

    def tearDown(self):
        self._tmp.cleanup()

    def test_target_ko_b_slice(self):
        # ko b : 묶음 g1~g3 마다 1/2 → 2/2 (차 +0.5, 묶음 다 같아 구간 [.5, .5]) · en b 는 0.5 그대로 (차 0)
        protocol, stats = self.fx.stats(b_miss_pred, ko_b_fix_pred)
        self.assertEqual(stats["metrics"]["b_recall@ko"]["pairs"], 6)
        self.assertEqual(stats["metrics"]["b_recall@ko"]["groups"], 3)
        self.assertAlmostEqual(stats["metrics"]["b_recall@ko"]["diff"], 0.5)
        self.assertAlmostEqual(stats["metrics"]["b_recall@ko"]["low"], 0.5)
        self.assertEqual(stats["metrics"]["b_recall@en"]["diff"], 0.0)
        self.assertAlmostEqual(stats["metrics"]["b_recall"]["diff"], 0.3)  # 쌍 전체 b 10 중 3 더 맞힘
        v = cmp.verdict(protocol, stats)
        self.assertEqual(v["target"]["metric"], "b_recall@ko")
        self.assertTrue(v["target"]["ok"])
        self.assertIn("b_recall@en", [g["metric"] for g in v["guards"]])
        self.assertEqual(v["verdict"], "먹힘")
        self.assertEqual(v["alarms"], [])

    def test_en_b_guard_slice_breaks(self):
        # en b 1.0 → 0.0 · 지킴 b_recall@en 아래끝 −1 < −.03 → 지킴 깨짐 (ko 목표는 못 채워도 지킴이 먼저)
        protocol, stats = self.fx.stats(base_pred_all_right, en_b_broken_pred)
        v = cmp.verdict(protocol, stats)
        guards = {g["metric"]: g for g in v["guards"]}
        self.assertIs(guards["b_recall@en"]["ok"], False)
        self.assertIn("b_recall@en 차 -1.0000", guards["b_recall@en"]["note"])
        self.assertEqual(v["verdict"], "지킴 깨짐")

    def test_mem_ko_alarm_slice(self):
        # broken_pred : g1·g2(mem·ko) 의 b 넷을 a 로 → mem·ko b 차 −0.6667 · ko a 는 그대로라 경보 하나
        protocol, stats = self.fx.stats(base_pred, broken_pred)
        self.assertAlmostEqual(stats["metrics"]["b_recall@mem·ko"]["diff"], -0.6667, places=4)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertEqual(len(alarms), 1)
        self.assertIn("b_recall@mem·ko 평균 차 -0.6667", alarms[0])

    def test_release_pick_slice(self):
        protocol, stats = self.fx.stats(base_pred, new_pred)
        self.assertEqual(stats["release_pick"]["metric"], "b_recall@ko")
        self.assertEqual(stats["release_pick"]["value"], 1.0)

    def test_empty_slice_refuses(self):
        protocol, stats = self.fx.stats(base_pred, new_pred)
        base, new = stats["_tables"]
        with self.assertRaises(cmp.Refuse):
            cmp.slice_compare(protocol, cmp.read_set(self.fx.set_path), base, new, {"metric": "b_recall", "lang": "jp"})

    def test_markdown_lists_slice_rulers(self):
        protocol, stats = self.fx.stats(b_miss_pred, ko_b_fix_pred)
        stats.pop("_tables")
        result = {"base": "B2", "new": "K", "protocol_sha256": "0" * 64, "set_sha256": "0" * 64,
                  "stats": stats, "verdict": cmp.verdict(protocol, stats)}
        md = cmp.to_markdown(result)
        self.assertIn("| 목표 b_recall@ko | O |", md)
        self.assertIn("| b_recall@mem·ko |", md)


PROTOCOL_G4_FROZEN = os.path.join(helpers.ROOT, "measure", "protocol-g4.json")
PROTOCOL_G5 = os.path.join(helpers.ROOT, "measure", "protocol-g5.draft.json")


def ko_c_miss_pred(row, seed):
    """ko(g1~g3) 의 c 를 다 a 로 놓친다. 나머지는 다 맞힘."""
    if row["label"] == "c" and row["lang"] == "ko":
        return "a"
    return row["label"]


def en_d_broken_pred(row, seed):
    """en(g4·g5) 의 d 를 다 c 로 쓸어 간다. 나머지는 다 맞힘."""
    if row["label"] == "d" and row["lang"] == "en":
        return "c"
    return row["label"]


class G5AlarmTest(unittest.TestCase):
    """protocol-g5 초안 — 양수 경보(c_recall@ko 평균 차 ≥ +.06)와 d_recall@en 경보 (K3 설계 2-2 · 5절 ⑤)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name, PROTOCOL_G5)

    def tearDown(self):
        self._tmp.cleanup()

    def test_draft_same_as_g4_but_named_rows(self):
        g5, _ = cmp.read_protocol(PROTOCOL_G5)
        g4, _ = cmp.read_protocol(PROTOCOL_G4_FROZEN)
        for key in ("sets", "compare_on", "calibrate_on", "release_pick", "bootstrap", "seeds", "majority", "release_gate"):
            self.assertEqual(g5[key], g4[key], key)
        for key in ("target", "guards"):
            self.assertEqual(g5["pair_gate"][key], g4["pair_gate"][key], key)
        self.assertEqual((g5["name"], g5["pair_gate"]["base"], g5["pair_gate"]["new"]), ("protocol-g5", "K", "K3"))
        self.assertEqual(g5["pair_gate"]["alarms"][:4], g4["pair_gate"]["alarms"])
        self.assertEqual(len(g5["pair_gate"]["alarms"]), 6)
        self.assertEqual(g5["train_sets"], dict(g4["train_sets"], **{"train-g4-d": ""}))
        with open(PROTOCOL_G5, encoding="utf-8") as f:
            self.assertNotIn("TODO", f.read())
        self.assertEqual(g5["pair_gate"]["report_only"][:len(g4["pair_gate"]["report_only"])], g4["pair_gate"]["report_only"])
        self.assertEqual(len(g5["pair_gate"]["report_only"]), len(g4["pair_gate"]["report_only"]) + 4)
        self.assertIn("K2 → K3", " ".join(g5["pair_gate"]["report_only"]))
        self.assertTrue(g5["pair_gate"]["verdicts"]["지킴 깨짐"].startswith("지킴 하나라도 미달 → 접는다"))
        self.assertNotIn("K2-h", g5["pair_gate"]["verdicts"]["지킴 깨짐"])

    def test_small_c_gain_fires_positive_alarm(self):
        # c 는 그대로(차 0) → 경보 c_recall@ko 평균 차 +0.0000 < 0.06 · en d 는 나아져 en d 경보 없음
        protocol, stats = self.fx.stats(base_pred, new_pred)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertIn("c_recall@ko 평균 차 +0.0000 < 0.06", alarms)
        self.assertFalse([a for a in alarms if a.startswith("d_recall@en")])

    def test_big_c_gain_no_alarm_but_en_d_alarm(self):
        # ko c 0 → 1 (차 +1, 경보 없음) · en d 1 → 0 (차 −1 < −.05 → 경보)
        protocol, stats = self.fx.stats(ko_c_miss_pred, en_d_broken_pred)
        self.assertEqual(stats["metrics"]["c_recall@ko"]["diff"], 1.0)
        self.assertEqual(stats["metrics"]["d_recall@en"]["diff"], -1.0)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertFalse([a for a in alarms if a.startswith("c_recall@ko")])
        self.assertIn("d_recall@en 평균 차 -1.0000 < -0.05", alarms)


PROTOCOL_G5_FROZEN = os.path.join(helpers.ROOT, "measure", "protocol-g5.json")
PROTOCOL_G6 = os.path.join(helpers.ROOT, "measure", "protocol-g6.draft.json")
PROTOCOL_G6_FROZEN = os.path.join(helpers.ROOT, "measure", "protocol-g6.json")
PROTOCOL_G7_DRAFT = os.path.join(helpers.ROOT, "measure", "protocol-g7.draft.json")


class G7ProtocolTest(unittest.TestCase):
    """protocol-g7 초안 — g6 에서 바뀌는 칸은 K5 설계 6-3 표뿐 (자는 g6 그대로)."""

    @staticmethod
    def assemble_g4():
        build = os.path.join(helpers.ROOT, "build")
        if build not in sys.path:
            sys.path.insert(0, build)
        import assemble_g4
        return assemble_g4

    def test_draft_same_rulers_as_g6(self):
        g7, _ = cmp.read_protocol(PROTOCOL_G7_DRAFT)
        g6, _ = cmp.read_protocol(PROTOCOL_G6_FROZEN)
        for key in ("sets", "compare_on", "calibrate_on", "release_pick", "bootstrap", "seeds", "majority", "release_gate",
                    "side", "group_key", "label_guide", "report_only_sets"):
            self.assertEqual(g7[key], g6[key], key)
        for key in ("target", "guards", "alarms", "slice_note", "base"):
            self.assertEqual(g7["pair_gate"][key], g6["pair_gate"][key], key)
        self.assertEqual((g7["name"], g7["pair_gate"]["new"]), ("protocol-g7", "K5"))
        self.assertEqual(g7["train_sets"], dict(g6["train_sets"], **{"train-g4-klb": ""}))
        self.assertEqual(g7["pair_gate"]["report_only"][:len(g6["pair_gate"]["report_only"])], g6["pair_gate"]["report_only"])
        self.assertIn("K4 → K5", " ".join(g7["pair_gate"]["report_only"]))
        cc = g7["cell_check"]
        self.assertEqual(cc["recipe"], g6["cell_check"]["recipe"] + ["train-g4-klb"])
        self.assertEqual(cc["base_recipe"], ["train-r2", "train-g1-arch", "train-g3-ko"])
        self.assertEqual((cc["max_shift"], cc["band_edges"], cc["band_min"], cc["weak"]), (0.2, [0.1, 0.2, 0.35], 100, 30))
        ag = self.assemble_g4()
        self.assertEqual((cc["max_shift"], cc["band_edges"], cc["band_min"], cc["weak"]),
                         (ag.MAX_SHIFT, list(ag.BAND_EDGES), ag.BAND_MIN, ag.WEAK_MIN))     # 규약 = 점검표 기본값
        ack = os.path.join(helpers.ROOT, "measure", cc["ack"]["file"])
        with open(ack, "rb") as f:
            self.assertEqual(cc["ack"]["sha256"], hashlib.sha256(f.read()).hexdigest())


PROTOCOL_G7_FROZEN = os.path.join(helpers.ROOT, "measure", "protocol-g7.json")
PROTOCOL_G8_DRAFT = os.path.join(helpers.ROOT, "measure", "protocol-g8.draft.json")
PROTOCOL_G8_FROZEN = os.path.join(helpers.ROOT, "measure", "protocol-g8.json")


class G8ProtocolTest(unittest.TestCase):
    """protocol-g8 — g7 에서 바뀌는 칸은 K6 설계 6-1 표뿐 (자는 g7 그대로)."""

    def test_draft_and_frozen_same_rulers_as_g7(self):
        g7, _ = cmp.read_protocol(PROTOCOL_G7_FROZEN)
        for path in (PROTOCOL_G8_DRAFT, PROTOCOL_G8_FROZEN):
            g8, _ = cmp.read_protocol(path)
            for key in ("sets", "compare_on", "calibrate_on", "release_pick", "bootstrap", "seeds", "majority", "release_gate",
                        "side", "group_key", "label_guide", "report_only_sets", "seed_rerun", "ckpt"):
                self.assertEqual(g8[key], g7[key], key)
            for key in ("target", "guards", "alarms", "alarms_note", "slice_note", "base"):
                self.assertEqual(g8["pair_gate"][key], g7["pair_gate"][key], key)
            self.assertEqual((g8["name"], g8["pair_gate"]["new"]), ("protocol-g8", "K6"))
            self.assertEqual(g8["pair_gate"]["report_only"][:len(g7["pair_gate"]["report_only"])], g7["pair_gate"]["report_only"])
            self.assertIn("K5 → K6", " ".join(g8["pair_gate"]["report_only"]))
            self.assertIn("K6 설계 2절", g8["pair_gate"]["verdicts"]["지킴 깨짐"])
            cc7, cc8 = g7["cell_check"], g8["cell_check"]
            self.assertEqual(cc8["recipe"], [n if n != "train-g4-bal" else "train-g4-bal2" for n in cc7["recipe"]])
            self.assertEqual({k: v for k, v in cc8.items() if k != "recipe"}, {k: v for k, v in cc7.items() if k != "recipe"})
        g8, _ = cmp.read_protocol(PROTOCOL_G8_DRAFT)
        self.assertEqual(g8["train_sets"], dict(g7["train_sets"], **{"train-g4-bal2": ""}))
        frozen, _ = cmp.read_protocol(PROTOCOL_G8_FROZEN)
        self.assertEqual(set(frozen["train_sets"]), set(g7["train_sets"]) | {"train-g4-bal2"})
        self.assertEqual({k: v for k, v in frozen["train_sets"].items() if k != "train-g4-bal2"}, g7["train_sets"])
        self.assertRegex(frozen["train_sets"]["train-g4-bal2"], "^[0-9a-f]{64}$")

    def test_frozen_is_draft_with_one_slot_filled(self):
        with open(PROTOCOL_G8_DRAFT, encoding="utf-8") as f:
            draft = f.read()
        with open(PROTOCOL_G8_FROZEN, encoding="utf-8") as f:
            frozen = f.read()
        digest = json.loads(frozen)["train_sets"]["train-g4-bal2"]
        self.assertEqual(draft.replace('"train-g4-bal2": ""', f'"train-g4-bal2": "{digest}"'), frozen)


class G6AlarmTest(unittest.TestCase):
    """protocol-g6 초안 — 새 경보 둘 「너무 오름」(recalls · max_mean_diff) · 「원천별 쏠림」(pred_shift) (원천 균형 판 설계 5-4 · 5-6)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name, PROTOCOL_G6)

    def tearDown(self):
        self._tmp.cleanup()

    def test_draft_same_as_g5_but_k4_rows(self):
        g6, _ = cmp.read_protocol(PROTOCOL_G6)
        g5, _ = cmp.read_protocol(PROTOCOL_G5_FROZEN)
        for key in ("sets", "compare_on", "calibrate_on", "release_pick", "bootstrap", "seeds", "majority", "release_gate",
                    "side", "group_key", "label_guide"):
            self.assertEqual(g6[key], g5[key], key)
        for key in ("target", "guards", "slice_note"):
            self.assertEqual(g6["pair_gate"][key], g5["pair_gate"][key], key)
        self.assertEqual((g6["name"], g6["pair_gate"]["base"], g6["pair_gate"]["new"]), ("protocol-g6", "K", "K4"))
        self.assertEqual(g6["pair_gate"]["alarms"][:6], g5["pair_gate"]["alarms"])
        self.assertEqual(len(g6["pair_gate"]["alarms"]), 8)
        self.assertEqual(g6["train_sets"], dict(g5["train_sets"], **{"train-g4-bal": ""}))
        self.assertEqual(g6["pair_gate"]["report_only"][:len(g5["pair_gate"]["report_only"])], g5["pair_gate"]["report_only"])
        self.assertIn("K3 → K4", " ".join(g6["pair_gate"]["report_only"]))
        self.assertEqual(g6["cell_check"]["recipe"], ["train-r2", "train-g1-arch", "train-g3-ko", "train-g4-bal"])
        self.assertEqual((g6["cell_check"]["dev"], g6["cell_check"]["thin"]), ("dev-cmp-g4", 10))
        with open(PROTOCOL_G6, encoding="utf-8") as f:
            self.assertNotIn("TODO", f.read())

    def test_old_alarms_same_as_g5(self):
        # 같은 판 짝에서 g6 의 옛 경보 여섯 줄 결과는 g5 와 같다 — 새 줄은 뒤에 붙기만 한다
        g5_dir = os.path.join(self._tmp.name, "g5")
        os.makedirs(g5_dir)
        g5_fx = Fixture(g5_dir, PROTOCOL_G5)
        p5, s5 = g5_fx.stats(ko_c_miss_pred, en_d_broken_pred)
        p6, s6 = self.fx.stats(ko_c_miss_pred, en_d_broken_pred)
        old, new = cmp.verdict(p5, s5)["alarms"], cmp.verdict(p6, s6)["alarms"]
        self.assertEqual(new[:len(old)], old)
        self.assertNotIn("pred_shift", s5)                         # 옛 규약 출력 칸은 그대로

    def test_too_high_fires(self):
        # ko c 0 → 1 : c_recall@ko · c_recall 평균 차 +1 > .20 · en d 1 → 0 은 내려감이라 「너무 오름」 아님
        protocol, stats = self.fx.stats(ko_c_miss_pred, en_d_broken_pred)
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertIn("c_recall@ko 평균 차 +1.0000 > 0.2", alarms)
        self.assertIn("c_recall 평균 차 +0.6000 > 0.2", alarms)       # c 다섯 중 ko 셋만 오름 → 3/5
        self.assertFalse([a for a in alarms if a.startswith("d_recall") and "> 0.2" in a])

    def test_too_high_quiet_when_small(self):
        protocol, stats = self.fx.stats(base_pred, base_pred)
        self.assertFalse([a for a in cmp.verdict(protocol, stats)["alarms"] if "> 0.2" in a])

    def test_pred_shift_fires_with_min_n(self):
        # jira(en) d 8쌍이 다 c 로 → 정답 d 중 c 몫 0 → 1. min_n 20 이면 8쌍 칸이라 안 센다, 5 면 운다
        protocol, stats = self.fx.stats(base_pred_all_right, en_d_broken_pred)
        self.assertEqual(stats["pred_shift"], [])
        self.assertFalse([a for a in cmp.verdict(protocol, stats)["alarms"] if a.startswith("domain ")])
        protocol["pair_gate"]["alarms"][-1]["min_n"] = 5
        _, base = cmp.load_runs(self.fx.runs("B0", base_pred_all_right))
        _, new = cmp.load_runs(self.fx.runs("B1", en_d_broken_pred))
        stats = cmp.compare(protocol, cmp.read_set(self.fx.set_path), base, new)
        line = next(r for r in stats["pred_shift"] if r["domain"] == "jira" and r["gold"] == "d" and r["pred"] == "c")
        self.assertEqual((line["pairs"], line["base"], line["new"], line["diff"]), (8, 0.0, 1.0, 1.0))
        alarms = cmp.verdict(protocol, stats)["alarms"]
        self.assertIn("domain jira 정답 d → c 몫 0.0000 → 1.0000 (차 +1.0000 > 0.3)", alarms)
        self.assertEqual(len([a for a in alarms if a.startswith("domain ")]), 1)

    def test_expand_alarm(self):
        specs = cmp.expand_alarm({"metric": "recalls", "langs": ["", "ko"], "max_mean_diff": 0.2})
        self.assertEqual([cmp.ruler_name(s) for s in specs],
                         ["a_recall", "b_recall", "c_recall", "d_recall",
                          "a_recall@ko", "b_recall@ko", "c_recall@ko", "d_recall@ko"])
        one = {"metric": "b_recall", "lang": "ko", "min_mean_diff": -0.03}
        self.assertEqual(cmp.expand_alarm(one), [one])


class OldFormCompatTest(unittest.TestCase):
    """옛 꼴(metric 문자열만)은 이름 · 지표 표가 그대로다 — g2 출력에 @ 자가 안 끼어든다."""

    def test_ruler_name_old_form(self):
        self.assertEqual(cmp.ruler_name({"metric": "b_recall", "ci_low_gt": 0.0}), "b_recall")
        self.assertFalse(cmp.is_slice({"metric": "b_recall"}))
        self.assertEqual(cmp.ruler_name({"metric": "b_recall", "lang": "ko", "domain": "mem"}), "b_recall@mem·ko")

    def test_g2_metrics_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, PROTOCOL_G2)
            protocol, stats = fx.stats(base_pred, broken_pred)
            self.assertEqual(list(stats["metrics"]), list(cmp.METRICS))
            self.assertEqual(stats["release_pick"]["metric"], "b_recall")
            self.assertEqual(cmp.gate_slice_specs(protocol), [])


if __name__ == "__main__":
    unittest.main()
