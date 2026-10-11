"""synth · label_batches(--judges · decide synth) · rulers source — ㄴ 합성 흐름. 입력은 모두 지어낸 짧은 글이다."""
import collections
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import label_batches as lb  # noqa: E402
import rulers  # noqa: E402
import synth  # noqa: E402
from test_label_batches import GUIDE, answer, read_jsonl, write_jsonl  # noqa: E402


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def train_rows():
    """want · 다섯 씨앗 예측을 함께 정한 가짜 학습 쌍. 값 = (want, [s1..s5 예측], origin)."""
    plan = {
        "x01": ("d", ["a", "a", "a", "d", "d"], "g1-II"),    # d→a 다수 → 씨앗
        "x02": ("a", ["d", "d", "d", "a", "c"], "g1-II"),    # a→d → 씨앗
        "x03": ("b", ["d", "d", "d", "d", "b"], "g1-ga"),    # b→d → 씨앗
        "x04": ("d", ["b", "b", "b", "d", "d"], "g1-ga"),    # d→b → 씨앗
        "x05": ("d", ["c", "c", "c", "c", "c"], "g1-II"),    # d↔c 는 씨앗 아님
        "x06": ("c", ["d", "d", "d", "d", "d"], "g1-II"),    # want c 는 씨앗 아님
        "x07": ("cd", ["a", "a", "a", "a", "a"], "g1-II"),   # cd 는 빼기
        "x08": ("a", ["d", "d", "a", "a", "c"], "g1-II"),    # 다수 없음
        "x09": ("a", ["a", "a", "a", "a", "a"], "g1-II"),    # 맞힘
    }
    for i in range(10, 30):                                  # cap 층화 시험용 d→a 씨앗
        plan[f"x{i:02d}"] = ("d", ["a"] * 5, "g1-II" if i < 25 else "g1-ga")
    rows, preds = [], collections.defaultdict(list)
    for pid, (want, pr, origin) in sorted(plan.items()):
        rows.append({"id": pid, "evidence": f"근거 {pid}", "claim": f"주장 {pid}", "want": want, "origin": origin,
                     "lang": "ko", "domain": "dom", "group": f"grp-{pid}", "groups": [f"grp-{pid}"], "lic": "CC-BY-4.0"})
        for s, p in enumerate(pr, 1):
            preds[s].append({"id": pid, "label": 0, "logits": [0, 0, 0, 0], "probs": [.25] * 4, "pred": p,
                             "kind": "", "src": ""})
    return rows, preds


class SeedsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        rows, preds = train_rows()
        self.train = os.path.join(self.root, "train.jsonl")
        write_jsonl(self.train, rows)
        self.preds = []
        for s, lines in preds.items():
            # II · arch 로 나눠 같은 씨앗 두 파일이 합쳐지는지 본다
            half = len(lines) // 2
            for part, chunk in (("II", lines[:half]), ("arch", lines[half:])):
                path = os.path.join(self.root, "pred", f"s{s}-{part}.pred.jsonl")
                write_jsonl(path, chunk)
                self.preds.append(path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_seeds(self, *extra):
        out = os.path.join(self.root, "seeds.jsonl")
        quiet(synth.main, ["seeds", "--train", self.train, "--pred", *self.preds, "--out", out, "--allow-git", *extra])
        return read_jsonl(out)

    def test_picks_ad_bd_flips_only(self):
        seeds = self.run_seeds()
        ids = {s["id"] for s in seeds}
        self.assertTrue({"x01", "x02", "x03", "x04"} <= ids)
        self.assertFalse(ids & {"x05", "x06", "x07", "x08", "x09"})
        self.assertEqual(len(seeds), 24)
        first = seeds[0]
        self.assertEqual(set(first["grid"]), set(synth.GRID))
        self.assertEqual(first["seed_id"], "s0001")
        for key in ("id", "evidence", "claim", "want", "origin", "lang", "grid"):
            self.assertIn(key, first)
        # 격자는 축마다 고르게 — 길이 둘은 12 · 12
        self.assertEqual(collections.Counter(s["grid"]["length"] for s in seeds), {"짧음": 12, "김": 12})

    def test_cap_stratified_by_origin(self):
        seeds = self.run_seeds("--cap", "12")
        self.assertEqual(len(seeds), 12)
        by_origin = collections.Counter(s["origin"] for s in seeds)
        # g1-II 17 · g1-ga 7 → 12 를 비례로 나눔 (8.5 · 3.5)
        self.assertEqual(sum(by_origin.values()), 12)
        self.assertIn(by_origin["g1-ga"], (3, 4))

    def test_bad_pred_name_stops(self):
        bad = os.path.join(self.root, "pred", "seed1.jsonl")
        write_jsonl(bad, [])
        with self.assertRaises(SystemExit):
            quiet(synth.main, ["seeds", "--train", self.train, "--pred", bad, "--out",
                               os.path.join(self.root, "s.jsonl"), "--allow-git"])


def seed_rows(n):
    rng = random.Random(1)
    rows = []
    for i in range(n):
        grid = {axis: rng.choice(values) for axis, values in synth.GRID.items()}
        rows.append({"seed_id": f"s{i + 1:04d}", "id": f"x{i:02d}", "evidence": f"원래 근거 {i}",
                     "claim": f"주장 {i}", "want": "d", "pred": "a", "origin": "g1-II", "lang": "ko", "grid": grid,
                     "domain": "dom", "group": f"grp{i}", "groups": [f"grp{i}"], "lic": "CC-BY-4.0"})
    return rows


class MakeCollectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.seeds = os.path.join(self.root, "seeds.jsonl")
        write_jsonl(self.seeds, seed_rows(25))
        self.guide = os.path.join(self.root, "guide.md")
        with open(self.guide, "w", encoding="utf-8") as f:
            f.write(GUIDE)
        self.out = os.path.join(self.root, "rw")
        quiet(synth.main, ["make", "--seeds", self.seeds, "--out", self.out, "--guide", self.guide, "--allow-git"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_batches_and_prompt(self):
        names = sorted(os.listdir(os.path.join(self.out, "batches")))
        self.assertEqual(names, ["g01.jsonl", "g01.txt", "g02.jsonl", "g02.txt"])
        shown = read_jsonl(os.path.join(self.out, "batches", "g01.jsonl"))
        self.assertEqual(len(shown), 20)
        self.assertEqual(set(shown[0]), {"seed_id", "lang", "grid", "claim", "evidence"})   # want · 원천은 안 보인다
        with open(os.path.join(self.out, "batches", "g02.txt"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("주장은 한 글자도 바꾸지 않는다", text)
        self.assertIn("## 3. 자주 갈리는 자리 규칙", text)    # 기준 1~3 절 전문
        self.assertNotIn("## 4.", text)
        self.assertIn("answers/g02.out.jsonl", text)
        self.assertIn("씨앗 5 개", text)
        with self.assertRaises(SystemExit):     # 같은 폴더에 다시 만들지 않는다
            quiet(synth.main, ["make", "--seeds", self.seeds, "--out", self.out, "--guide", self.guide, "--allow-git"])

    def test_collect_checks(self):
        def four(sid, texts):
            return [{"seed_id": sid, "cell": c, "evidence": t} for c, t in zip("abcd", texts)]
        lines = []
        lines += four("s0001", ["가 지지", "가 반대", "나 무관", "가 다른 속성"])          # 통과
        lines += four("s0002", ["지지", "반대", "무관", "계열"])[:3]                       # d 빠짐
        lines += four("s0003", ["지지", "", "무관", "계열"])                               # 빈 근거
        lines += four("s0004", ["주장 3", "반대", "무관", "계열"])                         # 주장과 같음
        lines += four("s0005", ["같은 글", "같은 글", "무관", "계열"])                     # 칸끼리 같음
        lines += four("s0006", ["지지", "반대", "무관", "계열"]) + [{"seed_id": "s0006", "cell": "a", "evidence": "또"}]
        lines += four("s0007", ["지지", "반대", "원래 근거 6", "계열"])                   # 원래 근거 그대로
        lines += [{"seed_id": "s0099", "cell": "a", "evidence": "x"}, {"nope": 1}]
        write_jsonl(os.path.join(self.out, "answers", "g01.out.jsonl"), lines)
        dest = os.path.join(self.root, "cands.jsonl")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            synth.main(["collect", "--out", self.out, "--dest", dest, "--allow-git"])
        report = json.loads(buf.getvalue())
        self.assertEqual(report["kept_seeds"], 1)
        for reason in ("missing_cell", "empty", "same_as_claim", "same_as_seed", "same_cells", "dup_cell"):
            self.assertEqual(report[reason], 1, reason)
        self.assertEqual(report["no_answer"], 25 - 7)
        self.assertEqual((report["unknown_seed"], report["bad_lines"]), (1, 1))
        rows = read_jsonl(dest)
        self.assertEqual([r["id"] for r in rows], [f"syn-s0001-{c}" for c in "abcd"])
        r = rows[3]
        self.assertEqual((r["hint"], r["origin"], r["synth_from"], r["claim"], r["lic"], r["label_guide"]),
                         ("d", "synth", "x00", "주장 0", "CC-BY-4.0", "support-v5"))


class SynthLabelFlowTest(unittest.TestCase):
    """합성 후보 → make --judges 5 → merge → decide --mode synth → assemble --mode train."""

    def test_flow(self):
        with tempfile.TemporaryDirectory() as root:
            guide = os.path.join(root, "guide.md")
            with open(guide, "w", encoding="utf-8") as f:
                f.write(GUIDE)
            cands = []
            for i in range(6):
                for cell in "abcd":
                    cands.append({"id": f"syn-s{i:04d}-{cell}", "evidence": f"새 근거 {i}{cell}", "claim": f"주장 {i}",
                                  "hint": cell, "origin": "synth", "src": "g1-II", "synth_from": f"x{i}",
                                  "grid": {"length": "짧음"}, "lang": "ko", "domain": "dom", "group": f"g{i}",
                                  "groups": [f"g{i}"], "lic": "CC-BY-4.0", "form": "synth"})
            cpath = os.path.join(root, "syn-cands.jsonl")
            write_jsonl(cpath, cands)
            out = os.path.join(root, "lab")
            quiet(lb.main, ["make", "--cands", cpath, "--out", out, "--mode", "train", "--judges", "5",
                            "--guide", guide, "--allow-git"])
            files = sorted(os.listdir(os.path.join(out, "batches")))
            self.assertEqual([f for f in files if f.endswith(".jsonl")], [f"b01-j{k}.jsonl" for k in range(1, 6)])
            mapping = read_jsonl(os.path.join(out, "map", "map.jsonl"))
            hint = {c["id"]: c["hint"] for c in cands}
            # 판 5 가 버려져 표는 넷 : s0001 은 4/4 → synth · s0002 는 2/4 + 빠짐 1 → drop
            for k in range(1, 6):
                lines = []
                for m in mapping:
                    want = hint[m["id"]]
                    seed = int(m["id"][5:9])
                    label = want
                    if seed == 1 and k == 5:
                        label = "c" if want != "c" else "d"
                    if seed == 2 and k in (3, 4):
                        label = "c" if want != "c" else "d"
                    if seed == 3 and k == 5:   # 판 5 는 답 없음 4/24 > 5% 라 판째 버려진다 — 다들 표 넷
                        continue
                    lines.append(answer(label, m["q"]))
                write_jsonl(os.path.join(out, "answers", f"b01-j{k}.jsonl"), lines)
            quiet(lb.main, ["merge", "--out", out])
            quiet(lb.main, ["decide", "--out", out, "--mode", "synth"])
            decided = {r["id"]: r for r in read_jsonl(os.path.join(out, "decided.jsonl"))}
            self.assertEqual(decided["syn-s0000-d"]["status"], "synth")
            self.assertEqual(decided["syn-s0000-d"]["label"], "d")
            self.assertEqual(decided["syn-s0003-a"]["status"], "synth")     # 판 5 가 버려져 4/4
            self.assertEqual(decided["syn-s0001-b"]["status"], "synth")     # 판 5 버림 → 4/4
            self.assertEqual(decided["syn-s0002-c"]["status"], "drop")      # 2/4 + 빠짐 1 < 4
            dest = os.path.join(root, "sets")
            quiet(lb.main, ["assemble", "--mode", "train", "--set", "train-g1-syn", "--dirs", out,
                            "--dest", dest, "--allow-git"])
            rows = read_jsonl(os.path.join(dest, "train-g1-syn.jsonl"))
            self.assertEqual(len(rows), 20)
            r = next(x for x in rows if x["id"] == "syn-s0000-b")
            self.assertEqual((r["want"], r["origin"], r["src"], r["lic"], r["synth_from"], r["status"]),
                             ("b", "synth", "g1-II", "CC-BY-4.0", "x0", "synth"))
            self.assertEqual(r["grid"], {"length": "짧음"})
            self.assertEqual(sorted(r["votes"]), ["j1", "j2", "j3", "j4"])

    def test_decide_synth_rule(self):
        five = {f"j{k}": "d" for k in range(1, 6)}
        self.assertEqual(lb.decide_synth(five, "d"), ("d", "synth"))
        self.assertEqual(lb.decide_synth(dict(five, j5="c"), "d"), ("d", "synth"))
        self.assertEqual(lb.decide_synth(dict(five, j4="c", j5="c"), "d"), (None, "drop"))
        self.assertEqual(lb.decide_synth({"j1": "d", "j2": "d", "j3": "d"}, "d"), (None, "missing"))
        self.assertEqual(lb.decide_synth({"j1": "d", "j2": "c", "j3": "d"}, "d"), (None, "missing"))
        self.assertEqual(lb.decide_synth({"j1": "d", "j2": "c", "j3": "c"}, "d"), (None, "drop"))
        self.assertEqual(lb.decide_synth(five, None), (None, "drop"))

    def test_decide_synth_needs_five_judges(self):
        with tempfile.TemporaryDirectory() as root:
            out = os.path.join(root, "lab")
            lb.write_json(os.path.join(out, "map", "manifest.json"), {"cands": "", "judges": 2})
            write_jsonl(os.path.join(out, "votes.jsonl"), [])
            with self.assertRaises(SystemExit):
                quiet(lb.main, ["decide", "--out", out, "--mode", "synth"])


class SourceRulerTest(unittest.TestCase):
    def test_tells_apart_only_when_style_differs(self):
        rng = random.Random(7)

        def words(k):
            return " ".join("".join(chr(0xAC00 + rng.randrange(400)) for _ in range(3)) for _ in range(k))
        real = [{"evidence": words(6), "group": f"r{i}"} for i in range(120)]
        alike = [{"evidence": words(6), "group": f"s{i}"} for i in range(120)]
        styled = [{"evidence": "합성 말투 표시 " + words(6), "group": f"t{i}"} for i in range(120)]
        same = rulers.source_probe(alike, real)
        diff = rulers.source_probe(styled, real)
        self.assertLess(abs(same["acc"] - 0.5), 0.15)
        self.assertGreater(diff["acc"], 0.9)
        self.assertEqual(len(diff["folds"]), 5)
        self.assertEqual(diff["n_each"], 120)
        self.assertNotIn("pass", diff)


class CapTest(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(synth.synth_d_cap(300), 100)     # 100 / (300 + 100) = 25%
        self.assertEqual(synth.synth_d_cap(1745), 581)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "set.jsonl")
            write_jsonl(path, [{"want": "d", "origin": "synth"}] * 4 + [{"want": "d", "origin": "g1-II"}] * 9)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    synth.main(["cap", "--d-total", "9", "--set", path])   # 상한 3 < 4
            res = json.loads(buf.getvalue())
            self.assertEqual((res["max_synth_d"], res["synth_d"], res["pass"], res["seeds_for_cap"]), (3, 4, False, 5))


class ArgCheckTest(unittest.TestCase):
    """리뷰 고침 — 범위 밖 숫자 인자는 셈 전에 멈춘다."""

    def test_synth_refuses_negative_cap_and_bad_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "seeds.jsonl")
            with self.assertRaises(SystemExit):
                quiet(synth.main, ["seeds", "--train", out, "--pred", out, "--out", out, "--cap", "-1", "--allow-git"])
        for share in ("0", "1", "1.5"):
            with self.subTest(share=share), self.assertRaises(SystemExit):
                quiet(synth.main, ["cap", "--d-total", "100", "--share", share])

    def test_rulers_refuses_folds_below_two(self):
        with self.assertRaises(SystemExit):
            quiet(rulers.main, ["source", "--synth", "x", "--real", "x", "--diagnostic", "--folds", "0"])


if __name__ == "__main__":
    unittest.main()
