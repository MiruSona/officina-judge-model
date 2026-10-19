"""layer_report — K7 층 점검 보고 둘 (새 보고 ① 원천 × 새 낱말 구간 · 꼴 표 원천 × 꼴 × 칸).

E4 1-1 · 1-2 의 새 낱말 몫 · 구간 규칙과 K7 설계 2-3 「새 ①」 표시 규칙을 지어낸 작은 줄로 본다. 글은 모두 지어낸 것이다.
"""
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

import layer_report as lr  # noqa: E402


def row(claim, evidence, want="b", domain="kowiki", lang="ko", form="labeled"):
    out = {"claim": claim, "evidence": evidence, "want": want, "lang": lang, "form": form}
    if domain is not None:
        out["domain"] = domain
    return out


class NovelShareTest(unittest.TestCase):
    def test_korean_two_syllable_prefix(self):
        # 「서울에서」 → 앞 두 음절 「서울」 이 근거에 있음 · 「바다」 없음 → 1/2
        self.assertEqual(lr.novel_share("서울에서 바다", "서울의 거리"), 0.5)

    def test_one_syllable_token(self):
        self.assertEqual(lr.novel_share("산이 높다", "산"), 0.0)

    def test_english_and_number_lowercase(self):
        # Apple 은 근거 apple 에 있음 · 2024 있음 · Pear 없음 → 1/3
        self.assertAlmostEqual(lr.novel_share("an APPLE in 2024", "Apple 2024 Pear"), 1 / 3)

    def test_counts_duplicates(self):
        # 같은 덩어리도 목록으로 센다 : 바다 바다 산 → 바다 둘 없음 · 산 있음 → 2/3
        self.assertAlmostEqual(lr.novel_share("산", "바다 바다 산"), 2 / 3)

    def test_no_tokens_is_zero(self):
        self.assertEqual(lr.novel_share("근거", "... !!"), 0.0)
        self.assertEqual(lr.novel_share(None, None), 0.0)


class BandTest(unittest.TestCase):
    def test_edges_left_closed(self):
        got = [lr.band_of(x) for x in (0.0, 0.0999, 0.1, 0.25, 0.3999, 0.4, 0.6, 0.7999, 0.8, 1.0)]
        self.assertEqual(got, ["B0", "B0", "B1", "B2", "B2", "B3", "B4", "B4", "B5", "B5"])

    def test_edges_are_e4_values(self):
        self.assertEqual(lr.BAND_EDGES, (0.1, 0.25, 0.4, 0.6, 0.8))


def high(domain, want="c"):
    return row("가나 다라 마바", "전혀 다른 글", want=want, domain=domain)      # 새 낱말 1.0 → B5


def low(domain, want="b"):
    return row("가나 다라", "가나 다라", want=want, domain=domain)             # 0 → B0


class SourceBandsTest(unittest.TestCase):
    def test_flag_when_over_80_percent_high(self):
        base = [low("kowiki"), low("klue")]
        add = [high("klue")] * 9 + [low("klue")] + [high("kowiki")] * 4 + [low("kowiki")]   # klue .9 · kowiki .8
        got = lr.source_bands(base, add)
        by = {r["domain"]: r for r in got["rows"]}
        self.assertTrue(by["klue"]["flag"])
        self.assertAlmostEqual(by["klue"]["add_high_share"], 0.9)
        self.assertFalse(by["kowiki"]["flag"])                    # .8 은 「80% 넘게」가 아니다
        self.assertEqual(got["flagged"], ["klue"])
        self.assertEqual(by["klue"]["add"], {"B0": 1, "B1": 0, "B2": 0, "B3": 0, "B4": 0, "B5": 9})
        self.assertEqual(by["klue"]["base"]["B0"], 1)

    def test_source_without_add_rows_not_flagged(self):
        got = lr.source_bands([high("mem")], [])
        self.assertEqual(got["rows"][0]["add_n"], 0)
        self.assertIsNone(got["rows"][0]["add_high_share"])
        self.assertFalse(got["rows"][0]["flag"])

    def test_b_by_band_counts_whole_recipe(self):
        base = [low("klue", "b")]
        add = [high("klue", "b"), high("klue", "c")]
        r = lr.source_bands(base, add)["rows"][0]
        self.assertEqual(r["b_all"], {"B0": 1, "B1": 0, "B2": 0, "B3": 0, "B4": 0, "B5": 1})

    def test_missing_domain_is_mem_and_cd_dropped(self):
        rows = [row("가", "가", want="support", domain=None), row("가", "가", want="cd", domain="kowiki")]
        got = lr.source_bands(rows, [])
        self.assertEqual([r["domain"] for r in got["rows"]], ["mem"])
        self.assertEqual(got["dropped_cell"], 1)

    def test_lang_bands_split_parts(self):
        base = [low("a"), row("x y", "z", lang="en", domain="a")]
        add = [high("a")]
        got = lr.source_bands(base, add)["lang"]
        self.assertEqual(got["all"]["ko"]["B0"], 1)
        self.assertEqual(got["all"]["ko"]["B5"], 1)
        self.assertEqual(got["base"]["en"]["B5"], 1)
        self.assertNotIn("en", got["add"])

    def test_rows_sorted_by_domain(self):
        got = lr.source_bands([low("z"), low("a"), low("m")], [])
        self.assertEqual([r["domain"] for r in got["rows"]], ["a", "m", "z"])


class FormTableTest(unittest.TestCase):
    def test_counts_per_domain_lang_form_cell(self):
        rows = [row("c", "e", want="c", form="near-x"), row("c", "e", want="unrelated", form="near-x"),
                row("c", "e", want="b", form="labeled"), row("c", "e", want="cd", form="labeled"),
                row("c", "e", want="a", form=None, domain=None, lang=None)]
        got = lr.form_rows(rows, {"near-x"})
        self.assertEqual([(r["domain"], r["lang"], r["form"]) for r in got],
                         [("kowiki", "ko", "labeled"), ("kowiki", "ko", "near-x"), ("mem", "?", "(none)")])
        near = got[1]
        self.assertEqual((near["c"], near["n"], near["gen"]), (2, 2, True))
        self.assertEqual((got[0]["b"], got[0]["x"], got[0]["gen"]), (1, 1, False))

    def test_gen_share_per_lang_cell(self):
        rows = [row("c", "e", want="c", form="near-x")] * 3 + [row("c", "e", want="c", form="labeled")] \
            + [row("c", "e", want="c", form="wiki-edit", lang="en")]
        got = lr.gen_share(lr.form_rows(rows, {"near-x"}))
        ko_c = [g for g in got if (g["lang"], g["cell"]) == ("ko", "c")][0]
        self.assertEqual((ko_c["n"], ko_c["gen"], ko_c["top_form"], ko_c["top_n"]), (4, 3, "near-x", 3))
        self.assertAlmostEqual(ko_c["gen_share"], 0.75)
        en_c = [g for g in got if (g["lang"], g["cell"]) == ("en", "c")][0]
        self.assertEqual(en_c["gen"], 0)


class CliTest(unittest.TestCase):
    def write(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def setup_files(self, tmp):
        paths = {}
        for name, rows in (("base", [low("kowiki")]), ("add", [high("klue")] * 5),
                           ("dev", [row("c", "e", want="c", form="near-x")])):
            paths[name] = os.path.join(tmp, f"train-{name}.jsonl")
            self.write(paths[name], rows)
        proto = os.path.join(tmp, "proto.json")
        with open(proto, "w", encoding="utf-8") as f:
            json.dump({"generator_forms": ["near-x"]}, f)
        return paths, proto

    def run_main(self, argv):
        out = io.StringIO()
        with mock.patch.object(sys, "stdout", out):
            code = lr.main(argv)
        return code, out.getvalue()

    def test_writes_json_and_md_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, proto = self.setup_files(tmp)
            out = os.path.join(tmp, "out")
            argv = ["--recipe", p["base"], p["add"], "--base-recipe", p["base"], "--dev", p["dev"],
                    "--protocol", proto, "--out", out, "--name", "K7"]
            code, text = self.run_main(argv)
            self.assertEqual(code, 0)
            with open(os.path.join(out, "K7.json"), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["bands"]["flagged"], ["klue"])
            self.assertEqual(doc["add_layers"], ["train-add.jsonl"])
            self.assertEqual(sorted(doc["forms"]), ["train-add", "train-base", "train-dev"])
            with open(os.path.join(out, "K7.md"), encoding="utf-8") as f:
                md1 = f.read()
            self.assertIn("klue", md1)
            self.assertIn("**표시**", md1)
            self.run_main(argv)
            with open(os.path.join(out, "K7.md"), encoding="utf-8") as f:
                self.assertEqual(f.read(), md1)

    def test_base_not_in_recipe_exit2(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, proto = self.setup_files(tmp)
            code, _ = self.run_main(["--recipe", p["add"], "--base-recipe", p["base"], "--dev", p["dev"],
                                     "--protocol", proto, "--out", os.path.join(tmp, "o")])
            self.assertEqual(code, 2)

    def test_missing_file_exit2(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, proto = self.setup_files(tmp)
            code, _ = self.run_main(["--recipe", p["base"], os.path.join(tmp, "none.jsonl"), "--base-recipe", p["base"],
                                     "--dev", p["dev"], "--protocol", proto, "--out", os.path.join(tmp, "o")])
            self.assertEqual(code, 2)

    def test_hashseed_does_not_change_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, proto = self.setup_files(tmp)
            texts = []
            for seed in ("1", "2"):
                out = os.path.join(tmp, f"o{seed}")
                env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
                subprocess.run([sys.executable, "-I", os.path.join(helpers.ROOT, "build", "layer_report.py"),
                                "--recipe", p["base"], p["add"], "--base-recipe", p["base"], "--dev", p["dev"],
                                "--protocol", proto, "--out", out], check=True, env=env, capture_output=True)
                with open(os.path.join(out, "layer-report.json"), encoding="utf-8") as f:
                    texts.append(f.read())
            self.assertEqual(texts[0], texts[1])


if __name__ == "__main__":
    unittest.main()
