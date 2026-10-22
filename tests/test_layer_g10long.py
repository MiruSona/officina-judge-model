"""layer_g10long — 글 꼴 셋 · 꼴 여덟 초안 · 토큰 자름 · 겹침 0 · 판정 뒤 고르기. 입력은 모두 지어낸 짧은 글이다."""
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

import layer_g10long as lg  # noqa: E402
import mine_wiki as mw  # noqa: E402
import scrub  # noqa: E402


class FakeTok:
    """글자 둘 = 토큰 하나. 쌍은 특수 토큰 넷을 더한다 (XLM-R 꼴)."""

    def count(self, text):
        return (len(text) + 1) // 2

    def pair(self, evidence, claim):
        return self.count(evidence) + self.count(claim) + 4


WIKI = "\n".join([
    "'''가나 마을'''은 산골에 있는 마을이다. 마을에는 3개의 우물이 있다. 주민은 120명이 산다. 마을은 1950년에 생겼다.",
    "== 역사 ==",
    "마을 회관은 1980년에 지어졌다. 그 뒤 2005년에 고쳐 지었다. 회관은 주민 모두가 쓴다. 회관에는 방이 4개 있다. "
    "회관 앞에는 큰 나무가 서 있다.",
    "== 시설 ==",
    "{| class=\"wikitable\"",
    "! 이름 !! 설명",
    "|-",
    "| 우물 || 마을 사람들이 함께 쓰는 우물로 물이 맑고 차갑다고 전해지며 겨울에도 얼지 않는다.",
    "|-",
    "| 회관 || 주민 회의를 여는 건물로 방이 4개 있고 마을 사람들이 함께 쓰는 곳이다.",
    "|}",
    "시설은 마을 사람들이 함께 돌본다. 해마다 봄에 청소를 한다.",
])


def page(pid="7", text=WIKI, title="가나 마을"):
    return {"id": pid, "title": title, "text": text}


class WikiParseTest(unittest.TestCase):
    def test_sections_tables_paragraphs(self):
        secs = lg.parse_page(WIKI)
        self.assertEqual([s["title"] for s in secs], ["", "역사", "시설"])
        table = secs[2]["tables"][0]
        self.assertEqual(table["head"], ["이름", "설명"])
        self.assertEqual([r[0] for r in table["rows"]], ["우물", "회관"])
        self.assertTrue(secs[0]["paras"][0].startswith("가나 마을은"))

    def test_merged_cells_table_dropped(self):
        lines = ["{|", "! 가 !! 나", "|-", "| rowspan=2 | 하나 || 둘", "|-", "| 셋 || 넷", "|}"]
        self.assertIsNone(lg.read_table(lines, 0)[1])

    def test_three_shapes(self):
        out = lg.ko_lines(page(), random.Random(1))
        head = out["head"]
        self.assertTrue(head, "표 칸 줄이 있어야 한다")
        self.assertTrue(head[0]["head"].startswith("[절 : 시설 · 줄 머리 : "))
        self.assertIn(" · 칸 머리 : 설명] ", head[0]["head"])          # conflict mdscan 꼴 머리
        self.assertTrue(all(any(sep in ln["body"] for sep in (" · ", " — ", " / ")) for ln in out["join"]))
        self.assertTrue(all(ln["head"] == "" for ln in out["para"]))
        self.assertTrue(all(ln["other"] for ln in out["para"]))           # c 꼴 재료 = 다른 절 문장
        self.assertTrue(all(o not in ln["body"] for ln in out["join"] for o in ln["other"]))

    def test_number_list_cells_are_not_prose(self):
        self.assertFalse(lg._prose_cell("1, 2, 3, 7, 11, 19, 43, 67, 163"))
        self.assertTrue(lg._prose_cell("주민 회의를 여는 건물로 방이 4개 있다"))


class SegmentTest(unittest.TestCase):
    def test_segments_and_rebuild(self):
        body = "하나는 3개다 · 둘은 5개다 — 셋은 끝이다. 넷은 남았다"
        spans = lg.segments(body)
        self.assertEqual([body[s:e] for s, e in spans], ["하나는 3개다", "둘은 5개다", "셋은 끝이다.", "넷은 남았다"])
        self.assertEqual(lg.rebuild(body, spans, [0, 2]), "하나는 3개다 — 셋은 끝이다.")
        self.assertEqual(lg.position(body, 0, 3), "front")
        self.assertEqual(lg.position(body, len(body) - 3, len(body)), "end")


class EnSlotTest(unittest.TestCase):
    def test_values_flips_alt(self):
        text = "The bridge opened in 1990 and carries 3 lanes for 1,500 cars, 40% more than before."
        slots = lg.en_value_slots(text)
        self.assertEqual([(s["kind"], s["text"]) for s in slots],
                         [("date", "1990"), ("num", "3"), ("num", "1,500"), ("num", "40%")])
        rng = random.Random(2)
        self.assertNotEqual(lg.en_new_value(slots[0], text, rng), "1990")
        self.assertTrue(lg.en_new_value(slots[3], text, rng).endswith("%"))
        self.assertEqual([lg.en_alt(s) for s in slots[1:]], ["three", "1500", "40 percent"])
        self.assertEqual([s["text"] for s in lg.flip_slots(text, "en")], ["before"])

    def test_vitaminc_dedupe_and_clean(self):
        rows = [{"revision_type": "real", "page": "P", "case_id": str(i), "evidence": e} for i, e in enumerate([
            "Her father is of Bengali and English descent , and her mother is Irish .",
            "Her father is of English and Hindu descent , and her mother is Irish .",   # 같은 문장의 다른 판
            "She has a brother who is two years older and works as a travel writer .",
            "The family moved to London in 1995 when she was seven years old .",
            "lower case start is not a sentence and should be dropped .",
        ])]
        pages = lg._vitaminc_pages("unused", read_rows=lambda path: rows)
        self.assertEqual(len(pages["P"]), 3)
        self.assertTrue(pages["P"][0].endswith("Irish."))
        self.assertNotIn(" ,", " ".join(pages["P"]))


def lines_for_make():
    """make 시험용 근거 줄 — ko 셋 · en 둘 (근거마다 값 · 부정어 · 반대말 · 때 자리)."""
    ko = [
        "1980년에 회관을 3개 지었다 · 회관은 주민이 함께 쓴다 — 회의는 처음 봄에 열었다 · 그 뒤 방을 고치지 않았다",
        "2005년에 우물을 5개 팠다 · 우물 물은 맑다 — 마을 사람 모두가 이용한다 / 겨울에는 얼지 않는다",
        "주민은 120명이 산다 · 마을은 1950년에 생겼다 — 학교는 마지막 해에 닫았다 · 시장은 열리지 않는다",
    ]
    en = [
        "The bridge opened in 1990 with 3 lanes · It is the longest bridge in the region — It was not repaired until 2010",
        "The club won 12 titles in 2001 · Its stadium is the largest in the city — The team did not play abroad",
    ]
    out = []
    for i, body in enumerate(ko):
        out.append({"id": f"ko{i:04d}", "lang": "ko", "shape": "join", "head": "[절 : 역사] ", "body": body,
                    "group": f"kowiki:{i}", "other": ["다른 절의 문장으로 마을 축제는 가을에 열린다."], "section": "역사"})
    for i, body in enumerate(en):
        out.append({"id": f"en{i:04d}", "lang": "en", "shape": "para", "head": "", "body": body,
                    "group": f"vitc:P{i}", "other": ["A different sentence about the weather in spring."], "section": "P"})
    return out


class MakeTest(unittest.TestCase):
    def setUp(self):
        self.scrubber = scrub.Scrubber([], ())

    def test_forms_and_one_chunk(self):
        rows, awkward, report = lg.make_rows(lines_for_make() * 1, 48, 5, FakeTok(), self.scrubber)
        forms = {r["form"] for r in rows}
        self.assertTrue({"long-num", "long-same", "long-other"} <= forms, forms)
        for r in rows:
            self.assertEqual(r["hint"], lg.FORMS[r["form"]][0])
            self.assertTrue(r["id"].startswith("g10l-") and "-s" in r["id"])
            self.assertEqual(r["id"][-8:], lg.sha_key(r)[:8])            # 꼬리 = 쌍 sha 앞 8자
            if r["form"] in ("long-num", "long-neg", "long-flip", "long-alt"):
                self.assertIn(r["seg_old"], r["evidence"])
                self.assertIn(r["seg_new"], r["claim"])
                self.assertEqual(r["claim"].replace(r["seg_new"], r["seg_old"], 1), r["evidence"])   # 한 토막만 바뀜
        per_pair = {}
        for r in rows:
            per_pair.setdefault(r["pair"], set()).add(r["hint"])
        self.assertTrue(any(len(v) >= 2 for v in per_pair.values()))      # 같은 근거에 a · b 가 같이
        self.assertEqual(report["count"]["rows"], len(rows))

    def test_num_never_changes_a_date(self):
        rows, _, _ = lg.make_rows(lines_for_make(), 48, 5, FakeTok(), self.scrubber)
        nums = [r for r in rows if r["form"] == "long-num"]
        self.assertTrue(nums)
        self.assertFalse([r for r in nums if r["slot"]["kind"] == "date"])

    def test_neg_rows_change_negation_by_one(self):
        rows, _, _ = lg.make_rows(lines_for_make(), 48, 5, FakeTok(), self.scrubber)
        import seed_contrast as sc
        for r in rows:
            if r["form"] == "long-neg":
                self.assertEqual(abs(sc.count_neg(r["claim"], r["lang"]) - sc.count_neg(r["evidence"], r["lang"])), 1)

    def test_time_changes_only_the_time(self):
        import seed_contrast as sc
        for lang, body in (("ko", "주민은 120명이 산다 · 1980년에 회관을 3개 지었다"),
                           ("en", "The bridge opened in 1990 with 3 lanes · It is long")):
            got = lg._time_changes(body, lang, random.Random(4))
            self.assertTrue(got, lang)
            for slot, draft, extra in got:
                self.assertEqual(slot["kind"], "time")
                self.assertEqual(draft, body[:slot["start"]] + extra["d"]["time_new"] + body[slot["end"]:])
                self.assertEqual(sc.chunks(body, draft), 1)        # 때 한 덩이만 — 값까지 바꾸면 b 로 읽힌다

    def test_token_cut_window_and_drop(self):
        tok = FakeTok()
        body = " · ".join(f"토막 {i}번은 값이 {i + 2}개다" for i in range(8))
        spans = lg.segments(body)
        evidence = body
        claim, fit = lg._fit_claim(tok, evidence, "", body, spans, 4)
        self.assertEqual(fit, "full")
        lg_max = lg.MAX_LEN
        try:
            lg.MAX_LEN = tok.count(evidence) + 30      # 주장 전체는 안 들어간다 → 바뀐 토막 둘레만
            claim, fit = lg._fit_claim(tok, evidence, "", body, spans, 4)
            self.assertTrue(fit.startswith("window"))
            self.assertIn("토막 4번", claim)
            self.assertLessEqual(tok.pair(evidence, claim), lg.MAX_LEN)
            lg.MAX_LEN = tok.count(evidence)           # 토막 하나도 못 넣는다 → 버림
            self.assertEqual(lg._fit_claim(tok, evidence, "", body, spans, 4), (None, None))
        finally:
            lg.MAX_LEN = lg_max

    def test_form_weights(self):
        w = lg.parse_weights("long-alt/ko=2,long-other=0.5")
        q, q0 = lg.quotas(1800, weights=w), lg.quotas(1800)
        self.assertEqual(sum(q.values()), 1800)
        self.assertGreater(q[("long-alt", "ko")], 1.8 * q0[("long-alt", "ko")])
        self.assertLess(q[("long-other", "en")], q0[("long-other", "en")])
        self.assertEqual(lg.quotas(1800, weights={}), q0)
        with self.assertRaises(SystemExit):
            lg.parse_weights("long-xx=2")
        for bad in ("long-alt=-1", ",".join(f"{f}=0" for f in lg.FORMS)):
            with self.assertRaises(SystemExit):
                lg.parse_weights(bad)

    def test_quotas_sum_and_share(self):
        q = lg.quotas(1800)
        self.assertEqual(sum(q.values()), 1800)
        self.assertAlmostEqual(sum(v for (f, l), v in q.items() if l == "ko") / 1800, 0.65, places=2)
        self.assertLessEqual(abs(q[("long-num", "ko")] + q[("long-num", "en")] - 270), 1)   # 큰 나머지 나눔이라 ±1


class CheckTest(unittest.TestCase):
    def test_tokens_overlap_scrub(self):
        scrubber = scrub.Scrubber(["비밀이름"], ())
        row = {"form": "long-same", "evidence": "마을 회관은 1980년에 지어졌다 · 방이 4개 있다", "claim": "방이 4개 있다",
               "lang": "ko"}
        self.assertEqual(lg.check_row(row, FakeTok(), set(), scrubber), [])
        banned = lg.shingles("게임 속 회관은 1980년에 지어졌다")
        self.assertIn("synth_overlap", lg.check_row(row, FakeTok(), banned, scrubber))
        self.assertEqual(lg.check_row(row, FakeTok(), lg.shingles("전혀 다른 기획 글 문장"), scrubber), [])
        self.assertIn("scrub", lg.check_row(dict(row, claim="비밀이름 이 왔다"), FakeTok(), set(), scrubber))
        long_row = dict(row, evidence="가" * 700)
        self.assertIn("tokens_over", lg.check_row(long_row, FakeTok(), set(), scrubber))

    def test_head_template_is_not_overlap(self):
        banned = lg.shingles("[절 : 가 · 줄 머리 : 2 · 칸 머리 : 나] 다른 글")
        row = "[절 : 역사 · 줄 머리 : 2 · 칸 머리 : 설명] 마을 회관은 오래되었다"
        self.assertFalse(lg.shingles(row) & banned)
        self.assertFalse(lg.shingles("see the reference list") & lg.shingles("a reference book"))

    def test_trivial_alt_and_glued(self):
        scrubber = scrub.Scrubber([], ())
        base = {"form": "long-alt", "lang": "en", "slot": {"kind": "num", "text": "428,000"}, "new": "428000",
                "evidence": "It had 428,000 people", "claim": "It had 428000 people",
                "seg_old": "It had 428,000 people", "seg_new": "It had 428000 people"}
        self.assertIn("alt_trivial", lg.check_row(base, FakeTok(), set(), scrubber))
        ok = dict(base, new="428 thousand", claim="It had 428 thousand people", seg_new="It had 428 thousand people")
        self.assertNotIn("alt_trivial", lg.check_row(ok, FakeTok(), set(), scrubber))
        glued = {"form": "long-other", "lang": "en", "evidence": "It won 10-0.New England came back", "claim": "x y"}
        self.assertIn("glued_sentence", lg.check_row(glued, FakeTok(), set(), scrubber))
        self.assertNotIn("glued_sentence", lg.check_row(dict(glued, evidence="It won 10-0. New England"), FakeTok(),
                                                        set(), scrubber))

    def test_changed_segment_check(self):
        scrubber = scrub.Scrubber([], ())
        ev = "회관은 3개다 · 방은 4개다"
        good = {"form": "long-num", "evidence": ev, "claim": "회관은 5개다 · 방은 4개다", "lang": "ko",
                "slot": {"kind": "num", "text": "3개"}, "new": "5개", "seg_old": "회관은 3개다", "seg_new": "회관은 5개다"}
        self.assertEqual(lg.check_row(good, FakeTok(), set(), scrubber), [])
        bad = dict(good, claim="회관은 5개를 새로 지어 모두 바꿨다 · 방은 4개다", seg_new="회관은 5개를 새로 지어 모두 바꿨다")
        self.assertTrue(set(lg.check_row(bad, FakeTok(), set(), scrubber)) & {"chunks", "tokens"})


class PickTest(unittest.TestCase):
    """평가 몫 문서 · 피할 묶음은 학습 근거로 안 가고, 평가 몫 문서만 ko-docs 로 간다 (겹침 0)."""

    def test_pick_excludes_eval_and_avoid(self):
        extra = "\n== 덧붙임 ==\n" + " ".join(f"마을의 {k}번째 이야기는 오래전부터 전해 내려온다." for k in range(12))
        pages = [page(str(i), text=WIKI + extra) for i in range(1, 60)]   # 평가 몫 문서는 글 300자 넘어야 ko-docs 로
        evals = {f"kowiki:{p['id']}" for p in pages if mw.eval_bucket(f"kowiki:{p['id']}") == 0}
        self.assertTrue(evals)
        with tempfile.TemporaryDirectory() as tmp:
            avoid = os.path.join(tmp, "avoid.jsonl")
            with open(avoid, "w", encoding="utf-8") as f:
                f.write(json.dumps({"group": "kowiki:3", "groups": ["vitc:P1"]}) + "\n")
            vit = {f"P{i}": [f"The town {i} had {i + 3} schools in 1990.", f"It is the largest town {i} in the area.",
                             f"The mayor {i} was not elected until 2001.", f"About {i + 10} people live there now.",
                             f"The river {i} floods every spring season."] for i in range(1, 6)}
            out, docs = os.path.join(tmp, "out"), os.path.join(tmp, "docs.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                lg.main(["pick", "--kowiki", tmp, "--vitaminc", tmp, "--avoid-sets", avoid, "--out", out,
                         "--docs-out", docs, "--n-ko", "30", "--n-en", "6", "--allow-git"],
                        tok=FakeTok(), pages=pages, vit_pages=vit)
            picked = lg.lb.read_jsonl(os.path.join(out, "pick.jsonl"))
            doc_rows = lg.lb.read_jsonl(docs)
            with open(os.path.join(out, "pick.json"), encoding="utf-8") as f:
                report = json.load(f)
        groups = {r["group"] for r in picked}
        self.assertFalse(groups & evals)
        self.assertNotIn("kowiki:3", groups)
        self.assertNotIn("vitc:P1", groups)
        self.assertEqual({d["group"] for d in doc_rows}, evals - {"kowiki:3"})
        self.assertFalse({d["group"] for d in doc_rows} & groups)
        self.assertLessEqual(max(sum(1 for r in picked if r["group"] == g) for g in groups), lg.GROUP_CAP)
        self.assertTrue({"head", "join", "para"} <= {r["shape"] for r in picked if r["lang"] == "ko"})
        self.assertTrue(all(lg.EV_TOKENS[0] <= r["ev_tokens"] <= lg.EV_TOKENS[1] for r in picked))
        self.assertEqual(report["count"]["ko_avoid"] + report["count"]["en_avoid"], 2)


class PagesCacheTest(unittest.TestCase):
    def test_cache_remembers_limit_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "pages.jsonl")
            lg.lb.write_jsonl(cache, [{"_meta": {"limit_pages": 5}}, page()])
            self.assertEqual(list(lg.cached_pages([], 5, cache)), [page()])
            with self.assertRaises(SystemExit):
                list(lg.cached_pages([], 0, cache))


class BuildCnliTest(unittest.TestCase):
    def test_build_keeps_pairs_whole(self):
        rows = []
        for p in range(30):
            for form in ("long-num", "long-same"):
                rows.append({"pair": f"p{p}", "form": form, "lang": "ko", "keep": p % 5 != 0, "want": lg.FORMS[form][0]})
        out, report = lg.build_rows(rows, 40, 1)
        pairs = {}
        for r in out:
            pairs.setdefault(r["pair"], []).append(r)
        self.assertTrue(all(len(v) == 2 for v in pairs.values()))
        self.assertFalse(any(not r["keep"] for r in out))
        self.assertLessEqual(report["have"].get("long-num/ko", 0), lg.quotas(40)[("long-num", "ko")])

    def test_build_drops_single_hint_pairs(self):
        rows = [{"pair": "p1", "form": "long-num", "lang": "ko", "hint": "b", "keep": True, "want": "b"},
                {"pair": "p1", "form": "long-same", "lang": "ko", "hint": "a", "keep": True, "want": "a"},
                {"pair": "p2", "form": "long-same", "lang": "ko", "hint": "a", "keep": True, "want": "a"},
                {"pair": "p3", "form": "long-same", "lang": "ko", "hint": "a", "keep": True, "want": "a"},
                {"pair": "p3", "form": "long-alt", "lang": "ko", "hint": "a", "keep": True, "want": "a"}]
        out, report = lg.build_rows(rows, 1200, 1)
        self.assertEqual({r["pair"] for r in out}, {"p1"})          # p2 칸 하나 · p3 도 칸은 a 하나뿐
        self.assertEqual(report["single_hint_pairs_dropped"], 2)

    def test_fixed_row(self):
        line = {"id": "ko0001", "lang": "ko", "shape": "join", "head": "[절 : 역사] ", "group": "kowiki:1",
                "body": "회관은 3개다 · 방은 넷이다"}
        ok = {"form": "long-alt", "lang": "ko", "claim": "[절 : 역사] 회관은 세 개다 · 방은 넷이다",
              "slot": {"kind": "num", "text": "3개"}, "new": "세 개"}
        row, why = lg.fixed_row(ok, line, FakeTok())
        self.assertEqual(why, [])
        self.assertEqual((row["seg_old"], row["seg_new"], row["how"]), ("회관은 3개다", "회관은 세 개다", "fixed"))
        two = dict(ok, claim="[절 : 역사] 회관은 세 개다 · 방은 다섯이다")
        self.assertEqual(lg.fixed_row(two, line, FakeTok())[1], ["chunks"])
        q = dict(ok, form="long-num", claim="[절 : 역사] 회관은 5개인가? · 방은 넷이다")
        self.assertEqual(lg.fixed_row(q, line, FakeTok())[1], ["question_b"])
        self.assertEqual(lg.fixed_row(dict(ok, claim=""), line, FakeTok())[1], ["empty"])
        self.assertEqual(lg.fixed_row(dict(ok, claim=ok["claim"] + "�"), line, FakeTok())[1], ["broken_char"])

    def test_fixed_neg_must_change_negation(self):
        line = {"id": "ko0001", "lang": "ko", "shape": "join", "head": "[절 : 역사] ", "group": "kowiki:1",
                "body": "회관은 크지 않다 · 방은 넷이다"}
        fixes = [{"pick_id": "ko0001", "form": "long-neg", "lang": "ko", "claim": "[절 : 역사] 회관은 크다 · 방은 넷이다",
                  "slot": {"kind": "neg", "text": "크지 않다"}, "new": "크다"},
                 {"pick_id": "ko0001", "form": "long-neg", "lang": "ko", "claim": "[절 : 역사] 회관은 작지 않다 · 방은 넷이다",
                  "slot": {"kind": "neg", "text": "크지 않다"}, "new": "크다"},     # 다듬다 부정어가 남음
                 {"pick_id": "ko9999", "form": "long-neg", "lang": "ko", "claim": "x"}]
        with tempfile.TemporaryDirectory() as tmp:
            pick, inp, out = (os.path.join(tmp, n) for n in ("pick.jsonl", "fixed.jsonl", "out.jsonl"))
            lg.lb.write_jsonl(pick, [line])
            lg.lb.write_jsonl(inp, fixes)
            with contextlib.redirect_stdout(io.StringIO()):
                lg.main(["fixed", "--in", inp, "--pick", pick, "--out", out, "--judge-root", tmp, "--allow-git"], tok=FakeTok())
            passed = lg.lb.read_jsonl(out)
            with open(os.path.join(tmp, "out.json"), encoding="utf-8") as f:
                report = json.load(f)["by_form_lang"]["long-neg/ko"]
        self.assertEqual([r["claim"] for r in passed], [fixes[0]["claim"]])
        self.assertEqual((report["negation"], report["no_pick"], report["pass"]), (1, 1, 1))

    def test_merge_judged_dedupes_dirs(self):
        def row(i, claim, want):
            return {"id": f"r{i}", "evidence": "근거 글", "claim": claim, "want": want, "keep": True, "pair": "p1"}
        rows = [row(1, "가", "a"), row(2, "가", "a"), row(3, "나", "b"), row(4, "나", "c"), row(5, "다", "b"),
                dict(row(6, "라", "a"), keep=False)]
        out, report = lg.merge_judged(rows)
        self.assertEqual([r["id"] for r in out], ["r1", "r5"])
        self.assertEqual(report, {"dup_rows_dropped": 1, "split_want_rows_dropped": 2})

    def test_build_picks_rare_form_groups_first(self):
        q = lg.quotas(40)
        rows = []
        for p in range(30):                           # 흔한 묶음: num + same — 몫보다 훨씬 많다
            rows += [{"pair": f"c{p}", "form": "long-num", "lang": "ko", "keep": True, "want": "b"},
                     {"pair": f"c{p}", "form": "long-same", "lang": "ko", "keep": True, "want": "a"}]
        for p in range(q[("long-alt", "ko")]):        # 귀한 묶음: num + alt — alt 공급이 몫과 같다
            rows += [{"pair": f"r{p}", "form": "long-num", "lang": "ko", "keep": True, "want": "b"},
                     {"pair": f"r{p}", "form": "long-alt", "lang": "ko", "keep": True, "want": "a"}]
        out, report = lg.build_rows(rows, 40, 1)
        self.assertEqual(report["have"].get("long-alt/ko", 0), q[("long-alt", "ko")])   # 귀한 꼴 몫이 다 찬다
        self.assertLessEqual(report["have"]["long-num/ko"], q[("long-num", "ko")])

    def test_to_pairs(self):
        rows = [{"id": f"r{i}", "pair": p, "hint": h, "evidence": "e", "claim": f"c{i}", "slot": {"kind": "num"},
                 "seg_old": "x", "form": "long-num"} for i, (p, h) in enumerate([("p1", "b"), ("p1", "a"), ("p2", "c")])]
        out, report = lg.to_pairs(rows)
        self.assertNotIn("seg_old", out[0])
        self.assertEqual(out[0]["slot_kind"], "num")
        self.assertEqual((report["pairs"], report["pairs_with_2plus_hints"]), (2, 1))
        clash = rows + [dict(rows[0], id="r9", hint="d", form="long-time")]   # 같은 근거·주장에 b 와 d
        out, report = lg.to_pairs(clash)
        self.assertNotIn("r0", {r["id"] for r in out})
        self.assertEqual(report["clash_rows_dropped"], 2)
        dated = rows + [dict(rows[0], id="r8", claim="c8", slot={"kind": "date"})]
        out, report = lg.to_pairs(dated, drop_num_date=True)
        self.assertNotIn("r8", {r["id"] for r in out})
        self.assertEqual((report["num_date_dropped"], report["clash_rows_dropped"]), (1, 0))

    def test_cnli_rows(self):
        spans = ["The Receiving Party shall not disclose Confidential Information.", "This Agreement is governed by law."]
        doc = {"id": 9, "annotation_sets": [{"annotations": {
            "nda-1": {"choice": "Contradiction", "spans": [0]}, "nda-2": {"choice": "NotMentioned", "spans": []}}}]}
        labels = {"nda-1": {"hypothesis": "The party may share information."},
                  "nda-2": {"hypothesis": "The party must return copies."}}
        rows, report = lg.cnli_rows([("train", doc, spans, labels), ("dev", dict(doc, id=1), spans, labels)],
                                    {"cnli:dev-1"}, FakeTok())
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["hint"] for r in rows], ["b", "d"])
        self.assertEqual(rows[0]["evidence"], spans[0])
        self.assertEqual(report["count"]["avoid"], 1)
        self.assertTrue(all(r["id"].endswith(lg.sha_key(r)[:8]) for r in rows))


if __name__ == "__main__":
    unittest.main()
