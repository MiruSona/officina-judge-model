"""mine_wiki — 위키 개정 이력 덤프에서 한 자리 고침 후보 뽑기. 입력은 지어낸 작은 XML 을 bz2 로 묶은 것이다."""
import bz2
import io
import json
import os
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import mine_wiki as mw  # noqa: E402

BASE = ("'''하늘시'''는 [[가상국]]의 도시이다. 2019년 인구는 10만 명이었다.{{출처 필요}} "
        "이 도시는 3번째로 생긴 시이다.<ref>가짜 책</ref>\n[[분류:가상 도시]]")


def rev(rid, text, sha, user="사람1", comment=""):
    return (f"<revision><id>{rid}</id><timestamp>2020-01-0{rid % 9 + 1}T00:00:00Z</timestamp>"
            f"<contributor><username>{user}</username><id>1</id></contributor><comment>{comment}</comment>"
            f"<text bytes=\"1\" xml:space=\"preserve\">{text}</text><sha1>{sha}</sha1></revision>")


def page(pid, title, ns, revs):
    return f"<page><title>{title}</title><ns>{ns}</ns><id>{pid}</id>{''.join(revs)}</page>"


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dump_bytes():
    city = [
        rev(11, esc(BASE), "s1"),
        rev(12, esc(BASE.replace("3번째로", "4번째로")), "s2"),                       # b · ordinal
        rev(13, esc(BASE.replace("3번째로", "4번째로").replace("2019년 인구는 10만", "2020년 인구는 11만")), "s3"),  # d
        rev(14, esc(BASE.replace("3번째로", "4번째로").replace("2019년 인구는 10만", "2020년 인구는 11만")
                    .replace("이 도시는 4", "이 도시는   4")), "s4"),                     # 빈칸만 → 문법만 바뀜으로 버림
        rev(15, esc(BASE.replace("3번째로", "4번째로").replace("2019년 인구는 10만", "2020년 인구는 11만")
                    .replace("가상국", "꿈나라")), "s5", user="CleanBot"),             # 봇 · 이름 바뀜 · 뒤에 되돌려짐
        rev(16, esc(BASE.replace("3번째로", "4번째로").replace("2019년 인구는 10만", "2020년 인구는 11만")), "s3",
            comment="편집 되돌림"),                                                    # s3 으로 되돌림
    ]
    talk = [rev(21, "토론 글이다. 숫자는 1이다.", "t1"), rev(22, "토론 글이다. 숫자는 2이다.", "t2")]
    other = [rev(31, "바다군은 가상국의 군으로 인구가 5만 명이다.", "o1"),
             rev(32, "바다군은 가상국의 군으로 인구가 6만 명이다.", "o2")]
    xml = ("<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\"><siteinfo><sitename>위키백과</sitename></siteinfo>"
           + page(100, "하늘시", 0, city) + page(200, "토론:하늘시", 1, talk) + page(300, "바다군", 0, other)
           + "</mediawiki>")
    return bz2.compress(xml.encode("utf-8"))


def mine(limit_pages=0):
    rows = []
    with bz2.open(io.BytesIO(dump_bytes()), "rb") as stream:
        counts = mw.mine_pairs(stream, rows.append, limit_pages, log=io.StringIO())
    return rows, counts


class StripTest(unittest.TestCase):
    def test_strip_wiki_removes_markup_keeps_link_text(self):
        line = "'''하늘시'''는 [[가상국|가상 나라]]의 [[도시]]이다.{{출처 필요|날짜=2020}}<ref name=\"a\">책</ref><!-- 메모 -->"
        self.assertEqual(mw.strip_wiki(line), "하늘시는 가상 나라의 도시이다.")

    def test_judge_pair_kinds(self):
        self.assertEqual(mw.judge_pair("이 역은 3번째로 개통한 역이다.", "이 역은 4번째로 개통한 역이다."), ("b", "ordinal"))
        self.assertEqual(mw.judge_pair("2019년 인구는 10만 명이었다.", "2020년 인구는 11만 명이었다."), ("d", "num"))
        self.assertEqual(mw.judge_pair("인구는 1만 명 이상이다.", "인구는 1만 명 이하이다."), ("b", "qual"))
        self.assertEqual(mw.judge_pair("1966년 스페인 월드컵에서 탈락했다.", "1966년 잉글랜드 월드컵에서 탈락했다."), ("b", "name"))
        self.assertEqual(mw.judge_pair("서울(首爾)은 한국의 수도이다.", "서울은 한국의 수도 이다."), (None, "style_only"))
        self.assertEqual(mw.judge_pair("그는 매우 유명한 가수이다.", "그는 아주 유명한 가수이다."), (None, "other_word"))
        self.assertEqual(mw.judge_pair("그는 서울에서 태어났다.", "그는 서울에서 태어나지 않았다."), ("b", "onoff"))
        self.assertEqual(mw.judge_pair("후보를 뽑을 수없고 비례대표만 뽑는다.", "후보를 뽑을 수 없으며 비례대표만 뽑는다."),
                         (None, "other_word"))
        self.assertEqual(mw.judge_pair("상금이 수여된다. 처음 수상했다.", "상금이 수여되면서 시상식 자리에서 102만 달러를 기탁했다."),
                         (None, "other_word"))


class PairsTest(unittest.TestCase):
    def test_pairs_keep_one_value_edits_with_flags(self):
        rows, counts = mine()
        by_new = {r["src_ids"][1]: r for r in rows}
        self.assertEqual(set(by_new), {"kowiki:12", "kowiki:13", "kowiki:15", "kowiki:16", "kowiki:32"})
        self.assertEqual((by_new["kowiki:12"]["hint"], by_new["kowiki:12"]["kind"]), ("b", "ordinal"))
        self.assertEqual(by_new["kowiki:13"]["hint"], "d")
        self.assertEqual(by_new["kowiki:15"]["kind"], "name")
        self.assertEqual(by_new["kowiki:15"]["flag"], ["bot", "reverted"])
        self.assertEqual(by_new["kowiki:16"]["flag"], ["revert"])
        self.assertEqual(by_new["kowiki:12"]["flag"], [])
        self.assertEqual(counts["pages"], 3)
        self.assertEqual(counts["pages_main"], 2)
        self.assertEqual(counts["drop_markup_only"], 1)

    def test_row_shape_follows_mine_archive(self):
        rows, _ = mine()
        row = next(r for r in rows if r["src_ids"][1] == "kowiki:12")
        self.assertEqual(row["evidence"], "하늘시 : 이 도시는 4번째로 생긴 시이다.")
        self.assertEqual(row["claim"], "이 도시는 3번째로 생긴 시이다.")
        self.assertEqual((row["group"], row["lic"], row["domain"], row["lang"]),
                         ("kowiki:100", "CC-BY-SA-4.0", "kowiki", "ko"))
        self.assertEqual(row["src_ids"], ["kowiki:11", "kowiki:12"])

    def test_limit_pages_stops_early(self):
        rows, counts = mine(limit_pages=1)
        self.assertEqual(counts["pages"], 1)
        self.assertTrue(all(r["group"] == "kowiki:100" for r in rows))


class SplitStatsTest(unittest.TestCase):
    def test_split_keeps_each_page_in_one_set(self):
        rows = [{"group": f"kowiki:{i}", "hint": "b", "claim": "x", "evidence": "y"} for i in range(200) for _ in range(2)]
        eval_rows, train_rows = mw.split_rows(rows, 0)
        self.assertEqual(len(eval_rows) + len(train_rows), len(rows))
        self.assertFalse({r["group"] for r in eval_rows} & {r["group"] for r in train_rows})
        self.assertTrue(5 <= len({r["group"] for r in eval_rows}) <= 40)

    def test_cli_pairs_split_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = os.path.join(tmp, "d.xml.bz2")
            with open(dump, "wb") as f:
                f.write(dump_bytes())
            out = os.path.join(tmp, "c.jsonl")
            with open(os.devnull, "w") as null:
                saved, sys.stdout, sys.stderr = (sys.stdout, sys.stderr), null, null
                try:
                    mw.main(["pairs", "--dump", dump, "--out", out])
                    mw.main(["split", "--in", out, "--out-eval", out + ".e", "--out-train", out + ".t"])
                finally:
                    sys.stdout, sys.stderr = saved
            rows = mw.read_jsonl(out)
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(mw.read_jsonl(out + ".e")) + len(mw.read_jsonl(out + ".t")), 5)
            with open(out + ".summary.json", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["kept"], 5)
            stats = mw.stats(rows)
            self.assertEqual(stats["hint"], {"b": 4, "d": 1})
            self.assertEqual(stats["flag"]["none"], 3)


def row(i, hint="b", kind="num", flag=(), claim="인구는 10만 명이다.", after="인구는 11만 명이다."):
    return {"group": f"kowiki:{i}", "hint": hint, "kind": kind, "flag": list(flag), "title": "가",
            "claim": claim, "evidence": f"가 : {after}"}


class FilterTest(unittest.TestCase):
    def test_time_anchor_only(self):
        self.assertTrue(mw.time_anchor_only("2009년 현재 노선은 8개이다.", "2010년 현재 노선은 8개이다."))
        self.assertTrue(mw.time_anchor_only("2010년 1월 기준 문서 수는 9개다.", "2011년 3월 기준 문서 수는 9개다."))
        self.assertFalse(mw.time_anchor_only("2009년 현재 노선은 8개이다.", "2010년 현재 노선은 9개이다."))
        self.assertFalse(mw.time_anchor_only("2009년에 노선은 8개이다.", "2010년에 노선은 8개이다."))

    def test_filter_kinds_and_rehint_keep_d(self):
        rows = [row(1), row(2, kind="name"), row(3, kind="qual"), row(4, hint="d", kind="name"),
                row(5, kind="date", claim="2009년 현재 노선은 8개이다.", after="2010년 현재 노선은 8개이다.")]
        out, counts = mw.filter_rows(rows, {"num", "date"}, time_anchor=True)
        self.assertEqual([r["group"] for r in out], ["kowiki:1", "kowiki:4", "kowiki:5"])
        self.assertEqual((out[2]["hint"], out[2]["rehint"]), ("d", "time-anchor"))
        self.assertEqual((counts["drop_kind_name"], counts["drop_kind_qual"], counts["rehint_d"]), (1, 1, 1))
        self.assertEqual(rows[4]["hint"], "b")                        # 입력 줄은 안 바꾼다
        self.assertEqual(mw.filter_rows(rows)[0], rows)               # 옵션 없으면 그대로

    def test_flag_cap_keeps_share_and_is_seeded(self):
        rows = [row(i) for i in range(80)] + [row(100 + i, flag=["reverted"]) for i in range(40)]
        rows += [row(200 + i, hint="d", flag=["revert"]) for i in range(5)]
        out, counts = mw.filter_rows(rows, flag_cap=0.2)
        b_rows = [r for r in out if r["hint"] == "b"]
        self.assertEqual(sum(1 for r in b_rows if r["flag"]), 20)     # 20 / (80 + 20) = 20%
        self.assertEqual(sum(1 for r in out if r["hint"] == "d"), 5)  # d 는 상한 밖
        self.assertEqual(counts["drop_flag_cap"], 20)
        self.assertEqual(out, mw.filter_rows(rows, flag_cap=0.2)[0])

    def test_date_d_turns_to_b_unless_time_anchor_only(self):
        rows = [row(1, hint="d", kind="date", claim="그는 1965년 3월 16일에 태어났다.", after="그는 1966년 3월 5일에 태어났다."),
                row(2, hint="d", kind="num"),
                row(3, hint="d", kind="date", claim="2009년 3월 현재 노선은 8개다.", after="2010년 4월 현재 노선은 8개다.")]
        out, counts = mw.filter_rows(rows, date_d_to_b=True)
        self.assertEqual([(r["hint"], r.get("rehint")) for r in out], [("b", "date-d"), ("d", None), ("d", None)])
        self.assertEqual(counts["rehint_b"], 1)
        self.assertEqual(mw.filter_rows(rows)[0], rows)               # 옵션 없으면 그대로

    def test_meta_sentence_dropped(self):
        for text in ("2023년 5월 13일 기준이다.", "현재 앵커는 2018년 1월 29일 기준으로 한다.",
                     "다음은 2019년 기준 호남대학교의 학부 과정 일람이다.", "2020년 2월 1일 기준.",
                     "본 방송 시간은 2018년 7월 2일 이후를 기준으로 한다.",
                     "2012년 5월 기준, 가나항공은 다음의 취항지를 운항중이다."):
            self.assertTrue(mw.meta_sentence(text), text)
        for text in ("인구는 2017년 7월 1일 기준으로 2,333명이다.", "2009년 현재 노선은 8개이다.",
                     "2009년 12월 31일 현재 신자의 교세는 다음과 같다."):
            self.assertFalse(mw.meta_sentence(text), text)
        rows = [row(1, hint="d", kind="date", claim="2023년 2월 7일 기준이다.", after="2023년 5월 13일 기준이다."), row(2)]
        out, counts = mw.filter_rows(rows, drop_meta=True)
        self.assertEqual(([r["group"] for r in out], counts["drop_meta"]), (["kowiki:2"], 1))

    def test_glued_tail_meta_dropped_only_when_asked(self):
        glued = "K리그와 FA컵 기록이 포함된 기록이다.2017년 7월 24일 기준"
        self.assertFalse(mw.meta_sentence(glued))
        self.assertTrue(mw.meta_sentence(glued, glued_tail=True))
        self.assertFalse(mw.meta_sentence("라디오를 진행하고 있다.(2008년 4월 현재)", glued_tail=True))
        self.assertFalse(mw.meta_sentence("인구는 2017년 7월 1일 기준으로 2,333명이다.", glued_tail=True))
        rows = [row(1, hint="d", kind="date", claim=glued, after=glued.replace("7월 24일", "8월 6일")), row(2)]
        out, counts = mw.filter_rows(rows, drop_meta=True, drop_meta_tail=True)
        self.assertEqual(([r["group"] for r in out], counts["drop_meta_tail"]), (["kowiki:2"], 1))
        self.assertEqual(len(mw.filter_rows(rows, drop_meta=True)[0]), 2)

    def test_merge_dedups_claims_and_keeps_eval_copy(self):
        eval_g = next(i for i in range(100) if mw.eval_bucket(f"kowiki:{i}") == 0)
        train_g = [i for i in range(100) if mw.eval_bucket(f"kowiki:{i}") != 0][:3]
        same = "인구는 10만 명이다."
        c1 = [row(train_g[0], claim=same), row(train_g[1], claim="다른 문장이다 하나.")]
        c2 = [row(eval_g, claim=same), row(train_g[2], claim="다른  문장이다 하나.")]
        eval_rows, train_rows, counts = mw.merge_rows([c1, c2])
        self.assertEqual([r["group"] for r in eval_rows], [f"kowiki:{eval_g}"])
        self.assertEqual([r["group"] for r in train_rows], [f"kowiki:{train_g[1]}"])
        self.assertEqual((counts["drop_dup"], counts["drop_train_in_eval"]), (1, 1))

    def test_hyeonjae_ui_is_not_time_anchor(self):
        self.assertFalse(mw.time_anchor_only("1952년 설립되었으며, 1994년 현재의 상호로 변경하였다.",
                                             "1952년 설립되었으며, 1995년 현재의 상호로 변경하였다."))
        self.assertTrue(mw.time_anchor_only("1994년 현재 상호는 가나이다.", "1995년 현재 상호는 가나이다."))

    def test_cli_filter_uses_recommended_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "c.jsonl"), os.path.join(tmp, "f.jsonl")
            mw.write_jsonl(src, [row(1), row(2, kind="name"), row(3, kind="onoff"), row(4, kind="ordinal")])
            with open(os.devnull, "w") as null:
                saved, sys.stdout = sys.stdout, null
                try:
                    mw.main(["filter", "--in", src, "--out", dst])
                finally:
                    sys.stdout = saved
            self.assertEqual([r["group"] for r in mw.read_jsonl(dst)], ["kowiki:1", "kowiki:4"])


if __name__ == "__main__":
    unittest.main()
