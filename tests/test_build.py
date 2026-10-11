"""build — BM25 순위 · 원천 파서 · 묶음 예약/겹침 검사 · 누출 거르기. 입력은 모두 지어낸 짧은 글이다."""
import gzip
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import bm25  # noqa: E402
import jira_dump as jd  # noqa: E402
import make_sets  # noqa: E402
import mine_public as mp  # noqa: E402
import parquet_lite  # noqa: E402
import scrub  # noqa: E402


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


class Bm25Test(unittest.TestCase):
    def test_tokens_mix_words_and_bigrams(self):
        toks = bm25.tokenize("판정모델 cache")
        self.assertIn("판정모델", toks)
        self.assertIn("판정", toks)
        self.assertIn("cache", toks)
        self.assertIn("ca", toks)

    def test_rank_puts_best_match_first(self):
        docs = ["캐시 지연이 줄었다", "로그 파일을 지웠다", "캐시를 붙였다 캐시 지연"]
        index = bm25.BM25(docs)
        self.assertEqual(index.rank("캐시 지연")[0], 2)
        self.assertNotIn(1, index.rank("캐시 지연"))
        self.assertEqual(index.rank("캐시 지연", exclude=[2])[0], 0)

    def test_near_is_rank_band_10_to_60(self):
        docs = [" ".join(["alpha"] * (80 - i) + ["filler"] * i) for i in range(80)]
        index = bm25.BM25(docs)
        ranked = index.rank("alpha")
        self.assertEqual(index.near("alpha"), ranked[9:60])
        self.assertEqual(len(index.near("alpha")), 51)


class ParquetLiteTest(unittest.TestCase):
    def test_snappy_literal_and_overlapping_copy(self):
        raw = bytes([12, 0x0C]) + b"abcd" + bytes([0x11, 0x04])
        self.assertEqual(parquet_lite.snappy_decompress(raw), b"abcdabcdabcd")

    def test_rle_hybrid_run_and_bitpacked(self):
        self.assertEqual(parquet_lite.rle_hybrid(b"\x0a\x01", 0, 2, 1, 5), [1] * 5)
        self.assertEqual(parquet_lite.rle_hybrid(bytes([3, 0b10110010]), 0, 2, 1, 8), [0, 1, 0, 0, 1, 1, 0, 1])


class FilterTest(unittest.TestCase):
    def test_common_filter(self):
        self.assertTrue(mp.text_ok("The cache keeps the last ten results."))
        self.assertFalse(mp.text_ok("short"))
        self.assertFalse(mp.text_ok("See https://example.com for the details here."))
        self.assertFalse(mp.text_ok("fix typo"))
        self.assertFalse(mp.text_ok("x = foo(); y = bar(); z == {}"))
        self.assertFalse(mp.text_ok("The following are allowed:"))
        self.assertFalse(mp.text_ok("A" * 401))


PEP_A = """PEP: 1
Title: Old cache rule
Author: Someone <someone@example.com>
Superseded-By: 2

Abstract
========

This PEP proposes a cache for the parser. The cache keeps the last ten parse results.

Motivation
==========

Parsing the same file twice wastes time in large builds. Users reported slow builds when the parser ran again.

Specification
=============

The parser stores each result in a cache keyed by file path. The cache keeps the last ten parse results in memory.
Entries are dropped when the file changes on disk.

Rejected Ideas
==============

Storing the cache on disk was rejected because it made builds slower.

Copyright
=========

This document is placed in the public domain or under the CC0-1.0-Universal license.
"""

PEP_B = PEP_A.replace("PEP: 1", "PEP: 2").replace("Superseded-By: 2", "Replaces: 1").replace(
    "Old cache rule", "New cache rule")


class ProposalsTest(unittest.TestCase):
    def test_sections_and_roles(self):
        sections = mp.parse_sections(PEP_A.split("\n\n", 1)[1], "rst")
        roles = [role for _, role, _ in sections]
        self.assertEqual(roles[:4], ["intro", "motivation", "spec", "rejected"])
        self.assertEqual(sections[0][2][1], "The cache keeps the last ten parse results.")

    def test_supersede_chain_is_one_group_and_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "python__peps", "peps", "pep-0001.rst"), PEP_A)
            write(os.path.join(tmp, "python__peps", "peps", "pep-0002.rst"), PEP_B)
            col = mp.mine_proposals(tmp, random.Random(1))
        self.assertEqual(set(col.groups), {"prop:pep-0001"})
        forms = {(r["hint"], r["form"]) for r in col.rows}
        self.assertIn(("d", "link"), forms)
        self.assertIn(("d", "same-doc"), forms)
        self.assertIn(("b", "rejected"), forms)
        for row in col.rows:
            self.assertEqual(row["want"], "")
            self.assertEqual(row["lic"], "CC0-1.0")
            self.assertNotIn("someone@example.com", row["evidence"] + row["claim"])


class VitaminCTest(unittest.TestCase):
    def test_real_only_and_page_group(self):
        def fake_reader(path):
            split = os.path.basename(path)
            rows = []
            for i in range(4):
                rows.append({"unique_id": f"{split}-{i}", "page": "Lake Town",
                             "claim": f"Lake Town had {100 + i} residents in the census year.",
                             "evidence": f"The census counted {100 + i} people living in Lake Town that year.",
                             "label": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "SUPPORTS"][i],
                             "revision_type": "real" if i < 3 else "synthetic"})
            return rows
        col = mp.mine_vitaminc("raw", random.Random(1), read_rows=fake_reader)
        labeled = [r for r in col.rows if r["form"] in ("labeled", "same-doc")]
        self.assertEqual(len(labeled), 9)                         # 셋 분할 × real 셋
        self.assertTrue(all(r["group"] == "vitc:Lake Town" for r in col.rows))
        self.assertEqual(col.groups["vitc:Lake Town"]["pool"], "mixed")
        self.assertEqual({r["hint"] for r in labeled}, {"a", "b", "d"})


class KlueTest(unittest.TestCase):
    def test_unanswerable_row_needs_rewrite(self):
        context = ("서울시는 올해 도서관 열 곳을 새로 연다고 밝혔다. 새 도서관은 모두 지하철역 근처에 들어선다. "
                   "개관 행사는 다음 달 첫 주에 열린다. 시는 예산 백억 원을 썼다고 설명했다.")
        article = {"title": "서울시 도서관 열 곳 개관", "news_category": "사회", "source": "x",
                   "paragraphs": [{"context": context, "qas": [
                       {"question": "새 도서관의 운영 시간은 몇 시까지인가?", "answers": [], "question_type": 3,
                        "is_impossible": True, "guid": "q-imp"},
                       {"question": "서울시가 쓴 예산은 얼마인가?", "answers": [{"text": "백억 원",
                        "answer_start": context.index("백억 원")}], "question_type": 1,
                        "is_impossible": False, "guid": "q-ans"}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "klue-mrc-v1.1_train.json"), json.dumps({"data": [article]}, ensure_ascii=False))
            write(os.path.join(tmp, "klue-mrc-v1.1_dev.json"), json.dumps({"data": []}))
            col = mp.mine_klue(tmp, random.Random(1))
        imp = [r for r in col.rows if r["form"] == "unanswerable"]
        self.assertEqual(len(imp), 1)
        self.assertTrue(imp[0]["needs_rewrite"])
        self.assertEqual(imp[0]["claim"], "")
        self.assertIn("도서관", imp[0]["entities"])
        self.assertEqual(imp[0]["group"], "klue:" + mp.short_hash("서울시 도서관 열 곳 개관"))
        self.assertEqual(imp[0]["split"], "train")


class MiniCheckTest(unittest.TestCase):
    def test_claim_and_doc_union_into_one_group(self):
        doc_a = ("The bridge opened in 1990 after six years of work. It carries two rail lines. "
                 "Engineers chose steel for the main span. The city paid for most of the cost.")
        doc_b = doc_a.replace("1990", "1991")
        claim = "The bridge opened in 1990 and carries rail lines."

        def fake_reader(path):
            if path.endswith("c2d.parquet"):
                return [{"claim": claim, "doc": doc_a, "label": 1}, {"claim": claim, "doc": doc_b, "label": 0}]
            return []
        col = mp.mine_minicheck("raw", random.Random(1), read_rows=fake_reader)
        self.assertEqual(len(col.groups), 1)


class JiraTest(unittest.TestCase):
    BODY = "The settings page shows the wrong theme after reload. It happens on every browser we tried so far."

    def issue(self, key, summary, links=(), resolution=None, comps=("ui",), body=None):
        return {"key": key, "repo": "R", "fields": {
            "summary": summary, "description": self.BODY if body is None else body,
            "project": {"key": "APP"}, "components": [{"name": c} for c in comps],
            "resolution": {"name": resolution} if resolution else None,
            "issuelinks": [{"type": {"name": kind}, "outwardIssue": {"key": other}} for kind, other in links],
            "comment": {"comments": []}}}

    def mine(self, issues, name="issues.jsonl"):
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, name), "".join(json.dumps(i) + "\n" for i in issues))
            return mp.mine_jira(tmp, random.Random(1))

    def test_relates_is_link_d_and_links_join_groups(self):
        issues = [self.issue("APP-1", "Theme resets after reload", [("Relates", "APP-2")]),
                  self.issue("APP-2", "Theme setting is not saved on reload", [("Relates", "APP-1")]),
                  self.issue("APP-3", "Export button crashes the app", comps=("export",))]
        col = self.mine(issues)
        links = [r for r in col.rows if r["form"] == "link"]
        self.assertEqual(len(links), 1)                      # 양쪽에 걸린 링크도 한 쌍
        self.assertEqual(links[0]["hint"], "d")
        self.assertEqual(links[0]["group"], "jira:APP-1")
        self.assertIn("jira:APP-3", col.groups)
        self.assertTrue(any(r["form"] == "other-component" for r in col.rows))

    def test_reference_and_duplicate_kinds_other_kinds_skipped(self):
        issues = [self.issue("APP-1", "Theme resets after reload", [("Reference", "APP-2"), ("Blocks", "APP-3")]),
                  self.issue("APP-2", "Theme setting is not saved on reload", [("Duplicate", "APP-3")]),
                  self.issue("APP-3", "Theme is lost after a reload")]
        col = self.mine(issues)
        kinds = sorted((r["hint"], r["form"]) for r in col.rows if r["form"] in ("link", "duplicate"))
        self.assertEqual(kinds, [("a", "duplicate"), ("d", "link")])
        self.assertEqual(len({r["group"] for r in col.rows}), 1)   # Relate · Duplicate 로 한 덩이

    def test_body_length_and_markup_filter(self):
        issues = [self.issue("APP-1", "Theme resets after reload", [("Relates", "APP-2")]),
                  self.issue("APP-2", "Theme setting is not saved", body="Too short."),
                  self.issue("APP-4", "Theme resets on save", [("Relates", "APP-5")]),
                  self.issue("APP-5", "Theme lost", body=self.BODY + " {code}x{code}")]
        col = self.mine(issues)
        self.assertFalse([r for r in col.rows if r["form"] == "link"])


class JiraDumpTest(unittest.TestCase):
    @staticmethod
    def bson(pairs):
        body = b""
        for name, value in pairs:
            key = name.encode() + b"\x00"
            if isinstance(value, str):
                raw = value.encode() + b"\x00"
                body += b"\x02" + key + struct.pack("<i", len(raw)) + raw
            elif isinstance(value, list):
                body += b"\x04" + key + JiraDumpTest.bson([(str(i), v) for i, v in enumerate(value)])
            elif isinstance(value, dict):
                body += b"\x03" + key + JiraDumpTest.bson(list(value.items()))
            elif isinstance(value, bool):
                body += b"\x08" + key + (b"\x01" if value else b"\x00")
            else:
                body += b"\x10" + key + struct.pack("<i", value)
        return struct.pack("<i", len(body) + 5) + body + b"\x00"

    def test_archive_stream_to_slim_issue(self):
        issue = {"_id": 7, "key": "APP-1", "changelog": {"histories": [{"x": "y"}]}, "fields": {
            "project": {"key": "APP"}, "summary": "Theme resets", "description": "Body text.",
            "components": [{"name": "ui"}], "resolution": None,
            "issuelinks": [{"type": {"name": "Relates"}, "inwardIssue": {"key": "APP-2"}}],
            "comment": {"comments": []}}}
        issue["fields"]["resolution"] = {"name": "Fixed"}
        term = struct.pack("<I", jd.TERMINATOR)
        archive = (struct.pack("<I", jd.MAGIC) + self.bson([("version", "0.1")]) + term
                   + self.bson([("db", "D"), ("collection", "Jira"), ("EOF", False)]) + self.bson(list(issue.items()))
                   + term + self.bson([("db", "D"), ("collection", "Jira"), ("EOF", True)]) + term)
        docs = list(jd.iter_docs(io.BytesIO(gzip.decompress(gzip.compress(archive)))))
        self.assertEqual([c for c, _ in docs], ["Jira"])
        slim = jd.slim_issue(jd.decode(docs[0][1], 0, jd._WANT), "Jira")
        self.assertEqual(slim["repo"], "Jira")
        self.assertEqual(slim["fields"]["issuelinks"], [{"type": {"name": "Relates"}, "inwardIssue": {"key": "APP-2"}}])
        self.assertEqual(mp._jira_issue(slim)["links"], [("Relates", "APP-2")])

    def test_broken_doc_size_stops(self):
        # 길이 칸이 5 보다 작으면(깨진 아카이브) 흐름 끝까지 읽지 않고 멈춘다
        term = struct.pack("<I", jd.TERMINATOR)
        archive = (struct.pack("<I", jd.MAGIC) + self.bson([("version", "0.1")]) + term
                   + struct.pack("<I", 2) + b"\x00" * 64)
        with self.assertRaises(EOFError):
            list(jd.iter_docs(io.BytesIO(archive)))


def make_build(tmp):
    """지어낸 후보 둘 원천 · 묶음 여럿."""
    for source, pool_name in (("proposals", "any"), ("vitaminc", "heldout")):
        rows, groups = [], {}
        for g in range(40):
            gid = f"{source}:{g}"
            groups[gid] = {"pool": pool_name if g < 30 else "train", "title": gid, "lic": "x"}
            for k in range(3):
                rows.append({"id": f"{source}-{g}-{k}", "evidence": f"{source} evidence {g} {k} sentence text",
                             "claim": f"{source} claim {g} {k} sentence text", "hint": "d",
                             "form": "same-doc", "group": gid, "groups": [gid],
                             "split": "any" if source == "proposals" else ("heldout" if g < 30 else "train")})
        write(os.path.join(tmp, source, "candidates.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        write(os.path.join(tmp, source, "groups.json"), json.dumps({"groups": groups}))


class MakeSetsTest(unittest.TestCase):
    def test_reserve_then_pool_has_zero_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            doc = make_sets.reserve(tmp, seed=1, factor=0.02)
            self.assertFalse(doc["jira"])
            self.assertAlmostEqual(doc["shares"]["proposals"], 0.21)
            reserved = set(doc["reserved"])
            self.assertTrue(reserved)
            for name in ("dev-cal", "dev-cmp", "blind-g1"):
                for gid in doc["sets"][name]["vitaminc"]["groups"]:
                    self.assertLess(int(gid.split(":")[1]), 30)   # heldout 묶음에서만
            chosen, _ = make_sets.pool(tmp, doc, seed=1, total=100)
            self.assertTrue(chosen)
            result = make_sets.check(chosen, doc, tmp)
            self.assertEqual(result, {"group_reserved": 0, "text_reserved": 0, "text_old": 0, "mem_id_old": 0})

    def test_check_catches_group_text_old_and_mem_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            doc = make_sets.reserve(tmp, seed=1, factor=0.02)
            gid = sorted(doc["reserved"])[0]
            source = gid.split(":")[0]
            g, k = gid.split(":")[1], 0
            bad = [
                {"evidence": "x evidence text here", "claim": "y claim text here", "group": gid, "groups": [gid]},
                {"evidence": f"{source} evidence {g} {k} sentence text", "claim": f"{source} claim {g} {k} sentence text",
                 "group": "other:1", "groups": ["other:1"]},
                {"evidence": "Old Evidence  text", "claim": "old claim", "group": "other:2", "groups": ["other:2"],
                 "src_ids": ["20261001-abcdef12"]},
            ]
            old_path = os.path.join(tmp, "old", "train.jsonl")
            write(old_path, json.dumps({"evidence": "old evidence text", "claim": "Old claim",
                                        "src_ids": ["20261001-abcdef12"]}) + "\n")
            keys, ids, missing = make_sets.load_old(tmp, {"t": "old/train.jsonl", "gone": "old/none.jsonl"})
            self.assertEqual(missing, ["gone"])
            result = make_sets.check(bad, doc, tmp, keys, ids)
        self.assertEqual(result, {"group_reserved": 1, "text_reserved": 1, "text_old": 1, "mem_id_old": 1})


class ScrubTest(unittest.TestCase):
    def test_rules(self):
        s = scrub.Scrubber(names=["Bluefin"], game_names=["Moonbake"])
        counts = __import__("collections").Counter()
        row = {"evidence": "Author: Kim <kim@example.com>\n로그는 C:\\Users\\kim\\proj\\a.txt 에 남는다. Moonbake 빌드",
               "claim": "/home/kim/logs 를 본다"}
        out = s.apply(row, counts)
        self.assertNotIn("Author", out["evidence"])
        self.assertIn("<경로>", out["evidence"])
        self.assertIn("<게임>", out["evidence"])
        self.assertEqual(out["claim"], "<경로> 를 본다")
        self.assertIsNone(s.apply({"evidence": "토큰 ghp_" + "a" * 36, "claim": "c"}, counts))
        self.assertIsNone(s.apply({"evidence": "bluefin 프로젝트 문서", "claim": "c"}, counts))
        self.assertEqual(sum(s.hits(out).values()), 0)
        self.assertGreater(sum(s.hits(row).values()), 0)

    def test_read_names_skips_table_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "names.md")
            write(path, "# 목록\n| 이름 | 메모 |\n| --- | --- |\n| `Alpha` | x |\nBeta\n")
            self.assertEqual(sorted(scrub.read_names([path])), ["Alpha", "Beta"])


import collections  # noqa: E402
import subprocess  # noqa: E402

import mine_archive as ma  # noqa: E402


def git_commit(repo, files, message):
    for name, text in files.items():
        write(os.path.join(repo, name), text)
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "-q", "-m", message],
                   check=True, capture_output=True)


MEM_MD = """---
id: {mid}
type: decision
title: {title}
summary: {summary}
tags: [cache]
scope: toolx
date: 2026-10-01
sources:
  - file:Docs/cache.md
{extra}---

왜 : 캐시가 커지면 메모리가 모자라서 열 개까지만 남기기로 했다.

다음 : 넘친 결과는 디스크로 옮기는 판을 따로 잰다.
"""


class MineArchiveTest(unittest.TestCase):
    def test_filter_drops_menu_and_keeps_sentences(self):
        self.assertTrue(ma.arch_text_ok("캐시는 마지막 열 개 결과를 남긴다."))
        self.assertTrue(ma.arch_text_ok("stage lint 가 새 파일 검사를 건너뛰던 버그 고침"))
        self.assertFalse(ma.arch_text_ok("캐시 · 로그 · 색인"))                     # 서술어 없는 토막
        self.assertFalse(ma.arch_text_ok("설정, 빌드, 시험, 배포, 문서 정리 목록과 메뉴 이름 모음 그리고 목차"))
        self.assertTrue(ma.is_line_heavy("가\n나\n다\n라\n마"))

    def test_git_same_file_d_and_revert_b(self):
        with tempfile.TemporaryDirectory() as repo:
            subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True)
            git_commit(repo, {"cache.py": "a=1\n"}, "캐시 크기를 열 개로 정했다\n\n- 결과가 넘치면 오래된 것부터 지운다.")
            git_commit(repo, {"cache.py": "a=2\n"}, "캐시 지우기 순서를 오래된 것부터로 바꿨다")
            git_commit(repo, {"cache.py": "a=1\n"}, "Revert 캐시 지우기 순서 바꾼 것을 되돌렸다")
            git_commit(repo, {"log.py": "b=1\n"}, "로그 파일 이름에 날짜를 붙였다")
            git_commit(repo, {"log.py": "b=2\n"}, "로그 파일을 하루마다 새로 연다")
            col = ma.mine_git([("t", repo)], random.Random(1))
        forms = collections.Counter((r["hint"], r["form"]) for r in col.rows)
        self.assertGreater(forms[("d", "same-file")], 0)
        self.assertEqual(forms[("b", "revert")], 1)
        self.assertTrue(all(r["group"].startswith("git:t:") and r["domain"] == "archive" for r in col.rows))
        self.assertTrue(all(r["want"] == "" and r["origin"] == "g1-ga" for r in col.rows))

    def test_docs_same_doc_and_handover_link(self):
        design = ("# 캐시 설계\n\n이 문서는 파서 캐시를 어떻게 둘지 정한 글이다.\n\n## 1. 크기\n\n"
                  "캐시는 마지막 열 개 결과를 남긴다. 넘치면 오래된 것부터 지운다.\n\n## 2. 저장\n\n"
                  "캐시 결과는 디스크에 쓰지 않고 메모리에만 둔다. 다시 켜면 캐시는 비어 있다.\n\n"
                  "## 3. 정한 것\n\n| 주제 | 정한 것 |\n| --- | --- |\n| 캐시 크기 | 열 개로 둔다 |\n")
        handover = "# 인수인계\n\n## 남은 일\n\n- 캐시 크기는 cache.md 에 적힌 대로 열 개로 둔다.\n"
        with tempfile.TemporaryDirectory() as root:
            write(os.path.join(root, "ToolX", "Docs", "Design", "cache.md"), design)
            write(os.path.join(root, "ToolX", "Docs", "History", "2026-10-01-인수인계.md"), handover)
            write(os.path.join(root, "ToolX", "Docs", "Guide", "secret.local.md"), "## 비밀\n\n여기는 읽으면 안 된다.\n")
            col = ma.mine_docs([("t", root)], random.Random(1))
        forms = collections.Counter((r["hint"], r["form"]) for r in col.rows)
        self.assertGreater(forms[("d", "same-doc")], 0)
        self.assertEqual(forms[("d", "link")], 1)
        link = next(r for r in col.rows if r["form"] == "link")
        self.assertEqual(len(link["groups"]), 2)
        self.assertFalse(any("secret.local" in g for g in col.groups))

    def test_mem_relabel_and_fresh_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "store")
            ids = ["20261001-0000000a", "20261001-0000000b", "20261001-0000000c"]
            write(os.path.join(store, "a.md"), MEM_MD.format(mid=ids[0], title="캐시는 열 개까지만 남긴다",
                  summary="캐시 크기를 열 개로 정했다", extra=f"superseded_by: {ids[1]}\n"))
            write(os.path.join(store, "b.md"), MEM_MD.format(mid=ids[1], title="캐시 크기를 스무 개로 늘렸다",
                  summary="캐시 크기를 열 개에서 스무 개로 바꿨다", extra=""))
            write(os.path.join(store, "c.md"), MEM_MD.format(mid=ids[2], title="로그는 하루마다 새 파일에 쓴다",
                  summary="로그 파일을 날짜별로 나눴다", extra=""))
            dev = os.path.join(tmp, "dev-r2.jsonl")
            write(dev, json.dumps({"id": "v00001", "evidence": "옛 근거 글이다", "claim": "옛 주장 글이다",
                                   "src_ids": [ids[2]], "kind": "near", "lang": "ko", "want": "related"},
                                  ensure_ascii=False) + "\n")
            col = ma.mine_mem([("t", store)], dev, {ids[0]}, random.Random(1))
        relabel = [r for r in col.rows if r.get("relabel")]
        self.assertEqual(len(relabel), 1)
        self.assertEqual((relabel[0]["hint"], relabel[0]["old_want"], relabel[0]["split"]), ("d", "related", "dev"))
        fresh = [r for r in col.rows if not r.get("relabel")]
        self.assertTrue(fresh)
        used = {i for r in fresh for i in r["src_ids"]}
        self.assertEqual(used, {ids[1]})                       # 옛 셋(a) · dev-r2(c) 기억은 새 후보에 안 씀
        self.assertEqual(col.groups[relabel[0]["group"]]["pool"], "mixed")     # 같은 출처 파일이라 한 덩이

    def test_scrub_drops_names_and_masks_game(self):
        col = ma.ArchiveCollector("docs", "archive")
        col.add("Bluefin 프로젝트의 캐시 구조를 참고했다.", "캐시는 열 개까지만 남긴다.", "d", "near", ["g1"], "", "any", ["x"])
        col.add("Moonbake 빌드는 캐시를 쓰지 않는다.", "캐시는 열 개까지만 남긴다.", "d", "near", ["g2"], "", "any", ["y"])
        col.scrub(scrub.Scrubber(["Bluefin"], ["Moonbake"]))
        self.assertEqual(len(col.rows), 1)
        self.assertTrue(col.rows[0]["evidence"].startswith("<게임>"))


def make_arch_build(tmp):
    def row(rid, group, split="any", relabel=False, src=None):
        out = {"id": rid, "evidence": f"근거 {rid}", "claim": f"주장 {rid}", "hint": "d", "form": "near",
               "group": group, "groups": [group], "split": split, "src_ids": src or [rid]}
        if relabel:
            out["relabel"] = True
        return out
    mem = [row(f"m{i}", f"mem:g{i % 40}") for i in range(120)]
    mem += [row(f"r{i}", f"mem:r{i % 5}", split="dev", relabel=True) for i in range(30)]
    arch = [row(f"a{i}", f"git:t:f{i % 300}") for i in range(3000)]
    for folder, rows in (("mem", mem), ("archive/git", arch)):
        write(os.path.join(tmp, folder, "candidates.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        write(os.path.join(tmp, folder, "groups.json"), json.dumps({"groups": {r["group"]: {} for r in rows}}))


class ReserveArchiveTest(unittest.TestCase):
    def test_sets_disjoint_blind_fresh_only_and_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_arch_build(tmp)
            doc = make_sets.reserve_archive(tmp, seed=1)
            rows = {r["id"]: r for f in ("mem", "archive/git")
                    for r in make_sets.read_jsonl(os.path.join(tmp, f, "candidates.jsonl"))}
        self.assertEqual(make_sets.arch_cross_check(doc), {"row_across_sets": 0, "group_across_sets": 0})
        blind = doc["sets"]["blind-g1"]["mem"]["row_ids"]
        self.assertFalse(any(rows[i].get("relabel") for i in blind))
        dev_relabel = [i for s in ("dev-cal", "dev-cmp") for i in doc["sets"][s]["mem"]["row_ids"] if rows[i].get("relabel")]
        self.assertEqual(len(dev_relabel), 30)                 # 다시 붙이기 줄은 하나도 안 잘린다
        self.assertGreater(sum(doc["shift"].values()), 0)      # mem 이 모자라면 아카이브로 넘긴다
        for set_name, short in doc["shift"].items():
            self.assertEqual(doc["sets"][set_name]["archive"]["pairs"],
                             round(make_sets.SET_SIZES[set_name] * 0.15) + short)


EVAL_WORDS = ("cache", "router", "logger", "parser", "scheduler", "codec", "buffer", "socket", "queue", "lock",
              "index", "mirror")
EVAL_KINDS = {"proposals": (("a", "same-sec"), ("b", "rejected"), ("d", "same-doc"), ("d", "near"), ("c", "other-doc")),
              "vitaminc": (("a", "labeled"), ("b", "labeled"), ("c", "same-page"), ("d", "same-doc"), ("d", "near")),
              "klue-mrc": (("a", "qa-answer"), ("d", "unanswerable"), ("d", "removed"))}
EVAL_SETS = ("dev-cal", "dev-cmp", "blind-g1")


def make_eval_build(tmp):
    """원천 셋 × 묶음 12(셋마다 4) — KLUE 는 claim 이 비어 있고 rewrite-reserved 가 채운다(한 줄은 skip)."""
    doc = {"sets": {s: {} for s in EVAL_SETS}}
    rewrites = []
    for source, kinds in EVAL_KINDS.items():
        rows = []
        for g, word in enumerate(EVAL_WORDS):
            gid = f"{source[:4]}:{word}"
            for k, (hint, form) in enumerate(kinds * 2):
                rid = f"{source}-{g:02d}{k:02d}"
                ev = (f"The {word} module stores entry {k} for group {g} and keeps it warm."
                      if source != "klue-mrc" else f"{word} 모듈은 {k}번 항목을 {g}번 묶음에 넣어 둔다.")
                claim = f"The {word} module was changed in step {k} to handle more traffic."
                row = {"id": rid, "evidence": ev, "claim": claim, "want": "", "hint": hint, "form": form,
                       "domain": source, "group": gid, "groups": [gid], "lang": "en", "lic": "x", "split": "any",
                       "src_ids": [rid]}
                if source == "klue-mrc":
                    row.update(claim="", needs_rewrite=True, lang="ko")
                    rewrites.append({"id": rid, "claim": f"{word} 모듈은 {k}단계에서 바뀌었다.",
                                     "skip": k == 1 and g == 0})
                rows.append(row)
        write(os.path.join(tmp, source, "candidates.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        write(os.path.join(tmp, source, "groups.json"), json.dumps({"groups": {r["group"]: {} for r in rows}}))
        for i, set_name in enumerate(EVAL_SETS):
            groups = [f"{source[:4]}:{w}" for w in EVAL_WORDS[i * 4:(i + 1) * 4]]
            doc["sets"][set_name][source] = {"pairs": 10 if source != "klue-mrc" else 5, "groups": groups}
    write(os.path.join(tmp, "klue-mrc", "rewrite-reserved.jsonl"), "".join(json.dumps(r) + "\n" for r in rewrites))
    return doc


class EvalPoolTest(unittest.TestCase):
    def test_sizes_and_hint_ratio_follow_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = make_eval_build(tmp)
            sets, report = make_sets.eval_pool(tmp, doc, seed=1)
        for set_name in EVAL_SETS:
            rows = sets[set_name]
            self.assertEqual(len(rows), 16 + 16 + 8)                      # pairs × 1.6 (올림)
            ratio = make_sets.EVAL_HINT[set_name]
            for hint, n in ratio.items():
                got = sum(1 for r in rows if r["hint"] == hint)
                self.assertLessEqual(abs(got - 40 * n / sum(ratio.values())), 2, (set_name, hint))
            klue = [r for r in rows if r["domain"] == "klue-mrc"]
            self.assertFalse([r for r in klue if r["hint"] == "b"])        # KLUE 는 b 가 없고 그 몫은 다른 원천이 진다
            self.assertTrue(all(r["claim"] and not r.get("needs_rewrite") for r in klue))
            self.assertNotIn("klue-mrc-0001", {r["id"] for r in klue})    # skip 줄은 안 쓴다
        self.assertEqual(make_sets.eval_cross(sets)["eval_group_across_sets"], 0)

    def test_c_is_redrawn_inside_the_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = make_eval_build(tmp)
            sets, _ = make_sets.eval_pool(tmp, doc, seed=2)
        for set_name, rows in sets.items():
            near = [r for r in rows if r["form"] == make_sets.NEAR_C]
            c_all = [r for r in rows if r["hint"] == "c"]
            self.assertGreaterEqual(len(near), len(c_all) / 2)           # c 의 절반 이상이 근접 c
            mine = {g for per in doc["sets"][set_name].values() for g in per["groups"]}
            for row in near:
                self.assertEqual(len(row["groups"]), 2)                   # 두 묶음을 다 적는다
                self.assertTrue(set(row["groups"]) <= mine)               # 같은 셋 예약 묶음끼리만
                self.assertTrue(10 <= row["near_rank"] <= 60)
            self.assertFalse([r for r in c_all if r["form"] == "other-doc" and not set(r["groups"]) <= mine])

    def test_list_like_evidence_is_dropped(self):
        listy = ["Colorado RapidsFC DallasHouston DynamoLos Angeles FCLA Galaxy",
                 "Second half — 2, Real Salt Lake, Findley 13, 64th. Shootout — Los Angeles 4, Beckham (Goal)",
                 "Media Play News – About Us – Contact",
                 "Home\nAbout\nContact us\nNews", "홈 · 소개 · 문의"]
        prose = ["The American Civil War was fought in the United States from 1861 to 1865 .",
                 "Formed in 2001 by Haim Saban , Saban Capital Group owns Saban Brands .",
                 "남한군은 바다로 도망치려고 했지만 탈출하지 못했다.",
                 "한국인 김보미 씨가 모차르트 팀의 상임지휘자로 발탁된 것."]
        for text in listy:
            self.assertTrue(make_sets.list_like(text), text)
        for text in prose:
            self.assertFalse(make_sets.list_like(text), text)
        with tempfile.TemporaryDirectory() as tmp:
            doc = make_eval_build(tmp)
            path = os.path.join(tmp, "proposals", "candidates.jsonl")
            rows = make_sets.read_jsonl(path)
            rows[0]["evidence"] = "Runtime information"
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            sets, report = make_sets.eval_pool(tmp, doc, seed=1)
        self.assertNotIn(rows[0]["id"], {r["id"] for r in sets["dev-cal"]})
        self.assertEqual(report["sets"]["dev-cal"]["proposals"]["list_dropped"], 1)


def make_arch_eval_build(tmp):
    """mem(새 줄 + dev-r2 다시 붙이기 줄) · 아카이브(b 없음) 후보와 reserve-archive 꼴 예약 — 문장 꼴 짧은 글."""
    words = ("캐시", "로그", "빌드", "저장", "검색", "묶음", "시험", "배포", "문서", "설정", "기억", "판정")
    kinds = (("a", "title-body"), ("c", "other-topic"), ("d", "same-doc"), ("d", "near"))
    mem, arch = [], []
    for g, word in enumerate(words):
        for k, (hint, form) in enumerate(kinds * 3):
            rid = f"m{g:02d}{k:02d}"
            mem.append({"id": rid, "evidence": f"{word} 기능은 {k}번째 판에서 {g}개 묶음을 다시 읽도록 바꾸었다.",
                        "claim": f"{word} 설정은 {k}단계 뒤에 기본값으로 돌아간다.", "hint": hint, "form": form,
                        "domain": "mem", "group": f"mem:{word}", "groups": [f"mem:{word}"], "lang": "ko",
                        "split": "any", "src_ids": [f"2026101{g % 10}-{k:08x}"]})
            arch.append({"id": f"git-{g:02d}{k:02d}", "evidence": f"{word} 모듈이 {k}번 커밋에서 {g}줄을 고쳤다.",
                         "claim": f"{word} 모듈의 {k}번 시험은 그대로 통과한다.",
                         "hint": hint, "form": "same-file" if hint == "d" else form, "domain": "archive",
                         "group": f"git:{word}", "groups": [f"git:{word}"], "lang": "ko", "split": "any",
                         "src_ids": [f"git-{g:02d}{k:02d}"]})
    relabel = [{"id": f"r{i:02d}", "evidence": f"옛 판 {i}번 기억은 캐시 크기를 {i}로 정했다.",
                "claim": f"캐시 크기는 {i + 1}로 정해졌다.", "hint": "b", "form": "r2-flip", "domain": "mem",
                "group": "mem:old", "groups": ["mem:old"], "lang": "ko", "split": "dev", "relabel": True,
                "old_want": "b", "old_id": f"r2-{i}", "src_ids": [f"20261001-{i:08x}"]} for i in range(6)]
    mem += relabel
    for folder, rows in (("mem", mem), ("archive/git", arch)):
        write(os.path.join(tmp, folder, "candidates.jsonl"), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        write(os.path.join(tmp, folder, "groups.json"), json.dumps({"groups": {r["group"]: {} for r in rows}}))
    split = {"dev-cal": words[:4], "dev-cmp": words[4:8], "blind-g1": words[8:]}
    doc = {"sets": {}}
    for set_name, ws in split.items():
        m_ids = [r["id"] for r in mem if r["group"] in {f"mem:{w}" for w in ws}]
        if set_name == "dev-cmp":
            m_ids += [r["id"] for r in relabel]
        doc["sets"][set_name] = {
            "mem": {"pairs": 10 if set_name != "dev-cmp" else 12, "row_ids": m_ids},
            "archive": {"pairs": 10, "row_ids": [r["id"] for r in arch if r["group"] in {f"git:{w}" for w in ws}]}}
    return doc


class EvalArchiveTest(unittest.TestCase):
    def test_relabel_kept_whole_fresh_scaled_and_b_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = make_arch_eval_build(tmp)
            sets, report = make_sets.eval_pool_archive(tmp, doc, seed=3)
        cmp_rows = sets["dev-cmp"]
        relabel = [r for r in cmp_rows if r.get("relabel")]
        self.assertEqual(len(relabel), 6)                                  # 다시 붙이기 줄은 전부, 1배
        self.assertTrue(all(r["old_want"] == "b" for r in relabel))
        fresh_mem = [r for r in cmp_rows if r["domain"] == "mem" and not r.get("relabel")]
        self.assertEqual(len(fresh_mem), 10)                               # (12 − 6) × 1.6 올림
        for set_name in ("dev-cal", "blind-g1"):
            self.assertEqual(len(sets[set_name]), 16 + 16)                 # 두 도메인 각 10 × 1.6
            self.assertGreater(report["sets"][set_name]["archive"]["b_moved_to_dc"], 0)   # b 후보가 없으면 d · c 로
            self.assertFalse([r for r in sets[set_name] if r["hint"] == "b"])
        for rows in sets.values():
            near = [r for r in rows if r["form"] == make_sets.NEAR_C and r["domain"] == "archive"]
            self.assertTrue(all(len(r["groups"]) == 2 for r in near))
            self.assertFalse([r for r in near if any(s.startswith("2026100") for s in r["src_ids"])])
        cross = make_sets.eval_cross(sets, old_keys={make_sets.pair_key(relabel[0])},
                                     old_ids={relabel[0]["src_ids"][0]}, skip_relabel_old=True)
        self.assertEqual(sum(cross.values()), 0)                           # relabel 줄은 옛 셋 겹침에서 뺀다
        loud = make_sets.eval_cross(sets, old_keys={make_sets.pair_key(relabel[0])})
        self.assertEqual(loud["eval_text_old"], 1)

    def test_traps_without_drops_pairs_in_the_set(self):
        rows = [{"evidence": "캐시는 열 개를 둔다.", "claim": "캐시는 열 개다."}]
        traps = [{"tid": "t1", "evidence": "캐시는 열 개다.", "claim": "캐시는 열 개를 둔다.", "gold": "a"},
                 {"tid": "t2", "evidence": "로그는 지운다.", "claim": "로그는 남긴다.", "gold": "b"}]
        self.assertEqual([t["tid"] for t in make_sets.traps_without(traps, rows)], ["t2"])


class AttachRewriteTest(unittest.TestCase):
    """학습 후보 KLUE 줄에 다시 쓴 claim 붙이기 (46 결함)."""

    @staticmethod
    def cands():
        rows = [{"id": "p0", "domain": "proposals", "evidence": "근거", "claim": "주장", "group": "p:0", "src_ids": ["p0"]}]
        for i in range(3):
            rows.append({"id": f"k{i}", "domain": "klue-mrc", "evidence": f"근거 {i}", "claim": "", "hint": "ad"[i % 2],
                         "group": f"klue:{i}", "src_ids": [f"s{i}"], "needs_rewrite": True})
        return rows

    def run_cli(self, tmp, cands, rewrites):
        cpath, rpath, out = (os.path.join(tmp, n) for n in ("c.jsonl", "r.jsonl", "o.jsonl"))
        write(cpath, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cands))
        write(rpath, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rewrites))
        code = make_sets.main(["attach-rewrite", "--cands", cpath, "--rewrite", rpath, "--out", out])
        return code, out

    def test_attach_fills_claim_and_drops_skip(self):
        rewrites = [{"cand_idx": 1, "group": "klue:0", "src_ids": ["s0"], "claim": " 주장 영 ", "skip": False},
                    {"cand_idx": 99, "group": "klue:1", "src_ids": ["s1"], "claim": "주장 일", "fake_answer": "가짜",
                     "skip": False},                                   # 줄 번호가 틀려도 group+src_ids 로 찾는다
                    {"cand_idx": 3, "group": "klue:2", "src_ids": ["s2"], "claim": "", "skip": True}]
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_cli(tmp, self.cands(), rewrites)
            rows = make_sets.read_jsonl(out)
        self.assertEqual(code, 0)
        self.assertEqual([r["id"] for r in rows], ["k0", "k1"])
        self.assertEqual([r["claim"] for r in rows], ["주장 영", "주장 일"])
        self.assertFalse(any(r["needs_rewrite"] for r in rows))
        self.assertEqual(rows[1]["fake_answer"], "가짜")
        self.assertNotIn("fake_answer", rows[0])

    def test_unmatched_or_empty_claim_stops(self):
        base = [{"cand_idx": 1, "group": "klue:0", "src_ids": ["s0"], "claim": "주장 영", "skip": False},
                {"cand_idx": 3, "group": "klue:2", "src_ids": ["s2"], "claim": "", "skip": True}]
        cases = {"unmatched": base + [{"cand_idx": 2, "group": "klue:9", "src_ids": ["s9"], "claim": "x"}],
                 "empty_claim": base + [{"cand_idx": 2, "group": "klue:1", "src_ids": ["s1"], "claim": " "}],
                 "no_rewrite": base}
        for name, rewrites in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                code, out = self.run_cli(tmp, self.cands(), rewrites)
                self.assertEqual(code, 2)
                self.assertFalse(os.path.exists(out))
                _, report = make_sets.attach_rewrite(self.cands(), rewrites)
                self.assertGreater(report[name], 0)


def make_arch_train_build(tmp):
    """git · docs · mem 학습 후보(b 는 적게) + reserve-archive 꼴 예약 — 예약 줄 · 묶음 · 출처 id 를 하나씩 섞는다."""
    words = ("캐시", "로그", "빌드", "저장", "검색", "묶음", "시험", "배포", "문서", "설정")
    kinds = {"archive/git": (("a", "subject-body"), ("c", "other-file"), ("d", "same-file"), ("b", "revert")),
             "archive/docs": (("a", "summary-body"), ("c", "other-doc"), ("d", "same-doc"), ("d", "near")),
             "mem": (("a", "title-body"), ("c", "other-topic"), ("d", "same-doc"), ("d", "near"))}
    rows_by = {}
    for folder, ks in kinds.items():
        tag = folder.split("/")[-1]
        rows = []
        for g, word in enumerate(words):
            for k in range(12):
                hint, form = ks[k % len(ks)]
                if hint == "b" and k > 3:
                    hint, form = "d", "same-file"
                rid = f"{tag}-{g:02d}{k:02d}"
                src = f"2026100{g % 9 + 1}-{k:08x}" if tag == "mem" else f"{tag}/{word}/{k}"
                rows.append({"id": rid, "evidence": f"{word} {tag} 기능은 {k}번째 판에서 {g}개 묶음을 다시 읽도록 바꾸었다.",
                             "claim": f"{word} {tag} 설정은 {k}단계 뒤에 {g}번 기본값으로 돌아간다.", "want": "",
                             "hint": hint, "form": form, "domain": "mem" if tag == "mem" else "archive",
                             "group": f"{tag}:{word}", "groups": [f"{tag}:{word}"], "lang": "ko", "split": "any",
                             "src_ids": [src], "source": tag})
        rows_by[folder] = rows
        write(os.path.join(tmp, folder, "candidates.jsonl"), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        write(os.path.join(tmp, folder, "groups.json"), json.dumps({"groups": {r["group"]: {} for r in rows}}))
    doc = {"sets": {"dev-cal": {"mem": {"row_ids": ["mem-0000"], "groups": ["mem:캐시"]},
                                "archive": {"row_ids": ["docs-0100"], "groups": ["docs:로그"]}}},
           "reserved": ["mem:캐시", "docs:로그"], "reserved_src_ids": ["git/빌드/3"]}
    return rows_by, doc


class PoolArchiveTest(unittest.TestCase):
    def test_reserve_forbid_quota_and_zero_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows_by, doc = make_arch_train_build(tmp)
            forbid_path = os.path.join(tmp, "blind.jsonl")
            write(forbid_path, json.dumps({"group": "mem:로그", "groups": ["mem:로그"], "evidence": "x", "claim": "y"}) + "\n"
                  + json.dumps({"id": "z", "note": "20261002-00000001 은 이 셋에 쓴 기억"}) + "\n")
            forbid = make_sets.forbidden_mem_ids([forbid_path])
            public = rows_by["archive/git"][5]
            key = make_sets.pair_key(public)
            goal = {"d": 60, "a": 30, "b": 20, "c": 25}
            chosen, report = make_sets.pool_archive(tmp, doc, seed=4, hint_goal=goal, forbid_ids=forbid,
                                                    taken_keys={key, (key[1], key[0])}, group_cap=8)
        self.assertEqual(forbid, {"20261002-00000001"})
        ids = {r["id"] for r in chosen}
        self.assertNotIn("mem-0000", ids)
        self.assertFalse([r for r in chosen if set(r["groups"]) & set(doc["reserved"])])
        self.assertFalse([r for r in chosen if "git/빌드/3" in r["src_ids"]])
        self.assertFalse([r for r in chosen if "20261002-00000001" in r["src_ids"]])   # 본문에서 뽑은 기억 id
        self.assertNotIn(public["id"], ids)                                            # 공개 학습 후보와 같은 쌍
        self.assertEqual(report["dropped"]["forbidden_mem_id"], 1)
        self.assertEqual(report["dropped"]["public_train_pair"], 1)
        self.assertEqual(report["dropped"]["reserved_src"], 1)
        free_b = report["free_hint_form"].get("b:revert", 0)
        self.assertEqual(report["hint_total"]["b"], free_b)                            # b 는 있는 만큼만
        moved = goal["b"] - free_b
        self.assertEqual(report["moved_to_dc"], {"b": moved})
        self.assertEqual(report["hint_total"]["d"], goal["d"] + (moved + 1) // 2)       # 모자란 몫은 d · c 에 반씩
        self.assertEqual(report["hint_total"]["c"], goal["c"] + moved // 2)
        self.assertEqual(report["hint_total"]["a"], goal["a"])
        a_forms = collections.Counter(r["form"] for r in chosen if r["hint"] == "a")
        self.assertLessEqual(max(a_forms.values()) - min(a_forms.values()), 1)          # 꼴(=원천)마다 고르게
        self.assertLessEqual(max(collections.Counter(r["group"] for r in chosen).values()), 8)
        self.assertEqual(make_sets.arch_overlap(chosen, doc), {"arch_group_reserved": 0, "arch_src_reserved": 0})
        self.assertEqual(len({make_sets.pair_key(r) for r in chosen}), len(chosen))

    def test_used_only_blocks_what_eval_really_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows_by, doc = make_arch_train_build(tmp)
            git, docs, mem = rows_by["archive/git"], rows_by["archive/docs"], rows_by["mem"]
            ea = os.path.join(tmp, "eval-archive")
            write(os.path.join(ea, "dev-cal-cands.jsonl"), json.dumps(docs[50], ensure_ascii=False) + "\n")   # docs:검색
            write(os.path.join(ea, "dev-cal-traps.jsonl"), "")
            write(os.path.join(ea, "dev-cal", "map", "map.jsonl"),
                  json.dumps({"q": "q1", "id": docs[50]["id"], "trap": False}) + "\n"
                  + json.dumps({"q": "q2", "id": mem[40]["id"], "trap": False}) + "\n"     # 후보 파일 밖 id → 원천에서
                  + json.dumps({"q": "q3", "id": "t9", "trap": True}) + "\n")
            twin = git[100]                                                               # 묶음은 안 막히고 글만 같은 쌍
            write(os.path.join(ea, "dev-cal", "batches", "b01-j1.jsonl"),
                  json.dumps({"q": "q3", "evidence": twin["evidence"], "claim": twin["claim"]}, ensure_ascii=False) + "\n")
            frozen = os.path.join(tmp, "dev-cmp.jsonl")
            write(frozen, json.dumps(dict(git[72], status="agree"), ensure_ascii=False) + "\n")   # git:시험
            rows, texts, unknown = make_sets.used_eval_rows(tmp, ea, [frozen])
            used = make_sets.used_blocks(rows)
            keys = set()
            for row in rows + texts:
                key = make_sets.pair_key(row)
                keys.update((key, (key[1], key[0])))
            chosen, report = make_sets.pool_archive(tmp, doc, seed=5, hint_goal={"d": 200, "a": 80, "b": 5, "c": 60},
                                                    eval_keys=keys, used=used, group_cap=12)
        self.assertEqual(unknown, 0)
        self.assertEqual(used["groups"], {"docs:검색", mem[40]["group"], "git:시험"})
        groups = {r["group"] for r in chosen}
        self.assertFalse(groups & used["groups"])                       # 실제로 쓴 묶음은 막힌다
        self.assertIn("docs:로그", groups)                              # 예약만 하고 안 쓴 묶음은 열린다
        self.assertNotIn(twin["id"], {r["id"] for r in chosen})         # 묶음 글(함정 포함)과 같은 쌍은 막힌다
        self.assertEqual(report["dropped"]["eval_pair"], 1)
        self.assertNotIn("reserved_row", report["dropped"])

    def test_cli_stops_without_forbid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, doc = make_arch_train_build(tmp)
            write(os.path.join(tmp, "reserve-archive.json"), json.dumps(doc))
            code = make_sets.main(["pool", "--archive", "--build", tmp, "--old-root", os.path.join(tmp, "없음")])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "train-cands-archive.jsonl")))


if __name__ == "__main__":
    unittest.main()
