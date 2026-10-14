"""filter_pairs — 고침 쌍 거르기(까닭별 버림 · 태그 벗기기 · 수 바뀜 앞세우기 · cap). 입력은 모두 지어낸 짧은 글이다."""
import json
import os
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import filter_pairs as fp  # noqa: E402


def row(i, evidence, claim, lang="ko", num_change=False, commit_edits=1):
    return {"id": f"t-{i:03d}", "evidence": evidence, "claim": claim, "lang": lang,
            "num_change": num_change, "commit_edits": commit_edits, "hint": "b"}


def run(rows, *extra):
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "pairs.jsonl"), os.path.join(tmp, "pairs-f.jsonl")
        fp.write_jsonl(src, rows)
        args = ["--in", src, "--out", out, "--lang", "ko,mix", "--drop-html", "--drop-list",
                "--min-chars", "15", "--sort", "num_change", *extra]
        fp.main(args)
        with open(os.path.join(tmp, "filter.summary.json"), encoding="utf-8") as f:
            return fp.read_jsonl(out), json.load(f)


class FilterPairsTest(unittest.TestCase):
    def test_reasons(self):
        rows = [
            row(1, "The cache keeps three entries.", "The cache keeps five entries.", lang="en"),
            row(2, '<img src="a.png" width="80%"/>', '<img src="b.png" width="80%"/>'),
            row(3, "[x] 운영체제 인터럽트 정리하기 - 주디", "[ ] 운영체제 인터럽트 정리하기"),
            row(4, "| 이름 | 크기 | 설명 칸이 여기 있다 |", "| 이름 | 길이 | 설명 칸이 여기 있다 |"),
            row(5, 'System.out.println("사진을 찍습니다");', 'System.out.println("사진 찍습니다");'),
            row(6, "짧은 문장이다.", "짧은 글이다."),
            row(7, "그 서버는 하루에 두 번 백업한다.", "그 서버는 하루에 세 번 백업한다."),
            row(8, "그 서버는 하루에 두 번 백업한다.", "그 서버는 하루에 세 번 백업한다."),
            row(9, "Hash table 의 collision 을 막는 법", "Hash table collision handling method", lang="mix"),
            row(10, "기본 설정에서 지연 시간은 40ms 이다.", "기본 설정에서 지연 시간은 90ms 이다.", lang="mix"),
        ]
        kept, summary = run(rows)
        self.assertEqual([r["id"] for r in kept], ["t-007", "t-010"])
        d = summary["dropped"]
        self.assertEqual((d["lang"], d["html"], d["checklist"], d["table"], d["code"], d["short"], d["dup"]),
                         (2, 1, 1, 1, 1, 1, 1))

    def test_inline_tag_stripped_and_prose_arrow_kept(self):
        rows = [
            row(1, "그렇다면 210만 원을 이체하는 <u>UPDATE문은</u> 실패한다.",
                "그렇다면 100만 원을 이체하는 UPDATE문은 실패한다."),
            row(2, "문서에서 <strong>굵은 글씨</strong>만 바꾼 줄이다.", "문서에서 굵은 글씨만 바꾼 줄이다."),
            row(3, "질의는 루트 -> com -> 회사 서버 차례로 간다.", "질의는 루트 → com → 회사 서버 차례로 간다."),
        ]
        kept, summary = run(rows)
        self.assertEqual([r["id"] for r in kept], ["t-001", "t-003"])
        self.assertEqual(kept[0]["evidence"], "그렇다면 210만 원을 이체하는 UPDATE문은 실패한다.")
        self.assertTrue(kept[0]["html_stripped"])
        self.assertEqual(summary["dropped"]["html"], 1)

    def test_sort_and_cap(self):
        rows = [row(i, f"그 목록에는 항목이 {i} 개 들어 있다고 적었다.", f"그 목록에는 항목이 {i + 1} 개 들어 있다고 적었다.",
                    num_change=(i % 4 == 0), commit_edits=10 - i % 10) for i in range(20)]
        kept, summary = run(rows, "--cap", "8", "--seed", "7")
        self.assertEqual(len(kept), 8)
        self.assertEqual(sum(r["num_change"] for r in kept), 5)
        self.assertTrue(all(r["num_change"] for r in kept[:5]))
        edits = [r["commit_edits"] for r in kept[:5]]
        self.assertEqual(edits, sorted(edits))
        again, _ = run(rows, "--cap", "8", "--seed", "7")
        self.assertEqual([r["id"] for r in kept], [r["id"] for r in again])
        self.assertEqual((summary["kept"], summary["out"], summary["out_num_change"]), (20, 8, 5))


if __name__ == "__main__":
    unittest.main()
