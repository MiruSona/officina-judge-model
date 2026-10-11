"""measure/use_sites/reports.py — ④ 보고 거두기 (실사용측정설계 4-1 덧붙임).
가짜 대화 기록(임시 폴더)과 가짜 mem(tests/fake_mem_reports.py)만 쓴다. 쓰기는 임시 폴더(JUDGEMODEL_USE1_HOME)에만 한다.
"""
import contextlib
import datetime
import io
import json
import os
import shutil
import tempfile
import unittest

import helpers  # noqa: F401  import 길

import use_eval
from use_sites import common, reports

HERE = os.path.dirname(os.path.abspath(__file__))
AT = datetime.datetime(2026, 10, 11, 10, 0, tzinfo=datetime.timezone.utc).astimezone()

REPORT = """## 2. 이번에 정한 것
| 무엇 | 정한 것 | 까닭 |
| --- | --- | --- |
| 포트 | 서버 포트는 8093 으로 바꾼다 | 충돌 |
| 키 | 비밀 키는 sk-ZYXWVUTSRQPONMLKJIHGFE 다 | 시험 |

정함 : 측정 결과는 저장소 안에 둔다
`mem add --type decision --scope judgemodel --title "보고 결정" --summary "보고 결정 요약 문장이다"`
남긴 기억 : 20261010-aaaa0005
"""


def transcript_lines(report, prompt):
    """사용자 지시 한 줄 · 도구 호출 · 마지막 SubagentHandback."""
    return [
        {"type": "user", "timestamp": "2026-10-11T09:00:00Z", "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "timestamp": "2026-10-11T09:30:00Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "정함 : 중간 생각은 거두지 않는다"}]}},
        {"type": "assistant", "timestamp": "2026-10-11T09:31:00Z",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "assistant", "timestamp": "2026-10-11T10:00:00Z",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "SubagentHandback",
                                                       "input": {"message": report}}]}},
    ]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = {k: os.environ.get(k) for k in (common.HOME_ENV, common.MEM_ENV)}
        os.environ[common.HOME_ENV] = os.path.join(self.tmp, "use1")
        os.environ[common.MEM_ENV] = os.path.join(HERE, "fake_mem_reports.py")
        self.repo = os.path.join(self.tmp, "repo")
        for name in ("JudgeModel", "ArtTool", "Docs"):
            os.makedirs(os.path.join(self.repo, name))
        self.trans = os.path.join(self.tmp, "trans")

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_agent(self, name, report, prompt="정함 : 사용자 지시는 거두지 않는다"):
        folder = os.path.join(self.trans, "sess1", "subagents")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"agent-{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in transcript_lines(report, prompt):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def run_cli(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = use_eval.main(argv)
        return code, out.getvalue()

    def pairs(self, alias="t"):
        return common.read_jsonl(common.alias_dir(alias, "collect", "report-pairs.jsonl"))


class ExtractTest(unittest.TestCase):
    def test_claim_kinds(self):
        got = dict((c, k) for c, k in reports.extract_claims(REPORT))
        self.assertEqual(got["포트 : 서버 포트는 8093 으로 바꾼다"], "table")
        self.assertEqual(got["측정 결과는 저장소 안에 둔다"], "decided")
        self.assertEqual(got["보고 결정 요약 문장이다"], "mem-add")
        self.assertFalse(any("충돌" in c for c in got))  # 까닭 칸은 안 넣는다

    def test_plain_table_ignored(self):
        text = "## 공수\n| 소단계 | 예상 | 실제 |\n| --- | --- | --- |\n| 1. 설계하기 | 10분 | 8분 |\n"
        self.assertEqual(reports.extract_claims(text), [])

    def test_titled_table_uses_first_two(self):
        text = "이번에 정한 것\n| 1 | 문자표 | UTF-8 로 통일한다 |\n| --- | --- | --- |\n| 2 | 줄끝 | LF 로만 쓴다고 정한다 |\n"
        self.assertEqual(reports.extract_claims(text), [("줄끝 : LF 로만 쓴다고 정한다", "table")])

    def test_cap_and_length(self):
        text = "\n".join(f"정함 : 열두 개를 넘는 결정 문장 {i}" for i in range(20)) + "\n정함 : 짧다"
        self.assertEqual(len(reports.extract_claims(text)), reports.CLAIMS_PER_REPORT)


class ScopeTest(unittest.TestCase):
    KNOWN = {"judgemodel", "arttool", "officina"}
    FOLDERS = ["ArtTool", "JudgeModel"]

    def test_order(self):
        self.assertEqual(reports.choose_scope("mem add --scope judgemodel · ArtTool ArtTool", self.KNOWN, self.FOLDERS),
                         ("judgemodel", None))
        self.assertEqual(reports.choose_scope("ArtTool/x.py 와 ArtTool 문서", self.KNOWN, self.FOLDERS), ("arttool", None))
        self.assertEqual(reports.choose_scope("아무 폴더도 없다", self.KNOWN, self.FOLDERS), ("officina", None))
        self.assertEqual(reports.choose_scope("--scope nosuch 뿐", self.KNOWN, self.FOLDERS), ("officina", None))

    def test_tie_and_unknown(self):
        self.assertEqual(reports.choose_scope("ArtTool · JudgeModel", self.KNOWN, self.FOLDERS), (None, "scope-tie"))
        self.assertEqual(reports.choose_scope("없음", {"arttool"}, []), (None, "scope-unknown"))


class HarvestTest(Base):
    def test_pairs_self_later_secret(self):
        self.write_agent("a1", REPORT)
        code, _ = self.run_cli(["harvest-reports", "--repo", self.repo, "--alias", "t", "--days", "100000",
                                "--transcripts", self.trans])
        self.assertEqual(code, 0)
        rows = self.pairs()
        olds = {r["old_id"] for r in rows if not r.get("refused")}
        self.assertEqual(olds, {"20261001-aaaa0001", "20261002-aaaa0002"})  # 0005 자기 짝 · 0006 보고 날 뒤는 빠진다
        live = [r for r in rows if not r.get("refused")]
        self.assertEqual(len(live), 6)  # 문장 셋 × 기존 결정 둘
        refused = [r for r in rows if r.get("refused")]
        self.assertEqual(len(refused), 1)
        self.assertEqual((refused[0]["claim"], refused[0]["evidence"], refused[0]["why"]), ("", "", "secret-key"))
        self.assertTrue(all(r["scope"] == "judgemodel" and r["src"] == "transcript" for r in rows))
        claims = {r["claim"] for r in live}
        self.assertNotIn("사용자 지시는 거두지 않는다", claims)
        self.assertNotIn("중간 생각은 거두지 않는다", claims)
        stats = common.read_json(common.alias_dir("t", "collect", "harvest-reports.json"))
        self.assertEqual((stats["self_dropped"], stats["later_dropped"]), (3, 3))

    def test_no_duplicates_on_rerun(self):
        self.write_agent("a1", REPORT)
        argv = ["harvest-reports", "--repo", self.repo, "--alias", "t", "--days", "100000", "--transcripts", self.trans]
        self.run_cli(argv)
        first = len(self.pairs())
        self.run_cli(argv)
        self.assertEqual(len(self.pairs()), first)
        keys = [r["pair"] for r in self.pairs()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_skip_tie_and_secret_evidence(self):
        self.write_agent("a2", "정함 : JudgeModel 과 ArtTool 을 같이 고친다고 정했다\n")
        self.write_agent("a3", "정함 : 공용 규칙은 루트 문서에 둔다고 정했다\n")
        self.run_cli(["harvest-reports", "--repo", self.repo, "--alias", "t", "--days", "100000",
                      "--transcripts", self.trans])
        skipped = common.read_jsonl(common.alias_dir("t", "collect", "skipped.jsonl"))
        self.assertEqual([r["why"] for r in skipped], ["scope-tie"])
        rows = self.pairs()
        self.assertTrue(all(r["scope"] == "officina" for r in rows))
        self.assertEqual(sorted(bool(r.get("refused")) for r in rows), [False, True])  # 키 든 기존 결정은 거른다

    def test_empty_refuses(self):
        os.makedirs(self.trans)
        code, _ = self.run_cli(["harvest-reports", "--repo", self.repo, "--alias", "t", "--transcripts", self.trans])
        self.assertEqual(code, common.REFUSE_EXIT)


class LogReportTest(Base):
    def test_one_file(self):
        path = os.path.join(self.tmp, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("정함 : JudgeModel 측정 기록은 집 폴더 아래에 둔다\n")
        code, _ = self.run_cli(["log-report", "--repo", self.repo, "--alias", "t", "--file", path])
        self.assertEqual(code, 0)
        rows = self.pairs()
        self.assertTrue(rows)
        self.assertTrue(all(r["src"] == "live" and r["scope"] == "judgemodel" for r in rows))


if __name__ == "__main__":
    unittest.main()
