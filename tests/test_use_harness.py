"""measure/use_sites/harness.py — ⑤ harvest-harness : 축약 길이 상한 · 중복 없음 · 비밀 거름 · 층 나누기 (실사용측정설계 4-2).
가짜 얼린 셋 · 가짜 runs 만 쓴다. 토큰은 글자 어림(--tokenizer "")으로 센다. 쓰기는 임시 폴더(JUDGEMODEL_USE1_HOME)에만 한다.
"""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import helpers  # noqa: F401  import 길

import use_eval
from use_sites import common, harness


def diff(path, added, removed=()):
    lines = [f"diff --git a/{path} b/{path}", "index 1111111..2222222 100644", f"--- a/{path}", f"+++ b/{path}", "@@ -1,3 +1,3 @@"]
    lines += [" 그대로 둔 줄"] + ["-" + r for r in removed] + ["+" + a for a in added]
    return "\n".join(lines)


EV_CACHE = diff("mods/cache.ts", ["// 상한을 정한다", "export function clampSec(sec) {", "  return Math.min(sec, 3600)", "}",
                                  "export const DEFAULT_RUNS = 'runs'"])
EV_PROBE = diff("mods/probe.ts", ["export const MODEL_MAX = 20"], ["export const MODEL_MAX = 16"])
EV_SECRET = diff("conf/app.ts", ["const api_key = \"sk-abcdefghijklmnopqrstuv\""])
EV_LONG = diff("Docs/long.md", ["긴 줄 " * 400, "또 긴 줄 " * 400])


def write_lines(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def probs(a):
    return {"A": a, "B": (1 - a) / 2, "C": (1 - a) / 2}


class HarvestHarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = os.path.join(self.tmp, "use1")
        os.environ[common.HOME_ENV] = self.home
        self.sets = os.path.join(self.tmp, "judge-sets")
        self.runs = os.path.join(self.tmp, "runs")
        item = [
            {"id": "t1-item-1-A", "evidence": EV_CACHE, "claim": "1 · mods/cache.ts · → export function clampSec", "want": "A"},
            {"id": "t1-item-1-B1", "evidence": EV_CACHE, "claim": "1 · other/file.ts · → export function clampSec", "want": "B"},
            {"id": "t1-item-1-C", "evidence": EV_PROBE, "claim": "1 · mods/cache.ts · → export function clampSec", "want": "C"},
            {"id": "t2-item-1-B0", "evidence": EV_PROBE, "claim": "1 · mods/probe.ts · MODEL_MAX 16 → 20", "want": "A"},
        ]
        claim = [
            {"id": "t1-claim-1-A", "evidence": EV_CACHE, "claim": "clampSec 함수를 넣었다", "want": "A"},
            {"id": "t1-claim-2-B2", "evidence": EV_PROBE, "claim": "MODEL_MAX 를 16에서 32로 바꿨다", "want": "B"},
            {"id": "t3-claim-1-A", "evidence": EV_SECRET, "claim": "키를 넣었다", "want": "A"},
            {"id": "t4-claim-1-A", "evidence": EV_LONG, "claim": "긴 줄을 더했다", "want": "A"},
        ]
        write_lines(os.path.join(self.sets, "2026-10-08-item.jsonl"), item)
        write_lines(os.path.join(self.sets, "2026-10-08-claim.jsonl"), claim)
        write_lines(os.path.join(self.sets, "2026-10-08-item.out.jsonl"),
                    [{"id": r["id"], "letter": "A", "probs": probs(0.9), "cached": True} for r in item])
        write_lines(os.path.join(self.sets, "2026-10-08-claim.out.jsonl"),
                    [{"id": r["id"], "letter": "A", "probs": probs(0.6)} for r in claim])
        run = os.path.join(self.runs, "t1-20261008")
        write_lines(os.path.join(run, "01-judge.jsonl"), [
            {"id": "claim-1", "kind": "claim", "evidence": EV_CACHE, "claim": "clampSec 함수를 넣었다", "probs": probs(0.85), "letter": "A",
             "at": "2026-10-08T03:00:00+09:00"},
            {"id": "item-1", "kind": "item", "evidence": EV_PROBE, "claim": "1 · mods/probe.ts · 16 → 20", "probs": probs(0.3), "letter": "B",
             "at": "2026-10-08T03:00:00+09:00"},
            {"id": "claim-2", "kind": "claim", "evidence": EV_PROBE, "claim": "C:\\Users\\someone\\x.txt 를 고쳤다", "probs": probs(0.7),
             "letter": "A", "at": "2026-10-08T03:00:00+09:00"},
            {"id": "claim-3", "evidence": EV_PROBE, "problem": "timeout"},
        ])
        with open(os.path.join(run, "run.json"), "w", encoding="utf-8") as f:
            json.dump({"메인검토": {"값": "통과", "까닭": "제자리"}}, f, ensure_ascii=False)

    def tearDown(self):
        os.environ.pop(common.HOME_ENV, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, *extra):
        argv = ["harvest-harness", "--alias", "t", "--tokenizer", "", "--judge-sets", self.sets, "--runs", self.runs, *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(use_eval.main(argv), 0)
        return common.read_jsonl(common.alias_dir("t", *harness.OUT_PARTS))

    def test_layers_secret_and_dedupe(self):
        rows = self.run_cmd()
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["js:t1-item-1-A"]["layer"], "judgeset-item")
        self.assertEqual(by_id["js:t2-item-1-B0"]["layer"], "judgeset-item")
        self.assertEqual(by_id["js:t1-claim-1-A"]["layer"], "judgeset-claim")
        for made in ("js:t1-item-1-B1", "js:t1-item-1-C", "js:t1-claim-2-B2"):
            self.assertEqual(by_id[made]["layer"], "judgeset-made")
        self.assertEqual(by_id["js:t1-item-1-C"]["gold"], "c")
        self.assertEqual(by_id["js:t1-item-1-A"]["semif"]["verdict"], "pass")
        self.assertEqual(by_id["js:t1-claim-1-A"]["semif"]["verdict"], "unsure")
        # 비밀 꼴 — 키 · 집 경로는 refused, 본문은 비운다
        for secret in ("js:t3-claim-1-A", "run:t1-20261008:01:claim-2"):
            self.assertTrue(by_id[secret]["refused"])
            self.assertEqual((by_id[secret]["evidence"], by_id[secret]["claim"]), ("", ""))
        # runs 의 claim-1 은 얼린 셋 A 줄과 같은 쌍 → 빠진다 · claim 없는 줄은 건너뛴다
        self.assertNotIn("run:t1-20261008:01:claim-1", by_id)
        self.assertNotIn("run:t1-20261008:01:claim-3", by_id)
        runs_item = by_id["run:t1-20261008:01:item-1"]
        self.assertEqual((runs_item["layer"], runs_item["main_review"], runs_item["semif"]["verdict"]), ("runs-item", "통과", "review"))
        pairs = [r["pair"] for r in rows if not r.get("refused")]
        self.assertEqual(len(pairs), len(set(pairs)))
        # 다시 돌려도 더해지지 않는다 (id · 쌍 열쇠 중복 없음)
        self.assertEqual(len(self.run_cmd()), len(rows))
        log = common.read_jsonl(common.alias_dir("t", *harness.MANIFEST_PARTS))
        self.assertEqual([(x["added"], x["token_mode"]) for x in log], [(len(rows), "chars"), (0, "chars")])

    def test_compress_under_limit(self):
        rows = self.run_cmd()
        counter = harness.TokenCounter()
        for r in rows:
            if r.get("refused"):
                continue
            total = counter.count(r["evidence"]) + counter.count(r["claim"]) + harness.PAIR_SPECIALS
            self.assertLessEqual(total, harness.MAX_TOKENS, r["id"])
            self.assertEqual(r["len"]["tokens"], total)
        self.assertTrue(next(r for r in rows if r["id"] == "js:t4-claim-1-A")["len"]["cut"])

    def test_pick_lines_follow_claim(self):
        got = harness.pick_lines(EV_CACHE, "DEFAULT_RUNS 를 runs 로 정했다")
        self.assertTrue(got.startswith("mods/cache.ts : "))
        self.assertIn("DEFAULT_RUNS", got)
        self.assertNotIn("상한을", got)  # 주석 줄은 claim 과 안 겹쳐 밀린다
        self.assertIn("(지움) ", harness.pick_lines(diff("a.md", [], ["지운 줄 하나", "지운 줄 둘"]), "지운 줄"))

    def test_needs_a_source(self):
        a = use_eval.build_parser().parse_args(["harvest-harness", "--alias", "t"])
        with self.assertRaises(common.Refuse):
            a.func(a)


if __name__ == "__main__":
    unittest.main()
