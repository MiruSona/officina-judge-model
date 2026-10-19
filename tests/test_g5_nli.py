"""measure/g5_nli.py — G5 얼린 셋 NLI 재기 : 셈 함수 손 셋 · 두 자 · 잘림 여부 = pairenc · 같은 입력 같은 출력 (v1.2 적용 설계 8절 #17).
가짜 토크나이저 · 가짜 서버만 쓴다. tokenizers 가 있으면 진짜 글자 토크나이저로 잘림 여부를 한 번 더 견준다.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import helpers

import g5_nli
from pairenc import encode_pair, pair_length

HAS_TOKENIZERS = helpers.has("tokenizers")


class _Enc:
    def __init__(self, ids, overflowing):
        self.ids = ids
        self.overflowing = overflowing


class FakeTok:
    """글자(공백 뺌) 하나 = 토큰 하나 · XLM-R 쌍 틀(특수 4). tokenizers.Tokenizer 가 encode_pair 에서 쓰는 만큼만 흉내 낸다."""

    def __init__(self):
        self.max_len = None
        self.strategy = None

    def no_padding(self):
        pass

    def no_truncation(self):
        self.max_len = None

    def enable_truncation(self, max_len, strategy="longest_first"):
        self.max_len, self.strategy = max_len, strategy

    def num_special_tokens_to_add(self, is_pair):
        return 4 if is_pair else 2

    @staticmethod
    def _ids(text):
        return [ord(ch) for ch in text if not ch.isspace()]

    def encode(self, a, b=None, add_special_tokens=True):
        first, second = self._ids(a), self._ids(b) if b is not None else None
        specials = (4 if second is not None else 2) if add_special_tokens else 0
        total = len(first) + len(second or []) + specials
        cut = False
        if self.max_len is not None and total > self.max_len:
            cut = True
            over = total - self.max_len
            if second is None or self.strategy == "only_first":
                first = first[:len(first) - over]
            else:
                while over > 0:
                    if len(first) >= len(second):
                        first = first[:-1]
                    else:
                        second = second[:-1]
                    over -= 1
        ids = [0] + first + [2] + ([2] + second + [2] if second is not None else []) if add_special_tokens else first + (second or [])
        return _Enc(ids, [1] if cut else [])


def diff(path, added):
    lines = [f"--- a/{path}", f"+++ b/{path}", "@@ -1,2 +1,3 @@", " 그대로 둔 줄"] + ["+" + a for a in added]
    return "\n".join(lines)


SHORT = diff("mods/a.ts", ["export const MAX = 20"])
LONG = diff("Docs/long.md", ["긴줄" * 300, "export const MAX = 20"])


def write_lines(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def semif(a):
    return {"id": None, "probs": {"A": a, "B": (1 - a) / 2, "C": (1 - a) / 2}, "letter": "A"}


def make_sets(folder):
    """손 셋 6줄(claim 4 · item 2) + 다른 날짜 파일 하나(안 읽혀야 한다)."""
    claim = [
        {"id": "t1-claim-1-A", "evidence": SHORT, "claim": "MAX 를 20 으로 넣었다", "want": "A"},
        {"id": "t1-claim-2-B1", "evidence": SHORT, "claim": "MAX 를 30 으로 넣었다", "want": "B"},
        {"id": "t1-claim-3-B2", "evidence": LONG, "claim": "항목 9 에서 MAX 를 넣었다", "want": "B"},
        {"id": "t1-claim-4-C", "evidence": SHORT, "claim": "테스트를 고쳤다", "want": "C"},
    ]
    item = [
        {"id": "t1-item-1-B0", "evidence": LONG, "claim": "1 · Docs/long.md · MAX 20", "want": "A"},
        {"id": "t1-item-2-C", "evidence": SHORT, "claim": "2 · other.ts · 다른 일", "want": "C"},
    ]
    p_a = {"t1-claim-1-A": 0.9, "t1-claim-2-B1": 0.9, "t1-claim-3-B2": 0.3, "t1-claim-4-C": 0.6,
           "t1-item-1-B0": 0.2, "t1-item-2-C": 0.85}
    write_lines(os.path.join(folder, "2026-10-08-claim.jsonl"), claim)
    write_lines(os.path.join(folder, "2026-10-08-item.jsonl"), item)
    write_lines(os.path.join(folder, "2026-10-08-claim.out.jsonl"), [{**semif(p_a[r["id"]]), "id": r["id"]} for r in claim])
    write_lines(os.path.join(folder, "2026-10-08-item.out.jsonl"), [{**semif(p_a[r["id"]]), "id": r["id"]} for r in item])
    write_lines(os.path.join(folder, "2026-10-19-nli-claim.jsonl"), [{"id": "x-claim-9-A", "evidence": "e", "claim": "c", "want": "A"}])


def fake_answer(evidence, claim):
    """글에 따라 정해진 확률 — 같은 글이면 같은 답."""
    if "30" in claim or "항목 9" in claim:
        p = (0.05, 0.9, 0.05)
    elif "다른" in claim or "테스트" in claim:
        p = (0.1, 0.1, 0.8)
    else:
        p = (0.95, 0.02, 0.03)
    return {"a": p[0], "b": p[1], "c": p[2], "model": "fake", "ms": 1.0, "label_set": "nli3"}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.01
        return self.t


class RulerTest(unittest.TestCase):
    def test_g5_edges(self):
        self.assertEqual(g5_nli.g5_verdict(0.8), "pass")
        self.assertEqual(g5_nli.g5_verdict(0.79), "unsure")
        self.assertEqual(g5_nli.g5_verdict(0.5), "unsure")
        self.assertEqual(g5_nli.g5_verdict(0.49), "review")

    def test_cal_edges(self):
        self.assertEqual(g5_nli.cal_verdict(0.89, 0.05, 0.06), "pass")
        self.assertEqual(g5_nli.cal_verdict(0.88, 0.0, 0.12), "unsure")
        self.assertEqual(g5_nli.cal_verdict(0.3, 0.5, 0.2), "review")
        self.assertEqual(g5_nli.cal_verdict(0.3, 0.1, 0.6), "review")
        self.assertEqual(g5_nli.cal_verdict(0.4, 0.3, 0.3), "unsure")


class TallyTest(unittest.TestCase):
    def test_hand_six(self):
        pairs = [("a", "pass"), ("a", "review"), ("b", "review"), ("b", "unsure"), ("c", "pass"), ("c", "review"),
                 ("b", None)]
        t = g5_nli.tally(pairs)
        self.assertEqual((t["n"], t["n_a"], t["n_b"], t["n_c"]), (6, 2, 2, 2))
        self.assertEqual((t["recall_b"], t["false_alarm_a"], t["c_misread"]), (0.5, 0.5, 0.5))
        self.assertEqual(t["unsure_share"], round(1 / 6, 4))

    def test_hit_matches_counts(self):
        self.assertTrue(g5_nli.hit("b", "review"))
        self.assertFalse(g5_nli.hit("b", "unsure"))
        self.assertTrue(g5_nli.hit("a", "unsure"))
        self.assertFalse(g5_nli.hit("a", "review"))
        self.assertFalse(g5_nli.hit("c", "pass"))
        self.assertTrue(g5_nli.hit("c", "unsure"))

    def test_check_answer(self):
        self.assertEqual(g5_nli.check_answer({"a": 0.5, "b": 0.3, "c": 0.2})["a"], 0.5)
        with self.assertRaises(ValueError):
            g5_nli.check_answer({"a": 0.5, "b": 0.3, "c": 0.1})
        with self.assertRaises(ValueError):
            g5_nli.check_answer({"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1})
        with self.assertRaises(KeyError):
            g5_nli.check_answer({"a": 0.5, "b": 0.5})


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        make_sets(self.tmp)
        self.max_len = 128

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_once(self, tok):
        rows, files = g5_nli.load_rows(self.tmp, "2026-10-08")
        prepared = g5_nli.prepare(rows, tok, self.max_len)
        g5_nli.mark_seen(prepared, [])
        lines = g5_nli.run_arms(prepared, fake_answer, clock=Clock())
        return rows, files, prepared, lines, g5_nli.build_summary(lines, {"sets": files})

    def test_reads_only_dated_files(self):
        rows, files, *_ = self.run_once(FakeTok())
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted(f["name"] for f in files),
                         ["2026-10-08-claim.jsonl", "2026-10-08-claim.out.jsonl", "2026-10-08-item.jsonl",
                          "2026-10-08-item.out.jsonl"])

    def test_cut_equals_pairenc(self):
        tok = FakeTok()
        rows, _, prepared, lines, _ = self.run_once(tok)
        for p in prepared:
            r = p["row"]
            self.assertEqual(p["raw_cut"], encode_pair(tok, r["evidence"], r["claim"], self.max_len)[1])
            self.assertEqual(p["arms"]["raw"]["tokens"], pair_length(tok, r["evidence"], r["claim"]))
        self.assertTrue(any(p["raw_cut"] for p in prepared))        # LONG 은 잘린다
        self.assertFalse(any(p["raw_cut"] for p in prepared if "long" not in p["row"]["evidence"]))
        raw = {l["id"]: l for l in lines if l["arm"] == "raw"}
        self.assertTrue(raw["t1-claim-3-B2"]["cut"])
        self.assertNotIn("evidence", raw["t1-claim-3-B2"])           # 본문은 출력에 안 남긴다

    def test_counts_and_semif(self):
        *_, summary = self.run_once(FakeTok())
        raw = summary["arms"]["raw"]
        g5 = raw["rulers"]["g5"]["all"]
        # want B 둘(30 · 항목 9) 모두 a .05 → 검토 · want A 둘 a .95 → 통과 · want C 둘 a .1 → 검토
        self.assertEqual((g5["recall_b"], g5["false_alarm_a"], g5["c_misread"]), (1.0, 0.0, 0.0))
        sem = raw["semif"]["all"]
        # SemIf : B1 .9 통과 · B2 .3 검토 → 재현 .5 · A .9 통과 · B0(want A) .2 검토 → 오탐 .5 · C .6 애매 · .85 통과 → 무관 오인 .5
        self.assertEqual((sem["recall_b"], sem["false_alarm_a"], sem["c_misread"]), (0.5, 0.5, 0.5))
        self.assertEqual(raw["rulers"]["g5"]["no_B2"]["n_b"], 1)
        self.assertEqual(raw["rulers"]["g5"]["tail:C"]["n"], 2)
        self.assertEqual(raw["mcnemar"]["g5"]["all"]["recall_b"]["c"], 1)   # NLI 만 맞힌 B 하나
        self.assertEqual(raw["p95_ms"], 10.0)
        self.assertEqual(summary["judgement"]["verdict"], "못 넘음")       # 표본이 작아 p ≥ .05

    def test_same_input_same_output(self):
        first = self.run_once(FakeTok())
        second = self.run_once(FakeTok())
        self.assertEqual(json.dumps(first[3], sort_keys=True), json.dumps(second[3], sort_keys=True))
        self.assertEqual(json.dumps(first[4], sort_keys=True), json.dumps(second[4], sort_keys=True))
        ids = [l["id"] for l in first[3] if l["arm"] == "raw"]
        self.assertEqual(ids, sorted(ids, key=lambda i: (i.split("-")[1], i)))

    def test_server_down_stops(self):
        rows, _ = g5_nli.load_rows(self.tmp, "2026-10-08")
        prepared = g5_nli.prepare(rows, FakeTok(), self.max_len)

        def down(evidence, claim):
            raise OSError("refused")
        with self.assertRaises(SystemExit):
            g5_nli.run_arms(prepared, down, clock=Clock())

    def test_mid_run_failures(self):
        """도중 한 줄 실패 → 끝까지 가되 판정 「셀 수 없음」 · partial. 도중 3줄 연속 실패 → 멈춤."""
        rows, _ = g5_nli.load_rows(self.tmp, "2026-10-08")
        prepared = g5_nli.prepare(rows, FakeTok(), self.max_len)
        calls = []

        def flaky(evidence, claim):
            calls.append(1)
            if len(calls) == 3:
                raise ValueError("bad json")
            return fake_answer(evidence, claim)
        lines = g5_nli.run_arms(prepared, flaky, clock=Clock())
        summary = g5_nli.build_summary(lines, {})
        self.assertEqual(summary["arms"]["raw"]["failed"], 1)
        self.assertTrue(summary["partial"])
        self.assertEqual(summary["judgement"]["verdict"], "셀 수 없음")
        self.assertEqual(summary["judgement_no_B2"]["verdict"], "셀 수 없음")

        calls.clear()

        def dies_later(evidence, claim):
            calls.append(1)
            if len(calls) >= 8:                                   # cmp 팔 두째 줄부터 죽음
                raise OSError("refused")
            return fake_answer(evidence, claim)
        with self.assertRaises(SystemExit):
            g5_nli.run_arms(prepared, dies_later, clock=Clock())

    def test_complete_run_not_partial(self):
        *_, summary = self.run_once(FakeTok())
        self.assertFalse(summary["partial"])

    def test_broken_set_line_exit2(self):
        with open(os.path.join(self.tmp, "2026-10-08-item.jsonl"), "a", encoding="utf-8") as f:
            f.write("{깨진 줄\n")
        with self.assertRaises(SystemExit) as got:
            g5_nli.load_rows(self.tmp, "2026-10-08")
        self.assertEqual(got.exception.code, 2)

    def test_health_fail_exit2(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        port = server.server_address[1]
        server.server_close()                                     # 닫힌 포트 — 연결 거절
        with self.assertRaises(SystemExit) as got:
            g5_nli.main(["--sets", self.tmp, "--url", f"http://127.0.0.1:{port}", "--seen", "none",
                         "--out-dir", os.path.join(self.tmp, "out"), "--stamp", "2099-01-01"])
        self.assertEqual(got.exception.code, 2)


class JudgeTest(unittest.TestCase):
    def arm(self, recall, fa, cmis, p):
        t = {"recall_b": recall, "false_alarm_a": fa, "c_misread": cmis}
        m = {"recall_b": {"p": p}, "false_alarm_a": {"p": 1.0}}
        return {"semif": {"all": {"recall_b": 0.5, "false_alarm_a": 0.37}},
                "rulers": {"g5": {"all": t}, "cal": {"all": t}}, "mcnemar": {"g5": {"all": m}, "cal": {"all": m}}}

    def test_three_outcomes(self):
        self.assertEqual(g5_nli.judge({"raw": self.arm(0.7, 0.2, 0.05, 0.01)})["verdict"], "넘음")
        self.assertEqual(g5_nli.judge({"raw": self.arm(0.96, 0.05, 0.05, 0.001)})["verdict"], "통과선")
        self.assertEqual(g5_nli.judge({"raw": self.arm(0.7, 0.2, 0.05, 0.2)})["verdict"], "못 넘음")
        self.assertEqual(g5_nli.judge({"raw": self.arm(0.7, 0.2, 0.2, 0.01)})["verdict"], "못 넘음")   # 무관 오인 > 10%
        self.assertEqual(g5_nli.judge({"raw": self.arm(0.7, 0.4, 0.05, 0.01)})["verdict"], "못 넘음")  # 오탐이 더 나쁨


class FakeServer(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"ok": True, "model": "fake", "name": "fake-nli", "backend": "test", "letters": ["a", "b", "c"]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        assert req["label_set"] == "nli3"
        self.reply(fake_answer(req["evidence"], req["claim"]))


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers 없음")
class EndToEndTest(unittest.TestCase):
    """진짜 tokenizers 글자 토크나이저 + 가짜 HTTP 서버로 main 을 끝까지 돌린다."""

    def setUp(self):
        from test_pairenc import char_tokenizer

        self.tmp = tempfile.mkdtemp()
        self.sets = os.path.join(self.tmp, "sets")
        os.makedirs(self.sets)
        make_sets(self.sets)
        text = SHORT + LONG + "MAX를2030으로넣었다항목9테스트고쳤다1·Docs/long.md2other.ts다른일"
        tok_dir = os.path.join(self.tmp, "model", "tokenizer")
        os.makedirs(tok_dir)
        char_tokenizer(text, xlmr=True).save(os.path.join(tok_dir, "tokenizer.json"))
        with open(os.path.join(self.tmp, "model", "model.json"), "w", encoding="utf-8") as f:
            json.dump({"input": {"max_len": 128}}, f)
        self.tok_dir = tok_dir
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_out_dir_is_nli_subfolder(self):
        g5_nli.main(["--sets", self.sets, "--url", self.url, "--tokenizer", self.tok_dir, "--seen", "none",
                     "--stamp", "2099-01-01"])
        self.assertEqual(len(os.listdir(os.path.join(self.sets, "nli"))), 3)
        self.assertFalse([n for n in os.listdir(self.sets) if n.startswith("2099-01-01")])   # 셋 폴더 바로 아래엔 안 씀

    def test_main_writes_three_files(self):
        out = os.path.join(self.tmp, "out")
        args = ["--sets", self.sets, "--url", self.url, "--tokenizer", self.tok_dir, "--seen", "none",
                "--out-dir", out, "--stamp", "2099-01-01"]
        summary = g5_nli.main(args)
        self.assertEqual(summary["pairs"], 6)
        self.assertEqual(summary["max_len"], 128)
        self.assertEqual(sorted(os.listdir(out)),
                         ["2099-01-01-nli-claim.jsonl", "2099-01-01-nli-item.jsonl", "2099-01-01-nli-summary.json"])
        with self.assertRaises(SystemExit):
            g5_nli.main(args)                                   # 이미 있으면 --force 없이 멈춤
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(self.tok_dir, "tokenizer.json"))
        with open(os.path.join(out, "2099-01-01-nli-claim.jsonl"), encoding="utf-8") as f:
            lines = [json.loads(x) for x in f]
        with open(os.path.join(self.sets, "2026-10-08-claim.jsonl"), encoding="utf-8") as f:
            rows = {r["id"]: r for r in map(json.loads, f)}
        for line in lines:
            if line["arm"] == "raw":
                r = rows[line["id"]]
                self.assertEqual(line["cut"], encode_pair(tok, r["evidence"], r["claim"], 128)[1])


if __name__ == "__main__":
    unittest.main()
