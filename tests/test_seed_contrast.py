"""seed_contrast — 씨앗 값 자리 · 새 값 · a 꼴 · 금지 목록 · 기계 검사 · count→make 흐름. 입력은 모두 지어낸 짧은 글이다."""
import collections
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
import unittest.mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import scrub  # noqa: E402
import seed_contrast as sc  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def kinds(text):
    return [(s["kind"], s["text"]) for s in sc.find_slots(text)]


class SlotTest(unittest.TestCase):
    def test_finds_value_slots_and_skips_identifiers(self):
        self.assertEqual(kinds("시험 3개를 40% 줄여 2.5배 빨라졌다."), [("num", "3개"), ("num", "40%"), ("num", "2.5배")])
        self.assertEqual(kinds("10-12 저녁에 2차 판을 마지막으로 돌렸다."),
                         [("date", "10-12"), ("date", "저녁"), ("ord", "2차"), ("firstlast", "마지막")])
        self.assertEqual(kinds("사용자가 0.65 이상이면 켬으로 둔다."),
                         [("who", "사용자"), ("num", "0.65"), ("bound", "이상"), ("onoff", "켬")])
        # 판 이름 · 절 번호 · 기억 id · 경로 속 날짜 · 옵션 뒤 수 · 네 자리 맨 수는 값 자리가 아니다
        self.assertEqual(kinds("v1.2 의 3-2 절과 20261012-abcd1234 를 보고 sets/2026-10-10/ 에 -np 1 로 8092 를 띄웠다."), [])
        self.assertEqual(kinds("19,880건은 15분쯤."), [("num", "19,880건"), ("num", "15분")])

    def test_new_value_same_kind_and_absent(self):
        rng = random.Random(1)
        seed = "시험 4개 중 3개가 10-12 에 2차로 통과했고 기능은 켬이다."
        slots = {s["text"]: s for s in sc.find_slots(seed)}
        for _ in range(20):
            new = sc.new_value(slots["3개"], seed, rng)
            n = int(new[:-1])
            self.assertTrue(new.endswith("개") and n != 3 and n != 4 and 1 <= n <= 6, new)
            day = sc.new_value(slots["10-12"], seed, rng)
            self.assertRegex(day, r"^10-(09|10|11|13|14|15)$")
            self.assertIn(sc.new_value(slots["2차"], seed, rng), ("1차", "3차"))
        self.assertEqual(sc.new_value(slots["켬"], seed, rng), "끔")
        rate = sc.find_slots("문턱 .65 를 넘으면 뺀다.")[0]
        for _ in range(10):
            v = float("0" + sc.new_value(rate, "문턱 .65 를 넘으면 뺀다.", rng))
            self.assertTrue(0.325 <= v < 1 and v != 0.65)

    def test_same_value_other_form(self):
        def a_of(text):
            return sc.same_value_other_form(sc.find_slots(text)[0])
        self.assertEqual(a_of("90초 걸렸다."), "1분 30초")
        self.assertEqual(a_of("47000원 들었다."), "4만 7천원")
        self.assertEqual(a_of("3개 남았다."), "세 개")
        self.assertEqual(a_of("10-12 에 정했다."), "10월 12일")
        self.assertEqual(a_of("2차 판이다."), "두 번째")
        self.assertIsNone(a_of("40% 줄었다."))          # 규칙으로 못 만드는 꼴 → 로컬 모델 몫

    def test_skips_versions_ranges_idioms_fixed_phrases(self):
        self.assertEqual(kinds("09-01 에 틀렸던 3.6 으로 다시 재야 한다."), [("date", "09-01")])   # 판 이름 꼴 맨 소수
        self.assertEqual(kinds("최근(9~10월) 발표다."), [])                                         # 범위 끝 날짜
        self.assertEqual(kinds("이름이 같아 100% 확답은 못 한다."), [])                              # 관용구
        self.assertEqual(kinds("100% 통과했다."), [("num", "100%")])
        self.assertEqual(kinds("처음부터 다시 만들고 마지막까지 본다."), [])                         # 굳은 말
        self.assertEqual(kinds("처음에 2개를 돌렸다."), [("firstlast", "처음"), ("num", "2개")])

    def test_new_values_keep_range_order(self):
        seed = "단계 4 는 08-30 23:35 ~ 08-31 01:15 (99분) 에 모델 5개로 돌았다."
        self.assertTrue(sc.order_broken("단계 4 는 09-01 23:35 ~ 08-31 01:15"))
        self.assertFalse(sc.order_broken(seed))
        row = {"text": seed, "slots": sc.find_slots(seed)}
        self.assertEqual([s["text"] for s in row["slots"] if s["kind"] == "date"], ["08-30"])   # 08-31 은 범위 끝
        plans = [p for p in (sc.plan_seed(row, "date", seed=k) for k in range(30)) if p]
        self.assertGreater(len(plans), 20)
        for plan in plans:
            self.assertFalse(sc.order_broken(plan["drafts"]["b"]), plan["drafts"]["b"])
            if plan["d"]:
                self.assertFalse(sc.order_broken(plan["d"]["draft"]), plan["d"]["draft"])


class ForbidTest(unittest.TestCase):
    def test_groups_forbid_and_reasons(self):
        self.assertEqual(sc.group_to_key("doc:studio:Docs/a.md"), ("doc", "Docs/a.md"))
        self.assertEqual(sc.group_to_key("doc:ArtTool:Docs/b.md"), ("doc", "ArtTool/Docs/b.md"))
        self.assertEqual(sc.group_to_key('git:SampleTool:"Docs/\\352\\260\\200.md"'), ("doc", "SampleTool/Docs/가.md"))
        self.assertIsNone(sc.group_to_key("doc:game:Docs/c.md"))
        self.assertIsNone(sc.group_to_key("vitc:Apeshit"))
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "sets", "g1"))
            os.makedirs(os.path.join(root, "sets", "g2"))
            os.makedirs(os.path.join(root, "measure", "use1", "x"))
            os.makedirs(os.path.join(root, "build", "g2"))
            row = {"evidence": "평가 문장 3개.", "claim": "다른 문장.", "group": "doc:studio:Docs/a.md",
                   "groups": ["mem:20260101-aaaaaaaa"]}
            for name in sc.G1_EVAL:
                with open(os.path.join(root, "sets", "g1", name + ".jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for name in sc.G2_EVAL:
                open(os.path.join(root, "sets", "g2", name + ".jsonl"), "w").close()
            with open(os.path.join(root, "sets", "g2", "blind-g2.forbid.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_paths": ["ArtTool:Docs/b.md"], "commits": ["DataTool:abc"], "mem_ids": ["20260202-bbbbbbbb"],
                           "groups": []}, f)
            with open(os.path.join(root, "measure", "use1", "x", "set-use1.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"evidence": "use1 문장 2개.", "claim": "c"}, ensure_ascii=False) + "\n")
            with open(os.path.join(root, "build", "g2", "forbid-ruler-ids.txt"), "w") as f:
                f.write("20260303-cccccccc\n")
            forbid = sc.SeedForbid(root, "studio", git_files=lambda repo, sha: ["Docs/c.md"])
        self.assertEqual(forbid.missing, [])
        self.assertTrue({"Docs/a.md", "ArtTool/Docs/b.md", "DataTool/Docs/c.md"} <= forbid.docs)

        def seed(**kw):
            base = {"source": "doc", "path": "Docs/z.md", "mem_id": "", "group": "doc:studio:Docs/z.md", "text": "새 문장 5개."}
            return dict(base, **kw)
        self.assertEqual(forbid.reasons(seed()), [])
        self.assertEqual(forbid.reasons(seed(path="ArtTool/Docs/b.md")), ["doc_eval"])
        self.assertEqual(forbid.reasons(seed(text="use1 문장 2개.")), ["text_eval"])
        self.assertIn("mem_eval", forbid.reasons(seed(source="mem", mem_id="20260202-bbbbbbbb", group="mem:x")))
        self.assertIn("mem_group_eval", forbid.reasons(seed(source="mem", mem_id="20260404-dddddddd",
                                                            group="mem:20260101-aaaaaaaa")))
        self.assertIn("mem_ruler", forbid.reasons(seed(source="mem", mem_id="20260303-cccccccc", group="mem:y")))


class CheckTest(unittest.TestCase):
    def test_check_lines(self):
        scrubber = scrub.Scrubber(["비밀이름"], ())
        seed = "10-12 에 시험 3개가 통과했다."
        row = {"seed": seed, "new": "5개", "slot": {"kind": "num"}, "d": {"time_new": "10-14"}}
        self.assertEqual(sc.check_line(seed, "10-12 에 시험 5개가 통과했다.", "b", row, scrubber), [])
        self.assertIn("chunks", sc.check_line(seed, "10-12 에 시험 5개가 실패했다.", "b", row, scrubber))
        self.assertIn("negation", sc.check_line(seed, "10-12 에 시험 5개가 통과하지 않았다.", "b", row, scrubber))
        self.assertIn("question", sc.check_line(seed, "10-12 에 시험 5개가 통과했나?", "b", row, scrubber))
        self.assertIn("scrub", sc.check_line(seed, "10-12 에 비밀이름 5개가 통과했다.", "b", row, scrubber))
        self.assertIn("new_missing", sc.check_line(seed, "10-12 에 시험 4개가 통과했다.", "b", row, scrubber))
        self.assertEqual(sc.check_line(seed, "10-12 에 시험 세 개가 통과했다.", "a", row, scrubber), [])
        self.assertEqual(sc.check_line(seed, "10-14 에 시험 7개가 통과했다.", "d", row, scrubber), [])
        self.assertIn("time_missing", sc.check_line(seed, "10-12 에 시험 7개가 통과했다.", "d", row, scrubber))
        self.assertTrue(sc.has_value("2", "2 가 나왔다") and not sc.has_value("1", "16 · 2026 년"))

    def test_check_skips_non_text_lines(self):
        scrubber = scrub.Scrubber([], ())
        seed = "10-12 에 시험 3개가 통과했다."
        row = {"seed": seed, "new": "5개", "slot": {"kind": "num", "text": "3개"}, "want_lines": ["b", "a"],
               "lines": {"b": ["10-12 에 시험 5개가 통과했다."], "a": 7}}
        results, count = sc.check_rows([row, {"seed": seed, "lines": "깨진 칸"}], scrubber)
        self.assertEqual(results[0]["why"], {"b": ["not_text"], "a": ["not_text"]})
        self.assertEqual((count["rows"], count["rows_pass"]), (2, 1))   # lines 가 dict 가 아니면 볼 줄이 없다

    def test_token_limit_catches_rewrite(self):
        scrubber = scrub.Scrubber([], ())
        seed = "10-12 에 시험 3개가 통과했고 판정은 그대로 두었다."
        row = {"seed": seed, "new": "5개", "slot": {"kind": "num", "text": "3개"},
               "d": {"time_slot": "10-12", "time_new": "10-14", "value_slot": "3개", "value_new": "6개"}}
        rewrite = "10-12 에 시험 5개를 다시 돌려 모두 바꿨고 판정은 그대로 두었다."   # 덩이 하나지만 토큰 여럿
        self.assertEqual(sc.chunks(seed, rewrite), 1)
        self.assertIn("tokens", sc.check_line(seed, rewrite, "b", row, scrubber))
        self.assertEqual(sc.check_line(seed, "10-12 에 시험 5개가 통과했고 판정은 그대로 두었다.", "b", row, scrubber), [])
        self.assertEqual(sc.check_line(seed, "10-14 에 시험 6개가 통과했고 판정은 그대로 두었다.", "d", row, scrubber), [])
        ymd_seed = "2026-08-28 기준으로 3개를 뺐다."
        ymd_row = {"seed": ymd_seed, "slot": {"kind": "date", "text": "2026-08-28"}, "a_value": "2026년 8월 28일"}
        self.assertNotIn("tokens", sc.check_line(ymd_seed, "2026년 8월 28일 기준으로 3개를 뺐다.", "a", ymd_row, scrubber))
        self.assertEqual(sc.token_limit("a", {"slot": {"text": "40%"}}), 3)

    def test_a_has_b_value_reads_korean_numerals(self):
        self.assertEqual(sc.ko_numerals_to_digits("구백 건 · 열흘 · 마흔한 명 · 이십구 시간 · 만들기"),
                         "900 건 · 10일 · 41 명 · 29 시간 · 만들기")
        self.assertTrue(sc.a_has_b_value("900건", "기억이 450건이다.", "기억이 구백 건이다."))
        self.assertTrue(sc.a_has_b_value("10일", "5일 걸렸다.", "열흘 걸렸다."))
        self.assertTrue(sc.a_has_b_value("41명", "40명 왔다.", "마흔한 명 왔다."))
        self.assertFalse(sc.a_has_b_value("5개", "3개 남았다.", "세 개 남았다."))
        self.assertFalse(sc.a_has_b_value("09-17", "09-15 에 쟀다.", "9월 15일에 쟀다.", kind="date"))


class FlowTest(unittest.TestCase):
    def test_count_make_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            studio, judge, out = (os.path.join(tmp, n) for n in ("studio", "judge", "out"))
            docs = os.path.join(studio, "ToolA", "Docs", "Research")
            os.makedirs(docs)
            for d in range(8):
                lines = ["# 조사", "", "## 1. 결과", ""]
                for i in range(6):
                    k = d * 6 + i
                    lines.append(f"- 10-{10 + k % 9:02d} 에 시험 {k % 7 + 2}개를 돌려 {k + 20}초 걸렸고 사용자가 판 {k}을 정했다.")
                with open(os.path.join(docs, f"2026-10-13-가짜{d}.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            os.makedirs(judge)
            run = lambda *argv: sc.main(list(argv))  # noqa: E731
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run("count", "--out", out, "--studio", studio, "--judge-root", judge, "--allow-missing", "--allow-git",
                    "--names", "--game-names")
                run("make", "--out", out, "--judge-root", judge, "--n", "12", "--allow-git", "--names", "--game-names")
                first = read(os.path.join(out, "input.jsonl"))
                run("make", "--out", out, "--judge-root", judge, "--n", "12", "--allow-git", "--names", "--game-names")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    run("check", "--in", os.path.join(out, "input.jsonl"), "--judge-root", judge, "--names", "--game-names")
            rows = [json.loads(line) for line in first.splitlines()]
            self.assertEqual(first, read(os.path.join(out, "input.jsonl")))   # 같은 입력 → 같은 출력
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(r["instruction"] == sc.INSTRUCTION and r["drafts"]["b"] != r["seed"] for r in rows))
            self.assertTrue(any(r["d"] for r in rows))
            self.assertLessEqual(max(sum(1 for r in rows if r["group"] == g) for g in {r["group"] for r in rows}), sc.GROUP_CAP)
            count = json.loads(buf.getvalue())
            self.assertEqual(count["b_pass"], count["b_total"])



def write_docs(studio):
    """FlowTest 와 같은 가짜 문서 여덟 — 절 하나에 문장 여섯."""
    docs = os.path.join(studio, "ToolA", "Docs", "Research")
    os.makedirs(docs)
    for d in range(8):
        lines = ["# 조사", "", "## 1. 결과", ""]
        for i in range(6):
            k = d * 6 + i
            lines.append(f"- 10-{10 + k % 9:02d} 에 시험 {k % 7 + 2}개를 돌려 {k + 20}초 걸렸고 사용자가 판 {k}을 정했다.")
        with open(os.path.join(docs, f"2026-10-13-가짜{d}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def sha_of(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ParagraphTest(unittest.TestCase):
    """K4 archive 씨앗 (원천 균형 판 설계 3-2 · 5-5 ③) — --evidence paragraph · K 씨앗 빼기 · 옛 기본값 바이트 그대로."""

    # 고치기 전 seed_contrast 로 write_docs 를 count · make(--n 12) 한 출력의 sha256 (2026-10-14 손질 전에 잼)
    OLD_SHA = {"seeds-all.jsonl": "fa6f5b14bddd75b14f1bd93d8ec63a99c90be2bd677ff6dda715b4c3d26c79cf",
               "input.jsonl": "df312da36c80a17d7a697c750fb2ba822faa760d373b4ad382df40bbb1878d5a",
               "sample50.jsonl": "df312da36c80a17d7a697c750fb2ba822faa760d373b4ad382df40bbb1878d5a",
               "seed-groups.txt": "8ecd4d723d95f80efe8720e7d662970155db995bf381d00b7abb12158894f3f1",
               "make.json": "8a70d917244c106256be011bac581df475bdb2405a906d0add6accf4ca02c59f"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name
        self.studio, self.judge, self.out = (os.path.join(tmp, n) for n in ("studio", "judge", "out"))
        write_docs(self.studio)
        os.makedirs(self.judge)

    def tearDown(self):
        self._tmp.cleanup()

    def run_sc(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            sc.main(list(argv))
        return buf.getvalue()

    def count(self, *extra):
        return self.run_sc("count", "--out", self.out, "--studio", self.studio, "--judge-root", self.judge,
                           "--allow-missing", "--allow-git", "--names", "--game-names", *extra)

    def make(self, *extra):
        return self.run_sc("make", "--out", self.out, "--judge-root", self.judge, "--n", "12", "--allow-git",
                           "--names", "--game-names", *extra)

    def test_default_output_bytes_same_as_before(self):
        self.count()
        self.make()
        for name, sha in self.OLD_SHA.items():
            self.assertEqual(sha_of(os.path.join(self.out, name)), sha, name)

    def test_paragraph_rows(self):
        self.count("--evidence", "paragraph")
        self.make("--evidence", "paragraph", "--domain", "archive")
        rows = [json.loads(line) for line in read(os.path.join(self.out, "input.jsonl")).splitlines()]
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertTrue(r["seed_id"].startswith(sc.PARA_ID))
            self.assertEqual((r["domain"], r["evidence_mode"], r["source"]), ("archive", "paragraph", "doc"))
            self.assertIn(r["seed"], r["evidence"])                       # 근거 문단에 씨앗 문장이 든다
            self.assertGreater(len(r["evidence"]), len(r["seed"]))        # 문장 하나보다 길다 (앞뒤 문장)
            self.assertLessEqual(r["evidence"].count("정했다."), 1 + 2 * sc.PARA_SPAN)
        make = json.loads(read(os.path.join(self.out, "make.json")))
        self.assertEqual((make["evidence"], make["domain"]), ("paragraph", "archive"))

    def test_k_seed_sentences_excluded(self):
        self.count("--evidence", "paragraph")
        seeds = [json.loads(line) for line in read(os.path.join(self.out, "seeds-all.jsonl")).splitlines()]
        k_input = os.path.join(self.judge, "k-input.jsonl")
        with open(k_input, "w", encoding="utf-8") as f:
            for r in seeds[:5]:
                f.write(json.dumps({"seed_id": "k0001", "seed": r["text"]}, ensure_ascii=False) + "\n")
        out = self.count("--evidence", "paragraph", "--exclude-seeds", k_input)
        again = [json.loads(line)["text"] for line in read(os.path.join(self.out, "seeds-all.jsonl")).splitlines()]
        self.assertEqual(len(again), len(seeds) - 5)
        self.assertFalse({r["text"] for r in seeds[:5]} & set(again))
        report = json.loads(out)
        self.assertEqual((report["count"]["k_seed"], report["exclude_seeds"]), (5, 5))

    def test_paragraph_default_k_seed_file_needed(self):
        # --allow-missing 없이 paragraph 꼴은 K 씨앗 기본 파일(<judge-root>/build/g3/seed/input.jsonl)이 있어야 한다
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            sc.main(["count", "--out", self.out, "--studio", self.studio, "--judge-root", self.judge, "--allow-git",
                     "--names", "--game-names", "--evidence", "paragraph"])

    def test_make_paragraph_needs_paragraph_seeds(self):
        self.count()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            sc.main(["make", "--out", self.out, "--judge-root", self.judge, "--allow-git", "--names", "--game-names",
                     "--evidence", "paragraph"])

    def test_paragraph_group_cap_two(self):
        self.count("--evidence", "paragraph")
        self.make("--evidence", "paragraph")
        rows = [json.loads(line) for line in read(os.path.join(self.out, "input.jsonl")).splitlines()]
        per = collections.Counter(r["group"] for r in rows)
        self.assertLessEqual(max(per.values()), sc.PARA_GROUP_CAP)
        self.assertEqual(sc.PARA_GROUP_CAP, 2)

    def test_paragraph_forbid_adds_g3_g4_eval(self):
        def put(rel, rows):
            path = os.path.join(self.judge, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        put("sets/g4/dev-cmp-g4.jsonl", [{"evidence": "g4 평가 문장.", "claim": "c", "group": "doc:studio:Docs/g4.md"}])
        put("sets/g3/blind-g3.jsonl", [{"evidence": "g3 평가 문장.", "claim": "c", "group": "doc:studio:Docs/g3.md"}])
        with open(os.path.join(self.judge, "sets", "g4", "blind-g4.forbid.json"), "w", encoding="utf-8") as f:
            json.dump({"groups": ["doc:ArtTool:Docs/f4.md"]}, f)
        old = sc.SeedForbid(self.judge)
        new = sc.SeedForbid(self.judge, evidence="paragraph")
        want = {"Docs/g4.md", "Docs/g3.md", "ArtTool/Docs/f4.md"}
        self.assertFalse(want & old.docs)                                   # seed 꼴은 옛 금지 그대로
        self.assertEqual(want & new.docs, want)
        self.assertIn(sc.norm("g4 평가 문장."), new.texts)
        self.assertNotIn("blind-g4.forbid.json", new.missing)
        self.assertIn("blind-g3.forbid.json", new.missing)
        self.assertNotIn("blind-g3.forbid.json", old.missing)

    def test_paragraph_window(self):
        doc = {"intro_ok": ["머리 하나다."], "sections": [("가", ["첫 문장 이다.", "둘째 문장 이다.", "셋째 문장 이다."])]}
        with unittest.mock.patch.object(sc.ma, "arch_text_ok", lambda t: True):
            got = sc._paragraphs(doc)
        self.assertEqual(got["첫 문장 이다."], "첫 문장 이다. 둘째 문장 이다.")
        self.assertEqual(got["둘째 문장 이다."], "첫 문장 이다. 둘째 문장 이다. 셋째 문장 이다.")
        self.assertEqual(got["머리 하나다."], "머리 하나다.")


if __name__ == "__main__":
    unittest.main()
