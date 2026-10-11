"""label_batches · rulers · teacher — 라벨 묶음 · 합의 · κ · 자 ①③ · 교사 거르기. 입력은 모두 지어낸 짧은 글이다."""
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
import teacher  # noqa: E402

GUIDE = """# 기준

## 1. 네 칸

칸 넷.

## 2. 2단 지침

Q1 Q2 Q3.

## 3. 자주 갈리는 자리 규칙

| 자리 | 규칙 | 보기 (지어냄) |
| --- | --- | --- |
| 수치 | 같은 때면 b | 보기1 |
| | | 보기2 |
| 다음 단계 | 새 일만 d | 보기3 |
| 같은 문서 | 대상이 다르면 c | 보기4 |
| | | 보기5 |

- 표에 없는 자리.

## 4. 쌍 파일 값

안 붙임.
"""


def cands(n, domain="proposals"):
    return [{"id": f"x{i:03d}", "evidence": f"근거 문장 {i}", "claim": f"주장 문장 {i}", "hint": "d",
             "domain": domain, "lang": "ko", "group": f"g{i // 3}"} for i in range(n)]


def traps():
    rows = []
    for label in "abcd":
        for i in range(4):
            rows.append({"tid": f"t{label}{i}", "evidence": f"함정 근거 {label}{i}",
                         "claim": f"함정 주장 {label}{i}", "gold": label})
    return rows


def answer(label, q):
    """칸 → 앞뒤가 맞는 판정자 답 줄."""
    q1, q2, q3 = {"a": ("예", "아니오", "예"), "b": ("예", "예", "-"), "c": ("아니오", "-", "-"),
                  "cd": ("아니오", "-", "-"), "d": ("예", "아니오", "아니오")}[label]
    return {"q": q, "q1": q1 + " — 까닭", "q2": q2, "q3": q3, "label": label}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.out = os.path.join(root, "lab")
        self.cands_path = os.path.join(root, "cands.jsonl")
        self.traps_path = os.path.join(root, "traps.jsonl")
        self.guide_path = os.path.join(root, "guide.md")
        write_jsonl(self.cands_path, cands(250))
        write_jsonl(self.traps_path, traps())
        with open(self.guide_path, "w", encoding="utf-8") as f:
            f.write(GUIDE)
        lb.main(["make", "--cands", self.cands_path, "--out", self.out, "--mode", "train",
                 "--traps", self.traps_path, "--guide", self.guide_path, "--allow-git"])
        self.map = read_jsonl(os.path.join(self.out, "map", "map.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_trap_share_and_spread(self):
        by_batch = collections.defaultdict(list)
        for m in self.map:
            by_batch[m["batch"]].append(m)
        self.assertEqual(sorted(by_batch), ["b01", "b02", "b03"])
        first = by_batch["b01"]
        self.assertEqual(len(first), 100)
        trap_rows = [m for m in first if m["trap"]]
        self.assertEqual(len(trap_rows), 10)
        per_label = collections.Counter(m["gold"] for m in trap_rows)
        self.assertEqual(set(per_label), set("abcd"))
        self.assertTrue(all(2 <= n <= 3 for n in per_label.values()))
        for rows in by_batch.values():
            share = sum(1 for m in rows if m["trap"]) / len(rows)
            self.assertAlmostEqual(share, 0.10, delta=0.02)

    def test_ids_opaque_and_map_restores(self):
        shown = read_jsonl(os.path.join(self.out, "batches", "b01-j1.jsonl"))
        for row in shown:
            self.assertEqual(set(row), {"q", "evidence", "claim"})
            self.assertRegex(row["q"], r"^q\d{5}$")
        qs = [m["q"] for m in self.map]
        self.assertEqual(len(qs), len(set(qs)))
        # q 차례가 원래 id 차례를 드러내지 않는다
        real = sorted((m["q"], m["id"]) for m in self.map if not m["trap"])
        self.assertNotEqual([i for _, i in real], sorted(i for _, i in real))
        # 대응표로 글을 원래 쌍에 되돌릴 수 있다
        by_q = {m["q"]: m for m in self.map}
        source = {c["id"]: c for c in read_jsonl(self.cands_path)}
        for row in shown:
            m = by_q[row["q"]]
            if not m["trap"]:
                self.assertEqual(source[m["id"]]["claim"], row["claim"])
        self.assertEqual(sum(1 for m in self.map if not m["trap"]), 250)
        self.assertFalse(any("hint" in r or "domain" in r for r in shown))

    def test_judges_get_different_order_and_examples(self):
        j1 = [r["q"] for r in read_jsonl(os.path.join(self.out, "batches", "b01-j1.jsonl"))]
        j2 = [r["q"] for r in read_jsonl(os.path.join(self.out, "batches", "b01-j2.jsonl"))]
        self.assertEqual(sorted(j1), sorted(j2))
        self.assertNotEqual(j1, j2)
        with open(os.path.join(self.out, "batches", "b01-j1.txt"), encoding="utf-8") as f:
            t1 = f.read()
        with open(os.path.join(self.out, "batches", "b01-j2.txt"), encoding="utf-8") as f:
            t2 = f.read()
        self.assertLess(t1.index("| 수치 |"), t1.index("| 같은 문서 |"))
        self.assertGreater(t2.index("| 수치 |"), t2.index("| 같은 문서 |"))
        self.assertIn("cd 를 쓸 수 있다", t1)
        self.assertNotIn("## 4.", t1)

    def test_variant3_reverses_examples(self):
        text = lb.reorder_places(lb.guide_sections(GUIDE), 3, 1)
        self.assertLess(text.index("보기5"), text.index("보기4"))
        self.assertIn("| 같은 문서 | 대상이 다르면 c | 보기5 |", text)

    def test_refuses_existing_out(self):
        with self.assertRaises(SystemExit):
            lb.main(["make", "--cands", self.cands_path, "--out", self.out, "--mode", "train",
                     "--traps", self.traps_path, "--guide", self.guide_path, "--allow-git"])

    def test_no_trap_pool_still_runs(self):
        out = os.path.join(self.tmp.name, "lab2")
        lb.main(["make", "--cands", self.cands_path, "--out", out, "--mode", "eval",
                 "--traps", os.path.join(self.tmp.name, "none.jsonl"), "--guide", self.guide_path, "--allow-git"])
        rows = read_jsonl(os.path.join(out, "map", "map.jsonl"))
        self.assertFalse(any(m["trap"] for m in rows))
        with open(os.path.join(out, "batches", "b01-j1.txt"), encoding="utf-8") as f:
            self.assertIn("cd 는 쓰지 않는다", f.read())

    def test_refuses_bad_size_or_trap_rate(self):
        for extra in (["--size", "-5"], ["--trap-rate", "1"], ["--trap-rate", "-0.1"]):
            out = os.path.join(self.tmp.name, "lab-bad")
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                lb.main(["make", "--cands", self.cands_path, "--out", out, "--mode", "train",
                         "--traps", self.traps_path, "--guide", self.guide_path, "--allow-git", *extra])
            self.assertFalse(os.path.exists(os.path.join(out, "map", "map.jsonl")))

    def test_merge_drops_low_trap_run_and_third_flow(self):
        by_batch = collections.defaultdict(list)
        for m in self.map:
            by_batch[m["batch"]].append(m)
        answers = os.path.join(self.out, "answers")
        for batch, rows in by_batch.items():
            for judge in (1, 2):
                lines = []
                for k, m in enumerate(rows):
                    if m["trap"]:
                        label = m["gold"]
                        if batch == "b02" and judge == 2 and k % 2 == 0:
                            label = "a" if m["gold"] != "a" else "b"   # 함정을 일부러 틀림
                    else:
                        label = "d"
                        if judge == 2 and int(m["id"][1:]) % 5 == 0:
                            label = "c"
                    lines.append(answer(label, m["q"]))
                write_jsonl(os.path.join(answers, f"{batch}-j{judge}.jsonl"), lines)
        lb.main(["merge", "--out", self.out])
        with open(os.path.join(self.out, "merge-report.json"), encoding="utf-8") as f:
            report = json.load(f)
        dropped = [r for r in report["runs"] if not r["kept"]]
        self.assertEqual([(r["batch"], r["judge"]) for r in dropped], [("b02", "j2")])
        votes = read_jsonl(os.path.join(self.out, "votes.jsonl"))
        b02 = {m["q"] for m in by_batch["b02"] if not m["trap"]}
        self.assertTrue(all("j2" not in v["votes"] for v in votes if v["q"] in b02))
        self.assertFalse(any(v["id"].startswith("t") for v in votes))
        lb.main(["decide", "--out", self.out, "--mode", "train"])
        decided = read_jsonl(os.path.join(self.out, "decided.jsonl"))
        status = collections.Counter(r["status"] for r in decided)
        self.assertGreater(status["need-third"], 0)
        self.assertGreater(status["missing"], 0)
        lb.main(["third", "--out", self.out])
        mapping = read_jsonl(os.path.join(self.out, "map", "map.jsonl"))
        third_real = [m for m in mapping if m["batch"].startswith("t") and not m["trap"]]
        need = {r["q"] for r in decided if r["status"] == "need-third"}
        self.assertEqual({m["q"] for m in third_real}, need)
        self.assertTrue(any(m["trap"] for m in mapping if m["batch"].startswith("t")))
        shown = read_jsonl(os.path.join(self.out, "batches", "t01-j3.jsonl"))
        self.assertTrue(need & {r["q"] for r in shown})
        # 셋째가 d 를 주면 2/3 로 d
        t_rows = [m for m in mapping if m["batch"] == "t01"]
        write_jsonl(os.path.join(answers, "t01-j3.jsonl"),
                    [answer(m["gold"] if m["trap"] else "d", m["q"]) for m in t_rows])
        lb.main(["merge", "--out", self.out])
        lb.main(["decide", "--out", self.out, "--mode", "train"])
        decided = {r["q"]: r for r in read_jsonl(os.path.join(self.out, "decided.jsonl"))}
        for q in need:
            self.assertEqual((decided[q]["label"], decided[q]["status"]), ("d", "third"))


class AssembleTest(unittest.TestCase):
    """셋 조립 — 작은 가짜 묶음 폴더 둘(공개 · 아카이브)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.dest = os.path.join(self.root, "sets")

    def tearDown(self):
        self.tmp.cleanup()

    def make_dir(self, kind, cands_rows, decided, trap_map=(), pool=()):
        parent = os.path.join(self.root, kind)
        out = os.path.join(parent, "dev-cal")
        cpath = os.path.join(parent, "dev-cal-cands.jsonl")
        write_jsonl(cpath, cands_rows)
        pool_path = os.path.join(parent, "traps.jsonl")
        write_jsonl(pool_path, list(pool))
        mapping = [{"q": d["q"], "batch": "b01", "id": d["id"], "trap": False} for d in decided]
        mapping += [{"q": q, "batch": "b01", "id": tid, "trap": True, "gold": "a"} for q, tid in trap_map]
        write_jsonl(os.path.join(out, "map", "map.jsonl"), mapping)
        lb.write_json(os.path.join(out, "map", "manifest.json"), {"cands": cpath, "traps": pool_path, "mode": "eval"})
        write_jsonl(os.path.join(out, "decided.jsonl"), decided)
        write_jsonl(os.path.join(out, "votes.jsonl"),
                    [{"q": d["q"], "id": d["id"], "votes": d["votes"]} for d in decided])
        lb.write_json(os.path.join(out, "merge-report.json"), {"runs": [{"kept": True}]})
        return out

    @staticmethod
    def cand(i, prefix, **extra):
        row = {"id": f"{prefix}{i:02d}", "evidence": f"근거 {prefix}{i}", "claim": f"주장 {prefix}{i}", "want": "",
               "hint": "d", "form": "near", "domain": "mem" if prefix == "m" else "vitaminc", "group": f"g{i}",
               "groups": [f"g{i}"], "lang": "ko", "lic": "x", "split": "dev", "src_ids": [], "origin": "o",
               "set": "dev-cal"}
        row.update(extra)
        return row

    @staticmethod
    def dec(i, prefix, label, status="agree"):
        votes = {"j1": label or "a", "j2": label or "c"}
        if status in ("third", "drop"):
            votes["j3"] = label or "d"
        return {"q": f"q{prefix}{i}", "id": f"{prefix}{i:02d}", "domain": "", "label": label, "status": status,
                "votes": votes}

    def public(self, extra_dec=(), extra_cand=()):
        cands_rows = [self.cand(i, "p") for i in range(4)] + list(extra_cand)
        decided = [self.dec(3, "p", "a"), self.dec(1, "p", "b", "third"), self.dec(0, "p", None, "drop"),
                   self.dec(2, "p", "c")] + list(extra_dec)
        return cands_rows, decided

    def archive(self):
        cands_rows = [self.cand(0, "m", relabel=True, old_want="related", source="mem"),
                      self.cand(1, "m", relabel=True, old_want="related", source="mem"),
                      self.cand(2, "m", relabel=True, old_want="contradict", source="mem"),
                      self.cand(3, "m"),
                      self.cand(4, "m", relabel=True, old_want="support")]
        decided = [self.dec(0, "m", "d"), self.dec(1, "m", "c"), self.dec(2, "m", "b", "third"),
                   self.dec(3, "m", "a"), self.dec(4, "m", None, "drop")]
        return cands_rows, decided

    def run_assemble(self, dirs):
        lb.main(["assemble", "--set", "dev-cal", "--dirs", *dirs, "--dest", self.dest, "--allow-git"])
        rows = read_jsonl(os.path.join(self.dest, "dev-cal.jsonl"))
        with open(os.path.join(self.dest, "dev-cal.summary.json"), encoding="utf-8") as f:
            return rows, json.load(f)

    def test_drop_trap_relabel_and_order(self):
        # 함정 둘 : 대응표의 함정 q 가 decided 에 섞인 줄 · 함정 풀과 같은 글인 후보
        pc, pd = self.public(extra_dec=[self.dec(9, "p", "a"), self.dec(8, "p", "d")],
                             extra_cand=[self.cand(8, "p")])
        pd[-2]["q"] = "qtrap"
        pool = [{"evidence": "근거 p8", "claim": "주장 p8", "gold": "d", "tid": "t0001"}]
        pub = self.make_dir("eval-public", pc, pd, trap_map=[("qtrap", "t0000")], pool=pool)
        ac, ad = self.archive()
        arc = self.make_dir("eval-archive", ac, ad)
        rows, summary = self.run_assemble([pub, arc])
        self.assertEqual([r["id"] for r in rows], ["m00", "m01", "m02", "m03", "p01", "p02", "p03"])
        self.assertEqual(summary["status"], {"agree": 5, "third": 2})
        self.assertEqual(summary["want"], {"a": 2, "b": 2, "c": 2, "d": 1})
        self.assertEqual(summary["dirs"][0]["traps_in_decided"], 2)
        self.assertEqual(summary["dirs"][1]["relabel_dropped"], 1)
        self.assertEqual(summary["relabel_matrix"], {"b": {"b": 1}, "d": {"c": 1, "d": 1}})
        self.assertEqual(summary["overlap"], 0)
        first = rows[0]
        self.assertEqual(list(first), ["id", "evidence", "claim", "want", "label_guide", "domain", "group", "groups",
                                       "lang", "form", "hint", "source", "relabel", "old_want", "votes", "status",
                                       "set"])
        self.assertEqual(first["label_guide"], "support-v5")
        self.assertNotIn("q", first)
        self.assertNotIn("source", rows[4])
        self.assertFalse(rows[4]["relabel"])
        self.assertEqual(rows[4]["votes"], {"j1": "b", "j2": "b", "j3": "b"})
        # 같은 입력이면 같은 바이트 (결정적)
        with open(os.path.join(self.dest, "dev-cal.jsonl"), "rb") as f:
            once = f.read()
        self.run_assemble([pub, arc])
        with open(os.path.join(self.dest, "dev-cal.jsonl"), "rb") as f:
            self.assertEqual(once, f.read())
        with open(os.path.join(self.dest, "dev-cal.summary.json"), encoding="utf-8") as f:
            import hashlib
            self.assertEqual(json.load(f)["sha256"], hashlib.sha256(once).hexdigest())

    def test_overlap_stops(self):
        pc, pd = self.public()
        pub = self.make_dir("eval-public", pc, pd)
        ac, ad = self.archive()
        ac[3]["evidence"], ac[3]["claim"] = "근거 p3", "주장 p3"   # 공개판 p03 과 같은 글
        arc = self.make_dir("eval-archive", ac, ad)
        with self.assertRaises(SystemExit) as cm:
            self.run_assemble([pub, arc])
        self.assertEqual(cm.exception.code, 2)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "dev-cal.jsonl")))

    def test_dedupe_within_dir_only_on_flag(self):
        pc, pd = self.public(extra_dec=[self.dec(7, "p", "c")],
                             extra_cand=[self.cand(7, "p", evidence="근거 p2", claim="주장 p2")])   # p02 와 같은 글 · 같은 칸
        pub = self.make_dir("eval-public", pc, pd)
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([pub], "dev-cal")
        rows, summary = lb.assemble([pub], "dev-cal", dedupe=True)
        self.assertEqual([r["id"] for r in rows], ["p01", "p02", "p03"])
        self.assertEqual(summary["dirs"][0]["deduped"], ["p07"])
        pd[-1]["label"] = "d"   # 칸이 다르면 줄이지 않고 멈춘다
        pub2 = self.make_dir("eval-public2", pc, pd)
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([pub2], "dev-cal", dedupe=True)

    def test_cd_stops(self):
        pc, pd = self.public()
        pd[0]["label"] = "cd"
        pub = self.make_dir("eval-public", pc, pd)
        with self.assertRaises(SystemExit):
            self.run_assemble([pub])

    def test_need_third_stops(self):
        pc, pd = self.public()
        pd[3] = dict(pd[3], label=None, status="need-third")
        pub = self.make_dir("eval-public", pc, pd)
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([pub], "dev-cal")

    def test_refuses_git_dest(self):
        pc, pd = self.public()
        pub = self.make_dir("eval-public", pc, pd)
        with self.assertRaises(SystemExit):
            lb.main(["assemble", "--set", "dev-cal", "--dirs", pub, "--dest", os.path.join(helpers.ROOT, "x")])


class AnswerCheckTest(unittest.TestCase):
    def test_consistency_table(self):
        self.assertEqual(lb.check_answer(answer("a", "q1"), "eval"), "a")
        self.assertEqual(lb.check_answer(answer("b", "q1"), "eval"), "b")
        self.assertEqual(lb.check_answer(answer("d", "q1"), "eval"), "d")
        self.assertEqual(lb.check_answer(answer("cd", "q1"), "train"), "cd")
        self.assertIsNone(lb.check_answer(answer("cd", "q1"), "eval"))
        bad = {"q": "q1", "q1": "예", "q2": "아니오", "q3": "예", "label": "d"}
        self.assertIsNone(lb.check_answer(bad, "eval"))
        self.assertIsNone(lb.check_answer({"q": "q1", "label": "a"}, "eval"))

    def test_none_rate_drops_run(self):
        mapping = [{"q": f"q{i}", "batch": "b01", "id": f"x{i}", "trap": False} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b01-j1.jsonl")
            lines = [answer("a", f"q{i}") for i in range(18)]
            write_jsonl(path, lines)   # 2/20 = 10% 답 없음 > 5%
            runs, votes = lb.merge_runs(mapping, [path], "eval")
        self.assertFalse(runs[0]["kept"])
        self.assertTrue(runs[0]["reason"].startswith("답 없음"))
        self.assertEqual(len(votes), 20)
        self.assertTrue(all(v["votes"] == {} for v in votes))   # 버린 판의 쌍도 줄은 남아 missing 이 된다

    def _careful_out(self, root):
        """신중 판정 한 판 — 실제 19쌍 + 함정 t0046(gold b) 하나. 판정자는 함정에 d 라고 답한다."""
        out = os.path.join(root, "gold")
        os.makedirs(os.path.join(out, "map"))
        with open(os.path.join(out, "map", "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "eval"}, f)
        mapping = [{"q": f"q{i}", "batch": "b06", "id": f"x{i}", "trap": False} for i in range(19)]
        mapping.append({"q": "q19", "batch": "b06", "id": "t0046", "trap": True, "gold": "b"})
        write_jsonl(os.path.join(out, "map", "map.jsonl"), mapping)
        write_jsonl(os.path.join(out, "answers", "b06-j1.jsonl"),
                    [answer("d", f"q{i}") for i in range(20)])
        return out

    def _merge(self, out, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            lb.main(["merge", "--out", out, *extra])
        with open(os.path.join(out, "merge-report.json"), encoding="utf-8") as f:
            return json.load(f)["runs"][0], read_jsonl(os.path.join(out, "votes.jsonl"))

    def test_merge_without_exempt_drops_doubtful_trap_run(self):
        with tempfile.TemporaryDirectory() as root:
            run, _ = self._merge(self._careful_out(root))
        self.assertFalse(run["kept"])
        self.assertEqual(run["reason"], "함정 0/1")
        self.assertNotIn("trap_exempt", run)

    def test_merge_exempt_trap_keeps_run(self):
        with tempfile.TemporaryDirectory() as root:
            run, votes = self._merge(self._careful_out(root), "--exempt-trap", "t0046")
        self.assertTrue(run["kept"])
        self.assertIsNone(run["trap_acc"])   # 남은 함정 0 → 관문 「없음」
        self.assertEqual((run["trap_n"], run["trap_exempt"]), (0, ["t0046"]))
        self.assertEqual(len(votes), 19)     # 면제 함정도 라벨 줄에는 안 들어간다
        self.assertNotIn("t0046", {v["id"] for v in votes})
        self.assertTrue(all(v["votes"] == {"j1": "d"} for v in votes))


class DecideTest(unittest.TestCase):
    def check(self, votes, mode, want):
        self.assertEqual(lb.decide_one(votes, mode), want)

    def test_train_rules(self):
        self.check({"j1": "d", "j2": "d"}, "train", ("d", "agree"))
        self.check({"j1": "cd", "j2": "cd"}, "train", ("cd", "agree"))
        self.check({"j1": "c", "j2": "d"}, "train", (None, "need-third"))
        self.check({"j1": "c", "j2": "d", "j3": "d"}, "train", ("d", "third"))
        self.check({"j1": "a", "j2": "d", "j3": "a"}, "train", ("a", "third"))
        self.check({"j1": "c", "j2": "d", "j3": "cd"}, "train", ("cd", "cd-split"))
        self.check({"j1": "a", "j2": "c", "j3": "d"}, "train", (None, "drop"))
        self.check({"j1": "d"}, "train", (None, "missing"))

    def test_eval_rules(self):
        self.check({"j1": "b", "j2": "b"}, "eval", ("b", "agree"))
        self.check({"j1": "c", "j2": "d"}, "eval", (None, "need-third"))
        self.check({"j1": "c", "j2": "d", "j3": "c"}, "eval", ("c", "third"))
        self.check({"j1": "c", "j2": "d", "j3": "a"}, "eval", (None, "drop"))
        self.check({"j2": "a"}, "eval", (None, "missing"))


class KappaTest(unittest.TestCase):
    def test_hand_computed(self):
        # 10 쌍 : (c,c)×4 (d,d)×3 (c,d)×2 (d,c)×1 → 관측 0.7 · 우연 (6·5 + 4·5)/100 = 0.5 → κ 0.4
        pairs = [("c", "c")] * 4 + [("d", "d")] * 3 + [("c", "d")] * 2 + [("d", "c")]
        self.assertAlmostEqual(lb.cohen_kappa(pairs), 0.4)
        self.assertIsNone(lb.cohen_kappa([]))
        self.assertIsNone(lb.cohen_kappa([("c", "c")] * 3))

    def test_cd_kappa_uses_cd_pairs_only(self):
        rows = [{"votes": {"j1": x, "j2": y}} for x, y in
                [("c", "c")] * 4 + [("d", "d")] * 3 + [("c", "d")] * 2 + [("d", "c")] + [("a", "a")] * 5]
        k = lb.kappas(rows)
        self.assertEqual(k["cd_n"], 10)
        self.assertAlmostEqual(k["cd"], 0.4)
        self.assertEqual(k["all_n"], 15)
        self.assertGreater(k["all"], k["cd"])


class GoldTest(unittest.TestCase):
    def test_pick_hard_then_domain_random(self):
        decided, votes = [], {}
        for i in range(400):
            status = "third" if i < 90 else "agree"
            domain = ("mem", "vitaminc")[i % 2]
            decided.append({"q": f"q{i}", "id": f"x{i}", "label": "d", "status": status, "domain": domain})
            sure = {"j1": "낮음"} if 90 <= i < 110 else {}
            votes[f"q{i}"] = {"sure": sure, "reasons": {}}
        decided.append({"q": "qz", "id": "xz", "label": None, "status": "drop", "domain": "mem"})
        picked = lb.pick_gold(decided, votes, 120, 80, 1)
        why = collections.Counter(r["why"] for r in picked)
        self.assertEqual(len(picked), 200)
        self.assertEqual(why["split"] + why["weak"], 110)
        self.assertEqual(why["random"], 90)
        self.assertNotIn("xz", {r["id"] for r in picked})
        rand = collections.Counter(r["domain"] for r in picked if r["why"] == "random")
        self.assertEqual(rand["mem"], 45)

    def test_gold_agreement(self):
        dev = [{"id": "1", "label": "d", "status": "agree"}, {"id": "2", "label": "c", "status": "third"},
               {"id": "3", "label": "a", "status": "agree"}, {"id": "4", "label": "d", "status": "agree"}]
        gold = [{"id": "1", "label": "d"}, {"id": "2", "label": "d"}, {"id": "3", "label": "a"},
                {"id": "4", "label": "d"}]
        res = lb.gold_agreement(dev, gold)
        self.assertEqual(res["all"], 0.75)
        self.assertEqual(res["agree_only"], 1.0)
        self.assertAlmostEqual(res["by_label"]["d"], 2 / 3, places=3)
        self.assertFalse(res["pass"])

    def test_gold_gate_on_random_layer(self):
        dev = [{"id": str(i), "label": "a", "status": "agree"} for i in range(10)]
        gold = [{"id": str(i), "label": "a" if i < 5 or i == 9 else "c"} for i in range(10)]
        why = {str(i): "split" if i < 8 else "random" for i in range(10)}
        why["7"] = ""   # 까닭 모름은 층에 안 넣는다
        res = lb.gold_agreement(dev, gold, why)
        self.assertEqual(res["all"], 0.6)
        self.assertEqual(res["by_why"], {"split": {"n": 7, "all": round(5 / 7, 4)},
                                         "random": {"n": 2, "all": 0.5}})
        self.assertEqual(res["gate_on"], "random")
        self.assertFalse(res["pass"])
        why = dict(why, **{"8": "random", "9": "random"})
        gold[8] = {"id": "8", "label": "a"}
        res = lb.gold_agreement(dev, gold, why)
        self.assertTrue(res["pass"])           # 전체 0.7 이어도 무작위 층 1.0 으로 통과
        res = lb.gold_agreement(dev, gold)
        self.assertEqual((res["gate_on"], res["by_why"], res["pass"]), ("all", {}, False))

    def _two_dev_dirs(self, root):
        """공개판 · 아카이브판 묶음 폴더 둘 — q 가 일부러 겹친다 (q0 · q1)."""
        dirs = []
        for k, (name, sure) in enumerate((("pub", "낮음"), ("arc", ""))):
            d = os.path.join(root, name)
            os.makedirs(d)
            write_jsonl(os.path.join(d, "decided.jsonl"),
                        [{"q": f"q{i}", "id": f"{name}{i}", "label": "c", "status": "agree", "domain": name}
                         for i in range(5)])
            write_jsonl(os.path.join(d, "votes.jsonl"),
                        [{"q": f"q{i}", "sure": {"j1": sure} if (sure and i == 0) else {}, "reasons": {}}
                         for i in range(5)])
            dirs.append(d)
        return dirs

    def test_load_dev_dirs_keeps_q_apart(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = self._two_dev_dirs(root)
            decided, votes = lb.load_dev_dirs(dirs, {"pub0", "pub1", "arc0", "arc1"})
            self.assertEqual(len(decided), 4)
            self.assertEqual(len(votes), 10)
            weak = [r["id"] for r in decided if lb.is_weak(votes[r["q"]])]
            self.assertEqual(weak, ["pub0"])   # 아카이브판 q0 의 표가 공개판 q0 을 덮지 않는다
            with self.assertRaises(SystemExit):
                lb.load_dev_dirs([dirs[0], dirs[0]])

    def test_gold_pick_and_score_two_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = self._two_dev_dirs(root)
            set_path = os.path.join(root, "dev-cmp.jsonl")
            write_jsonl(set_path, [{"id": f"{n}{i}", "evidence": "근거", "claim": "주장", "lang": "ko"}
                                   for n in ("pub", "arc") for i in range(4)])
            dest = os.path.join(root, "gold")
            os.makedirs(dest)
            lb.main(["gold-pick", "--out", *dirs, "--set", set_path, "--dest", dest,
                     "--n-hard", "1", "--n-random", "3"])
            rows = read_jsonl(os.path.join(dest, "gold-cands.jsonl"))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["id"] for r in rows if r["why"] == "weak"], ["pub0"])
            # 무작위 3 은 남은 7쌍(pub 3 · arc 4) 비례 — 1.29 · 1.71 → 1 · 2
            self.assertEqual(collections.Counter(r["domain"] for r in rows if r["why"] == "random"),
                             collections.Counter({"pub": 1, "arc": 2}))
            self.assertTrue(all(r["id"][-1] != "4" for r in rows))   # 셋에 없는 쌍은 안 뽑는다
            write_jsonl(os.path.join(dest, "decided.jsonl"),
                        [{"id": "pub0", "label": "c"}, {"id": "arc1", "label": "a"}])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lb.main(["gold-score", "--out", *dirs, "--gold", dest])
            res = json.loads(buf.getvalue().splitlines()[0])
            self.assertEqual(res["n"], 2)
            self.assertEqual(res["all"], 0.5)

    def test_review_order_and_train_sample(self):
        decided = [{"q": f"q{i}", "id": f"x{i}", "label": "d", "status": "third"} for i in range(80)]
        decided.append({"q": "qd", "id": "xd", "label": None, "status": "drop"})
        decided.append({"q": "qg", "id": "xg", "label": "c", "status": "agree"})
        decided.append({"q": "ql", "id": "xl", "label": "c", "status": "agree"})
        gold = {"xg": {"label": "d", "_votes": {}}, "xl": {"label": "c", "_votes": {"sure": {"j1": "낮음"}}}}
        rows = lb.review_rows(decided, {}, "train", gold, 50, 1)
        self.assertEqual((rows[0]["why"], rows[0]["gold"]), ("gold-low", "c"))   # 「낮음」 줄에도 gold 칸을 싣는다
        self.assertEqual(rows[1]["why"], "gold-diff")
        self.assertEqual(rows[2]["why"], "drop")
        self.assertEqual(sum(1 for r in rows if r["why"] == "third"), 50)
        rows = lb.review_rows(decided, {}, "eval", gold, 50, 1)
        self.assertEqual(sum(1 for r in rows if r["why"] == "third"), 80)

    def test_review_todo_two_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = self._two_dev_dirs(root)
            for d, name in zip(dirs, ("pub", "arc")):
                os.makedirs(os.path.join(d, "map"))
                with open(os.path.join(d, "map", "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump({"mode": "eval"}, f)
                rows = read_jsonl(os.path.join(d, "decided.jsonl"))
                rows[0].update(label=None, status="drop", votes={"j1": "a", "j2": "c", "j3": "d"})  # 두 폴더 모두 q0 을 버림 — q 가 겹친다
                write_jsonl(os.path.join(d, "decided.jsonl"), rows)
            set_path = os.path.join(root, "dev-cmp.jsonl")
            write_jsonl(set_path, [{"id": f"{n}{i}", "evidence": f"근거 {n}{i}", "claim": "주장"}
                                   for n in ("pub", "arc") for i in range(5)])
            dest = os.path.join(root, "todo")
            os.makedirs(dest)
            with contextlib.redirect_stdout(io.StringIO()):
                lb.main(["review-todo", "--out", *dirs, "--set", set_path, "--dest", dest])
            todo = read_jsonl(os.path.join(dest, "human-review-todo.jsonl"))
        self.assertEqual(sorted(r["id"] for r in todo), ["arc0", "pub0"])
        self.assertEqual({r["evidence"] for r in todo}, {"근거 arc0", "근거 pub0"})
        self.assertTrue(all(r["why"] == "drop" for r in todo))


class TrapPoolTest(unittest.TestCase):
    def test_pool_rules(self):
        dev = [{"evidence": "e1", "claim": "c1", "kind": "same", "want": "support"},
               {"evidence": "e2", "claim": "c2", "kind": "near", "want": "support"},
               {"evidence": "e3", "claim": "c3", "kind": "flip", "want": "contradict"},
               {"evidence": "e4", "claim": "c4", "kind": "supersede", "want": "contradict"},
               {"evidence": "e5", "claim": "c5", "kind": "xscope", "want": "unrelated"},
               {"evidence": "e6", "claim": "c6", "kind": "sscope", "want": "unrelated"}]
        inputs = [{"q": "q1", "evidence": "e7", "claim": "c7"}, {"q": "q2", "evidence": "e8", "claim": "c8"}]
        gold = [{"q": "q1", "gold": "d", "src": "audit"}, {"q": "q2", "gold": "d", "src": "wice"}]
        pool = lb.build_trap_pool(dev, inputs, gold)
        self.assertEqual([(r["evidence"], r["gold"]) for r in pool],
                         [("e1", "a"), ("e4", "b"), ("e5", "c"), ("e7", "d")])


class RulersTest(unittest.TestCase):
    def test_claim_only_catches_artifact(self):
        rng = random.Random(3)
        synth = []
        for i in range(300):
            is_d = i % 2 == 0
            word = "아마도 그런 편" if is_d else "확정 사실"
            synth.append({"claim": f"{word} {rng.randrange(1000)}", "want": "d" if is_d else "a", "group": f"g{i}"})
        real = [{"claim": f"평범한 글 {rng.randrange(1000)}", "want": "d" if i % 2 == 0 else "a"} for i in range(100)]
        res = rulers.claim_only(synth, real, "want")
        self.assertGreater(res["acc_valid"], 0.95)
        self.assertLess(res["acc_real"], 0.7)
        self.assertFalse(res["pass"])
        same = rulers.claim_only(synth, synth[:100], "want")
        self.assertTrue(same["pass"])

    def test_repeat_rate_and_diversity(self):
        self.assertAlmostEqual(rulers.repeat_rate(["aaaa"]), 0.5)   # 3묶음 aaa · aaa → 서로 다른 것 1/2
        same = ["같은 말 같은 말"] * 20
        rng = random.Random(5)
        varied = ["".join(chr(0xAC00 + rng.randrange(11172)) for _ in range(12)) for _ in range(20)]
        self.assertGreater(rulers.repeat_rate(same), rulers.repeat_rate(varied))
        synth = [{"claim": t, "want": "d"} for t in same]
        real = [{"claim": t, "want": "d"} for t in varied]
        res = rulers.diversity(synth, real, "want", sample=20)
        self.assertFalse(res["pass"])
        self.assertGreater(res["labels"]["d"]["repeat_ratio"], 1.5)
        self.assertTrue(rulers.diversity(real, real, "want", sample=20)["pass"])

    def test_source_needs_diagnostic_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.jsonl")
            write_jsonl(path, [{"evidence": "e", "claim": "c", "want": "d"}])
            with self.assertRaises(SystemExit):
                rulers.main(["source", "--synth", path, "--real", path])


class TeacherTest(unittest.TestCase):
    def rows(self):
        out = []
        for i in range(10):
            out.append({"id": f"a{i}", "hint": "a", "domain": "p", "form": "same-sec"})
            out.append({"id": f"b{i}", "hint": "b", "domain": "p", "form": "rejected"})
        out.append({"id": "d-hiA", "hint": "d", "domain": "p", "form": "link"})
        out.append({"id": "d-hiB", "hint": "d", "domain": "p", "form": "link"})
        out.append({"id": "d-low", "hint": "d", "domain": "p", "form": "link"})
        out.append({"id": "d-q", "hint": "d", "domain": "q", "form": "link"})   # q 원천엔 a·b 후보 없음
        for i in range(10):
            out.append({"id": f"n{i}", "hint": "d", "domain": "p", "form": "near"})
        return out

    def scores(self):
        s = {}
        for i in range(10):
            s[f"a{i}"] = {"a": 0.80 + i * 0.01, "b": 0.05, "c": 0.1, "d": 0.05}   # 중앙값 0.845
            s[f"b{i}"] = {"a": 0.05, "b": 0.60, "c": 0.2, "d": 0.15}               # 중앙값 0.60
        s["d-hiA"] = {"a": 0.81, "b": 0.0, "c": 0.09, "d": 0.10}   # 0.81 ≥ 0.845×0.95=0.80275 → a
        s["d-hiB"] = {"a": 0.0, "b": 0.58, "c": 0.12, "d": 0.30}   # 0.58 ≥ 0.57 → b
        s["d-low"] = {"a": 0.79, "b": 0.0, "c": 0.01, "d": 0.20}   # 0.79 < 0.80275 → d 그대로
        s["d-q"] = {"a": 0.99, "b": 0.0, "c": 0.0, "d": 0.01}
        for i in range(10):
            s[f"n{i}"] = {"a": 0.0, "b": 0.0, "c": i / 10, "d": 1 - i / 10}
        return s

    def test_filter_rules(self):
        kept, dropped = teacher.filter_rows(self.rows(), self.scores())
        by_id = {r["id"]: r for r in kept}
        self.assertEqual((by_id["d-hiA"]["hint"], by_id["d-hiA"]["teacher_rule"]), ("a", "d→a"))
        self.assertEqual(by_id["d-hiB"]["hint"], "b")
        self.assertEqual(by_id["d-low"]["hint"], "d")
        self.assertEqual(by_id["d-q"]["hint"], "d")
        rules = {r["id"]: r["teacher_rule"] for r in dropped}
        self.assertEqual(rules, {"n0": "near-top", "n9": "near-low"})
        self.assertEqual(sum(1 for r in kept if r["id"].startswith("n")), 8)
        self.assertEqual(len(kept) + len(dropped), len(self.rows()))

    def test_filter_floor(self):
        # 중앙값이 무너진 원천 : a 후보 p_a 가 다 낮으면 상대 문턱만으론 거의 모든 d 가 a 로 간다
        rows = [{"id": f"a{i}", "hint": "a", "domain": "k", "form": "same-doc"} for i in range(5)]
        rows += [{"id": "d-mid", "hint": "d", "domain": "k", "form": "removed"},
                 {"id": "d-sure", "hint": "d", "domain": "k", "form": "removed"}]
        scores = {f"a{i}": {"a": 0.01, "b": 0.0, "c": 0.9, "d": 0.09} for i in range(5)}
        scores["d-mid"] = {"a": 0.30, "b": 0.0, "c": 0.4, "d": 0.30}    # 중앙값 넘지만 바닥 0.5 아래
        scores["d-sure"] = {"a": 0.70, "b": 0.0, "c": 0.1, "d": 0.20}
        kept, _ = teacher.filter_rows(rows, scores)
        by_id = {r["id"]: r for r in kept}
        self.assertEqual(by_id["d-mid"]["hint"], "d")
        self.assertNotIn("teacher_rule", by_id["d-mid"])
        self.assertEqual(by_id["d-sure"]["hint"], "a")
        kept0, _ = teacher.filter_rows(rows, scores, floor=0.0)    # 바닥 끄면 옛 규칙
        self.assertEqual({r["id"]: r["hint"] for r in kept0}["d-mid"], "a")

    def test_filter_floor_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"id": "a0", "hint": "a", "domain": "k", "form": "x"},
                    {"id": "d0", "hint": "d", "domain": "k", "form": "x"}]
            sc = [{"id": "a0", "a": 0.02, "b": 0.0, "c": 0.9, "d": 0.08},
                  {"id": "d0", "a": 0.40, "b": 0.0, "c": 0.3, "d": 0.30}]
            cp, sp, op = (os.path.join(tmp, n) for n in ("c.jsonl", "s.jsonl", "o.jsonl"))
            write_jsonl(cp, rows)
            write_jsonl(sp, sc)
            for floor, want in (("0.5", "d"), ("0.3", "a")):
                with contextlib.redirect_stdout(io.StringIO()):
                    teacher.main(["filter", "--cands", cp, "--scores", sp, "--out", op, "--floor", floor])
                got = {r["id"]: r["hint"] for r in read_jsonl(op)}
                self.assertEqual(got["d0"], want, floor)

    def test_score_with_mock_server(self):
        calls = []

        def post(body):
            calls.append(body)
            return {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "model": "mock", "ms": 1.0, "label_set": "nli4"}
        cands_rows = [{"id": "x1", "evidence": "e", "claim": "c"}, {"id": "x2", "evidence": "e2", "claim": "c2"}]
        out = list(teacher.score_rows(cands_rows, post, done=["x1"]))
        self.assertEqual(calls, [{"evidence": "e2", "claim": "c2", "label_set": "nli4"}])
        self.assertEqual(out, [{"id": "x2", "a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "model": "mock"}])
        with self.assertRaises(RuntimeError):
            list(teacher.score_rows(cands_rows, lambda body: {"a": 0.5, "b": 0.3, "c": 0.2}))
        # --compat3 서버 꼴 — nli4 를 청해도 3칸 + label_set nli3 으로 답하면 멈춘다
        with self.assertRaises(RuntimeError):
            list(teacher.score_rows(cands_rows, lambda body: {"a": 0.5, "b": 0.3, "c": 0.2, "label_set": "nli3"}))

    def test_score_request_body_has_label_set(self):
        # urllib 를 가짜로 바꿔 score 가 실제로 보내는 요청 본문을 본다
        sent = []

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            sent.append((req.full_url, json.loads(req.data.decode("utf-8"))))
            reply = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "model": "m", "ms": 1.0, "label_set": "nli4"}
            return FakeResp(json.dumps(reply).encode("utf-8"))
        with tempfile.TemporaryDirectory() as d:
            cp, op = os.path.join(d, "cands.jsonl"), os.path.join(d, "scores.jsonl")
            with open(cp, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x1", "evidence": "근거", "claim": "주장"}, ensure_ascii=False) + "\n")
            real = teacher.urllib.request.urlopen
            teacher.urllib.request.urlopen = fake_urlopen
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    teacher.main(["score", "--cands", cp, "--out", op, "--url", "http://127.0.0.1:9/"])
            finally:
                teacher.urllib.request.urlopen = real
        self.assertEqual(sent, [("http://127.0.0.1:9/judge", {"evidence": "근거", "claim": "주장", "label_set": "nli4"})])


class AssembleTrainTest(unittest.TestCase):
    """학습 셋 조립(--mode train) — cd 를 넣고, 빈 쌍 · drop · 폴더 사이 겹침은 eval 과 같게 다룬다."""

    def setUp(self):
        self.t = AssembleTest()
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def train_dir(self, kind="train-II"):
        t = self.t
        pc, pd = t.public(extra_dec=[t.dec(5, "p", "cd"), t.dec(6, "p", "cd", "cd-split"), t.dec(7, "p", "a")],
                          extra_cand=[t.cand(5, "p"), t.cand(6, "p"), t.cand(7, "p", claim="  ")])
        for c in pc:
            c.pop("set")   # 학습 후보에는 set 칸이 없다
        pc[1]["set"] = "다른셋"   # 있어도 train 은 안 본다
        return t.make_dir(kind, pc, pd), pc, pd

    def run_train(self, dirs, name="train-g1-II"):
        lb.main(["assemble", "--mode", "train", "--set", name, "--dirs", *dirs, "--dest", self.t.dest, "--allow-git"])
        rows = read_jsonl(os.path.join(self.t.dest, f"{name}.jsonl"))
        with open(os.path.join(self.t.dest, f"{name}.summary.json"), encoding="utf-8") as f:
            return rows, json.load(f)

    def test_train_keeps_cd_and_drops_empty(self):
        d, _, _ = self.train_dir()
        with contextlib.redirect_stdout(io.StringIO()):
            rows, summary = self.run_train([d])
        self.assertEqual([r["id"] for r in rows], ["p01", "p02", "p03", "p05", "p06"])
        self.assertEqual(summary["mode"], "train")
        self.assertEqual(summary["want"], {"a": 1, "b": 1, "c": 1, "cd": 2})
        self.assertEqual(summary["cd"], 2)
        self.assertEqual(summary["dropped_empty"], 1)
        self.assertEqual(summary["status"], {"agree": 3, "cd-split": 1, "third": 1})
        self.assertEqual(summary["dirs"][0]["want"], {"a": 1, "b": 1, "c": 1, "cd": 2})
        cd_row = rows[4]
        self.assertEqual((cd_row["want"], cd_row["status"], cd_row["set"]), ("cd", "cd-split", "train-g1-II"))
        self.assertEqual(cd_row["label_guide"], "support-v5")
        self.assertEqual((cd_row["domain"], cd_row["origin"], cd_row["lic"]), ("vitaminc", "o", "x"))
        # 학습기가 읽는 꼴 — data_prep.normalize 가 want cd 를 CD 로 읽는다
        sys.path.insert(0, os.path.join(helpers.ROOT, "train"))
        try:
            import data_prep
        finally:
            sys.path.pop(0)
        self.assertEqual(data_prep.normalize(cd_row)["label"], data_prep.CD)
        self.assertEqual(data_prep.normalize(rows[0])["label"], 1)

    def test_eval_stops_on_same_cd_dir(self):
        d, _, _ = self.train_dir()
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([d], "dev-cal")

    def test_eval_rows_keep_old_shape(self):
        pc, pd = self.t.public()
        pub = self.t.make_dir("eval-public", pc, pd)
        rows, summary = lb.assemble([pub], "dev-cal")
        self.assertNotIn("origin", rows[0])
        self.assertNotIn("mode", summary)

    def test_train_overlap_between_dirs_stops(self):
        d1, _, _ = self.train_dir("train-II")
        t = self.t
        pc = [t.cand(0, "k"), t.cand(1, "k", evidence="근거 p2", claim="주장 p2")]   # train-II p02 와 같은 글
        pd = [t.dec(0, "k", "a"), t.dec(1, "k", "c")]
        d2 = t.make_dir("train-III", pc, pd)
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([d1, d2], "train-g1-II", mode="train")
        with self.assertRaises(lb.AssembleStop):   # --dedupe-same 은 폴더 안만 줄인다
            lb.assemble([d1, d2], "train-g1-II", dedupe=True, mode="train")

    def test_train_need_third_stops(self):
        d, pc, pd = self.train_dir()
        pd[3] = dict(pd[3], label=None, status="need-third")
        d2 = self.t.make_dir("train-IV", pc, pd)
        with self.assertRaises(lb.AssembleStop):
            lb.assemble([d2], "train-g1-II", mode="train")


class EmptyPairTest(unittest.TestCase):
    """빈 근거 · 주장 막기 — make 는 멈추고 assemble 은 빼고 센다 (46 KLUE 빈 주장)."""

    def test_make_stops_on_empty_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = cands(5)
            rows[2]["claim"] = ""
            cpath, gpath, out = (os.path.join(tmp, n) for n in ("c.jsonl", "guide.md", "lab"))
            write_jsonl(cpath, rows)
            with open(gpath, "w", encoding="utf-8") as f:
                f.write(GUIDE)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                lb.main(["make", "--cands", cpath, "--out", out, "--mode", "train", "--guide", gpath, "--allow-git"])
            self.assertEqual(cm.exception.code, 2)
            self.assertFalse(os.path.exists(os.path.join(out, "map", "map.jsonl")))

    def test_assemble_drops_and_counts_empty(self):
        t = AssembleTest()
        t.setUp()
        try:
            pc, pd = t.public(extra_dec=[t.dec(7, "p", "a"), t.dec(8, "p", "d")],
                              extra_cand=[t.cand(7, "p", claim=""), t.cand(8, "p", evidence="  ")])
            pub = t.make_dir("eval-public", pc, pd)
            rows, summary = t.run_assemble([pub])
        finally:
            t.tearDown()
        self.assertEqual([r["id"] for r in rows], ["p01", "p02", "p03"])
        self.assertEqual(summary["dropped_empty"], 2)
        self.assertEqual(summary["dirs"][0]["dropped_empty"], 2)


if __name__ == "__main__":
    unittest.main()
