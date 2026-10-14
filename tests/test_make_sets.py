"""make_sets reserve-g3 — 묶음째 예약 · MIT 셋 차례 · 몫 넘김 · 금지 목록 · 문단 겹침 · 빠진 금지 파일. 입력은 모두 지어낸 짧은 글이다."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import make_sets  # noqa: E402


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row(rid, group, label, evidence, claim, **kw):
    base = {"id": rid, "evidence": evidence, "claim": claim, "hint": label, "group": group, "groups": [group],
            "lang": "ko", "src_ids": [rid + "-src"]}
    return dict(base, **kw)


def make_root(tmp):
    """옛 셋 · g1 · g2 · use1 · 잣대 · 씨앗 금지 파일을 다 갖춘 가짜 뿌리 → (old_root, g3 build)."""
    old = os.path.join(tmp, "jm")
    for rel in make_sets.OLD_SETS.values():
        write_jsonl(os.path.join(old, rel), [])
    old_klue = {"evidence": "옛 평가 지문의 문장은 서울에서 열린 대회가 처음이었다.", "claim": "다른 주장 문장.",
                "domain": "klue-mrc", "lang": "ko", "group": "klue:old", "groups": ["klue:old"]}
    for name in make_sets.G1_SET_FILES + make_sets.G1_TRAIN_FILES:
        write_jsonl(os.path.join(old, "sets", "g1", name + ".jsonl"), [old_klue] if name == "dev-cmp" else [])
    for name in make_sets.G2_SET_FILES:
        write_jsonl(os.path.join(old, "sets", "g2", name + ".jsonl"), [])
    with open(os.path.join(old, "sets", "g2", "blind-g2.forbid.json"), "w", encoding="utf-8") as f:
        json.dump({"set": "blind-g2", "groups": ["kowiki:g2used"], "pair_keys": [], "src_ids": []}, f)
    os.makedirs(os.path.join(old, "measure", "use1"))
    open(os.path.join(old, "measure", "use1", "EXCLUDE.sha256list"), "w").close()
    os.makedirs(os.path.join(old, "build", "g2", "reserve"))
    open(os.path.join(old, "build", "g2", "forbid-ruler-ids.txt"), "w").close()
    with open(os.path.join(old, "build", "g2", "reserve", "reserve-g2.json"), "w", encoding="utf-8") as f:
        json.dump({"reserved": ["vitc:used"]}, f)
    g3 = os.path.join(tmp, "g3")
    os.makedirs(os.path.join(g3, "seed"))
    with open(os.path.join(g3, "seed", "forbid-g3.txt"), "w", encoding="utf-8") as f:
        f.write("# 씨앗 금지\ngroups\tkowiki:seeded\n")
    wiki = []
    for i in range(1, 9):
        wiki.append(row(f"w{i}", f"kowiki:{i}", "b", f"문서{i} : 위키 문장 {i}번은 2010년에 {i}0만 명이 찾았다고 적었다.",
                        f"위키 문장 {i}번은 2010년에 {i}1만 명이 찾았다고 적었다.", title=f"문서{i}"))
    wiki.append(row("w9", "kowiki:9", "d", "문서9 : 아홉째 문장은 때가 다른 고침이라 d 이다.", "다른 때.", title="문서9"))
    wiki.append(row("wseed", "kowiki:seeded", "b", "씨앗 문서의 문장은 평가에 못 간다고 정했다.", "씨앗 주장."))
    wiki.append(row("wg2", "kowiki:g2used", "b", "g2 금지 묶음의 문장은 평가에 못 간다고 정했다.", "g2 주장."))
    wiki.append(row("wold", "kowiki:old", "b", "옛문서 : 옛 평가 지문의 문장은 서울에서 열린 대회가 처음이었다.", "옛 주장.",
                    title="옛문서"))
    write_jsonl(os.path.join(g3, "kowiki", "wiki-labeled-eval.jsonl"), wiki)
    write_jsonl(os.path.join(g3, "kowiki", "wiki-labeled-train.jsonl"), [
        row("t1", "kowiki:t1", "b", "학습 문서 문장은 1999년에 세 번째로 열렸다고 적었다.", "학습 주장."),
        row("t2", "kowiki:t2", "a", "문서9 : 아홉째 문장은 때가 다른 고침이라 d 이다.", "겹치는 학습 줄.", title="문서9")])
    mit = [row("m1", "doc:r:a.md", "b", "MIT 고친 뒤 문장은 정렬이 3단계라고 적었다.", "MIT 고치기 전 문장은 4단계.",
               judge={"label": "b"}),
           row("m2", "doc:r:a.md", "a", "MIT 같은 문서의 오타 고침 문장 하나.", "MIT 같은 문서의 오타 고침 문장 하나..",
               judge={"label": "a"}),
           row("m3", "doc:r:a.md", "a", "MIT 같은 문서의 둘째 오타 고침 문장.", "MIT 같은 문서의 둘째 오타 고침 문장..",
               judge={"label": "a"}),
           row("m4", "doc:r:b.md", "a", "MIT 다른 문서의 a 줄은 학습 층으로 간다.", "MIT 다른 문서 a 줄.",
               judge={"label": "a"})]
    write_jsonl(os.path.join(g3, "mit", "judged.jsonl"), mit)
    write_jsonl(os.path.join(tmp, "vitc.jsonl"), [
        row(f"v{i}", f"vitc:p{i % 3}", "b", f"English evidence {i}.", f"English claim {i}.", lang="en") for i in range(6)]
        + [row("vused", "vitc:used", "b", "Used page.", "Used claim.", lang="en")])
    return old, g3


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = make_sets.main(argv)
    return code, out.getvalue()


class ReserveG3Test(unittest.TestCase):
    TARGETS = {"dev-cmp-g3": {"mit": 1, "wiki": 2}, "dev-cal-g3": {"wiki": 1},
               "blind-g3": {"mit": 1, "wiki": 2, "vitaminc": 2}}

    def test_reserve_whole_groups_shift_and_forbid(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, g3 = make_root(tmp)
            tpath = os.path.join(tmp, "targets.json")
            with open(tpath, "w", encoding="utf-8") as f:
                json.dump(self.TARGETS, f)
            argv = ["reserve-g3", "--build", g3, "--old-root", old, "--targets", tpath,
                    "--vitaminc", os.path.join(tmp, "vitc.jsonl")]
            code, _ = run(argv)
            self.assertEqual(code, 0)
            out = os.path.join(g3, "reserve")
            with open(os.path.join(out, "reserve-g3.json"), encoding="utf-8") as f:
                doc = json.load(f)
            sets = {s: make_sets.read_jsonl(os.path.join(out, f"{s}-cands.jsonl")) for s in make_sets.G3_SETS}
            ids = {s: {r["id"] for r in rows} for s, rows in sets.items()}
            # MIT 는 셋 차례대로 : b 묶음째 dev-cmp-g3 로, a 는 b 수만큼만. blind 몫 1 은 위키로 넘어간다
            self.assertEqual({"m1", "m2"} & ids["dev-cmp-g3"], {"m1", "m2"})
            self.assertNotIn("m3", set().union(*ids.values()))
            self.assertEqual(doc["shift"]["blind-g3"]["mit->wiki"], 1)
            self.assertEqual(doc["sets"]["blind-g3"]["wiki"]["b_rows"], 3)
            # 금지 : 씨앗 묶음 · blind-g2 금지 묶음 · 옛 KLUE 지문과 같은 문장 · g2 예약 VitaminC 묶음
            every = set().union(*ids.values())
            self.assertFalse({"wseed", "wg2", "wold", "vused"} & every)
            self.assertEqual(doc["dropped_eval"]["wiki"], {"seed_g3": 1, "blind_g2_forbid": 1, "para_old": 1})
            self.assertEqual(doc["dropped_eval"]["vitaminc"].get("g2_reserved"), 1)
            # 묶음은 한 셋에만, 영어 몫은 blind-g3 에만, 없는 원천(BoolQ · KLUE)은 몫 0 · 넘김은 대기
            groups = [g for rows in sets.values() for g in {r["group"] for r in rows}]
            self.assertEqual(len(groups), len(set(groups)))
            self.assertTrue(all(r["lang"] == "en" for r in sets["blind-g3"] if r["g3_source"] == "vitaminc"))
            self.assertGreater(sum(1 for r in sets["blind-g3"] if r["g3_source"] == "vitaminc"), 0)
            self.assertTrue(doc["sets"]["dev-cmp-g3"]["boolq"]["input_missing"])
            self.assertEqual(doc["pending"], {})
            self.assertEqual(sum(doc["check"].values()), 0)
            # 학습 층 : 예약 안 된 MIT 묶음 · 위키 학습 몫. 평가 문장과 겹친 학습 줄은 뺀다
            train = {r["id"] for r in make_sets.read_jsonl(os.path.join(out, "train-layer-cands.jsonl"))}
            self.assertIn("m4", train)
            self.assertIn("t1", train)
            self.assertNotIn("t2", train)
            self.assertFalse(train & every)
            with open(os.path.join(out, "blind-g3.forbid.json"), encoding="utf-8") as f:
                forbid = json.load(f)
            self.assertEqual(sorted(forbid["cand_ids"]), sorted(ids["blind-g3"]))
            self.assertEqual(len(forbid["pair_keys"]), len(ids["blind-g3"]))
            # 같은 씨앗이면 같은 결과
            run(argv)
            again = {s: [r["id"] for r in make_sets.read_jsonl(os.path.join(out, f"{s}-cands.jsonl"))]
                     for s in make_sets.G3_SETS}
            self.assertEqual(again, {s: [r["id"] for r in rows] for s, rows in sets.items()})

    def test_boolq_added_klue_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, g3 = make_root(tmp)
            boolq = [row(f"q{i}", f"boolq:p{i}", "b", f"불큐 문단 {i}번은 1990년에 {i}개 학교가 문을 열었다고 적었다.",
                         f"불큐 문단 {i}번은 {i}개 학교가 1980년에 문을 열었다.", judge={"final": "b"}, keep=True,
                         claim_question=f"불큐 {i}번 학교는 1980년에 열었나?") for i in range(1, 5)]
            boolq += [row("q5", "boolq:p5", "b", "불큐 다섯째 문단은 a 로 판정된 줄 하나만 가진다고 적었다.", "a 주장.",
                          judge={"final": "a"}, keep=True),
                      row("q6", "boolq:p6", "b", "불큐 여섯째 문단은 d 로 판정된 줄 하나만 가진다고 적었다.", "d 주장.",
                          judge={"final": "d"}, keep=True),
                      row("q7", "boolq:p7", "b", "불큐 일곱째 문단은 두 판정이 갈려 남기지 않는다고 적었다.", "갈린 주장.",
                          judge={"final": "disagree"}, keep=False)]
            write_jsonl(os.path.join(g3, "boolq", "judged-eval.jsonl"), boolq)
            tpath = os.path.join(tmp, "targets.json")
            with open(tpath, "w", encoding="utf-8") as f:
                json.dump({"dev-cmp-g3": {"klue": 2, "boolq": 1}, "dev-cal-g3": {}, "blind-g3": {"boolq": 1}}, f)
            code, _ = run(["reserve-g3", "--build", g3, "--old-root", old, "--targets", tpath,
                           "--vitaminc", os.path.join(tmp, "vitc.jsonl")])
            self.assertEqual(code, 0)
            out = os.path.join(g3, "reserve")
            with open(os.path.join(out, "reserve-g3.json"), encoding="utf-8") as f:
                doc = json.load(f)
            # KLUE 는 파일이 없어 넘기지 않고 대기, BoolQ 는 제 몫만
            self.assertEqual(doc["pending"], {"dev-cmp-g3": {"klue": 2}})
            self.assertEqual(doc["shift"], {})
            self.assertEqual(doc["sets"]["dev-cmp-g3"]["boolq"]["b_rows"], 1)
            self.assertEqual(doc["sets"]["blind-g3"]["boolq"]["b_rows"], 1)
            self.assertEqual(doc["dropped_eval"]["boolq"], {"keep_false": 1})
            rows = [r for s in make_sets.G3_SETS for r in make_sets.read_jsonl(os.path.join(out, f"{s}-cands.jsonl"))]
            ids = {r["id"] for r in rows}
            self.assertNotIn("q7", ids)
            self.assertTrue({"q5", "q6"} <= ids)     # b 없는 문단(a · d)도 셋에 간다
            self.assertEqual({r["g3_label"] for r in rows if r["id"] in ("q5", "q6")}, {"a", "d"})
            self.assertEqual(sum(doc["check"].values()), 0)

    def test_missing_forbid_file_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, g3 = make_root(tmp)
            os.remove(os.path.join(g3, "seed", "forbid-g3.txt"))
            code, text = run(["reserve-g3", "--build", g3, "--old-root", old])
            self.assertEqual(code, 2)
            self.assertIn("forbid-g3.txt", text)
            self.assertFalse(os.path.exists(os.path.join(g3, "reserve")))

    def test_para_index_trigram_similarity(self):
        index = make_sets.ParaIndex()
        index.add({"evidence": "이 문장은 글자 3묶음 닮음 검사를 위한 꽤 긴 한국어 문장이다."}, "x")
        self.assertEqual(index.hits({"evidence": "이 문장은 글자 3묶음 닮음 검사를 위한 꽤 긴 한국어 문장이다!"}), {"x"})
        self.assertEqual(index.hits({"evidence": "전혀 다른 내용의 문장은 겹침으로 잡히면 안 된다고 본다."}), set())
        self.assertEqual(make_sets.g3_label({"hint": "b", "want": "", "judge": {"label": "a"}}), "a")

    def test_g3_label_reads_override_before_final(self):
        for final in ("a", "b", "c", "d"):
            judge = {"final": final, "override": {"from": final, "to": "b" if final != "b" else "a"}}
            self.assertEqual(make_sets.g3_label({"hint": final, "want": judge["override"]["to"], "judge": judge}),
                             judge["override"]["to"])
        self.assertEqual(make_sets.g3_label({"judge": {"final": "c", "override": {"to": "x"}}}), "c")   # 칸 밖 override 는 무시


if __name__ == "__main__":
    unittest.main()
