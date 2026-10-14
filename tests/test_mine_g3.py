"""v1.2 K 판(g3) 후보 뽑기 — BoolQ · KLUE 답 있는 질문 · MIT 저장소 고침. 입력은 모두 지어낸 짧은 글이다."""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import mine_archive as ma  # noqa: E402
import mine_public as mp  # noqa: E402


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


class BoolqTest(unittest.TestCase):
    PARA = "타이완 요리는 대체로 담백하고 기름을 적게 쓰는 편이다. 해산물을 많이 쓴다. 차와 함께 먹는다."

    def raw(self, folder):
        write(os.path.join(folder, "train.jsonl"), jsonl([
            {"paragraph": self.PARA, "question": "타이완 요리는 기름을 많이 사용하는 것이다.", "label": 0},
            {"paragraph": "연하곤란은 식도가 막혀 삼키기 어려운 증상이다. 나이 든 사람에게 흔하다.",
             "question": "연하곤란이 생기면 식도가 막히나요?", "label": 0},
            {"paragraph": "빌보드 핫 100 은 미국 음악 산업의 대표 싱글 차트이다. 매주 새로 나온다.",
             "question": "빌보드 핫 100은 음악 차트인가?", "label": 1},
        ]))
        write(os.path.join(folder, "validation.jsonl"), jsonl([
            {"paragraph": self.PARA, "question": "타이완 요리는 해산물을 많이 쓰는가?", "label": 1},   # train 과 같은 문단
        ]))
        write(os.path.join(folder, "test.jsonl"), jsonl([
            {"paragraph": "감자의 원산지는 남미 안데스 지역이다. 유럽에는 16세기에 들어왔다.",
             "question": "감자의 원산지는 유럽이다.", "label": 0},
        ]))

    def test_rows_hint_rewrite_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.raw(tmp)
            out = os.path.join(tmp, "out")
            summary = mp.run_kobest_boolq(tmp, out)
            train, dev = read_jsonl(os.path.join(out, "train.jsonl")), read_jsonl(os.path.join(out, "dev.jsonl"))
        self.assertEqual([r["hint"] for r in train], ["b", "b", "a"])                 # 0 → b · 1 → a
        self.assertEqual([r["needs_rewrite"] for r in train], [False, True, True])     # 평서문 · ~나요? · ~인가?
        self.assertEqual(train[0]["claim"], "타이완 요리는 기름을 많이 사용하는 것이다.")
        self.assertEqual((train[0]["lic"], train[0]["domain"], train[0]["split"]), ("CC-BY-SA-4.0", "kobest-boolq", "train"))
        self.assertEqual(dev[0]["group"], train[0]["group"])                            # 묶음 = 문단 해시
        self.assertEqual(dev[0]["near_train"], "exact")
        self.assertEqual(summary["overlap_train"]["dev"]["exact"], 1)
        self.assertEqual(summary["overlap_train"]["test"]["exact"] + summary["overlap_train"]["test"]["near"], 0)
        self.assertAlmostEqual(summary["by_split"]["train"]["declarative_share"], 0.333)

    def test_long_paragraph_keeps_question_sentences(self):
        filler = " ".join(f"관계없는 이야기 {i}번이 길게 이어지는 문장이다." for i in range(20))
        para = filler + " 타이완 요리는 기름을 적게 쓰고 담백하다." + " 끝에 붙는 다른 이야기도 하나 있다."
        short = mp.shorten_paragraph(para, "타이완 요리는 기름을 많이 쓰는가?")
        self.assertLessEqual(len(short), mp.MAX_LEN)
        self.assertIn("타이완 요리는 기름을 적게 쓰고 담백하다.", short)
        self.assertTrue(mp.is_asking("명예 훈장을 받은 사람이 있는가"))
        self.assertFalse(mp.is_asking("그는 어머니."))                                 # 「~니」 는 물음표 없으면 평서문

    def test_paragraph_near_overlap(self):
        base = ["가나다라마바사아자차카타파하 거너더러머버서어저처커터퍼허 고노도로모보소오조초코토포호"]
        other = [base[0] + "구", "전혀 다른 문단이 여기에 있습니다 하나 둘 셋"]
        self.assertEqual(mp.overlap_against(base, other, 0.8), {0: "near"})


class KlueAnsweredTest(unittest.TestCase):
    CONTEXT = ("삼성테크윈은 2014년 한화그룹에 팔렸다. 회사 이름은 2015년 한화테크윈으로 바뀌었다. "
               "공장 직원은 3200명이고 연구소 직원은 450명이다.")

    def raw(self, folder):
        qas = [
            {"question": "삼성테크윈이 한화그룹에 팔린 해는?", "answers": [{"text": "2014년", "answer_start": self.CONTEXT.index("2014년")}],
             "question_type": 1, "is_impossible": False, "guid": "klue-mrc-v1_train_00001"},
            {"question": "공장 직원은 몇 명인가?", "answers": [{"text": "3200명", "answer_start": self.CONTEXT.index("3200명")}],
             "question_type": 1, "is_impossible": False, "guid": "klue-mrc-v1_train_00002"},
            {"question": "회사가 한 일은 무엇인가?", "answers": [{"text": "이름 바꾸기", "answer_start": 0}],
             "question_type": 1, "is_impossible": False, "guid": "klue-mrc-v1_train_00003"},
        ]
        data = {"data": [{"title": "한화테크윈", "news_category": "경제", "source": "x",
                          "paragraphs": [{"context": self.CONTEXT, "qas": qas}]}]}
        write(os.path.join(folder, "klue-mrc-v1.1_train.json"), json.dumps(data, ensure_ascii=False))
        write(os.path.join(folder, "klue-mrc-v1.1_dev.json"), json.dumps({"data": []}))
        cands = [{"form": "qa-answer", "answer": "x", "split": "train", "src_ids": [q["guid"]]} for q in qas]
        write(os.path.join(folder, "cands.jsonl"), jsonl(cands))

    def test_alt_value_same_kind_and_pair_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.raw(tmp)
            rows, summary = mp.mine_klue_answered(tmp, os.path.join(tmp, "cands.jsonl"), [], random.Random(1))
        by = {(r["src_ids"][0][-1], r["hint"]): r for r in rows}
        self.assertEqual(len(rows), 4)                                                # 「무엇을 했나」 질문은 버림
        self.assertEqual(by[("1", "b")]["alt_answer"], "2015년")
        self.assertEqual(by[("1", "b")]["fill"], "2015년")
        self.assertEqual(by[("1", "a")]["fill"], "2014년")
        self.assertEqual(by[("2", "b")]["alt_answer"], "450명")
        self.assertEqual(by[("2", "b")]["ans_type"], "number")
        self.assertTrue(all(r["needs_rewrite"] and r["claim"] == r["question"] for r in rows))
        self.assertIn("2014년", by[("1", "b")]["evidence"])
        self.assertEqual(summary["counts"]["drop_kind"], 1)

    def test_forbid_drops_whole_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.raw(tmp)
            gid = "klue:" + mp.short_hash("한화테크윈")
            write(os.path.join(tmp, "forbid", "set.jsonl"), jsonl([{"group": gid}]))
            rows, summary = mp.mine_klue_answered(tmp, os.path.join(tmp, "cands.jsonl"), [os.path.join(tmp, "forbid")],
                                                  random.Random(1))
        self.assertEqual(rows, [])
        self.assertEqual(summary["counts"]["drop_forbid"], 3)

    def test_answer_kinds(self):
        ctx = "김한규 부회장과 이재수 씨가 왔다. 서울대 교수도 왔다. 5년마다 받고 2001년에 처음 받았다. 지난해 14위에서 6위로 올랐다."
        self.assertEqual(mp.answer_kind("누가 왔나?", "김한규", ctx), ("person", ["김한규", "이재수"]))
        self.assertEqual(mp.answer_kind("주기는?", "5년", ctx)[1], ["5년"])           # 기간은 해(2001년)를 안 고른다
        self.assertEqual(mp.answer_kind("목표 순위는?", "6위", ctx), ("ordinal", ["14위", "6위"]))
        self.assertEqual(mp.pick_alt(["6위"], ["14위", "6위"], random.Random(1)), "14위")
        self.assertEqual(mp.klue_sentences("말했다.IMF 등은 3.3㎡당 1800만원이다."), ["말했다.", "IMF 등은 3.3㎡당 1800만원이다."])


def git_commit(repo, files, message):
    for name, text in files.items():
        write(os.path.join(repo, name), text)
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "-q", "-m", message],
                   check=True, capture_output=True)


class MitRevisionTest(unittest.TestCase):
    def test_outside_repo_name_domain_lic(self):
        with tempfile.TemporaryDirectory() as repo:
            subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True)
            doc = "# OAuth\n\nOAuth 는 Open Authentification 의 줄임말이다.\n\n전역 함수는 여러 곳에서 같이 쓴다고 정리한다.\n"
            git_commit(repo, {"Network/oauth.md": doc}, "정리")
            git_commit(repo, {"Network/oauth.md": doc.replace("Authentification", "Authorization")
                              .replace("전역 함수", "전역 변수")}, "잘못된 설명 고침")
            col = ma.mine_revision([("gyoogle", repo)], random.Random(1), "mit", "mit-cs", "MIT", "g3-mit")
        self.assertEqual(len(col.rows), 2)
        row = sorted(col.rows, key=lambda r: r["claim"])[0]
        self.assertEqual((row["domain"], row["lic"], row["origin"], row["source"]), ("mit-cs", "MIT", "g3-mit", "mit"))
        self.assertEqual(row["group"], "doc:gyoogle:Network/oauth.md")
        self.assertEqual(row["commit_edits"], 2)                                      # 한 커밋에 한 줄 고침 둘
        self.assertTrue(all("Authentification" in r["claim"] or "전역 함수" in r["claim"] for r in col.rows))


if __name__ == "__main__":
    unittest.main()
