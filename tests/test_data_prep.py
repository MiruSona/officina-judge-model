"""data_prep — 라벨 매핑 · NC 거부 · KorNLI 표본 균등 · max_len 고르기."""
import collections
import json
import os
import tempfile
import unittest

import helpers  # noqa: F401

import data_prep as dp


class StripTagsTest(unittest.TestCase):
    def test_strips_line_head_tags_only(self):
        self.assertEqual(dp.strip_tags("왜 : 스왑이 밀렸다"), "스왑이 밀렸다")
        self.assertEqual(dp.strip_tags("  무엇을：문턱을 올렸다"), "문턱을 올렸다")
        self.assertEqual(dp.strip_tags("정한 것 : A\n다음에 : B"), "A\nB")
        # 줄 가운데 것 · 꼬리표 아닌 낱말은 그대로
        self.assertEqual(dp.strip_tags("까닭은 왜 : 이다"), "까닭은 왜 : 이다")
        self.assertEqual(dp.strip_tags("왜냐 : 그렇다"), "왜냐 : 그렇다")
        self.assertEqual(dp.strip_tags("다음 판"), "다음 판")

    def test_read_pairs_strips_mem_rows_not_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "m1", "evidence": "근거 : 가", "claim": "결론 : 나", "want": "support"},
                                   ensure_ascii=False) + "\n")
                f.write(json.dumps({"id": "p1", "premise": "왜 : 다", "hypothesis": "라", "label": 2, "src": "klue"},
                                   ensure_ascii=False) + "\n")
            rows = dp.read_pairs(path)
        self.assertEqual((rows[0]["premise"], rows[0]["hypothesis"]), ("가", "나"))
        self.assertEqual(rows[1]["premise"], "왜 : 다")
        self.assertEqual(dp.text_pair({"evidence": "주의 : 가", "claim": "나"}), ("가", "나"))


class LabelMapTest(unittest.TestCase):
    def test_klue_ids_follow_design_table(self):
        # 설계 2-3 : entailment(0)→A · contradiction(2)→B · neutral(1)→C
        self.assertEqual(dp.KLUE_ID_TO_LABEL, {0: 0, 2: 1, 1: 2})
        # 넷 확장 설계 1절 : a·b·c 차례는 그대로 두고 d 를 넷째에 붙인다
        self.assertEqual(dp.LABELS, ["support", "contradict", "unrelated", "related"])
        self.assertEqual(dp.LETTERS, ["a", "b", "c", "d"])

    def test_text_and_want_agree_with_klue(self):
        for text, klue_id in (("entailment", 0), ("contradiction", 2), ("neutral", 1)):
            self.assertEqual(dp.TEXT_TO_LABEL[text], dp.KLUE_ID_TO_LABEL[klue_id])
        for want, letter in (("support", "A"), ("contradict", "B"), ("unrelated", "C")):
            self.assertEqual(dp.want_label(want), dp.want_label(letter))
        # 공개 NLI 에는 d 에 맞는 원본 라벨이 없다
        self.assertEqual(dp.SOURCE_LABELS, ["entailment", "contradiction", "neutral", None])

    def test_klue_record_forms(self):
        self.assertEqual(dp.klue_label({"gold_label": "neutral"}), 2)
        self.assertEqual(dp.klue_label({"label": 2}), 1)
        self.assertIsNone(dp.klue_label({"label": -1}))

    def test_convert_klue_drops_unlabeled(self):
        data = {"rows": [{"row": {"guid": "g1", "premise": "p", "hypothesis": "h", "label": 0}},
                         {"row": {"guid": "g2", "premise": "p", "hypothesis": "h", "label": -1}}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "klue-nli-train.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            rows, dropped = dp.convert_klue([path])
        self.assertEqual(len(rows), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(rows[0]["label"], 0)

    def test_mem_pair_normalizes(self):
        row = dp.normalize({"id": "t1", "evidence": "근거", "claim": "주장", "want": "contradict", "kind": "flip"})
        self.assertEqual((row["premise"], row["hypothesis"], row["label"], row["src"]), ("근거", "주장", 1, "mem"))


class NcRejectTest(unittest.TestCase):
    def test_kornli_dev_test_rejected(self):
        for name in ("xnli.dev.ko.tsv", "xnli.test.ko.tsv", "multinli.dev.ko.tsv"):
            with self.assertRaises(SystemExit):
                dp.check_allowed(os.path.join("raw", name), train_only=True)

    def test_xnli_anli_rejected_everywhere(self):
        for name in ("xnli-train.json", "anli_r1.json"):
            with self.assertRaises(SystemExit):
                dp.check_allowed(name, train_only=False)

    def test_train_files_allowed(self):
        dp.check_allowed("multinli.train.ko.tsv", train_only=True)
        dp.check_allowed("snli_1.0_train.ko.tsv", train_only=True)
        dp.check_allowed("klue-nli-v1.1_dev.json", train_only=False)

    def test_read_kornli_train_tsv(self):
        text = "sentence1\tsentence2\tgold_label\n가\t나\tentailment\n다\t라\tneutral\n마\t바\t-\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "multinli.train.ko.tsv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            rows, dropped = dp.read_kornli_tsv(path, "kornli-mnli")
        self.assertEqual([r["label"] for r in rows], [0, 2])
        self.assertEqual(dropped, 1)
        self.assertEqual(rows[1]["id"], "kornli-mnli-1")

    def test_read_kornli_refuses_before_reading(self):
        with self.assertRaises(SystemExit):
            dp.read_kornli_tsv("does-not-exist.dev.ko.tsv", "kornli-mnli")


def fake_pool(src, per_label):
    return [{"id": f"{src}-{label}-{i}", "premise": "p", "hypothesis": "h", "label": label, "src": src}
            for label in range(3) for i in range(per_label)]


class KornliSampleTest(unittest.TestCase):
    def setUp(self):
        self.by_src = {"kornli-mnli": fake_pool("kornli-mnli", 100), "kornli-snli": fake_pool("kornli-snli", 60)}

    def test_ratio_and_label_balance(self):
        epochs = dp.sample_kornli(self.by_src, per_epoch=90, epochs=2, seed=1)
        for rows in epochs:
            self.assertEqual(len(rows), 90)
            by_src = collections.Counter(r["src"] for r in rows)
            self.assertEqual(by_src["kornli-mnli"], 63)
            self.assertEqual(by_src["kornli-snli"], 27)
            for src in by_src:
                labels = collections.Counter(r["label"] for r in rows if r["src"] == src)
                self.assertLessEqual(max(labels.values()) - min(labels.values()), 1)

    def test_epochs_do_not_overlap_and_repeat_with_seed(self):
        a = dp.sample_kornli(self.by_src, per_epoch=90, epochs=2, seed=1)
        b = dp.sample_kornli(self.by_src, per_epoch=90, epochs=2, seed=1)
        ids = [{r["id"] for r in rows} for rows in a]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual([r["id"] for r in a[0]], [r["id"] for r in b[0]])

    def test_too_small_pool_stops(self):
        with self.assertRaises(SystemExit):
            dp.sample_kornli(self.by_src, per_epoch=90, epochs=5, seed=1)


class EnglishTest(unittest.TestCase):
    def write(self, tmp, name, recs):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_read_drops_dash_and_maps_labels(self):
        recs = [{"pairID": "1e", "sentence1": "A man.", "sentence2": "A person.", "gold_label": "entailment"},
                {"pairID": "2c", "sentence1": "A man.", "sentence2": "No one.", "gold_label": "contradiction"},
                {"pairID": "3n", "sentence1": "A man.", "sentence2": "He is tall.", "gold_label": "neutral"},
                {"pairID": "4x", "sentence1": "A man.", "sentence2": "Hm.", "gold_label": "-"},
                {"pairID": "5x", "sentence1": "", "sentence2": "Empty.", "gold_label": "neutral"}]
        with tempfile.TemporaryDirectory() as tmp:
            rows, dropped = dp.read_nli_jsonl(self.write(tmp, "snli_1.0_train.jsonl", recs), "snli")
        # 칸 차례 : entailment→A(0) · contradiction→B(1) · neutral→C(2)
        self.assertEqual([r["label"] for r in rows], [0, 1, 2])
        self.assertEqual(dropped, 2)
        self.assertEqual((rows[0]["id"], rows[0]["src"]), ("snli-1e", "snli"))

    def test_dev_test_and_nc_names_refused(self):
        for name in ("multinli_1.0_dev_matched.jsonl", "snli_1.0_test.jsonl", "anli_r1_train.jsonl"):
            with self.assertRaises(SystemExit):
                dp.read_nli_jsonl(name, "mnli")

    def test_eval_en_takes_dev_but_not_nc(self):
        recs = [{"pairID": f"{i}", "sentence1": "p", "sentence2": "h", "gold_label": lab}
                for i, lab in enumerate(("entailment", "-", "contradiction", "neutral", "neutral"))]
        with tempfile.TemporaryDirectory() as tmp:
            src = self.write(tmp, "multinli_1.0_dev_matched.jsonl", recs)
            out = os.path.join(tmp, "mnli-devm-eval.jsonl")
            dp.cmd_eval_en(dp.parse_args(["eval-en", "--src", "mnli", "--in", src, "--out", out, "--limit", "3"]))
            rows = dp.read_jsonl(out)
            # 학습 길(train_only 기본값)은 여전히 dev 이름을 거부한다
            with self.assertRaises(SystemExit):
                dp.read_nli_jsonl(src, "mnli")
            nc = self.write(tmp, "anli_r1_dev.jsonl", recs)
            with self.assertRaises(SystemExit):
                dp.read_nli_jsonl(nc, "mnli", train_only=False)
        self.assertEqual([r["label"] for r in rows], [0, 1, 2])
        self.assertEqual({r["src"] for r in rows}, {"mnli"})

    def test_train_names_pass_nc_filter(self):
        dp.check_allowed("multinli_1.0_train.jsonl", train_only=True)
        dp.check_allowed("snli_1.0_train.jsonl", train_only=True)

    def test_licenses_known_and_not_nc(self):
        for src in ("mnli", "snli"):
            self.assertIn(src, dp.LICENSES)
            self.assertFalse(dp.LICENSES[src][1])

    def test_balanced_epochs_for_one_source(self):
        pool = fake_pool("mnli", 50)
        epochs = dp.sample_balanced({"mnli": pool}, {"mnli": 30}, 2, 1)
        ids = [{r["id"] for r in rows} for rows in epochs]
        self.assertFalse(ids[0] & ids[1])
        for rows in epochs:
            self.assertEqual(collections.Counter(r["label"] for r in rows), {0: 10, 1: 10, 2: 10})

    def test_kornli_sampling_unchanged_by_refactor(self):
        by_src = {"kornli-mnli": fake_pool("kornli-mnli", 100), "kornli-snli": fake_pool("kornli-snli", 60)}
        old = dp.sample_kornli(by_src, per_epoch=90, epochs=2, seed=1)
        new = dp.sample_balanced(by_src, {"kornli-mnli": 63, "kornli-snli": 27}, 2, 1)
        self.assertEqual([r["id"] for r in old[1]], [r["id"] for r in new[1]])

    def test_cli_writes_epoch_files(self):
        recs = [{"pairID": f"{i}{lab[0]}", "sentence1": "p", "sentence2": "h", "gold_label": lab}
                for i in range(4) for lab in ("entailment", "contradiction", "neutral")]
        with tempfile.TemporaryDirectory() as tmp:
            src = self.write(tmp, "multinli_1.0_train.jsonl", recs)
            out = os.path.join(tmp, "prep")
            dp.cmd_english(dp.parse_args(["mnli", "--in", src, "--out-dir", out, "--per-epoch", "6",
                                          "--epochs", "2"]))
            e1 = dp.read_jsonl(os.path.join(out, "mnli-e1.jsonl"))
            e2 = dp.read_jsonl(os.path.join(out, "mnli-e2.jsonl"))
            self.assertTrue(os.path.exists(os.path.join(out, "mnli-sample-1.txt")))
        self.assertEqual((len(e1), len(e2)), (6, 6))
        self.assertFalse({r["id"] for r in e1} & {r["id"] for r in e2})
        self.assertEqual({r["src"] for r in e1}, {"mnli"})


class MaxLenTest(unittest.TestCase):
    def test_smallest_candidate_covering_95(self):
        lengths = [100] * 95 + [300] * 5
        self.assertEqual(dp.choose_max_len(lengths), 128)
        lengths = [100] * 90 + [200] * 10
        self.assertEqual(dp.choose_max_len(lengths), 256)

    def test_never_above_320(self):
        self.assertEqual(dp.choose_max_len([1000] * 10), 320)

    def test_stats_percentiles(self):
        stats = dp.length_stats(list(range(1, 101)))
        self.assertEqual((stats["p50"], stats["p95"], stats["p99"], stats["max"]), (50, 95, 99, 100))


if __name__ == "__main__":
    unittest.main()
