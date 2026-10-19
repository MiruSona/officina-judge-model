"""dev_g9 trim · check · blind (K7 설계 1-2 · 4-1 2단계). 글은 모두 지어낸 것이다."""
import hashlib
import os
import sys
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import dev_g9  # noqa: E402


def row(rid, group, guide="support-v6", want="a", ev="근거 글", cl="주장 글", lang="ko", src=None):
    return {"id": rid, "group": group, "groups": [group], "label_guide": guide, "want": want, "evidence": ev,
            "claim": cl, "lang": lang, "domain": "kowiki", "src_ids": src or [rid]}


class TrimTest(unittest.TestCase):
    def test_drop_layer_group_and_v5_list(self):
        dev = [row("d1", "g:1", "support-v5"), row("d2", "g:2", "support-v5"), row("d3", "g:3"),
               row("d4", "g:4", "support-v6", src=["s:9"])]
        layers = {"train-g1-arch": [row("t1", "g:2")], "train-r2": [row("t2", "g:x", src=["s:9"])]}
        drop, v5, why = dev_g9.trim(dev, layers)
        self.assertEqual(drop, ["d2", "d4"])           # 묶음 겹침 · 출처 id 겹침
        self.assertEqual(v5, ["d1"])                   # 뺀 쌍은 재라벨 대상에서도 빠진다
        self.assertEqual(why["train-g1-arch"], 1)


class CheckTest(unittest.TestCase):
    def test_counts_overlap_and_within(self):
        marks = {"train": dev_g9.Marks(), "blind-g4": dev_g9.Marks()}
        marks["train"].add_rows([row("t1", "g:1", ev="학습 근거 하나", cl="학습 주장 하나")], "train")
        hit = row("x", "g:9", ev="학습 근거 하나", cl="학습 주장 하나")
        sha = dev_g9.ms.sha_key({"evidence": "블라인드 근거", "claim": "블라인드 주장"})
        marks["blind-g4"].add_forbid({"groups": ["g:7"], "src_ids": [], "pair_keys": [sha]})
        cands = [{"id": "c1", "src": "kowiki", "form": "wiki-edit", "evidence": "새 근거", "claim": "새 주장"},
                 {"id": "c2", "src": "kowiki", "form": "wiki-edit", "evidence": hit["evidence"], "claim": hit["claim"]},
                 {"id": "c3", "src": "kowiki", "form": "wiki-edit", "evidence": "블라인드 근거", "claim": "블라인드 주장"}]
        hints = [{"id": "c1", "group": "g:5", "groups": ["g:5"], "src_ids": ["n1"]},
                 {"id": "c2", "group": "g:6", "groups": ["g:6"], "src_ids": ["n2"]},
                 {"id": "c3", "group": "g:7", "groups": ["g:7"], "src_ids": ["n3"]}]
        out = dev_g9.check(cands, hints, marks)
        self.assertEqual(out["text_train"], 1)
        self.assertEqual(out["group_blind-g4"], 1)
        self.assertEqual(out["sha_blind-g4"], 1)
        self.assertEqual(out["group_train"], 0)
        clean = dev_g9.check(cands[:1], hints[:1], marks)
        self.assertEqual(sum(clean.values()), 0)


class BlindTest(unittest.TestCase):
    def test_batches_hide_labels_and_map_back(self):
        new = [{"id": f"g9n-{i:012x}", "src": "kowiki", "form": "wiki-edit", "evidence": f"근거 {i}",
                "claim": f"주장 {i}"} for i in range(150)]
        old = [row(f"boolq-dev-{i:05d}", f"g:{i}", "support-v5", want="b") for i in range(60)]
        batches, table = dev_g9.blind(new, old, seed=7, size=100)
        self.assertEqual([len(b) for b in batches], [100, 100, 10])
        for b in batches:
            for r in b:
                self.assertEqual(set(r), {"bid", "evidence", "claim"})
        self.assertEqual(len(table), 210)
        first = next(m for m in table if m["id"] == "boolq-dev-00000")
        self.assertEqual(first["bid"], hashlib.sha256(b"g9-blind-v1:boolq-dev-00000").hexdigest()[:12])
        self.assertEqual((first["from"], first["old_want"]), ("old-dev", "b"))
        again, _ = dev_g9.blind(new, old, seed=7, size=100)
        self.assertEqual(batches, again)                 # 씨앗 고정 → 같은 차례
        self.assertNotEqual([r["bid"] for r in batches[0]][:5], sorted(m["bid"] for m in table)[:5])


class ExtraTest(unittest.TestCase):
    """보충 판 — 묶음 번호는 start 부터 · 앞 판 후보(avoid)와 겹치면 걸린다 · near-x 는 허용할 때만."""

    def test_blind_start_numbers_batches(self):
        new = [{"id": f"g9n-{i:012x}", "src": "kowiki", "form": "near-x", "evidence": f"근거 {i}",
                "claim": f"주장 {i}"} for i in range(120)]
        batches, table = dev_g9.blind(new, [], seed=7, size=100, start=24)
        self.assertEqual([len(b) for b in batches], [100, 20])
        self.assertEqual(sorted({m["batch"] for m in table}), ["g9-b024", "g9-b025"])
        self.assertTrue(all(m["from"] == "new-ko" for m in table))

    def test_avoid_marks_and_near_x_gate(self):
        old = row("o1", "g:1", ev="앞 판 근거", cl="앞 판 주장")
        marks, _ = dev_g9.load_marks(os.path.join(helpers.ROOT, "no-such-root"), avoid={"cands-g9": [old]})
        self.assertIn("group", marks["cands-g9"].reasons(row("n1", "g:1", ev="다른 근거", cl="다른 주장")))
        rows = [dict(row("r1", "g:5", ev="근접 근거 하나", cl="근접 주장 하나"), hint="c", form="near-x"),
                dict(row("r2", "g:6", ev="근접 근거 둘", cl="근접 주장 둘"), hint="c", form="near-x")]
        plain = {"train": dev_g9.Marks()}
        got, dropped = dev_g9.pick_source("kowiki", rows, {"total": 5, "b": 0}, None, plain, dev_g9.random.Random(1))
        self.assertEqual((len(got), dropped["generator_form"]), (0, 2))
        got, _ = dev_g9.pick_source("kowiki", rows, {"total": 5, "b": 0}, None, plain, dev_g9.random.Random(1),
                                    allow_forms={"near-x"})
        self.assertEqual(sorted(r["id"] for r in got), ["r1", "r2"])


def final(rid, label, set_name="new-ko", form="qa-true", l1=None, l2=None, l3=None, agree="2"):
    return {"id": rid, "bid": "b" + rid, "form": form, "set": set_name, "label": label,
            "l1": l1 or label, "l2": l2 or label, "l3": l3, "agree": agree}


def cand(rid, src, form="qa-true"):
    return {"id": rid, "src": src, "form": form, "evidence": f"근거 {rid}", "claim": f"주장 {rid}",
            "hint": "a", "group": f"n:{rid}", "groups": [f"n:{rid}"], "src_ids": [f"s:{rid}"], "lang": "ko",
            "lic": "CC-BY-SA-4.0", "orig_id": "o-" + rid, "split": "train", "origin": "g3-x"}


class SpreadTest(unittest.TestCase):
    """칸 몫을 원천에 나누기 — 고르게(또는 비대로), 모자란 원천 몫은 남은 원천에 다시 나눈다."""

    def test_even_with_shortage(self):
        got = dev_g9.spread(250, {"ka": 204, "mrc": 45, "boolq": 362, "wiki": 21})
        self.assertEqual(sum(got.values()), 250)
        self.assertEqual((got["wiki"], got["mrc"]), (21, 45))
        self.assertEqual(sorted((got["ka"], got["boolq"])), [92, 92])

    def test_weights_and_short_pool(self):
        got = dev_g9.spread(300, {"wiki": 149, "boolq": 157, "ka": 86, "mrc": 82},
                            {"wiki": 100, "boolq": 100, "ka": 60, "mrc": 40})
        self.assertEqual(got, {"wiki": 100, "boolq": 100, "ka": 60, "mrc": 40})
        self.assertEqual(dev_g9.spread(250, {"x": 100, "y": 27}), {"x": 100, "y": 27})   # 다 모자라면 있는 만큼


class AssembleTest(unittest.TestCase):
    """dev-cmp-g9 조립 — drop 빼기 · v5 라벨 교체 · 칸 몫 고르기 · form 보존 · 줄 꼴 · 결정성."""

    def setUp(self):
        self.dev = [dict(row("d1", "g:1", "support-v5", want="b"), form="near", set="dev-cmp-g4"),
                    dict(row("d2", "g:2", "support-v5", want="c"), form="same-doc", set="dev-cmp-g4"),
                    dict(row("d3", "g:3", want="d"), form="near-x", set="dev-cmp-g4"),
                    dict(row("d4", "g:4", want="a"), form="labeled", set="dev-cmp-g4")]
        self.drop, self.v5 = ["d4"], ["d1", "d2"]
        self.cands = {}
        finals = [final("d1", "a", "old-dev", form="near", l1="a", l2="b", l3="a", agree="3"),
                  final("d2", "c", "old-dev", form="same-doc")]
        n = 0
        for label, src, k, form in (("b", "kowiki", 4, "wiki-edit"), ("b", "klue-mrc", 3, "qa-alt"),
                                    ("a", "klue-answered", 3, "qa-true"), ("c", "kowiki", 2, "near-x"),
                                    ("c", "klue-answered", 1, "qa-true"), ("d", "klue-mrc", 1, "unanswerable")):
            for _ in range(k):
                rid = f"g9n-{n:04d}"
                n += 1
                self.cands[rid] = cand(rid, src, form)
                finals.append(final(rid, label, form=form))
        self.finals = finals
        self.quota = {"b": 4, "a": 2, "c": 2, "d": 2}

    def build(self, seed=5):
        return dev_g9.assemble(self.dev, self.drop, self.v5, self.finals, self.cands, self.quota, seed=seed)

    def test_drop_relabel_and_carry(self):
        rows, rep = self.build()
        ids = [r["id"] for r in rows]
        self.assertNotIn("d4", ids)
        self.assertEqual(ids[:3], ["d1", "d2", "d3"])          # 옛 줄은 옛 차례 그대로 앞에
        d1, d2, d3 = rows[:3]
        self.assertEqual((d1["want"], d1["g4_want"], d1["label_guide"], d1["g9_label"]), ("a", "b", "support-v6", "v6-relabel"))
        self.assertEqual((d1["votes"], d1["status"]), ({"j1": "a", "j2": "b", "j3": "a"}, "third"))
        self.assertEqual(d1["form"], "near")                   # form 은 옛 줄 그대로
        self.assertEqual((d3["want"], d3["g9_label"], d3["label_guide"]), ("d", "g4-carry", "support-v6"))
        self.assertNotIn("g4_want", d3)
        self.assertTrue(all(r["set"] == "dev-cmp-g9" for r in rows))
        self.assertEqual(d3["g4_set"], "dev-cmp-g4")
        self.assertEqual((rep["old"]["dropped"], rep["old"]["relabeled"], rep["old"]["changed"]), (1, 2, 1))

    def test_quota_and_new_row_shape(self):
        rows, rep = self.build()
        new = [r for r in rows if r["g9_label"] == "v6-new"]
        counts = dev_g9.collections.Counter(r["want"] for r in new)
        self.assertEqual(dict(counts), {"b": 4, "a": 2, "c": 2, "d": 1})   # d 는 1 뿐이라 있는 만큼
        self.assertEqual(rep["new"]["shortfall"], {"d": 1})
        by_src = dev_g9.collections.Counter(r["domain"] for r in new if r["want"] == "b")
        self.assertEqual(dict(by_src), {"kowiki": 3, "klue-mrc": 1})       # b 는 B_SHARE 비(100:40)
        c_src = dev_g9.collections.Counter(r["domain"] for r in new if r["want"] == "c")
        self.assertEqual(dict(c_src), {"kowiki": 1, "klue-answered": 1})  # 나머지 칸은 원천마다 고르게
        c_forms = sorted(r["form"] for r in new if r["want"] == "c")
        self.assertIn("qa-true", c_forms)                                   # 생성기 꼴 아닌 쌍 먼저
        r = new[0]
        for key in ("id", "evidence", "claim", "want", "form", "domain", "group", "groups", "lang", "label_guide",
                    "src_ids", "votes", "status", "set"):
            self.assertIn(key, r)
        self.assertEqual((r["label_guide"], r["lang"]), ("support-v6", "ko"))
        self.assertEqual(r["form"], self.cands[r["id"]]["form"])
        self.assertEqual([x["id"] for x in new], sorted(x["id"] for x in new))

    def test_deterministic_seed(self):
        a, _ = self.build(seed=5)
        b, _ = self.build(seed=5)
        self.assertEqual(a, b)
        picks = {tuple(r["id"] for r in self.build(seed=s)[0]) for s in range(1, 12)}
        self.assertGreater(len(picks), 1)                                   # 씨앗이 고르기를 바꾼다

    def test_label_only_supplement_keeps_other_picks(self):
        base, _ = self.build(seed=5)
        cands, finals = dict(self.cands), list(self.finals)
        for i, label in enumerate(("a", "b", "d", "d")):                   # 보충 판 : a · b 는 안 쓰고 d 만
            rid = f"g9x-{i:04d}"
            cands[rid] = dict(cand(rid, "kowiki", "wiki-edit"), cand_file="extra")
            finals.append(final(rid, label, form="wiki-edit"))
        rows, rep = dev_g9.assemble(self.dev, self.drop, self.v5, finals, cands, self.quota, seed=5,
                                    label_only={"extra": ("d",)})
        self.assertEqual(rep["new"]["label_only_unused"], {"extra:a": 1, "extra:b": 1})
        added = [r for r in rows if r["id"] not in {x["id"] for x in base}]
        self.assertEqual(sorted(r["want"] for r in added), ["d"])          # d 몫 2 중 빈 1 만 채운다
        self.assertEqual([r for r in rows if r["want"] != "d"], [r for r in base if r["want"] != "d"])
        self.assertNotIn("cand_file", added[0])

    def test_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):                                 # v5 인데 최종 라벨이 없다
            dev_g9.assemble(self.dev, self.drop, self.v5 + ["d3"], self.finals, self.cands, self.quota)
        with self.assertRaises(ValueError):                                 # 새 줄 id 가 후보 파일에 없다
            dev_g9.assemble(self.dev, self.drop, self.v5, self.finals + [final("g9n-zzzz", "a")], self.cands,
                            self.quota)
        with self.assertRaises(ValueError):                                 # drop 이 dev 에 없다
            dev_g9.assemble(self.dev, ["nope"], self.v5, self.finals, self.cands, self.quota)


class CheckSetTest(unittest.TestCase):
    def test_old_rows_skip_old_dev_marks_new_rows_do_not(self):
        marks = {"train": dev_g9.Marks(), "dev-cmp-g4": dev_g9.Marks(), "blind-g4": dev_g9.Marks()}
        old = dict(row("d1", "g:1", ev="옛 근거", cl="옛 주장"), g9_label="g4-carry")
        marks["dev-cmp-g4"].add_rows([old], "dev")
        clean_new = dict(row("n1", "g:9", ev="새 근거", cl="새 주장"), g9_label="v6-new")
        out = dev_g9.check_set([old, clean_new], marks)
        self.assertEqual(sum(out.values()), 0)
        bad_new = dict(row("n2", "g:1", ev="다른 근거", cl="다른 주장"), g9_label="v6-new")
        out = dev_g9.check_set([old, bad_new], marks)
        self.assertEqual(out["group_dev-cmp-g4"], 1)
        dup = dict(old, id="d9")
        self.assertEqual(dev_g9.check_set([old, dup], marks)["text_within"], 1)


class JudgeCheckTest(unittest.TestCase):
    def test_soft_only_for_old_para_and_other_layers(self):
        new = {"group_train": 0, "para_train": 0}
        old = {"group_recipe": 0, "para_recipe": 3, "group_other-train": 2, "group_blind-g4": 0}
        self.assertEqual(dev_g9.judge_check(new, old), (0, 5))
        self.assertEqual(dev_g9.judge_check(dict(new, para_train=1), old), (1, 5))     # 새 줄 문단 겹침은 막는다
        self.assertEqual(dev_g9.judge_check(new, dict(old, group_recipe=1)), (1, 5))   # 옛 줄 처방 층 묶음은 막는다


if __name__ == "__main__":
    unittest.main()
