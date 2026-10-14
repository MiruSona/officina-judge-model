"""layer_g4c — c 층 후보 뽑기 · 조립 (K2 설계 4-1 · 4-4 · 5절 ④). 글은 모두 지어낸 것이다."""
import collections
import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import layer_g4c as lc  # noqa: E402
from layer_g3 import Guard  # noqa: E402

WORDS = [a + b for a in "가나다라마바사아자차카타파하" for b in "거너더러머버서어저처"]


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def doc_rows(src, prefix, groups, rng, kinds="bad", **extra):
    """원천 하나 · 묶음마다 같은 문단에 b · a · d 주장."""
    rows = []
    for g in range(groups):
        gid = f"{src}:{prefix}{g}"
        ev = f"{src} 문서 {prefix}{g} : {text(rng, 12)} 이다."
        for k, label in enumerate(kinds):
            rid = f"{src}-{prefix}{g:03d}-{k}"
            rows.append(dict({"id": rid, "evidence": ev, "claim": f"{text(rng, 5)} {prefix}{g}-{k} 이다.", "want": label,
                              "hint": label, "domain": f"ko-{src}", "group": gid, "groups": [gid], "lang": "ko",
                              "lic": "CC-BY-SA", "src_ids": [rid]}, **extra))
    return rows


def fixture(tmp):
    rng = random.Random(3)
    layer, pools = [], {}
    for src in lc.SOURCES:
        layer += [dict(r, pick=src, set="train-g3-ko") for r in doc_rows(src, "L", 20, rng)]
        pools[src] = doc_rows(src, "P", 40, rng)
    pools["klue"] += doc_rows("klue", "U", 10, rng, kinds="c", form="unanswerable")
    blind = doc_rows("boolq", "B", 5, rng)                       # 평가 몫 후보 (묶음 boolq:B*)
    pools["boolq"] += [dict(r, id="dup-" + r["id"]) for r in blind[:3]]   # 평가와 같은 글 · 묶음 → 걸려야 한다
    path = os.path.join(tmp, "blind-g4-cands.jsonl")
    lc.write_jsonl(path, blind)
    forbid = os.path.join(tmp, "blind-g4.forbid.json")
    with open(forbid, "w", encoding="utf-8") as f:
        json.dump({"groups": ["wiki:P0"], "src_ids": [], "pair_keys": [], "cand_ids": []}, f)
    return layer, pools, Guard([path], [forbid])


TARGETS = {"pick": {"c-pair": {"boolq": 8, "klue": 6, "wiki": 6}, "near-x": {"boolq": 12, "wiki": 12, "klue": 8}},
           "build": {"c-pair": {"boolq": 6, "klue": 4, "wiki": 4}, "near-x": {"boolq": 10, "wiki": 10, "klue": 6}}}


def decide_all(cands, label_of=lambda r: "c"):
    return {r["id"]: {"id": r["id"], "label": label_of(r), "status": "agree", "votes": {"j1": label_of(r)}}
            for r in cands}


class PickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.layer, cls.pools, cls.guard = fixture(cls._tmp.name)
        cls.rows, cls.summary = lc.pick(cls.layer, cls.pools, cls.guard, TARGETS, 11)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_counts_and_fields(self):
        got = collections.Counter((r["pick"], r["form"]) for r in self.rows)
        for form in ("c-pair", "near-x"):
            for src, n in TARGETS["pick"][form].items():
                self.assertEqual(got[(src, form)], n, (src, form))
        for r in self.rows:
            self.assertEqual((r["split"], r["origin"], r["set"], r["hint"], r["want"]),
                             ("train", "g4-train", "train-g4-c", "c", ""))

    def test_no_forbidden_rows(self):
        self.assertEqual(lc.recheck(self.rows, self.guard), {})
        for r in self.rows:
            self.assertFalse({g for g in r["groups"] if g.startswith("boolq:B") or g == "wiki:P0"}, r["id"])
        self.assertGreater(self.summary["sources"]["boolq"]["dropped"].get("pool:eval_group", 0), 0)

    def test_c_pair_claims_are_layer_b(self):
        layer_b = {r["id"]: r for r in self.layer if r["want"] == "b"}
        pairs = [r for r in self.rows if r["form"] == "c-pair"]
        self.assertTrue(pairs)
        for r in pairs:
            self.assertEqual(r["claim"], layer_b[r["pair"]]["claim"])
            self.assertNotIn(layer_b[r["pair"]]["group"], {r["group"]})
        claims = collections.Counter(r["claim"] for r in self.rows)
        self.assertEqual(max(claims.values()), 1)                          # 주장 한 번

    def test_no_source_mixing_and_no_unanswerable(self):
        for r in self.rows:
            self.assertEqual({g.split(":")[0] for g in r["groups"]}, {r["pick"]}, r["id"])
            self.assertFalse(any(":U" in g for g in r["groups"]), r["id"])

    def test_near_x_layer_paragraphs_first(self):
        layer_ev = {r["evidence"] for r in self.layer}
        near = [r for r in self.rows if r["form"] == "near-x"]
        self.assertTrue(all(r["evidence"] in layer_ev for r in near))
        self.assertEqual(sum(s["near_from_layer"] for s in self.summary["sources"].values()), len(near))

    def test_two_per_evidence_group(self):
        per = collections.Counter(r["group"] for r in self.rows)
        self.assertLessEqual(max(per.values()), 2)

    def test_near_x_claims_never_layer_b(self):
        # 리뷰 #3 : near-x 주장에 train-g3-ko b 주장이 섞이지 않는다 (c 짝 · near-x 가 갈려야 -h 의 「c 짝 절반」 뜻이 선다)
        layer_b = {r["claim"] for r in self.layer if r["want"] == "b"}
        near = [r for r in self.rows if r["form"] == "near-x"]
        self.assertTrue(near)
        self.assertFalse({r["claim"] for r in near} & layer_b)

    def test_ids_carry_pair_hash(self):
        # 리뷰 #4 : id 꼬리 = 쌍 sha_key 앞 8자
        for r in self.rows:
            self.assertTrue(r["id"].endswith("-h" + lc.sha_key(r)[:8]), r["id"])

    def test_same_seed_same_rows(self):
        again, _ = lc.pick(self.layer, self.pools, self.guard, TARGETS, 11)
        self.assertEqual(json.dumps(again, ensure_ascii=False), json.dumps(self.rows, ensure_ascii=False))


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            layer, pools, guard = fixture(tmp)
            cls.cands, _ = lc.pick(layer, pools, guard, TARGETS, 11)

    def sha(self, rows):
        return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()

    def test_only_c_targets_and_half(self):
        not_c = {r["id"] for r in self.cands[::5]}
        decisions = decide_all(self.cands, lambda r: "d" if r["id"] in not_c else "c")
        rows, rows_h, summary = lc.build(self.cands, decisions, TARGETS, 5)
        self.assertFalse({r["id"] for r in rows} & not_c)
        self.assertTrue(all(r["want"] == "c" and r["set"] == "train-g4-c" for r in rows))
        got = collections.Counter((r["pick"], r["form"]) for r in rows)
        for form in ("c-pair", "near-x"):
            for src, n in TARGETS["build"][form].items():
                self.assertLessEqual(got[(src, form)], n)
        self.assertEqual(sum(got.values()), summary["rows"])
        near = [r["id"] for r in rows if r["form"] == "near-x"]
        self.assertEqual([r["id"] for r in rows_h if r["form"] == "near-x"], near)
        for src in lc.SOURCES:
            full = sum(1 for r in rows if r["form"] == "c-pair" and r["pick"] == src)
            half = sum(1 for r in rows_h if r["form"] == "c-pair" and r["pick"] == src)
            self.assertEqual(half, full // 2, src)
        self.assertTrue(all(r["set"] == "train-g4-c-h" for r in rows_h))
        per = collections.Counter(r["group"] for r in rows)
        self.assertLessEqual(max(per.values()), 2)

    def test_same_seed_same_sha(self):
        decisions = decide_all(self.cands)
        one = lc.build(self.cands, decisions, TARGETS, 5)
        two = lc.build(self.cands, decisions, TARGETS, 5)
        self.assertEqual(self.sha(one[0]), self.sha(two[0]))
        self.assertEqual(self.sha(one[1]), self.sha(two[1]))

    def test_cap_counts_only_kept_rows(self):
        # 리뷰 #11 : limit 에서 잘린 줄은 묶음 몫으로 안 센다
        rows = [{"group": g} for g in ("x", "y", "z", "w")]
        used = collections.Counter()
        got = lc._cap_groups(rows, 2, used, limit=2)
        self.assertEqual(len(got), 2)
        self.assertEqual(sum(used.values()), 2)
        self.assertEqual(used["z"], 0)

    def test_changed_pair_with_old_id_stops(self):
        # 리뷰 #4 : 다시 뽑아 같은 id 에 다른 쌍이 오면(옛 판정이 붙을 자리) 멈춘다
        decisions = decide_all(self.cands)
        bad = [dict(self.cands[0], claim=self.cands[0]["claim"] + " 바뀜")] + self.cands[1:]
        with self.assertRaises(lc.Stop):
            lc.build(bad, decisions, TARGETS, 5)

    def test_decision_pair_sha_mismatch_stops(self):
        decisions = decide_all(self.cands)
        decisions[self.cands[0]["id"]]["pair_sha"] = "0" * 64
        with self.assertRaises(lc.Stop):
            lc.build(self.cands, decisions, TARGETS, 5)
        decisions[self.cands[0]["id"]]["pair_sha"] = lc.sha_key(self.cands[0])
        lc.build(self.cands, decisions, TARGETS, 5)

    def test_boundary_label_cd_dropped_not_stopped(self):
        # train 판정이 경계 칸 cd 로 일치한 줄(agree · cd)이 들어도 멈추지 않고 버림으로 센다
        decisions = decide_all(self.cands)
        cd_ids = [r["id"] for r in self.cands[:3]]
        for rid in cd_ids:
            decisions[rid].update(label="cd", status="agree")
        decisions[self.cands[3]["id"]].update(label="cd", status="cd-split")
        rows, _, summary = lc.build(self.cands, decisions, TARGETS, 5)
        self.assertFalse({r["id"] for r in rows} & set(cd_ids))
        self.assertEqual(summary["boundary_dropped"], 4)
        self.assertEqual(summary["cand_state"]["dropped"], 4)

    def test_pending_stops_unless_draft(self):
        decisions = decide_all(self.cands[1:])
        with self.assertRaises(lc.Stop):
            lc.build(self.cands, decisions, TARGETS, 5)
        _, _, summary = lc.build(self.cands, decisions, TARGETS, 5, draft=True)
        self.assertEqual(summary["cand_state"]["pending"], 1)


class CliTest(unittest.TestCase):
    def test_refuses_output_inside_git(self):
        # 리뷰 #8 : 쌍 본문 출력 폴더가 git 안이면 exit 2
        for cmd in ("pick", "build"):
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(lc, "inside_git", lambda path: True), mock.patch("sys.stdout"):
                    code = lc.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", "train-g4-c")))

    def test_pick_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "g4", "train-g4-c")
            os.makedirs(out)
            with open(os.path.join(out, "pick.jsonl"), "w", encoding="utf-8") as f:
                f.write("old\n")
            with mock.patch("sys.stdout"):
                code = lc.main(["pick", "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
            self.assertEqual(code, 2)
            with open(os.path.join(out, "pick.jsonl"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "old\n")


if __name__ == "__main__":
    unittest.main()
