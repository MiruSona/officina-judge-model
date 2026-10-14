"""layer_g4d — d 층 후보 뽑기 · 조립 (K3 설계 4-1 · 4-2 · 4-4 · 5절 ①). 글은 모두 지어낸 것이다."""
import collections
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import layer_g4d as ld  # noqa: E402
from layer_g3 import Guard  # noqa: E402

FORMS = list(ld.FORMS)


def pair_text(tag, k):
    """근거 낱말 10 · 주장 낱말 10 중 k 개가 근거에도 있다 → 겹침 k/10."""
    ev = [f"근{tag}x{j}" for j in range(10)]
    claim = ev[:k] + [f"주{tag}x{j}" for j in range(10 - k)]
    return " ".join(ev) + " 이다", " ".join(claim)


def pool_rows(form, n, k_of, groups=None):
    """꼴 하나의 후보 n 줄 — i 번 줄의 겹침은 k_of(i)/10, 묶음은 groups 개를 돌려 쓴다."""
    rows = []
    for i in range(n):
        ev, claim = pair_text(f"{FORMS.index(form)}n{i}", k_of(i))
        group = f"{ld.FORMS[form][0]}:G{i % (groups or n)}"
        base = {"group": group, "domain": "dom", "lic": "CC"}
        rows.append(ld._row(form, base, ev, claim, [f"{form}-src-{i}"]))
    return rows


def all_pools(n=60, k_of=lambda i: i % 5):
    return {form: pool_rows(form, n, k_of) for form in FORMS}


class NoGuard:
    def reasons(self, row):
        return []


SMALL = {"pick": {ld.KLUE: 20, ld.WIKI: 20, ld.VITC[0]: 8, ld.VITC[1]: 6, ld.VITC[2]: 6},
         "build": {ld.KLUE: 10, ld.WIKI: 10, ld.VITC[0]: 4, ld.VITC[1]: 3, ld.VITC[2]: 3}}


class OverlapTest(unittest.TestCase):
    def test_by_hand(self):
        # 주장 낱말 {a1, 2024, 서울} 중 근거에 있는 것 {a1, 서울} → 2/3. 문장부호는 낱말이 아니다
        self.assertAlmostEqual(ld.overlap("서울 A1 이다.", "a1, 2024 서울!"), 2 / 3)
        self.assertEqual(ld.overlap("근거", ""), 0.0)

    def test_bins(self):
        edges = ld.TARGETS["edges"]
        self.assertEqual([ld.bin_of(v, edges) for v in (0.0, 0.0999, 0.1, 0.19, 0.2, 0.349, 0.35, 1.0)],
                         [0, 0, 1, 1, 2, 2, 3, 3])

    def test_quota_largest_remainder(self):
        self.assertEqual(ld._quota(750, ld.TARGETS["shares"]), [188, 187, 225, 150])
        self.assertEqual(sum(ld._quota(7, ld.TARGETS["shares"])), 7)


class SourceTest(unittest.TestCase):
    def klue(self):
        mrc, qa = [], []
        for i in range(6):
            src = f"klue-mrc-v1_train_{i:05d}"
            mrc.append({"id": f"m{i}", "evidence": f"문단 {i} 앞 문장이다. 뒤 문장이다.", "form": "removed",
                        "split": "train", "group": f"klue:P{i}", "groups": [f"klue:P{i}"], "domain": "klue-mrc",
                        "lic": "CC", "src_ids": [src], "answer": f"답{i}"})
            qa.append({"id": f"q{i}", "claim": f"무엇{i}인가?", "claim_rewrite": f"답{i} 이 그 일을 했다.",
                       "form": "qa-true", "split": "train", "domain": "klue-answered", "group": f"klue:P{i}",
                       "src_ids": [src], "answer": f"답{i}", "pair": src})
        mrc[1]["evidence"] += " 답1 이 남았다."                         # 답 글이 근거에 남음 → 뺌
        mrc[2]["split"] = "heldout"                                      # 평가 쪽 → 안 씀
        mrc[3]["form"] = "unanswerable"                                  # 답 없는 질문 → 안 씀
        qa.append(dict(qa[0], id="q0-alt", form="qa-alt", claim_rewrite="딴 답 이 그 일을 했다."))
        return mrc, qa

    def test_klue_answer_split_form_rules(self):
        mrc, qa = self.klue()
        c_rows = [{"id": "g4c-x1-klue-0001-h00000000", "pick": "klue", "form": "near-x", "claim": "답4 이 그 일을 했다."}]
        rows, dropped = ld.klue_cands(mrc, qa, [], c_rows)
        self.assertEqual(sorted(r["group"] for r in rows), ["klue:P0", "klue:P4", "klue:P5"])
        self.assertEqual(dropped["klue_answer_in_evidence"], 1)
        by_group = {r["group"]: r for r in rows}
        self.assertEqual(by_group["klue:P0"]["claim"], "답0 이 그 일을 했다.")    # qa-true 평서문만
        self.assertEqual(by_group["klue:P4"]["pair"], "g4c-x1-klue-0001-h00000000")   # d 짝 = train-g4-c id
        self.assertNotIn("pair", by_group["klue:P0"])
        for r in rows:
            self.assertEqual((r["form"], r["pick"], r["lang"], r["hint"], r["split"], r["origin"], r["set"]),
                             (ld.KLUE, "klue", "ko", "d", "train", "g4-train", "train-g4-d"))

    def test_klue_check_fail_dropped(self):
        mrc, qa = self.klue()
        rows, dropped = ld.klue_cands(mrc, qa, [{"id": "q5", "pass": False}])
        self.assertNotIn("klue:P5", {r["group"] for r in rows})
        self.assertEqual(dropped["klue_qa:check_fail"], 1)

    def test_wiki_other_sentence_title_and_same_line(self):
        def w(group, title, ev, claim, src):
            return {"evidence": f"{title} : {ev}", "claim": claim, "title": title, "group": group, "split": "any",
                    "domain": "kowiki", "lic": "CC", "src_ids": [src]}
        rows = [w("kowiki:1", "갑", "문장 하나 이다.", "문장 하나 였다.", "r1"),
                w("kowiki:1", "갑", "문장 하나 였다.", "문장 하나 일까.", "r2"),     # r1 고침 줄기 → 같은 문장
                w("kowiki:1", "갑", "다른 사실 둘 이다.", "다른 사실 셋 이다.", "r3"),
                w("kowiki:2", "을", "외톨이 문장 이다.", "외톨이 문장 였다.", "r4")]   # 문장 하나뿐인 글
        got, dropped = ld.wiki_cands(rows)
        self.assertEqual(dropped["wiki_same_line"], 2)
        self.assertEqual(len(got), 4)            # 줄기 {하나 이다 · 하나 였다} × {둘 이다} 양쪽
        for r in got:
            self.assertTrue(r["evidence"].startswith("갑 : "))
            self.assertFalse(r["claim"].startswith("갑 :"))                 # 「제목 :」 뗌
            self.assertNotEqual(ld.norm(r["evidence"][4:]), ld.norm(r["claim"]))
            self.assertEqual((r["group"], r["pick"], r["form"]), ("kowiki:1", "wiki", ld.WIKI))

    def test_wiki_near_dup_dropped(self):
        # 낱말 10 중 9 가 같은 두 문장(겹침 .9 ≥ .7)은 같은 문장의 다른 판으로 보고 뺀다 — 한 쌍 방향마다 1 씩
        def w(ev, src):
            return {"evidence": f"갑 : {ev}", "claim": "", "title": "갑", "group": "kowiki:1", "split": "any",
                    "domain": "kowiki", "lic": "CC", "src_ids": [src]}
        words = [f"낱{j}" for j in range(9)]
        rows = [w(" ".join(words + ["하나"]), "r1"), w(" ".join(words + ["둘"]), "r2")]
        got, dropped = ld.wiki_cands(rows)
        self.assertEqual((got, dropped["wiki_near_dup"]), ([], 2))
        got, dropped = ld.wiki_cands(rows, max_overlap=1.01)
        self.assertEqual((len(got), dropped["wiki_near_dup"]), (2, 0))

    def test_vitc_forms_split_and_same_sentence(self):
        def v(rid, form, split="train", claim=None):
            return {"id": rid, "evidence": f"Evidence {rid}.", "claim": claim or f"Claim {rid}.", "form": form,
                    "split": split, "group": "vitc:P", "domain": "vitaminc", "lic": "CC", "src_ids": [rid]}
        rows = [v("1", "same-page"), v("2", "near"), v("3", "same-doc"), v("4", "labeled"),
                v("5", "near", "heldout"), v("6", "same-doc", claim="evidence 6.")]
        got, dropped = ld.vitc_cands(rows)
        self.assertEqual({f: [r["src_ids"][0] for r in rs] for f, rs in got.items()},
                         {ld.VITC[0]: ["1"], ld.VITC[1]: ["2"], ld.VITC[2]: ["3"]})
        self.assertEqual(dropped["vitc_same_sentence"], 1)
        self.assertEqual(got[ld.VITC[0]][0]["lang"], "en")


class PickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = all_pools()
        cls.rows, cls.summary = ld.pick(cls.pools, NoGuard(), SMALL, 7)

    def test_targets_fields_and_ids(self):
        got = collections.Counter(r["form"] for r in self.rows)
        self.assertEqual(dict(got), SMALL["pick"])
        for r in self.rows:
            self.assertEqual((r["split"], r["origin"], r["set"], r["hint"], r["want"]),
                             ("train", "g4-train", "train-g4-d", "d", ""))
            self.assertTrue(r["id"].startswith(f"g4d-{r['form']}-"))
            self.assertTrue(r["id"].endswith("-h" + ld.sha_key(r)[:8]))
            self.assertAlmostEqual(r["overlap"], ld.overlap(r["evidence"], r["claim"]), places=4)

    def test_overlap_bin_quota_per_lang(self):
        # ko 40 → 10 · 10 · 12 · 8 · en 20 → 5 · 5 · 6 · 4 (25/25/30/20, 큰 나머지)
        edges = ld.TARGETS["edges"]
        for lang, quota in (("ko", [10, 10, 12, 8]), ("en", [5, 5, 6, 4])):
            self.assertEqual(self.summary["langs"][lang]["quota"], quota)
            got = collections.Counter(ld.bin_of(r["overlap"], edges) for r in self.rows if r["lang"] == lang)
            self.assertEqual([got[b] for b in range(4)], quota, lang)

    def test_scarce_low_bin_filled_by_other_source(self):
        # KLUE 에 낮은 구간 줄이 없으면 ko 0~.1 몫은 위키가 메우고, KLUE 는 다른 구간에서 목표를 채운다
        pools = dict(self.pools, **{ld.KLUE: pool_rows(ld.KLUE, 60, lambda i: 1 + i % 4)})
        rows, summary = ld.pick(pools, NoGuard(), SMALL, 7)
        edges = ld.TARGETS["edges"]
        low = collections.Counter(r["form"] for r in rows if ld.bin_of(r["overlap"], edges) == 0 and r["lang"] == "ko")
        self.assertEqual(dict(low), {ld.WIKI: 10})
        self.assertEqual(summary["forms"][ld.KLUE]["picked"], 20)
        self.assertEqual(summary["langs"]["ko"]["got"], summary["langs"]["ko"]["quota"])

    def test_short_bin_spills_to_neighbor(self):
        # ko 낮은 구간 줄이 아예 없으면 그 몫은 모자람으로 적고, 꼴 목표는 다른 구간에서 채운다
        pools = dict(self.pools, **{f: pool_rows(f, 60, lambda i: 1 + i % 4) for f in (ld.KLUE, ld.WIKI)})
        rows, summary = ld.pick(pools, NoGuard(), SMALL, 7)
        self.assertEqual(summary["langs"]["ko"]["short_bins_before_spill"][0], 10)
        self.assertEqual((summary["forms"][ld.KLUE]["picked"], summary["forms"][ld.WIKI]["picked"]), (20, 20))

    def test_no_source_mixing(self):
        for r in self.rows:
            self.assertEqual({g.split(":")[0] for g in r["groups"]}, {r["pick"]}, r["id"])
            self.assertEqual(r["pick"], ld.FORMS[r["form"]][0])

    def test_two_per_group_and_claim_once(self):
        pools = {form: pool_rows(form, 60, lambda i: i % 5, groups=6) for form in FORMS}
        rows, summary = ld.pick(pools, NoGuard(), SMALL, 7)
        per = collections.Counter(r["group"] for r in rows)
        self.assertLessEqual(max(per.values()), 2)
        self.assertEqual(sum(1 for r in rows if r["pick"] == "vitaminc"), 6 * 2)   # 쪽 6 × 2 — 꼴 셋이 같이 센다
        self.assertEqual(summary["forms"][ld.VITC[2]]["short"] + summary["forms"][ld.VITC[1]]["short"]
                         + summary["forms"][ld.VITC[0]]["short"], 20 - 12)
        claims = collections.Counter(ld.norm(r["claim"]) for r in self.rows)
        self.assertEqual(max(claims.values()), 1)

    def test_pair_rows_first(self):
        pools = dict(self.pools)
        pools[ld.KLUE] = [dict(r, pair=f"c-{i}") if i % 6 == 0 else r for i, r in enumerate(self.pools[ld.KLUE])]
        rows, summary = ld.pick(pools, NoGuard(), SMALL, 7)
        self.assertEqual(summary["forms"][ld.KLUE]["pair_pool"], 10)
        self.assertEqual(summary["forms"][ld.KLUE]["pair_picked"], 10)

    def test_same_seed_same_sha(self):
        def sha(rows):
            return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()
        again, _ = ld.pick(all_pools(), NoGuard(), SMALL, 7)
        self.assertEqual(sha(again), sha(self.rows))
        other, _ = ld.pick(all_pools(), NoGuard(), SMALL, 8)
        self.assertNotEqual(sha(other), sha(self.rows))


class ForbidTest(unittest.TestCase):
    def test_eval_and_c_pair_rows_never_picked(self):
        pools = all_pools()
        with tempfile.TemporaryDirectory() as tmp:
            blind = [dict(r, id=f"bl-{i}") for i, r in enumerate(pools[ld.WIKI][:5])]      # 평가와 같은 글 · 묶음
            path = os.path.join(tmp, "blind-g4.jsonl")
            ld.write_jsonl(path, blind)
            forbid = os.path.join(tmp, "blind-g4.forbid.json")
            with open(forbid, "w", encoding="utf-8") as f:
                json.dump({"groups": [pools[ld.VITC[0]][0]["group"]], "src_ids": [], "pair_keys": [], "cand_ids": []}, f)
            c_rows = [dict(r, id=f"c-{i}") for i, r in enumerate(pools[ld.KLUE][:30])]   # train-g4-c 와 같은 쌍
            guard = ld.DGuard(Guard([path], [forbid]), c_rows)
            rows, summary = ld.pick(pools, guard, SMALL, 7)
        self.assertEqual(ld.recheck(rows, guard), {})
        bad_groups = {r["group"] for r in blind} | {pools[ld.VITC[0]][0]["group"]}
        self.assertFalse({r["group"] for r in rows} & bad_groups)
        c_keys = {ld.sha_key(r) for r in c_rows}
        self.assertFalse({ld.sha_key(r) for r in rows} & c_keys)
        self.assertGreater(summary["dropped"].get("guard:c_pair", 0), 0)
        self.assertGreater(summary["dropped"].get("guard:eval_group", 0), 0)


def decide_all(cands, label_of=lambda r: "d"):
    return {r["id"]: {"id": r["id"], "label": label_of(r), "status": "agree", "votes": {"j1": label_of(r)}}
            for r in cands}


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cands, _ = ld.pick(all_pools(), NoGuard(), SMALL, 7)

    def test_only_d_and_targets(self):
        not_d = {r["id"] for r in self.cands[::4]}
        decisions = decide_all(self.cands, lambda r: "c" if r["id"] in not_d else "d")
        rows, summary = ld.build(self.cands, decisions, SMALL, 5)
        self.assertFalse({r["id"] for r in rows} & not_d)
        self.assertTrue(all(r["want"] == "d" and r["hint"] == "d" and r["set"] == "train-g4-d" for r in rows))
        got = collections.Counter(r["form"] for r in rows)
        self.assertEqual(dict(got), SMALL["build"])
        self.assertEqual(summary["rows"], sum(SMALL["build"].values()))
        per = collections.Counter(r["group"] for r in rows)
        self.assertLessEqual(max(per.values()), 2)

    def test_same_seed_same_sha(self):
        decisions = decide_all(self.cands)
        one, _ = ld.build(self.cands, decisions, SMALL, 5)
        two, _ = ld.build(self.cands, decisions, SMALL, 5)
        self.assertEqual(json.dumps(one, ensure_ascii=False), json.dumps(two, ensure_ascii=False))

    def test_boundary_cd_dropped_not_stopped(self):
        decisions = decide_all(self.cands)
        for r in self.cands[:3]:
            decisions[r["id"]].update(label="cd")
        rows, summary = ld.build(self.cands, decisions, SMALL, 5)
        self.assertFalse({r["id"] for r in rows} & {r["id"] for r in self.cands[:3]})
        self.assertEqual(summary["boundary_dropped"], 3)

    def test_pending_stops_unless_draft(self):
        decisions = decide_all(self.cands[1:])
        with self.assertRaises(ld.Stop):
            ld.build(self.cands, decisions, SMALL, 5)
        _, summary = ld.build(self.cands, decisions, SMALL, 5, draft=True)
        self.assertEqual(summary["cand_state"]["pending"], 1)

    def test_changed_pair_with_old_id_stops(self):
        decisions = decide_all(self.cands)
        bad = [dict(self.cands[0], claim=self.cands[0]["claim"] + " 바뀜")] + self.cands[1:]
        with self.assertRaises(ld.Stop):
            ld.build(bad, decisions, SMALL, 5)


class CliTest(unittest.TestCase):
    def test_refuses_output_inside_git(self):
        for cmd in ("pick", "build"):
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(ld, "inside_git", lambda path: True), mock.patch("sys.stdout"):
                    code = ld.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                self.assertFalse(os.path.exists(os.path.join(tmp, "g4", "train-g4-d")))

    def test_refuses_overwrite(self):
        for cmd, name in (("pick", "pick.jsonl"), ("build", "cands.jsonl")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "g4", "train-g4-d")
                os.makedirs(out)
                with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                    f.write("old\n")
                with mock.patch("sys.stdout"):
                    code = ld.main([cmd, "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
                self.assertEqual(code, 2, cmd)
                with open(os.path.join(out, name), encoding="utf-8") as f:
                    self.assertEqual(f.read(), "old\n")

    def test_build_without_pick_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(ld, "_guard", lambda *args: (NoGuard(), [])), mock.patch("sys.stdout"):
                code = ld.main(["build", "--g4", os.path.join(tmp, "g4"), "--old-root", tmp])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "g4", "train-g4-d", "cands.jsonl")))

    def test_missing_frozen_sets_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stdout"):
                code = ld.main(["pick", "--g4", os.path.join(tmp, "g4"), "--old-root", tmp, "--dry-run"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "g4", "train-g4-d")))


if __name__ == "__main__":
    unittest.main()
