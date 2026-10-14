"""layer_g3 — train-g3-ko 조립 (상한 · 씨앗 짝 규칙 · 위키 고르기 · 겹침). 입력은 모두 지어낸 짧은 글이다."""
import json
import os
import random
import sys
import tempfile
import unittest

import helpers  # noqa: F401

sys.path.insert(0, os.path.join(helpers.ROOT, "build"))

import layer_g3 as lg  # noqa: E402

CAPS = lg.TARGETS["caps"]


def row(rid, label, src, group=None, **extra):
    base = {"id": rid, "evidence": f"근거 문장 {rid} 은 이만큼 길다.", "claim": f"주장 문장 {rid} 이다.",
            "hint": label, "want": label, "group": group or f"g:{rid}", "src_ids": [rid], "_src": src, "_label": label}
    base.update(extra)
    return base


def seed_rows(pid, oks):
    """oks = {hint: 판정이 hint 와 같나}."""
    return [dict(row(f"seed-{pid}-{h}", h, "seed", group="doc:x"), pair=pid, _ok=ok) for h, ok in oks.items()]


class CapTest(unittest.TestCase):
    def test_caps_hold_and_only_shrink(self):
        want = {"wiki": 582, "boolq": 541, "seed": 280, "mit": 0, "klue": 0}
        n = lg.cap_b(want, CAPS)
        total = sum(n.values())
        self.assertLessEqual(n["wiki"], 0.40 * total)
        self.assertLessEqual(n["wiki"] + n["boolq"], 0.65 * total)
        self.assertLessEqual(n["seed"] + n["klue"], 0.25 * (767 + total))
        for s in want:
            self.assertLessEqual(n[s], want[s])
        self.assertEqual(n["seed"], 280)               # 씨앗이 묶는 자리라 줄지 않는다
        self.assertEqual(total, 800)                   # 280 / .35

    def test_synth_cap_binds_when_ko_small(self):
        n = lg.cap_b({"wiki": 0, "boolq": 0, "seed": 400, "mit": 100, "klue": 200}, CAPS)
        self.assertLessEqual(n["seed"] + n["klue"], 0.25 * (767 + sum(n.values())))
        self.assertEqual(n["mit"], 100)


class SeedUnitTest(unittest.TestCase):
    def test_whole_seed_dropped_when_one_row_off(self):
        rows = seed_rows("k1", {"b": True, "a": True, "d": True}) + seed_rows("k2", {"b": True, "a": True, "d": False}) \
            + seed_rows("k3", {"b": True}) + seed_rows("k4", {"a": True})
        units, why = lg.seed_units(rows)
        self.assertEqual([u[0]["pair"] for u in units], ["k1", "k3"])
        self.assertEqual(why, {"off_hint:d": 1, "no_b": 1})

    def test_order_puts_d_seeds_first(self):
        units, _ = lg.seed_units(seed_rows("k1", {"b": True}) + seed_rows("k2", {"b": True, "a": True, "d": True})
                                 + seed_rows("k3", {"b": True, "a": True}))
        order = [u[0]["pair"] for u in lg.order_seed_units(units, random.Random(1))]
        self.assertEqual(order, ["k2", "k3", "k1"])


class WikiPickTest(unittest.TestCase):
    def test_per_doc_and_flag_cap(self):
        rows = [row(f"w{i}", "b", "wiki", group=f"doc{i // 3}", flag=(["reverted"] if i % 2 else [])) for i in range(30)]
        got = lg.pick_per_doc(rows, 10, 2, random.Random(5), flag_cap=2)
        self.assertEqual(len(got), 10)
        self.assertLessEqual(sum(1 for r in got if r["flag"]), 2)
        groups = [r["group"] for r in got]
        self.assertTrue(all(groups.count(g) <= 2 for g in groups))
        # 문서 10 개 · 문서당 2 → flag 없는 15 + flag 5 (짝수 하나뿐인 문서) 중, flag ≤ 20% 로 고를 수 있는 최대 = 18
        self.assertEqual(lg.wiki_b_avail(rows, 2, 0.2), 18)


class AssembleTest(unittest.TestCase):
    def test_counts_caps_and_mit_fill(self):
        src = {s: [] for s in lg.SOURCES}
        src["wiki"] = [row(f"w{i}", "b", "wiki", group=f"doc{i}") for i in range(40)] + \
            [row("wd1", "d", "wiki"), row("wa1", "a", "wiki")]
        src["boolq"] = [row(f"q{i}", "b", "boolq") for i in range(40)] + [row(f"qa{i}", "a", "boolq") for i in range(5)]
        src["seed"] = [r for k in range(10) for r in seed_rows(f"k{k}", {"b": True, "a": True})]
        src["mit"] = [row(f"m{i}", "a", "mit", group=f"md{i}") for i in range(50)]
        out, summary = lg.assemble(src, lg.TARGETS, seed=7)
        n = summary["b_after_caps"]
        self.assertEqual(n["seed"], 10)
        self.assertLessEqual(n["wiki"] + n["boolq"], 0.65 * sum(n.values()))
        self.assertEqual(summary["final_total"]["b"], sum(n.values()))
        self.assertEqual(summary["final_by_source"]["boolq"].get("a"), 5)     # BoolQ a 는 가진 만큼
        a_total = summary["final_total"]["a"]
        self.assertEqual(a_total, sum(n.values()))                           # 모자란 a 를 MIT 로 채움
        self.assertTrue(all(r["set"] == "train-g3-ko" and r["want"] in "abcd" for r in out))
        self.assertTrue(all("_label" not in r and "_src" not in r for r in out))


class GuardTest(unittest.TestCase):
    def test_eval_text_group_para_dropped(self):
        ev = row("e1", "b", "wiki", group="kowiki:1")
        ev["evidence"] = "가나다라 마바사 아자차카 타파하 문장이 여기 길게 있다."
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eval.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in ev.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
            guard = lg.Guard([path])
        same_text = dict(row("t1", "b", "boolq"), evidence=ev["evidence"], claim=ev["claim"])
        same_group = row("t2", "b", "wiki", group="kowiki:1")
        clean = row("t3", "b", "wiki")
        self.assertIn("eval_text", guard.reasons(same_text))
        self.assertIn("eval_para", guard.reasons(same_text))
        self.assertEqual(guard.reasons(same_group), ["eval_group"])
        self.assertEqual(guard.reasons(clean), [])
        kept, dropped = lg.screen({"wiki": [same_group, clean], "seed": []}, guard)
        self.assertEqual([r["id"] for r in kept["wiki"]], ["t3"])
        self.assertEqual(dict(dropped), {"wiki:eval_group": 1})


class OverrideTest(unittest.TestCase):
    """wiki_pick finalize 의 override 는 judge.final 을 안 바꾼다 — 원천 읽기가 override.to 를 칸으로 쓴다."""

    def test_load_sources_uses_override(self):
        judge = {"final": "a", "override": {"from": "a", "to": "c"}}
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "reserve"))
            os.makedirs(os.path.join(tmp, "boolq"))
            files = {("reserve", "train-layer-cands.jsonl"): dict(row("w1", "c", "wiki"), g3_source="wiki-train", judge=judge),
                     ("boolq", "boolq-labeled-train.jsonl"): dict(row("q1", "c", "boolq"), judge=judge)}
            for parts, line in files.items():
                with open(os.path.join(tmp, *parts), "w", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            out, _ = lg.load_sources(tmp)
        self.assertEqual(out["wiki"][0]["_label"], "c")
        self.assertEqual(out["boolq"][0]["_label"], "c")


class DefaultEvalTest(unittest.TestCase):
    """--eval 기본값 — KLUE 학습 몫이 있으면 klue-labeled-eval 이 붙는다."""

    def test_klue_eval_added_only_with_klue_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            g3 = os.path.join(tmp, "g3")
            klue_eval = os.path.join(g3, "klue", "klue-labeled-eval.jsonl")
            self.assertNotIn(klue_eval, lg.default_eval_paths(g3, tmp))
            os.makedirs(os.path.join(g3, "klue"))
            open(os.path.join(g3, "klue", "klue-labeled-train.jsonl"), "w").close()
            paths = lg.default_eval_paths(g3, tmp)
            self.assertIn(klue_eval, paths)
            self.assertEqual(len(paths), 7)


if __name__ == "__main__":
    unittest.main()
