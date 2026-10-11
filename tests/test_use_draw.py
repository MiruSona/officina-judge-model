"""measure/use_eval.py draw · ④ ⑤ 판 얼리기 · 보고 · stop 이름 대조 (실사용측정설계 4 · 7절).
가짜 서버 · 가짜 mem 은 test_use_eval 의 것을 같이 쓴다. 쓰기는 임시 폴더에만 한다.
"""
import json
import os
import unittest

import helpers

from use_sites import common
from test_use_eval import TH, Base, probs

REPORT_ROWS = [("기본은 켠다.", "기본은 끈다."), ("로그는 지운다.", "로그는 오래 보관한다."), ("캐시를 쓴다.", "캐시를 쓴다고 했다.")]


class DrawTest(Base):
    def collect(self, alias, place, rows):
        name = "report-pairs.jsonl" if place == 4 else "harness-pairs.jsonl"
        common.write_jsonl(common.alias_dir(alias, "collect", name), rows)

    def draw(self, alias, place, *extra):
        return self.run_cli(["draw", "--place", str(place), "--repo", self.tmp, "--alias", alias, "--seen", "none",
                             "--models", self.two_models()] + list(extra))

    def test_draw_place4_ids_queries_and_set_aside(self):
        self.collect("d4", 4, [{"id": f"r{i}", "evidence": e, "claim": c, "report_at": f"2026-10-0{i + 1}"}
                               for i, (e, c) in enumerate(REPORT_ROWS)])
        code, out = self.draw("d4", 4)
        self.assertEqual(code, 0, out)
        paths = common.part_paths("d4", 4)
        cands = common.read_jsonl(paths["cands"])
        self.assertEqual(len(cands), 3)
        self.assertTrue(all(r["id"].startswith("u4-") and r["id"][3:15] == r["pair"][:12] for r in cands))
        self.assertTrue(all(r["src_id"] in ("r0", "r1", "r2") for r in cands))
        self.assertTrue(os.path.isfile(paths["queries"]))
        self.assertFalse(os.path.isfile(common.alias_dir("d4", "queries.jsonl")))  # ①~③ 기록과 섞지 않는다
        meta = common.read_json(paths["meta"])
        self.assertEqual(meta["collect"]["first"], "2026-10-01")
        hot = [r for r in cands if r["stratum"] == "sure-b"]
        self.assertTrue(hot)
        self.assertTrue(all("flag" in (common.place_outcome(r, "v110", TH), common.place_outcome(r, "v100", TH))
                            for r in hot))  # 어느 모델이든 b 확신이면 sure-b 층
        code, out = self.draw("d4", 4)
        self.assertEqual(code, 0, out)
        aside = [n for n in os.listdir(common.alias_dir("d4")) if n.startswith("cands-4.jsonl-") or n.startswith("cands-4-")]
        self.assertTrue(aside, os.listdir(common.alias_dir("d4")))  # 옛 cands-4 는 날짜 꼬리로

    def test_draw_place5_strips_tags_and_refuses_secret(self):
        self.collect("h5", 5, [{"id": "a", "evidence": "왜 : 캐시를 쓴다.", "claim": "캐시를 쓴다고 했다.",
                                "semif": {"verdict": "pass"}},
                               {"id": "b", "evidence": "서버 192.168.0.30 에 둔다.", "claim": "서버에 둔다."},
                               {"id": "c", "evidence": "x", "claim": "y", "refused": True}])
        code, out = self.draw("h5", 5)
        self.assertEqual(code, 0, out)
        paths = common.part_paths("h5", 5)
        asked = common.read_jsonl(paths["queries"])
        self.assertEqual(len(asked), 1)  # 비밀 꼴 · 이미 거른 줄은 안 묻는다
        self.assertEqual(asked[0]["pair"], common.pair_key("캐시를 쓴다.", "캐시를 쓴다고 했다."))
        meta = common.read_json(paths["meta"])
        self.assertEqual(meta["refused"]["draw-5"], 1)
        self.assertEqual(meta["pool"], 2)

    def test_draw_refuses_disabled_place(self):
        protocol, _ = common.load_protocol()
        protocol["places"]["4-report"]["enabled"] = False
        path = os.path.join(self.tmp, "protocol-off.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(protocol, f, ensure_ascii=False)
        self.collect("d0", 4, [{"id": "r0", "evidence": "가.", "claim": "나."}])
        code, out = self.draw("d0", 4, "--protocol", path)
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn("꺼져 있다", out)

    def test_freeze_place5_adds_queries_and_blocks_draw(self):
        self.collect("h6", 5, [{"id": "a", "evidence": e, "claim": c} for e, c in REPORT_ROWS])
        self.assertEqual(self.draw("h6", 5)[0], 0)
        cands = common.read_jsonl(common.part_paths("h6", 5)["cands"])
        lab = os.path.join(self.tmp, "lab.jsonl")
        common.write_jsonl(lab, [{"id": r["id"], "label": "a"} for r in cands])
        code, out = self.run_cli(["freeze", "--alias", "h6", "--place", "5", "--labels", lab])
        self.assertEqual(code, 0, out)
        with open(common.shared_path("EXCLUDE.sha256list"), encoding="utf-8") as f:
            exclude = f.read().split()
        asked = {r["pair"] for r in common.read_jsonl(common.part_paths("h6", 5)["queries"])}
        self.assertTrue(asked <= set(exclude))
        self.assertTrue(os.path.isfile(common.alias_dir("h6", "FROZEN-use1-h6-5.sha256")))
        code, out = self.draw("h6", 5)
        self.assertEqual(code, common.REFUSE_EXIT, out)  # 얼린 뒤 다시 안 뽑는다


def place5_rows():
    """⑤ 5쌍 — 정답 a a b c d. v110 통과 · 보류 · 거절 · 통과 · 보류, v100 통과 · 통과 · 통과 · 거절 · 거절."""
    sure_a, mid, sure_c = probs(.9, .05, .05), probs(.6, .3, .1), probs(.05, .05, .9)
    new = [sure_a, mid, sure_c, sure_a, mid]
    base = [sure_a, sure_a, sure_a, sure_c, sure_c]
    semif = ["pass", "unsure", "review", "review", "unsure"]
    return [{"place": 5, "stratum": "rest", "asked": True, "answers": {"v110": new[i], "v100": base[i]},
             "semif": {"verdict": semif[i]}, "label_fake": g} for i, g in enumerate("aabcd")]


class Place5ReportTest(Base):
    def test_report_place5_by_hand(self):
        lab = self.make_set("h7", place5_rows(), place=5)
        code, out = self.run_cli(["freeze", "--alias", "h7", "--place", "5", "--labels", lab])
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["report", "--alias", "h7", "--place", "5"])
        self.assertEqual(code, 0, out)
        res = common.read_json(common.alias_dir("h7", "result-5.json"))["results"][0]
        cells = res["places"]["5"]["rest"]["all"]
        self.assertAlmostEqual(cells["sure_false_support"]["v110"]["value"], 0.3333, places=4)
        self.assertAlmostEqual(cells["sure_false_support"]["v100"]["value"], 0.3333, places=4)
        self.assertEqual(cells["false_reject"]["v110"]["value"], 0.0)
        self.assertEqual(cells["false_reject"]["v100"]["value"], 0.0)
        self.assertEqual(cells["unsure"]["v110"]["value"], 0.4)
        self.assertEqual(cells["unsure"]["v100"]["value"], 0.0)
        self.assertEqual(res["collected"]["v110"]["semif_agree"], 0.8)
        self.assertEqual(res["collected"]["v100"]["semif_agree"], 0.4)
        with open(common.alias_dir("h7", "result-5.md"), encoding="utf-8") as f:
            self.assertIn("⑤", f.read())


class StopTest(Base):
    def test_stop_refuses_other_server(self):
        url = self.serve("nli-studio-v4", lambda c: probs(.9, .05, .05))
        port = url.rsplit(":", 1)[1]
        common.write_json(common.shared_path("servers.json"), {port: {"pid": 2 ** 30, "name": "nli-studio-v2"}})
        code, out = self.run_cli(["stop", "--port", port])
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn(port, common.read_json(common.shared_path("servers.json")))  # 기록도 그대로


_ = helpers
if __name__ == "__main__":
    unittest.main()
