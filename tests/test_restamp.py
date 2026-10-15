"""measure/restamp.py — 기준선 다시 찍기 (원천 균형 판 설계 5-3 · 5-5 ④). 판 폴더 · 예측은 모두 지어낸 것이다."""
import contextlib
import io
import json
import os
import tempfile
import unittest
import unittest.mock

import helpers  # noqa: F401

import compare as cmp
import restamp as rs


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


class RestampTest(unittest.TestCase):
    """runs/g1-K-s{1,2}/g4/ — eval-dev-cmp-g4/{pred, summary} · calibration · eval-side. 옛 규약 해시 OLD."""

    OLD = "a" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.root = os.path.join(self.tmp, "runs")
        self.set_rows = [{"id": "q1", "label": "c", "group": "g1"}, {"id": "q2", "label": "b", "group": "g2"}]
        self.set_path = os.path.join(self.tmp, "dev-cmp-g4.jsonl")
        write(self.set_path, "".join(json.dumps(r) + "\n" for r in self.set_rows))
        self.set_sha = cmp.sha256_file(self.set_path)
        self.runs = []
        for seed in (1, 2):
            g4 = os.path.join(self.root, f"g1-K-s{seed}", "g4")
            ev = os.path.join(g4, "eval-dev-cmp-g4")
            preds = [{"id": r["id"], "label": "abcd".index(r["label"]), "logits": [0, 0, 3, 0], "pred": "c"}
                     for r in self.set_rows]
            write(os.path.join(ev, "pred.jsonl"), "".join(json.dumps(p) + "\n" for p in preds))
            write(os.path.join(ev, "summary.json"),
                  json.dumps({"set_sha256": self.set_sha, "protocol_sha256": self.OLD, "acc3": 0.5}))
            write(os.path.join(g4, "calibration.json"), json.dumps({"temperature": 1.0, "calib_set_sha256": "cal"}))
            write(os.path.join(g4, "eval-side.txt"), "k48 63fd061c 48 0.9167 0.9167 1 {}\n")
            self.runs.append(f"g1-K-s{seed}/g4")
        self.protocol = os.path.join(self.tmp, "protocol-g6.json")
        write(self.protocol, json.dumps({"name": "protocol-g6", "compare_on": "dev-cmp-g4",
                                         "sets": {"dev-cmp-g4": self.set_sha}}))
        self.out = os.path.join(self.root, "g4-K4-eval", "restamp")

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *extra, runs=None, out=None):
        argv = ["--protocol", self.protocol, "--root", self.root, "--out", out or self.out,
                "--runs"] + (runs or self.runs) + list(extra)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return rs.main(argv)

    def test_links_and_new_hash(self):
        self.assertEqual(self.run_main(), 0)
        new_sha = cmp.sha256_file(self.protocol)
        for rel in self.runs:
            src, dst = os.path.join(self.root, rel), os.path.join(self.out, rel)
            for name in ("eval-dev-cmp-g4/pred.jsonl", "calibration.json", "eval-side.txt"):
                self.assertTrue(os.path.samefile(os.path.join(src, name), os.path.join(dst, name)), name)
            with open(os.path.join(dst, "eval-dev-cmp-g4", "summary.json"), encoding="utf-8") as f:
                got = json.load(f)
            self.assertEqual(got["protocol_sha256"], new_sha)
            self.assertEqual(got["restamp_from_protocol_sha256"], self.OLD)          # 옛 값 칸
            self.assertEqual(got["restamp_note"], rs.NOTE)                          # 까닭 칸
            self.assertEqual((got["set_sha256"], got["acc3"]), (self.set_sha, 0.5))  # 그 밖 칸은 그대로
            with open(os.path.join(src, "eval-dev-cmp-g4", "summary.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["protocol_sha256"], self.OLD)          # 원본은 안 바뀜
        with open(os.path.join(self.out, "restamp.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual((report["protocol_sha256"], report["summaries"]), (new_sha, 2))

    def test_compare_accepts_restamped_runs(self):
        # 다시 찍은 보기는 새 규약으로 해시 검사를 지나고, 원본은 거부된다
        self.assertEqual(self.run_main(), 0)
        protocol, sha = cmp.read_protocol(self.protocol)
        _, restamped = cmp.load_runs("K=" + ",".join(os.path.join(self.out, r, "eval-dev-cmp-g4") for r in self.runs))
        cmp.check_hashes(protocol, sha, self.set_sha, [("K", restamped)])
        _, original = cmp.load_runs("K=" + ",".join(os.path.join(self.root, r, "eval-dev-cmp-g4") for r in self.runs))
        with self.assertRaises(cmp.Refuse):
            cmp.check_hashes(protocol, sha, self.set_sha, [("K", original)])

    def test_refuses_out_inside_run_folder(self):
        inside = os.path.join(self.root, "g1-K-s1", "restamp")
        self.assertEqual(self.run_main(out=inside), rs.REFUSE_EXIT)
        self.assertFalse(os.path.exists(inside))
        self.assertEqual(self.run_main(out=os.path.join(self.root, "g1-K-s2", "g4", "x")), rs.REFUSE_EXIT)

    def test_refuses_overwrite_and_missing_run(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.run_main(), rs.REFUSE_EXIT)                       # 이미 있는 판 폴더
        self.assertEqual(self.run_main(runs=["g1-K-s9/g4"]), rs.REFUSE_EXIT)     # 없는 판

    def test_adds_runs_to_same_view(self):
        self.assertEqual(self.run_main(runs=self.runs[:1]), 0)
        self.assertEqual(self.run_main(runs=self.runs[1:]), 0)
        with open(os.path.join(self.out, "restamp.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual((sorted(report["runs"]), report["summaries"]), (sorted(self.runs), 2))

    def test_failed_run_still_writes_report(self):
        real = rs.link
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if "g1-K-s2" in dst:
                raise rs.Refuse("링크 실패 (지어낸 것)")
            return real(src, dst)
        with unittest.mock.patch.object(rs, "link", flaky):
            self.assertEqual(self.run_main(), rs.REFUSE_EXIT)
        with open(os.path.join(self.out, "restamp.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(sorted(report["runs"]), sorted(self.runs))
        self.assertEqual(len(report["runs"][self.runs[0]]), 4)                     # 앞 판은 다 됐다
        self.assertEqual(list(report["failed"]), [self.runs[1]])
        self.assertEqual(report["summaries"], 1)            # 둘째 판은 첫 링크(calibration.json)에서 멈췄다

    def test_follows_directory_symlink(self):
        real_dir = os.path.join(self.tmp, "elsewhere", "eval-side-dir")
        write(os.path.join(real_dir, "side.jsonl"), "{}\n")
        try:
            os.symlink(real_dir, os.path.join(self.root, self.runs[0], "side"), target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("디렉터리 심볼릭 링크를 못 만든다 (권한)")
        self.assertEqual(self.run_main(runs=self.runs[:1]), 0)
        got = os.path.join(self.out, self.runs[0], "side", "side.jsonl")
        self.assertTrue(os.path.samefile(got, os.path.join(real_dir, "side.jsonl")))
        with open(os.path.join(self.out, "restamp.json"), encoding="utf-8") as f:
            files = [e["file"] for e in json.load(f)["runs"][self.runs[0]]]
        self.assertIn("side/side.jsonl", files)

    def test_walk_follows_links(self):
        # 윈도에서 디렉터리 심볼릭 링크를 못 만들 때도 followlinks 를 켜는지는 본다
        real, seen = os.walk, []

        def spy(top, **kw):
            seen.append(kw.get("followlinks"))
            return real(top, **kw)
        with unittest.mock.patch.object(rs.os, "walk", spy):
            self.assertEqual(self.run_main(runs=self.runs[:1]), 0)
        self.assertEqual(seen, [True])

    def test_restamp_summary_keeps_order(self):
        got = rs.restamp_summary({"set_sha256": "s", "protocol_sha256": "old", "n": 3}, "new", "까닭")
        self.assertEqual(list(got), ["set_sha256", "protocol_sha256", "n", "restamp_from_protocol_sha256", "restamp_note"])
        self.assertEqual((got["protocol_sha256"], got["restamp_from_protocol_sha256"]), ("new", "old"))


if __name__ == "__main__":
    unittest.main()
