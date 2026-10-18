"""train_nli --smoke — 무작위 초기화 작은 모델로 저장 · 재개 · 평가 · 기록이 끝까지 도는지 (torch 가 있을 때만)."""
import os
import subprocess
import sys
import tempfile
import unittest

import helpers

HAS_TRAIN = helpers.has("torch", "transformers", "tokenizers")


@unittest.skipUnless(HAS_TRAIN, "torch · transformers · tokenizers 없음")
class SmokeTest(unittest.TestCase):
    def test_smoke_runs_to_end(self):
        script = os.path.join(helpers.ROOT, "train", "train_nli.py")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", script, "--smoke", "--out", tmp, "--threads", "2"],
                                  capture_output=True, text=True, encoding="utf-8", timeout=600)
            tail = (proc.stdout + proc.stderr)[-2000:]
            self.assertEqual(proc.returncode, 0, tail)
            self.assertIn("SMOKE OK", proc.stdout, tail)

    def test_smoke_select_last_writes_best_at_end(self):
        """last 는 바퀴 중엔 ckpt-best 를 안 쓰고 끝에 마지막 가중치를 한 번 낸다 — 멈췄다 재개해도 best 가 끝 스텝."""
        import json

        script = os.path.join(helpers.ROOT, "train", "train_nli.py")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", script, "--smoke", "--out", tmp, "--threads", "2",
                                   "--select", "last"],
                                  capture_output=True, text=True, encoding="utf-8", timeout=600)
            tail = (proc.stdout + proc.stderr)[-2000:]
            self.assertEqual(proc.returncode, 0, tail)
            with open(os.path.join(tmp, "train_info.json"), encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(info["hparams"]["select"], "last")
            self.assertEqual(info["best"]["step"], info["steps"])
            with open(os.path.join(tmp, "ckpt-best", "best.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["step"], info["steps"])

    def test_smoke_save_every_epoch_keeps_each_epoch(self):
        """--save-every-epoch 는 바퀴 끝마다 ckpt-e<N> 을 남긴다 — 마지막 바퀴 것은 끝 스텝."""
        import json

        script = os.path.join(helpers.ROOT, "train", "train_nli.py")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", script, "--smoke", "--out", tmp, "--threads", "2",
                                   "--select", "last", "--save-every-epoch"],
                                  capture_output=True, text=True, encoding="utf-8", timeout=600)
            tail = (proc.stdout + proc.stderr)[-2000:]
            self.assertEqual(proc.returncode, 0, tail)
            with open(os.path.join(tmp, "train_info.json"), encoding="utf-8") as f:
                info = json.load(f)
            epochs = info["hparams"]["epochs"]
            for n in range(1, epochs + 1):
                with open(os.path.join(tmp, f"ckpt-e{n}", "best.json"), encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["epoch"], n)
            with open(os.path.join(tmp, f"ckpt-e{epochs}", "best.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["step"], info["steps"])


class TrainHelpersTest(unittest.TestCase):
    """train_nli 가 heads 에서 가져다 쓰는 작은 셈 — torch 없이 돈다."""

    def test_min_recall_all_none_is_zero(self):
        """--select studio : 갈래 재현이 다 None(쌍 없는 dev)이어도 죽지 않고 0 으로 견준다."""
        import heads

        self.assertEqual(heads.min_recall({"a": None, "b": None, "c": None}), 0)
        self.assertEqual(heads.min_recall({"a": 0.7, "b": None, "c": 0.9}), 0.7)

    def test_epoch_end_metrics_falls_back_to_history(self):
        """--save-every-epoch : 바퀴 끝 run_eval 이 None(같은 스텝에서 이미 잼)이면 dev_history 마지막 줄을 쓴다."""
        import heads

        history = [{"epoch": 1, "step": 4, "acc": 0.5, "loss": 0.9}, {"epoch": 2, "step": 8, "acc": 0.75, "loss": 0.6}]
        self.assertEqual(heads.epoch_end_metrics(None, history), {"acc": 0.75, "loss": 0.6})
        fresh = {"acc": 0.8}
        self.assertIs(heads.epoch_end_metrics(fresh, history), fresh)
        self.assertIsNone(heads.epoch_end_metrics(None, []))


@unittest.skipUnless(HAS_TRAIN, "torch · transformers · tokenizers 없음")
class TrainPiecesTest(unittest.TestCase):
    def setUp(self):
        import train_nli

        self.t = train_nli

    def test_class_weights_mean_one_on_mem_only(self):
        rows = [{"src": "mem", "label": 0}] * 6 + [{"src": "mem", "label": 1}] * 2 + \
               [{"src": "mem", "label": 2}] * 4 + [{"src": "klue", "label": 1}] * 100
        w = self.t.class_weights(rows)
        self.assertAlmostEqual(sum(w) / 3, 1.0)
        self.assertGreater(w[1], w[2])
        self.assertGreater(w[2], w[0])

    def test_batches_repeat_with_seed_and_cover_all(self):
        items = [([1] * (i % 7 + 1), 0, 1.0, "") for i in range(50)]
        a = self.t.make_batches(items, 8, 1, 0)
        b = self.t.make_batches(items, 8, 1, 0)
        c = self.t.make_batches(items, 8, 1, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(sorted(i for batch in a for i in batch), list(range(50)))

    def test_metrics_false_support(self):
        logits = [[3, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]]
        m = self.t.metrics_from(logits, [0, 1, 1, 2], ["long", "", "", ""])
        self.assertEqual(m["false_support"], 1)
        self.assertEqual(m["acc"], 0.75)
        self.assertEqual(m["support_recall_by_kind"], {"long": 1.0})

    def test_select_studio_prefers_fewer_false_support(self):
        old = {"acc": 0.9, "false_support": 3, "recall": {"a": 0.9, "b": 0.9, "c": 0.9}}
        new = {"acc": 0.8, "false_support": 1, "recall": {"a": 0.7, "b": 0.8, "c": 0.8}}
        self.assertTrue(self.t.better("studio", new, old))
        self.assertFalse(self.t.better("base", new, old))

    def test_select_last_always_takes_newest(self):
        old = {"acc": 0.9, "false_support": 1, "recall": {"a": 0.9, "b": 0.9, "c": 0.9}}
        new = {"acc": 0.7, "false_support": 5, "recall": {"a": 0.6, "b": 0.6, "c": 0.6}}
        self.assertTrue(self.t.better("last", new, old))
        self.assertFalse(self.t.better("studio", new, old))

    def test_select_default_follows_stage(self):
        self.assertEqual(self.t.parse_args(["--out", "x", "--stage", "studio"]).select, "last")
        self.assertEqual(self.t.parse_args(["--out", "x"]).select, "base")
        self.assertEqual(self.t.parse_args(["--out", "x", "--stage", "studio", "--select", "studio"]).select, "studio")

    def test_class_weight_fixed_args(self):
        # --class-weight-fixed 는 네 값을 그대로 받고, --class-weight · --class-weight-scale 과 같이 오면 멈춘다
        fixed = "0.8646582405778421,0.9994239425532274,0.8340563960214578,1.3018614208474724"
        args = self.t.parse_args(["--out", "x", "--class-weight-fixed", fixed])
        self.assertEqual(args.weight_fixed, [float(v) for v in fixed.split(",")])
        self.assertIsNone(self.t.parse_args(["--out", "x"]).weight_fixed)
        for extra in (["--class-weight"], ["--class-weight-scale", "b=1.2"], ["--class-weight", "--class-weight-scale", "b=1.2"]):
            with self.assertRaises(SystemExit, msg=extra):
                self.t.parse_args(["--out", "x", "--class-weight-fixed", fixed] + extra)


if __name__ == "__main__":
    unittest.main()
