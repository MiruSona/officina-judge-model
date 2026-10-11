"""soup — 검사점 가중치 평균 (작은 가짜 state_dict 둘)."""
import json
import os
import tempfile
import unittest

import helpers

HAS = helpers.has("torch", "safetensors")


@unittest.skipUnless(HAS, "torch · safetensors 없음")
class SoupTest(unittest.TestCase):
    def make_ckpt(self, root, name, sd, extra=None):
        from safetensors.torch import save_file

        path = os.path.join(root, name)
        os.makedirs(path)
        save_file(sd, os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
        for fname in extra or ():
            with open(os.path.join(path, fname), "w", encoding="utf-8") as f:
                f.write(name)
        return path

    def test_mean_of_two(self):
        import torch
        from safetensors.torch import load_file

        import soup

        with tempfile.TemporaryDirectory() as root:
            a = self.make_ckpt(root, "a", {"w": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([0.0])},
                               extra=["tokenizer.json", "best.json", "state.json"])
            b = self.make_ckpt(root, "b", {"w": torch.tensor([[3.0, 6.0]]), "b": torch.tensor([4.0])})
            out = os.path.join(root, "soup")
            info = soup.soup([a, b], out)
            sd = load_file(os.path.join(out, "model.safetensors"))
            self.assertTrue(torch.equal(sd["w"], torch.tensor([[2.0, 4.0]])))
            self.assertTrue(torch.equal(sd["b"], torch.tensor([2.0])))
            files = set(os.listdir(out))
            self.assertIn("tokenizer.json", files)
            self.assertNotIn("best.json", files)
            self.assertNotIn("state.json", files)
            with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["name"], "a")  # 첫 입력 것
            with open(os.path.join(out, "soup.json"), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual([s["path"] for s in saved["sources"]], [a, b])
            self.assertEqual(saved["n"], 2)
            self.assertEqual(info["weights_sha256"], saved["weights_sha256"])

    def test_mismatch_stops(self):
        import torch

        import soup

        with tempfile.TemporaryDirectory() as root:
            a = self.make_ckpt(root, "a", {"w": torch.zeros(2, 2)})
            b = self.make_ckpt(root, "b", {"w": torch.zeros(2, 3)})
            c = self.make_ckpt(root, "c", {"v": torch.zeros(2, 2)})
            with self.assertRaises(SystemExit):
                soup.soup([a, b], os.path.join(root, "o1"))
            with self.assertRaises(SystemExit):
                soup.soup([a, c], os.path.join(root, "o2"))
            self.assertFalse(os.path.exists(os.path.join(root, "o1")))


if __name__ == "__main__":
    unittest.main()
