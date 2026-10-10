"""split_pairs — 기억 덩이 단위 가르기 · 겹침 0 · 제외."""
import collections
import json
import os
import tempfile
import unittest

import helpers  # noqa: F401

import split_pairs as sp

WANTS = ["support", "contradict", "unrelated"]


def make_rows(n_groups=60, per_group=3):
    """덩이마다 기억 둘을 쓰고, 이웃 쌍이 기억 하나를 같이 쓴다 (덩이 = 3쌍)."""
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            rows.append({"id": f"g{g}-{k}", "evidence": f"근거 {g}-{k}", "claim": f"주장 {g}-{k}",
                         "want": WANTS[(g + k) % 3], "kind": "same",
                         "src_ids": [f"m{g}-{k}", f"m{g}-{(k + 1) % per_group}"]})
    return rows


class GroupTest(unittest.TestCase):
    def test_shared_memory_makes_one_group(self):
        rows = make_rows(n_groups=4)
        groups = sp.make_groups(rows)
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(len(g) == 3 for g in groups))

    def test_src_plus_form(self):
        self.assertEqual(sp.mem_ids({"src": "a+b"}), ["a", "b"])


class SplitTest(unittest.TestCase):
    def test_no_memory_overlap(self):
        rows = make_rows()
        train, dev, seed, big = sp.split(rows, seed=1, dev_frac=0.15, band=(0.10, 0.20), min_contradict=5)
        self.assertEqual(len(train) + len(dev), len(rows))
        self.assertFalse(sp.overlap(train, dev))
        self.assertGreater(len(dev), 0)
        self.assertEqual(big, 0)

    def test_band_respected(self):
        rows = make_rows()
        _, dev, _, _ = sp.split(rows, seed=1, dev_frac=0.15, band=(0.10, 0.20), min_contradict=5)
        total = collections.Counter(r["want"] for r in rows)
        got = collections.Counter(r["want"] for r in dev)
        for want in WANTS:
            self.assertTrue(0.10 <= got[want] / total[want] <= 0.20)

    def test_letter_wants_count_as_names(self):
        # want 가 글자(a·b·c)여도 반대 수를 센다 — 이름만 세면 min_contradict 를 영영 못 맞춘다
        letters = {"support": "a", "contradict": "B", "unrelated": "c"}
        rows = [dict(r, want=letters[r["want"]]) for r in make_rows()]
        _, dev, _, _ = sp.split(rows, seed=1, dev_frac=0.15, band=(0.10, 0.20), min_contradict=5)
        self.assertGreater(len(dev), 0)
        self.assertEqual(set(sp.stats(dev)["want"]), set(WANTS))

    def test_impossible_band_stops(self):
        with self.assertRaises(SystemExit):
            sp.split(make_rows(), seed=1, band=(0.0, 0.01), min_contradict=0, max_tries=20)

    def test_exclusion_by_id_and_text(self):
        rows = make_rows(n_groups=3)
        kept, dropped = sp.drop_excluded(rows, {"m0-0"}, {("근거 2-1", "주장 2-1")})
        self.assertEqual(dropped, 3)  # m0-0 을 쓰는 쌍 둘 + 글이 같은 쌍 하나
        self.assertTrue(all("m0-0" not in sp.mem_ids(r) for r in kept))

    def test_main_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "pairs.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for r in make_rows():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out = os.path.join(tmp, "out")
            sp.main(["--in", src, "--out-dir", out, "--band", "0.10,0.20", "--min-contradict", "5"])
            for name in ("train.jsonl", "dev.jsonl", "MANIFEST.txt"):
                self.assertTrue(os.path.exists(os.path.join(out, name)))



class PinTest(unittest.TestCase):
    def test_pinned_groups_stay_in_train(self):
        rows = make_rows()
        pin = {f"m{g}-0" for g in range(0, 60, 2)}  # 짝수 덩이 30개를 묶어 둔다
        train, dev, _, _ = sp.split(rows, seed=1, dev_frac=0.15, band=(0.10, 0.20), min_contradict=5, pin_ids=pin)
        self.assertGreater(len(dev), 0)
        self.assertFalse(sp.overlap(train, dev))
        self.assertTrue(all(not pin.intersection(sp.mem_ids(r)) for r in dev))

    def test_no_pin_same_as_before(self):
        rows = make_rows()
        self.assertEqual(sp.split(rows, seed=3, band=(0.10, 0.20), min_contradict=5),
                         sp.split(rows, seed=3, band=(0.10, 0.20), min_contradict=5, pin_ids=set()))

    def test_main_pin_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "pairs.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for r in make_rows():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            pin = os.path.join(tmp, "pin.txt")
            with open(pin, "w", encoding="utf-8") as f:
                f.write("\n".join(f"m{g}-1" for g in range(30)) + "\n")
            out = os.path.join(tmp, "out")
            sp.main(["--in", src, "--out-dir", out, "--band", "0.10,0.20", "--min-contradict", "5",
                     "--pin-ids", pin])
            dev = sp.read_jsonl(os.path.join(out, "dev.jsonl"))
            self.assertTrue(dev)
            self.assertTrue(all(int(r["id"][1:].split("-")[0]) >= 30 for r in dev))


if __name__ == "__main__":
    unittest.main()
