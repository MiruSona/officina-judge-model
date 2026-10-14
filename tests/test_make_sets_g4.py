"""make_sets reserve-g4 · G4Forbid · check_g4 · near_c 인자 (K2 설계 5절 ① ② ③). 글은 모두 지어낸 것이다."""
import collections
import glob
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

import make_sets as ms  # noqa: E402

SYL = "가나다라마바사아자차카타파하"
EN = ["river", "stone", "music", "garden", "engine", "winter", "market", "island", "bridge", "planet",
      "silver", "forest", "doctor", "camera", "letter", "harbor", "castle", "pepper", "rocket", "valley"]
WORDS = [a + b for a in SYL for b in SYL]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False)


def ko_text(rng, n):
    return " ".join(rng.choice(WORDS[:60]) for _ in range(n))


def ko_source(src, groups=30, a_only=10):
    """원천 하나의 판정 끝 줄 — 묶음(문서)마다 b 2 · a 2 · d 1, 같은 문단. 뒤 a_only 묶음은 a 2 · d 1 (b 없음).
    boolq 는 실물처럼 문단마다 줄 하나 — 앞 groups 묶음은 b, 뒤 a_only 묶음은 a."""
    rng = random.Random(src)
    rows = []
    for g in range(groups + a_only):
        gid = f"doc:{src}:{g}"
        ev = f"{src} {g} 번 문단은 {ko_text(rng, 9)} 이야기다."
        if src == "boolq":
            labels = "b" if g < groups else "a"
        else:
            labels = "bbaad" if g < groups else "aad"
        for k, label in enumerate(labels):
            rid = f"{src}-{g:03d}-{k}"
            rows.append({"id": rid, "evidence": ev, "claim": f"{ko_text(rng, 5)} {g}-{k} 이다.", "want": label,
                         "hint": label, "domain": f"ko-{src}", "group": gid, "groups": [gid], "lang": "ko",
                         "lic": "CC-BY-SA", "src_ids": [rid]})
    return rows


def vit_rows(pages=30):
    rng = random.Random("vit")
    rows = []
    for p in range(pages):
        gid = f"vitc:page{p}"
        ev = f"Page {p} says " + " ".join(rng.choice(EN) for _ in range(10)) + "."
        for k, (hint, form) in enumerate((("b", "labeled"), ("b", "labeled"), ("b", "labeled"), ("a", "labeled"),
                                          ("a", "labeled"), ("a", "labeled"), ("c", "same-page"), ("c", "same-page"))):
            rid = f"vitaminc-{p:03d}-{k}"
            claim = f"The {rng.choice(EN)} {rng.choice(EN)} of {p} {k} " + " ".join(rng.choice(EN) for _ in range(3))
            rows.append({"id": rid, "evidence": ev, "claim": claim, "want": "", "hint": hint, "form": form,
                         "domain": "vitaminc", "group": gid, "groups": [gid], "lang": "en", "lic": "CC-BY-SA-3.0",
                         "split": "heldout", "src_ids": [rid]})
    rows.append(dict(rows[0], id="vitaminc-train-1", split="train", group="vitc:train", groups=["vitc:train"]))
    return rows


TARGETS = {"b": {"wiki": 10, "boolq": 8, "klue": 6}, "a": {"wiki": 14, "boolq": 6, "klue": 8, "vitaminc": 15},
           "vit_pages": 12,
           "c": {"blind-g4": {"vitaminc-same": 6, "vitaminc": 6, "wiki": 5, "boolq": 5, "klue": 4},
                 "dev-cmp-g4": {"boolq": 5, "wiki": 4, "klue": 4}}}


SMALL_MIN = {"b@en": 10, "a": 30, "c": 10, "b@ko": 20}   # 가짜 뿌리 크기에 맞춘 자 (main 의 exit 0 길)


class Fixture:
    """가짜 뿌리 — g3 셋 · blind-g3 금지 · train-g3-ko · reserve-g3 · g4 판정 원천 · VitaminC."""

    def __init__(self, tmp):
        self.root = tmp
        self.g3 = os.path.join(tmp, "build", "g3")
        self.g4 = os.path.join(tmp, "build", "g4")
        self.sources = {s: ko_source(s) for s in ("wiki", "boolq", "klue")}
        for src, rel in ms.G4_INPUTS.items():
            write_jsonl(os.path.join(self.g4, rel), self.sources[src])
        self.vit_path = os.path.join(tmp, "build", "g1", "vitaminc", "candidates.jsonl")
        write_jsonl(self.vit_path, vit_rows())
        sets = os.path.join(tmp, "sets", "g3")
        wiki0 = self.sources["wiki"][0]
        self.g3_row = dict(wiki0, id="g3-old-1", set="blind-g3")              # 묶음 doc:wiki:0 은 g3 셋 몫
        write_jsonl(os.path.join(sets, "blind-g3.jsonl"), [self.g3_row])
        write_jsonl(os.path.join(sets, "dev-cmp-g3.jsonl"), [])
        write_jsonl(os.path.join(sets, "dev-cal-g3.jsonl"), [])
        write_json(os.path.join(sets, "blind-g3.forbid.json"), {"set": "blind-g3", "groups": ["doc:boolq:0"],
                                                                "src_ids": [], "pair_keys": [], "cand_ids": []})
        self.train_row = dict(self.sources["klue"][0], id="tk-1", set="train-g3-ko")   # 학습 층 묶음 doc:klue:0
        write_jsonl(os.path.join(sets, "train-g3-ko.jsonl"), [self.train_row])
        write_json(os.path.join(self.g3, "reserve", "reserve-g3.json"), {"reserved": ["vitc:page0"]})
        write_jsonl(os.path.join(self.g3, "reserve", "train-layer-cands.jsonl"), [])

    def forbid(self):
        f = ms.G4Forbid(self.root, os.path.join(self.root, "build", "g1"), os.path.join(self.root, "measure", "use1"),
                        None, os.path.join(self.root, "build", "g2"), None, self.g3)
        f.missing = []                     # g1 · g2 옛 셋은 이 시험에 없다
        return f

    def reserve(self, seed=ms.G4_SEED):
        return ms.reserve_g4(self.g4, self.forbid(), self.vit_path, seed, TARGETS)


class ReserveG4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = Fixture(cls._tmp.name)
        cls.doc, cls.rows = cls.fx.reserve()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_vitaminc_two_per_page(self):
        per = collections.Counter((r["group"], r["g4_label"]) for r in self.rows["blind-g4"]
                                  if r["g4_source"] == "vitaminc" and r.get("form") == "labeled")
        self.assertTrue(per)
        self.assertLessEqual(max(per.values()), 2)
        pages = {g for g, label in per if label == "b"}
        self.assertEqual(len(pages), TARGETS["vit_pages"])

    def test_forbidden_groups_never_reserved(self):
        for rows in self.rows.values():
            for r in rows:
                groups = set(r["groups"])
                self.assertFalse(groups & {"doc:wiki:0", "doc:boolq:0", "doc:klue:0", "vitc:page0"}, r["id"])
        self.assertGreater(self.doc["dropped_eval"]["wiki"].get("group_g3", 0), 0)
        self.assertGreater(self.doc["dropped_eval"]["boolq"].get("blind_g3_forbid", 0), 0)
        self.assertGreater(self.doc["dropped_eval"]["klue"].get("g3_train", 0), 0)

    def test_blind_first_dev_cmp_c_from_rest(self):
        blind_groups = {g for r in self.rows["blind-g4"] for g in r["groups"]}
        dev = self.rows["dev-cmp-g4"]
        self.assertEqual(len(dev), sum(TARGETS["c"]["dev-cmp-g4"].values()))
        self.assertTrue(all(r["hint"] == "c" and r["form"] == ms.NEAR_C for r in dev))
        self.assertFalse({g for r in dev for g in r["groups"]} & blind_groups)
        b_ko = collections.Counter(r["g4_source"] for r in self.rows["blind-g4"]
                                   if r["g4_label"] == "b" and r["lang"] == "ko")
        self.assertEqual(dict(b_ko), {"wiki": 10, "boolq": 8, "klue": 6})

    def test_near_rows_carry_g4_origin(self):
        near = [r for rows in self.rows.values() for r in rows if r.get("form") == ms.NEAR_C]
        self.assertTrue(near)
        self.assertEqual({(r["split"], r["origin"]) for r in near}, {("eval", "g4-eval")})

    def test_check_zero_and_same_seed_same_output(self):
        self.assertEqual(sum(ms.check_g4(self.rows, [], self.fx.forbid()).values()), 0)
        doc2, rows2 = self.fx.reserve()
        self.assertEqual(json.dumps(rows2, ensure_ascii=False), json.dumps(self.rows, ensure_ascii=False))
        self.assertEqual(json.dumps(doc2, ensure_ascii=False), json.dumps(self.doc, ensure_ascii=False))


    def test_ko_a_only_groups_fill_a_target(self):
        # 리뷰 #1(가) : b 묶음 안 a 로 모자라면 a 만 있는 묶음으로 원천별 a 목표까지 — BoolQ(문단마다 줄 하나)도
        blind = self.rows["blind-g4"]
        b_groups = {r["group"] for r in blind if r["g4_label"] == "b"}
        for source in ("wiki", "boolq", "klue"):
            a_groups = {r["group"] for r in blind if r["g4_source"] == source and r["g4_label"] == "a"}
            self.assertEqual(len(a_groups), TARGETS["a"][source], source)
            self.assertEqual(self.doc["sets"]["blind-g4"][source]["a_groups"], TARGETS["a"][source])
        boolq_a = {r["group"] for r in blind if r["g4_source"] == "boolq" and r["g4_label"] == "a"}
        self.assertFalse(boolq_a & b_groups)                 # BoolQ a 는 전부 a 만 있는 묶음에서
        wiki_a_only = {r["group"] for r in blind if r["g4_source"] == "wiki" and r["g4_label"] == "a"} - b_groups
        self.assertEqual(len(wiki_a_only), TARGETS["a"]["wiki"] - TARGETS["b"]["wiki"])

    def test_vitaminc_a_target_in_groups(self):
        # 리뷰 #1(나) : VitaminC a 묶음은 b 쪽 안 a + 안 쓴 쪽의 a 로 G4_TARGETS["a"]["vitaminc"] 까지
        a_pages = {r["group"] for r in self.rows["blind-g4"] if r["g4_source"] == "vitaminc" and r["g4_label"] == "a"}
        self.assertEqual(len(a_pages), TARGETS["a"]["vitaminc"])

    def test_c_counted_in_groups_one_per_group(self):
        # 리뷰 #2 : 같은 쪽 c 와 near-x c 가 근거 묶음을 같이 센다 — c 줄 수 = c 묶음 수, 목표 = 묶음 수
        for set_name, rows in self.rows.items():
            c_rows = [r for r in rows if r["g4_label"] == "c"]
            self.assertEqual(len(c_rows), len({r["group"] for r in c_rows}), set_name)
        blind = self.doc["sets"]["blind-g4"]
        self.assertEqual(blind["vitaminc"]["c_same_groups"], TARGETS["c"]["blind-g4"]["vitaminc-same"])
        self.assertEqual(blind["vitaminc"]["c_groups"], TARGETS["c"]["blind-g4"]["vitaminc"])
        self.assertEqual(blind["wiki"]["c_groups"], TARGETS["c"]["blind-g4"]["wiki"])
        vit_c = {r["group"] for r in self.rows["blind-g4"] if r["g4_source"] == "vitaminc" and r["g4_label"] == "c"}
        self.assertEqual(len(vit_c), TARGETS["c"]["blind-g4"]["vitaminc-same"] + TARGETS["c"]["blind-g4"]["vitaminc"])
        for source, want in TARGETS["c"]["dev-cmp-g4"].items():
            self.assertEqual(self.doc["sets"]["dev-cmp-g4"][source]["c_groups"], want, source)

    def test_ko_b_one_per_doc_counted_in_groups(self):
        # 리뷰 #6 : b 는 문서당 1 먼저 · 목표는 묶음 수
        blind = self.rows["blind-g4"]
        for source, want in TARGETS["b"].items():
            b = [r for r in blind if r["g4_source"] == source and r["g4_label"] == "b"]
            self.assertEqual(len({r["group"] for r in b}), want, source)
            self.assertEqual(len(b), want, source)

    def test_second_b_only_when_short_and_last(self):
        # 리뷰 #6 : 넘길 곳 없는 원천이 묶음 모자라면 그때만 문서당 2 로 줄 수를 채운다
        good, _ = ms._g4_good(self.fx.sources["wiki"], self.fx.forbid(), lambda r: True)
        n_b_groups = sum(1 for g in good if any(ms.g3_label(r) == "b" for r in good[g]))
        groups, rows = ms._reserve_ko_b("wiki", good, n_b_groups + 5, ms._Owner(), [], 1, second=True)
        self.assertEqual((groups, rows), (n_b_groups, n_b_groups + 5))
        groups2, rows2 = ms._reserve_ko_b("wiki", good, n_b_groups + 5, ms._Owner(), [], 1, second=False)
        self.assertEqual((groups2, rows2), (n_b_groups, n_b_groups))

    def test_near_ids_carry_pair_hash(self):
        # 리뷰 #4 : near-x id 꼬리에 쌍 sha_key 앞 8자
        near = [r for rows in self.rows.values() for r in rows if r.get("form") == ms.NEAR_C]
        self.assertTrue(near)
        for r in near:
            self.assertTrue(r["id"].endswith("-h" + ms.sha_key(r)[:8]), r["id"])


class G4ForbidTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)
        self.forbid = self.fx.forbid()

    def tearDown(self):
        self._tmp.cleanup()

    def test_g3_set_row_caught_both_roles(self):
        for role in ("eval", "train"):
            why = self.forbid.reasons(self.fx.g3_row, role)
            self.assertIn("group_g3", why)
            self.assertIn("text_g3", why)

    def test_blind_g3_forbid_row_caught(self):
        self.assertIn("blind_g3_forbid", self.forbid.reasons(self.fx.sources["boolq"][0], "train"))

    def test_train_g3_ko_only_eval(self):
        self.assertIn("g3_train", self.forbid.reasons(self.fx.train_row, "eval"))
        self.assertNotIn("g3_train", self.forbid.reasons(self.fx.train_row, "train"))

    def test_g3_paragraph_caught(self):
        row = dict(self.fx.g3_row, id="x", group="doc:other", groups=["doc:other"], claim="다른 주장이다", src_ids=[])
        self.assertIn("para_g3", self.forbid.reasons(row, "eval"))

    def test_missing_files_listed(self):
        os.remove(os.path.join(self.fx.root, "sets", "g3", "train-g3-ko.jsonl"))
        f = ms.G4Forbid(self.fx.root, "", "", None, os.path.join(self.fx.root, "build", "g2"), None, self.fx.g3)
        self.assertIn("train-g3-ko.jsonl", f.missing)

    def test_forbid_doc_shape(self):
        doc = ms.forbid_doc("blind-g4", self.fx.sources["wiki"][:3], ["x"])
        self.assertEqual(doc["set"], "blind-g4")
        self.assertEqual(len(doc["pair_keys"]), 3)


class ReserveG4MainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._tmp.name)
        self.out = os.path.join(self.fx.g4, "reserve")
        targets = os.path.join(self._tmp.name, "targets.json")
        write_json(targets, TARGETS)
        self.argv = ["reserve-g4", "--build", self.fx.g4, "--old-root", self.fx.root, "--vitaminc", self.fx.vit_path,
                     "--targets", targets]

    def tearDown(self):
        self._tmp.cleanup()

    def main(self, *extra, argv=None, min_groups=None):
        with mock.patch.object(ms, "_g4_forbid", lambda args: (self.fx.forbid(), self.fx.root)), \
                mock.patch.dict(ms.G4_MIN_GROUPS, min_groups or SMALL_MIN), mock.patch("sys.stdout"):
            return ms.main((argv or self.argv) + list(extra))

    def reserve_doc(self):
        with open(os.path.join(self.out, "reserve-g4.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_preview_short_exit1_not_raised(self):
        # 리뷰 #1(나) : 미리보기가 min_groups 아래면 목표를 안 늘리고 short 에 찍어 exit 1
        self.assertEqual(self.main(min_groups=dict(ms.G4_MIN_GROUPS)), 1)
        doc = self.reserve_doc()
        self.assertEqual(doc["short"]["a"], 380 - doc["min_groups_preview"]["a"][0])
        self.assertEqual(doc["sets"]["blind-g4"]["vitaminc"]["a_groups"], TARGETS["a"]["vitaminc"])
        self.assertTrue(doc["notice"])

    def test_vitaminc_missing_exit2(self):
        # 리뷰 #5 : VitaminC 후보가 없으면 영어 0 으로 끝내지 않고 missing exit 2
        os.remove(self.fx.vit_path)
        self.assertEqual(self.main(), 2)
        self.assertFalse(os.path.exists(os.path.join(self.out, "reserve-g4.json")))

    def test_pending_source_exit1(self):
        # 리뷰 #5 : 판정 파일이 없는 원천(pending)이 있으면 exit 1 과 알림 칸
        os.remove(os.path.join(self.fx.g4, ms.G4_INPUTS["klue"]))
        self.assertEqual(self.main(), 1)
        doc = self.reserve_doc()
        self.assertEqual(doc["pending"], {"b:klue": 6, "a:klue": 8})
        self.assertTrue(doc["notice"])

    def test_inside_git_exit2(self):
        # 리뷰 #8 : 출력 폴더가 git 안이면 쓰지 않는다
        with mock.patch.object(ms, "_inside_git", lambda path: True):
            self.assertEqual(self.main(), 2)
        self.assertFalse(os.path.exists(self.out))

    def test_build_path_expands_user(self):
        # 리뷰 #10 : --build ~/… 를 집 폴더로 푼다
        argv = list(self.argv)
        argv[argv.index("--build") + 1] = "~/build/g4"
        with mock.patch.dict(os.environ, {"USERPROFILE": self.fx.root, "HOME": self.fx.root}):
            self.assertEqual(self.main(argv=argv), 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "reserve-g4.json")))

    def test_writes_then_refuses_overwrite(self):
        self.assertEqual(self.main(), 0)
        for name in ms.G4_OUT:
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        with open(os.path.join(self.out, "blind-g4.forbid.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["set"], "blind-g4")
        with open(os.path.join(self.out, "blind-g4-cands.jsonl"), "rb") as f:
            first = f.read()
        self.assertEqual(self.main(), 2)
        with open(os.path.join(self.out, "blind-g4-cands.jsonl"), "rb") as f:
            self.assertEqual(f.read(), first)               # 거부하면 손대지 않는다

    def test_force_moves_old_files_to_date_tail(self):
        self.assertEqual(self.main(), 0)
        with open(os.path.join(self.out, "blind-g4-cands.jsonl"), "rb") as f:
            first = f.read()
        self.assertEqual(self.main("--force"), 0)
        old = glob.glob(os.path.join(self.out, "blind-g4-cands.*.jsonl"))
        self.assertEqual(len(old), 1)
        with open(old[0], "rb") as f:
            self.assertEqual(f.read(), first)
        self.assertEqual(len(glob.glob(os.path.join(self.out, "*"))), 2 * len(ms.G4_OUT))


WORDS_NEAR = ["사과", "배", "포도", "강", "산", "바다", "도시", "기차", "학교", "시장", "축구", "야구"]


def near_rows():
    rows = []
    for i in range(60):
        w = [WORDS_NEAR[(i + k) % len(WORDS_NEAR)] for k in range(4)]
        rows.append({"id": f"r{i:03d}", "evidence": f"{w[0]} {w[1]} 이야기 {i}번 문단은 {w[2]} 과 {w[3]} 을 다룬다.",
                     "claim": f"{w[1]} 는 {w[3]} 와 닿아 있다 {i}.", "domain": "kowiki", "group": f"doc:{i // 2}",
                     "groups": [f"doc:{i // 2}"], "lang": "ko", "lic": "CC-BY-SA", "src_ids": [f"s{i}"]})
    return rows


class NearCArgsTest(unittest.TestCase):
    # 손질 전 near_c 가 같은 입력에 낸 출력의 sha256 (2026-10-14 손질 직전에 잼)
    GOLDEN = "a5f19982d04d5b1c4a3c3134d36c5d33908518c2346dcd1a7137418522ad3cba"

    def test_default_bytes_same_as_before(self):
        out = ms.near_c(near_rows(), 20, random.Random(7), collections.Counter(), "nx-")
        self.assertEqual(len(out), 20)
        text = json.dumps(out, ensure_ascii=False)
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), self.GOLDEN)

    def test_split_origin_args(self):
        out = ms.near_c(near_rows(), 5, random.Random(7), collections.Counter(), "nx-", split="train",
                        origin="g4-train")
        base = ms.near_c(near_rows(), 5, random.Random(7), collections.Counter(), "nx-")
        self.assertEqual({(r["split"], r["origin"]) for r in out}, {("train", "g4-train")})
        strip = [{k: v for k, v in r.items() if k not in ("split", "origin")} for r in out]
        self.assertEqual(strip, [{k: v for k, v in r.items() if k not in ("split", "origin")} for r in base])


if __name__ == "__main__":
    unittest.main()
