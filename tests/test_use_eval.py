"""measure/use_eval.py — 대리 서버 · 확신 가르기 · sample(가짜 mem) · 대조 멈춤 · 얼리기 · EXCLUDE · 보고 꼴 (실사용측정설계 3 · 8절).
가짜 서버 둘(고정 확률)과 가짜 mem(tests/fake_mem.py)만 쓴다. 쓰기는 임시 폴더(JUDGEMODEL_USE1_HOME)에만 한다.
"""
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import helpers

import use_eval
from use_sites import common, labels, proxy, report

HERE = os.path.dirname(os.path.abspath(__file__))
TH = {"nli_sure": 0.82, "nli_sure_support": 0.85}
FAKE_ENV = ("FAKE_MEM_DROP_MERGE", "FAKE_MEM_EXTRA_C19", "FAKE_MEM_NLI")
ID_RE = re.compile(r"^u[1-5]-[0-9a-f]{12}(-\d+)?$")


class FakeHandler(BaseHTTPRequestHandler):
    """가짜 NLI 서버. self.server.name · letters · decide(claim → 확률) 를 쓴다."""

    def log_message(self, *args):
        return

    def reply(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        name = self.server.name
        self.reply({"ok": True, "name": name, "model": name[-2:] + "000000", "letters": list(self.server.letters)})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.reply(dict(self.server.decide(body["claim"]), model=self.server.name[-2:] + "000000", ms=1.0))


def fake_server(name, decide, letters=("a", "b", "c"), port=0):
    """decide(claim) → 확률 dict. /health 는 name · letters 를 낸다. (서버, 주소)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), FakeHandler)
    server.name, server.letters, server.decide = name, letters, decide
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def v110_probs(claim):
    if "끈다" in claim:
        return {"a": 0.05, "b": 0.9, "c": 0.05}
    return {"a": 0.05, "b": 0.05, "c": 0.9}


def v100_probs(claim):
    if "보관" in claim or "끈다" in claim:
        return {"a": 0.03, "b": 0.92, "c": 0.05}
    return {"a": 0.9, "b": 0.05, "c": 0.05}


def probs(a, b, c):
    return {"a": a, "b": b, "c": c}


EXPECT = {"v100": {"name": "nli-studio-v2", "letters": "abc"}, "v110": {"name": "nli-studio-v4", "letters": "abc"}}
MODELS_OK = {"v100": {"ok": True, "name": "nli-studio-v2", "model": "aaaa"},
             "v110": {"ok": True, "name": "nli-studio-v4", "model": "bbbb"}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = {k: os.environ.get(k) for k in (common.HOME_ENV, common.MEM_ENV) + FAKE_ENV}
        os.environ[common.HOME_ENV] = os.path.join(self.tmp, "use1")
        os.environ[common.MEM_ENV] = os.path.join(HERE, "fake_mem.py")
        for k in FAKE_ENV:
            os.environ.pop(k, None)
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def serve(self, name, decide, letters=("a", "b", "c")):
        server, url = fake_server(name, decide, letters)
        self.servers.append(server)
        return url

    def two_models(self):
        return f"v100={self.serve('nli-studio-v2', v100_probs)},v110={self.serve('nli-studio-v4', v110_probs)}"

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = use_eval.main(argv)
        return code, out.getvalue() + err.getvalue()

    def make_set(self, alias, rows, place=None, models=None):
        """얼릴 셋을 손으로 — cands · 표본 기록(sample.json 또는 draw-N.json) · 손 라벨 파일. 손 라벨 경로를 돌려준다."""
        paths = common.part_paths(alias, place)
        taken = set()
        for r in rows:
            r.setdefault("evidence", f"근거 {len(taken)}")
            r.setdefault("claim", f"주장 {len(taken)}")
            r["pair"] = common.pair_key(r["evidence"], r["claim"])
            r["group"] = common.group_key(r["evidence"])
            r["id"] = common.pair_id(r["place"], r["pair"], taken)
            r.setdefault("weight", 1.0)
            r.setdefault("stratum", "s")
        common.write_jsonl(paths["cands"], [{k: v for k, v in r.items() if k != "label_fake"} for r in rows])
        _, sha = common.load_protocol()
        common.write_json(paths["meta"], {"protocol_sha256": sha, "thresholds": {"repo": TH}, "memories": 4,
                                          "models": models or MODELS_OK, "pool": len(rows),
                                          "queues": {"m0": {"contradict": 0, "c19": 0, "c19_nli": 0, "merge": 2, "nli": None}}
                                          if place is None else None})
        lab = os.path.join(self.tmp, f"labels-{alias}-{place}.jsonl")
        common.write_jsonl(lab, [{"id": r["id"], "label": r["label_fake"]} for r in rows])
        return lab


class JudgeAnswerTest(unittest.TestCase):
    def test_sure_lines(self):
        # 지지는 .85 이상만 확신, 반대 · 무관은 .82 이상
        self.assertEqual(common.judge_answer({"a": 0.84, "b": 0.1, "c": 0.06}, TH), ("a", False))
        self.assertEqual(common.judge_answer({"a": 0.86, "b": 0.1, "c": 0.04}, TH), ("a", True))
        self.assertEqual(common.judge_answer({"a": 0.1, "b": 0.83, "c": 0.07}, TH), ("b", True))
        self.assertEqual(common.judge_answer({"unavailable": True}, TH), (None, False))

    def test_unsure_rule_from_protocol(self):
        # 규약의 unsure_rule 을 넘기면 그 값으로 가른다 (기본 below .40 이면 확신 선이 이긴다)
        low = {"nli_sure": 0.3, "nli_sure_support": 0.3}
        answer = {"a": 0.45, "b": 0.30, "c": 0.25}
        self.assertEqual(common.judge_answer(answer, low), ("a", True))
        self.assertEqual(common.judge_answer(answer, low, {"below": 0.5, "tie_gap": 0.01}), ("a", False))
        protocol, _ = common.load_protocol()
        self.assertEqual(common.unsure_rule_of(protocol), {"below": 0.40, "tie_gap": 0.01})

    def test_place_outcome(self):
        row = {"place": 1, "rule": "b", "answers": {}}
        self.assertEqual(common.place_outcome(row, "v110", TH), "reject")  # retain 은 규칙 B 만 최종
        row = {"place": 1, "rule": "c", "answers": {"v110": {"a": 0.9, "b": 0.05, "c": 0.05}}}
        self.assertEqual(common.place_outcome(row, "v110", TH), "pass")
        row = {"place": 3, "asked": False, "answers": {}}
        self.assertEqual(common.place_outcome(row, "v110", TH), "keep")
        row = {"place": 2, "asked": True, "answers": {"v100": {"unavailable": True}}}
        self.assertIsNone(common.place_outcome(row, "v100", TH))
        # 물었는데 답이 없으면 none 이 아니라 None (셈에서 빠짐)
        self.assertIsNone(common.place_outcome({"place": 2, "asked": True, "answers": {}}, "v110", TH))

    def test_place_outcome_4_5(self):
        self.assertEqual(common.place_outcome({"place": 4, "rule": "b", "answers": {}}, "v110", TH), "flag")
        self.assertEqual(common.place_outcome({"place": 4, "answers": {"v110": probs(.05, .9, .05)}}, "v110", TH), "flag")
        self.assertEqual(common.place_outcome({"place": 4, "answers": {"v110": probs(.05, .5, .45)}}, "v110", TH), "none")
        self.assertEqual(common.place_outcome({"place": 5, "answers": {"v110": probs(.9, .05, .05)}}, "v110", TH), "pass")
        self.assertEqual(common.place_outcome({"place": 5, "answers": {"v110": probs(.6, .3, .1)}}, "v110", TH), "unsure")
        self.assertEqual(common.place_outcome({"place": 5, "answers": {"v110": probs(.05, .05, .9)}}, "v110", TH), "reject")

    def test_alias_jail(self):
        for bad in ("..", "a/b", "", "x" * 41, "-a"):
            with self.assertRaises(common.Refuse):
                common.alias_dir(bad)

    def test_pair_id_is_hash(self):
        taken = set()
        key = common.pair_key("가", "나")
        self.assertEqual(common.pair_id(2, key, taken), "u2-" + key[:12])
        self.assertEqual(common.pair_id(2, key, taken), "u2-" + key[:12] + "-2")  # 같은 쌍이 겹치면 꼬리


class SecretTest(unittest.TestCase):
    """비밀 거름 한 벌 — ④ reports · ⑤ harness 가 같이 쓴다 (두 모듈 규칙의 합집합)."""

    def test_kinds(self):
        self.assertEqual(common.secret_kind("키는 sk-ABCDEFGHIJKLMNOPQRSTUV"), "key")
        self.assertEqual(common.secret_kind("api_key = abcdef123"), "assign")
        self.assertEqual(common.secret_kind('password = "hunter22"'), "assign")
        self.assertEqual(common.secret_kind("서버 192.168.0.30 에 둔다"), "ip")
        self.assertEqual(common.secret_kind(r"C:\Users\someone\.judgemodel 아래"), "home")
        self.assertEqual(common.secret_kind("/home/someone/models"), "home")
        self.assertEqual(common.secret_kind("/Users/someone/.ssh"), "home")
        self.assertEqual(common.secret_kind("-----BEGIN RSA PRIVATE KEY-----"), "private-key")
        self.assertEqual(common.secret_kind("Authorization: Bearer abcdefgh12345678"), "bearer")
        self.assertEqual(common.secret_kind("메일 someone@example.com 로"), "email")

    def test_not_secret(self):
        self.assertIsNone(common.secret_kind("127.0.0.1:8092 로 묻는다 · ~/.judgemodel 아래"))
        self.assertIsNone(common.secret_kind("127.5.5.5 와 0.0.0.0 에 붙는다"))
        self.assertIsNone(common.secret_kind("커밋 77938b1 · sha256 " + "a" * 64))
        self.assertIsNone(common.secret_kind("key = strings.TrimSpace(key)"))
        self.assertIsNone(common.secret_kind("~/.localharness/runs 를 읽는다"))


class ProxyTest(Base):
    def post(self, url, evidence, claim):
        body = json.dumps({"evidence": evidence, "claim": claim}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url + "/judge", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read()

    def test_forwards_both_and_replies_bytes(self):
        v100 = self.serve("nli-studio-v2", v100_probs)
        v110 = self.serve("nli-studio-v4", v110_probs)
        log = os.path.join(self.tmp, "q.jsonl")
        p = proxy.Proxy({"v100": v100, "v110": v110}, "v110", log, "t", EXPECT)
        p.probe()
        url = p.start()
        try:
            got = self.post(url, "왜 : 기본은 켠다.", "기본은 끈다.")
            direct = urllib.request.urlopen(urllib.request.Request(
                v110 + "/judge", data=json.dumps({"evidence": "왜 : 기본은 켠다.", "claim": "기본은 끈다."},
                                                 ensure_ascii=False).encode("utf-8")), timeout=5).read()
        finally:
            p.stop()
        self.assertEqual(got, direct)  # reply 모델 답을 바이트 그대로
        rows = common.read_jsonl(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answers"]["v100"]["b"], 0.92)
        self.assertEqual(rows[0]["answers"]["v110"]["b"], 0.9)
        self.assertEqual(rows[0]["pair"], common.pair_key("기본은 켠다.", "기본은 끈다."))  # 꼬리표 뗀 열쇠
        for key in ("at", "alias", "place", "run"):
            self.assertIn(key, rows[0])

    def test_missing_or_wrong_model_is_unavailable(self):
        v110 = self.serve("nli-studio-v4", v110_probs)
        wrong = self.serve("nli-studio-v3", v110_probs, ("a", "b", "c", "d"))
        log = os.path.join(self.tmp, "q.jsonl")
        p = proxy.Proxy({"v100": "http://127.0.0.1:9", "v110": v110, "v3": wrong}, "v110", log, "t",
                        dict(EXPECT, v3=EXPECT["v100"]))
        status = p.probe()
        self.assertFalse(status["v100"]["ok"])
        self.assertFalse(status["v3"]["ok"])
        answers, data = p.ask_all("가", "나")
        self.assertEqual(answers["v100"], {"unavailable": True})
        self.assertIsNotNone(data)

    def test_bad_sum_is_error(self):
        bad = self.serve("nli-studio-v2", lambda c: {"a": 0.5, "b": 0.5, "c": 0.5})
        v110 = self.serve("nli-studio-v4", v110_probs)
        p = proxy.Proxy({"v100": bad, "v110": v110}, "v110", os.path.join(self.tmp, "q.jsonl"), "t", EXPECT)
        p.probe()
        answers, _ = p.ask_all("가", "나")
        self.assertEqual(answers["v100"]["error"], "bad-sum")

    def test_reply_down_refuses(self):
        p = proxy.Proxy({"v110": "http://127.0.0.1:9"}, "v110", os.path.join(self.tmp, "q.jsonl"), "t", EXPECT)
        with self.assertRaises(common.Refuse):
            p.probe()


class SampleTest(Base):
    def sample(self, alias, *extra):
        return self.run_cli(["sample", "--repo", self.tmp, "--alias", alias, "--seen", "none",
                             "--models", self.two_models()] + list(extra))

    def test_sample_with_fake_mem(self):
        code, out = self.sample("t1")
        self.assertEqual(code, 0, out)
        cands = common.read_jsonl(common.alias_dir("t1", "cands.jsonl"))
        by = {}
        for r in cands:
            by.setdefault((r["place"], r["stratum"]), []).append(r)
        # ② : 두 쌍 다 어느 모델이든 b 확신 → flagged 2 · rest 0, 보탬 1
        self.assertEqual(len(by[(2, "flagged")]), 2)
        self.assertEqual(len(by[(2, "plus-superseded")]), 1)
        # ③ : M0 merge 2 짝 — v110 판에서 m1·m2 가 숨고, v100 판에서는 둘 다 숨는다
        m0 = {tuple(r["mem_ids"]): r for r in by[(3, "m0")]}
        self.assertEqual(sorted(m0[("m1", "m2")]["hidden_in"]), ["v100", "v110"])
        self.assertEqual(m0[("m3", "m4")]["hidden_in"], ["v100"])
        self.assertEqual(m0[("m3", "m4")]["src"], "c19")
        self.assertEqual(m0[("m3", "m4")]["claim"], "로그는 오래 보관한다.")  # 기억 id 짝의 C19 까닭 글 문장
        # ① : 다시 지은 쌍에 규칙 글자 · 두 모델 답
        retain = [r for r in cands if r["place"] == 1]
        self.assertTrue(retain)
        self.assertTrue(all("v100" in r["answers"] and "v110" in r["answers"] for r in retain))
        # id 는 자리 + 쌍 열쇠 앞 12자, 겹침 없음
        self.assertTrue(all(ID_RE.match(r["id"]) and r["id"][3:15] == r["pair"][:12] for r in cands))
        self.assertEqual(len({r["id"] for r in cands}), len(cands))
        manifest = common.read_json(common.alias_dir("t1", "sample.json"))
        self.assertEqual(manifest["reconcile"]["v110"]["mem_nli_contradict"], 1)
        self.assertEqual(sum(manifest["no_answer"].values()), 0)
        self.assertIn("judge_dir_files_after", manifest)
        toml = read_text(common.alias_dir("t1", "llm-repo.toml"))
        self.assertNotIn("key", toml)  # 키를 옮기지 않는다
        self.assertNotIn("\nurl", toml)

    def test_resample_stops_on_labels_and_ids_stay(self):
        self.assertEqual(self.sample("t3")[0], 0)
        first = {r["id"]: r["pair"] for r in common.read_jsonl(common.alias_dir("t3", "cands.jsonl"))}
        common.write_jsonl(common.alias_dir("t3", "labels", "decided.jsonl"), [{"id": "x", "label": "a"}])
        code, out = self.sample("t3")
        self.assertEqual(code, common.REFUSE_EXIT, out)  # 라벨이 있으면 다시 안 뽑는다
        code, out = self.sample("t3", "--force")
        self.assertEqual(code, 0, out)
        self.assertFalse(os.path.isdir(common.alias_dir("t3", "labels")))
        aside = [n for n in os.listdir(common.alias_dir("t3")) if n.startswith("labels-")]
        self.assertEqual(len(aside), 1)  # 옛 labels 는 날짜 꼬리로 돌려 뒀다
        second = {r["id"]: r["pair"] for r in common.read_jsonl(common.alias_dir("t3", "cands.jsonl"))}
        self.assertEqual(first, second)  # 같은 쌍은 같은 id

    def test_v100_down_is_unavailable_not_fatal(self):
        v110 = self.serve("nli-studio-v4", v110_probs)
        code, out = self.run_cli(["sample", "--repo", self.tmp, "--alias", "t2", "--seen", "none", "--places", "2",
                                  "--models", f"v100=http://127.0.0.1:9,v110={v110}"])
        self.assertEqual(code, 0, out)
        rows = common.read_jsonl(common.alias_dir("t2", "queries.jsonl"))
        self.assertTrue(rows)
        self.assertTrue(all(r["answers"]["v100"] == {"unavailable": True} for r in rows))

    def test_c19_reconcile_mismatch_stops(self):
        os.environ["FAKE_MEM_EXTRA_C19"] = "1"
        code, out = self.sample("t4")
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn("대조 어긋남", out)

    def test_merge_hidden_mismatch_stops(self):
        # mem 이 merge 를 숨겼는데 기록 답으로는 숨길 까닭이 없다 → 멈춘다
        os.environ["FAKE_MEM_DROP_MERGE"] = "1"
        code, out = self.sample("t5", "--places", "3")
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn("③ 대조 어긋남", out)

    def test_no_answer_is_counted_not_none(self):
        os.environ["FAKE_MEM_NLI"] = "skip"
        code, out = self.sample("t6", "--places", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("경고", out)
        manifest = common.read_json(common.alias_dir("t6", "sample.json"))
        self.assertGreater(manifest["no_answer"]["retain"], 0)
        retain = common.read_jsonl(common.alias_dir("t6", "cands.jsonl"))
        unanswered = [r for r in retain if not r["answers"] and r.get("rule") != "b"]
        self.assertTrue(unanswered)
        self.assertTrue(all(common.place_outcome(r, "v110", TH) is None for r in unanswered))

    def test_mem_exit2_stops_but_secret_is_refused(self):
        os.environ["FAKE_MEM_NLI"] = "fail"
        code, out = self.sample("t7", "--places", "1")
        self.assertEqual(code, common.REFUSE_EXIT, out)
        os.environ["FAKE_MEM_NLI"] = "secret"
        code, out = self.sample("t8", "--places", "1")
        self.assertEqual(code, 0, out)
        manifest = common.read_json(common.alias_dir("t8", "sample.json"))
        self.assertGreater(manifest["refused"]["retain"], 0)
        self.assertEqual(manifest["no_answer"]["retain"], 0)


def fake_set_rows():
    """② flagged 10쌍. 정답 b 6 · d 2 · c 2. v110 은 b 6 + d 1 을 올리고, v100 은 b 4 + d 2 + c 1 을 올린다."""
    rows = []
    golds = ["b"] * 6 + ["d"] * 2 + ["c"] * 2
    flag_new = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    flag_base = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    for i, g in enumerate(golds):
        ans = lambda f: {"a": 0.04, "b": 0.92, "c": 0.04} if f else {"a": 0.04, "b": 0.04, "c": 0.92}
        rows.append({"place": 2, "stratum": "flagged", "evidence": f"근거 {i}", "claim": f"주장 {i}",
                     "answers": {"v100": ans(flag_base[i]), "v110": ans(flag_new[i])}, "asked": True,
                     "seen": i == 0, "label_fake": g})
    return rows


def place1_rows():
    """① 4쌍 — r1 a(둘 다 통과) · r2 b 규칙 B(둘 다 거절) · r3 c(v110 통과 · v100 거절) · r4 a(v110 거절 · v100 보류)."""
    return [{"place": 1, "stratum": "replay-hard", "answers": {"v110": probs(.9, .05, .05), "v100": probs(.9, .05, .05)},
             "label_fake": "a"},
            {"place": 1, "stratum": "replay-hard", "rule": "b", "answers": {"v110": probs(.9, .05, .05),
                                                                            "v100": probs(.9, .05, .05)}, "label_fake": "b"},
            {"place": 1, "stratum": "replay-hard", "answers": {"v110": probs(.9, .05, .05), "v100": probs(.05, .05, .9)},
             "label_fake": "c"},
            {"place": 1, "stratum": "replay-hard", "answers": {"v110": probs(.05, .9, .05), "v100": probs(.5, .3, .2)},
             "label_fake": "a"}]


def place3_rows():
    """③ 3쌍 — r1 d(v110 숨김 · v100 둠) · r2 b(둘 다 숨김) · r3 a(물은 적 없음 · 둘 다 둠)."""
    return [{"place": 3, "stratum": "m0", "asked": True, "answers": {"v110": probs(.05, .9, .05), "v100": probs(.05, .05, .9)},
             "label_fake": "d"},
            {"place": 3, "stratum": "m0", "asked": True, "answers": {"v110": probs(.05, .9, .05), "v100": probs(.05, .9, .05)},
             "label_fake": "b"},
            {"place": 3, "stratum": "m0", "asked": False, "answers": {}, "label_fake": "a"}]


class FreezeReportTest(Base):
    def test_freeze_once_and_exclude(self):
        lab = self.make_set("f1", fake_set_rows())
        code, out = self.run_cli(["freeze", "--alias", "f1", "--labels", lab])
        self.assertEqual(code, 0, out)
        meta = common.read_json(common.alias_dir("f1", "frozen.json"))
        self.assertIn("frozen_at", meta)
        mark = read_text(common.alias_dir("f1", "FROZEN-use1-f1.sha256"))
        self.assertTrue(mark.startswith(meta["set_sha256"]))
        exclude = read_text(common.shared_path("EXCLUDE.sha256list")).split()
        self.assertEqual(len(exclude), 10)
        code, _ = self.run_cli(["freeze", "--alias", "f1", "--labels", lab])
        self.assertEqual(code, common.REFUSE_EXIT)  # 다시 안 얼린다
        self.assertEqual(labels.add_exclude(exclude[:3] + ["new"]), 1)  # 있는 열쇠는 안 더한다

    def test_freeze_refuses_missing_labels(self):
        self.make_set("f2", fake_set_rows())
        code, _ = self.run_cli(["freeze", "--alias", "f2"])
        self.assertEqual(code, common.REFUSE_EXIT)

    def test_freeze_refuses_label_of_other_pair(self):
        # 라벨 줄의 쌍 열쇠가 cands 와 다르면(다시 뽑은 뒤 옛 라벨) 멈춘다
        rows = fake_set_rows()
        lab = self.make_set("f5", rows)
        given = common.read_jsonl(lab)
        given[0]["pair"] = common.pair_key("다른 근거", "다른 주장")
        common.write_jsonl(lab, given)
        code, out = self.run_cli(["freeze", "--alias", "f5", "--labels", lab])
        self.assertEqual(code, common.REFUSE_EXIT, out)
        # labels make 때 사본(pairs.jsonl)과 cands 가 어긋나도 멈춘다 — Opus 합의 줄
        lab = self.make_set("f6", fake_set_rows())
        cands = common.read_jsonl(common.alias_dir("f6", "cands.jsonl"))
        folder = common.part_paths("f6")["labels"]
        common.write_jsonl(os.path.join(folder, labels.PAIRS_FILE),
                           [{"id": r["id"], "pair": "0" * 64 if i == 0 else r["pair"]} for i, r in enumerate(cands)])
        common.write_jsonl(os.path.join(folder, "decided.jsonl"),
                           [{"id": r["id"], "label": "b", "status": "agree"} for r in cands])
        code, out = self.run_cli(["freeze", "--alias", "f6"])
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn("쌍 열쇠가 cands 와 다르다", out)
        # 지금 cands 에 없는 쌍의 사람 답도 멈춘다
        common.write_jsonl(os.path.join(folder, labels.PAIRS_FILE), [{"id": r["id"], "pair": r["pair"]} for r in cands])
        common.write_jsonl(os.path.join(folder, "human.jsonl"), [{"id": "u2-000000000000", "label": "a"}])
        code, out = self.run_cli(["freeze", "--alias", "f6"])
        self.assertEqual(code, common.REFUSE_EXIT, out)
        self.assertIn("지금 cands 에 없다", out)

    def test_report_values_and_shape(self):
        lab = self.make_set("f3", fake_set_rows())
        self.run_cli(["freeze", "--alias", "f3", "--labels", lab])
        code, out = self.run_cli(["report", "--alias", "f3"])
        self.assertEqual(code, 0, out)
        md = read_text(common.alias_dir("f3", "result.md"))
        for head in ("T0 측정 조건", "T1 자리", "T3 실제 큐", "T4 혼동표", "T5 오판 꼴", "T6 수집 자리", "T7 읽는 법"):
            self.assertIn(head, md)
        res = common.read_json(common.alias_dir("f3", "result.json"))["results"][0]
        cells = res["places"]["2"]["flagged"]["all"]
        # b 재현 : v110 6/6 = 1.0 · v100 4/6 → 차 +0.3333 · McNemar b(v100 만 맞힘)=0, c(v110 만)=2
        self.assertEqual(cells["b_recall"]["v110"]["value"], 1.0)
        self.assertAlmostEqual(cells["b_recall"]["v100"]["value"], 0.6667, places=4)
        self.assertAlmostEqual(cells["b_recall"]["pair"]["diff"], 0.3333, places=4)
        self.assertEqual(cells["b_recall"]["pair"]["mcnemar"], {"b": 0, "c": 2, "p": 0.5})
        # 거짓 반대 : v110 은 올린 7 중 1(d) · v100 은 올린 7 중 3(d 2 · c 1)
        self.assertAlmostEqual(cells["false_contradict"]["v110"]["value"], 0.1429, places=4)
        self.assertAlmostEqual(cells["false_contradict"]["v100"]["value"], 0.4286, places=4)
        self.assertEqual(res["places"]["2"]["flagged"]["n_unseen"], 9)  # 본 쌍 뺌 줄

    def test_report_place1_and_place3_by_hand(self):
        lab = self.make_set("f7", place1_rows() + place3_rows())
        self.assertEqual(self.run_cli(["freeze", "--alias", "f7", "--labels", lab])[0], 0)
        code, out = self.run_cli(["report", "--alias", "f7"])
        self.assertEqual(code, 0, out)
        res = common.read_json(common.alias_dir("f7", "result.json"))["results"][0]
        one = res["places"]["1"]["replay-hard"]["all"]
        # 확신 거짓 지지 (정답 ≠ a : r2 · r3) — v110 r3 통과 1/2 · v100 0/2
        self.assertEqual(one["sure_false_support"]["v110"]["value"], 0.5)
        self.assertEqual(one["sure_false_support"]["v100"]["value"], 0.0)
        self.assertEqual(one["sure_false_support"]["pair"]["diff"], 0.5)
        # 헛 거절 (정답 a : r1 · r4) — v110 r4 1/2 · v100 0 (r4 는 보류)
        self.assertEqual(one["false_reject"]["v110"]["value"], 0.5)
        self.assertEqual(one["false_reject"]["v100"]["value"], 0.0)
        self.assertEqual(one["unsure"]["v100"]["value"], 0.25)
        self.assertEqual(one["filter_rate"]["v110"]["value"], 0.5)
        self.assertEqual(one["filter_rate"]["v100"]["value"], 1.0)
        three = res["places"]["3"]["m0"]["all"]
        # 숨김 놓침 (숨겼는데 정답 ≠ b) — v110 r1 1/3 · v100 0/3 · 숨김 v110 2/3 · v100 1/3
        self.assertAlmostEqual(three["hidden_miss"]["v110"]["value"], 0.3333, places=4)
        self.assertEqual(three["hidden_miss"]["v100"]["value"], 0.0)
        self.assertAlmostEqual(three["hidden"]["v110"]["value"], 0.6667, places=4)
        self.assertAlmostEqual(three["hidden"]["v100"]["value"], 0.3333, places=4)

    def test_tell_needs_ten_points(self):
        self.assertFalse(report.tells({"diff": 0.05, "low": 0.01, "high": 0.09}, 10))  # 구간은 0 밖이지만 10%p 아래
        self.assertTrue(report.tells({"diff": 0.12, "low": 0.02, "high": 0.2}, 10))
        self.assertTrue(report.tells({"diff": -0.12, "low": -0.2, "high": -0.02}, 10))
        self.assertFalse(report.tells({"diff": 0.2, "low": -0.01, "high": 0.4}, 10))
        protocol, _ = common.load_protocol()
        self.assertEqual(protocol["tell_min_pp"], 10)

    def test_report_refuses_tampered_set(self):
        lab = self.make_set("f4", fake_set_rows())
        self.run_cli(["freeze", "--alias", "f4", "--labels", lab])
        with open(common.alias_dir("f4", "set-use1.jsonl"), "a", encoding="utf-8") as f:
            f.write("\n")
        code, _ = self.run_cli(["report", "--alias", "f4"])
        self.assertEqual(code, common.REFUSE_EXIT)

    def test_md_escapes_untrusted_text(self):
        self.assertEqual(report.safe("a|b`c<d"), "a\\|b\\`c&lt;d")
        self.assertTrue(report.safe("x" * 200).endswith("…"))


_ = helpers
if __name__ == "__main__":
    unittest.main()
