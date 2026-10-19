"""localharness G5 얼린 셋을 NLI 판정 서버로 다시 잰다 — 하네스 코드는 안 바꾸고 재기만 (v1.2 적용 설계 6절).

    <서빙 venv python> -I measure/g5_nli.py --sets ~/.localharness/judge-sets --url http://127.0.0.1:8092

- 입력 : `--sets` 폴더의 `<날짜>-claim.jsonl` · `<날짜>-item.jsonl` (+ 같은 이름 `.out.jsonl` 의 SemIf 답). 읽기만 한다.
  폴더에 다른 날짜 파일(이 스크립트의 출력 포함)이 있어도 `--date` 두 파일만 읽는다.
- 두 팔 : raw(그대로 보냄 · 서버가 근거 꼬리를 자름) · cmp(`use_sites.harness.compress` 로 줄여 보냄).
  잘림 여부 · 토큰 수는 `pairenc.encode_pair` 로 센다 — 서버와 같은 함수 · 같은 토크나이저다.
- 두 자 : G5 자(a ≥ .8 통과 · a < .5 검토 · 사이 애매) · 보정 자(a ≥ .89 통과 · b 나 c ≥ .50 검토 · 나머지 보류).
- 셈 : 어긋남 재현(want B 가 검토) · 정상 오탐(want A 가 검토) · 무관 오인(want C 가 통과) · p95(왕복 ms).
  꼬리별 · B2 뺌 · raw 잘림/안 잘림 · 본 쌍 뺌. SemIf 와 짝 비교(McNemar 정확 검정).
- 출력 : `--out-dir`(기본 `<--sets>/nli/` — 셋 폴더 바로 아래면 harvest 가 셋으로 읽는다) 의 `<오늘>-nli-{claim,item}.jsonl`(본문 없이 id · 확률 · 판정만) + `<오늘>-nli-summary.json`.
  이미 있으면 `--force` 없이는 멈춘다.
tokenizers 가 든 python(서빙 venv)으로 돌린다. 시스템 python 에는 없을 수 있다.
"""
import argparse
import datetime
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _sub in (HERE, ROOT):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

from pairenc import encode_pair, pair_length  # noqa: E402
from use_sites import common, harness  # noqa: E402  common 이 train/ · build/ 를 import 길에 넣는다
from use_sites.sample import seen_keys  # noqa: E402
from compare import mcnemar_exact  # noqa: E402
from run_measure import pct  # noqa: E402

KINDS = ("claim", "item")
ARMS = ("raw", "cmp")
TAILS = ("A", "B0", "B1", "B2", "C")
G5_PASS, G5_REVIEW = harness.SEMIF_PASS, harness.SEMIF_REVIEW   # .8 · .5 — 하네스 g5.go 문턱
CAL_SUPPORT, CAL_SURE = 0.89, 0.50                                 # nli-studio-v10 calibration.json 의 nli_sure_support · nli_sure
LINE_RECALL, LINE_FALSE_ALARM, LINE_C_MISREAD = 0.95, 0.10, 0.10   # 설계 6절 통과선 · 무관 오인 상한
SIG_P = 0.05
MODEL_ENV = harness.MODEL_ENV
DEFAULT_MODEL = os.path.join("~", ".judgemodel", "models", "nli-studio-v10")
DEFAULT_SEEN = ",".join(os.path.join("~", ".judgemodel", p) for p in ("sets", "build"))
DEFAULT_MAX_LEN = harness.MAX_TOKENS
FIRST_FAIL_STOP = 3    # 한 팔 안 어디서든 이만큼 줄줄이 실패하면 서버가 죽은 것으로 보고 멈춘다
REFUSE_EXIT = 2


# ---------------------------------------------------------------- 판정 자

def g5_verdict(a):
    """G5 자 — 하네스와 같은 문턱. pass · unsure · review."""
    if a >= G5_PASS:
        return "pass"
    if a >= G5_REVIEW:
        return "unsure"
    return "review"


def cal_verdict(a, b, c):
    """보정 자 — 지지 확신이면 통과, 반대 · 무관 확신이면 검토, 나머지 보류(unsure)."""
    if a >= CAL_SUPPORT:
        return "pass"
    if max(b, c) >= CAL_SURE:
        return "review"
    return "unsure"


RULERS = {"g5": lambda p: g5_verdict(p["a"]), "cal": lambda p: cal_verdict(p["a"], p["b"], p["c"])}


# ---------------------------------------------------------------- 셈

def hit(gold, verdict):
    """그 자의 결과가 want 와 맞나 — 셈 넷과 같은 뜻. B 는 검토면 맞힘 · A 는 검토가 아니면 · C 는 통과가 아니면."""
    if gold == "b":
        return verdict == "review"
    if gold == "a":
        return verdict != "review"
    if gold == "c":
        return verdict != "pass"
    return None


def share(n, d):
    return round(n / d, 4) if d else None


def tally(pairs):
    """[(gold, verdict)] → 셈 넷의 비율과 분자 · 분모. verdict 가 None(실패 · SemIf 없음)인 줄은 뺀다."""
    got = [(g, v) for g, v in pairs if v is not None]
    by = {k: [v for g, v in got if g == k] for k in "abc"}
    out = {"n": len(got), "n_a": len(by["a"]), "n_b": len(by["b"]), "n_c": len(by["c"]),
           "b_review": sum(v == "review" for v in by["b"]), "a_review": sum(v == "review" for v in by["a"]),
           "c_pass": sum(v == "pass" for v in by["c"]), "unsure": sum(v == "unsure" for _, v in got)}
    out["recall_b"] = share(out["b_review"], out["n_b"])
    out["false_alarm_a"] = share(out["a_review"], out["n_a"])
    out["c_misread"] = share(out["c_pass"], out["n_c"])
    out["unsure_share"] = share(out["unsure"], out["n"])
    return out


def scopes(rows):
    """이름 → 줄 목록. rows 는 (팔 하나의) 결과 줄."""
    out = {"all": rows, "no_B2": [r for r in rows if r["tail"] != "B2"]}
    for kind in KINDS:
        out[f"kind:{kind}"] = [r for r in rows if r["kind"] == kind]
    for tail in TAILS:
        out[f"tail:{tail}"] = [r for r in rows if r["tail"] == tail]
    out["raw_cut"] = [r for r in rows if r["raw_cut"]]
    out["raw_uncut"] = [r for r in rows if not r["raw_cut"]]
    if any(r.get("seen") for r in rows):
        out["unseen"] = [r for r in rows if not r.get("seen")]
    return out


def semif_verdict(r):
    return (r.get("semif") or {}).get("verdict")


def pair_test(rows, ruler):
    """같은 쌍에서 SemIf vs NLI(그 자) — 셈 셋(재현 · 오탐 · 무관 오인)마다 McNemar. 둘 다 답이 있는 줄만."""
    out = {}
    for name, gold in (("recall_b", "b"), ("false_alarm_a", "a"), ("c_misread", "c")):
        both = [r for r in rows if r["gold"] == gold and r["verdict"].get(ruler) and semif_verdict(r)]
        base = [hit(gold, semif_verdict(r)) for r in both]
        new = [hit(gold, r["verdict"][ruler]) for r in both]
        out[name] = {"n": len(both), **mcnemar_exact(base, new)}
    return out


def summarize_arm(rows):
    """팔 하나의 셈 — 자 × 범위 · SemIf 같은 범위 · McNemar · p95."""
    out = {"rulers": {}, "semif": {}, "mcnemar": {}}
    for name, part in scopes(rows).items():
        out["semif"][name] = tally([(r["gold"], semif_verdict(r)) for r in part])
        for ruler in RULERS:
            out["rulers"].setdefault(ruler, {})[name] = tally([(r["gold"], r["verdict"].get(ruler)) for r in part])
    for ruler in RULERS:
        out["mcnemar"][ruler] = {"all": pair_test(rows, ruler),
                                 "no_B2": pair_test([r for r in rows if r["tail"] != "B2"], ruler)}
    times = [r["rt_ms"] for r in rows if r.get("rt_ms") is not None]
    out["p95_ms"] = round(pct(times, 0.95), 1) if times else None
    out["failed"] = sum(1 for r in rows if r.get("error"))
    out["cut"] = sum(1 for r in rows if r["cut"])
    out["n"] = len(rows)
    return out


def judge(summary_arms, scope="all"):
    """설계 6절 판정을 미리 박은 대로 — 넘음 · 통과선 · 못 넘음 중 하나와 근거 줄.

    넘음 : 한 팔 · 한 자에서 재현 · 오탐 둘 다 SemIf 보다 낫고, 그중 하나는 McNemar p < .05, 무관 오인 ≤ 10%.
    통과선 : 넘음이면서 재현 ≥ .95 · 오탐 ≤ 10% 인 팔 · 자가 있다. 셋 다 아니면 못 넘음.
    범위는 B2 를 포함한 셋 전체(10-08 SemIf 숫자와 같은 셋) — B2 뺀 판정은 참고로 따로 낸다.
    """
    over, line, why = [], [], []
    for arm, s in summary_arms.items():
        base = s["semif"][scope]
        for ruler in RULERS:
            t = s["rulers"][ruler][scope]
            m = s["mcnemar"][ruler][scope]
            if None in (t["recall_b"], t["false_alarm_a"], base["recall_b"], base["false_alarm_a"]):
                why.append(f"{arm}·{ruler} : 셀 수 없음")
                continue
            better = t["recall_b"] > base["recall_b"] and t["false_alarm_a"] < base["false_alarm_a"]
            sig = min(m["recall_b"]["p"], m["false_alarm_a"]["p"]) < SIG_P
            c_ok = t["c_misread"] is not None and t["c_misread"] <= LINE_C_MISREAD
            ok = better and sig and c_ok
            reached = ok and t["recall_b"] >= LINE_RECALL and t["false_alarm_a"] <= LINE_FALSE_ALARM
            why.append(f"{arm}·{ruler} : 재현 {t['recall_b']} (SemIf {base['recall_b']}) · 오탐 {t['false_alarm_a']} "
                       f"(SemIf {base['false_alarm_a']}) · 무관 오인 {t['c_misread']} · p 재현 {m['recall_b']['p']} "
                       f"· p 오탐 {m['false_alarm_a']['p']} → {'넘음' if ok else '못 넘음'}{' · 통과선' if reached else ''}")
            if ok:
                over.append(f"{arm}·{ruler}")
            if reached:
                line.append(f"{arm}·{ruler}")
    verdict = "통과선" if line else ("넘음" if over else "못 넘음")
    return {"verdict": verdict, "scope": scope, "over": over, "line": line, "why": why}


# ---------------------------------------------------------------- 토큰

class TokCounter:
    """compress 가 쓰는 count(text) — 잘림 설정을 끄고 센다(encode_pair 가 설정을 바꿔 놓으므로)."""

    mode = "xlmr"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, text):
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def load_tokenizer(path):
    """tokenizer.json 또는 그 폴더. tokenizers 가 없으면 서빙 venv 로 돌리라고 멈춘다."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(path):
        raise SystemExit(f"토크나이저가 없다 : {path}")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("tokenizers 가 없다 — 서빙 venv 의 python 으로 돌린다")
    return Tokenizer.from_file(path), path


def model_max_len(tokenizer_path):
    """토크나이저 윗 폴더의 model.json input.max_len. 없으면 320."""
    spec = os.path.join(os.path.dirname(os.path.dirname(tokenizer_path)), "model.json")
    try:
        return int(common.read_json(spec)["input"]["max_len"]), spec
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_MAX_LEN, None


# ---------------------------------------------------------------- 입력

def refuse(message):
    """traceback 대신 까닭 한 줄을 stderr 에 찍고 exit 2."""
    print(message, file=sys.stderr)
    raise SystemExit(REFUSE_EXIT)


def check_jsonl(path):
    """줄마다 JSON 인지 본다. 깨진 줄이 있으면 파일 이름 · 줄 번호로 멈춘다."""
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except ValueError as err:
                refuse(f"셋 줄이 깨졌다 : {os.path.basename(path)} {number}줄 ({err.msg})")


def load_rows(folder, date):
    """`<date>-{claim,item}.jsonl` 두 파일만 임시 폴더에 베껴 harness.judge_set_rows 로 읽는다 (원본은 읽기만)."""
    names = [f"{date}-{kind}{tail}" for kind in KINDS for tail in (".jsonl", ".out.jsonl")]
    missing = [n for n in names if not os.path.isfile(os.path.join(folder, n))]
    if missing:
        refuse(f"얼린 셋 파일이 없다 : {', '.join(missing)}")
    for name in names:
        check_jsonl(os.path.join(folder, name))
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            shutil.copyfile(os.path.join(folder, name), os.path.join(tmp, name))
        rows, files = harness.judge_set_rows(tmp)
    for name in names:
        if name.endswith(".out.jsonl"):
            files.append({"name": name, "sha256": common.sha256_file(os.path.join(folder, name))})
    rows.sort(key=lambda r: (r["kind"], r["id"]))
    return rows, files


def prepare(rows, tokenizer, max_len):
    """줄마다 두 팔에 보낼 글과 길이 정보. 서버에 묻기 전에 다 정한다(차례 고정)."""
    counter = TokCounter(tokenizer)
    out = []
    for r in rows:
        evidence, claim = r["evidence"], r["claim"]
        raw_tokens = pair_length(tokenizer, evidence, claim)
        _, raw_cut = encode_pair(tokenizer, evidence, claim, max_len)
        ev_c, cl_c, info = harness.compress(counter, evidence, claim)
        cmp_tokens = pair_length(tokenizer, ev_c, cl_c)
        _, cmp_cut = encode_pair(tokenizer, ev_c, cl_c, max_len)
        out.append({"row": r, "raw_cut": raw_cut,
                    "arms": {"raw": {"evidence": evidence, "claim": claim, "tokens": raw_tokens, "cut": raw_cut},
                             "cmp": {"evidence": ev_c, "claim": cl_c, "tokens": cmp_tokens, "cut": cmp_cut,
                                     "compress_cut": info["cut"]}}})
    return out


def mark_seen(prepared, roots):
    """본 쌍 대조 — 학습 · 셋 jsonl 의 쌍 열쇠와 raw · cmp 열쇠를 견준다. 없는 폴더는 「못 봄」."""
    present = [r for r in roots if os.path.isdir(os.path.expanduser(r))]
    info = {"roots": roots, "missing": [r for r in roots if r not in present], "checked": bool(present)}
    if not present:
        for p in prepared:
            p["seen"] = None
        info["seen_pairs"] = None
        return info
    keys = seen_keys(present)
    info["keys"] = len(keys)
    for p in prepared:
        p["seen"] = any(common.pair_key(a["evidence"], a["claim"]) in keys for a in p["arms"].values())
    info["seen_pairs"] = sum(1 for p in prepared if p["seen"])
    return info


# ---------------------------------------------------------------- 서버

def http_json(url, body=None, timeout=30):
    data, headers = None, {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def http_asker(url):
    """(evidence, claim) → 서버 응답 dict. label_set nli3 를 실어 3칸을 청한다."""
    def ask(evidence, claim):
        return http_json(url.rstrip("/") + "/judge", {"evidence": evidence, "claim": claim, "label_set": "nli3"})
    return ask


def check_answer(got):
    """응답에서 a · b · c 를 꺼낸다. 칸 빠짐 · 3칸 아님 · 합 어긋남이면 ValueError."""
    if got.get("label_set") not in (None, "nli3"):
        raise ValueError(f"label_set {got.get('label_set')}")
    p = {k: float(got[k]) for k in "abc"}
    if "d" in got or abs(sum(p.values()) - 1) > 0.02 or not all(0 <= v <= 1 for v in p.values()):
        raise ValueError("확률 꼴이 3칸이 아니다")
    return p


def run_arms(prepared, ask, clock=time.perf_counter):
    """팔마다 차례대로 묻는다(raw 전부 → cmp 전부). 결과 줄 목록. 본문은 줄에 안 남긴다."""
    out = []
    for arm in ARMS:
        streak = 0
        for p in prepared:
            r, sent = p["row"], p["arms"][arm]
            line = {"id": r["id"][len("js:"):], "kind": r["kind"], "tail": r["variant"], "arm": arm,
                    "gold": r["gold"], "tokens": sent["tokens"], "cut": sent["cut"], "raw_cut": p["raw_cut"],
                    "seen": p.get("seen"), "semif": r["semif"] and {k: r["semif"][k] for k in ("p_a", "verdict")},
                    "a": None, "b": None, "c": None, "ms": None, "rt_ms": None, "verdict": {}}
            if "compress_cut" in sent:
                line["compress_cut"] = sent["compress_cut"]
            start = clock()
            try:
                got = ask(sent["evidence"], sent["claim"])
                probs = check_answer(got)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, http.client.HTTPException) as err:
                line["error"] = type(err).__name__
                streak += 1
                out.append(line)
                if streak >= FIRST_FAIL_STOP:
                    raise SystemExit(f"{arm} 팔에서 {streak} 줄 연속 실패 — 서버를 본다 ({line['error']} · {line['id']})")
                continue
            streak = 0
            line["rt_ms"] = round((clock() - start) * 1000, 1)
            line["ms"] = got.get("ms")
            line.update({k: round(v, 6) for k, v in probs.items()})
            line["verdict"] = {name: f(probs) for name, f in RULERS.items()}
            out.append(line)
    return out


# ---------------------------------------------------------------- 묶기

def build_summary(lines, meta):
    arms = {arm: summarize_arm([l for l in lines if l["arm"] == arm]) for arm in ARMS}
    total = len([l for l in lines if l["arm"] == "raw"])
    judgement, judgement_no_b2 = judge(arms, "all"), judge(arms, "no_B2")
    partial = any(s["failed"] > 0 for s in arms.values())
    if partial:
        # 한 줄이라도 답을 못 받으면 셈이 다른 쌍 위에서 나온 것이라 판정하지 않는다
        for j in (judgement, judgement_no_b2):
            j["verdict_if_complete"] = j["verdict"]
            j["verdict"] = "셀 수 없음"
    return {**meta, "pairs": total,
            "raw_cut_pairs": sum(1 for l in lines if l["arm"] == "raw" and l["cut"]),
            "rulers": {"g5": {"pass": G5_PASS, "review_below": G5_REVIEW},
                       "cal": {"pass": CAL_SUPPORT, "review_bc": CAL_SURE}},
            "lines": {"recall_b_min": LINE_RECALL, "false_alarm_a_max": LINE_FALSE_ALARM,
                      "c_misread_max": LINE_C_MISREAD, "p": SIG_P},
            "arms": arms, "judgement": judgement, "judgement_no_B2": judgement_no_b2, "partial": partial}


def write_outputs(out_dir, stamp, lines, summary):
    paths = {}
    for kind in KINDS:
        paths[kind] = os.path.join(out_dir, f"{stamp}-nli-{kind}.jsonl")
        common.write_jsonl(paths[kind], [l for l in lines if l["kind"] == kind])
    paths["summary"] = os.path.join(out_dir, f"{stamp}-nli-summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(paths["summary"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return paths


def print_table(summary):
    print(f"쌍 {summary['pairs']} · raw 잘림 {summary['raw_cut_pairs']}")
    for arm, s in summary["arms"].items():
        for ruler in RULERS:
            t = s["rulers"][ruler]["all"]
            print(f"  {arm}·{ruler} : 재현 {t['recall_b']} · 오탐 {t['false_alarm_a']} · 무관 오인 {t['c_misread']} "
                  f"· 보류 {t['unsure_share']} · p95 {s['p95_ms']}ms · 실패 {s['failed']}")
    t = summary["arms"]["raw"]["semif"]["all"]
    print(f"  SemIf : 재현 {t['recall_b']} · 오탐 {t['false_alarm_a']} · 무관 오인 {t['c_misread']}")
    print(f"판정 : {summary['judgement']['verdict']} (B2 뺌 : {summary['judgement_no_B2']['verdict']})")
    if summary.get("partial"):
        failed = {arm: s["failed"] for arm, s in summary["arms"].items()}
        print(f"경고 : 답을 못 받은 줄이 있다 {failed} — 셈은 일부 쌍뿐이라 판정하지 않는다")


def main(argv=None):
    model_dir = os.environ.get(MODEL_ENV) or DEFAULT_MODEL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", default=os.path.join(harness.LH_HOME, "judge-sets"), help="얼린 셋 폴더 (읽기만)")
    ap.add_argument("--date", default="2026-10-08", help="읽을 셋 파일 날짜 머리")
    ap.add_argument("--url", default="http://127.0.0.1:8092")
    ap.add_argument("--tokenizer", default=os.path.join(model_dir, "tokenizer"),
                    help=f"tokenizer 폴더나 tokenizer.json (기본 ${MODEL_ENV}/tokenizer, 없으면 nli-studio-v10)")
    ap.add_argument("--seen", default=DEFAULT_SEEN, help="본 쌍 대조 폴더 (쉼표) · none 이면 안 본다")
    ap.add_argument("--out-dir", default="", help="출력 폴더 (기본 <--sets>/nli)")
    ap.add_argument("--stamp", default=datetime.date.today().isoformat(), help="출력 파일 날짜 머리")
    ap.add_argument("--force", action="store_true", help="출력이 이미 있어도 덮는다")
    a = ap.parse_args(argv)

    folder = os.path.expanduser(a.sets)
    # 기본은 하위 폴더 nli/ — 셋 폴더 바로 아래에 두면 harness.judge_set_rows 가 출력까지 셋으로 읽는다
    out_dir = os.path.expanduser(a.out_dir) if a.out_dir else os.path.join(folder, "nli")
    if a.stamp == a.date:
        raise SystemExit("--stamp 가 --date 와 같다 — 원본 셋을 덮지 않는다")
    targets = [os.path.join(out_dir, f"{a.stamp}-nli-{k}") for k in ("claim.jsonl", "item.jsonl", "summary.json")]
    if not a.force and any(os.path.exists(t) for t in targets):
        raise SystemExit("출력이 이미 있다 — 다시 재려면 --force")

    try:
        health = http_json(a.url.rstrip("/") + "/health", timeout=5)
    except (OSError, ValueError, http.client.HTTPException) as err:
        refuse(f"판정 서버 health 실패 : {a.url} ({type(err).__name__})")
    if not isinstance(health, dict) or not health.get("ok"):
        refuse(f"서버가 ok 가 아니다 : {health}")
    rows, files = load_rows(folder, a.date)
    tokenizer, tok_path = load_tokenizer(a.tokenizer)
    max_len, spec_path = model_max_len(tok_path)
    prepared = prepare(rows, tokenizer, max_len)
    roots = [] if a.seen == "none" else [r for r in a.seen.split(",") if r]
    seen = mark_seen(prepared, roots)
    started = common.now_iso()
    lines = run_arms(prepared, http_asker(a.url))
    meta = {"at": started, "url": a.url, "model": {k: health.get(k) for k in ("model", "name", "backend", "letters")},
            "sets": files, "date": a.date, "tokenizer_sha256": common.sha256_file(tok_path),
            "max_len": max_len, "max_len_from": "model.json" if spec_path else "default", "seen": seen,
            "note": "p95 는 왕복 ms — 같은 때 다른 손님 요청이 섞였을 수 있다"}
    summary = build_summary(lines, meta)
    paths = write_outputs(out_dir, a.stamp, lines, summary)
    print_table(summary)
    for p in paths.values():
        print(f"  → {p}")
    return summary


if __name__ == "__main__":
    main()
