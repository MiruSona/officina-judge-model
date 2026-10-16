"""⑤ harvest-harness — localharness G5 얼린 셋 · 지난 runs 판정 기록을 규칙 축약(bm25 2줄 · 320토큰) 쌍으로 거둔다 (설계 4-2).
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
하네스 쪽(~/.localharness)은 읽기만 한다. 쓰기는 use1/<별명>/collect/ 안뿐이다.
"""
import glob
import json
import math
import os
import re
import statistics
import subprocess

from use_sites import common, servers
from bm25 import BM25  # noqa: E402  build/ 는 common 이 import 길에 넣는다

DEFAULT_ALIAS = "harness"
LH_HOME = os.path.join("~", ".localharness")
MODEL_ENV = "JUDGEMODEL_MODEL_DIR"   # 기본 토크나이저를 찾는 모델 폴더 환경 변수
OUT_PARTS = ("collect", "harness-pairs.jsonl")
MANIFEST_PARTS = ("collect", "harvest-harness.log.jsonl")

MAX_TOKENS = 320          # v4 model.json max_len
PAIR_SPECIALS = 4         # XLM-R 쌍 특수 토큰
CLAIM_MAX_TOKENS = 160
TOP_LINES = 2
CHARS_PER_TOKEN = 1.5     # 토크나이저가 없을 때 어림 — G5 셋 실측 평균 2.06 · 하위 5% 1.63 보다 짜게
SEMIF_PASS = 0.8
SEMIF_REVIEW = 0.5
REAL_TAILS = ("A", "B0")  # judgeset id 꼬리 — A 는 통과 판 줄 그대로 · B0 는 되돌림 판 줄(사람 want)

WORD_RE = re.compile(r"[0-9A-Za-z가-힣]")
COMMENT_HEAD = re.compile(r"^(?://+|#+|/\*+|\*+/?|<!--|--|;+)\s*")
COMMENT_TAIL = re.compile(r"\s*(?:\*/|-->)\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")

COUNT_SCRIPT = (
    "import json, sys, tokenizers\n"
    "t = tokenizers.Tokenizer.from_file(sys.argv[1])\n"
    "for line in sys.stdin:\n"
    "    print(len(t.encode(json.loads(line), add_special_tokens=False).ids), flush=True)\n"
)


# ---------------------------------------------------------------- 토큰 세기

def default_tokenizer():
    """모델 폴더($JUDGEMODEL_MODEL_DIR) 안의 tokenizer.json. 환경 변수가 없으면 빈 값 — 글자 수 어림으로 센다."""
    model_dir = os.environ.get(MODEL_ENV, "")
    return os.path.join(model_dir, "tokenizer", "tokenizer.json") if model_dir else ""


class TokenCounter:
    """XLM-R 토크나이저(서빙 venv 의 tokenizers)로 센다. 못 띄우면 글자 수 어림."""

    def __init__(self, tokenizer_path="", python=""):
        self.mode = "chars"
        self.proc = None
        if not tokenizer_path or not python or not os.path.isfile(tokenizer_path):
            return
        try:
            self.proc = subprocess.Popen([python, "-I", "-c", COUNT_SCRIPT, tokenizer_path], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
            if self._ask("ab") > 0:
                self.mode = "xlmr"
        except (OSError, ValueError):
            self.close()

    def _ask(self, text):
        self.proc.stdin.write(json.dumps(text) + "\n")
        self.proc.stdin.flush()
        return int(self.proc.stdout.readline())

    def count(self, text):
        if self.mode == "xlmr":
            return self._ask(text)
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.proc = None
        self.mode = "chars"


# ---------------------------------------------------------------- 축약

def clean_code(text):
    """코드 꾸밈(들여쓰기 · 주석 머리꼬리 · 겹친 공백)을 뗀다. 글자 · 숫자가 없으면 빈 글."""
    text = COMMENT_TAIL.sub("", COMMENT_HEAD.sub("", text.strip()))
    text = " ".join(text.split())
    if not WORD_RE.search(text):
        return ""
    return text


def diff_units(evidence):
    """diff → [(부호, 파일, 줄)]. 덩어리의 + · - 본문 줄만 담는다. diff 가 아니면 문장 단위."""
    if "@@" not in evidence:
        parts = [p for line in evidence.split("\n") for p in SENTENCE_END.split(line)]
        return [("+", "", t) for t in (clean_code(p) for p in parts) if t]
    units = []
    path = ""
    for line in evidence.split("\n"):
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            continue
        if line.startswith(("--- ", "diff --git ", "index ", "@@")) or not line or line[0] not in "+-":
            continue
        text = clean_code(line[1:])
        if text:
            units.append((line[0], path, text))
    return units


def pick_lines(evidence, claim):
    """claim 을 질의로 bm25 위 2줄(더한 줄 먼저, 없으면 지운 줄). 원래 차례로 잇고 파일 이름을 앞에 단다."""
    units = diff_units(evidence)
    pool = [u for u in units if u[0] == "+"]
    if not pool:
        pool = [u for u in units if u[0] == "-"]
    if not pool:
        return ""
    chosen = BM25([u[2] for u in pool]).rank(claim)[:TOP_LINES]
    for index in range(len(pool)):
        if len(chosen) >= TOP_LINES:
            break
        if index not in chosen:
            chosen.append(index)
    chosen.sort()
    lines = [("(지움) " if pool[i][0] == "-" else "") + pool[i][2] for i in chosen]
    path = pool[chosen[0]][1]
    if path and path != "/dev/null":
        return f"{path} : " + " ".join(lines)
    return " ".join(lines)


def fit(counter, text, budget):
    """토큰 budget 안으로 뒤를 자른다(자르면 끝에 …). (글, 잘렸나)."""
    if counter.count(text) <= budget:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid] + "…") <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…", True


def compress(counter, evidence, claim):
    """(축약 evidence, 축약 claim, 길이 정보). 쌍 전체가 MAX_TOKENS 안이다."""
    claim_text, claim_cut = fit(counter, " ".join(claim.split()), CLAIM_MAX_TOKENS)
    claim_tokens = counter.count(claim_text)
    budget = MAX_TOKENS - PAIR_SPECIALS - claim_tokens
    ev_text, ev_cut = fit(counter, pick_lines(evidence, claim_text), budget)
    ev_tokens = counter.count(ev_text)
    info = {"before_chars": len(evidence), "after_chars": len(ev_text), "tokens": ev_tokens + claim_tokens + PAIR_SPECIALS,
            "cut": ev_cut or claim_cut}
    return ev_text, claim_text, info


# ---------------------------------------------------------------- 읽기 (하네스 쪽은 읽기만)

def semif_of(row):
    """하네스 판정 줄 → {p_a, letter, verdict, cached}. P(A) 가 없으면 None. g5.go 문턱 .8 · .5."""
    probs = row.get("probs") or {}
    if "A" not in probs:
        return None
    p_a = float(probs["A"])
    verdict = "review"
    if p_a >= SEMIF_PASS:
        verdict = "pass"
    elif p_a >= SEMIF_REVIEW:
        verdict = "unsure"
    return {"p_a": p_a, "letter": str(row.get("letter", "")).lower(), "verdict": verdict, "cached": bool(row.get("cached"))}


def judge_set_rows(folder):
    """얼린 셋 폴더의 *-item.jsonl · *-claim.jsonl (+ 같은 이름 .out.jsonl 의 SemIf 답). (줄 목록, 파일 정보)."""
    rows, files = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        name = os.path.basename(path)
        if name.endswith(".out.jsonl"):
            continue
        kind = name[:-len(".jsonl")].rsplit("-", 1)[-1]
        if kind not in ("item", "claim"):
            continue
        answers = {r.get("id"): r for r in common.read_jsonl(path[:-len(".jsonl")] + ".out.jsonl")}
        got = common.read_jsonl(path)
        files.append({"name": name, "sha256": common.sha256_file(path), "lines": len(got)})
        for r in got:
            tail = str(r.get("id", "")).rsplit("-", 1)[-1]
            layer = "judgeset-made"
            if tail in REAL_TAILS:
                layer = f"judgeset-{kind}"
            rows.append({"id": "js:" + str(r.get("id")), "source": "judgeset", "kind": kind, "layer": layer, "variant": tail,
                         "evidence": r.get("evidence") or "", "claim": r.get("claim") or "",
                         "gold": str(r.get("want", "")).lower() or None, "semif": semif_of(answers.get(r.get("id")) or {})})
    return rows, files


def main_review(run_dir):
    """run.json 「메인검토」.값(통과 · 되돌림 · 고침) — 판 단위 약한 라벨. 없으면 None."""
    try:
        review = common.read_json(os.path.join(run_dir, "run.json")).get("메인검토")
    except (OSError, ValueError):
        return None
    if isinstance(review, dict):
        return review.get("값")
    return None


def runs_rows(runs_dir, since=""):
    """runs/*/NN-judge.jsonl → 줄 목록. evidence · claim · kind 가 없는 줄은 건너뛰고 센다. (줄, 판 수, 건너뛴 수)."""
    rows, skipped = [], 0
    paths = sorted(glob.glob(os.path.join(runs_dir, "*", "[0-9][0-9]-judge.jsonl")))
    for path in paths:
        run_dir = os.path.dirname(path)
        run, attempt = os.path.basename(run_dir), os.path.basename(path)[:2]
        review = main_review(run_dir)
        for r in common.read_jsonl(path):
            if not (r.get("evidence") and r.get("claim") and r.get("kind")):
                skipped += 1
                continue
            if since and str(r.get("at", ""))[:10] < since:
                continue
            rows.append({"id": f"run:{run}:{attempt}:{r.get('id')}", "source": "runs", "kind": r["kind"],
                         "layer": "runs-" + r["kind"], "evidence": r["evidence"], "claim": r["claim"],
                         "gold": None, "semif": semif_of(r), "main_review": review})
    return rows, len(paths), skipped


# ---------------------------------------------------------------- 거두기

def harvest(raw, existing, counter):
    """원문 줄 → 새 쌍 줄(열쇠 · id 중복 없음). (새 줄, 셈)."""
    seen_ids = {r.get("id") for r in existing}
    seen_pairs = {r.get("pair") for r in existing if r.get("pair")}
    out, counts = [], {"dup": 0, "refused": 0, "empty": 0}
    for r in raw:
        if r["id"] in seen_ids:
            counts["dup"] += 1
            continue
        seen_ids.add(r["id"])
        raw_evidence, raw_claim = common.strip_tags(r["evidence"]), common.strip_tags(r["claim"])
        if common.secret_kind(raw_evidence + "\n" + raw_claim):
            counts["refused"] += 1
            out.append({**r, "evidence": "", "claim": "", "semif": None, "refused": True})
            continue
        evidence, claim, info = compress(counter, raw_evidence, raw_claim)
        if not evidence:
            counts["empty"] += 1
            continue
        key = common.pair_key(evidence, claim)
        if key in seen_pairs:
            counts["dup"] += 1
            continue
        seen_pairs.add(key)
        out.append({**r, "evidence": evidence, "claim": claim, "pair": key, "len": info, "token_mode": counter.mode})
    return out, counts


def length_stats(rows):
    kept = [r["len"] for r in rows if r.get("len")]
    if not kept:
        return {}
    tokens = [x["tokens"] for x in kept]
    before = [x["before_chars"] for x in kept]
    return {"before_chars_median": statistics.median(before), "before_chars_max": max(before),
            "tokens_median": statistics.median(tokens), "tokens_max": max(tokens),
            "over_limit": sum(t > MAX_TOKENS for t in tokens), "cut": sum(x["cut"] for x in kept)}


def layer_counts(rows):
    out = {}
    for r in rows:
        name = r["layer"] + (" (refused)" if r.get("refused") else "")
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def cmd_harvest_harness(a):
    if a.judge_sets is None and a.runs is None:
        raise common.Refuse("--judge-sets 나 --runs 중 하나는 준다")
    alias = a.alias or DEFAULT_ALIAS
    out_path = common.alias_dir(alias, *OUT_PARTS)
    raw, files, runs_info = [], [], {}
    if a.judge_sets is not None:
        folder = os.path.expanduser(a.judge_sets)
        if not os.path.isdir(folder):
            raise common.Refuse(f"얼린 셋 폴더가 없다 : {folder}")
        got, files = judge_set_rows(folder)
        raw += got
    if a.runs is not None:
        runs_dir = os.path.expanduser(a.runs)
        if not os.path.isdir(runs_dir):
            raise common.Refuse(f"runs 폴더가 없다 : {runs_dir}")
        got, judged, skipped = runs_rows(runs_dir, a.since)
        raw += got
        runs_info = {"judge_files": judged, "rows": len(got), "skipped_no_claim": skipped, "since": a.since}
    if not a.tokenizer:
        print(f"토크나이저를 모른다 (--tokenizer · ${MODEL_ENV} 둘 다 없음) — 글자 수 어림으로 센다")
    counter = TokenCounter(os.path.expanduser(a.tokenizer), a.python or servers.serving_python())
    try:
        new, counts = harvest(raw, common.read_jsonl(out_path), counter)
    finally:
        mode = counter.mode
        counter.close()
    common.append_jsonl(out_path, new)
    summary = {"at": common.now_iso(), "token_mode": mode, "max_tokens": MAX_TOKENS, "read": len(raw), "added": len(new),
               **counts, "layers": layer_counts(new), "length": length_stats(new), "judge_sets": files, "runs": runs_info}
    common.append_jsonl(common.alias_dir(alias, *MANIFEST_PARTS), [summary])
    print(f"⑤ 거두기 : 읽음 {len(raw)} · 더함 {len(new)} · 중복 {counts['dup']} · refused {counts['refused']} · 토큰 셈 {mode}")
    for name, n in summary["layers"].items():
        print(f"  {name} : {n}")
    print(f"  길이 : {json.dumps(summary['length'], ensure_ascii=False)}")
    print(f"  → {out_path}")
    return summary


def register(sub):
    s = sub.add_parser("harvest-harness", help="⑤ localharness G5 얼린 셋 · runs 판정 기록 거두기 + 규칙 축약")
    s.add_argument("--repo", default=".", help="받되 쓰지 않는다 (다른 명령과 인자 꼴을 맞춘다 · ⑤ 는 저장소와 무관)")
    s.add_argument("--alias", default="")
    s.add_argument("--judge-sets", nargs="?", const=os.path.join(LH_HOME, "judge-sets"), default=None)
    s.add_argument("--runs", nargs="?", const=os.path.join(LH_HOME, "runs"), default=None)
    s.add_argument("--since", default="")
    s.add_argument("--tokenizer", default=default_tokenizer(),
                   help=f"tokenizer.json (기본 ${MODEL_ENV}/tokenizer/tokenizer.json · 없으면 글자 수 어림)")
    s.add_argument("--python", default="", help="tokenizers 가 든 python (기본 서빙 venv)")
    s.set_defaults(func=cmd_harvest_harness)
