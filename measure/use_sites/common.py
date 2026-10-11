"""use_eval 공용 — 별명 · 감옥 경로 · mem 부르기 · 쌍 열쇠 · 확신 가르기 · 비밀 꼴 (설계 `Docs/Design/2026-10-12-실사용측정설계.md`).
쌍 본문 · 기록은 전부 `~/.judgemodel/measure/use1/<별명>/` 아래(git 밖)에 둔다. 저장소에는 코드 · 규약만 둔다.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.dirname(HERE)
ROOT = os.path.dirname(MEASURE)
for _sub in (MEASURE, os.path.join(ROOT, "train"), os.path.join(ROOT, "build")):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

from data_prep import strip_tags  # noqa: E402  mem quality.StripTags 와 글자 하나까지 같은 규칙
from label_batches import inside_git  # noqa: E402

PROTOCOL_DEFAULT = os.path.join(MEASURE, "protocol-use1.json")
HOME_ENV = "JUDGEMODEL_USE1_HOME"
MEM_ENV = "JUDGEMODEL_MEM"
ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
LETTERS3 = ("a", "b", "c")
REFUSE_EXIT = 2
MEM_TIMEOUT_S = 1800
MEM_REFUSED = "secret-shape"   # mem llm.ErrRefused — 비밀 꼴이라 mem 이 안 물은 쌍
UNSURE_DEFAULT = {"below": 0.40, "tie_gap": 0.01}

# 비밀 꼴 — (종류, 꼴). 앞에서부터 처음 걸린 종류를 돌려준다. 127.x · 0.0.0.0 은 안 거른다.
SECRETS = (
    ("key", re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                       r"|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})")),
    ("assign", re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S{6,}")),
    ("assign", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd|bearer)\b\s*[:=]\s*"
                          r"(\"[^\"]{4,}\"|'[^']{4,}')")),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("token", re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{32,}\b")),
    ("ip", re.compile(r"(?<![0-9.])(?!127\.)(?!0\.0\.0\.0(?![0-9]))(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")),
    ("home", re.compile(r"(?i)(\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"'`]+|(?<![\w.])/home/[^/\s\"'`]+|(?<![\w.])/Users/[^/\s\"'`]+)")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
)
# 따옴표 없는 「이름 = 값」 은 값에 글자와 숫자가 섞이고 12자 넘을 때만 비밀로 본다 (코드 대입문을 덜 거르려고).
BARE_SECRET = re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd|bearer)\b\s*[:=]\s*"
                         r"([A-Za-z0-9_\-.+/]{12,})")


class Refuse(Exception):
    """멈춰야 하는 어긋남 — main 이 exit 2 로 바꾼다."""


def use1_home():
    given = os.environ.get(HOME_ENV, "").strip()
    if given:
        return os.path.abspath(given)
    return os.path.join(os.path.expanduser("~"), ".judgemodel", "measure", "use1")


def default_alias(repo):
    """별명이 없으면 저장소 경로 해시 앞 8자 — 이름이 결과에 새지 않게."""
    digest = hashlib.sha256(os.path.abspath(repo).lower().encode("utf-8")).hexdigest()
    return "repo-" + digest[:8]


def check_alias(alias):
    if not ALIAS_RE.match(alias or "") or alias in (".", ".."):
        raise Refuse(f"별명 꼴이 틀렸다 (영문 · 숫자 · . _ - 40자 안) : {alias!r}")
    return alias


def alias_dir(alias, *parts):
    """use1/<별명>/… 감옥 안 경로. 밖을 가리키거나 git 안이면 Refuse."""
    base = os.path.join(use1_home(), check_alias(alias))
    path = os.path.abspath(os.path.join(base, *parts))
    if os.path.commonpath([base, path]) != base:
        raise Refuse(f"감옥 밖 경로 : {path}")
    if inside_git(base):
        raise Refuse(f"쓰는 자리가 git 안이다 : {base} — {HOME_ENV} 를 git 밖으로")
    return path


def shared_path(name):
    """별명 사이 공용 파일(use1/EXCLUDE.sha256list · servers.json)."""
    base = use1_home()
    if inside_git(base):
        raise Refuse(f"쓰는 자리가 git 안이다 : {base}")
    return os.path.join(base, name)


def part_paths(alias, place=None):
    """한 별명의 한 판 파일들. place None = ①~③ (cands.jsonl …), 4 · 5 = ④ ⑤ (cands-4.jsonl …)."""
    tail = "" if place is None else f"-{place}"
    return {"cands": alias_dir(alias, f"cands{tail}.jsonl"), "labels": alias_dir(alias, f"labels{tail}"),
            "set": alias_dir(alias, f"set-use1{tail}.jsonl"), "frozen": alias_dir(alias, f"frozen{tail}.json"),
            "mark": alias_dir(alias, f"FROZEN-use1-{alias}{tail}.sha256"),
            "meta": alias_dir(alias, "sample.json" if place is None else f"draw-{place}.json"),
            "queries": alias_dir(alias, "queries.jsonl" if place is None else f"queries-draw-{place}.jsonl"),
            "result": alias_dir(alias, f"result{tail}")}


def set_aside(path, stamp):
    """있던 파일 · 폴더를 날짜 꼬리 이름으로 돌려 둔다(지우지 않는다). 돌려 둔 경로, 없었으면 ""."""
    if not os.path.exists(path):
        return ""
    stem, ext = os.path.splitext(path)
    if os.path.isdir(path):
        stem, ext = path, ""
    target = f"{stem}-{stamp}{ext}"
    n = 2
    while os.path.exists(target):
        target = f"{stem}-{stamp}-{n}{ext}"
        n += 1
    os.replace(path, target)
    return target


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp_now():
    return now_iso().replace(":", "")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pair_key(evidence, claim):
    """쌍 열쇠 — 꼬리표를 뗀 두 문장. queries · EXCLUDE · 본 쌍 대조가 같은 열쇠를 쓴다."""
    return sha256_text(strip_tags(evidence) + "\x1f" + strip_tags(claim))


def group_key(evidence):
    return sha256_text(strip_tags(evidence))


def pair_id(place, pair, taken):
    """쌍 id = 자리 + 쌍 열쇠 앞 12자. 다시 뽑아도 같은 쌍은 같은 id 다. 같은 쌍이 겹치면 꼬리 -2 …."""
    base = f"u{place}-{pair[:12]}"
    name, n = base, 2
    while name in taken:
        name = f"{base}-{n}"
        n += 1
    taken.add(name)
    return name


def secret_kind(text):
    """키 · 토큰 · 비밀번호 · 집 경로 · IP · 메일 꼴이면 그 종류, 아니면 None. 값은 돌려주지 않는다."""
    text = text or ""
    for name, pattern in SECRETS:
        if pattern.search(text):
            return name
    for m in BARE_SECRET.finditer(text):
        value = m.group(2)
        if re.search(r"[A-Za-z]", value) and re.search(r"[0-9]", value):
            return "assign"
    return None


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def mem_command():
    """PATH 의 mem 만 부른다. 시험은 JUDGEMODEL_MEM 으로 가짜 mem(파이썬 스크립트)을 준다."""
    given = os.environ.get(MEM_ENV, "").strip()
    if given:
        if given.endswith(".py"):
            return [sys.executable, "-I", given]
        return [given]
    return ["mem"]


def run_mem(args, repo=None, llm_config=None, stdin_text=None, check=True):
    """mem 한 번. (종료 코드, stdout, stderr). 출력은 믿지 않는 글로 다룬다."""
    cmd = mem_command() + list(args)
    if repo:
        cmd += ["--repo", os.path.abspath(repo)]
    env = dict(os.environ)
    if llm_config:
        env["MEM_LLM_CONFIG"] = llm_config
    try:
        done = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, timeout=MEM_TIMEOUT_S)
    except FileNotFoundError:
        raise Refuse("mem 을 PATH 에서 못 찾았다")
    if check and done.returncode not in (0,):
        raise Refuse(f"mem {' '.join(args[:3])} 가 {done.returncode} 로 끝났다 : {done.stderr.strip()[:300]}")
    return done.returncode, done.stdout, done.stderr


def mem_json(args, repo=None, llm_config=None):
    _, out, _ = run_mem(args, repo, llm_config)
    return json.loads(out)


def mem_jsonl(args, repo=None, llm_config=None):
    """줄마다 JSON 인 mem 출력. exit 2 는 멈춘다 — 단 못 받은 줄이 모두 비밀 꼴 거절(secret-shape)이면 받는다
    (그 줄은 부르는 쪽이 refused 로 센다)."""
    code, out, err = run_mem(args, repo, llm_config, check=False)
    rows = [json.loads(line) for line in out.splitlines() if line.strip().startswith("{")]
    if code == 0:
        return rows
    problems = [r.get("problem") for r in rows if r.get("problem")]
    if code == REFUSE_EXIT and problems and all(p == MEM_REFUSED for p in problems):
        return rows
    raise Refuse(f"mem {' '.join(args[:3])} 가 {code} 로 끝났다 (못 받은 줄 {len(problems)}) : {err.strip()[:300]}")


def judge_dir_files(repo):
    """그 저장소 판정 기록 폴더(mem judge clean 이 알려 주는 곳)의 파일 수. 폴더가 없으면 0."""
    clean = mem_json(["judge", "clean", "--json"], repo=repo)
    folder = clean.get("dir", "")
    if not os.path.isdir(folder):
        return 0
    return len(os.listdir(folder))


def repo_thresholds(config):
    """mem judge config --json → (nli_sure, nli_sure_support)."""
    return {"nli_sure": float(config["nli_sure"]), "nli_sure_support": float(config["nli_sure_support"])}


def write_llm_toml(path, proxy_url, thresholds):
    """임시 llm.toml — NLI 칸만. SemIf url · key 는 절대 옮기지 않는다."""
    lines = ["# use_eval 임시 설정 — 측정 동안 MEM_LLM_CONFIG 로만 쓴다 (키 없음 · SemIf 없음)",
             f'nli_url = "{proxy_url}"', "nli_timeout_ms = 5000",
             f"nli_sure = {thresholds['nli_sure']}", f"nli_sure_support = {thresholds['nli_sure_support']}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def unsure_rule_of(protocol):
    rule = (protocol or {}).get("unsure_rule") or UNSURE_DEFAULT
    return {"below": float(rule["below"]), "tie_gap": float(rule["tie_gap"])}


def judge_answer(answer, thresholds, unsure_rule=None):
    """모델 답 하나 → (글자, 확신 여부). 답이 없으면 (None, False). mem llm.askNLI 와 같은 가름."""
    if not isinstance(answer, dict) or "a" not in answer:
        return None, False
    rule = unsure_rule or UNSURE_DEFAULT
    probs = sorted(((float(answer[k]), k) for k in LETTERS3), reverse=True)
    top, letter = probs[0]
    second = probs[1][0]
    line = thresholds["nli_sure_support"] if letter == "a" else thresholds["nli_sure"]
    unsure = top < rule["below"] or top - second < rule["tie_gap"] or top < line
    return letter, not unsure


def models_from_arg(text, protocol):
    """「v100=http://…,v110=http://…」 → {이름: 주소}. 비면 규약의 default_url."""
    out = {}
    if text:
        for part in text.split(","):
            if "=" not in part:
                raise Refuse(f"--models 꼴은 이름=주소 : {part}")
            name, url = part.split("=", 1)
            out[name.strip()] = url.strip().rstrip("/")
        return out
    for name, spec in protocol["models"].items():
        if spec.get("enabled") and spec.get("default_url"):
            out[name] = spec["default_url"]
    return out


def load_protocol(path=None):
    path = path or PROTOCOL_DEFAULT
    return read_json(path), sha256_file(path)


def place_outcome(row, model, thresholds, unsure_rule=None):
    """그 자리의 실제 결과 — ① pass·reject·unsure ② ④ flag·none ③ hidden·keep ⑤ pass·unsure·reject. 답이 없으면 None.
    ⑤ 는 G5 의 P(A) 선을 NLI 확신으로 옮긴 것 : 확신 a 통과 · 확신 b·c 되돌림 · 나머지는 SemIf 로 넘김(unsure)."""
    place, rule = row["place"], row.get("rule", "")
    if place == 1 and rule == "b":
        return "reject"
    if place in (2, 4) and rule in ("b", "c"):
        return "flag" if rule == "b" else "none"
    if place == 3 and not row.get("asked", True):
        return "keep"
    letter, sure = judge_answer((row.get("answers") or {}).get(model), thresholds, unsure_rule)
    if letter is None:
        return None
    if place in (1, 5):
        if not sure:
            return "unsure"
        return "pass" if letter == "a" else "reject"
    hit = letter == "b" and sure
    if place in (2, 4):
        return "flag" if hit else "none"
    return "hidden" if hit else "keep"
