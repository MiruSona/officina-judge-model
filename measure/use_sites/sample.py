"""sample — 자리 ①~③ 표본 뽑기 (설계 3절 · 11-1 줄인 판). 대리 서버 → mem review 판 → ② 기록 · ③ merge 짝 · ① 다시 짓기 → cands.jsonl.
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
모든 쌍 줄에 두 모델 답(answers)을 함께 둔다 — report 는 이 파일과 라벨만 본다.
"""
import json
import os
import random
import re

from use_sites import common, proxy

C19 = "contradict-candidate"
C20 = "merge-candidate"
C13 = "stale-conflict-pair"
NLI_MARK = "NLI 반대"
REVIEW_ARGS = ["review", "--kind", "contradict,merge", "--json", "--all"]
NLI_ARGS = ["--nli", "--nli-pairs", "100000"]
PER_SCOPE_ARGS = ["--nli-per-scope", "100000"]
SENT_MIN, SENT_MAX = 8, 300
SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")
SPACE_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"「(.+)」 ↔ 「(.+)」\s*$", re.S)


# ---------------------------------------------------------------- 작은 셈

def squash(text):
    return SPACE_RE.sub(" ", common.strip_tags(text or "")).strip()


def sentences(text):
    out = []
    for part in SPLIT_RE.split(common.strip_tags(text or "")):
        part = part.strip(" |-*#\t")
        if SENT_MIN <= len(part) <= SENT_MAX:
            out.append(part)
    return out


def bigrams(text):
    t = SPACE_RE.sub("", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(x, y):
    a, b = bigrams(x), bigrams(y)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def draw(rows, size, rng):
    """size 0 이면 전부. 아니면 무작위 size. (뽑은 줄, 가중치 = 풀 / 뽑은 수)."""
    if size == 0 or len(rows) <= size:
        return list(rows), 1.0
    picked = rng.sample(rows, size)
    return picked, len(rows) / size


def pair_set(items, rule):
    return {tuple(sorted((it["id"], rel))) for it in items if it["rule"] == rule for rel in it.get("related", [])}


# ---------------------------------------------------------------- mem 판

class Session:
    """한 번의 sample · draw — 별명 · 대리 서버 · 임시 설정 · 기억 본문 캐시. place 4 · 5 는 queries-draw-N.jsonl 에 쓴다."""

    def __init__(self, a, protocol, place=None):
        self.a = a
        self.protocol = protocol
        self.alias = a.alias or common.default_alias(a.repo)
        self.stamp = common.stamp_now()
        self.rng = random.Random(protocol["sample_seed"])
        self.rule = common.unsure_rule_of(protocol)
        config = common.mem_json(["judge", "config", "--json"])
        self.thresholds = common.repo_thresholds(config)
        models = common.models_from_arg(a.models, protocol)
        log_path = common.part_paths(self.alias, place)["queries"]
        self.proxy = proxy.Proxy(models, protocol["reply"], log_path, self.alias, proxy.expected_models(protocol))
        self.memo = {}
        self.no_answer = {}
        self.refused = {}

    def run_tag(self, what):
        return f"{self.stamp}:{what}"

    def session_rows(self, what):
        tag = self.run_tag(what)
        return [r for r in common.read_jsonl(self.proxy.log_path) if r.get("run") == tag]

    def show(self, mem_id):
        if mem_id not in self.memo:
            try:
                self.memo[mem_id] = common.mem_json(["show", mem_id, "--json", "--no-index"], repo=self.a.repo)
            except (common.Refuse, ValueError):
                self.memo[mem_id] = {}
        return self.memo[mem_id]

    def review(self, reply):
        """reply 모델로 mem review --nli 한 판. reply None 이면 M0(NLI 없음)."""
        if reply is None:
            return common.mem_json(REVIEW_ARGS, repo=self.a.repo)
        self.proxy.reply, self.proxy.place, self.proxy.run = reply, "2", self.run_tag("review-" + reply)
        if getattr(self, "per_scope", True):
            try:
                report = common.mem_json(REVIEW_ARGS + NLI_ARGS + PER_SCOPE_ARGS, repo=self.a.repo, llm_config=self.toml)
                self.per_scope = True
                return report
            except common.Refuse as e:
                # 옛 mem(0.3.0 c6c0229)에는 --nli-per-scope 가 없다 — scope 당 20 상한이 남는다(manifest 에 적음)
                if "--nli-per-scope" not in str(e):
                    raise
                self.per_scope = False
        return common.mem_json(REVIEW_ARGS + NLI_ARGS, repo=self.a.repo, llm_config=self.toml)

    def judge_pairs(self, rows, place, what):
        """(id, evidence, claim) 줄 → rules 판 글자 + 대리 서버 경유 nli 판 답을 줄에 붙인다."""
        path = common.alias_dir(self.alias, "pools", f"{what}-pairs.jsonl")
        common.write_jsonl(path, [{"id": r["id"], "evidence": r["evidence"], "claim": r["claim"]} for r in rows])
        rules = {o["id"]: o for o in common.mem_jsonl(["judge", "support", "--file", path, "--stage", "rules", "--json"],
                                                      repo=self.a.repo)}
        self.proxy.reply, self.proxy.place, self.proxy.run = self.protocol["reply"], place, self.run_tag(what)
        done = common.mem_jsonl(["judge", "support", "--file", path, "--stage", "nli", "--json", "--fresh"],
                                repo=self.a.repo, llm_config=self.toml)
        refused = {o["id"] for o in done if o.get("problem") == common.MEM_REFUSED}
        answers = {r["pair"]: r["answers"] for r in self.session_rows(what)}
        missing = 0
        for r in rows:
            r["rule"] = (rules.get(r["id"], {}).get("letter") or "").lower()
            r["answers"] = answers.get(common.pair_key(r["evidence"], r["claim"]), {})
            r["asked"] = True
            if r["id"] in refused:
                r["refused"] = True
            elif not r["answers"]:
                missing += 1
        self.no_answer[what] = missing
        self.refused[what] = len(refused)
        return rows


# ---------------------------------------------------------------- 자리 ②

def flagged_by(answers, thresholds, models, rule=None):
    out = []
    for name in models:
        letter, sure = common.judge_answer(answers.get(name), thresholds, rule)
        if letter == "b" and sure:
            out.append(name)
    return out


def reconcile(session, report, reply):
    """mem 이 낸 NLI contradict 줄 수 = 대리 서버 기록으로 다시 센 수. 다르면 멈춘다 (설계 3-2 대조)."""
    mem_count = sum(1 for it in report["items"] if it["rule"] == C19 and NLI_MARK in it.get("why", ""))
    rows = session.session_rows("review-" + reply)
    ours = sum(1 for r in rows if flagged_by(r["answers"], session.thresholds, [reply], session.rule))
    if mem_count != ours:
        raise common.Refuse(f"대조 어긋남 : reply {reply} 판 mem contradict(NLI) {mem_count}줄 ≠ 기록으로 센 {ours}줄")
    return {"mem_nli_contradict": mem_count, "recount": ours, "asked": len(rows)}


def place2(session, strata):
    models = list(session.proxy.models)
    seen, pool = set(), []
    for reply in session.replies:
        for r in session.session_rows("review-" + reply):
            if r["pair"] in seen:
                continue
            seen.add(r["pair"])
            pool.append({"evidence": r["evidence"], "claim": r["claim"], "answers": r["answers"], "src": "c19"})
    flagged = [r for r in pool if flagged_by(r["answers"], session.thresholds, models, session.rule)]
    rest = [r for r in pool if not flagged_by(r["answers"], session.thresholds, models, session.rule)]
    out = []
    for name, rows in (("flagged", flagged), ("rest", rest)):
        picked, weight = draw(rows, strata.get(name, 0), session.rng)
        for r in picked:
            out.append({**r, "place": 2, "stratum": name, "weight": weight, "pool": len(rows), "asked": True})
    if strata.get("plus-superseded"):
        out += superseded(session, strata["plus-superseded"])
    return out


def superseded(session, size):
    """retired 짝(옛 id, 새 id)의 summary 쌍 — b 재현 보탬 층 (따로 보고)."""
    _, text, _ = common.run_mem(["review", "--kind", "retired", "--ids", "--all"], repo=session.a.repo)
    pairs = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3].strip():
            pairs.append((parts[2], parts[3].split(",")[0].strip()))
    picked, weight = draw(pairs, size, session.rng)
    rows = []
    for i, (old, new) in enumerate(picked):
        ev, cl = session.show(old).get("summary", ""), session.show(new).get("summary", "")
        if ev.strip() and cl.strip():
            rows.append({"id": f"sup{i}", "evidence": ev, "claim": cl, "src": "retired-summary"})
    rows = session.judge_pairs(rows, "2", "superseded")
    return [{**r, "place": 2, "stratum": "plus-superseded", "weight": weight, "pool": len(pairs)} for r in rows]


# ---------------------------------------------------------------- 자리 ③

def place3(session):
    m0 = pair_set(session.queues["m0"]["items"], C20)
    runs = {reply: pair_set(session.queues["review-" + reply]["items"], C20) for reply in session.replies}
    logged = []
    for reply in session.replies:
        logged += session.session_rows("review-" + reply)
    out = []
    for left, right in sorted(m0):
        row, matches = merge_text(session, left, right, logged)
        hidden_in = [reply for reply, kept in runs.items() if (left, right) not in kept]
        check_hidden(session, left, right, hidden_in, matches)
        row.update({"place": 3, "stratum": "m0", "weight": 1.0, "pool": len(m0), "mem_ids": [left, right],
                    "hidden_in": hidden_in})
        out.append(row)
    return out


def check_hidden(session, left, right, hidden_in, matches):
    """reply 모델마다 mem 이 실제로 숨긴 것(hidden_in) = 기록 답으로 다시 센 것. 다르면 멈춘다."""
    for reply in session.replies:
        own = [m for m in matches if m.get("run") == session.run_tag("review-" + reply)] or matches
        recount = any(flagged_by(m["answers"], session.thresholds, [reply], session.rule) for m in own)
        if recount != (reply in hidden_in):
            raise common.Refuse(f"③ 대조 어긋남 : merge 짝 {left}·{right} — reply {reply} 판 mem 숨김 {reply in hidden_in}"
                                f" ≠ 기록으로 센 {recount}")


def c19_quotes(session, left, right):
    """review 판들의 C19(NLI) 줄 중 이 두 기억 id 짝의 까닭 글에서 뽑은 문장 쌍 {frozenset(문장 둘)}."""
    out = set()
    for name, report in session.queues.items():
        if not name.startswith("review-"):
            continue
        for it in report["items"]:
            if it["rule"] != C19 or {it["id"], *it.get("related", [])} != {left, right}:
                continue
            m = QUOTE_RE.search(it.get("why", ""))
            if m:
                out.add(frozenset((squash(m.group(1)), squash(m.group(2)))))
    return out


def merge_text(session, left, right, logged):
    """merge 짝의 C19 글 → (줄, 맞은 기록들). 기억 id 짝의 C19 까닭 글 문장 → 없으면 두 본문에 든 문장 쌍이 하나뿐일 때.
    못 찾으면 summary 쌍 (물은 적 없음)."""
    quoted = c19_quotes(session, left, right)
    matches = [r for r in logged if frozenset((squash(r["evidence"]), squash(r["claim"]))) in quoted]
    src = "c19"
    if not matches:
        texts = {}
        for mem_id in (left, right):
            m = session.show(mem_id)
            texts[mem_id] = squash(m.get("summary", "") + "\n" + m.get("body", ""))
        inside = [r for r in logged if in_pair(squash(r["evidence"]), squash(r["claim"]), texts[left], texts[right])]
        if len({r["pair"] for r in inside}) == 1:
            matches = inside
        elif inside:
            src = "summary-ambiguous"
    if matches:
        flagged = [r for r in matches if flagged_by(r["answers"], session.thresholds, session.replies, session.rule)]
        pick = (flagged or matches)[0]
        return {"evidence": pick["evidence"], "claim": pick["claim"], "answers": pick["answers"], "src": "c19",
                "asked": True}, matches
    ml, mr = session.show(left), session.show(right)
    first, second = (ml, mr) if ml.get("date", "") <= mr.get("date", "") else (mr, ml)
    if src == "c19":
        src = "summary"
    return {"evidence": first.get("summary", ""), "claim": second.get("summary", ""), "answers": {},
            "src": src, "asked": False}, []


def in_pair(ev, cl, left_text, right_text):
    if not ev or not cl:
        return False
    return (ev in left_text and cl in right_text) or (ev in right_text and cl in left_text)


# ---------------------------------------------------------------- 자리 ①

def place1(session, strata):
    ids = set()
    for name, report in session.queues.items():
        for it in report["items"]:
            if it["rule"] in (C19, C20, C13):
                ids.add(it["id"])
                ids.update(it.get("related", []))
    ids = sorted(ids)
    session.rng.shuffle(ids)
    hard, own = [], []
    for mem_id in ids:
        if len(hard) >= strata.get("replay-hard", 0) and len(own) >= strata.get("replay-own", 0):
            break
        add_replays(session, mem_id, hard, own, strata)
    rows = [dict(r, id=f"r{i}") for i, r in enumerate(hard + own)]
    rows = session.judge_pairs(rows, "1", "retain")
    return [{**r, "place": 1, "weight": 1.0} for r in rows]


def add_replays(session, mem_id, hard, own, strata):
    m = session.show(mem_id)
    summary = (m.get("summary") or "").strip()
    if not summary:
        return
    mine = [s for s in sentences(m.get("body", "")) if squash(s) != squash(summary)]
    if mine and len(own) < strata.get("replay-own", 0):
        own.append({"evidence": session.rng.choice(mine), "claim": summary, "stratum": "replay-own",
                    "src": "body", "mem_ids": [mem_id]})
    if len(hard) >= strata.get("replay-hard", 0):
        return
    best, best_id = None, ""
    for other in neighbours(session, mem_id):
        for s in sentences(session.show(other).get("body", "")):
            score = similarity(s, summary)
            if best is None or score > best[0]:
                best, best_id = (score, s), other
    if best is not None:
        hard.append({"evidence": best[1], "claim": summary, "stratum": "replay-hard", "src": "neighbour",
                     "mem_ids": [mem_id, best_id], "sim": round(best[0], 3)})


def neighbours(session, mem_id):
    out = []
    for report in session.queues.values():
        for it in report["items"]:
            if it["id"] == mem_id:
                out += it.get("related", [])
            elif mem_id in it.get("related", []):
                out.append(it["id"])
    return sorted(set(out))


# ---------------------------------------------------------------- 본 쌍 대조

def seen_keys(roots):
    """학습 · 셋 jsonl 의 쌍 열쇠. 줄마다 evidence · claim 이 있는 것만."""
    keys = set()
    for root in roots:
        for folder, _, files in os.walk(os.path.expanduser(root)):
            for name in files:
                if name.endswith(".jsonl"):
                    collect_keys(os.path.join(folder, name), keys)
    return keys


def collect_keys(path, keys):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"evidence"' not in line or '"claim"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row.get("evidence"), str) and isinstance(row.get("claim"), str):
                keys.add(common.pair_key(row["evidence"], row["claim"]))


# ---------------------------------------------------------------- 명령

def guard_labels(alias, force, stamp):
    """labels/ 에 라벨이 있으면 멈춘다 — 다시 뽑으면 옛 라벨이 새 표본과 어긋난다. --force 면 날짜 꼬리로 돌려 둔다."""
    folder = common.part_paths(alias)["labels"]
    if not os.path.isdir(folder) or not os.listdir(folder):
        return ""
    if not force:
        raise common.Refuse(f"{folder} 에 라벨이 있다 — 다시 뽑지 않는다 (옛 라벨을 돌려 두고 뽑으려면 --force)")
    return common.set_aside(folder, stamp)


def finish_rows(rows, keys):
    """쌍 열쇠 · 묶음 열쇠 · 본 쌍 표시 · 해시 id 를 단다."""
    taken = set()
    for r in rows:
        r["pair"] = common.pair_key(r["evidence"], r["claim"])
        r["group"] = common.group_key(r["evidence"])
        r["seen"] = r["pair"] in keys
        r["id"] = common.pair_id(r["place"], r["pair"], taken)
    return rows


def seen_roots(text):
    if text == "none":
        return []
    return [r for r in text.split(",") if r]


def warn_no_answer(no_answer):
    total = sum(no_answer.values())
    if total:
        print(f"경고 : 물었는데 답을 못 찾은 쌍 {total} ({no_answer}) — 그 쌍의 결과는 None(셈에서 빠짐)")


def cmd_sample(a):
    protocol, protocol_sha = common.load_protocol(a.protocol)
    places = {int(p) for p in a.places.split(",") if p.strip()}
    if places - {1, 2, 3}:
        raise common.Refuse("sample 은 자리 1~3 만 뽑는다 (④⑤ 는 draw)")
    alias = common.check_alias(a.alias or common.default_alias(a.repo))
    paths = common.part_paths(alias)
    if os.path.isfile(paths["mark"]):
        raise common.Refuse("이 별명은 이미 얼렸다 — 다시 뽑지 않는다 (지킬 규칙 ①)")
    set_aside_labels = guard_labels(alias, a.force, common.stamp_now())
    a.alias = alias
    session = Session(a, protocol)
    status = session.proxy.probe()
    url = session.proxy.start()
    session.toml = common.write_llm_toml(common.alias_dir(alias, "llm-repo.toml"), url, session.thresholds)
    session.replies = [n for n in session.proxy.models if session.proxy.usable(n)]
    try:
        manifest = run_places(session, places)
    finally:
        session.proxy.stop()
    rows = manifest.pop("_rows")
    roots = seen_roots(a.seen)
    keys = seen_keys(roots)
    finish_rows(rows, keys)
    common.set_aside(paths["cands"], session.stamp)
    common.write_jsonl(paths["cands"], rows)
    manifest.update({"at": common.now_iso(), "alias": alias, "stamp": session.stamp,
                     "protocol_sha256": protocol_sha, "thresholds": {"repo": session.thresholds},
                     "models": status, "places": sorted(places), "seen_roots": roots, "seen_keys": len(keys),
                     "counts": count_strata(rows), "seen_pairs": sum(1 for r in rows if r["seen"]),
                     "no_answer": session.no_answer, "refused": session.refused,
                     "judge_dir_files_after": common.judge_dir_files(a.repo), "labels_set_aside": set_aside_labels})
    common.write_json(paths["meta"], manifest)
    print(f"cands {len(rows)}쌍 → {paths['cands']}")
    for key, n in sorted(manifest["counts"].items()):
        print(f"  {key} {n}")
    for name, s in status.items():
        print(f"  모델 {name} : {'ok ' + str(s.get('name')) if s['ok'] else UNAVAILABLE_NOTE + s['why']}")
    warn_no_answer(session.no_answer)
    if set_aside_labels:
        print(f"  옛 라벨을 돌려 뒀다 → {set_aside_labels}")
    return manifest


UNAVAILABLE_NOTE = "unavailable — "


def run_places(session, places):
    strata = {k.split("-", 1)[0]: v.get("strata", {}) for k, v in session.protocol["places"].items()}
    session.queues = {"m0": session.review(None)}
    common.write_json(common.alias_dir(session.alias, "queues", "m0.json"), session.queues["m0"])
    queue_rows = {"m0": queue_counts(session.queues["m0"])}
    checks = {}
    for reply in session.replies:
        report = session.review(reply)
        session.queues["review-" + reply] = report
        common.write_json(common.alias_dir(session.alias, "queues", f"review-{reply}.json"), report)
        queue_rows["review-" + reply] = queue_counts(report)
        queue_rows["review-" + reply]["hidden_merge"] = len(pair_set(session.queues["m0"]["items"], C20)
                                                           - pair_set(report["items"], C20))
        if reply == session.protocol["reply"]:
            checks[reply] = reconcile(session, report, reply)
    rows = []
    if 2 in places:
        rows += place2(session, strata["2"])
    if 3 in places:
        rows += place3(session)
    if 1 in places:
        rows += place1(session, strata["1"])
    return {"_rows": rows, "queues": queue_rows, "reconcile": checks, "memories": session.queues["m0"].get("checked"),
            "nli_per_scope_flag": getattr(session, "per_scope", None)}


def queue_counts(report):
    items = report["items"]
    return {"contradict": sum(1 for it in items if it["kind"] == "contradict"),
            "c19": sum(1 for it in items if it["rule"] == C19),
            "c19_nli": sum(1 for it in items if it["rule"] == C19 and NLI_MARK in it.get("why", "")),
            "merge": sum(1 for it in items if it["kind"] == "merge"), "nli": report.get("nli")}


def count_strata(rows):
    out = {}
    for r in rows:
        key = f"{r['place']}-{r['stratum']}"
        out[key] = out.get(key, 0) + 1
    return out


def register(sub):
    home = os.path.join("~", ".judgemodel")
    s = sub.add_parser("sample", help="자리 ①~③ 표본 (대리 서버 · review 판 · 다시 짓기 · 본 쌍 대조)")
    s.add_argument("--repo", default=".")
    s.add_argument("--alias", default="")
    s.add_argument("--models", default="", help="v100=http://127.0.0.1:8093,v110=http://127.0.0.1:8092")
    s.add_argument("--places", default="1,2,3")
    s.add_argument("--place", dest="places", help="--places 와 같다")
    s.add_argument("--seen", default=f"{os.path.join(home, 'sets')},{os.path.join(home, 'build')}",
                   help="본 쌍 대조 폴더 (쉼표) · none 이면 안 본다")
    s.add_argument("--protocol", default="")
    s.add_argument("--force", action="store_true", help="labels/ 에 라벨이 있어도 뽑는다 (옛 labels 는 날짜 꼬리로 돌려 둔다)")
    s.set_defaults(func=cmd_sample)
