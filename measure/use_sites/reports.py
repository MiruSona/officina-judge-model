"""④ 서브에이전트 보고 거두기 — harvest-reports · log-report (설계 `Docs/Design/2026-10-12-실사용측정설계.md` 4-1 덧붙임 2).
보고 전문에서 결정 문장을 뽑아 같은 scope 기존 mem 결정과 짝 짓고 `use1/<별명>/collect/report-pairs.jsonl` 에 덧붙인다.
"""
import datetime
import json
import os
import re
import sys

from use_sites import common

REPORT_TOOL = "SubagentHandback"
CLAIM_MIN, CLAIM_MAX, CLAIMS_PER_REPORT = 8, 200, 12
EVIDENCE_PER_CLAIM, SEARCH_LIMIT = 3, 10
DEFAULT_SCOPE = "officina"
DECIDE_WORDS = ("정한 것", "정한것", "결정")
DECIDE_HEAD = ("결정", "정한", "정함")
WHY_HEAD = ("까닭", "근거", "이유", "왜")

MEM_ID_RE = re.compile(r"\b(20\d{6}-[0-9a-f]{8})\b")
SCOPE_ARG_RE = re.compile(r"--scope[ =]+[\"']?([A-Za-z0-9_-]+)")
DECIDED_RE = re.compile(r"^\s*(?:[-*]\s+)?(?:\*\*)?정함(?:\*\*)?\s*[:：]\s*(.+)$")
MEM_ADD_RE = re.compile(r"mem add\b.*--type[ =]+decision")
SUMMARY_RE = re.compile(r"--(summary|title)[ =]+(?:\"([^\"]+)\"|'([^']+)')")
SEPARATOR_RE = re.compile(r"^\|?\s*:?-{2,}")


def sha8(text):
    return common.sha256_text(text or "")[:8]


def folded_name(repo):
    """Claude Code 가 저장소 경로를 접는 꼴 — 영문 · 숫자 밖은 모두 '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(repo))


def transcripts_dir(repo, given):
    if given:
        path = os.path.abspath(os.path.expanduser(given))
    else:
        path = os.path.join(os.path.expanduser("~"), ".claude", "projects", folded_name(repo))
    if not os.path.isdir(path):
        raise common.Refuse(f"대화 기록 폴더가 없다 : {path} — --transcripts 로 준다")
    return path


def parse_time(text):
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone()
    except (AttributeError, ValueError):
        return None


def last_report(path):
    """agent-*.jsonl 하나 → (보고 전문, 시각, 깨진 줄 수). 보고는 마지막 SubagentHandback message, 없으면 끝난 마지막 assistant 글."""
    handback, plain, bad = None, None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if row.get("type") != "assistant":
                continue
            content = (row.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            tools = [c for c in content if c.get("type") == "tool_use"]
            for c in tools:
                if c.get("name") == REPORT_TOOL and isinstance((c.get("input") or {}).get("message"), str):
                    handback = (c["input"]["message"], row.get("timestamp", ""))
            if tools:
                plain = None
            elif any(t.strip() for t in texts):
                plain = ("\n".join(texts), row.get("timestamp", ""))
    found = handback or plain
    if not found:
        return None, None, bad
    return found[0], parse_time(found[1]), bad


def clean(text):
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def cells(line):
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def is_number(cell):
    return re.fullmatch(r"[#\d.\s]*", clean(cell)) is not None


def table_claims(header, rows, titled):
    """표 하나 → 문장들. 머리줄 「결정」·「정한」 칸이 있으면 그 칸(앞에 주어 칸), 없고 제목이 결정이면 첫 칸 + 둘째 칸."""
    head = [clean(h) for h in header]
    why = {i for i, h in enumerate(head) if any(w in h for w in WHY_HEAD)}
    pick = [i for i, h in enumerate(head) if i not in why and any(w in h for w in DECIDE_HEAD)]
    if not pick and not titled:
        return []
    out = []
    for row in rows:
        usable = [i for i in range(len(row)) if i not in why and not is_number(row[i]) and clean(row[i]) not in ("", "—", "-")]
        if pick:
            value = pick[0]
            if value not in usable:
                continue
            subject = [i for i in usable if i < value][:1]
            parts = [row[i] for i in subject] + [row[value]]
        else:
            parts = [row[i] for i in usable[:2]]
        out.append(" : ".join(clean(p) for p in parts))
    return out


def tables(lines):
    """(머리줄, 몸줄들, 바로 위 글줄) 묶음."""
    found, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        start = i
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            i += 1
        block = [cells(x) for x in lines[start:i] if not SEPARATOR_RE.match(x.strip().strip("|").strip())]
        above = next((x for x in reversed(lines[:start]) if x.strip()), "")
        if len(block) >= 2:
            found.append((block[0], block[1:], above))
    return found


def extract_claims(text):
    """보고 전문 → [(문장, 종류)] (8~200자 · 12개까지 · 겹침 없음)."""
    lines = text.splitlines()
    found = []
    for header, rows, above in tables(lines):
        titled = any(w in above for w in DECIDE_WORDS)
        found += [(c, "table") for c in table_claims(header, rows, titled)]
    for line in lines:
        m = DECIDED_RE.match(line)
        if m:
            found.append((clean(m.group(1)), "decided"))
        if MEM_ADD_RE.search(line):
            got = {k: (a or b) for k, a, b in SUMMARY_RE.findall(line)}
            text_ = got.get("summary") or got.get("title")
            if text_:
                found.append((clean(text_), "mem-add"))
    out, seen = [], set()
    for claim, kind in found:
        if not CLAIM_MIN <= len(claim) <= CLAIM_MAX or claim in seen:
            continue
        seen.add(claim)
        out.append((claim, kind))
    return out[:CLAIMS_PER_REPORT]


def choose_scope(text, known, folders):
    """(scope, None) 또는 (None, 까닭). `--scope x` → 툴 폴더 이름 → officina 차례."""
    asked = [s.lower() for s in SCOPE_ARG_RE.findall(text) if s.lower() in known]
    counts = {}
    if asked:
        for s in asked:
            counts[s] = counts.get(s, 0) + 1
    else:
        for folder in folders:
            n = len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(folder) + r"(?![A-Za-z0-9])", text))
            if n:
                counts[folder.lower()] = n
    if not counts:
        if DEFAULT_SCOPE in known:
            return DEFAULT_SCOPE, None
        return None, "scope-unknown"
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, "scope-tie"
    return ranked[0][0], None


def known_scopes(repo):
    rows = common.mem_json(["search", "--type", "decision", "--json", "--facet", "scope", "--limit", "1"], repo)
    if isinstance(rows, dict):
        rows = rows.get("facets") or rows.get("scope") or []
    return {r["name"] for r in rows if isinstance(r, dict) and r.get("name")}


def tool_folders(repo, known):
    names = []
    for name in sorted(os.listdir(repo)):
        if os.path.isdir(os.path.join(repo, name)) and name.lower() in known:
            names.append(name)
    return names


def search_decisions(repo, scope, claim):
    query = claim.lstrip("-").strip() or claim
    got = common.mem_json(["search", query, "--type", "decision", "--scope", scope, "--json",
                           "--limit", str(SEARCH_LIMIT), "--no-index"], repo)
    if isinstance(got, dict):
        return got.get("hits") or []
    return got


class Harvest:
    """보고 여러 건을 같은 로그에 덧붙인다. 열쇠 겹침 · 자기 짝 · 비밀 · scope 를 센다."""

    def __init__(self, repo, alias, src):
        self.repo, self.alias, self.src = os.path.abspath(repo), alias, src
        self.pairs_path = common.alias_dir(alias, "collect", "report-pairs.jsonl")
        self.skipped_path = common.alias_dir(alias, "collect", "skipped.jsonl")
        self.seen = set()
        for r in common.read_jsonl(self.pairs_path):
            self.seen.add(r.get("pair") or common.pair_key(r.get("evidence", ""), r.get("claim", "")))
        self.skipped_seen = {(r.get("report_sha8"), r.get("why")) for r in common.read_jsonl(self.skipped_path)}
        self.known = known_scopes(self.repo)
        self.folders = tool_folders(self.repo, self.known)
        self.stats = {"reports": 0, "with_claims": 0, "claims": 0, "pairs_new": 0, "pairs_dup": 0, "refused": 0,
                      "skipped": 0, "self_dropped": 0, "later_dropped": 0, "no_evidence": 0}

    def skip(self, report_sha, why):
        self.stats["skipped"] += 1
        if (report_sha, why) in self.skipped_seen:
            return
        self.skipped_seen.add((report_sha, why))
        common.append_jsonl(self.skipped_path, [{"at": common.now_iso(), "report_sha8": report_sha, "why": why, "src": self.src}])

    def take(self, text, at, desc=""):
        """보고 하나 → 새로 덧붙인 줄 수."""
        self.stats["reports"] += 1
        claims = extract_claims(text)
        if not claims:
            return 0
        self.stats["with_claims"] += 1
        self.stats["claims"] += len(claims)
        report_sha = sha8(text)
        scope, why = choose_scope(text, self.known, self.folders)
        if not scope:
            self.skip(report_sha, why)
            return 0
        own_ids = set(MEM_ID_RE.findall(text))
        day = at.date().isoformat() if at else common.now_iso()[:10]
        base = {"report_at": at.isoformat(timespec="seconds") if at else "", "agent_desc_sha8": sha8(desc),
                "report_sha8": report_sha, "scope": scope, "src": self.src}
        rows = []
        for claim, kind in claims:
            rows += self.pairs_for(claim, kind, scope, own_ids, day, base)
        common.append_jsonl(self.pairs_path, rows)
        return len(rows)

    def pairs_for(self, claim, kind, scope, own_ids, day, base):
        if common.secret_kind(claim):
            return self.keep(self.refused_row("", claim, kind, common.secret_kind(claim), base))
        rows, kept = [], 0
        for hit in search_decisions(self.repo, scope, claim):
            if kept >= EVIDENCE_PER_CLAIM:
                break
            old_id = str(hit.get("id", ""))
            if old_id in own_ids:
                self.stats["self_dropped"] += 1
                continue
            if str(hit.get("date", "9999")) >= day:
                self.stats["later_dropped"] += 1
                continue
            evidence = hit.get("summary") or hit.get("title") or ""
            if not evidence:
                continue
            kept += 1
            reason = common.secret_kind(evidence)
            if reason:
                rows += self.keep(self.refused_row(evidence, claim, kind, reason, base, old_id))
                continue
            key = common.pair_key(evidence, claim)
            rows += self.keep({"id": "r-" + key[:12], "at": common.now_iso(), **base, "old_id": old_id,
                               "kind": kind, "evidence": evidence, "claim": claim, "pair": key})
        if kept == 0:
            self.stats["no_evidence"] += 1
        return rows

    def refused_row(self, evidence, claim, kind, reason, base, old_id=""):
        key = common.pair_key(evidence, claim)
        return {"id": "r-" + key[:12], "at": common.now_iso(), **base, "old_id": old_id, "kind": kind,
                "refused": True, "why": "secret-" + reason, "evidence": "", "claim": "", "pair": key}

    def keep(self, row):
        if row["pair"] in self.seen:
            self.stats["pairs_dup"] += 1
            return []
        self.seen.add(row["pair"])
        if row.get("refused"):
            self.stats["refused"] += 1
        else:
            self.stats["pairs_new"] += 1
        return [row]


def agent_files(folder, since):
    """세션/subagents/agent-*.jsonl 중 since 뒤에 고친 것 (차례 고정)."""
    out = []
    for root, _, names in os.walk(folder):
        if os.path.basename(root) != "subagents":
            continue
        for name in names:
            path = os.path.join(root, name)
            if name.startswith("agent-") and name.endswith(".jsonl") and os.path.getmtime(path) >= since.timestamp():
                out.append(path)
    return sorted(out)


def agent_desc(path):
    meta = path[:-len(".jsonl")] + ".meta.json"
    if not os.path.isfile(meta):
        return ""
    try:
        return str(common.read_json(meta).get("description", ""))
    except ValueError:
        return ""


def cmd_harvest(a):
    alias = a.alias or common.default_alias(a.repo)
    folder = transcripts_dir(a.repo, a.transcripts)
    since = datetime.datetime.now().astimezone() - datetime.timedelta(days=a.days)
    files = agent_files(folder, since)
    if not files:
        raise common.Refuse(f"지난 {a.days}일 서브에이전트 기록이 0건이다 — 0건을 「반대 없음」으로 읽지 않는다 : {folder}")
    harvest = Harvest(a.repo, alias, "transcript")
    bad_lines, no_report = 0, 0
    for path in files:
        try:
            text, at, bad = last_report(path)
        except OSError as e:
            raise common.Refuse(f"기록 파일을 못 읽었다 : {os.path.basename(path)} ({e})")
        bad_lines += bad
        if not text or (at and at < since):
            no_report += 1
            continue
        harvest.take(text, at, agent_desc(path))
    stats = dict(harvest.stats, files=len(files), no_report=no_report, bad_lines=bad_lines)
    common.write_json(common.alias_dir(alias, "collect", "harvest-reports.json"),
                      {"at": common.now_iso(), "days": a.days, **stats})
    print("④ 거두기 " + " · ".join(f"{k} {v}" for k, v in stats.items()) + f" → {harvest.pairs_path}")
    return stats


def cmd_log_report(a):
    alias = a.alias or common.default_alias(a.repo)
    if a.stdin:
        text = sys.stdin.read()
    elif os.path.isfile(a.file):
        with open(a.file, encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        raise common.Refuse(f"보고 파일이 없다 : {a.file}")
    harvest = Harvest(a.repo, alias, "live")
    harvest.take(text, datetime.datetime.now().astimezone(), a.desc)
    print("④ 보고 한 건 " + " · ".join(f"{k} {v}" for k, v in harvest.stats.items()))
    return harvest.stats


def register(sub):
    s = sub.add_parser("harvest-reports", help="④ 지난 대화 기록에서 서브에이전트 보고 거두기")
    s.add_argument("--repo", default=".")
    s.add_argument("--alias", default="")
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--transcripts", default="")
    s.set_defaults(func=cmd_harvest)

    s = sub.add_parser("log-report", help="④ 보고 한 건 거두기 (mod 가 부를 입구)")
    s.add_argument("--repo", default=".")
    s.add_argument("--alias", default="")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", default="")
    src.add_argument("--stdin", action="store_true")
    s.add_argument("--desc", default="")
    s.set_defaults(func=cmd_log_report)
