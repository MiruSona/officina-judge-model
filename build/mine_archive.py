"""mine_archive — 우리 아카이브(git · 문서 · mem)에서 열쇠 묶음과 후보 쌍을 뽑는다 (설계 d자료층재구축 4-1 · 4-2).

하위 명령 셋 : git · docs · mem. 후보 줄 꼴은 mine_public 과 같다(`want` 비움 · `hint` 는 라벨 아님).
쓰는 것 : <out>/archive/git · <out>/archive/docs · <out>/mem 아래 candidates.jsonl · groups.json.
쓰기 전에 모든 줄을 scrub 규칙으로 거른다. 저장소 · 기억 원본은 읽기만 한다.

    python -I mine_archive.py git  --repo studio=<저장소> --out <build>/g1 --names <목록> --game-names <목록>
    python -I mine_archive.py docs --root studio=<저장소> --out <build>/g1 --names … --game-names …
    python -I mine_archive.py mem  --store studio=<Memory/store> --dev-r2 <dev-r2.jsonl> --old-root ~/.judgemodel --out <build>/g1
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_sets  # noqa: E402
import mine_public as mp  # noqa: E402
import scrub  # noqa: E402
from bm25 import BM25  # noqa: E402

SEED = mp.SEED
FILE_COMMITS = (2, 30)        # 묶음이 되는 파일 : 고친 커밋 수 범위 (넘으면 모든 일에 걸리는 허브 파일)
NARROW_COMMIT = 10            # d 짝에 쓰는 커밋 : 고친 파일이 이 수 이하
MEM_SOURCE_CAP = 10           # 기억 묶음 : 이보다 많은 기억이 가리키는 출처 파일은 열쇠로 안 쓴다
OLD_WANT_HINT = {"support": "a", "contradict": "b", "unrelated": "c", "related": "d"}

_TRAILER = re.compile(r"^\s*(co-authored-by|signed-off-by|claude-session|change-id)\s*:", re.I)
_REVERT = re.compile(r"^revert\b|되돌[리림려렸]", re.I)
_VALUE_CHANGE = re.compile(r"([\w.]+)\s*(?:→|->)\s*\S")
_REVERSAL = re.compile(r"뒤집|바꿨다|바꾼다|바꿈|접는다|접었다|접음|철회|폐기|기각|취소|더는 안|안 쓴다|않기로")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_LIST_MARK = re.compile(r"^\s*(?:[-*+>]|\d+[.)])\s+")
_DOC_REF = re.compile(r"[\w가-힣.\-/]+\.md\b")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_KO_END = re.compile(r"(다|요|까|라|자|지|네|게)[\s.!?)」』'\"]*$")
_EN_VERB = re.compile(r"\b(is|are|was|were|be|has|have|had|do|does|did|can|will|should|must|uses?|adds?|keeps?|runs?|"
                      r"makes?|returns?|fix(es|ed)?|changed?|removed?|added|moved?)\b", re.I)
_META_LINE = re.compile(r"^(쓴 사람|바탕|날짜|상태|작성|기준|관련|참고)\s*[:：]")
_MEM_PART = re.compile(r"^(왜|무엇을|무엇|다음|결정|주의|까닭|결과|함정|방법|할 일)\s*[:：]\s*")
_MEM_ID = re.compile(r"\b\d{8}-[0-9a-f]{8}\b")


# ---------------------------------------------------------------- 거르기

def clean_inline(text):
    """마크다운 꾸밈을 벗긴다 (링크 → 글 · 따옴 코드 → 글 · 굵게 · 목록 표시)."""
    text = _MD_LINK.sub(r"\1", text)
    text = text.replace("`", "").replace("**", "").replace("__", "")
    text = _LIST_MARK.sub("", text)
    return text.strip().lstrip("#").strip()


def _ends_with_noun_verb(text):
    """「고침 · 바꿈 · 접음」처럼 ㅁ 받침으로 끝나는 한국어 명사형 끝맺음."""
    tail = text.rstrip(" .!?)」』'\"")
    if not tail:
        return False
    ch = tail[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 16


def has_predicate(text):
    """동사(서술어)가 있나 — 한국어 끝맺음 · 명사형 끝맺음 · 영어 동사 낱말."""
    return bool(_KO_END.search(text) or _ends_with_noun_verb(text) or _EN_VERB.search(text)
                or re.search(r"(한다|했다|된다|됐다|있다|없다|않는다|는다)[\s,]", text))


def is_menu_like(text):
    """목록 · 제목 · 메뉴만인 글 : 서술어 없는 짧은 토막이거나 낱말을 늘어놓기만 한 글."""
    if has_predicate(text):
        return False
    if len(text) < 40:
        return True
    return text.count(",") + text.count("·") + text.count("/") + text.count("|") >= 3


def is_line_heavy(raw):
    """줄 넘김만 많은 글 : 세 줄 넘게 끊겼는데 한 줄이 평균 25자도 안 된다."""
    lines = [line for line in raw.split("\n") if line.strip()]
    return len(lines) > 3 and sum(len(line) for line in lines) / len(lines) < 25


def arch_text_ok(text):
    """설계 3-1 공통 거르기(길이 · URL · 빈 글 · 코드 덩이) + 목록 · 메뉴 버림. 따옴 코드는 허용한다."""
    if not text:
        return False
    text = text.strip()
    if not mp.MIN_LEN <= len(text) <= mp.MAX_LEN:
        return False
    if mp._URL.search(text) or mp._TRIVIAL.match(text) or text.endswith(":"):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.5 * len(text.replace(" ", "")):
        return False
    code_marks = sum(text.count(mark) for mark in ("{", "}", ";", "==", "=>", "()", "$", "::", "</"))
    if code_marks > 2:
        return False
    return not is_menu_like(text)


def lang_of(text):
    hangul = len(_HANGUL.findall(text))
    latin = len(_LATIN.findall(text))
    if not hangul + latin:
        return "ko"
    ratio = hangul / (hangul + latin)
    return "ko" if ratio >= 0.3 else ("en" if ratio < 0.05 else "mix")


def units_of(raw):
    """여러 줄 글 → 문장 단위. 줄마다 꾸밈을 벗기고 한국어 문장 끝으로 자른다."""
    out = []
    for line in raw.split("\n"):
        line = clean_inline(line)
        if not line:
            continue
        out.extend(s for s in mp.split_sentences(line, "ko") if s)
    return out


class ArchiveCollector(mp.Collector):
    """mine_public.Collector 와 같은 꼴 · 상한으로 쓰되 거르기만 아카이브용으로 바꾼다."""

    def __init__(self, source, domain, origin="g1-ga"):
        super().__init__(source)
        self.domain = domain
        self.origin = origin

    def add(self, evidence, claim, hint, form, groups, lang, split, src_ids, lic="", needs_rewrite=False, **extra):
        key = (groups[0], hint, form)
        if self.per_cap[key] >= mp.PER_GROUP_CAP:
            return False
        evidence = " ".join(evidence.split())
        claim = " ".join(claim.split())
        if not arch_text_ok(evidence):
            self.dropped["evidence"] += 1
            return False
        if not arch_text_ok(claim):
            self.dropped["claim"] += 1
            return False
        if make_sets.norm(evidence) == make_sets.norm(claim):
            self.dropped["same"] += 1
            return False
        self.per_cap[key] += 1
        row = {
            "evidence": evidence, "claim": claim, "want": "", "hint": hint, "form": form,
            "domain": self.domain, "group": groups[0], "groups": sorted(set(groups)),
            "lang": lang or lang_of(evidence + " " + claim), "lic": lic, "split": split,
            "src_ids": list(src_ids), "origin": self.origin, "source": self.source,
        }
        row.update(extra)
        self.rows.append(row)
        return True

    def add_raw(self, row):
        """이미 쌍인 줄(dev-r2 다시 붙이기)을 거르기 없이 넣는다."""
        self.rows.append(row)

    def scrub(self, scrubber):
        counts = collections.Counter()
        self.rows = [r for r in (scrubber.apply(row, counts) for row in self.rows) if r is not None]
        self.dropped.update({"scrub_" + k: v for k, v in counts.items() if k.startswith("drop")})
        return counts



def _best(index, query, allowed, skip=(), top=1):
    return [j for j in index.rank(query, exclude=skip) if j in allowed][:top]


def _labelled(values):
    out = []
    for value in values or []:
        label, _, path = value.partition("=")
        if not path:
            label, path = os.path.basename(os.path.normpath(value)), value
        out.append((label, path))
    return out


# ---------------------------------------------------------------- git

def read_git(repo_path):
    """저장소 → [{hash, units:[(종류, 글)], files:[경로]}] (병합 커밋 뺌)."""
    out = subprocess.run(["git", "-C", repo_path, "log", "--no-merges", "--format=%x1e%H%x1f%B%x1f", "--name-only"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    commits = []
    for record in out.split("\x1e"):
        if "\x1f" not in record:
            continue
        sha, body, files = record.split("\x1f", 2)
        lines = [line for line in body.strip().split("\n") if not _TRAILER.match(line)]
        if not lines:
            continue
        units = [("subject", clean_inline(lines[0]))]
        body_text = "\n".join(lines[1:]).strip()
        if body_text and not is_line_heavy(body_text):
            units.extend(("body", u) for u in units_of(body_text))
        paths = [f.strip() for f in files.strip().split("\n") if f.strip()]
        commits.append({"hash": sha.strip()[:12], "units": units, "files": paths})
    return commits


def _path_words(path):
    stem = os.path.splitext(path)[0]
    return " ".join(w for w in re.split(r"[/\\_.\-]+", stem) if len(w) >= 2)


def mine_git(repos, rng):
    col = ArchiveCollector("git", "archive")
    for label, path in repos:
        commits = read_git(path)
        skip_dirs = {"Memory"}
        counts = collections.Counter(f for c in commits for f in set(c["files"]) if f.split("/")[0] not in skip_dirs)
        by_file = collections.defaultdict(list)
        for commit in commits:
            commit["eligible"] = [f for f in commit["files"] if FILE_COMMITS[0] <= counts.get(f, 0) <= FILE_COMMITS[1]]
            for f in commit["eligible"]:
                by_file[f].append(commit)
        all_units = [(c, kind, text) for c in commits for kind, text in c["units"]]
        index = BM25([text for _, _, text in all_units])
        for f in sorted(by_file):
            group = col.group(f"git:{label}:{f}", "any", title=f)
            narrow = [c for c in by_file[f] if len(c["files"]) <= NARROW_COMMIT]
            words = _path_words(f)
            # d : 같은 파일을 고친 서로 다른 커밋 메시지 (파일 이름 낱말에 가까운 글끼리)
            for i, left in enumerate(narrow):
                for right in narrow[i + 1:i + 3]:
                    lt = _pick_unit(left, words)
                    rt = _pick_unit(right, words)
                    if lt and rt:
                        groups = [group] + [f"git:{label}:{g}" for g in set(left["eligible"]) | set(right["eligible"])]
                        col.add(lt, rt, "d", "same-file", groups, "", "any", [left["hash"], right["hash"]])
            # b : 되돌림 · 값 바꿈 커밋 ↔ 앞 커밋 (log 는 새 것부터)
            ordered = by_file[f]
            for i, commit in enumerate(ordered[:-1]):
                prev = ordered[i + 1]
                for kind, text in commit["units"]:
                    if _REVERT.search(text) or _VALUE_CHANGE.search(text):
                        pt = _pick_unit(prev, text, _old_value(text))
                        if pt:
                            groups = [group] + [f"git:{label}:{g}" for g in set(commit["eligible"]) | set(prev["eligible"])]
                            col.add(pt, text, "b", "revert" if _REVERT.search(text) else "value-change", groups, "",
                                    "any", [prev["hash"], commit["hash"]])
                        break
            # a : 한 커밋의 제목 ↔ 가장 닮은 본문 줄
            for commit in rng.sample(narrow, min(3, len(narrow))):
                subject = commit["units"][0][1]
                bodies = [t for k, t in commit["units"] if k == "body"]
                if bodies:
                    best = BM25(bodies).rank(subject)[:1]
                    if best:
                        col.add(subject, bodies[best[0]], "a", "subject-body",
                                [group] + [f"git:{label}:{g}" for g in commit["eligible"]], "", "any", [commit["hash"]])
            # c : 같은 저장소 · 이 파일을 안 고친 커밋의 근접 글
            for commit in rng.sample(narrow, min(2, len(narrow))):
                subject = commit["units"][0][1]
                near = [j for j in index.rank(subject)[:40] if f not in all_units[j][0]["files"]][:10]
                if near:
                    other, _, text = all_units[rng.choice(near)]
                    groups = [group] + [f"git:{label}:{g}" for g in set(commit["eligible"]) | set(other["eligible"])]
                    col.add(subject, text, "c", "other-file", groups, "", "any", [commit["hash"], other["hash"]])
    return col


def _old_value(text):
    """「A → B」의 A — 앞 커밋 글에 A 가 있어야 같은 값 자리를 말한다고 본다. 되돌림은 조건 없음."""
    if _REVERT.search(text):
        return ""
    match = _VALUE_CHANGE.search(text)
    return match.group(1) if match else ""


def _pick_unit(commit, query, must=""):
    texts = [t for _, t in commit["units"] if arch_text_ok(t) and (not must or must in t)]
    if not texts:
        return ""
    ranked = BM25(texts).rank(query)
    return texts[ranked[0]] if ranked else texts[0]


# ---------------------------------------------------------------- docs

def parse_doc(text):
    """md 문서 → {intro:[문장], sections:[(머리, [문장])], decisions:[(주제, 글)]}. 코드 덩이 · 일반 표는 버린다."""
    intro, sections, decisions = [], [], []
    heading, sentences, in_fence, seen_h2 = "(head)", [], False, False
    decision_table = False
    paragraph = []

    def flush_para():
        if paragraph:
            raw = "\n".join(paragraph)
            if not is_line_heavy(raw):
                target = intro if not seen_h2 else sentences
                for line in paragraph:
                    target.extend(units_of(line))
            paragraph.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            flush_para()
            continue
        if in_fence:
            continue
        head = re.match(r"(#{1,6})\s+(.*)", stripped)
        if head:
            flush_para()
            if len(head.group(1)) >= 2:
                if seen_h2 and sentences:
                    sections.append((heading, list(sentences)))
                sentences.clear()
                heading, seen_h2 = head.group(2).strip(), True
                decision_table = bool(re.search(r"정한 것|결정", heading))
            continue
        if stripped.startswith("|"):
            flush_para()
            cells = [clean_inline(c) for c in stripped.strip("|").split("|")]
            if decision_table and len(cells) >= 2 and not re.fullmatch(r":?-{2,}:?", cells[0].replace(" ", "")):
                topic, body = cells[0], " ".join(c for c in cells[1:] if c)
                if topic and body and topic not in ("무엇", "주제", "항목", "#", "정한 것", "결정"):
                    decisions.append((topic, f"{topic} : {body}"))
            continue
        if not stripped:
            flush_para()
            continue
        if _META_LINE.match(clean_inline(stripped)):
            continue
        paragraph.append(stripped)
    flush_para()
    if seen_h2 and sentences:
        sections.append((heading, sentences))
    return {"intro": intro, "sections": sections, "decisions": decisions}


def read_docs(roots):
    docs = []
    for label, root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)):
            rel = os.path.relpath(path, root).replace("\\", "/")
            parts = rel.split("/")
            if "Docs" not in parts[:-1] or any(p in ("Memory", ".git", "node_modules", "Tools") for p in parts):
                continue
            if rel.endswith(".local.md"):
                continue
            doc = parse_doc(mp.read_text(path))
            doc.update({"label": label, "rel": rel, "tool": parts[0] if parts[0] != "Docs" else "_root",
                        "group": f"doc:{label}:{rel}", "name": parts[-1]})
            doc["flat"] = [(h, s) for h, ss in doc["sections"] for s in ss if arch_text_ok(s)]
            doc["intro_ok"] = [s for s in doc["intro"] if arch_text_ok(s)]
            docs.append(doc)
    return docs


def mine_docs(roots, rng):
    col = ArchiveCollector("docs", "archive")
    docs = read_docs(roots)
    by_name = collections.defaultdict(list)
    for doc in docs:
        col.group(doc["group"], "any", title=doc["rel"])
        by_name[(doc["label"], doc["name"])].append(doc)
    for doc in docs:
        texts = [s for _, s in doc["flat"]]
        heads = [h for h, _ in doc["flat"]]
        src = [doc["rel"]]
        if len(texts) >= 2:
            index = BM25(texts)
            # d same-doc : 다른 절에서 가장 가까운 문장
            for i in rng.sample(range(len(texts)), min(4, len(texts))):
                other = set(j for j, h in enumerate(heads) if h != heads[i])
                picks = _best(index, texts[i], other, top=5)
                if picks:
                    col.add(texts[i], texts[rng.choice(picks)], "d", "same-doc", [doc["group"]], "", "any", src)
            # d near : 문서 안 BM25 10~60 위
            for i in rng.sample(range(len(texts)), min(3, len(texts))):
                band = index.near(texts[i], exclude=range(i - 2, i + 3))
                if band:
                    col.add(texts[i], texts[rng.choice(band)], "d", "near", [doc["group"]], "", "any", src)
            # a : 요약(머리글) ↔ 본문의 가장 닮은 문장
            for summary in [s for s in doc["intro_ok"] if len(s) >= 25][:3]:
                picks = index.rank(summary)[:1]
                if picks:
                    col.add(summary, texts[picks[0]], "a", "summary-body", [doc["group"]], "", "any", src)
    _doc_cross(col, docs, by_name, rng)
    return col


def _doc_cross(col, docs, by_name, rng):
    """문서 사이 짝 : 인수인계 참조 · 정한 것 주제(d link) · 뒤집음(b) · 같은 툴 다른 문서(c)."""
    for doc in docs:
        if "인수인계" not in doc["name"]:
            continue
        for _, sentence in doc["flat"] + [("", s) for s in doc["intro_ok"]]:
            for ref in _DOC_REF.findall(sentence):
                for target in by_name.get((doc["label"], os.path.basename(ref)), []):
                    if target is doc or not target["flat"]:
                        continue
                    texts = [s for _, s in target["flat"]]
                    picks = BM25(texts).rank(sentence)[:1]
                    if picks:
                        col.add(texts[picks[0]], sentence, "d", "link", [target["group"], doc["group"]], "", "any",
                                [target["rel"], doc["rel"]], via="handover")
    topics = collections.defaultdict(list)
    for doc in docs:
        for topic, text in doc["decisions"]:
            topics[make_sets.norm(topic)].append((doc, text))
    for topic, items in sorted(topics.items()):
        owners = {id(d) for d, _ in items}
        if len(owners) < 2:
            continue
        (left_doc, left), (right_doc, right) = items[0], next(i for i in items if i[0] is not items[0][0])
        col.add(left, right, "d", "link", [left_doc["group"], right_doc["group"]], "", "any",
                [left_doc["rel"], right_doc["rel"]], via="decision-topic")
    by_tool = collections.defaultdict(list)
    for doc in docs:
        by_tool[(doc["label"], doc["tool"])].append(doc)
    for tool_docs in by_tool.values():
        reps, owners = [], []
        for doc in tool_docs:
            for _, s in doc["flat"][:8]:
                reps.append(s)
                owners.append(doc)
        if len(set(map(id, owners))) < 2:
            continue
        index = BM25(reps)
        for doc in tool_docs:
            texts = [s for _, s in doc["flat"]]
            # b : 「뒤집음 · 바꿈」 문장 ↔ 같은 툴 다른 문서의 가장 가까운 문장
            for sentence in [s for s in texts if _REVERSAL.search(s)][:2]:
                picks = [j for j in index.rank(sentence) if owners[j] is not doc][:1]
                if picks:
                    other = owners[picks[0]]
                    col.add(reps[picks[0]], sentence, "b", "reversal", [doc["group"], other["group"]], "", "any",
                            [other["rel"], doc["rel"]])
            # c : 같은 툴 다른 문서의 근접 문장
            for sentence in rng.sample(texts, min(2, len(texts))):
                picks = [j for j in index.rank(sentence) if owners[j] is not doc][:10]
                if picks:
                    j = rng.choice(picks)
                    col.add(reps[j], sentence, "c", "other-doc", [doc["group"], owners[j]["group"]], "", "any",
                            [owners[j]["rel"], doc["rel"]])


# ---------------------------------------------------------------- mem

def read_mem_store(store_path, label):
    """Memory/store → {id: {title, summary, scope, sources, superseded_by, parts:[(부분, 문장)]}}."""
    mems = {}
    for path in sorted(glob.glob(os.path.join(store_path, "**", "*.md"), recursive=True)):
        text = mp.read_text(path).replace("\r\n", "\n")
        if not text.startswith("---"):
            continue
        _, front, body = text.split("---", 2)
        field = lambda name: (re.search(rf"^{name}:\s*(.*)$", front, re.M) or [None, ""])[1].strip().strip("'\"")  # noqa: E731
        mid = field("id")
        if not mid:
            continue
        sources = [s.split("#")[0] for s in re.findall(r"^\s+-\s+'?file:([^\s']+)", front, re.M)]
        parts = []
        for block in body.strip().split("\n"):
            block = block.strip()
            if not block:
                continue
            match = _MEM_PART.match(block)
            part = match.group(1) if match else (parts[-1][0] if parts else "body")
            for unit in units_of(_MEM_PART.sub("", block)):
                parts.append((part, unit))
        mems[mid] = {"id": mid, "title": clean_inline(field("title")), "summary": clean_inline(field("summary")),
                     "scope": field("scope"), "sources": sources, "superseded_by": field("superseded_by"),
                     "parts": parts, "store": label}
    return mems


def mem_groups(mems):
    """기억 덩이 : 같은 출처 파일(허브 파일 뺌) · 덮음 사슬을 잇는다."""
    uf = mp.UnionFind()
    source_count = collections.Counter(s for m in mems.values() for s in set(m["sources"]))
    for mid, mem in mems.items():
        uf.find(mid)
        for source in mem["sources"]:
            if source_count[source] <= MEM_SOURCE_CAP:
                uf.union(mid, f"file:{mem['store']}:{source}")
        if mem["superseded_by"] in mems:
            uf.union(mid, mem["superseded_by"])
    return {mid: "mem:" + uf.find(mid) for mid in mems}


def mine_mem(stores, dev_r2_path, old_ids, rng):
    col = ArchiveCollector("mem", "mem")
    mems = {}
    for label, path in stores:
        mems.update(read_mem_store(path, label))
    group_of = mem_groups(mems)
    dev_rows = make_sets.read_jsonl(dev_r2_path) if dev_r2_path else []
    dev_ids = {i for row in dev_rows for i in _MEM_ID.findall(json.dumps(row.get("src_ids", [])))}
    fresh = {mid for mid in mems if mid not in old_ids and mid not in dev_ids}
    for mid in mems:
        pool = "fresh" if mid in fresh else ("dev-r2" if mid in dev_ids else "old")
        col.group(group_of[mid], pool)
    _relabel_rows(col, dev_rows, group_of)
    _mem_inside(col, mems, fresh, group_of, rng)
    _mem_between(col, mems, fresh, group_of, rng)
    col.fresh_count = len(fresh)
    return col


def _relabel_rows(col, dev_rows, group_of):
    """dev-r2 344 를 v5 로 다시 붙일 후보로 넣는다 (dev 몫만 · 옛 라벨은 old_want)."""
    for row in dev_rows:
        ids = _MEM_ID.findall(json.dumps(row.get("src_ids", [])))
        groups = sorted({group_of.get(i, "mem:" + i) for i in ids}) or ["mem:r2-" + row["id"]]
        for gid in groups:
            col.group(gid, "dev-r2")
        col.add_raw({
            "evidence": row["evidence"], "claim": row["claim"], "want": "",
            "hint": OLD_WANT_HINT.get(row.get("want"), ""), "form": "r2-" + str(row.get("kind", "")),
            "domain": "mem", "group": groups[0], "groups": groups, "lang": row.get("lang", "ko"), "lic": "",
            "split": "dev", "src_ids": list(row.get("src_ids", [])), "origin": "r2-relabel", "source": "mem",
            "relabel": True, "old_want": row.get("want", ""), "old_id": row["id"],
        })


def _mem_units(mem):
    units = [("title", mem["title"]), ("summary", mem["summary"])] + list(mem["parts"])
    return [(p, u) for p, u in units if arch_text_ok(u)]


def _mem_inside(col, mems, fresh, group_of, rng):
    """한 기억 안 : 제목 ↔ 요약 · 본문(a), 다른 부분 문장끼리(d same-doc)."""
    for mid in sorted(fresh):
        mem = mems[mid]
        units = _mem_units(mem)
        if len(units) < 2:
            continue
        texts = [u for _, u in units]
        parts = [p for p, _ in units]
        index = BM25(texts)
        group = [group_of[mid]]
        heads = [i for i, p in enumerate(parts) if p in ("title", "summary")]
        body = set(i for i, p in enumerate(parts) if p not in ("title", "summary"))
        for i in heads:
            picks = _best(index, texts[i], body | (set(heads) - {i}))
            if picks:
                col.add(texts[i], texts[picks[0]], "a", "title-body", group, "", "any", [mid])
        for i in rng.sample(sorted(body), min(3, len(body))):
            other = set(j for j in body if parts[j] != parts[i])
            picks = _best(index, texts[i], other, top=3)
            if picks:
                col.add(texts[i], texts[rng.choice(picks)], "d", "same-doc", group, "", "any", [mid], via="parts")


def _mem_between(col, mems, fresh, group_of, rng):
    """기억 사이 (둘 다 새 기억) : 같은 출처 · 덮음(d link / b) · 같은 scope 근접(d near · c)."""
    by_group = collections.defaultdict(list)
    for mid in sorted(fresh):
        by_group[group_of[mid]].append(mid)
    for gid, members in sorted(by_group.items()):
        for i, left in enumerate(members):
            for right in members[i + 1:i + 3]:
                lu, ru = _mem_units(mems[left]), _mem_units(mems[right])
                if not lu or not ru:
                    continue
                newer, older = (left, right) if mems[right]["superseded_by"] == left else (right, left)
                linked = mems[older]["superseded_by"] == newer
                rtexts = [u for _, u in ru]
                pick = BM25(rtexts).rank(lu[0][1])[:1]
                claim = rtexts[pick[0]] if pick else rtexts[0]
                if linked:
                    new_title = _mem_units(mems[newer])[0][1]
                    old_title = _mem_units(mems[older])[0][1]
                    reversed_ = any(_REVERSAL.search(u) for _, u in _mem_units(mems[newer]))
                    col.add(old_title, new_title, "b" if reversed_ else "d", "supersede" if reversed_ else "link",
                            [gid], "", "any", [older, newer], via="superseded_by")
                else:
                    col.add(lu[0][1], claim, "d", "same-doc", [gid], "", "any", [left, right], via="source")
    by_scope = collections.defaultdict(list)
    for mid in sorted(fresh):
        units = _mem_units(mems[mid])
        if units:
            by_scope[mems[mid]["scope"]].append((mid, units[0][1]))
    for scope, items in sorted(by_scope.items()):
        if len(items) < 3:
            continue
        index = BM25([t for _, t in items])
        for k, (mid, text) in enumerate(items):
            ranked = [j for j in index.rank(text, exclude=[k]) if group_of[items[j][0]] != group_of[mid]]
            if ranked[:10]:
                j = rng.choice(ranked[:10])
                other = items[j][0]
                col.add(text, items[j][1], "c", "other-topic", [group_of[mid], group_of[other]], "", "any", [mid, other])
            if len(ranked) >= 10 and rng.random() < 0.5:
                j = rng.choice(ranked[9:60])
                other = items[j][0]
                col.add(text, items[j][1], "d", "near", [group_of[mid], group_of[other]], "", "any", [mid, other])


# ---------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description="아카이브 · mem 열쇠 묶음 · 후보 쌍 뽑기")
    parser.add_argument("source", choices=("git", "docs", "mem"))
    parser.add_argument("--out", required=True, help="후보 뿌리 (<out>/archive/<git|docs> · <out>/mem 이 생긴다)")
    parser.add_argument("--repo", action="append", default=[], help="git : 이름=저장소 경로 (여러 번)")
    parser.add_argument("--root", action="append", default=[], help="docs : 이름=저장소 뿌리 (여러 번)")
    parser.add_argument("--store", action="append", default=[], help="mem : 이름=Memory/store 경로 (여러 번)")
    parser.add_argument("--dev-r2", help="mem : 다시 붙일 dev-r2 파일")
    parser.add_argument("--old-root", help="mem : 옛 셋 뿌리 (make_sets.OLD_SETS 의 기억 id 를 뺀다)")
    parser.add_argument("--names", nargs="*", default=[], help="버릴 이름 목록 파일 (git 밖)")
    parser.add_argument("--game-names", nargs="*", default=[], help="<게임> 으로 바꿀 이름 목록 파일 (git 밖)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    if args.source == "git":
        col, out_root = mine_git(_labelled(args.repo), rng), os.path.join(args.out, "archive")
    elif args.source == "docs":
        col, out_root = mine_docs(_labelled(args.root), rng), os.path.join(args.out, "archive")
    else:
        old_ids = make_sets.load_old(args.old_root)[1] if args.old_root else set()
        col, out_root = mine_mem(_labelled(args.store), args.dev_r2, old_ids, rng), args.out
    counts = col.scrub(scrub.Scrubber(scrub.read_names(args.names), scrub.read_names(args.game_names)))
    summary = col.write(out_root)
    summary["scrub"] = dict(counts)
    if hasattr(col, "fresh_count"):
        summary["fresh_memories"] = col.fresh_count
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
