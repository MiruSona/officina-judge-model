"""mine_public — 공개 원천에서 열쇠 묶음과 후보 쌍을 뽑는다 (설계 d자료층재구축 3-1 · 3-2).

하위 명령마다 원천 하나 : proposals · vitaminc · klue-mrc · minicheck · ood · jira(먼저 jira_dump.py slim)
· vitaminc-heldout · vitaminc-train (v1.2 B 판 b 후보, `--out <build>/g2`)
· kobest-boolq · klue-answered (v1.2 K 판 한국어 b 후보, `~/.judgemodel/build/g3`, 쓰는 꼴이 달라 아래 함수 설명을 본다).
각각 (가) 열쇠 묶음 `group` 을 만들고 (나) d·a·b·c 후보 규칙으로 쌍을 뽑아 `form` 을 적고
(다) 공통 거르기(10~400자 · URL · 코드 덩이 · 빈 글)를 한다.

쓰는 것 : <out>/<원천>/candidates.jsonl · <out>/<원천>/groups.json
라벨(want)은 비워 둔다 — 후보다. `hint` 는 뽑은 규칙이 바라는 칸이지 라벨이 아니다.
받은 원자료는 자료로만 읽는다 (안의 코드를 돌리지 않는다 · zip 은 메모리에서 읽는다).

    python -I mine_public.py proposals --raw <raw>/proposals --out <build>/g1
"""
import argparse
import bisect
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bm25 import BM25  # noqa: E402

SEED = 20261011
MIN_LEN = 10
MAX_LEN = 400
PER_GROUP_CAP = 4          # 한 묶음 · 한 꼴에서 뽑는 후보 상한 (뽑기 단계)
D_FORMS = ("link", "same-doc", "near", "removed", "unanswerable")

_URL = re.compile(r"https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
_TRIVIAL = re.compile(r"^\s*(\+1|-1|lgtm|fix(ed)? typos?|typo|thanks?( you)?|done|n/?a|tbd|todo)\W*$", re.I)
_SENT_EN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(`*])")
_SENT_KO = re.compile(r"(?<=[.!?。])\s+")


# ---------------------------------------------------------------- 공통 거르기

def text_ok(text):
    """설계 3-1 공통 거르기 : 길이 10~400 · URL · 코드 덩이 · 빈 글."""
    if not text:
        return False
    text = text.strip()
    if not MIN_LEN <= len(text) <= MAX_LEN:
        return False
    if _URL.search(text) or _TRIVIAL.match(text):
        return False
    if text.endswith(":") or ("a" <= text[0] <= "z"):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.6 * len(text.replace(" ", "")):
        return False
    code_marks = sum(text.count(mark) for mark in ("{", "}", ";", "==", "=>", "->", "()", "`", "$", "::", "</"))
    return code_marks <= 2


def split_sentences(text, lang):
    pattern = _SENT_KO if lang == "ko" else _SENT_EN
    out = []
    for piece in pattern.split(text):
        piece = " ".join(piece.split())
        if piece:
            out.append(piece)
    return out


def join_within(sentences, limit=MAX_LEN):
    """차례를 지키며 limit 안에 드는 만큼 잇는다."""
    out = ""
    for sentence in sentences:
        candidate = (out + " " + sentence).strip()
        if len(candidate) > limit:
            break
        out = candidate
    return out


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_json(path):
    return json.loads(read_text(path))


def short_hash(text, size=12):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:size]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            keep, drop = sorted((ra, rb))
            self.parent[drop] = keep


class Collector:
    """한 원천의 후보 · 묶음을 모아 파일로 쓴다."""

    def __init__(self, source, lic_default=""):
        self.source = source
        self.lic_default = lic_default
        self.rows = []
        self.groups = {}
        self.per_cap = collections.Counter()
        self.dropped = collections.Counter()

    def group(self, gid, pool, title="", lic=""):
        entry = self.groups.setdefault(gid, {"pool": pool, "title": title, "lic": lic or self.lic_default})
        if entry["pool"] != pool:
            entry["pool"] = "mixed"
        return gid

    def add(self, evidence, claim, hint, form, groups, lang, split, src_ids, lic="", needs_rewrite=False, **extra):
        key = (groups[0], hint, form)
        if self.per_cap[key] >= PER_GROUP_CAP:
            return False
        evidence = " ".join(evidence.split())
        claim = " ".join(claim.split())
        if not text_ok(evidence):
            self.dropped["evidence"] += 1
            return False
        if not needs_rewrite and not text_ok(claim):
            self.dropped["claim"] += 1
            return False
        if not needs_rewrite and evidence == claim:
            self.dropped["same"] += 1
            return False
        self.per_cap[key] += 1
        row = {
            "evidence": evidence, "claim": claim, "want": "", "hint": hint, "form": form,
            "domain": self.source, "group": groups[0], "groups": sorted(set(groups)),
            "lang": lang, "lic": lic or self.groups.get(groups[0], {}).get("lic") or self.lic_default,
            "split": split, "src_ids": list(src_ids), "origin": "g1-II",
        }
        if needs_rewrite:
            row["needs_rewrite"] = True
        row.update(extra)
        self.rows.append(row)
        return True

    def write(self, out_root):
        folder = os.path.join(out_root, self.source)
        os.makedirs(folder, exist_ok=True)
        self.rows.sort(key=lambda r: (r["group"], r["hint"], r["form"], r["evidence"], r["claim"]))
        sizes = collections.Counter()
        for index, row in enumerate(self.rows, 1):
            row["id"] = f"{self.source}-{index:06d}"
            for gid in row["groups"]:
                sizes[gid] += 1
        with open(os.path.join(folder, "candidates.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for row in self.rows:
                ordered = {"id": row.pop("id")}
                ordered.update(row)
                f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
        for gid, entry in self.groups.items():
            entry["n"] = sizes.get(gid, 0)
        summary = {
            "source": self.source,
            "candidates": len(self.rows),
            "groups": len(self.groups),
            "groups_used": sum(1 for g in self.groups.values() if g["n"]),
            "by_hint_form": dict(sorted(collections.Counter(f"{r['hint']}:{r['form']}" for r in self.rows).items())),
            "by_split": dict(collections.Counter(r["split"] for r in self.rows)),
            "dropped": dict(self.dropped),
        }
        with open(os.path.join(folder, "groups.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "groups": self.groups}, f, ensure_ascii=False, indent=1)
        return summary


# ---------------------------------------------------------------- 설계 제안서

_SECTION_ROLES = (
    ("skip", ("table of contents", "release signoff", "production readiness", "implementation history",
              "infrastructure needed", "copyright", "references", "footnotes", "acknowledg", "reference implementation",
              "open issues", "change history", "revision history", "graduation criteria", "test plan")),
    ("rejected", ("rejected", "alternative", "considered", "drawback")),
    ("compat", ("compatib", "abi", "migration", "impact on existing", "effect on api", "upgrade", "deprecat")),
    ("motivation", ("motivation", "rationale", "background", "why ", "problem", "goals")),
    ("intro", ("abstract", "summary", "introduction", "overview")),
    ("spec", ("specification", "proposed solution", "proposal", "detailed design", "design", "implementation",
              "semantics", "syntax", "api", "behavio")),
)


def section_role(heading):
    low = heading.lower()
    for role, words in _SECTION_ROLES:
        if any(word in low for word in words):
            return role
    return "other"


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_RST_ROLE = re.compile(r":[a-z:]+:`([^`<]*?)(?:\s*<[^>]*>)?`")
_INLINE_CODE = re.compile(r"``?([^`]+)``?")
_HTML = re.compile(r"<[^>]+>")


def _clean_inline(line):
    line = _MD_LINK.sub(r"\1", line)
    line = _RST_ROLE.sub(r"\1", line)
    line = _HTML.sub("", line)
    line = line.replace("**", "").replace("__", "")
    return line


def parse_sections(text, flavor):
    """제안서 본문 → [(절 머리, 역할, [문장…])]. 코드 덩이 · 표 · 지시문은 버린다."""
    lines = text.replace("\r\n", "\n").split("\n")
    sections = []
    heading, role = "(head)", "skip"
    paragraph = []
    paragraphs = []
    in_fence = False
    rst_literal = False

    def flush_paragraph():
        if paragraph:
            paragraphs.append(" ".join(paragraph))
            paragraph.clear()

    def flush_section():
        flush_paragraph()
        sentences = []
        for para in paragraphs:
            sentences.extend(split_sentences(_clean_inline(para), "en"))
        if sentences:
            sections.append((heading, role, sentences))
        paragraphs.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            flush_paragraph()
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        if flavor == "rst" and rst_literal:
            if stripped and not line[:1].isspace():
                rst_literal = False
            else:
                i += 1
                continue
        is_rst_head = (flavor == "rst" and i + 1 < len(lines) and stripped
                       and re.fullmatch(r"([=\-~^\"'`#*+])\1{2,}", lines[i + 1].strip() or "x")
                       and len(lines[i + 1].strip()) >= len(stripped) - 1)
        if flavor == "md" and re.match(r"#{1,6}\s", stripped):
            level = len(stripped) - len(stripped.lstrip("#"))
            name = stripped.lstrip("#").strip()
            if level == 1:
                flush_section()
                heading, role = name, "skip"
            elif level == 2 or role in ("skip", "other"):
                flush_section()
                heading, role = name, section_role(name)
            i += 1
            continue
        if is_rst_head:
            flush_section()
            heading, role = stripped, section_role(stripped)
            i += 2
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if flavor == "rst" and stripped.startswith(".."):
            flush_paragraph()
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                i += 1
            continue
        if stripped.startswith("|") or stripped.startswith("<") or (flavor == "md" and line.startswith("    ")):
            flush_paragraph()
            i += 1
            continue
        if flavor == "rst" and line[:1].isspace() and not paragraph and not re.match(r"\s*[-*+]\s|\s*\d+\.\s", line):
            i += 1
            continue
        body = re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped)
        if flavor == "rst" and body.endswith("::"):
            body = body[:-1]
            rst_literal = True
        paragraph.append(body)
        if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            flush_paragraph()
        i += 1
    flush_section()
    return sections


def _pep_headers(text):
    headers = {}
    key = None
    for line in text.split("\n"):
        if not line.strip():
            break
        match = re.match(r"^([A-Za-z-]+):\s*(.*)$", line)
        if match:
            key = match.group(1).lower()
            headers[key] = match.group(2).strip()
        elif key:
            headers[key] += " " + line.strip()
    return headers


def read_proposals(raw):
    """네 저장소 → 제안서 목록 [{id, repo, title, lic, sections, links:[다른 id]}]."""
    docs = []
    for path in sorted(glob.glob(os.path.join(raw, "python__peps", "peps", "pep-*.rst"))):
        text = read_text(path)
        if "CC0" not in text and "public domain" not in text.lower():
            continue
        headers = _pep_headers(text)
        number = int(re.search(r"pep-(\d+)", path).group(1))
        links = []
        for key in ("replaces", "superseded-by"):
            for num in re.findall(r"\d+", headers.get(key, "")):
                links.append(f"pep-{int(num):04d}")
        body = text.split("\n\n", 1)[1] if "\n\n" in text else text
        docs.append({"id": f"pep-{number:04d}", "repo": "pep", "title": headers.get("title", ""),
                     "lic": "CC0-1.0", "sections": parse_sections(body, "rst"), "links": links})
    for path in sorted(glob.glob(os.path.join(raw, "swiftlang__swift-evolution", "proposals", "*.md"))):
        name = os.path.basename(path)
        match = re.match(r"(\d{4})-", name)
        if not match:
            continue
        text = read_text(path)
        links = []
        for line in text.split("\n")[:30]:
            if re.match(r"^\*\s*(Previous Proposals?|Superseded by)\s*:", line, re.I):
                links.extend(f"se-{num}" for num in re.findall(r"SE-(\d{4})", line))
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        docs.append({"id": f"se-{match.group(1)}", "repo": "swift", "title": title, "lic": "Apache-2.0",
                     "sections": parse_sections(text, "md"), "links": links})
    for path in sorted(glob.glob(os.path.join(raw, "golang__proposal", "design", "*.md"))):
        name = os.path.basename(path)
        match = re.match(r"(\d+)-", name)
        if not match:
            continue
        text = read_text(path)
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        docs.append({"id": f"go-{match.group(1)}", "repo": "go", "title": title, "lic": "BSD-3-Clause",
                     "sections": parse_sections(text, "md"), "links": []})
    for yaml_path in sorted(glob.glob(os.path.join(raw, "kubernetes__enhancements", "keps", "*", "*", "kep.yaml"))):
        folder = os.path.dirname(yaml_path)
        readme = os.path.join(folder, "README.md")
        if not os.path.exists(readme):
            continue
        meta = read_text(yaml_path)
        number = re.search(r"^kep-number:\s*\"?(\d+)", meta, re.M)
        if not number:
            continue
        links = []
        for key in ("replaces", "superseded-by"):
            block = re.search(rf"^{key}:\s*\n((?:\s+-.*\n?)+)", meta, re.M)
            if block:
                links.extend(f"kep-{int(n)}" for n in re.findall(r"/(\d+)-", block.group(1)))
        text = read_text(readme)
        title = re.search(r"^title:\s*(.*)$", meta, re.M)
        docs.append({"id": f"kep-{int(number.group(1))}", "repo": "kep",
                     "title": title.group(1).strip().strip('"') if title else "",
                     "lic": "Apache-2.0", "sections": parse_sections(text, "md"), "links": links})
    return docs


def mine_proposals(raw, rng):
    col = Collector("proposals")
    docs = read_proposals(raw)
    by_id = {doc["id"]: doc for doc in docs}
    uf = UnionFind()
    for doc in docs:
        uf.find(doc["id"])
        for other in doc["links"]:
            if other in by_id and other != doc["id"]:
                uf.union(doc["id"], other)
    for doc in docs:
        col.group("prop:" + uf.find(doc["id"]), "any", title=doc["title"], lic=doc["lic"])
        doc["group"] = "prop:" + uf.find(doc["id"])
        doc["flat"] = [(role, s) for _, role, sentences in doc["sections"] if role != "skip"
                       for s in sentences if text_ok(s)]

    def first_sentences(doc, count):
        for role in ("intro", "motivation", "spec", "other"):
            picked = [s for r, s in doc["flat"] if r == role]
            if picked:
                return picked[:count]
        return []

    # link : 대체 사슬의 요약끼리 (d 후보)
    for doc in docs:
        for other_id in doc["links"]:
            other = by_id.get(other_id)
            if not other or other_id == doc["id"]:
                continue
            evidence = join_within(first_sentences(other, 2))
            claim = (first_sentences(doc, 1) or [""])[0]
            col.add(evidence, claim, "d", "link", [doc["group"]], "en", "any", [doc["id"], other_id], lic=doc["lic"])

    pairs_same_doc = (("motivation", "spec"), ("spec", "compat"), ("intro", "compat"), ("motivation", "compat"),
                      ("intro", "spec"))
    for doc in docs:
        flat = doc["flat"]
        if len(flat) < 4:
            continue
        texts = [s for _, s in flat]
        index = BM25(texts)
        roles = [r for r, _ in flat]
        lic = doc["lic"]
        group = [doc["group"]]
        # same-doc : 다른 절에서 가장 가까운 문장 (d 후보)
        for left, right in pairs_same_doc:
            lefts = [i for i, r in enumerate(roles) if r == left]
            rights = set(i for i, r in enumerate(roles) if r == right)
            if not lefts or not rights:
                continue
            for i in rng.sample(lefts, min(2, len(lefts))):
                ranked = [j for j in index.rank(texts[i]) if j in rights][:5]
                if ranked:
                    j = rng.choice(ranked)
                    col.add(texts[i], texts[j], "d", "same-doc", group, "en", "any", [doc["id"]], lic=lic)
        # near : 묶음 안 BM25 10~60 위 (d 후보)
        for i in rng.sample(range(len(texts)), min(3, len(texts))):
            band = [j for j in index.near(texts[i], exclude=range(i - 2, i + 3))]
            if band:
                j = rng.choice(band)
                col.add(texts[i], texts[j], "d", "near", group, "en", "any", [doc["id"]], lic=lic)
        # a : 같은 절, 떨어진 자리의 가장 닮은 문장 (바꿔 말함 후보)
        for i in rng.sample(range(len(texts)), min(4, len(texts))):
            same = [j for j in index.rank(texts[i], exclude=range(i - 1, i + 2)) if roles[j] == roles[i]][:1]
            if same:
                col.add(texts[i], texts[same[0]], "a", "same-sec", group, "en", "any", [doc["id"]], lic=lic)
        # b : Rejected/Alternatives 문장 ↔ 명세에서 가장 가까운 문장
        rejected = [i for i, r in enumerate(roles) if r == "rejected"]
        specs = set(i for i, r in enumerate(roles) if r in ("spec", "intro"))
        for i in rng.sample(rejected, min(3, len(rejected))):
            ranked = [j for j in index.rank(texts[i]) if j in specs][:1]
            if ranked:
                col.add(texts[ranked[0]], texts[i], "b", "rejected", group, "en", "any", [doc["id"]], lic=lic)

    # c : 같은 저장소 다른 제안서의 근접 문장 (근접 c)
    by_repo = collections.defaultdict(list)
    for doc in docs:
        if doc["flat"]:
            by_repo[doc["repo"]].append(doc)
    for repo, repo_docs in by_repo.items():
        reps = []
        owners = []
        for doc in repo_docs:
            for _, sentence in doc["flat"][:6]:
                reps.append(sentence)
                owners.append(doc)
        index = BM25(reps)
        for doc in repo_docs:
            texts = [s for _, s in doc["flat"]]
            for sentence in rng.sample(texts, min(2, len(texts))):
                ranked = [j for j in index.rank(sentence) if owners[j]["group"] != doc["group"]][:10]
                if ranked:
                    j = rng.choice(ranked)
                    other = owners[j]
                    col.add(reps[j], sentence, "c", "other-doc", [doc["group"], other["group"]], "en", "any",
                            [doc["id"], other["id"]], lic=doc["lic"])
    return col


# ---------------------------------------------------------------- VitaminC

def mine_vitaminc(raw, rng, read_rows=None):
    if read_rows is None:
        from parquet_lite import read_rows
    col = Collector("vitaminc", "CC-BY-SA-3.0")
    by_page = collections.defaultdict(list)
    for split_file, split in (("train.parquet", "train"), ("validation.parquet", "heldout"), ("test.parquet", "heldout")):
        for row in read_rows(os.path.join(raw, split_file)):
            if row.get("revision_type") != "real":
                continue
            row["split"] = split
            by_page[row["page"]].append(row)
    hints = {"SUPPORTS": ("a", "labeled"), "REFUTES": ("b", "labeled"), "NOT ENOUGH INFO": ("d", "same-doc")}
    for page in sorted(by_page):
        rows = by_page[page]
        gid = "vitc:" + page
        for row in rows:
            col.group(gid, row["split"], title=page)
        rng.shuffle(rows)
        for row in rows:
            hint, form = hints[row["label"]]
            col.add(row["evidence"], row["claim"], hint, form, [gid], "en", row["split"], [row["unique_id"]])
        evidences = sorted(set(r["evidence"] for r in rows))
        if len(evidences) < 3:
            continue
        index = BM25(evidences)
        for row in rows[:6]:
            ranked = index.rank(row["claim"])
            own = evidences.index(row["evidence"])
            # c : 같은 문서 다른 대상 — 가장 덜 닮은 증거 (Opus 가 가린다)
            scored = set(ranked)
            far = [j for j in range(len(evidences)) if j not in scored and j != own] + [j for j in ranked[-2:] if j != own]
            if far:
                j = rng.choice(far)
                col.add(evidences[j], row["claim"], "c", "same-page", [gid], "en", row["split"], [row["unique_id"]])
            band = [j for j in index.near(row["claim"], exclude=[own])]
            if band:
                j = rng.choice(band)
                col.add(evidences[j], row["claim"], "d", "near", [gid], "en", row["split"], [row["unique_id"]])
    return col


# ---------------------------------------------------------------- VitaminC b (v1.2 B 판)

def mine_vitaminc_g2(raw, rng, split, read_rows=None):
    """VitaminC 실제 판 고침의 b 후보. 설계 v1.2-B판 3-1 · 5-1 B.

    heldout(평가용) : REFUTES 줄 전부 (form labeled). train(학습 층용) : 같은 판 고침(case_id) · 같은 주장에
    SUPPORTS 근거와 REFUTES 근거가 다 있는 짝만 a · b 두 줄로 (form contrast, 같은 `pair`). 묶음은 g1 과 같은 「vitc:문서」.
    """
    if read_rows is None:
        from parquet_lite import read_rows
    files = {"train": ("train.parquet",), "heldout": ("validation.parquet", "test.parquet")}[split]
    col = Collector("vitaminc-" + split, "CC-BY-SA-3.0")
    by_page = collections.defaultdict(list)
    for name in files:
        for row in read_rows(os.path.join(raw, name)):
            if row.get("revision_type") == "real":
                by_page[row["page"]].append(row)
    for page in sorted(by_page):
        gid = col.group("vitc:" + page, split, title=page)
        rows = sorted(by_page[page], key=lambda r: r["unique_id"])
        if split == "heldout":
            for row in rows:
                if row["label"] == "REFUTES":
                    col.add(row["evidence"], row["claim"], "b", "labeled", [gid], "en", split, [row["unique_id"]],
                            domain="vitaminc")
            continue
        by_claim = collections.defaultdict(dict)
        for row in rows:
            by_claim[(row["case_id"], row["claim"])].setdefault(row["label"], row)
        keys = sorted(by_claim)
        rng.shuffle(keys)
        for case_id, claim in keys:
            both = by_claim[(case_id, claim)]
            sup, ref = both.get("SUPPORTS"), both.get("REFUTES")
            if not sup or not ref or sup["evidence"] == ref["evidence"]:
                continue
            pair = case_id + ":" + short_hash(claim, 8)
            kept = len(col.rows)
            ok = col.add(ref["evidence"], claim, "b", "contrast", [gid], "en", split, [ref["unique_id"]],
                         domain="vitaminc", pair=pair)
            if ok and not col.add(sup["evidence"], claim, "a", "contrast", [gid], "en", split, [sup["unique_id"]],
                                  domain="vitaminc", pair=pair):
                del col.rows[kept:]           # 짝 한쪽만 남으면 버린다 (설계 5-1 「한쪽만 남은 짝은 버림」)
                col.dropped["half_pair"] += 1
    return col


# ---------------------------------------------------------------- KLUE-MRC

_JOSA = ("에서는", "에서", "으로", "에게", "까지", "부터", "이라", "라는", "은", "는", "이", "가", "을", "를", "의",
         "에", "로", "도", "와", "과", "만")


def question_entities(question, context):
    """질문 낱말 중 지문에도 있는 것 (조사를 대충 떼고). 개체 인식기 대신 쓴다."""
    out = []
    for word in re.findall(r"[0-9A-Za-z가-힣]+", question):
        for josa in _JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[:-len(josa)]
                break
        if len(word) >= 2 and word in context and word not in out:
            out.append(word)
    return out


def _sentence_at(sentences, context, offset):
    pos = 0
    for index, sentence in enumerate(sentences):
        found = context.find(sentence[:20], pos)
        start = found if found >= 0 else pos
        if start <= offset < start + len(sentence) + 2:
            return index
        pos = start + len(sentence)
    return None


def mine_klue(raw, rng):
    col = Collector("klue-mrc", "CC-BY-SA-4.0")
    articles = []
    for name, split in (("klue-mrc-v1.1_train.json", "train"), ("klue-mrc-v1.1_dev.json", "heldout")):
        data = read_json(os.path.join(raw, name))["data"]
        for article in data:
            article["split"] = split
            articles.append(article)
    by_category = collections.defaultdict(list)
    for article in articles:
        gid = "klue:" + short_hash(article["title"])
        col.group(gid, article["split"], title=article["title"])
        article["gid"] = gid
        for para in article["paragraphs"]:
            sentences = [s for s in split_sentences(para["context"], "ko")]
            para["sentences"] = sentences
            reps = [s for s in sentences if text_ok(s)][:3]
            for rep in reps:
                by_category[(article["split"], article.get("news_category") or article.get("source"))].append(
                    (rep, article))
    cat_index = {key: (BM25([rep for rep, _ in items]), items) for key, items in by_category.items()}

    for article in articles:
        gid = article["gid"]
        split = article["split"]
        for para in article["paragraphs"]:
            sentences = para["sentences"]
            if len(sentences) < 3:
                continue
            index = BM25(sentences)
            for qa in para["qas"]:
                question = qa["question"]
                entities = question_entities(question, para["context"])
                base = {"question": question, "entities": entities}
                if qa.get("is_impossible"):
                    top = index.rank(question)[:2]
                    evidence = join_within([sentences[i] for i in sorted(top)])
                    col.add(evidence, "", "d", "unanswerable", [gid], "ko", split, [qa["guid"]],
                            needs_rewrite=True, **base)
                    continue
                answer = qa["answers"][0]
                at = _sentence_at(sentences, para["context"], answer["answer_start"])
                if at is None:
                    continue
                base["answer"] = answer["text"]
                role = int(short_hash(qa["guid"], 4), 16) % 3
                if role == 0:
                    col.add(sentences[at], "", "a", "qa-answer", [gid], "ko", split, [qa["guid"]],
                            needs_rewrite=True, **base)
                elif role == 1:
                    rest = [i for i in index.rank(question) if i != at][:2]
                    evidence = join_within([sentences[i] for i in sorted(rest)])
                    col.add(evidence, "", "d", "removed", [gid], "ko", split, [qa["guid"]],
                            needs_rewrite=True, removed=sentences[at], **base)
                else:
                    key = (split, article.get("news_category") or article.get("source"))
                    cat, items = cat_index[key]
                    ranked = [j for j in cat.rank(question)[:50] if items[j][1]["gid"] != gid
                              and any(word in items[j][0] for word in entities)][:5]
                    if ranked:
                        rep, other = items[rng.choice(ranked)]
                        col.add(rep, "", "c", "other-doc", [gid, other["gid"]], "ko", split, [qa["guid"]],
                                needs_rewrite=True, **base)
    return col


# ---------------------------------------------------------------- MiniCheck

def mine_minicheck(raw, rng, read_rows=None):
    if read_rows is None:
        from parquet_lite import read_rows
    col = Collector("minicheck", "MIT")
    rows = []
    for name in ("c2d", "d2c"):
        for index, row in enumerate(read_rows(os.path.join(raw, f"{name}.parquet"))):
            row["rid"] = f"{name}-{index}"
            rows.append(row)
    uf = UnionFind()
    for row in rows:
        uf.union("c:" + short_hash(row["claim"]), "d:" + short_hash(row["doc"]))
    for row in rows:
        row["gid"] = "mc:" + uf.find("d:" + short_hash(row["doc"]))
        col.group(row["gid"], "any")
    docs = {}
    for row in rows:
        docs.setdefault(row["doc"], split_sentences(row["doc"], "en"))
    reps = []
    owners = []
    for row in rows:
        if row["doc"] not in docs:
            continue
        first = [s for s in docs[row["doc"]] if text_ok(s)][:2]
        for sentence in first:
            reps.append(sentence)
            owners.append(row["gid"])
    cross = BM25(reps)
    rng.shuffle(rows)
    for row in rows:
        if str(row["label"]) != "1":
            continue
        sentences = docs[row["doc"]]
        index = BM25(sentences)
        ranked = index.rank(row["claim"])
        if not ranked:
            continue
        role = int(short_hash(row["rid"], 4), 16) % 5
        if role in (0, 1):
            top = sorted(ranked[:3])
            col.add(join_within([sentences[i] for i in top]), row["claim"], "a", "doc-top", [row["gid"]], "en",
                    "any", [row["rid"]])
        elif role in (2, 3):
            kept = sorted(ranked[2:5])
            col.add(join_within([sentences[i] for i in kept]), row["claim"], "d", "removed", [row["gid"]], "en",
                    "any", [row["rid"]], removed=join_within([sentences[i] for i in sorted(ranked[:2])], 800))
        else:
            near = [j for j in cross.rank(row["claim"]) if owners[j] != row["gid"]][:10]
            if near:
                j = rng.choice(near)
                col.add(reps[j], row["claim"], "c", "other-doc", [row["gid"], owners[j]], "en", "any", [row["rid"]])
    return col


# ---------------------------------------------------------------- OOD (WiCE · SufficientFacts · ContractNLI)

def mine_ood(raw, rng, per_source=240):
    col = Collector("ood")
    # WiCE (claim 수준, dev · test)
    wice_rows = []
    for split in ("dev", "test"):
        path = os.path.join(raw, "wice", "wice", "data", "entailment_retrieval", "claim", f"{split}.jsonl")
        for line_no, line in enumerate(read_text(path).splitlines()):
            wice_rows.append((split, line_no, json.loads(line)))
    rng.shuffle(wice_rows)
    wice_hint = {"supported": "a", "not_supported": "d"}
    for split, line_no, row in wice_rows:
        hint = wice_hint.get(row["label"])
        if not hint:
            continue
        sentences = row["evidence"]
        if hint == "a":
            picked = sorted(set(i for chunk in row["supporting_sentences"][:1] for i in chunk))
        else:
            picked = sorted(BM25(sentences).rank(row["claim"])[:2])
        gid = col.group("wice:" + short_hash(row["claim"]), "heldout", lic="ODC-BY+CC-BY-SA")
        col.add(join_within([sentences[i] for i in picked if i < len(sentences)]), row["claim"], hint,
                "wice-" + row["label"], [gid], "en", split, [f"wice-{split}-{line_no}"], source_label=row["label"])
    # SufficientFacts (뺀 뒤 라벨)
    sf_hint = {"SUPPORTS": "a", "REFUTES": "b", "NOT ENOUGH": "d"}
    for name in ("fever", "hover", "vitaminc"):
        path = os.path.join(raw, "sufficientfacts", f"{name}_sufficient_facts.jsonl")
        for line_no, line in enumerate(read_text(path).splitlines()):
            row = json.loads(line)
            hint = sf_hint.get(row.get("label_after"))
            if not hint:
                continue
            evidence = join_within([sent for _, sent in row["evidence"]])
            gid = col.group("sf:" + short_hash(row["claim"]), "heldout", lic="MIT")
            col.add(evidence, row["claim"], hint, "sf-" + name, [gid], "en", "heldout", [f"sf-{name}-{line_no}"],
                    source_label=row["label_after"])
    # ContractNLI (dev · test, zip 을 메모리에서 읽음)
    cnli_hint = {"Entailment": "a", "Contradiction": "b", "NotMentioned": "d"}
    with zipfile.ZipFile(os.path.join(raw, "contractnli", "contract-nli.zip")) as archive:
        for split in ("dev", "test"):
            data = json.loads(archive.read(f"contract-nli/{split}.json"))
            labels = data["labels"]
            for doc in data["documents"]:
                text = doc["text"]
                spans = [" ".join(text[a:b].split()) for a, b in doc["spans"]]
                gid = col.group(f"cnli:{split}-{doc['id']}", "heldout", lic="CC-BY-4.0")
                index = BM25(spans)
                for key, ann in doc["annotation_sets"][0]["annotations"].items():
                    hint = cnli_hint[ann["choice"]]
                    claim = labels[key]["hypothesis"]
                    picked = ann["spans"] if ann["spans"] else index.rank(claim)[:1]
                    col.add(join_within([spans[i] for i in picked]), claim, hint, "cnli-" + ann["choice"], [gid],
                            "en", split, [f"cnli-{split}-{doc['id']}-{key}"], source_label=ann["choice"])
    # 원천마다 per_source 로 줄이되 hint 를 고르게
    kept = []
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in col.rows:
        by_src[row["form"].split("-")[0]][row["hint"]].append(row)
    for src in sorted(by_src):
        buckets = by_src[src]
        share = per_source // max(1, len(buckets))
        for hint in sorted(buckets):
            rng.shuffle(buckets[hint])
            kept.extend(buckets[hint][:share])
    col.rows = kept
    used = set(g for r in kept for g in r["groups"])
    col.groups = {g: v for g, v in col.groups.items() if g in used}
    return col


# ---------------------------------------------------------------- Jira (The Public Jira Dataset 익명판)
#
# 원자료는 mongodump 아카이브라 먼저 `jira_dump.py slim` 으로 얇은 jsonl 을 만들고 그 폴더를 --raw 로 준다.
# 링크 이름은 원자료의 jira_issue_linktype_mapping.json 을 따른다 (Relate 꼴 다섯 · Duplicate).

JIRA_RELATE = {"reference", "relate", "related", "relates", "relationship"}
JIRA_DUPLICATE = {"duplicate"}
JIRA_WONTFIX = ("won't fix", "not a bug", "not a problem", "invalid", "won't do")
JIRA_BODY = (80, 1200)          # 설명(본문) 길이 — 로그 · 빈 설명을 거른다
JIRA_PEER_TRIES = 20
_JIRA_MARKUP = re.compile(r"\{(code|noformat|quote|panel)|\[~|\|\||^h[1-6]\.", re.I | re.M)


def _jira_issue(raw_issue):
    """Jira REST 꼴(얇게 줄인 것도 같은 꼴)에서 필요한 칸만 뽑는다."""
    fields = raw_issue.get("fields", raw_issue)
    links = []
    for link in fields.get("issuelinks") or []:
        kind = (link.get("type") or {}).get("name", "")
        other = link.get("outwardIssue") or link.get("inwardIssue") or {}
        if other.get("key"):
            links.append((kind, other["key"]))
    comments = [c.get("body", "") for c in ((fields.get("comment") or {}).get("comments") or [])]
    return {
        "key": raw_issue.get("key", ""),
        "repo": raw_issue.get("repo", ""),
        "project": (fields.get("project") or {}).get("key", raw_issue.get("key", "").split("-")[0]),
        "summary": fields.get("summary") or "",
        "description": fields.get("description") or "",
        "components": sorted(c.get("name", "") for c in fields.get("components") or []),
        "resolution": (fields.get("resolution") or {}).get("name", ""),
        "links": links,
        "comments": comments,
    }


def _jira_items(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    loaded = read_json(path)
    yield from (loaded if isinstance(loaded, list) else loaded.get("issues", []))


def read_jira(raw):
    issues = []
    for path in sorted(glob.glob(os.path.join(raw, "**", "*.json*"), recursive=True)):
        issues.extend(_jira_issue(item) for item in _jira_items(path) if isinstance(item, dict) and item.get("key"))
    return issues


def _jira_body_ok(issue):
    body = " ".join(issue["description"].split())
    return JIRA_BODY[0] <= len(body) <= JIRA_BODY[1] and not _JIRA_MARKUP.search(issue["description"])


def _jira_kind(kind):
    low = re.sub(r"^\d+\s*-\s*", "", kind.strip().lower())     # RedHat 꼴 「1 - Relate」
    if low in JIRA_RELATE:
        return "relate"
    if low in JIRA_DUPLICATE:
        return "duplicate"
    return ""


def mine_jira(raw, rng):
    col = Collector("jira", "CC-BY-4.0")
    issues = read_jira(raw)
    by_key = {issue["key"]: issue for issue in issues}
    uf = UnionFind()
    for issue in issues:
        uf.find(issue["key"])
        for kind, other in issue["links"]:
            if other in by_key and _jira_kind(kind):
                uf.union(issue["key"], other)
    for issue in issues:
        issue["gid"] = "jira:" + uf.find(issue["key"])
        issue["ok"] = _jira_body_ok(issue)
        col.group(issue["gid"], "any", title=issue["project"])

    def head(issue, count):
        return join_within([issue["summary"].rstrip(".") + "."] + split_sentences(issue["description"], "en")[:count])

    by_project = collections.defaultdict(list)
    for issue in issues:
        if issue["ok"] and issue["components"]:
            by_project[(issue["repo"], issue["project"])].append(issue)
    done = set()
    for issue in issues:
        if not issue["ok"]:
            continue
        for kind, other_key in issue["links"]:
            other = by_key.get(other_key)
            pair = frozenset((issue["key"], other_key))
            kind = _jira_kind(kind)
            if not kind or not other or not other["ok"] or pair in done or len(pair) < 2:
                continue
            done.add(pair)
            hint, form = ("d", "link") if kind == "relate" else ("a", "duplicate")
            col.add(head(issue, 2), head(other, 1), hint, form, [issue["gid"]], "en", "any", [issue["key"], other_key])
        if issue["resolution"].lower() in JIRA_WONTFIX:
            for comment in issue["comments"]:
                if "[~" not in comment and re.search(r"won'?t fix|not a bug|works as (intended|designed)|by design",
                                                     comment, re.I):
                    col.add(join_within(split_sentences(comment, "en")[:2]), head(issue, 1), "b", "wontfix",
                            [issue["gid"]], "en", "any", [issue["key"]])
                    break
        peers = by_project.get((issue["repo"], issue["project"])) if issue["components"] else None
        for _ in range(JIRA_PEER_TRIES if peers else 0):
            other = rng.choice(peers)
            if other["gid"] != issue["gid"] and not set(other["components"]) & set(issue["components"]):
                col.add(head(other, 2), head(issue, 1), "c", "other-component", [issue["gid"], other["gid"]], "en",
                        "any", [issue["key"], other["key"]])
                break
    return col


# ---------------------------------------------------------------- v1.2 K 판 (g3) 공통 — 문단 겹침

def _norm(text):
    return " ".join(unicodedata.normalize("NFKC", text or "").split()).lower()


def char3(text):
    """정규화한 글(빈칸 뺌)의 글자 3묶음 집합. 문단끼리 닮음을 볼 때 쓴다."""
    flat = "".join(_norm(text).split())
    return {flat[i:i + 3] for i in range(len(flat) - 2)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_against(base_texts, other_texts, threshold=0.8):
    """other 글마다 base 에 같은 글(정규화 일치)이나 글자 3묶음 닮음 ≥ threshold 인 글이 있나.

    돌려줌 : {other 차례: "exact" | "near"}. 닮음은 크기가 threshold 배 안쪽인 것끼리만 잰다(그 밖은 닿을 수 없다).
    """
    base_norm = {_norm(t) for t in base_texts}
    grams = sorted((len(g), i, g) for i, g in enumerate(char3(t) for t in sorted(set(base_texts))))
    sizes = [n for n, _, _ in grams]
    found = {}
    for index, text in enumerate(other_texts):
        if _norm(text) in base_norm:
            found[index] = "exact"
            continue
        mine = char3(text)
        if not mine:
            continue
        low = bisect.bisect_left(sizes, len(mine) * threshold)
        high = bisect.bisect_right(sizes, len(mine) / threshold)
        if any(jaccard(mine, g) >= threshold for _, _, g in grams[low:high]):
            found[index] = "near"
    return found


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stem_words(text):
    """낱말에서 조사를 대충 뗀 줄기 (2자 이상). question_entities 와 같은 규칙이되 지문을 안 본다."""
    out = []
    for word in re.findall(r"[0-9A-Za-z가-힣]+", text):
        for josa in _JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[:-len(josa)]
                break
        if len(word) >= 2 and word not in out:
            out.append(word)
    return out


# ---------------------------------------------------------------- KoBEST BoolQ (v1.2 K 판 · 설계 3-2)

BOOLQ_FILES = (("train.jsonl", "train"), ("validation.jsonl", "dev"), ("test.jsonl", "test"))
BOOLQ_HINT = {0: "b", 1: "a"}          # 라벨 0(아니오) → b 후보 · 1(예) → a 짝
BOOLQ_KEEP = 3                         # 문단이 길면 질문과 낱말이 많이 겹치는 문장 이만큼까지
BOOLQ_NEAR = 0.8                       # 문단 겹침 : 글자 3묶음 닮음 하한
# 물음꼴 : 물음표로 끝나거나 「~나요 · ~인가 · ~는가 · ~을까 …」로 끝난다. 「~니 · ~냐」는 「어머니」 같은 말과 갈려 물음표 있을 때만.
_ASKING = re.compile(r"(?:[?？]|(?:나요|까요|니까|는가|은가|인가|던가|을까|일까|느냐)\s*\.?)\s*$")


def is_asking(text):
    return bool(_ASKING.search((text or "").strip()))


def shorten_paragraph(paragraph, question, keep=BOOLQ_KEEP, limit=MAX_LEN):
    """limit 를 넘는 문단은 질문 낱말이 많이 든 문장을 keep 개까지(limit 안에서) 골라 원래 차례로 잇는다."""
    paragraph = " ".join(paragraph.split())
    if len(paragraph) <= limit:
        return paragraph
    sentences = split_sentences(paragraph, "ko")
    words = stem_words(question)
    ranked = sorted(range(len(sentences)), key=lambda i: (-sum(w in sentences[i] for w in words), i))
    picked = []
    for i in ranked:
        if len(" ".join(sentences[j] for j in sorted(picked + [i]))) <= limit:
            picked.append(i)
        if len(picked) == keep:
            break
    return " ".join(sentences[i] for i in sorted(picked))


def mine_kobest_boolq(raw):
    """BoolQ 세 판 → 줄 목록 · 요약. 근거 = 문단(길면 줄임) · 주장 = 질문 · 물음꼴이면 needs_rewrite."""
    rows, dropped, paragraphs = [], collections.Counter(), {}
    for name, split in BOOLQ_FILES:
        paragraphs[split] = []
        for index, line in enumerate(read_text(os.path.join(raw, name)).splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            paragraph = " ".join(item["paragraph"].split())
            question = " ".join(item["question"].split())
            paragraphs[split].append(paragraph)
            evidence = shorten_paragraph(paragraph, question)
            if not text_ok(evidence):
                dropped["evidence"] += 1
                continue
            if len(question) < 5 or _norm(question) == _norm(evidence):
                dropped["claim"] += 1
                continue
            gid = "boolq:" + short_hash(_norm(paragraph))
            rows.append({
                "id": f"boolq-{split}-{index + 1:05d}", "evidence": evidence, "claim": question, "want": "",
                "hint": BOOLQ_HINT[int(item["label"])], "form": "labeled", "domain": "kobest-boolq",
                "group": gid, "groups": [gid], "lang": "ko", "lic": "CC-BY-SA-4.0", "split": split,
                "src_ids": [f"kobest-boolq:{split}:{index + 1:05d}"], "origin": "g3-boolq",
                "needs_rewrite": is_asking(question), "shortened": evidence != paragraph,
            })
    overlap = {}
    for split in ("dev", "test"):
        found = overlap_against(paragraphs["train"], paragraphs[split], BOOLQ_NEAR)
        hit_norm = {_norm(paragraphs[split][i]): kind for i, kind in found.items()}
        unique = {_norm(p) for p in paragraphs[split]}
        overlap[split] = {"paragraphs": len(unique), "rows": 0,
                          "exact": sum(1 for k in hit_norm.values() if k == "exact"),
                          "near": sum(1 for k in hit_norm.values() if k == "near")}
        # 근거는 줄여졌을 수 있어 문단 정규형의 해시(= group)로 줄을 찾는다
        by_group = {"boolq:" + short_hash(n): kind for n, kind in hit_norm.items()}
        for row in rows:
            if row["split"] == split and row["group"] in by_group:
                row["near_train"] = by_group[row["group"]]
                overlap[split]["rows"] += 1
    summary = {
        "source": "kobest-boolq", "rows": len(rows), "dropped": dict(dropped), "overlap_train": overlap,
        "by_split": {}, "near_threshold": BOOLQ_NEAR,
    }
    for split in ("train", "dev", "test"):
        mine = [r for r in rows if r["split"] == split]
        asking = sum(r["needs_rewrite"] for r in mine)
        summary["by_split"][split] = {
            "rows": len(mine), "hint": dict(sorted(collections.Counter(r["hint"] for r in mine).items())),
            "needs_rewrite": asking, "declarative_share": round(1 - asking / len(mine), 3) if mine else 0.0,
            "shortened": sum(r["shortened"] for r in mine), "groups": len({r["group"] for r in mine}),
        }
    return rows, summary


def run_kobest_boolq(raw, out_dir):
    rows, summary = mine_kobest_boolq(raw)
    for _, split in BOOLQ_FILES:
        write_jsonl(os.path.join(out_dir, split + ".jsonl"), [r for r in rows if r["split"] == split])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


# ---------------------------------------------------------------- KLUE-MRC 답 있는 질문 (v1.2 K 판 · 설계 3-5)
#
# g1 klue-mrc 후보 가운데 지문이 답을 글자로 적는 질문(유형 1 · 2)만, 답이 날짜 · 서수 · 수 · 사람 · 곳인 것.
# b = 질문 + 같은 지문 속 같은 종류의 다른 값(alt_answer) · a = 질문 + 참 답. 평서문 만들기는 다음 판(로컬 생성)이라
# claim 은 질문 그대로 두고 needs_rewrite 를 단다. 같은 종류의 다른 값이 없으면 그 질문은 버린다.

KLUE_ANS_CAP = 4                       # 한 문서(묶음)에서 고르는 질문 상한 — 한 질문이 줄 둘(a · b)
_KLUE_GROUP = re.compile(r"klue:[0-9a-f]{12}")
_KLUE_GUID = re.compile(r"klue-mrc-v1_(?:train|dev)_\d+")
_LEAD = re.compile(r"^(?:약|총|무려|최대|최소|최고|단|불과)\s?")
_DATE = re.compile(r"^(?:\d{4}년(?:\s?\d{1,2}월(?:\s?\d{1,2}일)?)?|\d{1,2}월\s?\d{1,2}일|\d{1,2}월|\d{1,2}일|"
                   r"\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?)$")
_ORDINAL = re.compile(r"^(?:제\s?)?\d+\s?(?:위|번째|차|회|등|기|세대|라운드|호|번)$")
_ORDINAL_WORD = re.compile(r"^(?:첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s?번째$")
_NUMBER = re.compile(r"^\d[\d,.]*(?:\s?[만억조천]\s?[\d,.]*)*\s?([%％]|[가-힣㎡㎞㎏]{1,3})$")
# 사람 : 「이름 씨」 · 「이름 직함」(사이 빈칸) 꼴의 세 자 이름(성 한 자) · 네 자 이름(두 자 성)만. 「기초과학연구원」처럼
# 낱말에 붙은 직함을 이름으로 잘못 읽지 않으려고 직함 앞 빈칸을 요구하고, 성 글자로 거른다.
_TITLES = ("대표|감독|의원|장관|회장|교수|선수|위원장|사장|대통령|총리|부총리|단장|원장|청장|지사|기자|작가|박사|"
           "변호사|코치|부회장|전무|상무|팀장|의장|본부장|실장|부장|차관|총장|소장|대변인|검사|판사")
_PERSON = re.compile(r"(?<![가-힣])([가-힣]{3,4})(?:\s?씨|\s(?:" + _TITLES + "))")
_SURNAME = frozenset("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공"
                     "현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉")
_SURNAME2 = ("남궁", "선우", "제갈", "황보", "독고", "사공", "서문")
_NAME_BAD_END = frozenset("을의를에당부대이가는도과로")   # 「한국을 · 민주당 · 국방부 · 서울대 · 경험이」 꼴 (이 · 가로 끝나는 이름은 버림을 감수)
_NOT_NAME = frozenset("이사회 마케팅 경제학 지주사 정책위 정치국 국내외 주니어 유전자 하지만 지난해 최연소 원로원 전교조 "
                      "신세계 기업체 전경련 현대차 공화당 고용부".split())
_PLACE_Q = re.compile(r"어디|어느\s?(?:나라|도시|지역|국가|곳|장소|섬|대륙)|(?:나라|도시|지역|국가|장소|곳)(?:은|는|이|가)?\s*[?？]?\s*$")
_PLACE_SUFFIX = ("특별시", "광역시", "시", "군", "구", "현", "주", "역", "공항", "항", "섬", "국")
_JOSA_AFTER = r"(?=[^가-힣]|$|은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만)"
COUNTRIES = frozenset(
    "한국 대한민국 북한 미국 중국 일본 러시아 영국 독일 프랑스 이탈리아 스페인 포르투갈 네덜란드 벨기에 스위스 오스트리아 "
    "스웨덴 노르웨이 덴마크 핀란드 폴란드 체코 헝가리 그리스 터키 튀르키예 이집트 이란 이라크 이스라엘 사우디아라비아 "
    "인도 파키스탄 베트남 태국 필리핀 인도네시아 말레이시아 싱가포르 대만 홍콩 몽골 캐나다 멕시코 브라질 아르헨티나 칠레 "
    "페루 콜롬비아 호주 오스트레일리아 뉴질랜드 남아프리카공화국 나이지리아 케냐 우크라이나 아일랜드 쿠바 카타르".split())
CITIES = frozenset(
    "서울 부산 대구 인천 광주 대전 울산 세종 수원 성남 고양 용인 창원 청주 전주 천안 포항 제주 강릉 춘천 원주 "
    "도쿄 오사카 교토 베이징 상하이 홍콩 타이베이 뉴욕 워싱턴 로스앤젤레스 시카고 샌프란시스코 런던 파리 베를린 로마 "
    "마드리드 모스크바 시드니 토론토 밴쿠버 싱가포르 방콕 하노이 호치민 뭄바이 두바이 빈 프라하 암스테르담 브뤼셀".split())


_KLUE_SENT = re.compile(r"(?<=[.!?。])\s+|(?<=[다요음]\.)(?=[^\s\d.,)」』’”'\"])")


def klue_sentences(context):
    """KLUE 뉴스 지문은 「했다.다음 문장」처럼 마침표 뒤 빈칸이 없는 곳이 많아 그 자리도 끊는다."""
    return [" ".join(p.split()) for p in _KLUE_SENT.split(context) if p and p.strip()]


def read_forbid_klue(paths):
    """금지 파일 · 폴더(*.json · *.jsonl 아래 전부)에서 KLUE 묶음(klue:…)과 질문 guid 를 모은다."""
    groups, guids = set(), set()
    files = []
    for path in paths or []:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, n) for n in sorted(names) if n.endswith((".json", ".jsonl")))
        else:
            files.append(path)
    for path in files:
        text = read_text(path)
        groups.update(_KLUE_GROUP.findall(text))
        guids.update(_KLUE_GUID.findall(text))
    return groups, guids


def _squash(text):
    return re.sub(r"\s+", "", text or "")


def _find_values(pattern, context):
    out = []
    for match in re.finditer(pattern, context):
        value = (match.group(1) if match.groups() else match.group(0)).strip()
        if value and value not in out:
            out.append(value)
    return out


def is_name(word):
    if word in _NOT_NAME or word[-1] in _NAME_BAD_END:
        return False
    if len(word) == 3:
        return word[0] in _SURNAME
    return len(word) == 4 and word[:2] in _SURNAME2


def _shape_pattern(core, kind):
    """답 꼴 → 같은 꼴 값을 찾는 정규식. 날짜는 네 자리 해를 네 자리로 지킨다."""
    out = []
    for piece in re.split(r"(\d+|\s+)", core):
        if not piece:
            continue
        if piece.isdigit():
            out.append(r"\d{4}" if kind == "date" and len(piece) == 4 else (r"\d{1,2}" if kind == "date" else r"\d+"))
        elif piece.isspace():
            out.append(r"\s?")
        else:
            out.append(re.escape(piece))
    return r"(?<![\d.,])" + "".join(out) + r"(?![\d])"


def answer_kind(question, answer, context):
    """답 유형과 같은 종류 값 후보(지문 속, 답 뺌 전) → (kind, [값]) · 맞는 유형이 없으면 (None, [])."""
    core = _LEAD.sub("", answer.strip())
    if len(core) > 20:
        return None, []
    if _DATE.match(core):
        return "date", _find_values(_shape_pattern(core, "date"), context)
    if _ORDINAL.match(core):
        return "ordinal", _find_values(_shape_pattern(core, "ordinal"), context)
    if _ORDINAL_WORD.match(core):
        return "ordinal", _find_values(r"(?:첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s?번째", context)
    number = _NUMBER.match(core)
    if number and number.group(1):
        unit = number.group(1).lstrip("만억조천")
        if unit:
            # 「5년」 같은 기간이 「2001년」 같은 해를 고르지 않게 년 단위는 세 자리까지만
            lead = r"\d{1,3}(?![\d])" if unit.startswith("년") else r"\d[\d,.]*"
            pattern = r"(?<![\d.,])" + lead + r"(?:\s?[만억조천]\s?[\d,.]*)*\s?[만억조천]?" + re.escape(unit)
            return "number", _find_values(pattern, context)
    if re.fullmatch(r"[가-힣]{2,4}", core):
        names = [n for n in _find_values(_PERSON, context) if is_name(n)]
        if core in names:
            return "person", names
    if _PLACE_Q.search(question) and re.fullmatch(r"[가-힣]{2,8}", core):
        for pool in (COUNTRIES, CITIES):
            if core in pool:
                return "place", [w for w in pool if re.search(r"(?<![가-힣])" + w + _JOSA_AFTER, context)]
        for suffix in sorted(_PLACE_SUFFIX, key=len, reverse=True):
            if core.endswith(suffix) and len(core) > len(suffix):
                pattern = r"(?<![가-힣])([가-힣]{1,6}" + suffix + ")" + _JOSA_AFTER
                return "place", _find_values(pattern, context)
    return None, []


def pick_alt(answers, values, rng):
    """같은 종류 값 가운데 참 답(여러 표기)과 다르고 서로 품지 않는 것 하나. 없으면 None."""
    truth = [_squash(_LEAD.sub("", a)) for a in answers if a]
    pool = [v for v in values
            if _squash(v) and all(_squash(v) != t and _squash(v) not in t and t not in _squash(v) for t in truth)]
    return rng.choice(pool) if pool else None


def mine_klue_answered(raw, cands_path, forbid_paths, rng, cap=KLUE_ANS_CAP):
    """g1 klue-mrc 후보 · KLUE 원자료 → (줄 목록, 요약). 한 질문이 b(alt) · a(참) 두 줄, 같은 `pair`."""
    allowed = {}
    for line in read_text(cands_path).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("form") != "unanswerable" and row.get("answer"):
            for guid in row.get("src_ids", []):
                allowed[guid] = row.get("split", "")
    forbid_groups, forbid_guids = read_forbid_klue(forbid_paths)
    counts = collections.Counter()
    by_group = collections.defaultdict(list)
    for name in ("klue-mrc-v1.1_train.json", "klue-mrc-v1.1_dev.json"):
        for article in read_json(os.path.join(raw, name))["data"]:
            gid = "klue:" + short_hash(article["title"])
            for para in article["paragraphs"]:
                context = para["context"]
                sentences = klue_sentences(context)
                index = None
                for qa in para["qas"]:
                    guid = qa["guid"]
                    if guid not in allowed or qa.get("is_impossible"):
                        continue
                    counts["g1_answered"] += 1
                    if qa.get("question_type") not in (1, 2):
                        counts["drop_qtype"] += 1
                        continue
                    if gid in forbid_groups or guid in forbid_guids:
                        counts["drop_forbid"] += 1
                        continue
                    answers = [a["text"] for a in qa["answers"]]
                    answer = qa["answers"][0]
                    kind, values = answer_kind(qa["question"], answer["text"], context)
                    if kind is None:
                        counts["drop_kind"] += 1
                        continue
                    alt = pick_alt(answers, values, rng)
                    if alt is None:
                        counts["drop_no_alt"] += 1
                        continue
                    at = _sentence_at(sentences, context, answer["answer_start"])
                    if at is None or answer["text"] not in sentences[at]:
                        counts["drop_sentence"] += 1
                        continue
                    chosen = [at]
                    if qa["question_type"] == 2:      # 여러 문장 추론 : 질문과 가장 닮은 다른 문장 하나를 더한다
                        if index is None:
                            index = BM25(sentences)
                        chosen += [i for i in index.rank(qa["question"]) if i != at][:1]
                    evidence = join_within([sentences[i] for i in sorted(chosen)]) or sentences[at]
                    if not text_ok(evidence):
                        counts["drop_evidence"] += 1
                        continue
                    by_group[gid].append({
                        "guid": guid, "question": " ".join(qa["question"].split()), "answer": answer["text"],
                        "alt": alt, "kind": kind, "evidence": evidence, "split": allowed[guid],
                        "qtype": qa["question_type"], "alt_in_evidence": alt in evidence,
                    })
    rows = []
    for gid in sorted(by_group):
        items = by_group[gid]
        rng.shuffle(items)
        counts["drop_cap"] += max(0, len(items) - cap)
        for item in sorted(items[:cap], key=lambda x: x["guid"]):
            for hint, fill, form in (("b", item["alt"], "qa-alt"), ("a", item["answer"], "qa-true")):
                rows.append({
                    "id": f"klue-ans-{len(rows) + 1:06d}", "evidence": item["evidence"], "claim": item["question"],
                    "want": "", "hint": hint, "form": form, "domain": "klue-answered", "group": gid, "groups": [gid],
                    "lang": "ko", "lic": "CC-BY-SA-4.0", "split": item["split"], "src_ids": [item["guid"]],
                    "origin": "g3-klue", "needs_rewrite": True, "question": item["question"],
                    "answer": item["answer"], "alt_answer": item["alt"], "fill": fill, "ans_type": item["kind"],
                    "qtype": item["qtype"], "alt_in_evidence": item["alt_in_evidence"], "pair": item["guid"],
                })
    questions = [r for r in rows if r["hint"] == "b"]
    summary = {
        "source": "klue-answered", "rows": len(rows), "questions": len(questions),
        "groups": len({r["group"] for r in rows}), "cap_per_group": cap,
        "by_type": dict(sorted(collections.Counter(r["ans_type"] for r in questions).items())),
        "by_split": dict(sorted(collections.Counter(r["split"] for r in questions).items())),
        "alt_in_evidence": sum(r["alt_in_evidence"] for r in questions),
        "forbid": {"groups": len(forbid_groups), "guids": len(forbid_guids)},
        "counts": dict(sorted(counts.items())),
    }
    return rows, summary


def run_klue_answered(raw, cands_path, forbid_paths, out_path, rng):
    rows, summary = mine_klue_answered(raw, cands_path, forbid_paths, rng)
    write_jsonl(out_path, rows)
    with open(os.path.splitext(out_path)[0] + ".summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


MINERS = {
    "proposals": mine_proposals,
    "vitaminc": mine_vitaminc,
    "klue-mrc": mine_klue,
    "minicheck": mine_minicheck,
    "ood": mine_ood,
    "jira": mine_jira,
    "vitaminc-heldout": lambda raw, rng: mine_vitaminc_g2(raw, rng, "heldout"),
    "vitaminc-train": lambda raw, rng: mine_vitaminc_g2(raw, rng, "train"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="공개 원천 열쇠 묶음 · 후보 쌍 뽑기")
    parser.add_argument("source", choices=sorted(MINERS) + ["kobest-boolq", "klue-answered"])
    parser.add_argument("--raw", required=True, help="원천 폴더 (ood 는 raw 뿌리)")
    parser.add_argument("--out", required=True,
                        help="후보를 쓸 뿌리 (원천 이름 폴더가 생긴다) · kobest-boolq 는 그 폴더에 split 별 jsonl · "
                             "klue-answered 는 jsonl 경로")
    parser.add_argument("--cands", help="klue-answered : g1 klue-mrc candidates.jsonl")
    parser.add_argument("--forbid", nargs="*", default=[],
                        help="klue-answered : g1 · g2 가 쓴 KLUE 묶음 · 줄을 뽑아 뺄 파일 · 폴더")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    if args.source == "kobest-boolq":
        summary = run_kobest_boolq(args.raw, args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    if args.source == "klue-answered":
        if not args.cands:
            parser.error("klue-answered 는 --cands 가 있어야 한다")
        summary = run_klue_answered(args.raw, args.cands, args.forbid, args.out, rng)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    collector = MINERS[args.source](args.raw, rng)
    summary = collector.write(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
