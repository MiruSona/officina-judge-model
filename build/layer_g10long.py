"""layer_g10long — K8 긴 줄 층 train-g10-long 후보 · 목표 셋 dev-long-g10 재료 (비공개 설계 K8 2-1 ~ 2-6).

근거 = 여러 토막을 이은 긴 줄, 주장 = 같은 줄(또는 그 일부)에서 한 토막만 바꾼 것. 같은 꼴로 a · c · d 도 짓는다.
공개 원천만 읽는다 — 한국어 위키 덤프(본문 · 표)와 VitaminC(학습 쪽). 사설 저장소를 읽는 seed_contrast 와 섞지 않고
토막 찾기 · 바꾸기 함수만 import 한다. 쌍 본문은 git 밖(`~/.judgemodel/build/g10/`)에만 쓴다.

    python -I layer_g10long.py pick  --kowiki <dir> --vitaminc <dir> --avoid-sets <jsonl …> --out <B>/long
                                     [--docs-out <B>/dev-long/ko-docs.jsonl]       # 근거 후보 pick.jsonl (+ 목표 셋 ko 문서 풀)
    python -I layer_g10long.py make  --in <B>/long/pick.jsonl --n 1800 --seed 20262201 --out <B>/long   # out.jsonl · awkward.jsonl
    python -I layer_g10long.py check --in <B>/long/out.jsonl --synth <synth-30> --real <real-50> [--out <pass.jsonl>]
    python -I layer_g10long.py build --dirs <B>/long/lab --out <B>/train-g10-long   # 판정 뒤 칸 몫대로 → cands.jsonl
    python -I layer_g10long.py cnli  --raw <raw/2026-10-11> --avoid-sets <ood-g1.jsonl> --out <B>/dev-long/en-cands.jsonl

글 꼴(shape) : head = `[절 : … · 줄 머리 : … · 칸 머리 : …] 칸 글`(위키 표, conflict mdscan 과 같은 꼴) · join = 토막 3~6개를
` · ` ` — ` ` / ` 로 이은 줄(앞에 `[절 : …]` 를 붙이기도) · para = 머리 없는 2~5문장 문단.
토큰은 학습 · 평가 · 서버와 같은 토크나이저(tokenizer.json, `pairenc.pair_length`)로 센다 — tokenizers 가 든 python 으로 돈다.
id 꼬리는 `-s<쌍 sha 앞 8자>`(build/README 34항 주의).
"""
import argparse
import bz2
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

import label_batches as lb  # noqa: E402
import mine_public as mp  # noqa: E402
import mine_wiki as mw  # noqa: E402
import scrub  # noqa: E402
import seed_contrast as sc  # noqa: E402
from make_sets import sha_key  # noqa: E402

SEED_PICK = 20262200
MAX_LEN = 320                       # 학습 · 평가 --max-len — 쌍이 이보다 길면 근거 뒤가 잘려 라벨이 거짓이 될 수 있다
EV_TOKENS = (30, 200)               # 근거 토큰 범위 — 주장까지 붙여 320 안에 들게 (설계 2-1 「120~600자」는 어림이라 토큰으로 묶음)
KO_BODY_CHARS = (120, 600)          # join · para 꼴 본문 글자 (설계 2-1)
KO_CELL_CHARS = 30                  # head 꼴 칸 글 하한 — 위키 표 칸은 짧아서 본문 하한을 따로 둔다
EN_BODY_WORDS = (40, 250)           # en 본문 낱말 (설계 2-1 의 80~400 은 320 토큰에 안 들어 줄임)
GROUP_CAP = 2                       # 한 문서 · 한 쪽에서 근거 둘까지 (학습 층 「묶음당 2」)
SHAPES = ("head", "join", "para")
LANG_SHARE = {"ko": 0.65, "en": 0.35}
FORMS = {  # 꼴 : (칸, 1,200 줄 기준 몫) — 설계 2-1 표
    "long-num": ("b", 180), "long-neg": ("b", 150), "long-flip": ("b", 90),
    "long-same": ("a", 160), "long-alt": ("a", 120), "long-bothneg": ("a", 140),
    "long-other": ("c", 240), "long-time": ("d", 120),
}
FAMILY = {"b": ("long-num", "long-neg", "long-flip"), "a": ("long-same", "long-alt", "long-bothneg"),
          "cd": ("long-other", "long-time")}   # 근거 하나에 칸 갈래마다 한 줄까지 — 같은 근거의 a · b 가 같이 든다
SEPS = ((" · ", 0.6), (" — ", 0.25), (" / ", 0.15))
LIC = {"ko": "CC-BY-SA-4.0", "en": "CC-BY-SA-3.0"}
DOMAIN = {"ko": "kowiki", "en": "vitaminc"}
ORIGIN = "g10-long"

# ---------------------------------------------------------------- 토큰 셈

class Tok:
    """tokenizer.json 하나 — 근거 토큰 수 · 쌍 길이(`pairenc.pair_length`, 특수 토큰 포함, 자르기 없이)."""

    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            sys.exit("tokenizers 가 없다 — 서빙 venv python 으로 돌린다 (학습 · 평가와 같은 셈이어야 한다)")
        import pairenc
        self._pair = pairenc.pair_length
        self.tok = Tokenizer.from_file(path)

    def count(self, text):
        self.tok.no_truncation()
        self.tok.no_padding()
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def pair(self, evidence, claim):
        return self._pair(self.tok, evidence, claim)


def default_tokenizer():
    return os.path.join(os.path.expanduser("~"), ".judgemodel", "models", "nli-studio-v10", "tokenizer", "tokenizer.json")


# ---------------------------------------------------------------- 위키 문서 읽기

_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def latest_pages(paths, limit_pages=0, log=sys.stderr):
    """역사 덤프들 → 본문 이름공간 · 넘겨주기 아닌 문서의 마지막 판 {id, title, text} (문서마다 하나)."""
    for path in paths:
        with bz2.open(path, "rb") as stream:
            last, n = None, 0
            for event, page, rev in mw.iter_pages(stream, limit_pages):
                if event == "rev":
                    if rev["text"] is not None:
                        last = rev["text"]
                    continue
                n += 1
                if page["ns"] == "0" and not page["redirect"] and last:
                    yield {"id": page["id"], "title": page["title"], "text": last}
                last = None
                if n % 2000 == 0:
                    print(f"[layer_g10long] {os.path.basename(path)} 문서 {n:,}", file=log, flush=True)


def cached_pages(paths, limit_pages=0, cache=None):
    """latest_pages 를 cache(jsonl)에 받아 두고 다음엔 그걸 읽는다 — 역사 덤프를 다시 훑는 데 30분 넘게 든다.
    첫 줄 {"_meta": {"limit_pages": n}} 이 이번 --limit-pages 와 다르면 멈춘다 (덜 읽은 캐시를 다 읽은 것으로 쓰지 않게)."""
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:     # 한 줄씩 — 통째로 읽으면 메모리를 많이 먹는다
            first = True
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if first and "_meta" in row:
                    if row["_meta"].get("limit_pages") != limit_pages:
                        raise SystemExit(f"[layer_g10long] --pages-cache 는 --limit-pages {row['_meta'].get('limit_pages')}"
                                         f" 로 받은 것이다 (이번 {limit_pages}) — 캐시를 지우거나 값을 맞춘다")
                    first = False
                    continue
                if first:            # shortcut: 한도 기록 없는 캐시는 확인 못 하고 읽는다, 다시 받으면 meta 가 붙는다
                    print("[layer_g10long] --pages-cache 에 limit_pages 기록이 없다 — 한도를 확인 못 함", file=sys.stderr)
                first = False
                yield row
        return
    if not cache:
        yield from latest_pages(paths, limit_pages)
        return
    tmp = cache + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_meta": {"limit_pages": limit_pages}}) + "\n")
        for page in latest_pages(paths, limit_pages):
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
            yield page
    os.replace(tmp, cache)


def _cell_text(cell):
    """위키 표 칸 하나 → 글. 「style=… | 글」 꼴이면 속성을 뗀다."""
    if "|" in cell:
        attrs, _, rest = cell.partition("|")
        if "=" in attrs or not attrs.strip():
            cell = rest
    return mw.strip_wiki(cell)


def read_table(lines, i):
    """lines[i] 가 `{|` — (다음 줄 번호, 표 {head, rows} 또는 None). 칸 합치기 · 안긴 표 · 칸 수가 머리와 다른 줄은 버린다."""
    depth, rows, j = 0, [], i
    broken = False
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith("{|"):
            depth += 1
            broken |= depth > 1
        elif s.startswith("|}"):
            depth -= 1
            if depth == 0:
                break
        elif depth == 1 and not s.startswith("|+"):
            if "rowspan" in s or "colspan" in s:
                broken = True
            if s.startswith("|-"):
                rows.append([])
            elif s.startswith("!") or s.startswith("|"):
                if not rows:
                    rows.append([])
                header = s.startswith("!")
                parts = re.split(r"!!|\|\|", s[1:]) if header else s[1:].split("||")
                rows[-1].extend((header, _cell_text(p)) for p in parts)
            elif rows and rows[-1]:
                head, text = rows[-1][-1]
                rows[-1][-1] = (head, (text + " " + mw.strip_wiki(s)).strip())
        j += 1
    rows = [r for r in rows if r]
    if broken or len(rows) < 3 or not all(h for h, _ in rows[0]):
        return j + 1, None
    head = [t for _, t in rows[0]]
    body = [[t for _, t in r] for r in rows[1:] if len(r) == len(head)]
    return j + 1, ({"head": head, "rows": body} if len(head) >= 2 and len(body) >= 2 else None)


def parse_page(text):
    """위키 글 → 절 목록 [{title, paras, tables}]. 첫 절(머리말)의 title 은 빈 글."""
    sections = [{"title": "", "paras": [], "tables": []}]
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        m = _HEADING.match(stripped)
        if m:
            sections.append({"title": mw.strip_wiki(m.group(2)), "paras": [], "tables": []})
            i += 1
            continue
        if stripped.startswith("{|"):
            i, table = read_table(lines, i)
            if table:
                sections[-1]["tables"].append(table)
            continue
        clean = mw.strip_wiki(raw)
        if mw.prose_ok(raw, clean) and len(clean) >= 20:
            sections[-1]["paras"].append(clean)
        i += 1
    return [s for s in sections if s["paras"] or s["tables"]]


# ---------------------------------------------------------------- 근거 줄 짓기

def head_ctx(section, row_head=None, col_head=None):
    """conflict mdscan 꼴 머리 — `[절 : S · 줄 머리 : R · 칸 머리 : C] `."""
    parts = [f"절 : {section}"]
    if row_head is not None:
        parts += [f"줄 머리 : {row_head}", f"칸 머리 : {col_head}"]
    return "[" + " · ".join(parts) + "] "


def _sep(rng):
    r = rng.random()
    for sep, w in SEPS:
        if r < w:
            return sep
        r -= w
    return SEPS[0][0]


def join_body(pieces, rng):
    """토막들을 섞인 이음표로 — 끝 마침표는 떼어 「줄」 꼴로."""
    out = pieces[0].rstrip(". ")
    for p in pieces[1:]:
        out += _sep(rng) + p.rstrip(". ")
    return out


def _ko_body_ok(body):
    return KO_BODY_CHARS[0] <= len(body) <= KO_BODY_CHARS[1]


def _en_body_ok(body):
    return EN_BODY_WORDS[0] <= len(body.split()) <= EN_BODY_WORDS[1]


def ko_lines(page, rng):
    """위키 문서 하나 → 근거 줄 후보 {shape: [줄]}. 줄 = {head, body, section, other}. other = 다른 절 문장(c 꼴 재료)."""
    sections = parse_page(page["text"])
    sent_by_sec = [[t for p in s["paras"] for t in mw.sentences(p) if 20 <= len(t) <= 200] for s in sections]
    out = {k: [] for k in SHAPES}
    out["_tables"] = sum(len(s["tables"]) for s in sections)

    def other_of(k):
        pool = [t for j, ss in enumerate(sent_by_sec) if j != k for t in ss]
        rng.shuffle(pool)
        return pool[:3]

    for k, sec in enumerate(sections):
        title = sec["title"] or page["title"]
        for table in sec["tables"]:
            for row in table["rows"]:
                if not row[0].strip():
                    continue
                for c in range(1, len(row)):
                    cell = row[c]
                    if KO_CELL_CHARS <= len(cell) <= KO_BODY_CHARS[1] and _prose_cell(cell) and _has_target(cell, "ko"):
                        out["head"].append({"head": head_ctx(title, row[0], table["head"][c]), "body": cell,
                                            "section": title, "other": other_of(k)})
        sents = sent_by_sec[k]
        for start in range(0, len(sents), 6):
            chunk = sents[start:start + 6]
            if len(chunk) >= 3:
                n = rng.randint(3, min(6, len(chunk)))
                body = join_body(chunk[:n], rng)
                if _ko_body_ok(body):
                    head = head_ctx(title) if rng.random() < 0.5 else ""
                    out["join"].append({"head": head, "body": body, "section": title, "other": other_of(k)})
            if len(chunk) >= 2:
                n = rng.randint(2, min(5, len(chunk)))
                body = " ".join(chunk[:n])
                if _ko_body_ok(body):
                    out["para"].append({"head": "", "body": body, "section": title, "other": other_of(k)})
    return out


def _prose_cell(cell):
    """표 칸이 글인가 — 수 · 기호 나열(「1, 2, 3, 7」 「≈ 0.66016 …」)은 빼고 한글 낱말 셋 이상."""
    return len(re.findall(r"[가-힣]{2,}", cell)) >= 3


def _has_target(text, lang):
    """바꿀 자리(값 · 부정어 · 반대말)가 하나라도 있나."""
    return bool(value_slots(text, lang) or sc.find_neg_slots(text, lang) or flip_slots(text, lang))


# ---------------------------------------------------------------- en 자리 (seed_contrast 는 한국어 꼴이라 따로)

_EN_NUM = re.compile(r"(?<![\w.,\-/:#$])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(%)?(?![\w\-/:]|[.,]\d)")
_EN_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
_EN_FLIP = {"first": "last", "last": "first", "before": "after", "after": "before", "above": "below", "below": "above",
            "increased": "decreased", "decreased": "increased", "higher": "lower", "lower": "higher",
            "largest": "smallest", "smallest": "largest", "earlier": "later", "later": "earlier",
            "maximum": "minimum", "minimum": "maximum", "won": "lost", "lost": "won", "older": "younger",
            "younger": "older", "oldest": "youngest", "youngest": "oldest", "northern": "southern",
            "southern": "northern", "eastern": "western", "western": "eastern"}
_EN_FLIP_RE = re.compile(r"\b(" + "|".join(sorted(_EN_FLIP, key=len, reverse=True)) + r")\b")


def en_value_slots(text):
    out = []
    for m in _EN_NUM.finditer(text):
        num, pct = m.group(1), m.group(2) or ""
        if not pct and re.fullmatch(r"1[5-9]\d\d|20\d\d", num):
            out.append({"kind": "date", "form": "year_en", "start": m.start(), "end": m.end(), "text": m.group(0)})
        else:
            out.append({"kind": "num", "start": m.start(), "end": m.end(), "text": m.group(0), "num": num, "unit": pct})
    return out


def en_new_value(slot, text, rng):
    if slot["kind"] == "date":
        for _ in range(4):
            cand = str(int(slot["text"]) + rng.choice((-3, -2, -1, 1, 2, 3)))
            if not sc.has_value(cand, text):
                return cand
        return None
    for cand in sc._num_candidates(slot["num"], slot["unit"], rng):
        if cand != slot["num"] and not sc.has_value(cand, text):
            return cand + slot["unit"]
    return None


def en_alt(slot):
    """a 짝 : 같은 값 다른 꼴 — 10 이하 정수 → 낱말 · 1,500 ↔ 1500 · % → percent."""
    if slot["kind"] != "num":
        return None
    num, unit = slot["num"], slot["unit"]
    if unit == "%":
        return f"{num} percent"
    if "," in num:
        return num.replace(",", "")
    if num.isdigit() and int(num) in _EN_WORDS:
        return _EN_WORDS[int(num)]
    if num.isdigit() and int(num) >= 1000:
        return f"{int(num):,}"
    return None


def value_slots(text, lang):
    """수 · 날짜 · 차례 자리 (ko 는 seed_contrast.find_slots, 누가 자리는 사설 저장소 낱말이라 뺀다)."""
    if lang == "en":
        return en_value_slots(text)
    return [s for s in sc.find_slots(text) if s["kind"] in ("num", "date", "ord")]


def flip_slots(text, lang):
    if lang == "en":
        return [{"kind": "flip", "start": m.start(), "end": m.end(), "text": m.group(0)} for m in _EN_FLIP_RE.finditer(text)]
    return [s for s in sc.find_slots(text) if s["kind"] in ("onoff", "bound", "firstlast")]


# ---------------------------------------------------------------- 토막

_SEG_SPLIT = re.compile(r"\s[·—/]\s|(?<=[.!?])\s+")   # 쉼표에서는 안 끊는다 — 쉼표 조각을 빼면 문장이 깨진다


def segments(body):
    """본문 → 토막 [(시작, 끝)]. 이음표 · 문장 끝에서 끊는다."""
    out, start = [], 0
    for m in _SEG_SPLIT.finditer(body):
        if m.start() > start:
            out.append((start, m.start()))
        start = m.end()
    if start < len(body):
        out.append((start, len(body)))
    return out


def rebuild(body, spans, keep):
    """keep(토막 번호, 차례대로)만 남긴 본문 — 각 토막 앞의 원래 이음표를 따라 붙인다(첫 토막은 이음표 없이)."""
    out = ""
    for n, i in enumerate(keep):
        if n:
            out += body[spans[i - 1][1]:spans[i][0]] if i > 0 else " · "
        out += body[spans[i][0]:spans[i][1]]
    return out


def position(body, start, end):
    """바꾼 자리가 본문 앞 · 가운데 · 끝 어디인가 (가운데 글자 자리로)."""
    at = (start + end) / 2 / max(1, len(body))
    return "front" if at < 1 / 3 else ("end" if at >= 2 / 3 else "mid")


# ---------------------------------------------------------------- make : 꼴별 주장 초안

def _changes(line, lang, rng):
    """본문 안 바꿀 자리 후보 → {form: [(slot, new, extra)]}. new 는 자리를 바꿀 새 글."""
    body = line["body"]
    out = collections.defaultdict(list)
    for slot in value_slots(body, lang):
        new = en_new_value(slot, body, rng) if lang == "en" else sc._new_keeping_order(slot, body, rng)
        if new:
            out["long-num"].append((slot, new, {}))
        alt = en_alt(slot) if lang == "en" else sc.same_value_other_form(slot)
        out["long-alt"].append((slot, alt, {"rule": bool(alt)}))
    for slot in sc.find_neg_slots(body, lang):
        out["long-neg"].append((slot, slot["flip"], {"awkward": slot["awkward"], "dir": slot["dir"]}))
    for slot in flip_slots(body, lang):
        new = _EN_FLIP[slot["text"]] if lang == "en" else sc.new_value(slot, body, rng)
        if new:
            out["long-flip"].append((slot, new, {}))
    out["long-time"] = _time_changes(body, lang, rng)
    return out


def _time_changes(body, lang, rng):
    """d : 같은 토막의 때 자리 하나만 바꿈 — 값까지 바꾸면 b 로 읽힌다.
    값 새 글도 뽑아 버린다 — rng 쓰임이 같아야 다른 꼴 출력이 흔들리지 않는다."""
    out = []
    spans = segments(body)
    slots = value_slots(body, lang)
    for s, e in spans:
        inside = [x for x in slots if s <= x["start"] and x["end"] <= e]
        times = [x for x in inside if x["kind"] == "date"]
        values = [x for x in inside if x["kind"] != "date"]
        if not times or not values:
            continue
        if lang == "ko":
            d = sc._plan_d(body, values[0], times[0], "", "", rng)
            tnew = d and d["time_new"]
        else:
            tnew, vnew = en_new_value(times[0], body, rng), en_new_value(values[0], body, rng)
            tnew = vnew and tnew
        if not tnew:
            continue
        t = times[0]
        draft = body[:t["start"]] + tnew + body[t["end"]:]
        if lang == "ko" and sc.order_broken(draft) and not sc.order_broken(body):
            continue
        out.append(({"kind": "time", "start": t["start"], "end": t["end"], "text": t["text"]}, draft,
                    {"d": {"time_slot": t["text"], "time_new": tnew, "draft": draft}}))
    return out


def _seg_of(spans, start):
    for i, (s, e) in enumerate(spans):
        if s <= start < e:
            return i
    return len(spans) - 1


def _window(body, spans, i, k):
    lo, hi = max(0, i - k), min(len(spans), i + k + 1)
    return rebuild(body, spans, list(range(lo, hi)))


def _fit_claim(tok, evidence, head, body, spans, seg):
    """주장 = 머리 + 본문 전체, 320 토큰에 안 들면 바뀐 토막 둘레만 (2 → 1 → 0). 못 맞추면 (None, None)."""
    claim = head + body
    if tok.pair(evidence, claim) <= MAX_LEN:
        return claim, "full"
    for k in (2, 1, 0):
        if len(spans) > 2 * k + 1 or k == 0:
            claim = head + _window(body, spans, seg, k)
            if tok.pair(evidence, claim) <= MAX_LEN:
                return claim, f"window{k}"
    return None, None


def _subset_claim(line, lang, rng, need_neg=False):
    """a (same · bothneg) : 토막 일부만 남기거나 차례를 바꾸거나 머리를 뗀다 — 값은 그대로.
    need_neg 면 부정어가 든 토막을 꼭 남긴다(bothneg)."""
    body, head = line["body"], line["head"]
    spans = segments(body)
    neg_idx = [i for i, (s, e) in enumerate(spans) if sc.count_neg(body[s:e], lang)]
    if need_neg and not neg_idx:
        return None
    options = []
    if len(spans) >= 3:
        options.append("drop")
    if head:
        options.append("nohead")
    if len(spans) >= 2 and line.get("shape") == "join":
        options.append("swap")
    if not options:
        return None
    how = rng.choice(options)
    if how == "nohead":
        return body, how, (neg_idx[0] if need_neg else None)
    keep = list(range(len(spans)))
    if how == "drop":
        must = rng.choice(neg_idx) if need_neg else rng.randrange(len(spans))
        n_keep = rng.randint(max(2, len(spans) // 2), len(spans) - 1)
        rest = [i for i in keep if i != must]
        rng.shuffle(rest)
        keep = sorted([must] + rest[:n_keep - 1])
        return head + rebuild(body, spans, keep), how, must
    i = rng.randrange(len(spans) - 1)
    keep[i], keep[i + 1] = keep[i + 1], keep[i]
    text = " · ".join(body[spans[j][0]:spans[j][1]].rstrip(". ") for j in keep)
    return head + text, how, (neg_idx[0] if need_neg else i)


def _other_claim(line, rng):
    """c : 같은 머리 + 같은 문서 다른 절 문장 — 겹침은 머리뿐."""
    if not line.get("other"):
        return None
    n = rng.randint(1, min(2, len(line["other"])))
    return line["head"] + " ".join(line["other"][:n])


def plan_line(line, rng, tok, want_forms, pos_target):
    """근거 줄 하나 → 이 줄로 지을 수 있는 꼴의 초안 {form: 초안} (want_forms 안에서만).
    초안 = {claim, slot, new, seg_old, seg_new, pos, fit, awkward(까닭 목록)}."""
    lang, body, head = line["lang"], line["body"], line["head"]
    evidence = head + body
    spans = segments(body)
    changes = _changes(line, lang, rng)
    out = {}
    for form in want_forms:
        if form in ("long-num", "long-neg", "long-flip", "long-alt", "long-time"):
            cands = changes.get(form) or []
            if not cands:
                continue
            target = pos_target.get(form, "mid")
            cands = sorted(cands, key=lambda c: (position(body, c[0]["start"], c[0]["end"]) != target, rng.random()))
            if form == "long-num":                    # 때만 바꾸면 v6 에서 d 라 b 가 서지 않는다 — 정렬 뒤에 걸러 rng 쓰임은 그대로
                cands = [c for c in cands if c[0]["kind"] != "date"]
                if not cands:
                    continue
            slot, new, extra = cands[0]
            seg = _seg_of(spans, slot["start"])
            draft = {"slot": {k: slot[k] for k in ("kind", "start", "end", "text")}, "new": new, "awkward": [],
                     "pos": position(body, slot["start"], slot["end"]), "seg_old": body[spans[seg][0]:spans[seg][1]]}
            if form == "long-alt" and not extra["rule"]:
                draft.update(claim=None, awkward=["no_rule"])
                out[form] = draft
                continue
            if form == "long-time":
                new_body = new
                draft["d"] = extra["d"]
            else:
                new_body = body[:slot["start"]] + new + body[slot["end"]:]
            new_spans = segments(new_body)
            new_seg = _seg_of(new_spans, slot["start"])     # 바꾼 자리 앞은 그대로라 시작 글자로 찾는다
            draft["seg_new"] = new_body[new_spans[new_seg][0]:new_spans[new_seg][1]]
            claim, fit = _fit_claim(tok, evidence, head, new_body, new_spans, new_seg)
            if claim is None:
                continue
            draft.update(claim=claim, fit=fit)
            if extra.get("awkward"):
                draft["awkward"].append("neg_rule_awkward")
            out[form] = draft
        elif form in ("long-same", "long-bothneg"):
            got = _subset_claim(line, lang, rng, need_neg=form == "long-bothneg")
            if not got:
                continue
            claim, how, seg = got
            if tok.pair(evidence, claim) > MAX_LEN:
                continue
            out[form] = {"claim": claim, "how": how, "fit": "full", "awkward": [],
                         "pos": position(body, *spans[seg]) if seg is not None else None}
        elif form == "long-other":
            claim = _other_claim(line, rng)
            if claim and tok.pair(evidence, claim) <= MAX_LEN:
                out[form] = {"claim": claim, "fit": "full", "awkward": [], "pos": None}
    return out


def seg_check(draft, form, line, scrubber):
    """바뀐 토막만 놓고 seed_contrast.check_line — 한 덩이 · 새 값 · 부정어 차이 · scrub. 까닭 목록."""
    if form not in ("long-num", "long-neg", "long-flip", "long-alt", "long-time") or not draft.get("claim"):
        return []
    role = {"long-alt": "a", "long-time": "d"}.get(form, "b")
    slot = dict(draft["slot"])
    if form == "long-neg":
        slot["kind"] = "neg"
    row = {"seed": draft["seg_old"], "slot": slot, "lang": line["lang"], "new": draft["new"] if role == "b" else None,
           "a_value": draft["new"] if role == "a" else None, "d": draft.get("d")}
    why = sc.check_line(draft["seg_old"], draft["seg_new"], role, row, scrubber)
    return [w for w in why if not (w == "length" and len(draft["seg_new"]) >= 2)]   # 짧은 토막은 길이 비가 흔들린다


def quotas(n, seed_counts=FORMS, share=LANG_SHARE, weights=None):
    """n 줄 → {(꼴, 말): 몫} — 1,200 기준 몫을 n 에 맞추고 말 몫으로 큰 나머지 나눔.
    weights {"꼴" 또는 "꼴/말": 배수} 를 주면 그 칸 몫을 곱한 뒤 n 에 다시 맞춘다 (모자란 꼴을 더 만들 때)."""
    weights = weights or {}
    w = {(f, lang): weights.get(f"{f}/{lang}", weights.get(f, 1.0)) for f in seed_counts for lang in share}
    total = sum(c * s * w[(f, lang)] for f, (_, c) in seed_counts.items() for lang, s in share.items())
    raw = {(f, lang): n * c * s * w[(f, lang)] / total for f, (_, c) in seed_counts.items() for lang, s in share.items()}
    out = {k: int(v) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:rest]:
        out[k] += 1
    return out


def parse_weights(text):
    """「long-alt/ko=2,long-other=0.6」 → {키: 배수}. 꼴 이름이 틀리면 멈춘다."""
    out = {}
    for item in filter(None, (x.strip() for x in (text or "").split(","))):
        key, _, val = item.partition("=")
        if key.split("/")[0] not in FORMS or (("/" in key) and key.split("/")[1] not in LANG_SHARE):
            raise SystemExit(f"[layer_g10long] --form-weights 의 꼴 이름이 틀렸다 : {key}")
        out[key] = float(val)
        if out[key] < 0:
            raise SystemExit(f"[layer_g10long] --form-weights 배수는 0 이상이다 : {item}")
    if out and not any(out.get(f"{f}/{lang}", out.get(f, 1.0)) for f in FORMS for lang in LANG_SHARE):
        raise SystemExit("[layer_g10long] --form-weights 가 모든 칸을 0 으로 만든다")
    return out


def pair_row(line, form, draft):
    evidence, claim = line["head"] + line["body"], draft["claim"]
    key = sha_key({"evidence": evidence, "claim": claim})
    row = {"id": f"g10l-{line['id']}-{form[5:]}-s{key[:8]}", "evidence": evidence, "claim": claim, "want": "",
           "hint": FORMS[form][0], "form": form, "domain": DOMAIN[line["lang"]], "group": line["group"],
           "groups": [line["group"]], "lang": line["lang"], "lic": LIC[line["lang"]], "split": "train",
           "src_ids": [line["id"]], "origin": ORIGIN, "pair": line["id"], "shape": line["shape"],
           "pos": draft.get("pos"), "fit": draft.get("fit")}
    for k in ("slot", "new", "seg_old", "seg_new", "how", "d"):
        if draft.get(k) is not None:
            row[k] = draft[k]
    return row


def make_rows(lines, n, seed, tok, scrubber, weights=None):
    """근거 줄들 → (쌍 줄, 어색한 줄, 셈). 근거 하나에 칸 갈래(b · a · c/d)마다 한 꼴씩, 몫이 가장 모자란 꼴부터."""
    rng = random.Random(seed)
    order = list(lines)
    rng.shuffle(order)
    quota = quotas(n, weights=weights)
    have = collections.Counter()
    pos_turn = collections.Counter()
    rows, awkward, count = [], [], collections.Counter()
    for line in order:
        if all(have[k] >= q for k, q in quota.items()):
            break
        lang = line["lang"]
        want = [f for f in FORMS if have[(f, lang)] < quota[(f, lang)]]
        if not want:
            continue
        pos_target = {f: ("front", "mid", "end")[pos_turn[f] % 3] for f in want}
        drafts = plan_line(line, random.Random(f"{seed}:{line['id']}"), tok, want, pos_target)
        for family in FAMILY.values():
            got = [f for f in family if f in drafts]
            got.sort(key=lambda f: have[(f, lang)] / max(1, quota[(f, lang)]))
            for form in got:
                draft = drafts[form]
                why = draft["awkward"] + seg_check(draft, form, line, scrubber)
                if why:
                    count[f"awkward_{form}"] += 1
                    if count[f"awkward_{form}/{lang}"] >= quota[(form, lang)]:   # 다듬기 재료는 꼴 몫만큼만
                        continue
                    count[f"awkward_{form}/{lang}"] += 1
                    awkward.append({"pick_id": line["id"], "form": form, "why": why, "lang": lang,
                                    "evidence": line["head"] + line["body"], "draft": draft.get("claim"),
                                    "slot": draft.get("slot"), "new": draft.get("new"), "seg_old": draft.get("seg_old"),
                                    "d": draft.get("d"),
                                    "instruction": sc.NEG_INSTRUCTION if form == "long-neg" else sc.INSTRUCTION})
                    continue
                rows.append(pair_row(line, form, draft))
                have[(form, lang)] += 1
                pos_turn[form] += 1
                break
    count["rows"] = len(rows)
    count["awkward"] = len(awkward)
    short = {f"{f}/{lang}": q - have[(f, lang)] for (f, lang), q in quota.items() if have[(f, lang)] < q}
    return rows, awkward, {"count": dict(count), "quota": {f"{f}/{l}": q for (f, l), q in quota.items()},
                           "by_form_lang": {f"{f}/{l}": have[(f, l)] for f, l in quota}, "short": short,
                           "by_pos": dict(collections.Counter(f"{r['form']}/{r['pos']}" for r in rows)),
                           "by_shape": dict(collections.Counter(f"{r['form']}/{r['shape']}" for r in rows)),
                           "by_fit": dict(collections.Counter(r["fit"] for r in rows))}


# ---------------------------------------------------------------- check

_HEAD_LABEL = re.compile(r"\[|\]|(?:절|줄 머리|칸 머리) :| · ")


def shingles(text, size=8):
    """8글자 조각 — 머리 틀 낱말(절 · 줄 머리 · 칸 머리)을 넘는 조각과 영문 낱말 하나뿐인 조각은 뺀다 (틀만 같아도 겹침으로 잡혔다)."""
    out = set()
    for piece in _HEAD_LABEL.split(text or ""):
        flat = re.sub(r"\s+", "", piece)
        out |= {flat[i:i + size] for i in range(len(flat) - size + 1)}
    return {x for x in out if not re.fullmatch(r"[A-Za-z]+", x)}


_GLUED = re.compile(r"[a-z0-9)][.!?][A-Z][a-z]")


def _plain(text):
    return re.sub(r"[,\s]", "", text or "")


def check_row(row, tok, banned, scrubber):
    """후보 한 줄 → 까닭 목록. 한 덩이(바뀐 꼴) · 쌍 320 토큰 안 · 합성/실제 셋과 8글자 겹침 0 · scrub."""
    why = []
    form = row.get("form")
    if form in ("long-num", "long-neg", "long-flip", "long-alt", "long-time"):
        if row.get("seg_old") not in row["evidence"] or row.get("seg_new") not in row["claim"]:
            why.append("seg_missing")
        draft = {"claim": row["claim"], "slot": row.get("slot") or {}, "new": row.get("new"),
                 "seg_old": row.get("seg_old") or "", "seg_new": row.get("seg_new") or "", "d": row.get("d")}
        why += seg_check(draft, form, row, scrubber)
    if form == "long-alt" and _plain(row.get("new")) == _plain((row.get("slot") or {}).get("text")):
        why.append("alt_trivial")                     # 「428,000」→「428000」 처럼 쉼표·띄어쓰기만 다르면 꼴 바꿈이 아니다
    if _GLUED.search(row["evidence"]):
        why.append("glued_sentence")                  # VitaminC 의 「10–0.New England」 처럼 문장이 붙은 근거
    if tok.pair(row["evidence"], row["claim"]) > MAX_LEN:
        why.append("tokens_over")
    if banned and (shingles(row["evidence"]) | shingles(row["claim"])) & banned:
        why.append("synth_overlap")
    if sum(scrubber.hits({"evidence": row["evidence"], "claim": row["claim"]}).values()):
        why.append("scrub")
    return why


def read_banned(paths):
    out = set()
    for path in paths:
        for row in lb.read_jsonl(path):
            for side in ("evidence", "claim", "a", "b"):
                if isinstance(row.get(side), str):
                    out |= shingles(row[side])
    return out


# ---------------------------------------------------------------- build (판정 뒤)

def build_rows(rows, n, seed):
    """판정 남은 줄(keep) → 꼴 × 말 몫대로. 근거(pair) 하나의 줄은 통째로 넣거나 뺀다.
    귀한 꼴이 든 묶음부터 고른다 — 섞은 차례대로 고르면 흔한 꼴이 몫을 먼저 채워 귀한 꼴 묶음이 막힌다."""
    quota = quotas(n)
    by_pair = collections.defaultdict(list)
    for r in rows:
        if r.get("keep") and r.get("want"):
            by_pair[r["pair"]].append(r)
    single = sorted(k for k, g in by_pair.items() if len({r["want"] for r in g}) < 2)   # 판정 칸 기준
    for k in single:                                  # 칸 하나뿐인 근거는 지름길(근거만 보고 칸을 맞힘)이라 뺀다
        del by_pair[k]
    keys = sorted(by_pair)
    random.Random(seed).shuffle(keys)
    order = {k: i for i, k in enumerate(keys)}       # 점수가 같으면 섞은 차례
    needs = {k: collections.Counter((r["form"], r["lang"]) for r in by_pair[k]) for k in keys}
    supply = collections.Counter()
    for need in needs.values():
        supply.update(need)
    have, out, left = collections.Counter(), [], set(keys)

    def drop(key):
        left.discard(key)
        supply.subtract(needs[key])

    def score(key):                                   # 귀한 꼴 = 남은 몫에 견줘 남은 공급이 적은 꼴
        return max((quota.get(k, 0) - have[k]) / max(1, supply[k]) for k in needs[key])

    while left:
        for key in [k for k in left if any(have[f] + v > quota.get(f, 0) for f, v in needs[k].items())]:
            drop(key)                                 # 몫이 찬 꼴이 든 묶음은 통째로 뺀다 (have 는 늘기만 한다)
        if not left:
            break
        key = max(left, key=lambda k: (score(k), -order[k]))
        drop(key)
        have.update(needs[key])
        out.extend(by_pair[key])
    return out, {"rows": len(out), "quota": {f"{f}/{l}": q for (f, l), q in quota.items()},
                 "have": {f"{f}/{l}": v for (f, l), v in sorted(have.items())}, "pairs": len({r["pair"] for r in out}),
                 "single_hint_pairs_dropped": len(single)}


# ---------------------------------------------------------------- 묶음 · 금지

def avoid_groups(paths):
    """평가 셋들의 묶음(group · groups) — 쌍 본문은 안 본다."""
    out = set()
    for path in paths:
        for row in lb.read_jsonl(path):
            out.update(g for g in [row.get("group")] + list(row.get("groups") or []) if g)
    return out


def _vitaminc_pages(raw, read_rows=None):
    """VitaminC 학습 쪽(train.parquet) real 판 → {쪽: [근거 문장]} — 판 고침 짝(case_id)마다 한 문장 · 거의 같은 문장은 하나만."""
    if read_rows is None:
        from parquet_lite import read_rows
    by_page = collections.defaultdict(dict)
    for row in read_rows(os.path.join(raw, "train.parquet")):
        if row.get("revision_type") == "real" and row.get("evidence"):
            by_page[row["page"]].setdefault(row["case_id"], row["evidence"])
    out = {}
    for page in sorted(by_page):
        kept, grams, heads = [], [], set()
        for case_id in sorted(by_page[page]):
            text = en_clean(by_page[page][case_id])
            g = mp.char3(text)
            head = " ".join(text.split()[:4])
            # 같은 문장의 다른 판(고치기 전 · 뒤)이 한 줄에 같이 들면 근거 안에서 서로 부딪친다 — 닮은 문장은 하나만
            if en_sentence_ok(text) and head not in heads and all(mp.jaccard(g, h) < 0.3 for h in grams):
                kept.append(text)
                grams.append(g)
                heads.add(head)
        if len(kept) >= 3:
            out[page] = kept
    return out


def en_clean(text):
    """VitaminC 토큰 띄어쓰기(「 , 」 「( 」 「`` ''」)를 글 꼴로."""
    text = " ".join(text.split()).replace("``", '"').replace("''", '"')
    text = re.sub(r" ([,.;:%)!?])", r"\1", text)
    return re.sub(r"\( ", "(", text)


def en_sentence_ok(text):
    """문장 꼴인가 — 30~400자 · 대문자 · 숫자로 시작 · 마침표로 끝 · 위키 틀 찌꺼기 없음."""
    return (30 <= len(text) <= 400 and (text[0].isupper() or text[0].isdigit()) and text.endswith(".")
            and not re.search(r"[{}|]|\[\[", text))


def en_lines(page, sents, rng):
    """VitaminC 한 쪽 → {join: [줄], para: [줄]}. 머리는 `[절 : <쪽>]`(conflict 꼴을 그대로)."""
    out = {"head": [], "join": [], "para": []}
    order = list(sents)
    rng.shuffle(order)
    for start in range(0, len(order), 6):
        chunk = order[start:start + 6]
        if len(chunk) >= 3:
            used = chunk[:rng.randint(3, len(chunk))]
            body = join_body(used, rng)
            if _en_body_ok(body):
                out["join"].append({"head": head_ctx(page.replace("_", " ")) if rng.random() < 0.5 else "",
                                    "body": body, "section": page, "other": [t for t in order if t not in used][:3]})
        if len(chunk) >= 2:
            used = chunk[:rng.randint(2, min(5, len(chunk)))]
            body = " ".join(used)
            if _en_body_ok(body):
                out["para"].append({"head": "", "body": body, "section": page,
                                    "other": [t for t in order if t not in used][:3]})
    return out


def sample_lines(pools, n_by_shape, tok, rng, group_cap=GROUP_CAP):
    """꼴별 풀 → 몫만큼 (묶음당 group_cap · 근거 토큰 EV_TOKENS 안 · 바꿀 자리 있음). 앞 꼴이 모자라면 다음 꼴 몫에 더한다."""
    per_group, out, short = collections.Counter(), [], {}
    carry = 0
    for shape in SHAPES:
        want = n_by_shape.get(shape, 0) + carry
        pool = list(pools.get(shape, []))
        rng.shuffle(pool)
        got = 0
        for line in pool:
            if got >= want:
                break
            if per_group[line["group"]] >= group_cap or not _has_target(line["body"], line["lang"]):
                continue
            ntok = tok.count(line["head"] + line["body"])
            if not EV_TOKENS[0] <= ntok <= EV_TOKENS[1]:
                continue
            per_group[line["group"]] += 1
            out.append(dict(line, shape=shape, ev_tokens=ntok))
            got += 1
        short[shape] = want - got
        carry = max(0, want - got) if shape != "para" else 0
    return out, short


# ---------------------------------------------------------------- 명령

def _guard(path, allow_git):
    if lb.inside_git(path) and not allow_git:
        sys.exit(f"{path} 가 git 안이다 — 쌍 본문은 git 밖에 둔다")


def _scrubber(judge_root):
    g1 = os.path.join(judge_root, "build", "g1")
    paths = [p for p in (os.path.join(g1, "names.txt"), os.path.join(g1, "game-names.txt")) if os.path.exists(p)]
    return scrub.Scrubber(scrub.read_names(paths), ())


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0


def cmd_pick(a):
    _guard(a.out, a.allow_git)
    tok = a.tok or Tok(a.tokenizer)
    avoid = avoid_groups(a.avoid_sets)
    rng = random.Random(a.seed)
    files = sorted(glob.glob(os.path.join(a.kowiki, "*.bz2")))
    pools = {"ko": {k: [] for k in SHAPES}, "en": {k: [] for k in SHAPES}}
    count = collections.Counter()
    docs = []
    for page in (a.pages if a.pages is not None else cached_pages(files, a.limit_pages, a.pages_cache)):
        group = f"kowiki:{page['id']}"
        count["ko_pages"] += 1
        if group in avoid:
            count["ko_avoid"] += 1
            continue
        if mw.eval_bucket(group) == a.eval_digit:
            count["ko_eval_pages"] += 1
            sections = parse_page(page["text"])
            if sum(len(p) for s in sections for p in s["paras"]) >= 300:
                docs.append({"doc_id": page["id"], "group": group, "title": page["title"], "sections": sections})
            continue
        lines = ko_lines(page, random.Random(f"{a.seed}:{page['id']}"))
        count["ko_train_pages"] += 1
        count["ko_table_docs"] += bool(lines.pop("_tables"))      # 읽을 수 있는 표(칸 합치기 없음)가 든 문서
        count["ko_head_docs"] += bool(lines["head"])              # 그중 head 꼴 줄이 나온 문서
        for shape, items in lines.items():
            for i, line in enumerate(items):
                pools["ko"][shape].append(dict(line, group=group, title=page["title"], lang="ko",
                                               src=f"kowiki:{page['id']}:{shape}:{i}"))
    vit = a.vit_pages if a.vit_pages is not None else _vitaminc_pages(a.vitaminc)
    for page, sents in vit.items():
        group = f"vitc:{page}"
        count["en_pages"] += 1
        if group in avoid:
            count["en_avoid"] += 1
            continue
        for shape, items in en_lines(page, sents, random.Random(f"{a.seed}:{page}")).items():
            for i, line in enumerate(items):
                pools["en"][shape].append(dict(line, group=group, title=page, lang="en", src=f"vitc:{page}:{shape}:{i}"))
    picked, short = [], {}
    for lang, n in (("ko", a.n_ko), ("en", a.n_en)):
        shapes = SHAPES if lang == "ko" else ("join", "para")
        base = {s: n // len(shapes) + (1 if i < n % len(shapes) else 0) for i, s in enumerate(shapes)}
        got, short[lang] = sample_lines(pools[lang], base, tok, rng)
        picked += got
    for i, line in enumerate(picked):
        line["id"] = f"{a.id_prefix}{line['lang']}{i + 1:04d}"      # 2차 pick 은 머리를 달아 근거 열쇠(pair)가 1차와 안 겹친다
    lb.write_jsonl(os.path.join(a.out, "pick.jsonl"), picked)
    by = collections.Counter(f"{r['lang']}/{r['shape']}" for r in picked)
    report = {"count": dict(count), "pool": {f"{lang}/{s}": len(v) for lang, d in pools.items() for s, v in d.items()},
              "picked": dict(sorted(by.items())), "short_before_fill": short, "groups": len({r["group"] for r in picked}),
              "avoid_groups": len(avoid), "ev_tokens_p50": _pct([r["ev_tokens"] for r in picked], .5),
              "ev_tokens_p90": _pct([r["ev_tokens"] for r in picked], .9), "eval_digit": a.eval_digit, "seed": a.seed}
    if a.docs_out:
        _guard(a.docs_out, a.allow_git)
        lb.write_jsonl(a.docs_out, docs)
        report["ko_docs"] = len(docs)
        report["ko_docs_with_table"] = sum(1 for d in docs if any(s["tables"] for s in d["sections"]))
    lb.write_json(os.path.join(a.out, "pick.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_make(a):
    _guard(a.out, a.allow_git)
    tok = a.tok or Tok(a.tokenizer)
    lines = lb.read_jsonl(a.inp)
    rows, awkward, report = make_rows(lines, a.n, a.seed, tok, _scrubber(a.judge_root), parse_weights(a.form_weights))
    lb.write_jsonl(os.path.join(a.out, "out.jsonl"), rows)
    lb.write_jsonl(os.path.join(a.out, "awkward.jsonl"), awkward)
    report.update(lines=len(lines), n=a.n, seed=a.seed)
    lb.write_json(os.path.join(a.out, "make.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_check(a):
    tok = a.tok or Tok(a.tokenizer)
    banned = read_banned([p for p in (a.synth, a.real) if p])
    scrubber = _scrubber(a.judge_root)
    rows = lb.read_jsonl(a.inp)
    passed, count = [], collections.Counter()
    by = collections.defaultdict(collections.Counter)      # 꼴/말 → {rows, pass, 까닭...}
    for row in rows:
        why = check_row(row, tok, banned, scrubber)
        key = f"{row.get('form')}/{row.get('lang')}"
        count["rows"] += 1
        by[key]["rows"] += 1
        for w in why:
            count[w] += 1
            by[key][w] += 1
        if not why:
            passed.append(row)
            by[key]["pass"] += 1
    count["pass"] = len(passed)
    report = {"count": dict(count), "by_form_lang": {k: dict(v) for k, v in sorted(by.items())}, "banned_shingles": len(banned)}
    if a.out:
        _guard(a.out, a.allow_git)
        lb.write_jsonl(a.out, passed)
        lb.write_json(os.path.splitext(a.out)[0] + ".json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


B_FORMS = ("long-num", "long-neg", "long-flip")


def fixed_row(fix, line, tok):
    """다듬은 어색한 줄 하나 + 그 pick 줄 → (쌍 줄, 까닭 목록). 다른 덩이가 정확히 하나인지만 보고 slot 범위는 안 본다."""
    claim, head, body = fix.get("claim"), line["head"], line["body"]
    if not claim:
        return None, ["empty"]
    if "\ufffd" in line["head"] + body + claim:
        return None, ["broken_char"]
    if not claim.startswith(head):
        return None, ["head_changed"]
    new_body = claim[len(head):]
    if sc.chunks(body, new_body) != 1:
        return None, ["chunks"]
    at = next((i for i, (x, y) in enumerate(zip(body, new_body)) if x != y), min(len(body), len(new_body)))
    spans, new_spans = segments(body), segments(new_body)
    seg, new_seg = _seg_of(spans, at), _seg_of(new_spans, at)
    seg_old, seg_new = body[spans[seg][0]:spans[seg][1]], new_body[new_spans[new_seg][0]:new_spans[new_seg][1]]
    if fix["form"] in B_FORMS and sc.is_question(seg_new):
        return None, ["question_b"]                    # 의문문은 b 라벨이 서지 않는다 (메인 결정)
    fitted, fit = _fit_claim(tok, head + body, head, new_body, new_spans, new_seg)
    if fitted is None:
        return None, ["window"]
    if seg_new not in fitted:
        return None, ["window"]                        # 바뀐 토막이 창 밖
    draft = {"claim": fitted, "fit": fit, "pos": position(body, at, at + 1), "slot": fix.get("slot"),
             "new": fix.get("new"), "seg_old": seg_old, "seg_new": seg_new, "how": "fixed", "d": fix.get("d")}
    return pair_row(line, fix["form"], draft), []


# 다듬은 줄은 낱말 수 · 물음꼴이 바뀌어도 되지만, 칸을 세우는 부정어 차이 · 새 값 · 때 값은 남아야 한다
FIXED_SEG_WHY = ("negation", "new_missing", "time_missing")


def cmd_fixed(a):
    """awkward.fixed → 검사 통과 쌍 줄. 바뀐 덩이 하나 · 320 토큰 · 겹침 · scrub · 붙은 문장 · 쉼표만 바꾼 alt ·
    원래 꼴 seg_check 의 부정어 차이 1 · 새 값 · 때 값까지 본다."""
    tok = a.tok or Tok(a.tokenizer)
    banned = read_banned([p for p in (a.synth, a.real) if p])
    scrubber = _scrubber(a.judge_root)
    lines = {r["id"]: r for r in lb.read_jsonl(a.pick)}
    passed, by = [], collections.defaultdict(collections.Counter)
    for fix in lb.read_jsonl(a.inp):
        key = f"{fix['form']}/{fix['lang']}"
        by[key]["rows"] += 1
        line = lines.get(fix["pick_id"])
        if line is None:
            by[key]["no_pick"] += 1
            continue
        row, why = fixed_row(fix, line, tok)
        if row is not None:
            why = check_row(dict(row, form="fixed"), tok, banned, scrubber)
            if fix["form"] == "long-alt" and _plain(row.get("new")) == _plain((row.get("slot") or {}).get("text")):
                why.append("alt_trivial")
            seg = seg_check(dict(row, slot=row.get("slot") or {}, new=row.get("new")), fix["form"], line, scrubber)
            why += [w for w in seg if w in FIXED_SEG_WHY]
        for w in why:
            by[key][w] += 1
        if not why:
            passed.append(row)
            by[key]["pass"] += 1
    _guard(a.out, a.allow_git)
    lb.write_jsonl(a.out, passed)
    report = {"pass": len(passed), "by_form_lang": {k: dict(v) for k, v in sorted(by.items())}}
    lb.write_json(os.path.splitext(a.out)[0] + ".json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


PAIR_FIELDS = ("id", "evidence", "claim", "want", "hint", "form", "domain", "group", "groups", "lang", "lic", "split",
               "src_ids", "origin", "pair", "shape", "pos", "fit")


def to_pairs(rows, drop_num_date=False):
    """check 통과 줄 → seed_to_pairs pairs 꼴 쌍 줄 (make 안쪽 칸은 뺌) + 근거(pair)별 칸 묶음 셈."""
    hints_of = collections.defaultdict(set)
    for r in rows:
        hints_of[sha_key(r)].add(r["hint"])
    clash = {k for k, v in hints_of.items() if len(v) > 1}   # 같은 근거·주장에 다른 칸 (num 이 때 자리를 바꾼 줄 = time 줄)
    out, num_date = [], 0
    for r in rows:
        if sha_key(r) in clash:
            continue
        if drop_num_date and r.get("form") == "long-num" and (r.get("slot") or {}).get("kind") == "date":
            num_date += 1                             # 옛 판 make 가 낸 「때만 바꾼 b」 (v6 에선 d)
            continue
        row = {k: r[k] for k in PAIR_FIELDS if k in r}
        row["want"] = ""
        row["slot_kind"] = (r.get("slot") or {}).get("kind")
        out.append(row)
    hints = collections.defaultdict(set)
    for r in out:
        hints[r["pair"]].add(r["hint"])
    report = {"rows": len(out), "clash_rows_dropped": len(rows) - len(out) - num_date, "num_date_dropped": num_date, "by_hint": dict(collections.Counter(r["hint"] for r in out)),
              "pairs": len(hints), "pairs_with_2plus_hints": sum(1 for v in hints.values() if len(v) >= 2),
              "pairs_by_hints": dict(collections.Counter("".join(sorted(v)) for v in hints.values()))}
    return out, report


def cmd_pairs(a):
    _guard(a.out, a.allow_git)
    out, report = to_pairs(lb.read_jsonl(a.inp), a.drop_num_date)
    lb.write_jsonl(a.out, out)
    lb.write_json(os.path.splitext(a.out)[0] + ".json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def merge_judged(rows):
    """여러 판정 폴더 줄 → (근거, 주장) sha 하나에 한 줄. 같은 sha 에 판정 칸이 갈리면 그 sha 줄은 다 뺀다. (줄, 셈)."""
    by_sha = collections.defaultdict(list)
    for r in rows:
        if r.get("keep") and r.get("want"):
            by_sha[sha_key(r)].append(r)
    agree = [g for g in by_sha.values() if len({r["want"] for r in g}) == 1]
    split = sum(len(g) for g in by_sha.values()) - sum(len(g) for g in agree)
    return [g[0] for g in agree], {"dup_rows_dropped": sum(len(g) - 1 for g in agree), "split_want_rows_dropped": split}


def cmd_build(a):
    _guard(a.out, a.allow_git)
    rows = []
    for d in a.dirs:
        rows += lb.read_jsonl(os.path.join(d, "judged.jsonl"))
    rows, merged = merge_judged(rows)
    out, report = build_rows(rows, a.n, a.seed)
    report.update(merged)
    lb.write_jsonl(os.path.join(a.out, "cands.jsonl"), out)
    lb.write_json(os.path.join(a.out, "summary.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cnli_rows(docs, avoid, tok):
    """ContractNLI 문서들 → 목표 셋 en 후보 (본래 라벨 쌍 · 근거 = 걸린 조항 글, 320 토큰 안). (줄, 셈)."""
    hint = {"Entailment": "a", "Contradiction": "b", "NotMentioned": "d"}
    out, count, lens = [], collections.Counter(), collections.defaultdict(list)
    for split, doc, spans, labels in docs:
        group = f"cnli:{split}-{doc['id']}"
        if group in avoid:
            count["avoid"] += 1
            continue
        index = None
        for key, ann in sorted(doc["annotation_sets"][0]["annotations"].items()):
            claim = labels[key]["hypothesis"]
            picked = ann["spans"]
            if not picked:
                index = index or mp.BM25(spans)
                picked = index.rank(claim)[:1]
            evidence = " ".join(spans[i] for i in sorted(picked))
            choice = ann["choice"]
            count[f"all_{choice}"] += 1
            ntok = tok.count(evidence)
            lens[choice].append(ntok)
            if tok.pair(evidence, claim) > MAX_LEN:
                count[f"over_{choice}"] += 1
                continue
            row = {"evidence": evidence, "claim": claim, "want": "", "hint": hint[choice], "form": "cnli-" + choice,
                   "domain": "contractnli", "group": group, "groups": [group], "lang": "en", "lic": "CC-BY-4.0",
                   "split": split, "src_ids": [f"cnli-{split}-{doc['id']}-{key}"], "source_label": choice,
                   "ev_tokens": ntok, "origin": "g10-dev-long"}
            row["id"] = f"g10d-cnli-{split}-{doc['id']}-{key}-s{sha_key(row)[:8]}"
            out.append(row)
            count[f"kept_{choice}"] += 1
    dist = {c: {"n": len(v), "p50": _pct(v, .5), "p90": _pct(v, .9), "max": max(v) if v else 0} for c, v in lens.items()}
    return out, {"count": dict(count), "ev_tokens": dist}


def cmd_cnli(a):
    _guard(a.out, a.allow_git)
    tok = a.tok or Tok(a.tokenizer)
    rows, report = cnli_rows(mp.read_contractnli(a.raw, ("train", "dev", "test")), avoid_groups(a.avoid_sets), tok)
    lb.write_jsonl(a.out, rows)
    lb.write_json(os.path.splitext(a.out)[0] + ".summary.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main(argv=None, tok=None, pages=None, vit_pages=None):
    """tok · pages · vit_pages 는 시험용 주입 (토크나이저 · 위키 문서 · VitaminC 쪽)."""
    parser = argparse.ArgumentParser(description="K8 긴 줄 층 후보 · 목표 셋 재료")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pick", "make", "check", "fixed", "pairs", "build", "cnli"):
        p = sub.add_parser(name)
        p.add_argument("--tokenizer", default=default_tokenizer())
        p.add_argument("--judge-root", default=os.path.join(os.path.expanduser("~"), ".judgemodel"))
        p.add_argument("--allow-git", action="store_true", help="시험용 — git 안에 써도 된다")
    p = sub.choices["pick"]
    p.add_argument("--kowiki", required=True)
    p.add_argument("--vitaminc", required=True)
    p.add_argument("--avoid-sets", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--docs-out", help="목표 셋 ko 재료 — 평가 몫 문서 풀 (ko-docs.jsonl)")
    p.add_argument("--n-ko", type=int, default=1100)
    p.add_argument("--n-en", type=int, default=600)
    p.add_argument("--id-prefix", default="", help="2차 pick 줄 id 머리 (예 L2-)")
    p.add_argument("--eval-digit", type=int, default=0)
    p.add_argument("--limit-pages", type=int, default=0, help="덤프 파일마다 문서 몇 개에서 멈추나 (0 = 끝까지)")
    p.add_argument("--pages-cache", help="마지막 판 모음 jsonl — 있으면 덤프 대신 읽고, 없으면 덤프를 읽으며 쓴다")
    p.add_argument("--seed", type=int, default=SEED_PICK)
    p = sub.choices["make"]
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--n", type=int, default=1800)
    p.add_argument("--seed", type=int, default=20262201)
    p.add_argument("--out", required=True)
    p.add_argument("--form-weights", default="", help="꼴 몫 배수 — 예 long-alt/ko=2,long-other=0.6")
    p = sub.choices["check"]
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--synth")
    p.add_argument("--real")
    p.add_argument("--out")
    p = sub.choices["fixed"]
    p.add_argument("--in", dest="inp", required=True, help="awkward.fixed.jsonl")
    p.add_argument("--pick", required=True)
    p.add_argument("--synth")
    p.add_argument("--real")
    p.add_argument("--out", required=True)
    p = sub.choices["pairs"]
    p.add_argument("--drop-num-date", action="store_true", help="long-num 이 때 자리를 바꾼 줄을 뺀다")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p = sub.choices["build"]
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=20262204)
    p = sub.choices["cnli"]
    p.add_argument("--raw", required=True)
    p.add_argument("--avoid-sets", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = parser.parse_args(argv)
    a.tok, a.pages, a.vit_pages = tok, pages, vit_pages
    return {"pick": cmd_pick, "make": cmd_make, "check": cmd_check, "fixed": cmd_fixed, "pairs": cmd_pairs, "build": cmd_build, "cnli": cmd_cnli}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
