"""mine_wiki — 한국어 위키백과 개정 이력 덤프(.bz2)에서 「한 문장 · 한 자리만 바뀐」 b · d 후보를 뽑는다.

비공개 설계 3-1. 표준 라이브러리만 쓴다(8-2 ⑧).
덤프는 디스크에 풀지 않고 bz2 + iterparse 로 흘려 읽는다. 줄 꼴은 mine_archive revision 과 같다.

    python -I mine_wiki.py pairs --dump <history.xml-….bz2> --out <chunk.jsonl> [--limit-pages N]
    python -I mine_wiki.py split --in <chunk.jsonl> --out-eval <eval.jsonl> --out-train <train.jsonl> [--eval-digit 0]
    python -I mine_wiki.py stats --in <chunk.jsonl>
    python -I mine_wiki.py sample --in <chunk.jsonl> --out <sample.jsonl> --n 200 [--seed 20261013]
    python -I mine_wiki.py filter --in <chunk.jsonl> --out <chunk-f.jsonl> [--kinds num,date,ordinal] [--flag-cap 0.2]
        [--no-date-d-to-b] [--no-drop-meta] [--no-drop-meta-tail]
    python -I mine_wiki.py merge --in <c1-f.jsonl> <c2-f.jsonl> … --out <all-f.jsonl> --out-eval <eval> --out-train <train>

거르기(sets-g3.md 2 · 3절)는 pairs 에선 옵션으로만 켜고(옛 결과 재현), filter 는 권장값이 기본이다.
"""
import argparse
import bz2
import collections
import difflib
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_sets  # noqa: E402

SEED = 20261013
LIC = "CC-BY-SA-4.0"
DOMAIN = "kowiki"
MIN_LEN = 15                  # 두 문장 다 이 글자 수 이상 (mine_archive REV_MIN_LEN 과 같다)
MAX_LEN = 400                 # 근거 · 주장 상한 (mine_public MAX_LEN 과 같다)
MIN_RATIO = 0.6               # 두 문장 글자 닮음 하한 (mine_archive REV_MIN_RATIO 와 같다)
MAX_MID_CHARS = 8000          # 바뀐 줄들이 이보다 길면 문단이 아니라 표 · 틀 덩이로 보고 건너뜀
MAX_MID_LINES = 8             # 앞뒤 같은 줄을 뗀 가운데가 이보다 많으면 큰 고침으로 보고 건너뜀
MAX_CHUNK_TOKENS = 4          # 바뀐 덩이 한쪽 토큰 수 상한 — 넘으면 「한 자리」가 아니라 고쳐 쓴 것
LOG_EVERY = 1000              # 문서 몇 개마다 진행 로그를 찍나
KEEP_KINDS = ("num", "date", "ordinal")   # 표본 판정 권장 (sets-g3.md 2-4 · 3절) — name · qual · onoff 는 b 비율이 낮아 뺀다
FLAG_CAP = 0.20               # flag 붙은 b 는 남은 b 의 이 몫까지 (설계 3-1 「되돌림 b ≤ 위키 b 의 20%」)

# ---------------------------------------------------------------- 위키 문법 벗기기

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF_PAIR = re.compile(r"<ref\b[^>/]*>.*?</ref\s*>", re.I | re.S)
_REF_SELF = re.compile(r"<ref\b[^>]*/>", re.I)
_DROP_TAG = re.compile(r"<(math|gallery|timeline|score|syntaxhighlight|source|pre|code|nowiki)\b[^>]*>.*?</\1\s*>",
                       re.I | re.S)
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE_PARAM = re.compile(r"\{\|[^{}]*\|\}")
_FILE_LINK = re.compile(r"\[\[(?:파일|File|그림|Image|분류|Category|틀|Template):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
                        re.I)
_LANG_LINK = re.compile(r"\[\[[a-z\-]{2,12}:[^\[\]]*\]\]")
_PIPE_LINK = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_EXT_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_BOLD = re.compile(r"'{2,}")
_EMPTY_PAREN = re.compile(r"\(\s*[,;·]?\s*\)")
_SPACE = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?。])\s+")
_SKIP_LINE = ("{|", "|", "!", "*", "#", ":", ";", "=", "{{", "}}", "[[파일:", "[[File:", "[[분류:", "__")


def strip_wiki(line):
    """문단 한 줄의 위키 문법을 벗긴다. 틀 · 각주 · 주석 · 표 · 꾸밈은 지우고 링크는 보이는 글만 남긴다."""
    text = _COMMENT.sub("", line)
    text = _REF_PAIR.sub("", text)
    text = _REF_SELF.sub("", text)
    text = _DROP_TAG.sub("", text)
    for _ in range(6):                       # 틀은 안쪽부터 벗긴다 (겹친 틀)
        text, n = _TEMPLATE.subn("", text)
        if not n:
            break
    text = _TABLE_PARAM.sub("", text)
    text = _FILE_LINK.sub("", text)
    text = _LANG_LINK.sub("", text)
    text = _PIPE_LINK.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _BOLD.sub("", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = _EMPTY_PAREN.sub("", text)
    return _SPACE.sub(" ", text).strip()


def prose_ok(raw_line, text):
    """벗긴 줄이 문단 글인가 — 표 · 목록 · 제목 · 틀 줄이나 벗기다 남은 괄호 조각은 버린다."""
    head = raw_line.lstrip()
    if not text or head.startswith(_SKIP_LINE):
        return False
    return not any(mark in text for mark in ("{{", "}}", "[[", "]]", "{|", "|}"))


def sentences(text):
    return [s for s in _SENT.split(text) if s]


# ---------------------------------------------------------------- 바뀐 덩이 가르기

_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[A-Za-z]+|[가-힣]+|[㐀-鿿豈-﫿]+|\S")
_DIGIT = re.compile(r"[0-9]")
_HANJA = re.compile(r"^[㐀-鿿豈-﫿]+$")
_HANJA_PAREN = re.compile(r"\(\s*[㐀-鿿豈-﫿][㐀-鿿豈-﫿\s,·]*\)")
_NOT_LETTER = re.compile(r"[\W_]+")
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
_DATE_GAP = {"년", "월", "일", "시", "분", ".", "-", "/", "~", "–", ",", " "}
_DATE_UNIT = re.compile(r"^(년|월|일|년도|년대|세기|시|분)")
_ORDINAL_NEXT = re.compile(r"^(번째|째|위|등|차|대|기|호|세대|편|화|라운드|분기|학년|군단|사단|연대)")
_ORDINAL_WORD = re.compile(r"^(첫째|둘째|셋째|넷째|다섯째|첫$)")
_NUM_WORD = {"한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열", "스무", "서른", "마흔", "쉰", "백", "천",
             "만", "억", "조", "하나", "둘", "셋", "넷", "수십", "수백", "수천", "수만"}
_QUAL = ("이상", "이하", "초과", "미만", "넘는", "넘게", "넘어", "넘은", "앞", "뒤", "이전", "이후", "직전", "직후", "처음",
         "마지막", "최초", "최후", "최대", "최소", "최고", "최저", "가장", "유일", "약", "거의", "모두", "일부", "전부",
         "대부분", "전", "후")
_HANJA_WORD = {"前": "전", "後": "후", "現": "현", "元": "원", "新": "신", "舊": "구"}
_NEGATION = ("않", "안", "못", "아니", "없", "비", "불", "무", "미")
_NEG_RE = re.compile(r"않|못하|못했|아니|없")      # 붙여 쓴 글에서 세어 띄어쓰기만 바뀐 고침을 부정 바뀜으로 안 본다
_ONOFF = (("있", "없"), ("가능", "불가능"), ("허용", "금지"), ("찬성", "반대"), ("승리", "패배"), ("승", "패"),
          ("남성", "여성"), ("남자", "여자"), ("생존", "사망"), ("공개", "비공개"), ("포함", "제외"), ("상승", "하락"),
          ("증가", "감소"), ("흑자", "적자"), ("성공", "실패"), ("합격", "불합격"), ("유료", "무료"), ("공립", "사립"),
          ("국립", "사립"), ("활성", "비활성"), ("현역", "은퇴"), ("동", "서"), ("남", "북"), ("상", "하"), ("좌", "우"),
          ("우승", "준우승"), ("당선", "낙선"), ("진출", "탈락"), ("개통", "폐지"), ("존속", "폐지"), ("운행", "운휴"),
          ("단선", "복선"), ("여당", "야당"), ("진보", "보수"), ("직접", "간접"), ("정규", "비정규"), ("일반계", "전문계"))
_STYLE_WORD = {"매우", "아주", "주로", "특히", "또한", "그리고", "그러나", "하지만", "및", "등", "또는", "같은", "다른",
               "여러", "많은", "그", "이", "저", "것", "수", "때", "중", "곳", "데", "현재", "당시", "이후", "한편",
               "그녀", "그들", "이들", "자신", "본인", "해당", "이곳", "그곳"}
_VERB_END = ("한", "된", "는", "던", "고", "며", "서", "게", "지", "했", "됐", "었", "았", "였", "해", "돼", "할", "될", "던")
_VERBISH = ("하다", "했다", "한다", "된다", "되었", "하였", "했", "된", "하는", "되는", "하고", "되고", "하며", "되며",
            "하여", "되어", "해서", "돼", "하게", "스럽", "스러운", "적인", "적으로", "롭게", "로운", "였다", "었다",
            "았다", "이다", "있다", "없다", "한다고", "했으며", "하였다")
_JOSA = ("에서는", "에서", "으로는", "으로", "로는", "에게", "까지", "부터", "이라", "라는", "이며", "이고", "은", "는",
         "이", "가", "을", "를", "의", "에", "로", "도", "와", "과", "만")


def _trivial(tokens):
    """문장부호 · 한자 병기 · 숫자 쉼표만 바뀐 덩이면 True (띄어쓰기는 토큰에 안 잡혀 이미 빠진다)."""
    return all(not re.search(r"[0-9A-Za-z가-힣]", t) or _HANJA.match(t) for t in tokens)


def _qual_in(qual, token):
    """한정어가 토큰에 드나 — 두 글자 넘는 한정어는 토큰 머리에서, 한 글자는 조사를 뗀 토큰과 같을 때만."""
    if len(qual) >= 2:
        return _strip_josa(token).startswith(qual)
    return _strip_josa(token) == qual or token == qual


def _strip_josa(word):
    for josa in _JOSA:
        if len(word) > len(josa) + 1 and word.endswith(josa):
            return word[:-len(josa)]
    return word


def _core(old, new):
    """두 글에서 앞뒤 같은 글자를 떼고 남은 가운데 (바뀐 핵심)."""
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    j = 0
    while j < min(len(old), len(new)) - i and old[-1 - j] == new[-1 - j]:
        j += 1
    return old[i:len(old) - j], new[i:len(new) - j]


def chunks_of(old, new):
    """두 문장 → 바뀐 덩이 [(옛 토큰, 새 토큰, 앞 토큰, 뒤 토큰, 서술어 덩이인가)]. 날짜 단위만 사이에 둔 덩이는 잇는다.

    한자 병기 괄호는 먼저 지우고, 띄어쓰기 · 문장부호만 다르면 덩이 없음(빈 목록)으로 낸다.
    """
    old, new = _HANJA_PAREN.sub("", old), _HANJA_PAREN.sub("", new)
    for hanja, hangul in _HANJA_WORD.items():
        old, new = old.replace(hanja, hangul), new.replace(hanja, hangul)
    if _NOT_LETTER.sub("", old).lower() == _NOT_LETTER.sub("", new).lower():
        return []
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    words = [k for k, t in enumerate(a) if re.search(r"[0-9A-Za-z가-힣]", t)]
    last_word = words[-1] if words else -1
    a_cmp = [t.replace(",", "") if _DIGIT.match(t) else t.lower() for t in a]
    b_cmp = [t.replace(",", "") if _DIGIT.match(t) else t.lower() for t in b]
    ops = [op for op in difflib.SequenceMatcher(None, a_cmp, b_cmp, autojunk=False).get_opcodes() if op[0] != "equal"]
    ops = [op for op in ops if not _trivial(a[op[1]:op[2]] + b[op[3]:op[4]])]
    merged = []
    for op in ops:
        if merged and all(t in _DATE_GAP for t in a[merged[-1][2]:op[1]]) and op[1] - merged[-1][2] <= 2:
            last = merged[-1]
            merged[-1] = ("replace", last[1], op[2], last[3], op[4])
            continue
        merged.append(op)
    out = []
    for _, i1, i2, j1, j2 in merged:
        prev = a[i1 - 1] if i1 > 0 else ""
        nxt = a[i2] if i2 < len(a) else ""
        out.append((a[i1:i2], b[j1:j2], prev, nxt, i1 <= last_word < i2 or (i1 == i2 and i1 > last_word)))
    return out


def _is_year(tokens, nxt):
    return len(tokens) == 1 and bool(_YEAR.match(tokens[0])) and nxt.startswith("년")


def classify(chunk):
    """바뀐 덩이 하나의 종류 — num · date · ordinal · qual · onoff · name. 해당 없으면 None (말투 · 표기 · 덧붙임).

    한쪽이 빈 덩이(덧붙임 · 지움)는 정보를 더하거나 뺀 것이라 a 쪽이다. 한정어 · 부정어 한두 토큰일 때만 남긴다.
    """
    old_t, new_t, prev, nxt, at_end = chunk
    if max(len(old_t), len(new_t)) > MAX_CHUNK_TOKENS:
        return None
    old_s, new_s = "".join(old_t), "".join(new_t)
    old_d, new_d = _DIGIT.findall(old_s), _DIGIT.findall(new_s)
    quals_o = {q for q in _QUAL if any(_qual_in(q, t) for t in old_t)}
    quals_n = {q for q in _QUAL if any(_qual_in(q, t) for t in new_t)}
    neg_o = len(_NEG_RE.findall(old_s)) + sum(1 for t in old_t if t in ("안", "못"))
    neg_n = len(_NEG_RE.findall(new_s)) + sum(1 for t in new_t if t in ("안", "못"))
    if not old_t or not new_t:
        if len(old_t + new_t) > 2:
            return None
        if quals_o != quals_n:
            return "qual"
        return "onoff" if neg_o != neg_n else None
    if old_d and new_d and old_d != new_d:
        if _DATE_UNIT.match(nxt) or any(_DATE_UNIT.match(t) for t in old_t[1:] + new_t[1:]):
            return "date"
        if _ORDINAL_NEXT.match(nxt) or prev.endswith("제"):
            return "ordinal"
        return "num"
    if old_d or new_d:
        return None                          # 한쪽에만 수 — 「그 해」 → 「2009년」 같은 풀어 쓰기
    core_o, core_n = _core(old_s, new_s)
    if (set(old_t) & _NUM_WORD) != (set(new_t) & _NUM_WORD):
        return "num"
    if _ORDINAL_WORD.match(core_o or "-") or _ORDINAL_WORD.match(core_n or "-"):
        return "ordinal"
    if quals_o != quals_n:
        return "qual"
    if neg_o != neg_n:
        return "onoff"
    for left, right in _ONOFF:
        if len(left) == 1 and (len(core_o) > 2 or len(core_n) > 2):
            continue                         # 한 글자 짝(동/서 · 있/없)은 짧은 핵심에서만 본다
        for a, b in ((left, right), (right, left)):
            if a in old_s and a not in new_s and b in new_s and b not in old_s:
                return "onoff"
    if _name_like(old_t, new_t, core_o, core_n, at_end):
        return "name"
    return None


def _name_like(old_t, new_t, core_o, core_n, at_end):
    """고유명사 · 이름 자리 바뀜 어림 — 낱말 1~3개가 통째로 바뀌고, 서술어 · 꾸밈말 · 조사 고침이 아닐 때."""
    if at_end or not old_t or not new_t or len(old_t) > 3 or len(new_t) > 3:
        return False
    if len(core_o) < 2 or len(core_n) < 2:
        return False
    stems = [_strip_josa(t) for t in old_t + new_t]
    if set(stems) & _STYLE_WORD or any(len(t) >= 2 and t.endswith(_VERB_END) for t in stems):
        return False
    if sorted(core_o) == sorted(core_n):
        return False                         # 글자 자리만 바뀐 오타 고침
    latin_o, latin_n = bool(re.search(r"[A-Za-z]", core_o)), bool(re.search(r"[A-Za-z]", core_n))
    if latin_o != latin_n:
        return False                         # 로마자 ↔ 한글 바꿔 적기 (m.net → 엠넷)
    return not any(c.endswith(_VERBISH) or c.startswith(_VERBISH) for c in (core_o, core_n))


def judge_pair(old, new):
    """한 문장 쌍 → (hint, kind) 또는 (None, 버린 까닭). 덩이 하나 → b, 연도 덩이 + 값 덩이 하나 → d."""
    chunks = chunks_of(old, new)
    if not chunks:
        return None, "style_only"
    if len(chunks) == 1:
        kind = classify(chunks[0])
        if kind is None:
            return None, "other_word"
        return "b", kind
    if len(chunks) == 2:
        years = [c for c in chunks if _is_year(c[0], c[3]) and _is_year(c[1], c[3])]
        rest = [c for c in chunks if c not in years]
        if len(years) == 1 and len(rest) == 1:
            kind = classify(rest[0])
            if kind is not None:
                return "d", kind
    return None, "multi_chunk"


# ---------------------------------------------------------------- 덤프 흘려 읽기

_REVERT_COMMENT = re.compile(r"되돌|\brevert|\brv\b|\bundo|편집 취소|복구", re.I)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def iter_pages(stream, limit_pages=0):
    """덤프 → ("rev", 문서, 판) 을 판마다, ("end", 문서, None) 을 문서 끝마다 낸다.

    판 본문을 문서 하나 동안 다 쥐면 긴 문서가 메모리를 먹어서, 판을 하나씩 흘려보낸다.
    """
    context = ET.iterparse(stream, events=("start", "end"))
    root = None
    page = None
    count = 0
    for event, elem in context:
        tag = _local(elem.tag)
        if event == "start":
            if root is None:
                root = elem
            if tag == "page":
                page = {"id": "", "title": "", "ns": "", "redirect": False}
            continue
        if page is None:
            continue
        if tag == "revision":
            yield "rev", page, _read_revision(elem)
            elem.clear()
        elif tag == "title" and not page["title"]:
            page["title"] = elem.text or ""
        elif tag == "ns":
            page["ns"] = elem.text or ""
        elif tag == "id" and not page["id"]:          # 문서 id 는 첫 판보다 앞에 나온다
            page["id"] = elem.text or ""
        elif tag == "redirect":
            page["redirect"] = True
        elif tag == "page":
            yield "end", page, None
            page = None
            root.clear()
            count += 1
            if limit_pages and count >= limit_pages:
                return


def _read_revision(elem):
    rev = {"id": "", "user": "", "anon": False, "text": None, "sha1": "", "comment": "", "ts": ""}
    for child in elem:
        tag = _local(child.tag)
        if tag == "id":
            rev["id"] = child.text or ""
        elif tag == "timestamp":
            rev["ts"] = child.text or ""
        elif tag == "comment":
            rev["comment"] = child.text or ""
        elif tag == "sha1":
            rev["sha1"] = child.text or ""
        elif tag == "text":
            rev["text"] = child.text if child.get("deleted") is None else None
        elif tag == "contributor":
            for part in child:
                if _local(part.tag) == "username":
                    rev["user"] = part.text or ""
                elif _local(part.tag) == "ip":
                    rev["user"], rev["anon"] = part.text or "", True
    return rev


def _middle_lines(old_text, new_text):
    """두 판 본문에서 앞뒤 같은 줄을 떼고 남은 가운데 줄들 (옛, 새)."""
    a, b = old_text.split("\n"), new_text.split("\n")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return a[i:len(a) - j], b[i:len(b) - j]


def _prose_sentences(lines):
    """가운데 줄들 → 문단 글 줄만 벗겨 문장 목록으로. 표 · 목록 · 틀 줄은 빠진다(그 줄만 바뀐 고침은 문장 0개)."""
    out = []
    for raw in lines:
        text = strip_wiki(raw)
        if prose_ok(raw, text):
            out.extend(sentences(text))
    return out


def is_bot(user):
    low = (user or "").lower()
    return "bot" in low or "봇" in low


class PageMiner:
    """문서 하나의 판을 차례로 받아 이웃 판 쌍을 거른다. 되돌리기(같은 sha1 으로 돌아감) 표시는 문서 끝에 붙인다."""

    def __init__(self, page, counts):
        self.page = page
        self.counts = counts
        self.prev = None
        self.sha_at = {}
        self.revs = []           # [(rev id, 되돌린 판인가)]
        self.reverted = set()    # 나중에 되돌려진 판 id (mw-reverted 와 같은 뜻)
        self.found = []          # [(옛 판 id, 새 판 id, 옛 문장, 새 문장, hint, kind, 편집 정보)]
        self.seen_pairs = set()

    def add(self, rev):
        self.counts["revisions"] += 1
        index = len(self.revs)
        restores = rev["sha1"] and rev["sha1"] in self.sha_at and self.sha_at[rev["sha1"]] < index - 1
        if restores:
            for k in range(self.sha_at[rev["sha1"]] + 1, index):
                self.reverted.add(self.revs[k][0])
        self.revs.append((rev["id"], bool(restores) or bool(_REVERT_COMMENT.search(rev["comment"]))))
        if rev["sha1"]:
            self.sha_at[rev["sha1"]] = index
        if self.prev is not None and rev["text"] is not None and self.prev["text"] is not None:
            self._compare(self.prev, rev)
        self.prev = rev

    def _compare(self, old, new):
        if old["sha1"] and old["sha1"] == new["sha1"]:
            return
        mid_a, mid_b = _middle_lines(old["text"], new["text"])
        if len(mid_a) > MAX_MID_LINES or len(mid_b) > MAX_MID_LINES:
            self.counts["drop_many_lines"] += 1
            return
        if sum(map(len, mid_a)) > MAX_MID_CHARS or sum(map(len, mid_b)) > MAX_MID_CHARS:
            self.counts["drop_long_line"] += 1
            return
        sent_a, sent_b = _prose_sentences(mid_a), _prose_sentences(mid_b)
        if not sent_a or not sent_b:
            self.counts["drop_not_prose"] += 1
            return
        ops = [op for op in difflib.SequenceMatcher(None, sent_a, sent_b, autojunk=False).get_opcodes()
               if op[0] != "equal"]
        if not ops:
            self.counts["drop_markup_only"] += 1
            return
        if len(ops) != 1 or ops[0][0] != "replace" or ops[0][2] - ops[0][1] != 1 or ops[0][4] - ops[0][3] != 1:
            self.counts["drop_multi_sentence"] += 1
            return
        self.counts["one_sentence"] += 1
        self._keep(old, new, sent_a[ops[0][1]], sent_b[ops[0][3]])

    def _keep(self, old, new, before, after):
        if len(before) < MIN_LEN or len(after) < MIN_LEN or len(before) > MAX_LEN or len(after) > MAX_LEN:
            self.counts["drop_length"] += 1
            return
        if difflib.SequenceMatcher(None, before, after, autojunk=False).ratio() < MIN_RATIO:
            self.counts["drop_not_similar"] += 1
            return
        if make_sets.list_like(before) or make_sets.list_like(after):
            self.counts["drop_list_like"] += 1
            return
        hint, kind = judge_pair(before, after)
        if hint is None:
            self.counts["drop_" + kind] += 1
            return
        if (before, after) in self.seen_pairs:
            self.counts["drop_dup_in_page"] += 1
            return
        self.seen_pairs.add((before, after))
        info = {"user_bot": is_bot(new["user"]), "anon": new["anon"], "ts": new["ts"]}
        self.found.append((old["id"], new["id"], before, after, hint, kind, info))

    def rows(self):
        """문서 끝 — 되돌리기 표시를 붙여 후보 줄을 낸다."""
        revert_of = dict(self.revs)
        title = self.page["title"]
        out = []
        for old_id, new_id, before, after, hint, kind, info in self.found:
            flags = []
            if info["user_bot"]:
                flags.append("bot")
            if new_id in self.reverted:
                flags.append("reverted")
            if revert_of.get(new_id):
                flags.append("revert")
            out.append({
                "evidence": f"{title} : {after}", "claim": before, "want": "", "hint": hint, "form": "wiki-edit",
                "kind": kind, "flag": flags, "anon": info["anon"], "ts": info["ts"],
                "domain": DOMAIN, "group": f"kowiki:{self.page['id']}", "groups": [f"kowiki:{self.page['id']}"],
                "lang": "ko", "lic": LIC, "split": "any", "src_ids": [f"kowiki:{old_id}", f"kowiki:{new_id}"],
                "origin": "g3-wiki", "source": DOMAIN, "title": title,
            })
        return out


def mine_pairs(stream, write_row, limit_pages=0, log=sys.stderr):
    """덤프 흐름 → 후보 줄을 write_row 로 낸다. 반환 : 셈 Counter."""
    counts = collections.Counter()
    miner = None
    start = time.time()
    for kind, page, rev in iter_pages(stream, limit_pages):
        if kind == "rev":
            if page["ns"] != "0" or page["redirect"]:
                continue
            if miner is None or miner.page is not page:
                miner = PageMiner(page, counts)
            miner.add(rev)
            continue
        counts["pages"] += 1
        if miner is not None and miner.page is page:
            counts["pages_main"] += 1
            for row in miner.rows():
                counts["kept"] += 1
                counts["hint_" + row["hint"]] += 1
                write_row(row)
        miner = None
        if counts["pages"] % LOG_EVERY == 0:
            print(f"[mine_wiki] 문서 {counts['pages']:,} · 본문 {counts['pages_main']:,} · 판 {counts['revisions']:,}"
                  f" · 한 문장 {counts['one_sentence']:,} · 남은 쌍 {counts['kept']:,} · {time.time() - start:.0f}초",
                  file=log, flush=True)
    counts["seconds"] = int(time.time() - start)
    return counts


# ---------------------------------------------------------------- split · stats · sample

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_bucket(group):
    """문서 id 묶음 → 해시 끝자리(0~9). 같은 문서는 늘 같은 자리라 한 셋에만 간다."""
    return int(hashlib.sha1(group.encode("utf-8")).hexdigest(), 16) % 10


def split_rows(rows, eval_digit=0):
    eval_rows = [r for r in rows if eval_bucket(r["group"]) == eval_digit]
    train_rows = [r for r in rows if eval_bucket(r["group"]) != eval_digit]
    return eval_rows, train_rows


def _pct(values, q):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def stats(rows):
    flags = collections.Counter()
    for row in rows:
        flags.update(row.get("flag") or ["none"])
    claims = [len(r["claim"]) for r in rows]
    evidences = [len(r["evidence"]) for r in rows]
    return {
        "rows": len(rows), "pages": len({r["group"] for r in rows}),
        "kind": dict(collections.Counter(r.get("kind", "") for r in rows).most_common()),
        "hint": dict(collections.Counter(r["hint"] for r in rows).most_common()),
        "hint_kind": dict(collections.Counter(f"{r['hint']}:{r.get('kind', '')}" for r in rows).most_common()),
        "flag": dict(flags.most_common()),
        "anon": sum(1 for r in rows if r.get("anon")),
        "claim_len_p50_p95": [_pct(claims, 0.5), _pct(claims, 0.95)],
        "evidence_len_p50_p95": [_pct(evidences, 0.5), _pct(evidences, 0.95)],
    }


# ---------------------------------------------------------------- 거르기 (filter)

# 「현재의」는 「1994년 현재의 상호」처럼 기준 때가 아니라 지금 것을 가리켜서 뺀다 (sets-g3.md 3-3)
_TIME_ANCHOR = re.compile(r"(?:1[0-9]{3}|20[0-9]{2})년(?:\s*[0-9]{1,2}월)?(?:\s*[0-9]{1,2}일)?\s*(?:현재(?!의)|기준)")
# 문서 기준일 메타 문장 (sets-g3.md 3-3 noise) — 「N년 M월 D일 기준이다.」「현재 앵커는 … 기준으로 한다.」「다음은 … 일람이다.」
# 「다음과 같다」는 「N년 현재 신자 교세는 다음과 같다」처럼 판정 d 가 섞여 넣지 않는다.
_META_DATE = re.compile(r"(?:1[0-9]{3}|20[0-9]{2})년(?:\s*[0-9]{1,2}월)?(?:\s*[0-9]{1,2}일)?\s*(?:이후|부터)?\s*(?:을|를)?"
                        r"(?:\s*[가-힣]+\s*시간을)?\s*(?:현재(?!의)|기준)")
_META_TAIL = re.compile(r"^[\s,.]*(?:으로|이다|이며|임)?[\s.]*$")
# 표 아래 기준일 주석이 앞 문장 마침표 뒤에 붙은 꼴 — 「… 기록이다.2017년 7월 24일 기준」 (sets-g3.md 4-5 · 5-1).
# 괄호에 싼 꼴 「… 진행하고 있다.(2008년 4월 현재)」는 판정 d 가 섞여 넣지 않는다.
_META_GLUED_TAIL = re.compile(r"[.。]\s*(?:1[0-9]{3}|20[0-9]{2})년(?:\s*[0-9]{1,2}월)?(?:\s*[0-9]{1,2}일)?\s*(?:현재|기준)[\s.]*$")
_META_MARK = re.compile(r"^(?:다음은|아래는|본\s)|기준으로\s*한다|(?:일람|목록)이다|아래와\s*같다"
                        r"|다음\s+[가-힣]+(?:과|와)\s|다음의\s")    # 다음의 : 「N년 M월 기준, ○○항공은 다음의 취항지를 운항중이다.」 (5-1)


def _after_text(row):
    """근거에서 「제목 : 」 머리를 뗀 고친 뒤 문장."""
    head = row.get("title", "") + " : "
    return row["evidence"][len(head):] if row["evidence"].startswith(head) else row["evidence"]


def time_anchor_only(before, after):
    """「N년 현재」 · 「N년 기준」 의 때만 바뀌고 나머지는 같은가 — 그러면 둘 다 참일 수 있어 d 후보다."""
    old_a, new_a = _TIME_ANCHOR.findall(before), _TIME_ANCHOR.findall(after)
    if not old_a or not new_a or old_a == new_a:
        return False
    return _anchor_norm(before) == _anchor_norm(after)


def _anchor_norm(text):
    """때 덩이를 자리표로 바꾸고 빈칸을 뺀 글 — 때 밖이 같은지 견주는 데 쓴다."""
    return _SPACE.sub("", _TIME_ANCHOR.sub("@", text))


def meta_sentence(text, glued_tail=False):
    """문서 기준일만 알리는 메타 문장인가 — 사실 내용이 없어 noise 다. 기준 때 덩이가 있을 때만 본다."""
    if glued_tail and _META_GLUED_TAIL.search(text):
        return True
    found = _META_DATE.search(text)
    if not found:
        return False
    rest = text[:found.start()] + text[found.end():]
    if _META_TAIL.match(rest) or rest.lstrip().startswith("("):
        return True                          # 「2023년 5월 13일 기준이다.」 · 「2020년 1월 1일 기준('○' …)」
    return bool(_META_MARK.search(text))


def _flagged(row):
    return bool(row.get("flag"))


def filter_rows(rows, kinds=None, time_anchor=False, flag_cap=None, seed=SEED, date_d_to_b=False, drop_meta=False,
                drop_meta_tail=False):
    """후보 줄 → 거른 줄 · 셈. 차례 : ⓪ 메타 문장 버림 ① 때만 바뀐 b 를 d 로 · 때만 바뀜 아닌 date d 를 b 로
    ② b 의 kind 거르기 ③ flag 붙은 b 상한.

    d 줄은 kind · flag 로 거르지 않는다(모자라서 다 판정에 넘긴다). 줄 차례는 입력 차례를 지킨다.
    """
    counts = collections.Counter(rows_in=len(rows))
    out = []
    for row in rows:
        row = dict(row)
        before, after = row["claim"], _after_text(row)
        if drop_meta and (meta_sentence(before) or meta_sentence(after)):
            counts["drop_meta"] += 1
            continue
        if drop_meta_tail and (meta_sentence(before, glued_tail=True) or meta_sentence(after, glued_tail=True)):
            counts["drop_meta_tail"] += 1
            continue
        if time_anchor and row["hint"] == "b" and time_anchor_only(before, after):
            row["hint"], row["rehint"] = "d", "time-anchor"
            counts["rehint_d"] += 1
        elif (date_d_to_b and row["hint"] == "d" and row.get("kind") == "date"
              and not time_anchor_only(before, after)):
            row["hint"], row["rehint"] = "b", "date-d"     # 한 날짜 고침이라 b 후보 (sets-g3.md 3-3 · d 통과 7%)
            counts["rehint_b"] += 1
        if kinds is not None and row["hint"] == "b" and row.get("kind") not in kinds:
            counts["drop_kind_" + str(row.get("kind"))] += 1
            continue
        out.append(row)
    if flag_cap is not None:
        b_rows = [k for k, r in enumerate(out) if r["hint"] == "b"]
        flagged = [k for k in b_rows if _flagged(out[k])]
        plain = len(b_rows) - len(flagged)
        room = int(flag_cap * plain / (1 - flag_cap)) if flag_cap < 1 else len(flagged)
        if len(flagged) > room:
            keep = set(random.Random(seed).sample(flagged, room))
            drop = set(flagged) - keep
            counts["drop_flag_cap"] = len(drop)
            out = [r for k, r in enumerate(out) if k not in drop]
    counts["rows_out"] = len(out)
    return out, counts


# ---------------------------------------------------------------- 조각 합치기 (merge)

def claim_hash(row):
    """주장 문장 해시 — 빈칸을 뺀 글로 센다. 다른 문서에 같은 문장이 있으면 같은 값이다."""
    return hashlib.sha1(_SPACE.sub("", row["claim"]).encode("utf-8")).hexdigest()


def merge_rows(chunks, eval_digit=0):
    """조각들 → (평가 줄, 학습 줄, 셈). 평가 · 학습은 조각별 split 과 같은 문서 해시로 가른다.

    같은 주장 문장은 한 줄만 남긴다. 평가와 학습에 같이 있으면 평가 쪽을 남겨 학습 → 평가 새는 길을 막는다.
    """
    rows = [row for chunk in chunks for row in chunk]
    eval_hashes = {claim_hash(r) for r in rows if eval_bucket(r["group"]) == eval_digit}
    counts = collections.Counter(rows_in=len(rows))
    seen = set()
    eval_rows, train_rows = [], []
    for row in rows:
        key = claim_hash(row)
        is_eval = eval_bucket(row["group"]) == eval_digit
        if key in seen:
            counts["drop_dup"] += 1
            continue
        if not is_eval and key in eval_hashes:
            counts["drop_train_in_eval"] += 1
            continue
        seen.add(key)
        (eval_rows if is_eval else train_rows).append(row)
    counts["eval"], counts["train"] = len(eval_rows), len(train_rows)
    return eval_rows, train_rows, counts


# ---------------------------------------------------------------- main

def _add_filter_args(parser, recommended):
    """거르기 옵션. pairs 는 기본이 꺼짐(옛 결과 그대로), filter 는 권장값이 기본."""
    parser.add_argument("--kinds", default=",".join(KEEP_KINDS) if recommended else "",
                        help="남길 b 의 kind 목록 (쉼표). 빈 값 = 다 남김. d 줄은 거르지 않는다")
    parser.add_argument("--time-anchor-d", action=argparse.BooleanOptionalAction, default=recommended,
                        help="「N년 현재 · 기준」 때만 바뀐 b 를 d 로 돌림")
    parser.add_argument("--date-d-to-b", action=argparse.BooleanOptionalAction, default=recommended,
                        help="때만 바뀜이 아닌 kind date 의 d 를 b 로 돌림 (한 날짜 고침)")
    parser.add_argument("--drop-meta", action=argparse.BooleanOptionalAction, default=recommended,
                        help="「N년 M월 D일 기준이다」류 문서 기준일 메타 문장을 버림")
    parser.add_argument("--drop-meta-tail", action=argparse.BooleanOptionalAction, default=recommended,
                        help="「… 이다.2017년 7월 24일 기준」처럼 마침표 뒤에 붙은 기준일 주석 문장을 버림")
    parser.add_argument("--flag-cap", type=float, default=FLAG_CAP if recommended else -1,
                        help="flag 붙은 b 를 남은 b 의 이 몫까지 (음수 = 안 함)")
    parser.add_argument("--filter-seed", type=int, default=SEED, help="flag 상한에서 남길 줄을 고르는 씨앗")


def _filter_opts(args):
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()} or None
    cap = args.flag_cap if args.flag_cap >= 0 else None
    return kinds, cap


def main(argv=None):
    parser = argparse.ArgumentParser(description="위키 개정 이력 → 한 자리 고침 b · d 후보")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs", help="덤프(.bz2) → 후보 jsonl")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit-pages", type=int, default=0, help="문서 몇 개에서 멈추나 (0 = 끝까지)")
    _add_filter_args(p, recommended=False)
    s = sub.add_parser("split", help="문서 id 해시로 평가 · 학습 몫 가르기")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out-eval", required=True)
    s.add_argument("--out-train", required=True)
    s.add_argument("--eval-digit", type=int, default=0)
    t = sub.add_parser("stats", help="종류 · hint · flag · 길이 셈")
    t.add_argument("--in", dest="inp", required=True)
    m = sub.add_parser("sample", help="고정 씨앗 무작위 표본")
    m.add_argument("--in", dest="inp", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=200)
    m.add_argument("--seed", type=int, default=SEED)
    g = sub.add_parser("merge", help="거른 조각들 → 하나로 (주장 문장 해시 중복 제거 · 평가 / 학습 가름)")
    g.add_argument("--in", dest="inputs", nargs="+", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--out-eval", required=True)
    g.add_argument("--out-train", required=True)
    g.add_argument("--eval-digit", type=int, default=0)
    f = sub.add_parser("filter", help="후보 jsonl → 거른 jsonl (kind · 때만 바뀜 → d · flag 상한)")
    f.add_argument("--in", dest="inp", required=True)
    f.add_argument("--out", required=True)
    _add_filter_args(f, recommended=True)
    args = parser.parse_args(argv)

    if args.cmd == "pairs":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        kinds, cap = _filter_opts(args)
        if (kinds is None and not args.time_anchor_d and cap is None and not args.date_d_to_b and not args.drop_meta
                and not args.drop_meta_tail):
            with bz2.open(args.dump, "rb") as stream, open(args.out, "w", encoding="utf-8", newline="\n") as out:
                counts = mine_pairs(stream, lambda row: out.write(json.dumps(row, ensure_ascii=False) + "\n"),
                                    args.limit_pages)
        else:                                # 거르기를 켜면 flag 상한 때문에 다 모은 뒤 쓴다
            found = []
            with bz2.open(args.dump, "rb") as stream:
                counts = mine_pairs(stream, found.append, args.limit_pages)
            kept, fcounts = filter_rows(found, kinds, args.time_anchor_d, cap, args.filter_seed,
                                        args.date_d_to_b, args.drop_meta, args.drop_meta_tail)
            write_jsonl(args.out, kept)
            counts.update({"filter_" + k: v for k, v in fcounts.items()})
        summary = dict(sorted(counts.items()))
        with open(args.out + ".summary.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "merge":
        eval_rows, train_rows, counts = merge_rows([read_jsonl(p) for p in args.inputs], args.eval_digit)
        write_jsonl(args.out, eval_rows + train_rows)
        write_jsonl(args.out_eval, eval_rows)
        write_jsonl(args.out_train, train_rows)
        print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
        return 0
    rows = read_jsonl(args.inp)
    if args.cmd == "filter":
        kinds, cap = _filter_opts(args)
        kept, counts = filter_rows(rows, kinds, args.time_anchor_d, cap, args.filter_seed,
                                   args.date_d_to_b, args.drop_meta, args.drop_meta_tail)
        write_jsonl(args.out, kept)
        print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "split":
        eval_rows, train_rows = split_rows(rows, args.eval_digit)
        write_jsonl(args.out_eval, eval_rows)
        write_jsonl(args.out_train, train_rows)
        print(json.dumps({"eval": len(eval_rows), "train": len(train_rows),
                          "eval_pages": len({r["group"] for r in eval_rows}),
                          "train_pages": len({r["group"] for r in train_rows})}, ensure_ascii=False))
        return 0
    if args.cmd == "stats":
        print(json.dumps(stats(rows), ensure_ascii=False, indent=1))
        return 0
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.n, len(rows)))
    write_jsonl(args.out, picked)
    print(json.dumps({"sampled": len(picked), "from": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
