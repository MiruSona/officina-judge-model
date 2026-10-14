"""seed_contrast — 실물 한국어 문장 씨앗 → 한 자리만 바꾼 b · 같은 값 다른 꼴 a · 다른 때 d 의 생성 입력 (K 판 ⑤ 앞부분).

설계 `../Docs/Design/2026-10-13-v1.2-한국어b보강판설계.md` 3-3 · 4-1 · 7절. 씨앗 · 초안 본문은 git 밖(`~/.judgemodel/build/g3/seed/`)에만 쓴다.

    python -I seed_contrast.py count --out ~/.judgemodel/build/g3/seed                 # 씨앗 세기 → seeds-all.jsonl · count.json
    python -I seed_contrast.py make  --out ~/.judgemodel/build/g3/seed [--n 450]       # 자리 · 새 값 · 초안 → input.jsonl · sample50.jsonl
    python -I seed_contrast.py check --in <생성 결과 또는 input.jsonl> [--out 결과.jsonl]  # 기계 검사 (한 덩이 · 새 값 · 길이 · 물음꼴 · 부정어 · scrub)

게임 저장소 문장은 쓰지 않는다. 금지 목록(g1 · g2 평가 묶음 · blind-g2.forbid · use1 · 잣대 기억)에 걸린 문서 · 기억은 통째로 뺀다.
"""
import argparse
import collections
import datetime
import difflib
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_batches as lb  # noqa: E402
import mine_archive as ma  # noqa: E402
import scrub  # noqa: E402

SEED = 20261301
TARGET = 450
SAMPLE = 50
GROUP_CAP = 3                    # 한 문서 · 한 기억 덩이에서 뽑는 씨앗 상한 — 문서 하나로 쏠리지 않게
SEED_LEN = (15, 220)
MAX_SLOTS = 6                    # 값 자리가 이보다 많으면 표 · 나열 글로 보고 뺀다
KINDS = ("num", "date", "ord", "onoff", "bound", "firstlast", "who")
TIME_KINDS = ("date",)
WEIGHTS = {"num": 4, "date": 2}  # 한 바퀴에 고르는 수 (나머지 종류 1) — 수 자리가 가장 많고 설계 보기(「3월 120만」)도 수다
G1_EVAL = ("dev-cal", "dev-cmp", "blind-g1", "ood-g1")
G2_EVAL = ("dev-cmp-g2", "dev-cal-g2", "blind-g2")
INSTRUCTION = ("문장 다듬기만 한다 : 초안의 조사 · 단위 · 띄어쓰기만 자연스럽게 고친다. "
               "바꾼 자리 밖의 낱말 · 값은 씨앗 그대로 두고, 새 값을 바꾸거나 부정어(않다 · 안 · 못 · 아니다)를 넣지 않는다.")

# ---------------------------------------------------------------- 값 자리 찾기

_UNITS = ("개월", "개", "가지", "번", "회", "명", "줄", "건", "쌍", "판", "장", "칸", "층", "바퀴", "글자", "자", "토큰", "턴",
          "%", "배", "초", "분", "시간", "일", "주", "달", "년", "TB", "GB", "MB", "KB", "K", "만", "천", "원", "달러", "점",
          "곳", "조각", "문장", "파일", "단계", "살")
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_NUM_CORE = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+"
_NUM = re.compile(rf"(?<![\w.,\-/:#~–×=])(?P<n>{_NUM_CORE})(?:\s?(?P<u>{_UNIT_RE}))?(?![\w\-~–/:×=]|[.,]\d)"
                  rf"|(?<![\w.,\-/:#~–×=])(?P<n2>{_NUM_CORE})\s?(?P<u2>{_UNIT_RE})(?=[가-힣])")
_NOT_VALUE_AFTER = re.compile(r"^\s?(?:절|호|항)")
_AFTER_NAME = re.compile(r"[A-Za-z_@\-][A-Za-z0-9_@\-]*\s$")   # 「-np 1」 「SA 4.0」 처럼 이름 · 옵션 뒤 맨 수는 값이 아니라 설정 · 판 이름
_FULL_DATE = re.compile(r"(?<![\w\-/])(?P<y>20\d\d)-(?P<m>\d\d)-(?P<d>\d\d)(?![\w\-/])")
_MD_DATE = re.compile(r"(?<![\w\-./:])(?P<m>0?[1-9]|1[0-2])-(?P<d>[0-2]\d|3[01])(?![\w\-/])")
_KO_DATE = re.compile(r"(?<!\d)(?P<m>1[0-2]|0?[1-9])월(?:\s?(?P<d>[1-9]|[12]\d|3[01])일)?")
_YEAR = re.compile(r"(?<!\d)(?P<y>(?:19|20)\d\d)년")
_CLOCK = re.compile(r"(?<!\d)(?P<h>[01]?\d|2[0-3])시(?![간작])")
_TIME_WORD = re.compile(r"(?<![가-힣])(아침|오전|오후|점심|저녁|밤|새벽)(?=$|[^가-힣]|[에은는의도과와까부])")
_QUOTED = re.compile(r'「[^」]*」|“[^”]*”|"[^"]*"|『[^』]*』')   # 따옴 안의 값은 보기 글이라 씨앗 자리로 안 쓴다
_KO_TAIL = r"(?=$|[^가-힣]|[이으은을의에로도])"   # 뒤에 낱말이 더 붙으면(「이상한」) 값 자리가 아니다 — 조사만 허용
_ORD_NUM = re.compile(rf"(?<![\w.\-])(?P<n>\d{{1,2}})(?P<u>차|번째|회차|판째|바퀴째){_KO_TAIL}")
_ORD_WORD = re.compile(r"(?<![가-힣])(첫째|둘째|셋째|넷째|첫 번째|두 번째|세 번째|네 번째|다섯 번째|첫(?=\s[가-힣]))")
_BOUND = re.compile(rf"(?<![가-힣])(이상|이하|미만|초과|넘는|넘으면|넘었다|넘게|못 미치는){_KO_TAIL}")
_ONOFF = re.compile(rf"(?<![가-힣])(켬|끔|켠다|끈다|켰다|껐다|켜짐|꺼짐|켜진|꺼진|켜 둔다|꺼 둔다|있음|없음){_KO_TAIL}"
                    r"|(?<![A-Za-z\-.])(on|off|ON|OFF)(?![A-Za-z(_\-])")
_FIRSTLAST = re.compile(r"(?<![가-힣])(처음|마지막|먼저|나중에|맨 앞|맨 뒤)(?=$|[^가-힣]|[에은는의도과와까부으로이])")
_PARTICLE = r"(?=$|[^\w가-힣]|[이가은는을를의에와과도만께로])"
_WHO = re.compile(rf"(?<![\w가-힣\-])(사용자|메인|Opus|Sonnet|Fable|Qwen3\.6|Flash-Next|서브에이전트|Gemma 4){_PARTICLE}")

_FLIP = {"켬": "끔", "끔": "켬", "켠다": "끈다", "끈다": "켠다", "켰다": "껐다", "껐다": "켰다", "켜짐": "꺼짐", "꺼짐": "켜짐",
         "켜진": "꺼진", "꺼진": "켜진", "켜 둔다": "꺼 둔다", "꺼 둔다": "켜 둔다", "있음": "없음", "없음": "있음",
         "on": "off", "off": "on", "ON": "OFF", "OFF": "ON",
         "이상": "이하", "이하": "이상", "미만": "초과", "초과": "미만", "넘는": "못 미치는", "못 미치는": "넘는",
         "넘으면": "못 미치면", "넘었다": "못 미쳤다", "넘게": "못 미치게",
         "처음": "마지막", "마지막": "처음", "먼저": "나중에", "나중에": "먼저", "맨 앞": "맨 뒤", "맨 뒤": "맨 앞"}
_TIME_SWAP = {"아침": "저녁", "오전": "오후", "오후": "오전", "점심": "저녁", "저녁": "아침", "밤": "아침", "새벽": "저녁"}
_WHO_SWAP = {"사용자": ("메인", "Opus"), "메인": ("사용자",), "Opus": ("Sonnet", "사용자"), "Sonnet": ("Opus",),
             "Fable": ("Opus",), "Qwen3.6": ("Flash-Next",), "Flash-Next": ("Qwen3.6",), "서브에이전트": ("메인", "사용자"),
             "Gemma 4": ("Qwen3.6",)}
_ORD_KO = {1: "첫 번째", 2: "두 번째", 3: "세 번째", 4: "네 번째", 5: "다섯 번째", 6: "여섯 번째", 7: "일곱 번째"}
_ORD_WORD_NEXT = {"첫째": "둘째", "둘째": "셋째", "셋째": "넷째", "넷째": "셋째", "첫 번째": "두 번째", "두 번째": "세 번째",
                  "세 번째": "네 번째", "네 번째": "세 번째", "다섯 번째": "네 번째", "첫": "두 번째"}
_NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열", 20: "스무"}
_NATIVE_UNITS = ("개", "가지", "명", "줄", "건", "쌍", "판", "장", "칸", "층", "바퀴", "곳", "조각", "달", "시간", "살", "문장")


def _slot(kind, m, start=None, end=None, **extra):
    s = m.start() if start is None else start
    e = m.end() if end is None else end
    return dict({"kind": kind, "start": s, "end": e, "text": m.string[s:e]}, **extra)


def find_slots(text):
    """문장 → 값 자리 목록 (겹치면 먼저 찾은 것이 이긴다 : 날짜 → 서수 → 비교말 → 수 → 켬/끔 → 처음/마지막 → 누가)."""
    found = []
    for m in _FULL_DATE.finditer(text):
        found.append(_slot("date", m, form="ymd"))
    for m in _MD_DATE.finditer(text):
        found.append(_slot("date", m, form="md"))
    for m in _KO_DATE.finditer(text):
        found.append(_slot("date", m, form="ko"))
    for m in _YEAR.finditer(text):
        found.append(_slot("date", m, form="year"))
    for m in _CLOCK.finditer(text):
        found.append(_slot("date", m, form="clock"))
    for m in _TIME_WORD.finditer(text):
        found.append(_slot("date", m, form="word"))
    for m in _ORD_NUM.finditer(text):
        found.append(_slot("ord", m, form="num"))
    for m in _ORD_WORD.finditer(text):
        found.append(_slot("ord", m, form="word"))
    for m in _BOUND.finditer(text):
        if re.search(r"\d", text[max(0, m.start() - 12):m.start()]):
            found.append(_slot("bound", m))
    for m in _NUM.finditer(text):
        num, unit = (m.group("n"), m.group("u")) if m.group("n") else (m.group("n2"), m.group("u2"))
        if _NOT_VALUE_AFTER.match(text[m.end():]) or (not unit and re.fullmatch(r"(19|20)\d\d", num)):
            continue
        if not unit and ("," not in num and "." not in num and int(num) >= 1000 or _AFTER_NAME.search(text[:m.start()])):
            continue
        found.append(_slot("num", m, num=num, unit=unit or ""))
    for m in _ONOFF.finditer(text):
        found.append(_slot("onoff", m))
    for m in _FIRSTLAST.finditer(text):
        found.append(_slot("firstlast", m))
    for m in _WHO.finditer(text):
        found.append(_slot("who", m))
    taken, out = [], []
    quoted = [m.span() for m in _QUOTED.finditer(text)]
    for slot in found:
        if any(slot["start"] < e and s < slot["end"] for s, e in taken):
            continue
        taken.append((slot["start"], slot["end"]))
        if any(s <= slot["start"] and slot["end"] <= e for s, e in quoted) or _not_a_value(text, slot):
            continue
        out.append(slot)
    return sorted(out, key=lambda s: s["start"])


_IDIOM_AFTER = re.compile(r"\s?(?:확답|확신|확실|장담|보장|동의|믿|자신)")   # 「100% 확답은 못 한다」 — 값이 아니라 관용구
_FIXED_PHRASE = re.compile(r"부터|까지")                                      # 「처음부터 다시」 「마지막까지」 — 뒤집으면 뜻이 안 선다


def _in_range(text, slot):
    """범위의 한쪽 끝(「16GB ~ 18GB」 「9~10월」) — 한쪽만 바꾸면 범위가 깨지거나 뒤집힌다."""
    return bool(re.match(r"\s?[~–]", text[slot["end"]:]) or re.search(r"[~–]\s?$", text[:slot["start"]]))


def _not_a_value(text, slot):
    """범위 끝 · 판 이름 꼴 맨 소수(「3.6 으로」) · 관용구(「100% 확답」) · 굳은 말(「처음부터」) ·
    부정이 붙은 비교말 · 「깔려 있음」 꼴의 있음/없음은 값 자리가 아니다."""
    if slot["kind"] in ("num", "date", "ord") and _in_range(text, slot):
        return True
    if slot["kind"] == "num":
        if not slot["unit"] and "." in slot["num"] and not slot["num"].startswith(".") and _parse_num(slot["num"]) >= 1:
            return True   # 단위 없는 1 이상 소수는 모델 · 판 이름(「3.6」 「1.8」)이 많아 값으로 안 친다
        return slot["text"] == "100%" and bool(_IDIOM_AFTER.match(text, slot["end"]))
    if slot["kind"] == "firstlast" and _FIXED_PHRASE.match(text, slot["end"]):
        return True
    if slot["kind"] == "bound" and re.search(r"(못|안)\s?$", text[:slot["start"]]):
        return True   # 「못 넘었다」를 뒤집으면 「못 못 미쳤다」 — 부정이 붙은 비교말은 뺀다
    if slot["kind"] == "onoff" and slot["text"] in ("있음", "없음"):
        before = text[:slot["start"]]
        if len(before) >= 2 and before[-1] == " " and "가" <= before[-2] <= "힣":
            code = ord(before[-2]) - 0xAC00
            return code % 28 == 0 and (code // 28) % 21 in (0, 1, 4, 6, 9, 14)   # 받침 없는 ㅏ ㅐ ㅓ ㅕ ㅘ ㅝ — 「-아/어 있음」
    return False


# ---------------------------------------------------------------- 새 값

def _parse_num(text):
    return float(text.replace(",", "")) if not text.startswith(".") else float("0" + text)


def _fmt_num(value, like):
    """like(원래 글자)와 같은 꼴로 — 소수 자리 · 천 단위 쉼표 · 앞 0 생략."""
    if "." in like:
        places = len(like.split(".")[1])
        out = f"{value:.{places}f}"
        return out[1:] if like.startswith(".") and out.startswith("0.") else out
    out = str(int(round(value)))
    return f"{int(out):,}" if "," in like else out


def _round_sig(value, sig):
    if value <= 0:
        return value
    digits = len(str(int(value)))
    step = 10 ** max(0, digits - sig)
    return round(value / step) * step


def _num_candidates(num, unit, rng):
    """같은 꼴 · ×0.5~×2 안쪽의 그럴듯한 값들 (차례 섞음)."""
    v = _parse_num(num)
    out = []
    if "." in num and v < 1:
        places = len(num.split(".")[1])
        step = 10 ** -places
        for delta in (0.05, 0.08, 0.1, 0.12, 0.15, -0.05, -0.08, -0.1, -0.12):
            w = round(v + delta, places)
            if step <= w < 1 and v * 0.5 <= w <= v * 2:
                out.append(w)
    elif "." in num:
        out = [v * f for f in (0.5, 0.6, 0.75, 1.25, 1.5, 2)]
    elif v <= 10:
        out = [w for w in range(max(1, int(v // 2)), int(v * 2) + 1) if w != v]
    else:
        stripped = num.replace(",", "").rstrip("0") or "0"
        sig = max(1, min(3, len(stripped)))
        out = [_round_sig(v * f, sig) for f in (0.5, 0.6, 0.75, 1.25, 1.5, 2)]
    if unit == "%":
        out = [w for w in out if 1 <= w <= 100]
    rng.shuffle(out)
    return [_fmt_num(w, num) for w in out]


def _shift_date(slot, rng):
    text = slot["text"]
    form = slot["form"]
    k = rng.choice((-3, -2, -1, 1, 2, 3))
    if form in ("ymd", "md"):
        m = (_FULL_DATE if form == "ymd" else _MD_DATE).fullmatch(text)
        year = int(m.group("y")) if form == "ymd" else 2026
        try:
            day = datetime.date(year, int(m.group("m")), int(m.group("d"))) + datetime.timedelta(days=k)
        except ValueError:
            return None
        if form == "ymd":
            return day.isoformat()
        width = len(m.group("m"))
        return f"{day.month:0{width}d}-{day.day:02d}"
    if form == "ko":
        m = _KO_DATE.fullmatch(text)
        if m.group("d"):
            try:
                day = datetime.date(2026, int(m.group("m")), int(m.group("d"))) + datetime.timedelta(days=k)
            except ValueError:
                return None
            return f"{day.month}월 {day.day}일" if " " in text else f"{day.month}월{day.day}일"
        month = (int(m.group("m")) - 1 + k) % 12 + 1
        return f"{month}월"
    if form == "year":
        return f"{int(text[:4]) + (1 if k > 0 else -1)}년"
    if form == "clock":
        hour = (int(text[:-1]) + k) % 24
        return f"{hour}시"
    return _TIME_SWAP.get(text)


def new_value(slot, seed_text, rng):
    """바꿀 자리의 새 값 — 같은 종류 · 씨앗에 이미 없는 값. 못 정하면 None."""
    kind, text = slot["kind"], slot["text"]
    if kind == "num":
        for cand in _num_candidates(slot["num"], slot["unit"], rng):
            if cand != slot["num"] and not has_value(cand, seed_text):
                return text.replace(slot["num"], cand, 1)
        return None
    if kind == "date":
        for _ in range(4):
            cand = _shift_date(slot, rng)
            if cand and cand != text and cand not in seed_text:
                return cand
        return None
    if kind == "ord":
        if slot["form"] == "num":
            m = _ORD_NUM.fullmatch(text)
            n = int(m.group("n"))
            n2 = n + 1 if n == 1 or rng.random() < 0.5 else n - 1
            cand = f"{n2}{m.group('u')}"
        else:
            cand = _ORD_WORD_NEXT.get(text)
        return cand if cand and cand not in seed_text else None
    if kind == "who":
        for cand in _WHO_SWAP.get(text, ()):
            if cand not in seed_text:
                return cand
        return None
    cand = _FLIP.get(text)
    return cand if cand and cand not in seed_text else None


def _korean_big(n):
    """47000 → 4만 7천 · 1300 → 1천3백 (1,000 이상 정수만)."""
    parts = []
    man, rest = divmod(n, 10000)
    if man:
        parts.append(f"{man}만")
    cheon, rest = divmod(rest, 1000)
    baek, rest = divmod(rest, 100)
    small = (f"{cheon}천" if cheon else "") + (f"{baek}백" if baek else "") + (str(rest) if rest else "")
    if small:
        parts.append(small)
    return " ".join(parts)


def same_value_other_form(slot):
    """a 짝 : 같은 값을 다른 꼴로 (규칙으로 되는 것만). 못 만들면 None → 로컬 모델 몫."""
    kind, text = slot["kind"], slot["text"]
    if kind == "num":
        num, unit = slot["num"], slot["unit"]
        v = _parse_num(num)
        whole = v == int(v) and "." not in num
        if unit == "초" and whole and v >= 60:
            mins, secs = divmod(int(v), 60)
            return f"{mins}분" + (f" {secs}초" if secs else "")
        if unit == "분" and whole and v >= 60:
            hours, mins = divmod(int(v), 60)
            return f"{hours}시간" + (f" {mins}분" if mins else "")
        if unit == "개월" and whole and v % 12 == 0:
            return f"{int(v) // 12}년"
        if unit == "%" and v == 50:
            return "절반"
        if unit == "배" and v == 0.5:
            return "절반"
        if unit == "배" and v == 2.5:
            return "두 배 반"
        if num.startswith("."):
            return text.replace(num, "0" + num, 1)
        if whole and int(v) in _NATIVE and unit in _NATIVE_UNITS + ("배",):
            return f"{_NATIVE[int(v)]} {unit}"
        if whole and v >= 1000 and v % 100 == 0:
            return text.replace(num, _korean_big(int(v)), 1)
        return None
    if kind == "date":
        if slot["form"] == "md":
            m = _MD_DATE.fullmatch(text)
            return f"{int(m.group('m'))}월 {int(m.group('d'))}일"
        if slot["form"] == "ymd":
            m = _FULL_DATE.fullmatch(text)
            return f"{m.group('y')}년 {int(m.group('m'))}월 {int(m.group('d'))}일"
        return None
    if kind == "ord":
        if slot["form"] == "num":
            m = _ORD_NUM.fullmatch(text)
            n = int(m.group("n"))
            if m.group("u") in ("차", "번째") and n in _ORD_KO:
                return _ORD_KO[n]
        if text == "첫째":
            return "첫 번째"
        return None
    if kind == "onoff":
        return {"켬": "켜 둠", "끔": "꺼 둠", "켠다": "켜 둔다", "끈다": "꺼 둔다"}.get(text)
    return None


# ---------------------------------------------------------------- 금지 목록

def _decode_git_path(path):
    """git 의 따옴표 · 8진 이스케이프 경로 → 글자."""
    path = path.strip()
    if not (path.startswith('"') and path.endswith('"')):
        return path
    raw = path[1:-1]
    data = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 3 < len(raw) + 1 and re.fullmatch(r"[0-7]{3}", raw[i + 1:i + 4]):
            data.append(int(raw[i + 1:i + 4], 8))
            i += 4
        elif raw[i] == "\\" and i + 1 < len(raw):
            data.extend(raw[i + 1].encode("utf-8"))
            i += 2
        else:
            data.extend(raw[i].encode("utf-8"))
            i += 1
    return data.decode("utf-8", errors="replace")


def _studio_path(repo, path):
    """(저장소 이름, 그 안 경로) → 스튜디오 뿌리 기준 경로. 게임 저장소면 None."""
    if repo == "game":
        return None
    path = unicodedata.normalize("NFC", path.replace("\\", "/").lstrip("/"))
    return path if repo == "studio" else f"{repo}/{path}"


def group_to_key(group):
    """평가 셋 묶음 → ("doc", 스튜디오 경로) · ("mem", 덩이) · None (공개 자료 · 게임)."""
    if not group:
        return None
    if group.startswith("doc:"):
        _, repo, path = group.split(":", 2)
        key = _studio_path(repo, path)
        return ("doc", key) if key else None
    if group.startswith("git:"):
        _, repo, path = group.split(":", 2)
        key = _studio_path(repo, _decode_git_path(path))
        return ("doc", key) if key else None
    if group.startswith("mem:"):
        return ("mem", group)
    return None


class SeedForbid:
    """씨앗이 피할 것 — 문서 경로 · 기억 id · 기억 덩이 · 평가 · use1 문장."""

    def __init__(self, judge_root, studio_root=None, git_files=None):
        self.missing = []
        self.docs, self.mem_groups, self.mem_ids, self.texts = set(), set(), set(), set()
        self.ruler_ids = set()
        eval_paths = [os.path.join(judge_root, "sets", "g1", n + ".jsonl") for n in G1_EVAL]
        eval_paths += [os.path.join(judge_root, "sets", "g2", n + ".jsonl") for n in G2_EVAL]
        for path in eval_paths:
            if not os.path.exists(path):
                self.missing.append(os.path.relpath(path, judge_root))
                continue
            self._eat_rows(path)
        forbid_path = os.path.join(judge_root, "sets", "g2", "blind-g2.forbid.json")
        if os.path.exists(forbid_path):
            self._eat_forbid(forbid_path, studio_root, git_files or _git_files)
        else:
            self.missing.append("blind-g2.forbid.json")
        use1 = os.path.join(judge_root, "measure", "use1")
        use1_files = sorted(glob.glob(os.path.join(use1, "*", "*.jsonl")))
        if not use1_files:
            self.missing.append("measure/use1")
        for path in use1_files:
            with open(path, encoding="utf-8") as f:
                body = f.read()
            self.mem_ids.update(ma._MEM_ID.findall(body))
            for line in body.splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.texts.update(norm(row.get(side)) for side in ("evidence", "claim"))
        ruler = os.path.join(judge_root, "build", "g2", "forbid-ruler-ids.txt")
        if os.path.exists(ruler):
            with open(ruler, encoding="utf-8") as f:
                self.ruler_ids = set(ma._MEM_ID.findall(f.read()))
        else:
            self.missing.append("forbid-ruler-ids.txt")
        self.mem_ids |= self.ruler_ids
        self.texts.discard("")

    def _eat_rows(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.mem_ids.update(ma._MEM_ID.findall(line))
                row = json.loads(line)
                self.texts.update(norm(row.get(side)) for side in ("evidence", "claim"))
                for group in [row.get("group")] + list(row.get("groups") or []):
                    self._eat_group(group)

    def _eat_group(self, group):
        key = group_to_key(group)
        if key and key[0] == "doc":
            self.docs.add(key[1])
        elif key:
            self.mem_groups.add(key[1])
            self.mem_ids.update(ma._MEM_ID.findall(key[1]))

    def _eat_forbid(self, path, studio_root, git_files):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.mem_ids.update(data.get("mem_ids") or [])
        for group in data.get("groups") or []:
            self._eat_group(group)
        for item in data.get("doc_paths") or []:
            repo, _, rel = item.partition(":")
            key = _studio_path(repo, rel) if rel else None
            if key:
                self.docs.add(key)
        for item in data.get("commits") or []:
            repo, _, sha = item.partition(":")
            if repo == "game" or not studio_root:
                continue
            files = git_files(studio_root if repo == "studio" else os.path.join(studio_root, repo), sha)
            if files is None:
                self.missing.append(f"commit {item}")
                continue
            self.docs.update(k for k in (_studio_path(repo, p) for p in files) if k)

    def reasons(self, seed):
        """seed = {kind: doc|mem, path|mem_id, group, text} → 못 쓰는 까닭 목록."""
        out = []
        if seed["source"] == "doc" and seed["path"] in self.docs:
            out.append("doc_eval")
        if seed["source"] == "mem":
            if seed["mem_id"] in self.ruler_ids:
                out.append("mem_ruler")
            if seed["mem_id"] in self.mem_ids:
                out.append("mem_eval")
            if seed["group"] in self.mem_groups:
                out.append("mem_group_eval")
        if norm(seed["text"]) in self.texts:
            out.append("text_eval")
        return out


def _git_files(repo_path, sha):
    try:
        res = subprocess.run(["git", "-C", repo_path, "-c", "core.quotepath=off", "show", "--name-only", "--format=", sha],
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().lower()


# ---------------------------------------------------------------- 씨앗 세기

def read_sources(studio_root):
    """스튜디오 문서(Docs 아래 md) · 기억 → 문장 줄 [{source, path|mem_id, group, text}]."""
    rows = []
    for doc in ma.read_docs([("studio", studio_root)]):
        sentences = list(doc["intro_ok"]) + [s for _, s in doc["flat"]]
        for text in sentences:
            rows.append({"source": "doc", "path": unicodedata.normalize("NFC", doc["rel"]), "mem_id": "",
                         "group": f"doc:studio:{doc['rel']}", "text": text})
    store = os.path.join(studio_root, "Memory", "store")
    if os.path.isdir(store):
        mems = ma.read_mem_store(store, "studio")
        groups = ma.mem_groups(mems)
        for mid, mem in mems.items():
            for _, text in ma._mem_units(mem):
                rows.append({"source": "mem", "path": "", "mem_id": mid, "group": groups[mid], "text": text})
    return rows


def seed_ok(text):
    """씨앗 꼴 거르기 — 한국어 · 길이 · 물음꼴 아님. 통과면 None, 아니면 까닭."""
    if ma.lang_of(text) != "ko":
        return "not_ko"
    if not SEED_LEN[0] <= len(text) <= SEED_LEN[1]:
        return "length"
    if is_question(text):
        return "question"
    if "|" in text:
        return "table"
    return None


def is_question(text):
    tail = text.rstrip(" )」』'\"")
    return tail.endswith("?") or bool(re.search(r"(까|나요|가요|니)[.]?$", tail))


def count_seeds(rows, forbid, scrubber):
    """문장 줄 → (쓸 씨앗, 셈). 같은 문장은 한 번만."""
    count = collections.Counter()
    kinds_before, kinds_after = collections.Counter(), collections.Counter()
    by_source = collections.Counter()
    seen, out = set(), []
    for row in rows:
        key = norm(row["text"])
        if key in seen:
            count["dup"] += 1
            continue
        seen.add(key)
        why = seed_ok(row["text"])
        if why:
            count[f"skip_{why}"] += 1
            continue
        slots = find_slots(row["text"])
        if not slots:
            count["no_slot"] += 1
            continue
        if len(slots) > MAX_SLOTS:
            count["too_many_slots"] += 1
            continue
        count["with_slot"] += 1
        kinds = {s["kind"] for s in slots}
        kinds_before.update(kinds)
        reasons = forbid.reasons(row)
        cleaned = scrubber.apply({"evidence": row["text"]}, collections.Counter())
        if cleaned is None or sum(scrubber.hits({"evidence": row["text"]}).values()):
            reasons.append("scrub")
        if reasons:
            count["forbidden"] += 1
            for r in reasons:
                count[f"forbid_{r}"] += 1
            continue
        kinds_after.update(kinds)
        by_source[row["source"]] += 1
        out.append(dict(row, slots=slots))
    count["kept"] = len(out)
    return out, {"count": dict(count), "kinds_before": dict(kinds_before), "kinds_after": dict(kinds_after),
                 "by_source": dict(by_source)}


# ---------------------------------------------------------------- 자리 고르기 · 초안

def _rng_for(text, seed=SEED):
    return random.Random(int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:12], 16))


def _replace(text, slot, value):
    return text[:slot["start"]] + value + text[slot["end"]:]


_ORDER_DATE = r"(?:20\d\d-)?\d{1,2}-\d\d"
_ORDER_PAIR = re.compile(rf"(?<![\w\-/])(?P<a>{_ORDER_DATE})(?:\s\d{{1,2}}:\d\d)?\s?[~–→]\s?(?P<b>{_ORDER_DATE})(?![\w\-/])")


def _date_key(text):
    parts = [int(p) for p in text.split("-")]
    return tuple([2026] * (3 - len(parts)) + parts)


def order_broken(text):
    """「08-30 23:35 ~ 08-31」 처럼 범위 · 순서로 이은 두 날짜가 거꾸로 섰나 (앞 > 뒤)."""
    return any(_date_key(m.group("a")) > _date_key(m.group("b")) for m in _ORDER_PAIR.finditer(text))


def _new_keeping_order(slot, text, rng, avoid=None, tries=8):
    """새 값 — 씨앗에서 멀쩡하던 범위 · 순서를 깨는 값은 다시 뽑는다."""
    was_broken = order_broken(text)
    for _ in range(tries):
        cand = new_value(slot, text, rng)
        if cand and cand != avoid and (was_broken or not order_broken(_replace(text, slot, cand))):
            return cand
    return None


def plan_seed(row, kind, seed=SEED):
    """씨앗 하나 · 자리 종류 하나 → 입력 줄 (못 만들면 None). 때 자리와 값 자리가 같이 있으면 d 초안도."""
    text = row["text"]
    rng = _rng_for(text, seed)
    options = [s for s in row["slots"] if s["kind"] == kind]
    options.sort(key=lambda s: (s["kind"] == "num" and not s["unit"], same_value_other_form(s) is None, s["start"]))
    for slot in options:
        new = _new_keeping_order(slot, text, rng)
        if not new:
            continue
        a_value = same_value_other_form(slot)
        d = _d_for(row, slot, new, rng)
        return {"slot": {k: slot[k] for k in ("kind", "start", "end", "text")}, "new": new,
                "a_value": a_value, "a_mode": "rule" if a_value else "llm",
                "d": d, "drafts": {"b": _replace(text, slot, new), "a": _replace(text, slot, a_value) if a_value else None,
                                   "d": d["draft"] if d else None}}
    return None


def _d_for(row, slot, new, rng):
    """d 짝 자리 : b 가 값 자리를 바꿨으면 그 값 + 다른 때 자리, b 가 때를 바꿨으면 그 때(다른 새 때) + 값 자리 하나."""
    if slot["kind"] in TIME_KINDS:
        values = [s for s in row["slots"] if s["kind"] not in TIME_KINDS]
        return _plan_d(row["text"], values[0], slot, "", new, rng) if values else None
    times = [s for s in row["slots"] if s["kind"] in TIME_KINDS]
    return _plan_d(row["text"], slot, times[0], new, "", rng) if times else None


def _plan_d(text, value_slot, time_slot, avoid_value, avoid_time, rng):
    """d : 같은 자리 · 다른 때 — 때 자리와 값 자리를 함께 바꾼다 (b 의 새 값과는 다르게)."""
    time_new = _new_keeping_order(time_slot, text, rng, avoid=avoid_time, tries=6)
    value_new = _new_keeping_order(value_slot, text, rng, avoid=avoid_value, tries=6)
    if not time_new or not value_new:
        return None
    first, second = sorted(((value_slot, value_new), (time_slot, time_new)), key=lambda p: p[0]["start"], reverse=True)
    draft = _replace(_replace(text, *first), *second)
    if order_broken(draft) and not order_broken(text):
        return None
    return {"time_slot": time_slot["text"], "time_new": time_new, "value_slot": value_slot["text"],
            "value_new": value_new, "draft": draft}


def _d_able(row):
    kinds = {s["kind"] in TIME_KINDS for s in row["slots"]}
    return kinds == {True, False}


def _take(kind, queues, pos, used, per_group, picked, group_cap, seed):
    """kind 줄에서 다음 씨앗 하나를 골라 picked 에 넣는다. 줄이 바닥나면 False."""
    while pos[kind] < len(queues[kind]):
        row = queues[kind][pos[kind]]
        pos[kind] += 1
        if id(row) in used or per_group[row["group"]] >= group_cap:
            continue
        plan = plan_seed(row, kind, seed)
        if plan:
            used.add(id(row))
            per_group[row["group"]] += 1
            picked.append((row, plan))
            return True
    return False


def choose(seeds, n=TARGET, seed=SEED, group_cap=GROUP_CAP, weights=None):
    """씨앗 n 개 — 종류를 무게(WEIGHTS)대로 돌아가며 고른다. 같은 종류 안에서는 d 를 만들 수 있는 씨앗 먼저, 묶음마다 group_cap 까지."""
    weights = weights or WEIGHTS
    rng = random.Random(seed)
    order = list(seeds)
    rng.shuffle(order)
    queues = {}
    for kind in KINDS:
        has = [r for r in order if any(s["kind"] == kind for s in r["slots"])]
        queues[kind] = [r for r in has if _d_able(r)] + [r for r in has if not _d_able(r)]
    used, per_group, picked = set(), collections.Counter(), []
    pos = {k: 0 for k in KINDS}
    active = [k for k in KINDS if queues[k]]
    while len(picked) < n and active:
        for kind in list(active):
            for _ in range(weights.get(kind, 1)):
                if len(picked) >= n or kind not in active:
                    break
                if not _take(kind, queues, pos, used, per_group, picked, group_cap, seed):
                    active.remove(kind)
    out = []
    for i, (row, plan) in enumerate(picked):
        lines = ["b", "a", "d"] if plan["d"] else ["b", "a"]
        out.append({"seed_id": f"k{i + 1:04d}", "source": row["source"], "group": row["group"], "seed": row["text"],
                    **plan, "want_lines": lines, "instruction": INSTRUCTION})
    return out


# ---------------------------------------------------------------- 기계 검사

_TOKEN = re.compile(r"\d{4}-\d\d-\d\d|\d{1,2}-\d\d(?!\d)|[A-Za-z0-9_.,%]+|[가-힣]+|\S")
_NEG = re.compile(r"않|안 |못 |아니|없")
LIMITS = {"b": 1, "a": 2, "d": 3}


def has_value(value, text):
    """text 에 value 가 따로 서 있나 — 「1」이 「16」 · 「2026」 안에 든 것은 안 친다."""
    head = r"(?<![\d.,])" if value[:1].isdigit() else ""
    tail = r"(?![\d.,]\d|\d)" if value[-1:].isdigit() else ""
    return re.search(head + re.escape(value) + tail, text) is not None


def chunks(old, new):
    """낱말 단위로 견준 바뀐 덩이 수."""
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    return sum(1 for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal")


def changed_tokens(old, new):
    """바뀐 토큰 수 — 덩이 수(`chunks`)는 문장 통째 바꿈도 1 로 세어, 덩이마다 긴 쪽 토큰 수를 더한다.
    `rewrite_check.changed_tokens` 를 베꼈다 (그쪽이 이 모듈을 import 해서 거꾸로 import 하면 돈다)."""
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")


TOKEN_SLACK = {"b": 1, "a": 2, "d": 2}   # 바뀐 토큰 상한 = 자리 토큰 + 이만큼 (조사가 따로 떨어진 꼴 · a 의 풀어 쓴 꼴)


def _ntok(text):
    return len(_TOKEN.findall(text or ""))


def token_limit(role, row):
    """줄 종류별 바뀐 토큰 상한. 자리 토큰 = 옛 자리 · 새 자리 중 긴 쪽 (「2026-08-28」 → 「2026년 8월 28일」)."""
    slot = (row.get("slot") or {}).get("text", "")
    if role == "b":
        base = max(_ntok(slot), _ntok(row.get("new")))
    elif role == "a":
        base = max(_ntok(slot), _ntok(row.get("a_value")))
    else:
        d = row.get("d") or {}
        base = max(_ntok(d.get("time_slot")), _ntok(d.get("time_new"))) + max(_ntok(d.get("value_slot")),
                                                                             _ntok(d.get("value_new")))
    return max(1, base) + TOKEN_SLACK[role]


# 한글 수 → 아라비아 수 (a 줄에 b 의 새 값이 한글로 숨었나 보려고). 하루 · 열흘 꼴은 「N일」 로.
_DAYS = {"하루": 1, "이틀": 2, "사흘": 3, "나흘": 4, "닷새": 5, "엿새": 6, "이레": 7, "여드레": 8, "아흐레": 9, "열흘": 10, "보름": 15}
_NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
_NATIVE_ONES = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "석": 3, "넷": 4, "네": 4, "넉": 4, "다섯": 5,
                "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_SINO_DIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_MUL = {"십": 10, "백": 100, "천": 1000}
_ALT = lambda words: "|".join(sorted(words, key=len, reverse=True))  # noqa: E731
_KO_DAYS = re.compile(rf"(?<![가-힣])({_ALT(_DAYS)})")
_KO_NATIVE = re.compile(rf"(?<![가-힣])(?:(?P<t>{_ALT(_NATIVE_TENS)})(?P<o>{_ALT(_NATIVE_ONES)})?"
                        rf"|(?P<o2>{_ALT(_NATIVE_ONES)})(?=\s|$|[^가-힣]))")
_KO_SINO = re.compile(r"(?<![가-힣])[일이삼사오육칠팔구]?(?:[십백천만][일이삼사오육칠팔구]?)+")


def _sino_value(word):
    total, section, digit = 0, 0, 0
    for ch in word:
        if ch in _SINO_DIGIT:
            digit = _SINO_DIGIT[ch]
        elif ch == "만":
            total += (section + digit or 1) * 10000
            section, digit = 0, 0
        else:
            section += (digit or 1) * _SINO_MUL[ch]
            digit = 0
    return total + section + digit


def ko_numerals_to_digits(text):
    """「구백 건」 「열흘」 「마흔한 명」 → 「900 건」 「10일」 「41 명」. 받침 · 조사에 붙은 말은 앞쪽만 본다(어림)."""
    text = _KO_DAYS.sub(lambda m: f"{_DAYS[m.group(1)]}일", text)
    text = _KO_SINO.sub(lambda m: str(_sino_value(m.group(0))) if len(m.group(0)) > 1 else m.group(0), text)  # 「만들기」의 만 하나는 안 친다

    def native(m):
        if m.group("t"):
            return str(_NATIVE_TENS[m.group("t")] + (_NATIVE_ONES[m.group("o")] if m.group("o") else 0))
        return str(_NATIVE_ONES[m.group("o2")])
    return _KO_NATIVE.sub(native, text)


def a_has_b_value(new, seed, line, kind="num"):
    """a 줄에 b 의 새 값이 들어갔나 — 그대로(「900건」) 또는 수 자리면 한글 수(「구백 건」)로도. 씨앗에도 있던 수는 안 친다."""
    if has_value(new, line) and not has_value(new, seed):
        return True
    m = re.search(_NUM_CORE, new) if kind == "num" else None
    if not m:
        return False
    num = str(int(_parse_num(m.group(0)))) if _parse_num(m.group(0)).is_integer() else m.group(0)
    return has_value(num, ko_numerals_to_digits(line)) and not has_value(num, ko_numerals_to_digits(seed))


def check_line(seed, line, role, row, scrubber):
    """한 줄의 기계 검사 → 까닭 목록 (빈 목록이면 통과)."""
    if line is None or line == "":
        return ["empty"]
    if not isinstance(line, str):
        return ["not_text"]
    out = []
    n = chunks(seed, line)
    if n == 0:
        out.append("same_as_seed")
    elif n > LIMITS[role]:
        out.append("chunks")
    if n and changed_tokens(seed, line) > token_limit(role, row):
        out.append("tokens")
    new = row.get("new")
    if role == "b" and new and not has_value(new, line):
        out.append("new_missing")
    if role == "b" and new and has_value(new, seed):
        out.append("new_in_seed")
    if role == "a" and new and a_has_b_value(new, seed, line, (row.get("slot") or {}).get("kind", "num")):
        out.append("a_has_b_value")
    if role == "d" and row.get("d") and not has_value(row["d"]["time_new"], line):
        out.append("time_missing")
    ratio = len(line) / max(1, len(seed))
    if not 0.5 <= ratio <= 1.6 or len(line) < 10:
        out.append("length")
    if is_question(line):
        out.append("question")
    if (row.get("slot") or {}).get("kind") not in ("onoff", "bound") and len(_NEG.findall(line)) > len(_NEG.findall(seed)):
        out.append("negation")
    if sum(scrubber.hits({"evidence": line}).values()):
        out.append("scrub")
    return out


def check_rows(rows, scrubber):
    """생성 결과(lines 칸의 b · a · d) 또는 입력 줄(drafts) → (줄별 결과, 셈)."""
    count = collections.Counter()
    results = []
    for row in rows:
        lines = row.get("lines") or row.get("drafts") or {}
        lines = dict(lines) if isinstance(lines, dict) else {}
        verdict = {}
        for role in row.get("want_lines") or [r for r, v in lines.items() if v is not None]:
            line = lines.get(role)
            if line is None and role == "a" and row.get("a_mode") == "llm" and "lines" not in row:
                count["a_llm"] += 1
                continue
            why = check_line(row["seed"], line, role, row, scrubber)
            verdict[role] = why
            count[f"{role}_total"] += 1
            count[f"{role}_pass"] += not why
            for w in why:
                count[f"{role}_{w}"] += 1
        ok = all(not v for v in verdict.values())
        count["rows"] += 1
        count["rows_pass"] += ok
        results.append({"seed_id": row.get("seed_id"), "pass": ok, "why": {k: v for k, v in verdict.items() if v}})
    return results, dict(count)


# ---------------------------------------------------------------- 명령

def _default_studio():
    return os.path.dirname(os.path.dirname(HERE))


def _default_judge_root():
    return os.path.join(os.path.expanduser("~"), ".judgemodel")


def _scrubber(a):
    g1 = os.path.join(a.judge_root, "build", "g1")
    names = a.names if a.names is not None else [p for p in [os.path.join(g1, "names.txt")] if os.path.exists(p)]
    games = a.game_names if a.game_names is not None else [p for p in [os.path.join(g1, "game-names.txt")]
                                                            if os.path.exists(p)]
    # 이름이 든 씨앗은 바꿔 쓰지 않고 버린다 — 게임 이름도 버릴 목록에 같이 넣는다.
    return scrub.Scrubber(scrub.read_names(names + games), ())


def _guard_out(path, allow_git):
    if lb.inside_git(path) and not allow_git:
        sys.exit("--out 이 git 안이다 — 씨앗 본문은 git 밖에 둔다")


def cmd_count(a):
    _guard_out(a.out, a.allow_git)
    forbid = SeedForbid(a.judge_root, a.studio)
    if forbid.missing and not a.allow_missing:
        sys.exit(f"금지 목록 파일이 없다 : {forbid.missing}")
    rows = read_sources(a.studio)
    seeds, report = count_seeds(rows, forbid, _scrubber(a))
    lb.write_jsonl(os.path.join(a.out, "seeds-all.jsonl"), seeds)
    report.update({"sentences": len(rows), "forbid_docs": len(forbid.docs), "forbid_mem_ids": len(forbid.mem_ids),
                   "forbid_mem_groups": len(forbid.mem_groups), "forbid_texts": len(forbid.texts),
                   "missing": forbid.missing, "target": TARGET})
    lb.write_json(os.path.join(a.out, "count.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    if report["count"].get("kept", 0) < TARGET:
        print(f"주의 : 쓸 씨앗 {report['count'].get('kept', 0)} < {TARGET} — 씨앗 몫을 줄여야 한다(설계 7절)", file=sys.stderr)
    return 0


def refresh_slots(seeds):
    """seeds-all 의 자리를 지금 규칙(find_slots)으로 다시 찾는다 — 자리 규칙을 고쳐도 다시 세기(count) 없이 make 만 돌리면 된다."""
    out = []
    for row in seeds:
        slots = find_slots(row["text"])
        if slots and len(slots) <= MAX_SLOTS:
            out.append(dict(row, slots=slots))
    return out


def cmd_make(a):
    _guard_out(a.out, a.allow_git)
    seeds = refresh_slots(lb.read_jsonl(os.path.join(a.out, "seeds-all.jsonl")))
    rows = choose(seeds, a.n, a.seed)
    rng = random.Random(a.seed + 1)
    sample = sorted(rng.sample(rows, min(SAMPLE, len(rows))), key=lambda r: r["seed_id"])
    lb.write_jsonl(os.path.join(a.out, "input.jsonl"), rows)
    lb.write_jsonl(os.path.join(a.out, "sample50.jsonl"), sample)
    with open(os.path.join(a.out, "seed-groups.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# 씨앗으로 쓴 묶음 — g3 예약 · 평가 셋 금지 목록에 더한다\n")
        f.writelines(g + "\n" for g in sorted({r["group"] for r in rows}))
    report = {"seeds": len(seeds), "picked": len(rows), "sample": len(sample),
              "by_kind": dict(collections.Counter(r["slot"]["kind"] for r in rows)),
              "by_source": dict(collections.Counter(r["source"] for r in rows)),
              "with_d": sum(1 for r in rows if r["d"]), "a_rule": sum(1 for r in rows if r["a_mode"] == "rule"),
              "groups": len({r["group"] for r in rows})}
    lb.write_json(os.path.join(a.out, "make.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_check(a):
    rows = lb.read_jsonl(a.inp)
    results, count = check_rows(rows, _scrubber(a))
    if a.out:
        _guard_out(a.out, a.allow_git)
        lb.write_jsonl(a.out, results)
    print(json.dumps(count, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="실물 문장 씨앗 → 한 자리 바꾼 b · a · d 생성 입력")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("count", "make", "check"):
        p = sub.add_parser(name)
        p.add_argument("--judge-root", default=_default_judge_root())
        p.add_argument("--names", nargs="*", help="버릴 이름 목록 (기본 <judge-root>/build/g1/names.txt)")
        p.add_argument("--game-names", nargs="*", help="버릴 게임 이름 목록 (기본 <judge-root>/build/g1/game-names.txt)")
        p.add_argument("--allow-git", action="store_true", help="시험용 — git 안에 써도 된다")
    for name in ("count", "make"):
        sub.choices[name].add_argument("--out", required=True)
    sub.choices["count"].add_argument("--studio", default=_default_studio(), help="스튜디오 뿌리")
    sub.choices["count"].add_argument("--allow-missing", action="store_true", help="시험용 — 금지 목록 파일이 없어도 돈다")
    sub.choices["make"].add_argument("--n", type=int, default=TARGET)
    sub.choices["make"].add_argument("--seed", type=int, default=SEED)
    sub.choices["check"].add_argument("--in", dest="inp", required=True)
    sub.choices["check"].add_argument("--out")
    a = parser.parse_args(argv)
    return {"count": cmd_count, "make": cmd_make, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
