"""layer_report — 학습 층 점검 보고 둘 (K7 설계 2-3 「새 ①」 · 1-3 꼴 표). 막지 않는다 — 보고만.

    python -I layer_report.py --recipe <학습 층 …> --base-recipe <K 처방 층 …> --dev <dev-cmp-g9.jsonl> \
        [--protocol ../measure/protocol-g9.json] --out <git 밖 폴더> [--name layer-report]

- 새 보고 ① (원천 × 새 낱말 구간) : 학습 줄마다 주장의 새 낱말 몫(E4 1-1 규칙)을 구간 B0~B5(E4 1-2 끝값
  .1 · .25 · .4 · .6 · .8)로 나눠 원천마다 센다. 덧층 = --recipe 중 --base-recipe 에 없는 층(K7 이면 train-g4-bal ·
  train-g4-klb). 한 원천의 덧층 줄 80% 넘게(> .8)가 B3 이상이면 「표시」 — 그 원천 b 를 지킬 줄이 같은 구간에
  있나를 사람이 본다(b_all 칸). 칸은 assemble_g4.cell_key 와 같고 cd 줄은 뺀다(E4 1-3).
- 꼴 표 (원천 × 언어 × 꼴 × 칸) : --recipe 층 파일마다와 --dev 의 form 칸을 센다. 생성기 꼴 여부는 규약
  generator_forms. 언어 × 칸마다 생성기 꼴 몫 · 가장 많은 꼴도 낸다.

출력은 <out>/<name>.json · <name>.md (쌍 본문은 안 담는다). 파일 없음 · base 가 recipe 밖 · out 이 git 안이면 exit 2.
정렬은 모두 이름 차례라 PYTHONHASHSEED 와 상관없이 같은 바이트가 나온다.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemble_g4 import CELL_OF, cell_key  # noqa: E402
from label_batches import inside_git, read_jsonl, write_json  # noqa: E402
from layer_g4d import bin_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.join(HERE, "..", "measure", "protocol-g9.json")
TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")     # E4 1-1 (B-2 b2_diag2.novel 과 같은 낱말 나누기)
HANGUL = re.compile(r"[가-힣]")
BAND_EDGES = (0.1, 0.25, 0.4, 0.6, 0.8)           # E4 1-2 박제 끝값
BANDS = tuple(f"B{i}" for i in range(len(BAND_EDGES) + 1))
HIGH_FROM = 3                                    # B3 이상 = 새 낱말 .4 이상
FLAG_SHARE = 0.8                                 # 덧층 줄 중 B3 이상 몫이 이 값을 넘으면 표시 (설계 2-3 「80% 넘게」)
CELLS = ("a", "b", "c", "d", "x")                # x = cd · 그 밖 칸
NO_FORM = "(none)"
NO_LANG = "?"


def novel_share(evidence, claim):
    """주장 덩어리 중 근거에 없는 몫 (E4 1-1). 한글은 앞 두 음절이 근거(소문자)에 있으면 있음, 영어 · 숫자는 소문자 그대로."""
    tokens = TOKEN.findall(claim or "")
    if not tokens:
        return 0.0
    ev = (evidence or "").lower()
    missing = 0
    for tok in tokens:
        probe = tok[:2] if HANGUL.match(tok) else tok.lower()
        if probe not in ev:
            missing += 1
    return missing / len(tokens)


def band_of(value):
    return BANDS[bin_of(value, BAND_EDGES)]


def _zero():
    return {b: 0 for b in BANDS}


def _lang(row):
    return str(row.get("lang") or NO_LANG)


def source_bands(base_rows, add_rows):
    """새 보고 ① — 원천마다 기본 층 · 덧층 줄의 구간 분포, 덧층 B3 이상 몫과 표시, 학습 전체 b 줄의 구간 분포.
    lang = 부분(all · base · add) × 언어 × 구간 줄 수 (E4 2-1 과 견줄 수 있게)."""
    per = collections.defaultdict(lambda: {"base": _zero(), "add": _zero(), "b_all": _zero()})
    lang = {"all": collections.defaultdict(_zero), "base": collections.defaultdict(_zero),
            "add": collections.defaultdict(_zero)}
    dropped = 0
    for part, rows in (("base", base_rows), ("add", add_rows)):
        for row in rows:
            key = cell_key(row)
            if key is None:
                dropped += 1
                continue
            domain, cell = key
            band = band_of(novel_share(row.get("evidence"), row.get("claim")))
            per[domain][part][band] += 1
            if cell == "b":
                per[domain]["b_all"][band] += 1
            lang[part][_lang(row)][band] += 1
            lang["all"][_lang(row)][band] += 1
    out_rows, flagged = [], []
    for domain in sorted(per):
        d = per[domain]
        add_n = sum(d["add"].values())
        high = sum(d["add"][b] for b in BANDS[HIGH_FROM:])
        share = high / add_n if add_n else None
        flag = share is not None and share > FLAG_SHARE
        if flag:
            flagged.append(domain)
        out_rows.append({"domain": domain, "base": d["base"], "add": d["add"], "b_all": d["b_all"],
                         "base_n": sum(d["base"].values()), "add_n": add_n, "add_high": high,
                         "add_high_share": share, "flag": flag})
    return {"edges": list(BAND_EDGES), "high_from": BANDS[HIGH_FROM], "flag_share": FLAG_SHARE,
            "rows": out_rows, "flagged": flagged, "dropped_cell": dropped,
            "lang": {part: {k: v[k] for k in sorted(v)} for part, v in lang.items()}}


def form_rows(rows, generator_forms):
    """꼴 표 — (원천, 언어, 꼴) 마다 칸 a~d · x(cd 와 그 밖) 줄 수. 원천이 없으면 mem, 꼴이 없으면 (none)."""
    counts = collections.defaultdict(lambda: dict.fromkeys(CELLS, 0))
    for row in rows:
        domain = str(row.get("domain") or "mem")
        form = str(row.get("form") or NO_FORM)
        cell = CELL_OF.get(str(row.get("want") or ""), "x")
        counts[(domain, _lang(row), form)][cell] += 1
    out = []
    for (domain, lang, form) in sorted(counts):
        c = counts[(domain, lang, form)]
        out.append({"domain": domain, "lang": lang, "form": form, "gen": form in generator_forms,
                    **c, "n": sum(c.values())})
    return out


def gen_share(table):
    """언어 × 칸(a~d)마다 줄 수 · 생성기 꼴 줄 수 · 몫 · 가장 많은 꼴(같으면 이름 차례 앞)."""
    acc = collections.defaultdict(lambda: {"n": 0, "gen": 0, "forms": collections.Counter()})
    for r in table:
        for cell in "abcd":
            if r[cell]:
                slot = acc[(r["lang"], cell)]
                slot["n"] += r[cell]
                slot["gen"] += r[cell] if r["gen"] else 0
                slot["forms"][r["form"]] += r[cell]
    out = []
    for (lang, cell) in sorted(acc):
        slot = acc[(lang, cell)]
        top_form, top_n = sorted(slot["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        out.append({"lang": lang, "cell": cell, "n": slot["n"], "gen": slot["gen"],
                    "gen_share": slot["gen"] / slot["n"], "top_form": top_form, "top_n": top_n})
    return out


# ---------------------------------------------------------------- md


def _share(x):
    return "—" if x is None else f"{x:.3f}"


def bands_md(bands):
    out = [f"## 새 보고 ① 원천 × 새 낱말 구간 (끝값 {' · '.join(str(e) for e in bands['edges'])} · 덧층 줄 중 "
           f"{bands['high_from']} 이상 몫 > {bands['flag_share']} 이면 표시)", "",
           f"- 표시 원천 : {', '.join(bands['flagged']) or '없음'} · cd 로 뺀 줄 {bands['dropped_cell']}", "",
           "| 원천 | 기본 줄 | 덧층 줄 | 덧층 " + " · ".join(BANDS) + f" | 덧층 {bands['high_from']}+ 몫 | 학습 b "
           + " · ".join(BANDS) + " | 판정 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in bands["rows"]:
        add = " · ".join(str(r["add"][b]) for b in BANDS)
        b_all = " · ".join(str(r["b_all"][b]) for b in BANDS)
        out.append(f"| {r['domain']} | {r['base_n']} | {r['add_n']} | {add} | {_share(r['add_high_share'])} | {b_all} | "
                   f"{'**표시**' if r['flag'] else ''} |")
    out += ["", "### 언어 × 구간 줄 수 (all = 기본 + 덧층 · replay 층은 안 셈)", "",
            "| 부분 | 언어 | " + " | ".join(BANDS) + " | 합 |", "| --- | --- | " + " | ".join("---" for _ in BANDS) + " | --- |"]
    for part in ("all", "base", "add"):
        for lang, counts in bands["lang"][part].items():
            out.append(f"| {part} | {lang} | " + " | ".join(str(counts[b]) for b in BANDS) + f" | {sum(counts.values())} |")
    return out


def forms_md(name, block):
    out = [f"### {name} ({block['rows']}줄)", "", "| 언어 | 칸 | 줄 | 생성기 꼴 | 몫 | 가장 많은 꼴 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for g in block["gen_share"]:
        out.append(f"| {g['lang']} | {g['cell']} | {g['n']} | {g['gen']} | {g['gen_share']:.3f} | {g['top_form']} {g['top_n']} |")
    out += ["", "| 원천 | 언어 | 꼴 | 생성기 | a | b | c | d | x | 합 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in block["table"]:
        out.append(f"| {r['domain']} | {r['lang']} | {r['form']} | {'G' if r['gen'] else ''} | "
                   + " | ".join(str(r[c]) for c in CELLS) + f" | {r['n']} |")
    return out + [""]


def report_md(doc):
    out = [f"# 층 점검 보고 — {doc['name']}", "",
           f"- 학습 : {', '.join(doc['recipe'])} · 기본 : {', '.join(doc['base_recipe'])} · 덧층 : {', '.join(doc['add_layers'])}",
           f"- dev : {doc['dev']} · 생성기 꼴 {len(doc['generator_forms'])}개 (규약 {doc['protocol']})",
           "- 막지 않는 보고다. 새 낱말 몫은 E4 1-1 규칙(주장 낱말 중 근거에 없는 몫), 칸 x = cd · 그 밖", ""]
    out += bands_md(doc["bands"])
    out += ["", "## 꼴 표 원천 × 언어 × 꼴 × 칸 (G = 규약 generator_forms)", ""]
    for name in sorted(doc["forms"]):
        out += forms_md(name, doc["forms"][name])
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------- 명령


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _generator_forms(path):
    with open(path, encoding="utf-8") as f:
        return sorted(json.load(f).get("generator_forms") or [])


def build_report(a):
    """인자 → 보고 dict. 파일 없음 · base 가 recipe 밖이면 ValueError."""
    missing = [p for p in list(a.recipe) + list(a.base_recipe) + [a.dev, a.protocol] if not os.path.exists(p)]
    if missing:
        raise ValueError(f"파일이 없다 {missing}")
    recipe = {_norm(p): p for p in a.recipe}
    outside = [p for p in a.base_recipe if _norm(p) not in recipe]
    if outside:
        raise ValueError(f"--base-recipe 가 --recipe 에 없다 {outside}")
    base_keys = {_norm(p) for p in a.base_recipe}
    layers = {p: read_jsonl(p) for p in a.recipe}
    base_rows = [r for p in a.recipe if _norm(p) in base_keys for r in layers[p]]
    add_paths = [p for p in a.recipe if _norm(p) not in base_keys]
    add_rows = [r for p in add_paths for r in layers[p]]
    gen = _generator_forms(a.protocol)
    gen_set = set(gen)
    forms = {}
    for path, rows in list(layers.items()) + [(a.dev, read_jsonl(a.dev))]:
        table = form_rows(rows, gen_set)
        forms[_stem(path)] = {"rows": len(rows), "table": table, "gen_share": gen_share(table)}
    return {"name": a.name, "recipe": [os.path.basename(p) for p in a.recipe],
            "base_recipe": [os.path.basename(p) for p in a.base_recipe],
            "add_layers": [os.path.basename(p) for p in add_paths], "dev": os.path.basename(a.dev),
            "protocol": os.path.basename(a.protocol), "generator_forms": gen,
            "bands": source_bands(base_rows, add_rows), "forms": forms}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="층 점검 보고 — 원천 × 새 낱말 구간 · 원천 × 꼴 × 칸")
    p.add_argument("--recipe", nargs="+", required=True, help="학습 층 파일 전체 (K 처방 + 덧층)")
    p.add_argument("--base-recipe", nargs="+", required=True, help="K 처방 층 — 나머지 recipe 가 덧층")
    p.add_argument("--dev", required=True, help="꼴 표에 같이 셀 dev 셋 (dev-cmp-g9.jsonl)")
    p.add_argument("--protocol", default=PROTOCOL, help="generator_forms 를 읽을 규약 (기본 measure/protocol-g9.json)")
    p.add_argument("--out", required=True, help="출력 폴더 (git 밖)")
    p.add_argument("--name", default="layer-report", help="출력 파일 이름 앞부분")
    a = p.parse_args(argv)
    if inside_git(a.out):
        print("멈춤 : --out 이 git 안이다 — 층 보고는 git 밖에 둔다", file=sys.stderr)
        return 2
    try:
        doc = build_report(a)
    except ValueError as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    write_json(os.path.join(a.out, f"{a.name}.json"), doc)
    md = report_md(doc)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.name}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(json.dumps({"out": os.path.join(a.out, a.name), "flagged": doc["bands"]["flagged"],
                      "add_layers": doc["add_layers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
