"""label_batches — Opus 라벨 묶음 만들기 · 답 합치기 · 합의 · κ · 정답셋 뽑기 · 사람 목록.

비공개 설계 2-4 · 3-3, 프롬프트 `../Docs/Guide/라벨프롬프트-g1.md`.
판정자는 `<out>/batches/` 만 본다. 대응표 `<out>/map/` 는 판정자에게 안 주고, `--out` 은 git 밖이어야 한다.

    python -I label_batches.py traps  --dev-r2 <dev-r2.jsonl> --ruler4 <ruler4 폴더> --pool <trap-pool.jsonl>
    python -I label_batches.py make   --cands <후보.jsonl> --out <묶음 폴더> --mode train|eval [--traps <trap-pool.jsonl>] [--judges 5]
    python -I label_batches.py third  --out <묶음 폴더>          # decide 가 고른 갈린 쌍 → 셋째 묶음
    python -I label_batches.py merge  --out <묶음 폴더> [--exempt-trap <함정id> …] # answers/<묶음>-j<판정자>.jsonl → votes.jsonl
    python -I label_batches.py decide --out <묶음 폴더> --mode train|eval|synth   # synth : 5 중 4 가 hint 와 같을 때만
    python -I label_batches.py kappa  --out <묶음 폴더>
    python -I label_batches.py gold-pick   --out <dev-cmp 묶음 폴더> [<묶음 폴더> …] --set <dev-cmp.jsonl> [--dest <폴더>]
    python -I label_batches.py gold-score  --out <dev-cmp 묶음 폴더> [<묶음 폴더> …] --gold <gold 묶음 폴더> [--cands <gold-cands.jsonl>]
    python -I label_batches.py gold-score  --dev-from <gold-cands.jsonl> --gold <gold 묶음 폴더>        # dev 칸이 묶음 폴더 밖에서 정해진 쌍
    python -I label_batches.py review-todo --out <묶음 폴더> [<묶음 폴더> …] --set <쌍.jsonl> --dest ~/.judgemodel/sets/g1
    python -I label_batches.py assemble --set <셋> --dirs <묶음 폴더> [<묶음 폴더> …] --dest ~/.judgemodel/sets/g1 [--allow-mixed-guide]
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys

SEED = 20261011
BATCH_SIZE = 100
CAREFUL_SIZE = 20
TRAP_RATE = 0.10
TRAP_PASS = 0.80
NONE_MAX = 0.05
KAPPA_CD_MIN = 0.60
LABELS = ("a", "b", "c", "d")
YES, NO, DASH = "예", "아니오", "-"
WEAK_WORDS = ("근거 약함", "근거가 약")
GUIDE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "Docs", "Guide", "라벨기준support-v5.md")

# 판정자 k 의 예시 차례 · 쌍 차례 (라벨프롬프트-g1 1-3). 값 = (자리 차례, 자리 안 보기 차례)
VARIANTS = {1: ("keep", "keep"), 2: ("reverse", "keep"), 3: ("shuffle", "reverse"),
            4: ("reverse", "reverse"), 5: ("shuffle", "keep")}
JUDGE_MAX = len(VARIANTS)
# 합성 쌍(synth.py) 합의 : 판정 k 중 SYNTH_NEED 이상이 의도 칸(hint)과 같을 때만 남긴다 (설계 5절 ⑤)
SYNTH_NEED = 4


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def inside_git(path):
    """path 나 그 위 폴더에 .git 이 있으면 True — 쌍 본문 · 대응표가 git 안으로 새는 것을 막는다."""
    cur = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return True
        parent = os.path.dirname(cur)
        if parent == cur:
            return False
        cur = parent


# ---------------------------------------------------------------- 함정 풀

def build_trap_pool(dev_rows, ruler4_inputs, ruler4_gold):
    """사람 라벨이 확실한 쌍만 함정으로 — a : same·long 지지 · b : flip 아닌 반대 · c : xscope 무관 · d : 사람 감사 d.

    flip 반대는 뜻이 안 뒤집힌 쌍이 섞여 함정으로 못 쓴다 (2026-10-10 확인).
    """
    pool = []
    for row in dev_rows:
        kind, want = row.get("kind"), row.get("want")
        gold = None
        if want == "support" and kind in ("same", "long"):
            gold = "a"
        elif want == "contradict" and kind != "flip":
            gold = "b"
        elif want == "unrelated" and kind == "xscope":
            gold = "c"
        if gold:
            pool.append({"evidence": row["evidence"], "claim": row["claim"], "gold": gold, "from": "dev-r2"})
    gold_src = {g["q"]: g for g in ruler4_gold}
    for row in ruler4_inputs:
        g = gold_src.get(row["q"])
        if g and g.get("src") == "audit" and g.get("gold") == "d":
            pool.append({"evidence": row["evidence"], "claim": row["claim"], "gold": "d", "from": "audit"})
    for i, row in enumerate(pool):
        row["tid"] = f"t{i:04d}"
    return pool


def cmd_traps(a):
    dev = read_jsonl(a.dev_r2)
    inputs, gold = [], []
    if a.ruler4 and os.path.exists(os.path.join(a.ruler4, "ruler4-gold.jsonl")):
        inputs = read_jsonl(os.path.join(a.ruler4, "ruler4-input.jsonl"))
        gold = read_jsonl(os.path.join(a.ruler4, "ruler4-gold.jsonl"))
    pool = build_trap_pool(dev, inputs, gold)
    if inside_git(a.pool):
        sys.exit("함정 풀은 git 밖에 쓴다")
    write_jsonl(a.pool, pool)
    print(json.dumps({"traps": len(pool), "by_label": dict(collections.Counter(r["gold"] for r in pool))},
                     ensure_ascii=False))


# ---------------------------------------------------------------- 안내 글

def guide_sections(guide_text):
    """support-v5 의 1 ~ 3 절 (4 절 앞까지)을 고치지 않고 떼어 낸다."""
    start = guide_text.find("## 1.")
    end = guide_text.find("## 4.")
    if start < 0 or end < 0:
        raise ValueError("support-v5 에서 1 ~ 3 절을 못 찾음")
    return guide_text[start:end].rstrip() + "\n"


def reorder_places(section_text, variant, seed):
    """3 절 표의 자리 차례 · 자리 안 보기 차례만 바꾼다. 다른 줄은 그대로 둔다."""
    lines = section_text.split("\n")
    marks = [i for i, line in enumerate(lines) if line.startswith("## 3.")]
    if not marks:
        return section_text
    table = []
    for i in range(marks[0] + 1, len(lines)):
        if lines[i].startswith("|"):
            table.append(i)
        elif table:
            break
    body = table[2:]   # 머리 줄 · 구분 줄 뒤
    if not body:
        return section_text
    places = []
    for i in body:
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if cells[0]:
            places.append({"name": cells[0], "rule": cells[1], "examples": [cells[2]]})
        else:
            places[-1]["examples"].append(cells[2])
    order_mode, example_mode = VARIANTS[variant]
    if order_mode == "reverse":
        places.reverse()
    elif order_mode == "shuffle":
        random.Random(seed).shuffle(places)
    out_rows = []
    for place in places:
        examples = list(place["examples"])
        if example_mode == "reverse":
            examples.reverse()
        out_rows.append(f"| {place['name']} | {place['rule']} | {examples[0]} |")
        out_rows.extend(f"| | | {ex} |" for ex in examples[1:])
    first, last = body[0], body[-1]
    return "\n".join(lines[:first] + out_rows + lines[last + 1:])


LABEL_ROLE = ("너는 두 글의 관계를 칸 넷(a 지지 · b 반대 · c 무관 · d 비슷한 계열) 중 하나로 가르는 라벨러다.\n"
              "쌍마다 「근거(evidence)」와 「주장(claim)」이 있다. 근거가 주장에 대해 무엇을 말해 주는지를 본다.")
CAREFUL_ROLE = ("너는 두 글의 관계 라벨을 다시 보는 검토자다. 칸은 a 지지 · b 반대 · c 무관 · d 비슷한 계열 넷이다.\n"
                "쌍마다 따로, 천천히 본다. 앞 쌍의 답에 끌리지 않는다.")
LABEL_QS = ("쌍마다 아래 셋을 차례로 답한다. 답은 「예」 또는 「아니오」와 까닭 한 마디(20자 안팎)다.\n"
            "- Q1 같은 대상인가?  아니오면 Q2 · Q3 은 「-」 로 두고 c.\n"
            "- Q2 함께 참일 수 없는 말이 있나?  예면 Q3 은 「-」 로 두고 b.\n"
            "- Q3 근거가 맞다고 해 주는 말이 주장에 하나라도 있나?  예면 a, 아니오면 d.\n"
            "d 를 고르기 전에 숨은 부분 지지(다른 말 · 다른 단위 · 이름 · 날짜)를 한 번 더 본다.")
CAREFUL_QS = ("쌍마다 「앞 문맥」 · 「뒤 문맥」을 준다. 문맥은 근거가 무엇을 가리키는지(대명사 · 생략된 주어 · 때)를\n"
              "알아내는 데만 쓴다. 판정은 근거와 주장으로만 한다 — 문맥에만 있는 말로 a 나 b 를 주지 않는다.\n\n"
              "Q1 → Q2 → Q3 을 차례로 답한다. 값마다 「예/아니오/-」 뒤에 까닭을 한 문장(40자 안팎)으로 붙인다.\n"
              "d 를 고르기 전에 숨은 부분 지지를 한 번 더 보고, 본 결과를 q3 까닭에 적는다.\n"
              "칸은 a · b · c · d 중 하나. cd 는 쓰지 않는다. 정말 못 고르겠으면 칸을 고르고 \"sure\": \"낮음\" 을 적는다.")
CD_TRAIN = "칸은 a · b · c · d 중 하나만 고른다. Q1 을 정말 못 고를 때만 cd 를 쓸 수 있다."
CD_EVAL = "칸은 a · b · c · d 중 하나만 고른다. cd 는 쓰지 않는다."
FORBID = ("- 이 안내와 아래 쌍 파일 말고 다른 파일을 열지 말라. 다른 쌍의 답을 보지 말라.\n"
          "- 쌍이 어디서 왔는지(원천 · 자료 이름 · 묶음 순서)를 짐작하지 말라. 글에 적힌 말만 보고 가린다.\n"
          "- 칸 비율을 맞추려 하지 말라. 한 묶음에 한 칸이 몰려도 된다.")
LABEL_OUT = ('쌍 하나에 한 줄. JSON 한 줄 말고 아무것도 쓰지 않는다. 까닭은 q1 · q2 · q3 값 뒤에 「 — 」로 붙인다.\n'
             '{"q": "q04121", "q1": "예 — 같은 앨범", "q2": "아니오", "q3": "아니오 — 출시일만 말함", "label": "d"}')
CAREFUL_OUT = ('쌍 하나에 한 줄. JSON 한 줄 말고 아무것도 쓰지 않는다. "sure" 는 높음 · 보통 · 낮음.\n'
               '{"q": "q04121", "q1": "예 — 둘 다 같은 앨범의 출시를 말한다", "q2": "아니오 — 날짜와 회사는 서로 '
               '부딪치지 않는다", "q3": "아니오 — 주장의 회사 이름을 근거가 말하지 않는다", "label": "d", "sure": "보통"}')


def render_guide(criteria, style, mode, pairs_file, answer_file, count):
    careful = style == "careful"
    cd_line = CD_EVAL
    if mode == "train" and not careful:
        cd_line = CD_TRAIN
    parts = [
        "[① 역할]", CAREFUL_ROLE if careful else LABEL_ROLE, "",
        "[② 기준 전문]", "아래 기준을 그대로 따른다. 기준에 없는 자리는 Q1 → Q2 → Q3 차례로만 가린다.",
        "---", criteria.rstrip(), "---", "",
        "[③ 예/아니오 셋]", CAREFUL_QS if careful else LABEL_QS, "",
        "[④ 칸 하나]", cd_line, "",
        "[⑤ 금지]", FORBID, "",
        "[출력]", CAREFUL_OUT if careful else LABEL_OUT, "",
        "[쌍]", f"쌍 {count} 개가 이 안내와 같은 폴더의 `{pairs_file}` 에 줄마다 하나씩 있다.",
        f"답은 쌍 차례대로 묶음 폴더 뿌리의 `{answer_file}` 에 쓴다 (부르는 쪽이 다른 자리를 주면 그쪽).", "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- 묶음 만들기

def trap_quota(n_trap, labels, batch_index):
    """함정 수를 칸에 고르게 — 남는 몫은 묶음마다 돌아가며 준다."""
    if not labels:
        return {}
    base, extra = divmod(n_trap, len(labels))
    quota = {}
    for i, label in enumerate(labels):
        shifted = (i - batch_index) % len(labels)
        quota[label] = base + (1 if shifted < extra else 0)
    return quota


def pick_traps(pool_by_label, n_trap, batch_index, rng):
    labels = [lab for lab in LABELS if pool_by_label.get(lab)]
    picked = []
    short = 0
    for label, want in trap_quota(n_trap, labels, batch_index).items():
        rows = pool_by_label[label]
        take = min(want, len(rows))
        picked.extend(rng.sample(rows, take))
        short += want - take
    rest = [r for lab in labels for r in pool_by_label[lab] if r not in picked]
    picked.extend(rng.sample(rest, min(short, len(rest))))
    return picked


def judge_order(rows, judge, batch_index):
    out = list(rows)
    if judge != 1:
        random.Random(SEED * 10 + judge * 1000 + batch_index).shuffle(out)
    return out


def fresh_qids(rng, count, used):
    out = []
    while len(out) < count:
        q = f"q{rng.randrange(10000, 100000):05d}"
        if q not in used:
            used.add(q)
            out.append(q)
    return out


def emit_batch(out_dir, name, judges, rows, criteria, style, mode, batch_index):
    """판정자마다 쌍 파일(jsonl) 하나와 안내 txt 하나를 쓴다. 판정자마다 예시 · 쌍 차례가 다르다."""
    keys = ("q", "before", "evidence", "after", "claim") if style == "careful" else ("q", "evidence", "claim")
    for judge in judges:
        stem = f"{name}-j{judge}"
        ordered = judge_order(rows, judge, batch_index)
        shown = [{k: r.get(k, "없음") for k in keys} for r in ordered]
        write_jsonl(os.path.join(out_dir, "batches", stem + ".jsonl"), shown)
        text = render_guide(reorder_places(criteria, judge, SEED + judge * 7 + batch_index), style, mode,
                            stem + ".jsonl", f"answers/{stem}.jsonl", len(shown))
        with open(os.path.join(out_dir, "batches", stem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def build_batches(cands, traps, mode, style, size, trap_rate, seed, prefix, used_q, judges):
    """후보 → [(묶음 이름, [쌍 줄], [대응 줄])] — 쌍 줄에는 q · 글만, 대응 줄에는 원래 id · 함정 정답."""
    rng = random.Random(seed)
    order = list(cands)
    rng.shuffle(order)
    per_real = size - round(size * trap_rate) if traps else size
    pool_by_label = collections.defaultdict(list)
    for t in traps:
        pool_by_label[t["gold"]].append(t)
    batches = []
    for b, start in enumerate(range(0, len(order), per_real)):
        real = order[start:start + per_real]
        n_trap = 0
        if traps:
            n_trap = max(1, round(len(real) * trap_rate / (1 - trap_rate)))
        trap_rows = pick_traps(pool_by_label, n_trap, b, rng)
        qids = fresh_qids(rng, len(real) + len(trap_rows), used_q)
        name = f"{prefix}{b + 1:02d}"
        items, mapping = [], []
        for q, row in zip(qids, real):
            items.append({"q": q, **{k: row.get(k) for k in ("evidence", "claim", "before", "after") if k in row}})
            mapping.append({"q": q, "batch": name, "id": row.get("id"), "trap": False,
                            "domain": row.get("domain", ""), "lang": row.get("lang", "")})
        for q, t in zip(qids[len(real):], trap_rows):
            items.append({"q": q, "evidence": t["evidence"], "claim": t["claim"]})
            mapping.append({"q": q, "batch": name, "id": t["tid"], "trap": True, "gold": t["gold"]})
        rng.shuffle(items)
        batches.append((name, items, mapping, judges))
    return batches


def load_manifest(out_dir):
    with open(os.path.join(out_dir, "map", "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def load_map(out_dir):
    path = os.path.join(out_dir, "map", "map.jsonl")
    return read_jsonl(path) if os.path.exists(path) else []


def cmd_make(a):
    if inside_git(a.out) and not a.allow_git:
        sys.exit("--out 이 git 안이다 — 쌍 본문과 대응표는 git 밖(~/.judgemodel/build/g1/…)에 둔다")
    cands = read_jsonl(a.cands)
    empty = [c.get("id") for c in cands if is_empty_pair(c)]
    if empty:
        print(f"멈춤 : 근거나 주장이 빈 후보 {len(empty)}줄 (처음 : {empty[:5]}) — 주장을 먼저 채운다", file=sys.stderr)
        sys.exit(2)
    if a.size < 0 or not 0 <= a.trap_rate < 1:
        sys.exit("--size 는 0 이상, --trap-rate 는 0 이상 1 미만이다")
    traps = []
    if a.traps and os.path.exists(a.traps):
        traps = read_jsonl(a.traps)
    else:
        print("함정 풀 없음 — 함정 없이 묶는다 (merge 의 함정 관문도 꺼진다)", file=sys.stderr)
    with open(a.guide, encoding="utf-8") as f:
        criteria = guide_sections(f.read())
    size = a.size or (CAREFUL_SIZE if a.style == "careful" else BATCH_SIZE)
    if load_map(a.out):
        sys.exit("--out 에 이미 묶음이 있다 — 새 폴더를 쓴다 (대응표를 덮지 않으려고)")
    if not 2 <= a.judges <= JUDGE_MAX:
        sys.exit(f"--judges 는 2 ~ {JUDGE_MAX} 다")
    judges = tuple(range(1, a.judges + 1))
    batches = build_batches(cands, traps, a.mode, a.style, size, a.trap_rate, a.seed, "b", set(), judges)
    mapping = []
    for name, items, rows, judges in batches:
        emit_batch(a.out, name, judges, items, criteria, a.style, a.mode, int(name[1:]))
        mapping.extend(rows)
    write_jsonl(os.path.join(a.out, "map", "map.jsonl"), mapping)
    write_json(os.path.join(a.out, "map", "manifest.json"), {
        "mode": a.mode, "style": a.style, "size": size, "trap_rate": a.trap_rate, "seed": a.seed,
        "traps": os.path.abspath(a.traps) if traps else "", "guide": os.path.abspath(a.guide),
        "cands": os.path.abspath(a.cands), "batches": [b[0] for b in batches], "judges": a.judges})
    os.makedirs(os.path.join(a.out, "answers"), exist_ok=True)
    real = sum(1 for m in mapping if not m["trap"])
    print(json.dumps({"batches": len(batches), "pairs": real, "traps": len(mapping) - real}, ensure_ascii=False))


def cmd_third(a):
    """decide 가 need-third 로 남긴 쌍을 셋째 판정자(j3) 묶음으로. q 는 그대로, 함정은 새 q."""
    man = load_manifest(a.out)
    mapping = load_map(a.out)
    by_q = {m["q"]: m for m in mapping if not m["trap"]}
    sent = {m["q"] for m in mapping if m["batch"].startswith("t") and not m["trap"]}
    need = [r for r in read_jsonl(os.path.join(a.out, "decided.jsonl"))
            if r["status"] == "need-third" and r["q"] not in sent]
    if not need:
        print("셋째 판정이 필요한 쌍 없음")
        return
    cands = {c.get("id"): c for c in read_jsonl(man["cands"])}
    traps = read_jsonl(man["traps"]) if man.get("traps") and os.path.exists(man["traps"]) else []
    with open(man["guide"], encoding="utf-8") as f:
        criteria = guide_sections(f.read())
    done = {m["batch"] for m in mapping}
    start = 1 + sum(1 for b in done if b.startswith("t"))
    rows = [dict(cands[by_q[r["q"]]["id"]], _q=r["q"]) for r in need]
    used = {m["q"] for m in mapping}
    batches = build_batches(rows, traps, man["mode"], man["style"], man["size"], man["trap_rate"],
                            man["seed"] + 3 + start, "t", used, (3,))
    for i, (_, items, rows_map, judges) in enumerate(batches):
        name = f"t{start + i:02d}"
        for m in rows_map:
            m["batch"] = name
        relabel_real(items, rows_map, rows)   # 실제 쌍은 첫 판 q 그대로 — 셋의 표를 한 쌍으로 모으려고
        emit_batch(a.out, name, judges, items, criteria, man["style"], man["mode"], start + i)
        mapping.extend(rows_map)
    write_jsonl(os.path.join(a.out, "map", "map.jsonl"), mapping)
    print(json.dumps({"third_batches": len(batches), "pairs": len(need)}, ensure_ascii=False))


def relabel_real(items, rows_map, rows):
    """셋째 묶음의 실제 쌍 q 를 처음 판의 q 로 되돌린다 (대응표 id 로 찾음)."""
    first_q = {r.get("id"): r["_q"] for r in rows}
    swap = {}
    for m in rows_map:
        if not m["trap"]:
            swap[m["q"]] = first_q[m["id"]]
            m["q"] = first_q[m["id"]]
    for item in items:
        item["q"] = swap.get(item["q"], item["q"])


# ---------------------------------------------------------------- 답 합치기

def first_word(value):
    text = str(value or "").strip()
    for word in (NO, YES, DASH):
        if text.startswith(word):
            return word
    return ""


def check_answer(ans, mode):
    """줄 하나의 앞뒤 검사 (라벨프롬프트-g1 1-1 표). 맞으면 칸, 아니면 None(답 없음)."""
    if not isinstance(ans, dict):
        return None
    q1, q2, q3 = (first_word(ans.get(k)) for k in ("q1", "q2", "q3"))
    label = str(ans.get("label", "")).strip().lower()
    if q1 == NO and q2 == DASH and q3 == DASH:
        allowed = ("c", "cd") if mode == "train" else ("c",)
        return label if label in allowed else None
    if q1 == YES and q2 == YES and q3 == DASH:
        return "b" if label == "b" else None
    if q1 == YES and q2 == NO and q3 == YES:
        return "a" if label == "a" else None
    if q1 == YES and q2 == NO and q3 == NO:
        return "d" if label == "d" else None
    return None


def parse_answer_file(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "q" in row:
                out[str(row["q"])] = row
    return out


ANSWER_NAME = re.compile(r"^([bt]\d+)-j(\d+)(?:\.[^.]+)?\.jsonl$")


def merge_runs(mapping, answer_files, mode, exempt=()):
    """판(묶음 × 판정자)마다 함정 정답률 · 답 없음 비율을 재고, 통과한 판의 표만 쌍별로 모은다.
    exempt : 관문에서 뺄 함정 id (map 의 id). 정답률 분모·분자에서 빠지고, 라벨에는 여전히 안 들어간다."""
    exempt = set(exempt)
    by_batch = collections.defaultdict(list)
    for m in mapping:
        by_batch[m["batch"]].append(m)
    runs, chosen = [], {}
    for path in sorted(answer_files):
        hit = ANSWER_NAME.match(os.path.basename(path))
        if not hit or hit.group(1) not in by_batch:
            continue
        batch, judge = hit.group(1), f"j{hit.group(2)}"
        answers = parse_answer_file(path)
        rows = by_batch[batch]
        traps = [m for m in rows if m["trap"] and m["id"] not in exempt]
        skipped = sorted(m["id"] for m in rows if m["trap"] and m["id"] in exempt)
        real = [m for m in rows if not m["trap"]]
        trap_ok = sum(1 for m in traps if check_answer(answers.get(m["q"]), mode) == m["gold"])
        none = sum(1 for m in real if check_answer(answers.get(m["q"]), mode) is None)
        trap_acc = (trap_ok / len(traps)) if traps else None
        none_rate = (none / len(real)) if real else 0.0
        reason = ""
        if trap_acc is not None and trap_acc < TRAP_PASS:
            reason = f"함정 {trap_ok}/{len(traps)}"
        elif none_rate > NONE_MAX:
            reason = f"답 없음 {none}/{len(real)}"
        run = {"file": os.path.basename(path), "batch": batch, "judge": judge, "trap_acc": trap_acc,
               "trap_n": len(traps), "none": none, "kept": not reason, "reason": reason}
        if skipped:
            run["trap_exempt"] = skipped
        runs.append(run)
        if not reason:
            chosen[(batch, judge)] = answers
    # 판을 다 버린 쌍도 줄을 남긴다 — decide 가 missing 으로 알린다
    votes = {m["q"]: {"q": m["q"], "id": m["id"], "domain": m.get("domain", ""), "lang": m.get("lang", ""),
                      "votes": {}, "reasons": {}, "sure": {}} for m in mapping if not m["trap"]}
    for (batch, judge), answers in chosen.items():
        for m in by_batch[batch]:
            if m["trap"]:
                continue
            row = votes[m["q"]]
            ans = answers.get(m["q"])
            label = check_answer(ans, mode)
            if label is None:
                continue
            row["votes"][judge] = label
            row["reasons"][judge] = " | ".join(str(ans.get(k, "")) for k in ("q1", "q2", "q3"))
            if ans.get("sure"):
                row["sure"][judge] = str(ans["sure"])
    return runs, [votes[q] for q in sorted(votes)]


def cmd_merge(a):
    man = load_manifest(a.out)
    answer_dir = os.path.join(a.out, "answers")
    files = [os.path.join(answer_dir, n) for n in os.listdir(answer_dir)] if os.path.isdir(answer_dir) else []
    mapping = load_map(a.out)
    unknown = sorted(set(a.exempt_trap) - {m["id"] for m in mapping if m["trap"]})
    if unknown:
        sys.exit(f"--exempt-trap 의 함정 id {unknown} 가 map.jsonl 에 없다")
    runs, votes = merge_runs(mapping, files, man["mode"], a.exempt_trap)
    write_jsonl(os.path.join(a.out, "votes.jsonl"), votes)
    write_json(os.path.join(a.out, "merge-report.json"), {"runs": runs})
    dropped = [r for r in runs if not r["kept"]]
    for r in dropped:
        print(f"버린 판 {r['file']} — {r['reason']} (같은 묶음을 새 판으로 다시 돌린다)")
    twice = [k for k, n in collections.Counter((r["batch"], r["judge"]) for r in dropped).items() if n >= 2]
    if twice:
        print(f"두 번 이상 버린 판 {twice} — 메인에 알린다")
    print(json.dumps({"runs": len(runs), "kept": len(runs) - len(dropped), "pairs": len(votes)}, ensure_ascii=False))


# ---------------------------------------------------------------- 합의

def decide_one(votes, mode):
    """판정자 표 → (칸 또는 None, 상태). 규칙 표는 build/README.md 「합의 규칙」."""
    v1, v2, v3 = votes.get("j1"), votes.get("j2"), votes.get("j3")
    if v1 is None or v2 is None:
        return None, "missing"
    if v1 == v2:
        return v1, "agree"
    if v3 is None:
        return None, "need-third"
    counts = collections.Counter((v1, v2, v3))
    label, n = counts.most_common(1)[0]
    if n >= 2:
        return label, "third"
    if mode == "train" and {v1, v2, v3} <= {"c", "d", "cd"}:
        return "cd", "cd-split"
    return None, "drop"


def decide_synth(votes, hint, judges=5, need=SYNTH_NEED):
    """합성 쌍 : 판정 judges 개 중 need 개 이상이 의도 칸(hint)과 같으면 (hint, synth).
    빠진 표를 다 채워도 need 에 못 닿으면 drop, 아직 닿을 수 있으면 missing(버린 판을 다시 돌린다)."""
    if hint not in LABELS:
        return None, "drop"
    same = sum(1 for v in votes.values() if v == hint)
    if same >= need:
        return hint, "synth"
    if same + max(0, judges - len(votes)) >= need:
        return None, "missing"
    return None, "drop"


def cmd_decide(a):
    rows = read_jsonl(os.path.join(a.out, "votes.jsonl"))
    out, third = [], []
    per_domain = collections.defaultdict(collections.Counter)
    hints, judges = {}, 5
    if a.mode == "synth":
        man = load_manifest(a.out)
        judges = man.get("judges", 2)
        if judges < SYNTH_NEED:
            sys.exit(f"synth 합의는 판정자가 {SYNTH_NEED} 이상이어야 한다 (make --judges 5) — 지금 {judges}")
        hints = {c.get("id"): c.get("hint") for c in read_jsonl(man["cands"])}
    for row in rows:
        if a.mode == "synth":
            label, status = decide_synth(row["votes"], hints.get(row["id"]), judges)
        else:
            label, status = decide_one(row["votes"], a.mode)
        out.append({"q": row["q"], "id": row["id"], "domain": row.get("domain", ""), "label": label,
                    "status": status, "votes": row["votes"]})
        per_domain[row.get("domain", "")][status] += 1
        if status == "need-third":
            third.append(row["q"])
    write_jsonl(os.path.join(a.out, "decided.jsonl"), out)
    # synth 는 「첫 두 판 일치」 대신 남김률(synth 상태 몫)을 원천별로 본다
    hit = "synth" if a.mode == "synth" else "agree"
    agree = {d: round(c[hit] / max(1, sum(c.values()) - c["missing"]), 3) for d, c in per_domain.items()}
    low = [d for d, r in agree.items() if r < 0.60] if a.mode != "synth" else []
    summary = {"status": dict(collections.Counter(r["status"] for r in out)),
               "labels": dict(collections.Counter(r["label"] for r in out if r["label"])),
               ("keep_by_domain" if a.mode == "synth" else "agree_by_domain"): agree, "need_third": len(third)}
    if a.mode == "synth":
        # 의도 칸별 남김 수 — 어느 칸이 판정자와 자주 어긋나나
        by_hint = collections.defaultdict(collections.Counter)
        for r in out:
            by_hint[hints.get(r["id"]) or ""][r["status"]] += 1
        summary["by_hint"] = {h: dict(c) for h, c in sorted(by_hint.items())}
    write_json(os.path.join(a.out, "decide-summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))
    if low:
        print(f"첫 두 판 일치율 60% 아래 원천 {low} — 그 원천의 바꾸는 규칙을 본다 (설계 3-3 ⑤)")


# ---------------------------------------------------------------- κ

def cohen_kappa(pairs):
    """[(판정자1 칸, 판정자2 칸)] → κ. 쌍이 없거나 우연 일치가 1 이면 None."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    left = collections.Counter(x for x, _ in pairs)
    right = collections.Counter(y for _, y in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def kappas(vote_rows):
    both = [(r["votes"]["j1"], r["votes"]["j2"]) for r in vote_rows
            if "j1" in r["votes"] and "j2" in r["votes"]]
    cd = [(x, y) for x, y in both if x in ("c", "d") and y in ("c", "d")]
    return {"all": cohen_kappa(both), "all_n": len(both), "cd": cohen_kappa(cd), "cd_n": len(cd)}


def cmd_kappa(a):
    k = kappas(read_jsonl(os.path.join(a.out, "votes.jsonl")))
    k["pass"] = k["cd"] is not None and k["cd"] >= KAPPA_CD_MIN
    print(json.dumps({key: (round(v, 4) if isinstance(v, float) else v) for key, v in k.items()},
                     ensure_ascii=False))
    if not k["pass"]:
        print(f"c/d κ < {KAPPA_CD_MIN} — 기준 글을 한 번 고치고 표본 60쌍만 다시 붙인다")
        sys.exit(1)


# ---------------------------------------------------------------- 정답셋 · 사람 목록

def is_weak(vote_row):
    if any(s.strip().startswith("낮") for s in vote_row.get("sure", {}).values()):
        return True
    return any(w in text for text in vote_row.get("reasons", {}).values() for w in WEAK_WORDS)


def share_counts(total, keys_count):
    """total 을 도메인 크기 비례로 나눈다 (큰 나머지 차례)."""
    size = sum(keys_count.values())
    if not size:
        return {}
    raw = {k: total * v / size for k, v in keys_count.items()}
    out = {k: math.floor(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[:total - sum(out.values())]:
        out[k] += 1
    return out


def pick_gold(decided, votes_by_q, n_hard, n_random, seed):
    """갈린 쌍 · 「근거 약함」 쌍 n_hard + 도메인별 무작위 n_random. 모자란 갈린 몫은 무작위로 채운다."""
    rng = random.Random(seed)
    labeled = [r for r in decided if r["label"] in LABELS]
    hard = [r for r in labeled if r["status"] != "agree" or is_weak(votes_by_q.get(r["q"], {}))]
    rng.shuffle(hard)
    hard = hard[:n_hard]
    taken = {r["q"] for r in hard}
    rest = [r for r in labeled if r["q"] not in taken]
    want_random = n_random + (n_hard - len(hard))
    by_domain = collections.defaultdict(list)
    for r in rest:
        by_domain[r.get("domain", "")].append(r)
    picks = []
    for domain, n in share_counts(want_random, {d: len(v) for d, v in by_domain.items()}).items():
        picks.extend(rng.sample(by_domain[domain], min(n, len(by_domain[domain]))))
    out = [dict(r, why="split" if r["status"] != "agree" else "weak") for r in hard]
    out.extend(dict(r, why="random") for r in picks)
    return out


def load_dev_dirs(dirs, set_ids=None):
    """dev 묶음 폴더 여럿 → (decided, votes_by_q). q 는 폴더마다 따로 뽑혀 겹칠 수 있어 「폴더번호:q」 로 바꾼다.
    set_ids 를 주면 얼린 셋에 든 쌍만 남긴다 (조립에서 빠진 쌍은 정답셋 후보가 아니다)."""
    decided, votes = [], {}
    for i, out_dir in enumerate(dirs):
        for r in read_jsonl(os.path.join(out_dir, "votes.jsonl")):
            votes[f"{i}:{r['q']}"] = r
        for r in read_jsonl(os.path.join(out_dir, "decided.jsonl")):
            if set_ids is None or r["id"] in set_ids:
                decided.append(dict(r, q=f"{i}:{r['q']}"))
    ids = collections.Counter(r["id"] for r in decided)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        sys.exit(f"묶음 폴더끼리 id 가 겹친다 ({len(dup)}건) — 같은 폴더를 두 번 줬는지 본다")
    return decided, votes


def cmd_gold_pick(a):
    texts = {r.get("id"): r for r in read_jsonl(a.set)}
    decided, votes = load_dev_dirs(a.out, set(texts))
    picked = pick_gold(decided, votes, a.n_hard, a.n_random, a.seed)
    rows = []
    for r in picked:
        src = texts.get(r["id"], {})
        rows.append({"id": r["id"], "evidence": src.get("evidence", ""), "claim": src.get("claim", ""),
                     "before": src.get("before", "없음"), "after": src.get("after", "없음"),
                     "domain": r.get("domain", ""), "lang": src.get("lang", ""), "dev_label": r["label"],
                     "why": r["why"]})
    dest = a.dest or a.out[0]
    if inside_git(dest):
        sys.exit("--dest 가 git 안이다")
    path = os.path.join(dest, "gold-cands.jsonl")
    write_jsonl(path, rows)
    print(json.dumps({"gold": len(rows), "from": len(decided),
                      "why": dict(collections.Counter(r["why"] for r in rows)), "out": path}, ensure_ascii=False))


def gold_agreement(dev_decided, gold_decided, why_by_id=None):
    """「dev 라벨 = gold 라벨」 비율 — 전체 · 칸별(gold 기준) · 첫 두 판이 같았던 쌍만 · 뽑힌 까닭(why)별.
    관문(0.85)은 무작위 층(random)이 있으면 그 층으로 잰다. gold-pick 이 갈린 쌍을 일부러 많이 뽑아서
    전체 평균은 아래로 치우친다. 무작위 층이 없으면 전체로 잰다."""
    dev = {r["id"]: r for r in dev_decided if r["label"]}
    pairs = [(dev[g["id"]], g) for g in gold_decided if g["label"] and g["id"] in dev]
    if not pairs:
        return {"n": 0}

    def rate(items):
        return round(sum(1 for d, g in items if d["label"] == g["label"]) / len(items), 4) if items else None
    by_label = {lab: rate([(d, g) for d, g in pairs if g["label"] == lab]) for lab in LABELS}
    agreed = [(d, g) for d, g in pairs if d["status"] == "agree"]
    layers = collections.defaultdict(list)
    for d, g in pairs:
        layers[(why_by_id or {}).get(g["id"], "")].append((d, g))
    by_why = {w: {"n": len(v), "all": rate(v)} for w, v in sorted(layers.items()) if w}
    gate_on = "random" if "random" in by_why else "all"
    gate = by_why["random"]["all"] if gate_on == "random" else rate(pairs)
    return {"n": len(pairs), "all": rate(pairs), "by_label": by_label, "agree_only": rate(agreed),
            "by_why": by_why, "gate_on": gate_on, "pass": gate >= 0.85}


def gold_why(gold_dir, cands=""):
    """gold-cands.jsonl 의 id → why. 기본 자리는 gold 묶음 폴더, 없으면 그 위 폴더(gold-pick 의 --dest)."""
    paths = [cands] if cands else [os.path.join(gold_dir, "gold-cands.jsonl"),
                                   os.path.join(os.path.dirname(os.path.abspath(gold_dir)), "gold-cands.jsonl")]
    for path in paths:
        if os.path.exists(path):
            return {r["id"]: r.get("why", "") for r in read_jsonl(path)}
    if cands:
        sys.exit(f"--cands 파일이 없다 : {cands}")
    return {}


def dev_from_cands(path):
    """gold-cands.jsonl 의 dev_label 칸을 dev 라벨로 쓴다 — dev 칸이 묶음 폴더 decided.jsonl 밖에서
    정해진 쌍(g3 ko b : 위키 · BoolQ · KLUE 판정 파일)을 잴 때. 합의 여부는 몰라 status 는 비운다."""
    return [{"id": r["id"], "label": r.get("dev_label"), "status": ""} for r in read_jsonl(path)]


def cmd_gold_score(a):
    if bool(a.out) == bool(a.dev_from):
        sys.exit("--out (dev 묶음 폴더) 과 --dev-from (gold-cands.jsonl) 중 하나만 준다")
    dev = dev_from_cands(a.dev_from) if a.dev_from else load_dev_dirs(a.out)[0]
    res = gold_agreement(dev, read_jsonl(os.path.join(a.gold, "decided.jsonl")), gold_why(a.gold, a.cands))
    print(json.dumps(res, ensure_ascii=False))
    if res.get("n") and not res["pass"]:
        print("0.85 아래 — dev 라벨에서 어긋난 칸 규칙을 찾아 지침을 고치고 그 칸만 다시 붙인다 (한 번)")


def review_rows(decided, votes_by_q, mode, gold_by_id, split_sample, seed):
    """사람 목록 차례 : 신중 판정자 「낮음」 → gold ≠ dev → 버림 · cd → 셋째가 가른 쌍(학습은 무작위 split_sample)."""
    rng = random.Random(seed)
    first, second, third, split = [], [], [], []
    for r in decided:
        g = gold_by_id.get(r["id"])
        if g and is_weak(g.get("_votes", {})):
            first.append(dict(r, why="gold-low", gold=g.get("label")))
        elif g and g.get("label") and r["label"] and g["label"] != r["label"]:
            second.append(dict(r, why="gold-diff", gold=g["label"]))
        elif r["status"] in ("drop", "cd-split"):
            third.append(dict(r, why=r["status"]))
        elif r["status"] == "third":
            split.append(dict(r, why="third"))
    if mode == "train":
        rng.shuffle(split)
        split = split[:split_sample]
    out = first + second + third + split
    for row in out:
        v = votes_by_q.get(row["q"], {})
        row["reasons"] = v.get("reasons", {})
    return out


REVIEW_README = """human-review-todo.jsonl — 사람이 나중에 볼 라벨 목록 (JudgeModel 설계 d자료층재구축 2-4 ④ · 3-3)

줄마다 쌍 하나다. 칸 : a 지지 · b 반대 · c 무관 · d 비슷한 계열 (기준 Docs/Guide/라벨기준support-vN.md — 줄의 label_guide 판).
- id · set : 원래 쌍 id 와 어느 셋에서 왔나
- evidence · claim : 근거 · 주장
- votes : 판정자 j1 · j2 · j3 의 칸 / reasons : 판정자의 Q1 | Q2 | Q3 답과 까닭
- final : 지금 셋에 들어간 칸 (null 이면 버린 쌍) / gold : 신중 판정자 칸 (있을 때만)
- why : gold-low(신중 판정자가 「낮음」) · gold-diff(gold ≠ dev) · drop(셋 다 다름) · cd-split · third(셋째가 가름)

보는 법 : 줄마다 "human" 칸에 a · b · c · d 또는 "skip" 을 적는다. 감사 결과는 다음 판 셋에 반영한다.
이번 판의 얼린 셋(FROZEN-*.sha256)은 고치지 않는다.
"""


def cmd_review_todo(a):
    modes = {load_manifest(d)["mode"] for d in a.out}
    if len(modes) > 1:
        sys.exit(f"--out 묶음 폴더의 mode 가 다르다 {sorted(modes)}")
    decided, votes = load_dev_dirs(a.out)
    gold = {}
    if a.gold:
        gvotes = {r["q"]: r for r in read_jsonl(os.path.join(a.gold, "votes.jsonl"))}
        for g in read_jsonl(os.path.join(a.gold, "decided.jsonl")):
            gold[g["id"]] = dict(g, _votes=gvotes.get(g["q"], {}))
    texts = {r.get("id"): r for r in read_jsonl(a.set)}
    rows = review_rows(decided, votes, modes.pop(), gold, a.split_sample, a.seed)
    if inside_git(a.dest):
        sys.exit("--dest 가 git 안이다")
    path = os.path.join(a.dest, "human-review-todo.jsonl")
    old = read_jsonl(path) if os.path.exists(path) else []
    seen = {(r.get("set"), r.get("id")) for r in old}
    set_name = a.set_name or os.path.splitext(os.path.basename(a.set))[0]
    for r in rows:
        if len(old) >= a.max:
            break
        if (set_name, r["id"]) in seen:
            continue
        src = texts.get(r["id"], {})
        old.append({"id": r["id"], "set": set_name, "evidence": src.get("evidence", ""),
                    "claim": src.get("claim", ""), "votes": r["votes"], "reasons": r["reasons"],
                    "final": r["label"], "gold": r.get("gold"), "why": r["why"]})
    write_jsonl(path, old)
    with open(os.path.join(a.dest, "README.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(REVIEW_README)
    print(json.dumps({"todo": len(old), "max": a.max, "out": path}, ensure_ascii=False))


# ---------------------------------------------------------------- 셋 조립 (소단계 45-바)

KEEP_STATUS = ("agree", "third")
DONE_STATUS = ("agree", "third", "drop")
# 학습 셋(--mode train)은 cd-split 도 판정이 끝난 줄로 보고 넣는다 — cd 는 학습 전용 칸 (라벨기준support-v5 4절)
# 합성 쌍(decide --mode synth)의 남김 상태 synth 도 학습 셋에서만 받는다
TRAIN_KEEP_STATUS = KEEP_STATUS + ("cd-split", "synth")
TRAIN_DONE_STATUS = DONE_STATUS + ("cd-split", "synth")
ASSEMBLE_MODES = ("eval", "train")
# 학습 줄에 남기는 원천 칸 (있는 것만)
TRAIN_SOURCE_KEYS = ("origin", "src", "lic", "synth_from", "grid")
JUDGE_KEYS = tuple(f"j{i}" for i in range(1, JUDGE_MAX + 1))
OLD_WANT_LETTER = {"support": "a", "contradict": "b", "unrelated": "c", "related": "d",
                   "a": "a", "b": "b", "c": "c", "d": "d", "cd": "cd"}
LABEL_GUIDE = "support-v5"   # 판의 기준 문서 이름에서 판을 못 뽑을 때 쓰는 기본 (옛 판 manifest 에는 guide 가 없다)
GUIDE_VERSION = re.compile(r"support-v\d+")


def guide_label(guide_path):
    """기준 문서 경로 → 줄에 적을 label_guide 값. 파일 이름의 support-vN 을 쓰고, 없으면 LABEL_GUIDE."""
    found = GUIDE_VERSION.search(os.path.basename(guide_path or ""))
    return found.group(0) if found else LABEL_GUIDE


class AssembleStop(Exception):
    """조립을 멈춰야 하는 자료 이상 (cd · 겹침 · 함정 섞임 · 판정 덜 끝남)."""


def is_empty_pair(row):
    """근거나 주장이 비었나 — 빈 쌍은 판정자에게 보내지도, 셋에 넣지도 않는다 (46 KLUE 빈 주장 1,000쌍)."""
    return not str(row.get("evidence") or "").strip() or not str(row.get("claim") or "").strip()


def text_key(row):
    return (str(row.get("evidence", "")).strip(), str(row.get("claim", "")).strip())


def trap_keys(out_dir, man):
    """묶음 폴더의 함정 — 대응표의 함정 q · id 와 함정 풀의 (근거, 주장)."""
    mapping = load_map(out_dir)
    qs = {m["q"] for m in mapping if m["trap"]}
    ids = {m["id"] for m in mapping if m["trap"]}
    texts = set()
    pool = man.get("traps") or ""
    if pool and os.path.exists(pool):
        texts = {text_key(t) for t in read_jsonl(pool)}
    return qs, ids, texts


def set_row(cand, dec, set_name, mode="eval", label_guide=LABEL_GUIDE):
    """후보 줄 + 합의 줄 → 셋 줄 (q 는 뺀다, want 는 칸 글자).

    train 이면 원천 칸(TRAIN_SOURCE_KEYS 중 있는 것)을 더 남긴다. eval 줄 꼴은 그대로다 (얼린 셋 바이트를 안 바꾸려고).
    줄 꼴은 train/data_prep.normalize 가 읽는 mem 쌍 꼴(evidence · claim · want)이다.
    """
    row = {"id": cand["id"], "evidence": cand["evidence"], "claim": cand["claim"], "want": dec["label"],
           "label_guide": label_guide}
    for key in ("domain", "group", "groups", "lang", "form", "hint"):
        row[key] = cand.get(key, "")
    if cand.get("source"):
        row["source"] = cand["source"]
    row["relabel"] = bool(cand.get("relabel"))
    if cand.get("old_want"):
        row["old_want"] = cand["old_want"]
    row["votes"] = {j: dec["votes"][j] for j in JUDGE_KEYS if j in dec.get("votes", {})}
    if mode == "train":
        for key in TRAIN_SOURCE_KEYS:
            if cand.get(key):
                row[key] = cand[key]
    row["status"] = dec["status"]
    row["set"] = set_name
    return row


def collect_dir(out_dir, set_name, mode="eval"):
    """묶음 폴더 하나 → (넣을 줄, 폴더 요약). 판정이 덜 끝났거나 함정이 섞였으면 AssembleStop.

    eval 은 cd 칸 · 후보의 셋 이름이 다르면 멈춘다. train 은 cd(cd-split 포함)를 want "cd" 로 넣고 셋 이름을 안 본다.
    """
    train = mode == "train"
    keep_status = TRAIN_KEEP_STATUS if train else KEEP_STATUS
    done_status = TRAIN_DONE_STATUS if train else DONE_STATUS
    man = load_manifest(out_dir)
    cand_path = man["cands"]
    if not os.path.exists(cand_path):
        cand_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), f"{os.path.basename(out_dir)}-cands.jsonl")
    cands = {c["id"]: c for c in read_jsonl(cand_path)}
    decided = read_jsonl(os.path.join(out_dir, "decided.jsonl"))
    trap_q, trap_id, trap_text = trap_keys(out_dir, man)
    status = collections.Counter(r["status"] for r in decided)
    undone = {s: n for s, n in status.items() if s not in done_status}
    if undone:
        raise AssembleStop(f"{out_dir} : 판정이 덜 끝났다 {undone}")
    rows, traps_hit, relabel_dropped, dropped_empty = [], 0, 0, 0
    for dec in decided:
        if dec["q"] in trap_q or dec["id"] in trap_id:
            traps_hit += 1
            continue
        if dec["id"] in cands and is_empty_pair(cands[dec["id"]]):
            dropped_empty += 1
            continue
        if dec["status"] not in keep_status:
            if cands.get(dec["id"], {}).get("relabel"):
                relabel_dropped += 1
            continue
        if dec["label"] == "cd" and not train:
            raise AssembleStop(f"{out_dir} : 평가 셋에 cd 칸 ({dec['id']})")
        if dec["label"] not in LABELS and not (train and dec["label"] == "cd"):
            raise AssembleStop(f"{out_dir} : 모르는 칸 {dec['label']!r} ({dec['id']})")
        cand = cands.get(dec["id"])
        if cand is None:
            raise AssembleStop(f"{out_dir} : 후보에 없는 id {dec['id']}")
        if not train and cand.get("set") and cand["set"] != set_name:
            raise AssembleStop(f"{out_dir} : 후보의 셋 {cand['set']} ≠ {set_name}")
        if text_key(cand) in trap_text:
            traps_hit += 1
            continue
        rows.append(set_row(cand, dec, set_name, mode, guide_label(man.get("guide"))))
    votes_path = os.path.join(out_dir, "votes.jsonl")
    kap = kappas(read_jsonl(votes_path)) if os.path.exists(votes_path) else {}
    report_path = os.path.join(out_dir, "merge-report.json")
    runs = []
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            runs = json.load(f).get("runs", [])
    info = {"dir": os.path.basename(os.path.dirname(os.path.abspath(out_dir))) + "/" + os.path.basename(out_dir),
            "decided": len(decided), "kept": len(rows), "status": dict(status), "traps_in_decided": traps_hit,
            "relabel_dropped": relabel_dropped, "dropped_empty": dropped_empty,
            "kappa": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kap.items()},
            "runs": len(runs), "runs_dropped": sum(1 for r in runs if not r.get("kept"))}
    return rows, info


def relabel_matrix(rows):
    """dev-r2 다시 붙인 줄의 v4(old_want) → v5(want) 옮김 수."""
    matrix = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("relabel") and r.get("old_want"):
            matrix[OLD_WANT_LETTER.get(str(r["old_want"]).lower(), str(r["old_want"]))][r["want"]] += 1
    return {old: dict(sorted(c.items())) for old, c in sorted(matrix.items())}


def dedupe_same(rows):
    """한 폴더 안 같은 (근거, 주장) 이 같은 칸이면 id 가 작은 줄만 남긴다. 칸이 다르면 그대로 둔다(뒤에서 멈춤)."""
    first, out, dropped = {}, [], []
    for r in sorted(rows, key=lambda r: r["id"]):
        key = text_key(r)
        if key in first and first[key]["want"] == r["want"]:
            dropped.append(r["id"])
            continue
        first.setdefault(key, r)
        out.append(r)
    return out, dropped


def assemble(dirs, set_name, dedupe=False, mode="eval", allow_mixed_guide=False):
    """묶음 폴더들 → (id 차례로 정렬한 셋 줄, 요약). 폴더 사이 (근거, 주장) · id 겹침이 있으면 AssembleStop.

    dedupe 이면 한 폴더 안의 같은 글 · 같은 칸 겹침만 하나로 줄인다 (폴더 사이 겹침은 늘 멈춤).
    mode 는 eval(평가 셋, cd 멈춤) · train(학습 셋, cd 넣음) — collect_dir 참고.
    라벨 기준이 섞이면 멈춘다. allow_mixed_guide 면 줄마다 제 기준을 두고 요약 label_guide 를 "mixed" 로 적는다.
    """
    if mode not in ASSEMBLE_MODES:
        raise ValueError(f"mode 는 {ASSEMBLE_MODES} 중 하나다: {mode!r}")
    rows, infos = [], []
    for d in dirs:
        got, info = collect_dir(d, set_name, mode)
        info["deduped"] = []
        if dedupe:
            got, info["deduped"] = dedupe_same(got)
            info["kept"] = len(got)
        if mode == "train":
            info["want"] = dict(sorted(collections.Counter(r["want"] for r in got).items()))
        rows.extend(got)
        infos.append(info)
    by_text = collections.Counter(text_key(r) for r in rows)
    by_id = collections.Counter(r["id"] for r in rows)
    overlap = sum(n - 1 for n in by_text.values() if n > 1)
    dup_id = sum(n - 1 for n in by_id.values() if n > 1)
    if overlap or dup_id:
        raise AssembleStop(f"{set_name} : 같은 (근거, 주장) {overlap} · 같은 id {dup_id}")
    guides = sorted({r["label_guide"] for r in rows})
    if len(guides) > 1 and not allow_mixed_guide:
        raise AssembleStop(f"{set_name} : 폴더마다 라벨 기준이 다르다 {guides} — 섞어 쓰려면 --allow-mixed-guide")
    guide = LABEL_GUIDE
    if len(guides) == 1:
        guide = guides[0]
    elif guides:
        guide = "mixed"
    rows.sort(key=lambda r: r["id"])
    by_domain = collections.defaultdict(collections.Counter)
    for r in rows:
        by_domain[r["domain"]][r["want"]] += 1
    summary = {
        "set": set_name, "pairs": len(rows), "label_guide": guide,
        "label_guides": dict(sorted(collections.Counter(r["label_guide"] for r in rows).items())),
        "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
        "domain": {d: sum(c.values()) for d, c in sorted(by_domain.items())},
        "domain_want": {d: dict(sorted(c.items())) for d, c in sorted(by_domain.items())},
        "lang": dict(sorted(collections.Counter(r["lang"] for r in rows).items())),
        "status": dict(sorted(collections.Counter(r["status"] for r in rows).items())),
        "relabel": sum(1 for r in rows if r["relabel"]), "relabel_matrix": relabel_matrix(rows),
        "overlap": overlap, "dropped_empty": sum(i["dropped_empty"] for i in infos), "dirs": infos}
    if mode == "train":
        summary = {"mode": mode, **summary, "cd": summary["want"].get("cd", 0),
                   "deduped": sum(len(i["deduped"]) for i in infos)}
    return rows, summary


def cmd_assemble(a):
    if inside_git(a.dest) and not a.allow_git:
        sys.exit("--dest 가 git 안이다 — 셋 본문은 git 밖(~/.judgemodel/sets/g1)에 둔다")
    try:
        rows, summary = assemble(a.dirs, a.set, a.dedupe_same, a.mode, a.allow_mixed_guide)
    except AssembleStop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        sys.exit(2)
    path = os.path.join(a.dest, f"{a.set}.jsonl")
    write_jsonl(path, rows)
    with open(path, "rb") as f:
        summary["sha256"] = hashlib.sha256(f.read()).hexdigest()
    write_json(os.path.join(a.dest, f"{a.set}.summary.json"), summary)
    keys = ("set", "pairs", "label_guides", "want", "domain", "status", "relabel", "overlap", "dropped_empty",
            "sha256")
    if a.mode == "train":
        keys = ("mode",) + keys + ("cd", "deduped")
    print(json.dumps({k: summary[k] for k in keys}, ensure_ascii=False))


# ---------------------------------------------------------------- CLI

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("traps", help="사람 라벨이 확실한 함정 풀 만들기")
    s.add_argument("--dev-r2", required=True)
    s.add_argument("--ruler4", default="")
    s.add_argument("--pool", required=True)
    s.set_defaults(func=cmd_traps)

    s = sub.add_parser("make", help="후보 → 묶음 (불투명 q · 함정 · 판정자별 차례)")
    s.add_argument("--cands", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("train", "eval"), required=True)
    s.add_argument("--style", choices=("label", "careful"), default="label")
    s.add_argument("--traps", default="")
    s.add_argument("--guide", default=GUIDE_DEFAULT)
    s.add_argument("--size", type=int, default=0)
    s.add_argument("--trap-rate", type=float, default=TRAP_RATE)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--judges", type=int, default=2,
                   help="판정자 수 (기본 2 = j1 · j2, 셋째는 third 가 따로). 합성 쌍(synth.py)은 5")
    s.add_argument("--allow-git", action="store_true", help="시험용 — git 안에 써도 된다")
    s.set_defaults(func=cmd_make)

    s = sub.add_parser("third", help="갈린 쌍 → 셋째 판정자 묶음")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_third)

    s = sub.add_parser("merge")
    s.add_argument("--out", required=True)
    s.add_argument("--exempt-trap", nargs="+", default=[],
                   help="함정 관문에서 뺄 함정 id (map.jsonl 의 id) — 함정 자체가 의심스러울 때")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("kappa")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_kappa)

    s = sub.add_parser("decide")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("train", "eval", "synth"), required=True,
                   help="synth = 합성 쌍 (판정 5 중 4 이상이 hint 와 같을 때만 남김)")
    s.set_defaults(func=cmd_decide)

    s = sub.add_parser("gold-pick")
    s.add_argument("--out", nargs="+", required=True, help="dev 묶음 폴더 (공개판 · 아카이브판)")
    s.add_argument("--set", required=True)
    s.add_argument("--dest", default="", help="gold-cands.jsonl 을 둘 곳 (기본 : 첫 --out)")
    s.add_argument("--n-hard", type=int, default=120)
    s.add_argument("--n-random", type=int, default=80)
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_gold_pick)

    s = sub.add_parser("gold-score")
    s.add_argument("--out", nargs="*", default=[], help="dev 묶음 폴더 (공개판 · 아카이브판)")
    s.add_argument("--dev-from", default="", help="dev 라벨을 이 gold-cands.jsonl 의 dev_label 칸에서 읽는다 (--out 대신)")
    s.add_argument("--gold", required=True)
    s.add_argument("--cands", default="", help="gold-cands.jsonl (기본 : --gold 폴더, 없으면 그 위 폴더)")
    s.set_defaults(func=cmd_gold_score)

    s = sub.add_parser("review-todo")
    s.add_argument("--out", nargs="+", required=True, help="묶음 폴더 (공개판 · 아카이브판)")
    s.add_argument("--set", required=True)
    s.add_argument("--set-name", default="")
    s.add_argument("--gold", default="")
    s.add_argument("--dest", required=True)
    s.add_argument("--max", type=int, default=200)
    s.add_argument("--split-sample", type=int, default=50)
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_review_todo)

    s = sub.add_parser("assemble", help="합의 끝난 묶음 폴더들 → 평가 · 학습 셋 jsonl + summary")
    s.add_argument("--mode", choices=ASSEMBLE_MODES, default="eval",
                   help="eval = 평가 셋(cd 멈춤 · 후보 셋 이름 검사) · train = 학습 셋(cd 넣음 · 셋 이름 자유)")
    s.add_argument("--set", required=True,
                   help="셋 이름 (eval : dev-cal · dev-cmp · blind-g1 · ood-g1, train : 자유 — 예 train-g1-II)")
    s.add_argument("--dirs", nargs="+", required=True, help="묶음 폴더 (공개판 · 아카이브판)")
    s.add_argument("--dest", required=True)
    s.add_argument("--dedupe-same", action="store_true",
                   help="한 폴더 안 같은 글 · 같은 칸 겹침은 id 작은 줄만 남긴다 (기본은 멈춤)")
    s.add_argument("--allow-git", action="store_true", help="시험용 — git 안에 써도 된다")
    s.add_argument("--allow-mixed-guide", action="store_true",
                   help="폴더마다 라벨 기준(support-vN)이 달라도 잇는다 — 줄마다 label_guide · summary 에 기준별 수 (기본은 멈춤)")
    s.set_defaults(func=cmd_assemble)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
