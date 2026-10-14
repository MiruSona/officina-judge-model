"""synth — ㄴ 실물 씨앗 다시 쓰기(합성) : 씨앗 뽑기 · Opus 다시쓰기 묶음 · 답 거두기 · 합성 d 상한 셈.

설계 `../Docs/Design/2026-10-11-d자료층재구축설계.md` 5절. 쌍 본문은 git 밖(`~/.judgemodel/build/g1/synth/`)에만 쓴다.

    python -I synth.py seeds   --train <학습 셋.jsonl> … --pred <sN-II.pred.jsonl> <sN-arch.pred.jsonl> … --out seeds.jsonl [--cap N]
    python -I synth.py make    --seeds seeds.jsonl --out <다시쓰기 폴더> [--size 20]
    python -I synth.py collect --out <다시쓰기 폴더> --dest synth-cands.jsonl
    python -I synth.py cap     --d-total <학습 d 수> [--keep-rate 0.6] [--set <조립한 합성 셋.jsonl>]

그다음 라벨은 `label_batches.py make --judges 5 --mode train` → merge → `decide --mode synth` → `assemble --mode train`.
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_batches as lb  # noqa: E402

SEED = 20261050
SIZE = 20
CELLS = ("a", "b", "c", "d")
# 다수결 예측과 gold 가 이 짝으로 엇갈린 쌍이 씨앗이다 (설계 5절 ①)
FLIP_PAIRS = ({"a", "d"}, {"b", "d"})
SEED_WANTS = ("a", "b", "d")
SYNTH_D_SHARE = 0.25
PRED_NAME = re.compile(r"^s(\d+)-(II|arch)\.pred\.jsonl$")
OLD_LETTER = {"support": "a", "contradict": "b", "unrelated": "c", "related": "d"}

# 속성 격자 (설계 5절 ②) — 축마다 고르게 돌려 씨앗에 하나씩 준다
GRID = {
    "target": ("사물", "사건", "기능", "규칙", "수치", "결정"),
    "d_kind": ("다른 속성", "다른 시점", "다음 단계", "원인↔결과"),
    "length": ("짧음", "김"),
    "style": ("메모", "문서", "이슈"),
}


# ---------------------------------------------------------------- 씨앗

def letter(value):
    v = str(value or "").strip().lower()
    return OLD_LETTER.get(v, v)


def load_preds(paths):
    """pred 파일들 → {씨앗 번호: {쌍 id: 예측 칸}}. 파일 이름 sN-II|arch.pred.jsonl 로 씨앗을 묶는다."""
    by_seed = collections.defaultdict(dict)
    for path in paths:
        hit = PRED_NAME.match(os.path.basename(path))
        if not hit:
            sys.exit(f"pred 파일 이름이 sN-II.pred.jsonl · sN-arch.pred.jsonl 꼴이 아니다 : {os.path.basename(path)}")
        seed = int(hit.group(1))
        for row in lb.read_jsonl(path):
            if row.get("id"):
                by_seed[seed][row["id"]] = letter(row.get("pred"))
    return dict(by_seed)


def majority(preds_by_seed, pid):
    """씨앗들의 예측 다수결 — 예측한 씨앗의 절반을 넘는 칸만. 없으면 None."""
    votes = [p[pid] for p in preds_by_seed.values() if pid in p]
    if not votes:
        return None, 0
    label, n = collections.Counter(votes).most_common(1)[0]
    return (label if n * 2 > len(votes) else None), len(votes)


def balanced_grid(n, rng):
    """축마다 값을 n 개까지 고르게 되풀이한 뒤 섞는다 — 한 값으로 몰리지 않게."""
    columns = {}
    for axis, values in GRID.items():
        col = [values[i % len(values)] for i in range(n)]
        rng.shuffle(col)
        columns[axis] = col
    return [{axis: columns[axis][i] for axis in GRID} for i in range(n)]


def pick_seeds(train_rows, preds_by_seed, cap, seed):
    """학습 쌍 → (씨앗 줄, 셈). want ∈ a·b·d 이고 다수결 예측과 a↔d · b↔d 로 엇갈린 쌍만."""
    count = collections.Counter()
    picked = []
    for row in train_rows:
        want = letter(row.get("want"))
        if want not in SEED_WANTS:
            count["skip_want"] += 1
            continue
        pred, n = majority(preds_by_seed, row["id"])
        if n == 0:
            count["no_pred"] += 1
            continue
        if pred is None:
            count["no_majority"] += 1
            continue
        if {want, pred} not in FLIP_PAIRS:
            continue
        count[f"{want}->{pred}"] += 1
        picked.append(dict(row, _pred=pred))
    count["flipped"] = len(picked)
    rng = random.Random(seed)
    if cap and len(picked) > cap:
        picked = stratified(picked, cap, rng)
    picked.sort(key=lambda r: r["id"])
    grids = balanced_grid(len(picked), rng)
    out = []
    for i, (row, grid) in enumerate(zip(picked, grids)):
        out.append({"seed_id": f"s{i + 1:04d}", "id": row["id"], "evidence": row["evidence"], "claim": row["claim"],
                    "want": letter(row["want"]), "pred": row["_pred"], "origin": row.get("origin", ""),
                    "lang": row.get("lang", ""), "grid": grid, "domain": row.get("domain", ""),
                    "group": row.get("group", ""), "groups": row.get("groups") or [], "lic": row.get("lic", "")})
    count["seeds"] = len(out)
    return out, count


def stratified(rows, cap, rng):
    """origin 별 크기 비례로 cap 개를 무작위로 뽑는다."""
    by_origin = collections.defaultdict(list)
    for r in rows:
        by_origin[r.get("origin", "")].append(r)
    out = []
    for origin, n in lb.share_counts(cap, {k: len(v) for k, v in by_origin.items()}).items():
        out.extend(rng.sample(by_origin[origin], n))
    return out


def cmd_seeds(a):
    if lb.inside_git(a.out) and not a.allow_git:
        sys.exit("--out 이 git 안이다 — 씨앗 본문은 git 밖에 둔다")
    rows, seen = [], set()
    for path in a.train:
        for row in lb.read_jsonl(path):
            if row.get("id") in seen:
                sys.exit(f"학습 셋끼리 id 가 겹친다 : {row.get('id')}")
            seen.add(row.get("id"))
            rows.append(row)
    preds = load_preds(a.pred)
    seeds, count = pick_seeds(rows, preds, a.cap, a.seed)
    lb.write_jsonl(a.out, seeds)
    report = {"train": len(rows), "pred_seeds": sorted(preds), **dict(count),
              "by_origin": dict(collections.Counter(s["origin"] for s in seeds)),
              "by_lang": dict(collections.Counter(s["lang"] for s in seeds)), "out": a.out}
    print(json.dumps(report, ensure_ascii=False))
    if len(preds) < 5:
        print(f"주의 : 씨앗 {len(preds)} 개의 예측만 받았다 (설계는 다섯 씨앗 다수결)", file=sys.stderr)


# ---------------------------------------------------------------- 다시쓰기 묶음

REWRITE_ROLE = ("너는 판정 모델 학습용 쌍의 「근거(evidence)」를 다시 쓰는 작성자다.\n"
                "씨앗마다 주장(claim) 하나와 원래 근거 하나를 준다. 주장은 그대로 두고, 근거만 칸 넷(a · b · c · d)에 하나씩 새로 쓴다.")
REWRITE_RULES = """- **주장은 한 글자도 바꾸지 않는다.** 출력에 주장을 다시 적지도 않는다.
- 원래 근거는 대상 · 분야를 알아보는 소재로만 쓴다. 그대로 베끼지 않는다.
- a 지지 : 주장의 어느 한 부분이라도 맞다고 해 주는 근거. 주장을 글자 그대로 옮기지 말고 바꿔 말한다.
- b 반대 : 주장과 함께 참일 수 없는 말이 든 근거 (같은 대상 · 같은 때 · 같은 자리).
- c 무관 : **같은 분야의 다른 대상**을 말하는 근거. 주장 대상과 다른 사물 · 사건 · 기능 · 규칙 · 수치 · 결정이다.
- d 비슷한 계열 : 주장과 **같은 대상**의 **다른 사실**. 씨앗의 `grid.d_kind` 대로 쓴다
  (다른 속성 = 같은 대상의 다른 성질 · 다른 시점 = 같은 대상의 다른 때 · 다음 단계 = 그 뒤에 이어진 일 · 원인↔결과 = 그 일의 까닭이나 결과).
  주장을 받치지도 뒤집지도 않는다. 숨은 부분 지지(같은 사실을 다른 말 · 다른 단위 · 이름 · 날짜로 쓴 것)가 없게 한 번 더 본다.
- 네 칸은 **같은 말투**로 쓴다. 글꼴은 `grid.style` 하나로 맞춘다
  (메모 = 짧은 개인 메모 말투 · 문서 = 설명서 · 설계 문서의 평서문 · 이슈 = 이슈 트래커 글의 현상 · 원인 · 제안 말투).
- 길이는 `grid.length` 대로 : 짧음 = 한 문장 40~120자 · 김 = 두~네 문장 150~400자. 영어면 글자 수를 1.5배로 본다.
- `grid.target` 은 근거가 대상을 어떤 면(사물 · 사건 · 기능 · 규칙 · 수치 · 결정)에서 다룰지에 대한 바람이다. 주장과 안 맞으면 따르지 않아도 된다.
- **낱말 겹침으로 칸이 새지 않게** : 네 칸의 근거 길이를 비슷하게(가장 긴 것이 가장 짧은 것의 1.5배 안) 하고,
  주제 낱말(주장의 핵심 낱말 · 분야 낱말)이 네 칸에 비슷하게 나오게 한다. c 도 같은 분야 낱말을 쓰고, a 만 주장 낱말을 많이 겹치게 쓰지 않는다.
- 언어는 씨앗의 `lang` 그대로 (ko = 한국어 · en = English).
- 실제 사람 이름 · 연락처 · 파일 경로 · 비밀값을 새로 지어 넣지 않는다.
- 네 근거는 서로 달라야 하고, 어느 것도 주장과 같은 글이면 안 된다."""
REWRITE_FORBID = ("- 이 안내와 아래 씨앗 파일 말고 다른 파일을 열지 말라.\n"
                  "- 씨앗이 어디서 왔는지(원천 · 자료 이름)를 짐작하지 말라. 글에 적힌 말만 본다.")
REWRITE_OUT = ('씨앗 하나에 네 줄, 칸 차례 a → b → c → d. JSON 한 줄 말고 아무것도 쓰지 않는다.\n'
               '{"seed_id": "s0001", "cell": "a", "evidence": "…"}')


def render_rewrite(criteria, seeds_file, answer_file, count):
    parts = [
        "[① 역할]", REWRITE_ROLE, "",
        "[② 쓰는 규칙]", REWRITE_RULES, "",
        "[③ 칸 기준 전문]", "칸의 뜻은 아래 기준 그대로다. 쓴 근거가 이 기준으로 의도한 칸에 들어가야 한다.",
        "---", criteria.rstrip(), "---", "",
        "[④ 금지]", REWRITE_FORBID, "",
        "[출력]", REWRITE_OUT, "",
        "[씨앗]", f"씨앗 {count} 개가 이 안내와 같은 폴더의 `{seeds_file}` 에 줄마다 하나씩 있다.",
        f"답은 씨앗 차례대로 다시쓰기 폴더 뿌리의 `{answer_file}` 에 쓴다 (부르는 쪽이 다른 자리를 주면 그쪽).", "",
    ]
    return "\n".join(parts)


def cmd_make(a):
    if lb.inside_git(a.out) and not a.allow_git:
        sys.exit("--out 이 git 안이다 — 씨앗 본문은 git 밖에 둔다")
    if os.path.exists(os.path.join(a.out, "manifest.json")):
        sys.exit("--out 에 이미 묶음이 있다 — 새 폴더를 쓴다")
    seeds = lb.read_jsonl(a.seeds)
    empty = [s["seed_id"] for s in seeds if lb.is_empty_pair(s)]
    if empty:
        print(f"멈춤 : 근거나 주장이 빈 씨앗 {len(empty)}줄 (처음 : {empty[:5]})", file=sys.stderr)
        sys.exit(2)
    with open(a.guide, encoding="utf-8") as f:
        criteria = lb.guide_sections(f.read())
    names = []
    for b, start in enumerate(range(0, len(seeds), a.size)):
        name = f"g{b + 1:02d}"
        shown = [{k: s[k] for k in ("seed_id", "lang", "grid", "claim", "evidence")}
                 for s in seeds[start:start + a.size]]
        lb.write_jsonl(os.path.join(a.out, "batches", name + ".jsonl"), shown)
        text = render_rewrite(criteria, name + ".jsonl", f"answers/{name}.out.jsonl", len(shown))
        with open(os.path.join(a.out, "batches", name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        names.append(name)
    os.makedirs(os.path.join(a.out, "answers"), exist_ok=True)
    lb.write_json(os.path.join(a.out, "manifest.json"), {
        "seeds": os.path.abspath(a.seeds), "size": a.size, "guide": os.path.abspath(a.guide), "batches": names})
    print(json.dumps({"batches": len(names), "seeds": len(seeds)}, ensure_ascii=False))


# ---------------------------------------------------------------- 답 거두기

def norm_text(text):
    return " ".join(str(text or "").split())


def check_seed(seed, cells):
    """씨앗 하나의 네 칸 앞뒤 검사. 맞으면 "", 아니면 버린 까닭."""
    if sorted(cells) != list(CELLS):
        return "missing_cell"
    texts = [norm_text(cells[c]) for c in CELLS]
    if any(not t for t in texts):
        return "empty"
    if norm_text(seed["claim"]) in texts:
        return "same_as_claim"
    if norm_text(seed["evidence"]) in texts:
        # 원래 근거를 그대로 옮기면 실물 학습 쌍과 같은 (근거, 주장) 이 다른 칸으로 들 수 있다
        return "same_as_seed"
    if len(set(texts)) < len(CELLS):
        return "same_cells"
    return ""


def read_answers(answer_dir):
    """answers/gNN.out.jsonl → {seed_id: {cell: evidence}} · 같은 칸이 두 번 나온 씨앗 · 못 읽은 줄 수."""
    got = collections.defaultdict(dict)
    dup, bad = set(), 0
    if not os.path.isdir(answer_dir):
        return got, dup, bad
    for name in sorted(os.listdir(answer_dir)):
        if not re.match(r"^g\d+\.out(\.[^.]+)?\.jsonl$", name):
            continue
        with open(os.path.join(answer_dir, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(row, dict) or "seed_id" not in row:
                    bad += 1
                    continue
                sid, cell = str(row["seed_id"]), str(row.get("cell", "")).strip().lower()
                if cell not in CELLS:
                    bad += 1
                    continue
                if cell in got[sid]:
                    dup.add(sid)
                got[sid][cell] = row.get("evidence", "")
    return got, dup, bad


def collect(seeds, answers, dup, label_guide=None):
    """씨앗 + 답 → (후보 쌍 줄, 셈). 어긋난 씨앗은 네 칸을 통째로 버린다."""
    rows, count = [], collections.Counter()
    for seed in seeds:
        sid = seed["seed_id"]
        cells = answers.get(sid, {})
        reason = "no_answer" if not cells else ("dup_cell" if sid in dup else check_seed(seed, cells))
        if reason:
            count[reason] += 1
            continue
        count["kept_seeds"] += 1
        for cell in CELLS:
            rows.append({"id": f"syn-{sid}-{cell}", "evidence": norm_text(cells[cell]), "claim": seed["claim"],
                         "want": "", "hint": cell, "origin": "synth", "src": seed.get("origin", ""),
                         "synth_from": seed["id"], "grid": seed["grid"], "lang": seed.get("lang", ""),
                         "label_guide": label_guide or lb.LABEL_GUIDE, "domain": seed.get("domain", ""),
                         "form": "synth",
                         "group": seed.get("group", ""), "groups": seed.get("groups") or [],
                         "lic": seed.get("lic", "")})
    count["pairs"] = len(rows)
    return rows, count


def cmd_collect(a):
    if lb.inside_git(a.dest) and not a.allow_git:
        sys.exit("--dest 가 git 안이다 — 후보 본문은 git 밖에 둔다")
    with open(os.path.join(a.out, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    seeds = lb.read_jsonl(man["seeds"])
    answers, dup, bad = read_answers(os.path.join(a.out, "answers"))
    known = {s["seed_id"] for s in seeds}
    rows, count = collect(seeds, answers, dup, lb.guide_label(man.get("guide")))
    lb.write_jsonl(a.dest, rows)
    report = {"seeds": len(seeds), **dict(count), "bad_lines": bad,
              "unknown_seed": len(set(answers) - known), "out": a.dest}
    print(json.dumps(report, ensure_ascii=False))


# ---------------------------------------------------------------- 상한

def synth_d_cap(real_d, share=SYNTH_D_SHARE):
    """합성 d ≤ (실물 d + 합성 d) × share → 합성 d 상한 = floor(실물 d × share / (1 − share))."""
    return math.floor(real_d * share / (1 - share) + 1e-9)


def cmd_cap(a):
    cap = synth_d_cap(a.d_total, a.share)
    res = {"real_d": a.d_total, "share": a.share, "max_synth_d": cap,
           # 씨앗 하나가 d 하나를 낸다. 4/5 남김률을 keep_rate 로 보면 씨앗이 이만큼 있어야 상한을 채운다
           "seeds_for_cap": math.ceil(cap / a.keep_rate) if a.keep_rate > 0 else None, "keep_rate": a.keep_rate}
    if a.set:
        got = sum(1 for r in lb.read_jsonl(a.set) if letter(r.get("want")) == "d" and r.get("origin") == "synth")
        res["synth_d"] = got
        res["pass"] = got <= cap
    print(json.dumps(res, ensure_ascii=False))
    if res.get("pass") is False:
        print(f"합성 d {res['synth_d']} > 상한 {cap} — 합성 d 를 줄인다 (설계 5절 ⑥)", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seeds", help="B2 다섯 씨앗 다수결이 gold 와 a↔d · b↔d 로 엇갈린 학습 쌍 → 씨앗")
    s.add_argument("--train", nargs="+", required=True, help="학습 셋 (train-g1-II · train-g1-arch …)")
    s.add_argument("--pred", nargs="+", required=True, help="eval_nli pred.jsonl — 이름 sN-II|arch.pred.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--cap", type=int, default=0,
                   help="씨앗 상한 (넘으면 origin 별 층화 무작위). 값은 `cap --d-total` 의 seeds_for_cap 을 본다")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--allow-git", action="store_true", help="시험용")
    s.set_defaults(func=cmd_seeds)

    s = sub.add_parser("make", help="씨앗 → Opus 다시쓰기 묶음 (gNN.jsonl · gNN.txt)")
    s.add_argument("--seeds", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=SIZE)
    s.add_argument("--guide", default=lb.GUIDE_DEFAULT)
    s.add_argument("--allow-git", action="store_true", help="시험용")
    s.set_defaults(func=cmd_make)

    s = sub.add_parser("collect", help="answers/gNN.out.jsonl → 합성 후보 쌍 (네 칸 앞뒤 검사)")
    s.add_argument("--out", required=True, help="make 의 다시쓰기 폴더")
    s.add_argument("--dest", required=True, help="후보 jsonl (label_batches make --cands 로 줌)")
    s.add_argument("--allow-git", action="store_true", help="시험용")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("cap", help="합성 d 상한 (d 전체의 25%%) 셈")
    s.add_argument("--d-total", type=int, required=True, help="합성 뺀 학습 d 수")
    s.add_argument("--share", type=float, default=SYNTH_D_SHARE)
    s.add_argument("--keep-rate", type=float, default=0.6, help="4/5 남김률 어림 (씨앗 수 셈에만)")
    s.add_argument("--set", default="", help="조립한 합성 학습 셋 — 주면 합성 d 가 상한 안인지 본다 (넘으면 exit 1)")
    s.set_defaults(func=cmd_cap)

    a = p.parse_args(argv)
    if getattr(a, "size", 1) < 1:
        sys.exit("--size 는 1 이상")
    if getattr(a, "cap", 0) < 0:
        sys.exit("--cap 은 0 이상 (0 = 상한 없음)")
    if a.cmd == "cap" and not 0 < a.share < 1:
        sys.exit("--share 는 0 과 1 사이다")
    a.func(a)


if __name__ == "__main__":
    main()
