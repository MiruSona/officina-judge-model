"""assemble_g3 — v1.2 K 판 g3 평가 셋 셋 조립 · 얼림 · 검사 · 규약 해시 채우기.

설계 `../Docs/Design/2026-10-13-v1.2-한국어b보강판설계.md` 2-3 · 5-1. 쌍 본문은 화면에 찍지 않는다. 수는 `../measure/sets-g3.md` 11절.

    python -I assemble_g3.py assemble --set dev-cmp-g3|dev-cal-g3|blind-g3 [--dest ~/.judgemodel/sets/g3] [--draft] \
        [--build ~/.judgemodel/build/g3] [--g2-dir ~/.judgemodel/sets/g2] [--dirs <판정 폴더> …]
    python -I assemble_g3.py freeze --set <셋> [--dest …]
    python -I assemble_g3.py check [--dest …] [--build …] [--draft]
    python -I assemble_g3.py protocol --draft ../measure/protocol-g3.draft.json --dest … --out ../measure/protocol-g3.json

셋 = 이을 g2 셋(dev-cmp-g3 ← dev-cmp-g2 · dev-cal-g3 ← dev-cal-g2 · blind-g3 는 안 잇는다) + 예약 후보
`<build>/reserve/<셋>-cands.jsonl` 중 확정 칸이 있는 줄.

확정 칸 (차례)
  1. 후보 줄의 `judge`(판정 결과) — `override.to`(사람이 되돌린 칸) → `final` → `label`. 위키 · BoolQ · MIT · KLUE 판정 파일 몫.
     줄의 `want` 가 따로 있고 판정과 다르면 멈춘다(exit 2). BoolQ 의 `hint` 는 원래 칸이라 확정 칸으로 안 본다.
  2. 판정 폴더(`label_batches.py decide` 가 낸 `decided.jsonl`, 기본 `<build>/eval-public/<셋>`) — agree · third 는 칸,
     drop 은 「버림」, missing · need-third 와 폴더에 없는 줄은 「대기」.
  3. 둘 다 없으면 「대기」(판정 전 — VitaminC · 새 영어 · c 후보). `--draft` 면 세어서 빼고, 아니면 exit 2.

새 줄 꼴 : `want` · `hint` = 확정 칸, 옛 hint 는 `hint_pick`(이미 있으면 그대로), `label_guide: support-v6`, `set` = 새 셋,
`votes` · `status` 는 판정에서, `g3_label: v6-new`. `judge` 는 뺀다(본문은 예약 파일에 남는다). 이은 g2 줄은 칸 · 라벨 기준을
그대로 두고 `set` 만 새 이름으로, 옛 이름은 `g2_set`, `g3_label: g2-carry`. 셋 전체를 id 차례로 쓴다(g2 와 같은 꼴).
같은 (근거, 주장) 정규형(양쪽 차례)은 하나만 — 새 줄끼리는 id 작은 줄, g2 줄과 겹치면 새 줄을 뺀다. 칸이 다르면 멈춘다.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scrub  # noqa: E402
from label_batches import KEEP_STATUS, LABELS, inside_git, read_jsonl, write_json, write_jsonl  # noqa: E402
from make_sets import judge_pick, sha_key, text_keys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.join(os.path.expanduser("~"), ".judgemodel")
G3_SETS = ("dev-cmp-g3", "dev-cal-g3", "blind-g3")
G2_BASE = {"dev-cmp-g3": "dev-cmp-g2", "dev-cal-g3": "dev-cal-g2", "blind-g3": None}
LABEL_GUIDE = "support-v6"
NEW_TAG, CARRY_TAG = "v6-new", "g2-carry"
KO_B_MIN = {"dev-cmp-g3": 250, "blind-g3": 250}       # 설계 2-3 (dev-cmp-g3 ko b 약 285 · blind-g3 ≥ 250)
WAIT_STATUS = ("missing", "need-third")


class Stop(Exception):
    """조립을 멈춰야 하는 자료 이상 → exit 2."""


# ---------------------------------------------------------------- 열쇠

def groups_of(row):
    return set(row.get("groups") or [row.get("group")]) - {None, ""}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- 확정 칸

def judge_label(row):
    """후보 줄 judge 의 확정 칸과 (votes, status). 없으면 ("", None, None)."""
    judge = row.get("judge")
    if not isinstance(judge, dict):
        return "", None, None
    label = judge_pick(judge)
    if not label:
        return "", None, None
    votes = {}
    for j, alt in (("j1", "label1"), ("j2", "label2"), ("j3", "label3")):
        v = judge.get(j)
        v = v.get("label") if isinstance(v, dict) else v
        v = v if v in LABELS else judge.get(alt)
        if v in LABELS:
            votes[j] = v
    status = judge.get("status")
    if status not in KEEP_STATUS:
        status = "agree" if len(votes) >= 2 and len(set(votes.values())) == 1 else "third"
    return label, votes, status


def load_decisions(dirs):
    """판정 폴더들의 decided.jsonl → {id: 결정 줄}. 같은 id 가 두 폴더에 있으면 멈춘다."""
    out = {}
    for d in dirs:
        path = os.path.join(d, "decided.jsonl")
        if not os.path.exists(path):
            continue
        for dec in read_jsonl(path):
            if dec["id"] in out:
                raise Stop(f"판정 폴더 둘에 같은 id {dec['id']}")
            out[dec["id"]] = dec
    return out


def resolve(cand, decisions):
    """후보 → (칸, votes, status, 상태). 상태 : judge · decided · dropped · pending."""
    label, votes, status = judge_label(cand)
    if label:
        want = cand.get("want")
        if want in LABELS and want != label:
            raise Stop(f"{cand['id']} : want {want} ≠ 판정 {label}")
        return label, votes, status, "judge"
    dec = decisions.get(cand["id"])
    if dec is None or dec.get("status") in WAIT_STATUS:
        return "", None, None, "pending"
    if dec.get("status") not in KEEP_STATUS:
        return "", None, None, "dropped"
    if dec.get("label") not in LABELS:
        raise Stop(f"{cand['id']} : 평가 셋에 모르는 칸 {dec.get('label')!r}")
    return dec["label"], dict(dec.get("votes") or {}), dec["status"], "decided"


def new_row(cand, label, votes, status, set_name):
    row = {k: v for k, v in cand.items() if k not in ("judge", "g3_label")}
    if "hint_pick" not in row:
        row["hint_pick"] = cand.get("hint", "")
    row.update(want=label, hint=label, label_guide=LABEL_GUIDE, set=set_name, votes=votes or {}, status=status,
               g3_label=NEW_TAG)
    return row


def carry_row(row, set_name):
    return dict(row, set=set_name, g2_set=row.get("set", ""), g3_label=CARRY_TAG)


# ---------------------------------------------------------------- 조립

def assemble(set_name, cands, g2_rows, decisions, draft):
    """→ (셋 줄, 요약). 판정 전 줄은 draft 면 세어서 빼고 아니면 Stop."""
    by_state = collections.Counter()
    pending = collections.Counter()
    dropped = collections.Counter()
    new = []
    for cand in cands:
        if cand.get("set") and cand["set"] != set_name:
            raise Stop(f"{cand['id']} : 후보의 셋 {cand['set']} ≠ {set_name}")
        if not str(cand.get("evidence") or "").strip() or not str(cand.get("claim") or "").strip():
            raise Stop(f"{cand['id']} : 근거나 주장이 비었다")
        label, votes, status, state = resolve(cand, decisions)
        by_state[state] += 1
        if state == "pending":
            pending[(cand.get("domain", ""), cand.get("hint", ""))] += 1
            continue
        if state == "dropped":
            dropped[cand.get("domain", "")] += 1
            continue
        new.append(new_row(cand, label, votes, status, set_name))
    if pending and not draft:
        raise Stop(f"{set_name} : 판정 전 줄 {sum(pending.values())} — 판정을 끝내거나 --draft")
    carried = [carry_row(r, set_name) for r in g2_rows]
    # 같은 (근거, 주장) 정규형 — g2 줄 먼저 잡고, 새 줄끼리는 id 작은 줄
    seen, deduped = {}, []
    for r in carried:
        for k in text_keys(r):
            seen.setdefault(k, r)
    kept = []
    for r in sorted(new, key=lambda r: r["id"]):
        hit = next((seen[k] for k in text_keys(r) if k in seen), None)
        if hit is not None:
            if hit["want"] != r["want"]:
                raise Stop(f"{r['id']} : 같은 (근거, 주장) 인 {hit['id']} 와 칸이 다르다 ({r['want']} ≠ {hit['want']})")
            deduped.append(r["id"])
            continue
        for k in text_keys(r):
            seen[k] = r
        kept.append(r)
    rows = sorted(carried + kept, key=lambda r: r["id"])
    ids = collections.Counter(r["id"] for r in rows)
    dup = sorted(i for i, n in ids.items() if n > 1)
    if dup:
        raise Stop(f"{set_name} : 같은 id {len(dup)} (예 {dup[0]})")
    summary = summarize(set_name, rows, draft)
    summary.update({
        "g2_base": G2_BASE[set_name], "g2_pairs": len(carried), "new_pairs": len(kept),
        "cands": len(cands), "cand_state": dict(sorted(by_state.items())),
        "pending": sum(pending.values()),
        "pending_by": {f"{d} · hint {h}": n for (d, h), n in sorted(pending.items())},
        "dropped": dict(sorted(dropped.items())), "deduped": deduped,
    })
    return rows, summary


def table(rows, key):
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        out[r.get(key) or "-"][r["want"]] += 1
    return {k: dict(sorted(c.items())) for k, c in sorted(out.items())}


def summarize(set_name, rows, draft):
    new = [r for r in rows if r.get("g3_label") == NEW_TAG]
    return {
        "set": set_name, "draft": bool(draft), "pairs": len(rows),
        "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
        "domain_want": table(rows, "domain"), "lang_want": table(rows, "lang"),
        "new_domain_want": table(new, "domain"), "new_lang_want": table(new, "lang"),
        "ko_b": sum(1 for r in rows if r.get("lang") == "ko" and r["want"] == "b"),
        "label_guides": dict(sorted(collections.Counter(r.get("label_guide", "") for r in rows).items())),
        "g3_label": dict(sorted(collections.Counter(r["g3_label"] for r in rows).items())),
    }


def pending_quota(build, set_name):
    """reserve-g3.json 의 pending — 판정 파일이 아직 없어 줄도 없는 원천 몫(KLUE)."""
    path = os.path.join(build, "reserve", "reserve-g3.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("pending", {}).get(set_name, {})


def cmd_assemble(a):
    if inside_git(a.dest):
        sys.exit("--dest 가 git 안이다 — 셋 본문은 git 밖에 둔다")
    dirs = a.dirs if a.dirs is not None else [os.path.join(a.build, "eval-public", a.set)]
    cands = read_jsonl(os.path.join(a.build, "reserve", f"{a.set}-cands.jsonl"))
    base = G2_BASE[a.set]
    g2_rows = read_jsonl(os.path.join(a.g2_dir, f"{base}.jsonl")) if base else []
    try:
        rows, summary = assemble(a.set, cands, g2_rows, load_decisions(dirs), a.draft)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    if base:
        summary["g2_sha256"] = sha256_file(os.path.join(a.g2_dir, f"{base}.jsonl"))
    summary["pending_quota"] = pending_quota(a.build, a.set)
    summary["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    path = os.path.join(a.dest, f"{a.set}.jsonl")
    write_jsonl(path, rows)
    summary["sha256"] = sha256_file(path)
    write_json(os.path.join(a.dest, f"{a.set}.summary.json"), summary)
    keys = ("set", "draft", "pairs", "g2_pairs", "new_pairs", "want", "ko_b", "pending", "pending_quota",
            "dropped", "deduped", "sha256")
    print(json.dumps({k: (len(summary[k]) if k == "deduped" else summary[k]) for k in keys}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- 얼림

def freeze_tool(*args):
    cmd = [sys.executable, "-I", os.path.join(HERE, "..", "train", "freeze.py"), *args]
    got = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return got.returncode, got.stdout.strip() or got.stderr.strip()


def cmd_freeze(a):
    pairs = os.path.join(a.dest, f"{a.set}.jsonl")
    frozen = os.path.join(a.dest, f"FROZEN-{a.set}.sha256")
    with open(os.path.join(a.dest, f"{a.set}.summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("draft") or summary.get("pending"):
        print("멈춤 : 미리 조립(draft) 한 셋이거나 판정 전 줄이 남았다 — 얼리지 않는다", file=sys.stderr)
        return 2
    if summary.get("sha256") != sha256_file(pairs):
        print("멈춤 : 셋 파일이 조립 뒤 바뀌었다 (summary sha256 ≠ 파일)", file=sys.stderr)
        return 2
    if os.path.exists(frozen):
        print(f"멈춤 : 이미 얼렸다 {frozen}", file=sys.stderr)
        return 2
    code, out = freeze_tool("write", "--pairs", pairs, "--out", frozen)
    print(out)
    if code:
        return code
    code, out = freeze_tool("check", "--pairs", pairs, "--frozen", frozen)
    print(out)
    return code


# ---------------------------------------------------------------- 검사

def overlap_counts(sets, train_rows):
    """셋 사이 · 학습 층 × 셋 겹침 수 (글 = 정규형 양쪽 차례 · 묶음 · 쌍 열쇠 · 출처 id)."""
    counts = collections.Counter()
    owner = {"text": {}, "group": {}, "key": {}, "src": {}}
    for set_name, rows in sets.items():
        for r in rows:
            feats = {"text": text_keys(r), "group": groups_of(r), "key": {sha_key(r)}, "src": set(r.get("src_ids") or [])}
            for kind, vals in feats.items():
                hit = False
                for v in vals:
                    if owner[kind].setdefault(v, set_name) != set_name:
                        hit = True
                if hit:
                    counts[f"sets_{kind}"] += 1
    for r in train_rows:
        feats = {"text": text_keys(r), "group": groups_of(r), "key": {sha_key(r)}, "src": set(r.get("src_ids") or [])}
        for kind, vals in feats.items():
            if any(v in owner[kind] for v in vals):
                counts[f"train_{kind}"] += 1
    names = [f"{w}_{k}" for w in ("sets", "train") for k in ("text", "group", "key", "src")]
    return {n: counts.get(n, 0) for n in names}


def scrub_hits(rows, names, game_names):
    s = scrub.Scrubber(scrub.read_names(names), scrub.read_names(game_names))
    total = collections.Counter()
    for r in rows:
        total.update(s.hits(r))
    return sum(total.values())


def cmd_check(a):
    sets = {}
    for name in G3_SETS:
        path = os.path.join(a.dest, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"멈춤 : 셋이 없다 {path}", file=sys.stderr)
            return 2
        sets[name] = read_jsonl(path)
    train_rows = []
    for rel in a.train:
        path = os.path.join(a.build, rel)
        if os.path.exists(path):
            train_rows.extend(read_jsonl(path))
            continue
        if not a.draft:
            print(f"멈춤 : 학습 층 후보가 없어 겹침을 못 잰다 {path} (초안 검사는 --draft)", file=sys.stderr)
            return 2
        print(f"(학습 층 후보 없음 : {rel})", file=sys.stderr)
    overlap = overlap_counts(sets, train_rows)
    result = {"overlap": overlap, "train_rows": len(train_rows), "sets": {}}
    bad = sum(overlap.values())
    short = []
    for name, rows in sets.items():
        hits = scrub_hits(rows, a.names, a.game_names)
        bad += hits
        ko_b = sum(1 for r in rows if r.get("lang") == "ko" and r["want"] == "b")
        need = KO_B_MIN.get(name)
        if need and ko_b < need:
            short.append(name)
        result["sets"][name] = {"pairs": len(rows), "ko_b": ko_b, "ko_b_min": need, "scrub_hits": hits,
                                "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
                                "lang_want": table(rows, "lang"), "domain_want": table(rows, "domain")}
    result["ko_b_short"] = short
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if bad:
        return 1
    if short and not a.draft:
        return 1
    return 0


# ---------------------------------------------------------------- 규약 해시

def fill_protocol(draft_text, hashes):
    """초안 바이트에서 "TODO-<셋>" 자리만 sha256 으로 바꾼다(그 밖 글자는 그대로). 빠진 자리 · 남은 TODO 면 Stop."""
    out = draft_text
    for name, digest in hashes.items():
        token = f'"TODO-{name}"'
        if out.count(token) != 1:
            raise Stop(f"초안에 {token} 가 한 번 있어야 한다 ({out.count(token)})")
        out = out.replace(token, f'"{digest}"')
    doc = json.loads(out)
    left = [k for k, v in doc.get("sets", {}).items() if str(v).startswith("TODO")]
    if left:
        raise Stop(f"TODO 가 남았다 {left}")
    if {k: doc["sets"].get(k) for k in hashes} != hashes:
        raise Stop("채운 sets 가 해시와 다르다")
    return out


def cmd_protocol(a):
    if os.path.exists(a.out):
        print(f"멈춤 : {a.out} 이 이미 있다 — 규약은 첫 비교 뒤 안 고친다(hash_rule)", file=sys.stderr)
        return 2
    hashes = {}
    for name in G3_SETS:
        pairs = os.path.join(a.dest, f"{name}.jsonl")
        frozen = os.path.join(a.dest, f"FROZEN-{name}.sha256")
        if not os.path.exists(frozen):
            print(f"멈춤 : 얼리지 않은 셋 {name}", file=sys.stderr)
            return 2
        code, out = freeze_tool("check", "--pairs", pairs, "--frozen", frozen)
        if code:
            print(f"멈춤 : {name} 해시가 FROZEN 과 다르다 {out}", file=sys.stderr)
            return 2
        hashes[name] = sha256_file(pairs)
    with open(a.draft_protocol, encoding="utf-8", newline="") as f:
        text = f.read()
    try:
        out = fill_protocol(text, hashes)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(json.dumps({"out": a.out, "sets": hashes, "protocol_sha256": sha256_file(a.out)}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="g3 평가 셋 조립 · 얼림 · 검사")
    sub = p.add_subparsers(dest="cmd", required=True)
    dest = os.path.join(HOME, "sets", "g3")
    build = os.path.join(HOME, "build", "g3")
    s = sub.add_parser("assemble")
    s.add_argument("--set", required=True, choices=G3_SETS)
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--g2-dir", default=os.path.join(HOME, "sets", "g2"))
    s.add_argument("--dirs", nargs="*", default=None, help="판정 폴더 (decided.jsonl). 기본 <build>/eval-public/<셋>")
    s.add_argument("--draft", action="store_true", help="판정 전 줄을 세어서 빼고 조립 (얼리지 않는다)")
    s = sub.add_parser("freeze")
    s.add_argument("--set", required=True, choices=G3_SETS)
    s.add_argument("--dest", default=dest)
    s = sub.add_parser("check")
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--train", nargs="*", default=["train-g3-ko/cands.jsonl", "reserve/train-layer-cands.jsonl"])
    s.add_argument("--names", nargs="*", default=[os.path.join(HOME, "build", "g1", "names.txt")])
    s.add_argument("--game-names", nargs="*", default=[os.path.join(HOME, "build", "g1", "game-names.txt")])
    s.add_argument("--draft", action="store_true", help="ko b 모자람 · 학습 층 후보 없음을 실패로 안 본다")
    s = sub.add_parser("protocol")
    s.add_argument("--draft", dest="draft_protocol", required=True, help="protocol-g3.draft.json")
    s.add_argument("--dest", default=dest)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"assemble": cmd_assemble, "freeze": cmd_freeze, "check": cmd_check, "protocol": cmd_protocol}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
