"""assemble_g4 — v1.2 K2 판 g4 평가 셋 둘 조립 · 얼림 · 검사(min_groups) · 규약 해시 채우기.

설계 `../Docs/Design/2026-10-14-v1.2-K2-한국어c보강판설계.md` 3-2 · 3-4 · 3-5 · 5절 ⑤. 쌍 본문은 화면에 찍지 않는다.

    python -I assemble_g4.py assemble --set dev-cmp-g4|blind-g4 [--dest ~/.judgemodel/sets/g4] [--draft] \
        [--build ~/.judgemodel/build/g4] [--g3-dir ~/.judgemodel/sets/g3] [--dirs <판정 폴더> …]
    python -I assemble_g4.py freeze --set <셋> [--dest …]
    python -I assemble_g4.py check [--dest …] [--build …] [--g3-dir …] [--protocol ../measure/protocol-g4.draft.json] [--draft]
    python -I assemble_g4.py protocol --draft ../measure/protocol-g4.draft.json --dest … --out ../measure/protocol-g4.json
    python -I assemble_g4.py protocol --draft ../measure/protocol-g5.draft.json --dest … --out ../measure/protocol-g5.json   (K3)

셋 = 이을 g3 셋(dev-cmp-g4 ← dev-cmp-g3 · blind-g4 는 안 잇는다) + 예약 후보 `<build>/reserve/<셋>-cands.jsonl` 중 확정 칸이 있는 줄.
확정 칸 · 새 줄 꼴 · 중복 규칙은 assemble_g3 와 같다. 꼬리표만 g4 다 — 새 줄 `g4_label: v6-new`,
이은 g3 줄은 칸 · 라벨 기준을 그대로 두고 `set` 만 새 이름으로, 옛 이름은 `g3_set`, `g4_label: g3-carry`.
check 는 규약 초안 release_gate.min_groups 를 읽어 그 셋(blind-g4)의 칸별 묶음 수가 모자라면 exit 1 — 얼리기를 막는다.
protocol 은 초안의 빈 "" 해시 자리(sets 의 g4 셋 둘 · train_sets 의 train-g4-c · -h · -d 중 초안에 있는 것)만 채운다.
이미 해시가 적힌 자리(protocol-g5 초안의 g4 셋 · train-g4-c 같은 것)는 얼린 파일과 같은지만 본다.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemble_g3 import (Gen, Stop, assemble, cmd_freeze, freeze_tool, load_decisions, overlap_counts,  # noqa: E402
                         scrub_hits, sha256_file, table)
from label_batches import inside_git, read_jsonl, write_json, write_jsonl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.join(os.path.expanduser("~"), ".judgemodel")
G4_SETS = ("dev-cmp-g4", "blind-g4")
G4 = Gen("g4_label", "g3_set", "g3-carry", {"dev-cmp-g4": "dev-cmp-g3", "blind-g4": None}, "g3")
G3_SIDE_SETS = ("dev-cal-g3", "blind-g3")         # 겹침만 같이 본다 (dev-cal-g3 는 g4 규약에서도 쓴다)
TRAIN_SETS = ("train-g4-c", "train-g4-c-h", "train-g4-d")   # train/freeze.py 로 얼린 학습 층 — 규약 train_sets 자리
DRAFT = os.path.join(HERE, "..", "measure", "protocol-g4.draft.json")


def group_counts(min_groups, rows, group_key="group"):
    """min_groups 자 이름(칸 글자 · 꼬리 @lang) → 그 칸(want) 줄의 묶음 수. compare.group_counts 와 같은 셈."""
    out = {}
    for name in min_groups:
        letter, _, lang = name.partition("@")
        out[name] = len({str(r.get(group_key) or r["id"]) for r in rows
                         if r.get("want") == letter and (not lang or str(r.get("lang")) == lang)})
    return out


def read_gate(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc["release_gate"], doc.get("group_key", "group")


def pending_quota(build, set_name):
    """reserve-g4.json 의 pending — 판정 파일이 아직 없어 줄도 없는 원천 몫(blind-g4 ko b)."""
    path = os.path.join(build, "reserve", "reserve-g4.json")
    if set_name != "blind-g4" or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("pending", {})


def cmd_assemble(a):
    if inside_git(a.dest):
        sys.exit("--dest 가 git 안이다 — 셋 본문은 git 밖에 둔다")
    dirs = a.dirs if a.dirs is not None else [os.path.join(a.build, "eval-public", a.set)]
    cands = read_jsonl(os.path.join(a.build, "reserve", f"{a.set}-cands.jsonl"))
    base = G4.base[a.set]
    g3_rows = read_jsonl(os.path.join(a.g3_dir, f"{base}.jsonl")) if base else []
    try:
        rows, summary = assemble(a.set, cands, g3_rows, load_decisions(dirs), a.draft, G4)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    if base:
        summary["g3_sha256"] = sha256_file(os.path.join(a.g3_dir, f"{base}.jsonl"))
    summary["pending_quota"] = pending_quota(a.build, a.set)
    summary["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    os.makedirs(a.dest, exist_ok=True)
    path = os.path.join(a.dest, f"{a.set}.jsonl")
    write_jsonl(path, rows)
    summary["sha256"] = sha256_file(path)
    write_json(os.path.join(a.dest, f"{a.set}.summary.json"), summary)
    keys = ("set", "draft", "pairs", "g3_pairs", "new_pairs", "want", "ko_b", "pending", "pending_quota",
            "dropped", "deduped", "sha256")
    print(json.dumps({k: (len(summary[k]) if k == "deduped" else summary[k]) for k in keys}, ensure_ascii=False))
    return 0


def cmd_check(a):
    sets = {}
    paths = {n: os.path.join(a.dest, f"{n}.jsonl") for n in G4_SETS}
    paths.update({n: os.path.join(a.g3_dir, f"{n}.jsonl") for n in G3_SIDE_SETS})
    for name, path in paths.items():
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
            print(f"멈춤 : 학습 층이 없어 겹침을 못 잰다 {path} (초안 검사는 --draft)", file=sys.stderr)
            return 2
        print(f"(학습 층 없음 : {rel})", file=sys.stderr)
    gate, group_key = read_gate(a.protocol)
    want = gate.get("min_groups") or {}
    have = group_counts(want, sets.get(gate.get("set"), []), group_key)
    short = {name: [have[name], need] for name, need in want.items() if have[name] < need}
    overlap = overlap_counts(sets, train_rows)
    result = {"overlap": overlap, "train_rows": len(train_rows), "sets": {},
              "min_groups": {"set": gate.get("set"), "have": have, "need": want, "short": short}}
    bad = sum(overlap.values())
    for name in G4_SETS:
        rows = sets[name]
        hits = scrub_hits(rows, a.names, a.game_names)
        bad += hits
        result["sets"][name] = {"pairs": len(rows), "scrub_hits": hits,
                                "want": dict(sorted(collections.Counter(r["want"] for r in rows).items())),
                                "lang_want": table(rows, "lang"), "domain_want": table(rows, "domain")}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if bad:
        return 1
    if short and not a.draft:
        return 1
    return 0


def fill_protocol(draft_text, hashes):
    """초안 바이트에서 `"<이름>": ""` 자리만 sha256 으로 바꾼다(그 밖 글자는 그대로). 자리가 한 번이 아니거나 빈 칸이 남으면 Stop."""
    out = draft_text
    for name, digest in hashes.items():
        token = f'"{name}": ""'
        if out.count(token) != 1:
            raise Stop(f"초안에 {token} 가 한 번 있어야 한다 ({out.count(token)})")
        out = out.replace(token, f'"{name}": "{digest}"')
    doc = json.loads(out)
    left = [k for part in ("sets", "train_sets") for k, v in doc.get(part, {}).items() if not v or "TODO" in str(v)]
    if left:
        raise Stop(f"빈 해시가 남았다 {left}")
    filled = {**doc.get("sets", {}), **doc.get("train_sets", {})}
    if {k: filled.get(k) for k in hashes} != hashes:
        raise Stop("채운 해시가 얼린 파일과 다르다")
    return out


def frozen_hash(dest, name):
    """FROZEN-<이름>.sha256 과 맞는 파일의 sha256. 안 얼렸거나 다르면 Stop."""
    pairs = os.path.join(dest, f"{name}.jsonl")
    frozen = os.path.join(dest, f"FROZEN-{name}.sha256")
    if not os.path.exists(frozen):
        raise Stop(f"얼리지 않은 파일 {name}")
    code, out = freeze_tool("check", "--pairs", pairs, "--frozen", frozen)
    if code:
        raise Stop(f"{name} 해시가 FROZEN 과 다르다 {out}")
    return sha256_file(pairs)


def slots_to_fill(doc, dest):
    """초안의 g4 셋 · 학습 층 자리 중 빈 "" 자리 → {이름: 얼린 sha256}. 해시가 적힌 자리는 얼린 파일과 다르면 Stop, 초안에 없는 이름은 건너뛴다."""
    written = {**doc.get("sets", {}), **doc.get("train_sets", {})}
    hashes = {}
    for name in G4_SETS + TRAIN_SETS:
        if name not in written:
            continue
        digest = frozen_hash(dest, name)
        if written[name] == "":
            hashes[name] = digest
        elif written[name] != digest:
            raise Stop(f"초안의 {name} 해시가 얼린 파일과 다르다")
    return hashes


def cmd_protocol(a):
    if os.path.exists(a.out):
        print(f"멈춤 : {a.out} 이 이미 있다 — 규약은 첫 비교 뒤 안 고친다(hash_rule)", file=sys.stderr)
        return 2
    with open(a.draft_protocol, encoding="utf-8", newline="") as f:
        text = f.read()
    try:
        hashes = slots_to_fill(json.loads(text), a.dest)
        cal = json.loads(text)["sets"].get("dev-cal-g3")
        if cal != frozen_hash(a.g3_dir, "dev-cal-g3"):
            raise Stop("초안의 dev-cal-g3 해시가 얼린 dev-cal-g3 와 다르다")
        out = fill_protocol(text, hashes)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(json.dumps({"out": a.out, "hashes": hashes, "protocol_sha256": sha256_file(a.out)}, ensure_ascii=False))
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="g4 평가 셋 조립 · 얼림 · 검사")
    sub = p.add_subparsers(dest="cmd", required=True)
    dest = os.path.join(HOME, "sets", "g4")
    build = os.path.join(HOME, "build", "g4")
    g3_dir = os.path.join(HOME, "sets", "g3")
    s = sub.add_parser("assemble")
    s.add_argument("--set", required=True, choices=G4_SETS)
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--g3-dir", default=g3_dir)
    s.add_argument("--dirs", nargs="*", default=None, help="판정 폴더 (decided.jsonl). 기본 <build>/eval-public/<셋>")
    s.add_argument("--draft", action="store_true", help="판정 전 줄을 세어서 빼고 조립 (얼리지 않는다)")
    s = sub.add_parser("freeze")
    s.add_argument("--set", required=True, choices=G4_SETS)
    s.add_argument("--dest", default=dest)
    s = sub.add_parser("check")
    s.add_argument("--dest", default=dest)
    s.add_argument("--build", default=build)
    s.add_argument("--g3-dir", default=g3_dir)
    s.add_argument("--protocol", default=DRAFT, help="min_groups 를 읽을 규약 (기본 measure/protocol-g4.draft.json)")
    s.add_argument("--train", nargs="*", default=["train-g4-c/cands.jsonl", "train-g4-c/cands-h.jsonl",
                                                  "train-g4-d/cands.jsonl", os.path.join(g3_dir, "train-g3-ko.jsonl")],
                   help="학습 층 (<build> 기준 상대 경로 또는 절대 경로)")
    s.add_argument("--names", nargs="*", default=[os.path.join(HOME, "build", "g1", "names.txt")])
    s.add_argument("--game-names", nargs="*", default=[os.path.join(HOME, "build", "g1", "game-names.txt")])
    s.add_argument("--draft", action="store_true", help="min_groups 모자람 · 학습 층 없음을 실패로 안 본다")
    s = sub.add_parser("protocol")
    s.add_argument("--draft", dest="draft_protocol", required=True, help="protocol-g4.draft.json · protocol-g5.draft.json")
    s.add_argument("--dest", default=dest)
    s.add_argument("--g3-dir", default=g3_dir, help="dev-cal-g3 · FROZEN-dev-cal-g3.sha256 이 있는 곳")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"assemble": cmd_assemble, "freeze": cmd_freeze, "check": cmd_check, "protocol": cmd_protocol}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
