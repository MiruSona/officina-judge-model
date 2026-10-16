"""labels · freeze — 라벨 묶음(build/label_batches.py 감싸기) · 사람 목록 · 얼리기 (설계 6절 · 2-3 freeze).
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
사람 답은 `<라벨 폴더>/human.jsonl` 에 {"id","label"} 로 적고 Opus 합의를 이긴다. 얼린 쌍 열쇠는 use1/EXCLUDE.sha256list 에 더한다.
"""
import json
import os
import subprocess
import sys

from use_sites import common

LABEL_BATCHES = os.path.join(common.ROOT, "build", "label_batches.py")
GOOD_STATUS = ("agree", "third")
LABELS = ("a", "b", "c", "d")
PAIRS_FILE = "pairs.jsonl"   # labels make 때 cands 의 {id, pair} 사본 — freeze 가 라벨과 cands 를 대조한다


def run_batches(args):
    done = subprocess.run([sys.executable, "-I", LABEL_BATCHES] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        raise common.Refuse(f"label_batches {args[0]} 가 {done.returncode} 로 끝났다")
    return done.stdout


def outcomes(row, models, thresholds, rule):
    return {m: common.place_outcome(row, m, thresholds, rule) for m in models}


def human_todo(cands, decided, models, thresholds, rule=None):
    """사람이 볼 쌍 (11-1 줄인 판) — 두 모델 결과가 갈린 쌍 + Opus 가 못 가른 쌍."""
    out = []
    for row in cands:
        got = outcomes(row, models, thresholds, rule)
        known = [v for v in got.values() if v is not None]
        why = []
        if len(known) >= 2 and len(set(known)) > 1:
            why.append("모델 갈림")
        d = decided.get(row["id"])
        if d is not None and d["status"] not in GOOD_STATUS:
            why.append("Opus " + d["status"])
        if why:
            out.append({"id": row["id"], "pair": row.get("pair"), "place": row["place"], "evidence": row["evidence"],
                        "claim": row["claim"], "opus": (d or {}).get("label"), "why": why})
    return out


def place_arg(a):
    return getattr(a, "place", None)


def cmd_labels(a):
    paths = common.part_paths(a.alias, place_arg(a))
    out, cands = paths["labels"], paths["cands"]
    if a.step == "make":
        args = ["make", "--cands", cands, "--out", out, "--mode", "eval", "--judges", "2"]
        if a.traps:
            args += ["--traps", a.traps]
        run_batches(args)
        common.write_jsonl(os.path.join(out, PAIRS_FILE),
                           [{"id": r["id"], "pair": r["pair"]} for r in common.read_jsonl(cands)])
        return
    if a.step in ("merge", "third"):
        run_batches([a.step, "--out", out])
        return
    if a.step == "decide":
        run_batches(["decide", "--out", out, "--mode", "eval"])
        return
    if a.step == "kappa":
        text = run_batches(["kappa", "--out", out])
        common.write_json(os.path.join(out, "kappa.json"), json.loads(text.splitlines()[0]))
        return
    protocol, _ = common.load_protocol(a.protocol)
    meta = common.read_json(paths["meta"])
    decided = {r["id"]: r for r in common.read_jsonl(os.path.join(out, "decided.jsonl"))}
    todo = human_todo(common.read_jsonl(cands), decided, list(meta["models"]), meta["thresholds"]["repo"],
                      common.unsure_rule_of(protocol))
    path = os.path.join(out, "human-todo.jsonl")
    common.write_jsonl(path, todo)
    print(f"사람 몫 {len(todo)}쌍 → {path} · 답은 {os.path.join(out, 'human.jsonl')} 에 {{\"id\",\"label\"}} 로")


def final_labels(alias, place=None, override=""):
    """id → (칸, 출처, 라벨 때 쌍 열쇠 또는 None). 차례 : Opus 합의 → 사람 답 → --labels(시험 · 손 라벨)."""
    out = common.part_paths(alias, place)["labels"]
    snapshot = {r["id"]: r.get("pair") for r in common.read_jsonl(os.path.join(out, PAIRS_FILE))}
    labels = {}
    for r in common.read_jsonl(os.path.join(out, "decided.jsonl")):
        if r["status"] in GOOD_STATUS and r["label"] in LABELS:
            labels[r["id"]] = (r["label"], "opus", snapshot.get(r["id"]))
    changed = 0
    human = common.read_jsonl(os.path.join(out, "human.jsonl"))
    for r in human:
        if r["label"] not in LABELS:
            raise common.Refuse(f"사람 답 칸이 틀렸다 : {r}")
        if r["id"] in labels and labels[r["id"]][0] != r["label"]:
            changed += 1
        labels[r["id"]] = (r["label"], "human", r.get("pair") or snapshot.get(r["id"]))
    if override:
        for r in common.read_jsonl(override):
            labels[r["id"]] = (r["label"], "given", r.get("pair"))
    return labels, {"human": len(human), "human_changed": changed}, set(snapshot)


def check_label_pairs(cands, labels, snapshot_ids):
    """라벨 줄의 쌍 열쇠가 지금 cands 의 쌍과 같은지. 다른 판의 라벨이 섞였으면 멈춘다."""
    by_id = {r["id"]: r for r in cands}
    for label_id, (_, src, pair) in labels.items():
        row = by_id.get(label_id)
        if row is None:
            # 함정 쌍은 decided 에 남을 수 있다 — 라벨 때 cands 에 없던 Opus 줄만 건너뛴다
            if src == "opus" and label_id not in snapshot_ids:
                continue
            raise common.Refuse(f"라벨 {label_id} 의 쌍이 지금 cands 에 없다 — 다른 sample · draw 판의 라벨이다")
        if pair and pair != row["pair"]:
            raise common.Refuse(f"라벨 {label_id} 의 쌍 열쇠가 cands 와 다르다 — 다시 뽑은 뒤 옛 라벨이 붙었다")


def cmd_freeze(a):
    place = place_arg(a)
    _, protocol_sha = common.load_protocol(a.protocol)
    paths = common.part_paths(a.alias, place)
    if os.path.isfile(paths["mark"]):
        raise common.Refuse("이미 얼렸다 — 다시 안 얼린다 (지킬 규칙 ①)")
    meta_in = common.read_json(paths["meta"])
    if meta_in.get("protocol_sha256") != protocol_sha:
        raise common.Refuse("표본 뽑을 때 규약과 지금 규약 해시가 다르다")
    cands = common.read_jsonl(paths["cands"])
    labels, stats, snapshot_ids = final_labels(a.alias, place, a.labels)
    check_label_pairs(cands, labels, snapshot_ids)
    missing = [r["id"] for r in cands if r["id"] not in labels]
    if missing:
        raise common.Refuse(f"라벨 없는 쌍 {len(missing)} (처음 {missing[:5]}) — 사람 몫(labels todo)을 채운다")
    rows = [dict(r, label=labels[r["id"]][0], label_src=labels[r["id"]][1]) for r in cands]
    common.write_jsonl(paths["set"], rows)
    set_sha = common.sha256_file(paths["set"])
    frozen_at = common.now_iso()
    kappa_path = os.path.join(paths["labels"], "kappa.json")
    meta = {"frozen_at": frozen_at, "alias": a.alias, "place": place, "set_sha256": set_sha,
            "protocol_sha256": protocol_sha, "pairs": len(rows), "thresholds": meta_in["thresholds"],
            "models": meta_in["models"], "queues": meta_in.get("queues"), "reconcile": meta_in.get("reconcile"),
            "memories": meta_in.get("memories"), "collect": meta_in.get("collect"), "pool": meta_in.get("pool"),
            "seen_pairs": sum(1 for r in rows if r.get("seen")), "labels": stats,
            "kappa": common.read_json(kappa_path) if os.path.isfile(kappa_path) else None}
    common.write_json(paths["frozen"], meta)
    with open(paths["mark"], "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{set_sha}  {os.path.basename(paths['set'])}\n")
    keys = [r["pair"] for r in rows]
    if place is not None:
        # draw 가 물은 쌍 전부 — 뽑히지 않은 쌍도 이 판의 측정에 쓰였으니 학습 풀에서 뺀다
        keys += [r["pair"] for r in common.read_jsonl(paths["queries"]) if r.get("pair")]
    added = add_exclude(keys)
    print(f"얼림 {len(rows)}쌍 · {set_sha[:12]} · frozen_at {frozen_at} · EXCLUDE 에 {added}줄 더함")
    return meta


def add_exclude(keys):
    """use1/EXCLUDE.sha256list 에 없는 열쇠만 덧붙인다. 학습 캐기 스크립트가 이 파일로 쌍을 뺀다."""
    path = common.shared_path("EXCLUDE.sha256list")
    have = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            have = {line.strip() for line in f if line.strip()}
    new = [k for k in dict.fromkeys(keys) if k not in have]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for k in new:
            f.write(k + "\n")
    return len(new)


def register(sub):
    s = sub.add_parser("labels", help="라벨 묶음 (make · merge · third · decide · kappa · todo)")
    s.add_argument("step", choices=("make", "merge", "third", "decide", "kappa", "todo"))
    s.add_argument("--alias", required=True)
    s.add_argument("--place", type=int, choices=(4, 5), default=None, help="④ ⑤ 판 (cands-N.jsonl · labels-N/)")
    s.add_argument("--traps", default="", help="함정 풀 (g1 것이 있으면)")
    s.add_argument("--protocol", default="")
    s.set_defaults(func=cmd_labels)

    s = sub.add_parser("freeze", help="라벨 붙은 셋을 얼린다 (별명 · 판마다 한 번)")
    s.add_argument("--alias", required=True)
    s.add_argument("--place", type=int, choices=(4, 5), default=None, help="④ ⑤ 판 (set-use1-N.jsonl)")
    s.add_argument("--labels", default="", help="손 라벨 jsonl {id,label[,pair]} — 합의 · 사람 답 위에 덮는다")
    s.add_argument("--protocol", default="")
    s.set_defaults(func=cmd_freeze)
