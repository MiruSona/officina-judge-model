"""draw — ④ ⑤ 수집 로그에서 두 모델에 묻고 층으로 뽑아 cands-N.jsonl 을 만든다 (설계 4절 · 7절).
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
물음 기록은 queries-draw-N.jsonl 에 따로 둔다(얼리면 그 쌍 열쇠가 EXCLUDE 로 간다).
"""
import os
import random

from use_sites import common
from use_sites.sample import Session, draw, finish_rows, seen_keys, seen_roots, warn_no_answer

PLACE_FILES = {4: ("collect", "report-pairs.jsonl"), 5: ("collect", "harness-pairs.jsonl")}
PLACE_KEYS = {4: "4-report", 5: "5-harness"}
PLACE_LETTER = {4: "b", 5: "a"}
HARNESS_ALIAS = "harness"


def sure_by(answers, thresholds, models, letter, rule):
    for m in models:
        got, sure = common.judge_answer(answers.get(m), thresholds, rule)
        if got == letter and sure:
            return True
    return False


def ask_harness(session, rows):
    """⑤ 는 mem 을 안 지나 서버에 직접 묻는다 — mem 과 같은 꼬리표 떼기 뒤 비밀 꼴은 묻지 않는다."""
    session.proxy.place, session.proxy.run = "5", session.run_tag("draw-5")
    refused = 0
    for r in rows:
        evidence, claim = common.strip_tags(r["evidence"]), common.strip_tags(r["claim"])
        r["asked"] = True
        if common.secret_kind(evidence + "\n" + claim):
            r["answers"], r["refused"] = {}, True
            refused += 1
            continue
        r["answers"], _ = session.proxy.ask_all(evidence, claim)
    session.refused["draw-5"] = refused
    session.no_answer["draw-5"] = 0
    return rows


def collect_span(rows):
    times = sorted(str(r.get("report_at") or r.get("at") or "") for r in rows if r.get("report_at") or r.get("at"))
    if not times:
        return {"rows": len(rows), "first": None, "last": None}
    return {"rows": len(rows), "first": times[0], "last": times[-1]}


def cmd_draw(a):
    protocol, protocol_sha = common.load_protocol(a.protocol)
    place = a.place
    spec = protocol["places"][PLACE_KEYS[place]]
    if not spec.get("enabled"):
        raise common.Refuse(f"규약에서 자리 {place} 가 꺼져 있다 (enabled false) — 뽑지 않는다")
    alias = common.check_alias(a.alias or (HARNESS_ALIAS if place == 5 else common.default_alias(a.repo)))
    paths = common.part_paths(alias, place)
    if os.path.isfile(paths["mark"]):
        raise common.Refuse(f"이 별명의 자리 {place} 는 이미 얼렸다 — 다시 뽑지 않는다 (지킬 규칙 ①)")
    pairs_path = common.alias_dir(alias, *PLACE_FILES[place])
    logged = common.read_jsonl(pairs_path)
    rows = [dict(r, src_id=r.get("id")) for r in logged if not r.get("refused")]
    if not rows:
        raise common.Refuse(f"수집 로그가 비었다 : {pairs_path} — harvest 를 먼저 돌린다")
    a.alias = alias
    session = Session(a, protocol, place)
    status = session.proxy.probe()
    url = session.proxy.start()
    session.toml = common.write_llm_toml(common.alias_dir(alias, f"llm-draw-{place}.toml"), url, session.thresholds)
    for i, r in enumerate(rows):
        r["id"] = f"p{place}-{i:05d}"
    try:
        if place == 4:
            rows = session.judge_pairs(rows, "4", "draw-4")
        else:
            rows = ask_harness(session, rows)
    finally:
        session.proxy.stop()
    models = list(session.proxy.models)
    letter = PLACE_LETTER[place]
    target = spec["target"]
    hot = [r for r in rows if sure_by(r["answers"], session.thresholds, models, letter, session.rule)]
    cold = [r for r in rows if not sure_by(r["answers"], session.thresholds, models, letter, session.rule)]
    rng = random.Random(protocol["sample_seed"])
    out = []
    for name, part, size in ((f"sure-{letter}", hot, target // 2), ("rest", cold, target - min(len(hot), target // 2))):
        picked, weight = draw(part, size, rng)
        for r in picked:
            out.append({**r, "place": place, "stratum": name, "weight": weight, "pool": len(part)})
    roots = seen_roots(a.seen)
    keys = seen_keys(roots)
    finish_rows(out, keys)
    old = [common.set_aside(paths["cands"], session.stamp), common.set_aside(paths["meta"], session.stamp)]
    common.write_jsonl(paths["cands"], out)
    manifest = {"at": common.now_iso(), "alias": alias, "place": place, "stamp": session.stamp,
                "protocol_sha256": protocol_sha, "pool": len(rows), "drawn": len(out),
                "collect": dict(collect_span(logged), refused=sum(1 for r in logged if r.get("refused"))),
                "thresholds": {"repo": session.thresholds}, "models": status,
                "no_answer": session.no_answer, "refused": session.refused,
                "seen_roots": roots, "seen_keys": len(keys), "seen_pairs": sum(1 for r in out if r["seen"]),
                "set_aside": [p for p in old if p]}
    common.write_json(paths["meta"], manifest)
    print(f"자리 {place} 층 뽑기 {len(out)}쌍 (풀 {len(rows)} · 본 쌍 {manifest['seen_pairs']}) → {paths['cands']}")
    warn_no_answer(session.no_answer)
    return out


def register(sub):
    home = os.path.join("~", ".judgemodel")
    s = sub.add_parser("draw", help="④ · ⑤ 수집 로그에서 층 뽑기 + 두 모델에 묻기")
    s.add_argument("--place", type=int, choices=(4, 5), required=True)
    s.add_argument("--repo", default=".")
    s.add_argument("--alias", default="")
    s.add_argument("--models", default="")
    s.add_argument("--seen", default=f"{os.path.join(home, 'sets')},{os.path.join(home, 'build')}",
                   help="본 쌍 대조 폴더 (쉼표) · none 이면 안 본다")
    s.add_argument("--protocol", default="")
    s.set_defaults(func=cmd_draw)
