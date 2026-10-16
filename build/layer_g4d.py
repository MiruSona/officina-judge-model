"""layer_g4d — v1.2 K3 판 d 층 train-g4-d (같은 대상 · 근거에 없는 사실 d 후보 뽑기 → 판정 → 조립).

비공개 설계 4-1 · 4-2 · 4-4 · 4-5 · 5절 ①. 쌍 본문은 화면에 찍지 않는다.

    python -I layer_g4d.py pick  --g4 ~/.judgemodel/build/g4 [--old-root ~/.judgemodel] [--dry-run] [--force]
    → <g4>/train-g4-d/pick.jsonl · pick-summary.json   (label_batches.py make --mode train 에 넣는다)
    python -I layer_g4d.py build --g4 … [--dirs <판정 폴더> …] [--draft]
    → <g4>/train-g4-d/cands.jsonl (train-g4-d) · summary.json

꼴 다섯 (같은 원천 · 같은 묶음 안에서만 짓는다 — 같은 대상은 꼴이 보장한다) :
  klue-removed    KLUE 학습 쪽 지문에서 답 문장을 뺀 문단 × 같은 질문의 평서문. train-g4-c KLUE near-x 주장과 이어지면 먼저(`pair`)
  wiki-other      「제목 : 문장 X」 × 같은 위키 글의 다른 문장 Y(「제목 :」 뗌). 고침으로 이어진 같은 문장끼리는 짓지 않는다
  vitc-same-page · vitc-near · vitc-same-doc   VitaminC 학습 쪽 g1 후보(같은 쪽 c 꼴 · 가까운 근거 · NEI) 그대로
규칙 : 겹침 구간 몫 25/25/30/20(언어마다) · 근거 묶음당 2 · 주장 한 번 · KLUE 답 글이 근거에 남으면 뺌 · 같은 문장 뺌 ·
  금지 = layer_g4c.make_guard + 얼린 blind-g4 · dev-cmp-g4 + train-g4-c 쌍 열쇠. build 는 두 판이 다 d 로 정한 줄만 넣는다.
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import public_pick  # noqa: E402
from assemble_g3 import Stop, load_decisions, resolve  # noqa: E402
from label_batches import inside_git, read_jsonl, write_json, write_jsonl  # noqa: E402
from layer_g3 import merge_targets, recheck  # noqa: E402
from layer_g4c import _cap_groups, _drop_boundary, _refuse_or_move, _targets, make_guard  # noqa: E402
from make_sets import norm, sha_key  # noqa: E402

SEED = 20261403
LAYER = "train-g4-d"
KLUE, WIKI, VITC = "klue-removed", "wiki-other", ("vitc-same-page", "vitc-near", "vitc-same-doc")
# 꼴 → (원천 pick, 언어). 차례가 곧 뽑는 차례다
FORMS = {KLUE: ("klue", "ko"), WIKI: ("wiki", "ko"),
         VITC[0]: ("vitaminc", "en"), VITC[1]: ("vitaminc", "en"), VITC[2]: ("vitaminc", "en")}
VITC_FORM = {"same-page": VITC[0], "near": VITC[1], "same-doc": VITC[2]}
TARGETS = {
    "pick": {KLUE: 360, WIKI: 390, VITC[0]: 160, VITC[1]: 120, VITC[2]: 120},
    "build": {KLUE: 250, WIKI: 250, VITC[0]: 100, VITC[1]: 75, VITC[2]: 75},
    "per_group": 2,
    "edges": [0.1, 0.2, 0.35],
    "shares": [0.25, 0.25, 0.30, 0.20],
    "wiki_max_overlap": 0.7,     # 위키 같은 글 두 문장이 이만큼 겹치면 같은 문장의 다른 판으로 보고 뺀다 (리뷰 10-14)
}
_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


def overlap(evidence, claim):
    """주장 낱말 덩어리(영문 · 숫자 · 한글) 중 근거에도 있는 몫. d 진단(K2-d붕괴진단)과 같은 셈이다."""
    claim_tokens = set(_TOKEN.findall((claim or "").lower()))
    ev_tokens = set(_TOKEN.findall((evidence or "").lower()))
    return len(claim_tokens & ev_tokens) / max(1, len(claim_tokens))


def bin_of(value, edges):
    """겹침 값 → 구간 번호. edges [.1, .2, .35] 면 0~.1 → 0 · .1~.2 → 1 · .2~.35 → 2 · .35~ → 3."""
    return sum(1 for edge in edges if value >= edge)


def _row(form, base, evidence, claim, src_ids, **extra):
    src, lang = FORMS[form]
    row = {"id": None, "evidence": evidence, "claim": claim, "want": "", "hint": "d", "form": form,
           "domain": base.get("domain", ""), "group": base["group"], "groups": [base["group"]], "lang": lang,
           "lic": base.get("lic", ""), "split": "train", "src_ids": list(dict.fromkeys(src_ids)),
           "origin": "g4-train", "set": LAYER, "pick": src}
    row.update(extra)
    return row


# ---------------------------------------------------------------- 원천 → 꼴 후보


def klue_cands(mrc_rows, qa_rows, check_rows=(), c_rows=()):
    """KLUE removed 문단 × 같은 질문(src_ids)의 평서문 → (후보, 뺀 까닭). train-g4-c KLUE near-x 주장과 같으면 `pair` 에 그 c 줄 id."""
    dropped = collections.Counter()
    qa_train = [r for r in qa_rows if r.get("split") == "train" and r.get("form") == "qa-true"]
    prepared, why = public_pick.prepare(qa_train, "train", drop_ids=public_pick.failed_ids(check_rows))
    for key, n in why.items():
        dropped["klue_qa:" + key] += n
    qa_by_src = {}
    for row in prepared:
        for src in row.get("src_ids") or []:
            qa_by_src.setdefault(src, row)
    c_ids = {norm(r["claim"]): r["id"] for r in c_rows if r.get("pick") == "klue" and r.get("form") == "near-x"}
    out = []
    for row in mrc_rows:
        if row.get("split") != "train" or row.get("form") != "removed":
            continue
        qa = next((qa_by_src[s] for s in row.get("src_ids") or [] if s in qa_by_src), None)
        if qa is None:
            dropped["klue_no_claim"] += 1
            continue
        answers = {a.strip() for a in (row.get("answer"), qa.get("answer")) if a and a.strip()}
        if any(a in row["evidence"] for a in answers):
            dropped["klue_answer_in_evidence"] += 1
            continue
        extra = {}
        pair = c_ids.get(norm(qa["claim"]))
        if pair:
            extra["pair"] = pair
        out.append(_row(KLUE, row, row["evidence"], qa["claim"], row["src_ids"] + qa["src_ids"], **extra))
    return out, dropped


def _strip_title(text, title):
    if title and text.startswith(title + " : "):
        return text[len(title) + 3:]
    return text


class _Lines:
    """한 위키 글 안에서 고침(근거 → 주장)으로 이어진 문장을 한 줄기로 묶는다 — 줄기가 같으면 「같은 문장」이다."""

    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, one, two):
        self.parent[self.find(one)] = self.find(two)


def wiki_cands(wiki_rows, max_overlap=TARGETS["wiki_max_overlap"]):
    """같은 글(group)의 두 문장 X · Y → 근거 「제목 : X」 · 주장 Y. 같은 문장(고침 줄기 같음)과
    겹침 max_overlap 이상(같은 문장의 다른 판)은 뺀다 → (후보, 뺀 까닭)."""
    dropped = collections.Counter()
    by_group = collections.defaultdict(list)
    for row in wiki_rows:
        if row.get("split") not in ("any", "train"):
            continue
        by_group[row["group"]].append(row)
    out = []
    for group in sorted(by_group):
        rows = sorted(by_group[group], key=lambda r: (tuple(r.get("src_ids") or []), r["evidence"]))
        lines = _Lines()
        sentences = {}
        for row in rows:
            body = norm(_strip_title(row["evidence"], row.get("title")))
            edited = norm(_strip_title(row.get("claim") or "", row.get("title")))
            lines.find(body)
            if edited:
                lines.join(body, edited)
            sentences.setdefault(body, row)
        keys = sorted(k for k in sentences if k)
        for x in keys:
            for y in keys:
                if x == y:
                    continue
                if lines.find(x) == lines.find(y):
                    dropped["wiki_same_line"] += 1
                    continue
                ev_row, claim_row = sentences[x], sentences[y]
                claim = _strip_title(claim_row["evidence"], claim_row.get("title")).strip()
                if overlap(ev_row["evidence"], claim) >= max_overlap:
                    dropped["wiki_near_dup"] += 1
                    continue
                out.append(_row(WIKI, ev_row, ev_row["evidence"], claim,
                                (ev_row.get("src_ids") or []) + (claim_row.get("src_ids") or []),
                                title=ev_row.get("title", "")))
    return out, dropped


def vitc_cands(vit_rows):
    """VitaminC g1 후보 중 학습 쪽 same-page · near · same-doc 줄 → (꼴별 후보, 뺀 까닭). 근거와 주장이 같은 문장이면 뺀다."""
    dropped = collections.Counter()
    out = {form: [] for form in VITC}
    for row in vit_rows:
        form = VITC_FORM.get(row.get("form"))
        if form is None or row.get("split") != "train":
            continue
        if norm(row["evidence"]) == norm(row["claim"]):
            dropped["vitc_same_sentence"] += 1
            continue
        out[form].append(_row(form, row, row["evidence"], row["claim"], row.get("src_ids") or []))
    return out, dropped


# ---------------------------------------------------------------- 금지


class DGuard:
    """layer_g4c 의 Guard(평가 겹침 · 옛 셋 금지 · forbid.json · scrub)에 train-g4-c 쌍 열쇠를 더한다 — 같은 쌍이 c 와 d 로 두 번 안 나오게."""

    def __init__(self, guard, c_rows):
        self.guard = guard
        self.c_keys = set()
        for row in c_rows:
            self.c_keys.add(sha_key(row))
            self.c_keys.add(sha_key({"evidence": row.get("claim"), "claim": row.get("evidence")}))

    def reasons(self, row):
        out = list(self.guard.reasons(row))
        if sha_key(row) in self.c_keys:
            out.append("c_pair")
        return out


# ---------------------------------------------------------------- 뽑기


def _quota(total, shares):
    """total 을 shares 로 나눈다 (큰 나머지, 같으면 앞 구간)."""
    raw = [total * s for s in shares]
    out = [int(v) for v in raw]
    order = sorted(range(len(shares)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[:total - sum(out)]:
        out[i] += 1
    return out


class _Picker:
    """꼴 × 구간 줄 차례에서 한 줄씩 꺼낸다 — 묶음당 per_group · 주장 한 번 · 같은 쌍 한 번 · 금지를 꺼낼 때 본다."""

    def __init__(self, queues, guard, per_group, dropped):
        self.queues, self.guard, self.per_group, self.dropped = queues, guard, per_group, dropped
        self.pos = collections.Counter()
        self.used_group = collections.Counter()
        self.claims, self.keys = set(), set()

    def take(self, form, b):
        queue = self.queues[(form, b)]
        while self.pos[(form, b)] < len(queue):
            row = queue[self.pos[(form, b)]]
            self.pos[(form, b)] += 1
            if self.used_group[row["group"]] >= self.per_group:
                continue
            claim, key = norm(row["claim"]), sha_key(row)
            if claim in self.claims or key in self.keys:
                continue
            why = self.guard.reasons(row)
            if why:
                self.dropped.update("guard:" + w for w in why)
                continue
            self.used_group[row["group"]] += 1
            self.claims.add(claim)
            self.keys.add(key)
            return row
        return None


def _queues(pools, edges, seed):
    """꼴 × 구간 → 씨앗 고정으로 섞은 줄 차례. KLUE 는 d 짝(`pair`) 줄을 앞으로."""
    queues = {}
    for form in FORMS:
        bins = collections.defaultdict(list)
        for row in pools.get(form) or []:
            row = dict(row, overlap=round(overlap(row["evidence"], row["claim"]), 4))
            bins[bin_of(row["overlap"], edges)].append(row)
        for b in range(len(edges) + 1):
            rows = sorted(bins[b], key=sha_key)
            random.Random(f"{seed}:{form}:{b}").shuffle(rows)
            rows.sort(key=lambda r: 0 if r.get("pair") else 1)
            queues[(form, b)] = rows
    return queues


def _fill_lang(forms, need, queues, picker, quota):
    """한 언어 — 구간 몫을 모자란 구간부터 채우고(남은 몫 큰 꼴 먼저), 그래도 꼴 목표가 남으면 가장 덜 찬 구간에서 더 꺼낸다."""
    bins = range(len(quota))
    left = {f: need[f] for f in forms}
    got = {f: [[] for _ in bins] for f in forms}
    done = set()
    count = [0 for _ in bins]
    avail = {b: sum(len(queues[(f, b)]) for f in forms) / max(1, quota[b]) for b in bins}
    for b in sorted(bins, key=lambda k: (avail[k], k)):
        while count[b] < quota[b]:
            open_forms = [f for f in forms if left[f] > 0 and (f, b) not in done]
            if not open_forms:
                break
            form = min(open_forms, key=lambda f: (-left[f], forms.index(f)))
            row = picker.take(form, b)
            if row is None:
                done.add((form, b))
                continue
            got[form][b].append(row)
            left[form] -= 1
            count[b] += 1
    short_bins = [quota[b] - count[b] for b in bins]      # 첫 판(구간 몫 채우기) 뒤 값 — 넘침 판 앞이다
    for form in forms:
        while left[form] > 0:
            open_bins = [b for b in bins if (form, b) not in done]
            if not open_bins:
                break
            b = min(open_bins, key=lambda k: (count[k] - quota[k], k))
            row = picker.take(form, b)
            if row is None:
                done.add((form, b))
                continue
            got[form][b].append(row)
            left[form] -= 1
            count[b] += 1
    return got, short_bins, count


def pick(pools, guard, targets=None, seed=SEED):
    """꼴별 후보 줄(pools) → (뽑은 줄, 요약). 겹침 구간 몫은 언어마다 · 금지는 꺼낼 때 본다. guard 는 DGuard 꼴(reasons)."""
    targets = merge_targets(TARGETS, targets)
    edges, shares = targets["edges"], targets["shares"]
    dropped = collections.Counter()
    queues = _queues(pools, edges, seed)
    picker = _Picker(queues, guard, targets["per_group"], dropped)
    out, summary = [], {"seed": seed, "targets": targets["pick"], "edges": edges, "shares": shares,
                        "forms": {}, "langs": {}}
    for lang in ("ko", "en"):
        forms = [f for f in FORMS if FORMS[f][1] == lang]
        need = {f: targets["pick"].get(f, 0) for f in forms}
        quota = _quota(sum(need.values()), shares)
        got, short_bins, count = _fill_lang(forms, need, queues, picker, quota)
        summary["langs"][lang] = {"quota": quota, "got": count, "short_bins_before_spill": short_bins}
        for form in forms:
            rows = [r for per_bin in got[form] for r in per_bin]
            for k, row in enumerate(rows, start=1):
                row["id"] = f"g4d-{form}-{k:04d}-h{sha_key(row)[:8]}"
            out += rows
            pool_bins = [len(queues[(form, b)]) for b in range(len(edges) + 1)]
            summary["forms"][form] = {"pool": sum(pool_bins), "pool_bins": pool_bins,
                                      "picked_bins": [len(per_bin) for per_bin in got[form]], "picked": len(rows),
                                      "short": need[form] - len(rows),
                                      "pair_pool": sum(1 for r in pools.get(form) or [] if r.get("pair")),
                                      "pair_picked": sum(1 for r in rows if r.get("pair"))}
    summary["dropped"] = dict(sorted(dropped.items()))
    summary["rows"] = len(out)
    return sorted(out, key=lambda r: r["id"]), summary


# ---------------------------------------------------------------- 조립


def build(cands, decisions, targets=None, seed=SEED, draft=False):
    """판정 끝 후보 → (train-g4-d 줄, 요약). d 로 정한 줄만 · 꼴 목표 · 묶음당 2. 경계 칸(cd)은 버린다."""
    targets = merge_targets(TARGETS, targets)
    decisions, boundary = _drop_boundary(decisions)
    state = collections.Counter()
    decided = collections.defaultdict(list)
    for cand in sorted(cands, key=lambda r: r["id"]):
        label, votes, status, how = resolve(dict(cand, want=""), decisions)
        state[how if how in ("pending", "dropped") else f"label_{label}"] += 1
        if label != "d":
            continue
        row = {k: v for k, v in cand.items() if k != "judge"}
        row.update(want="d", hint="d", label_guide="support-v6", status=status, votes=votes or {})
        decided[cand["form"]].append(row)
    if state["pending"] and not draft:
        raise Stop(f"판정 전 줄 {state['pending']} — 판정을 끝내거나 --draft")
    used = collections.Counter()
    chosen = {}
    for form in FORMS:
        rows = decided[form]
        random.Random(f"{seed}:{form}:build").shuffle(rows)
        chosen[form] = _cap_groups(rows, targets["per_group"], used, targets["build"].get(form, 0))
    rows = sorted((r for got in chosen.values() for r in got), key=lambda r: r["id"])
    edges = targets["edges"]
    summary = {"seed": seed, "targets": targets["build"], "cand_state": dict(sorted(state.items())),
               "decided_d": {f: len(v) for f, v in sorted(decided.items())},
               "chosen": {f: len(v) for f, v in chosen.items()},
               "chosen_bins": {f: [sum(1 for r in v if bin_of(r.get("overlap", 0), edges) == b)
                                   for b in range(len(edges) + 1)] for f, v in chosen.items()},
               "short": {f: targets["build"].get(f, 0) - len(v) for f, v in chosen.items()
                         if len(v) < targets["build"].get(f, 0)},
               "rows": len(rows), "boundary_dropped": boundary}
    return rows, summary


# ---------------------------------------------------------------- 명령


def _inputs(a, old_root):
    build_dir = os.path.join(old_root, "build")
    paths = {"klue_mrc": os.path.join(build_dir, "g1", "klue-mrc", "candidates.jsonl"),
             "klue_rw": os.path.join(build_dir, "g3", "klue", "answered-rw.jsonl"),
             "klue_check": os.path.join(build_dir, "g3", "klue", "answered-rw.check.jsonl"),
             "wiki": os.path.join(build_dir, "g3", "kowiki", "wiki-all-train.jsonl"),
             "vitc": os.path.join(build_dir, "g1", "vitaminc", "candidates.jsonl"),
             "train_c": os.path.join(old_root, "sets", "g4", "train-g4-c.jsonl")}
    for key in paths:
        given = getattr(a, key, None)
        if given:
            paths[key] = os.path.expanduser(given)
    return paths


def _guard(g4, old_root, train_c, no_old_forbid):
    """→ (DGuard, 빠진 파일). 얼린 blind-g4 · dev-cmp-g4 가 없으면 빠진 파일로 센다(make_guard 는 있을 때만 읽는다)."""
    frozen = [os.path.join(old_root, "sets", "g4", f"{n}.jsonl") for n in ("blind-g4", "dev-cmp-g4")]
    missing = [p for p in frozen + [train_c] if not os.path.exists(p)]
    if missing:
        return None, missing
    guard, missing = make_guard(g4, old_root, no_old_forbid)
    if missing:
        return None, missing
    return DGuard(guard, read_jsonl(train_c)), []


def _stop(message, **extra):
    print(json.dumps(dict({"stop": message}, **extra), ensure_ascii=False))
    return 2


def cmd_pick(a):
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    out_dir = os.path.join(g4, LAYER)
    if not a.dry_run and inside_git(out_dir):
        return _stop("출력 폴더가 git 안이다 — 쌍 본문을 git 안에 안 쓴다", out=out_dir)
    outs = [os.path.join(out_dir, n) for n in ("pick.jsonl", "pick-summary.json")]
    if not a.dry_run:
        clash = _refuse_or_move(outs, a.force)
        if clash:
            return _stop("출력이 이미 있다 — 덮지 않는다 (--force 면 날짜 꼬리로 돌려 둔다)", files=clash)
    paths = _inputs(a, old_root)
    missing = [p for k, p in paths.items() if k != "train_c" and not os.path.exists(p)]
    guard, more = _guard(g4, old_root, paths["train_c"], a.no_old_forbid)
    missing += more
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    c_rows = read_jsonl(paths["train_c"])
    pools, dropped = {}, collections.Counter()
    pools[KLUE], why = klue_cands(read_jsonl(paths["klue_mrc"]), read_jsonl(paths["klue_rw"]),
                                  read_jsonl(paths["klue_check"]), c_rows)
    dropped.update(why)
    targets = _targets(a.targets)
    pools[WIKI], why = wiki_cands(read_jsonl(paths["wiki"]), merge_targets(TARGETS, targets)["wiki_max_overlap"])
    dropped.update(why)
    vitc, why = vitc_cands(read_jsonl(paths["vitc"]))
    pools.update(vitc)
    dropped.update(why)
    rows, summary = pick(pools, guard, targets, a.seed)
    summary["dropped_source"] = dict(sorted(dropped.items()))
    summary["inputs"] = {k: os.path.basename(p) for k, p in paths.items()}
    summary["recheck"] = recheck(rows, guard)
    summary["dry_run"] = a.dry_run
    if not a.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        write_jsonl(outs[0], rows)
        write_json(outs[1], summary)
    keys = ("rows", "recheck", "langs", "forms", "dropped", "dropped_source")
    print(json.dumps({k: summary[k] for k in keys}, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def cmd_build(a):
    g4, old_root = os.path.expanduser(a.g4), os.path.expanduser(a.old_root)
    out_dir = os.path.join(g4, LAYER)
    if inside_git(out_dir):
        return _stop("출력 폴더가 git 안이다 — 쌍 본문을 git 안에 안 쓴다", out=out_dir)
    outs = [os.path.join(out_dir, n) for n in ("cands.jsonl", "summary.json")]
    clash = _refuse_or_move(outs, a.force)
    if clash:
        return _stop("출력이 이미 있다 — 덮지 않는다 (--force 면 날짜 꼬리로 돌려 둔다)", files=clash)
    guard, missing = _guard(g4, old_root, _inputs(a, old_root)["train_c"], a.no_old_forbid)
    pick_path = os.path.join(out_dir, "pick.jsonl")
    if not os.path.exists(pick_path):
        missing.append(pick_path)
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False))
        return 2
    dirs = a.dirs
    if dirs is None:
        dirs = [os.path.join(g4, "judge", LAYER)]
    dirs = [os.path.expanduser(d) for d in dirs]
    try:
        rows, summary = build(read_jsonl(pick_path), load_decisions(dirs),
                              _targets(a.targets), a.seed, a.draft)
    except Stop as e:
        print(f"멈춤 : {e}", file=sys.stderr)
        return 2
    summary["recheck"] = recheck(rows, guard)
    summary["dirs"] = [d for d in dirs if os.path.exists(os.path.join(d, "decided.jsonl"))]
    write_jsonl(outs[0], rows)
    write_json(outs[1], summary)
    print(json.dumps({k: summary[k] for k in ("cand_state", "chosen", "short", "rows", "recheck")}, ensure_ascii=False))
    return 1 if sum(summary["recheck"].values()) else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, func in (("pick", cmd_pick), ("build", cmd_build)):
        s = sub.add_parser(name)
        s.add_argument("--g4", default="~/.judgemodel/build/g4")
        s.add_argument("--old-root", default="~/.judgemodel")
        s.add_argument("--targets", help="TARGETS 를 덮을 json (일부 칸만 적어도 됨)")
        s.add_argument("--seed", type=int, default=SEED)
        s.add_argument("--force", action="store_true", help="출력이 있으면 옛 파일을 날짜 꼬리로 돌려 두고 쓴다")
        s.add_argument("--no-old-forbid", action="store_true", help="G4Forbid · 옛 평가 문단 검사를 끈다 (시험용)")
        s.add_argument("--train-c", help="train-g4-c (쌍 열쇠 금지 · KLUE d 짝). 기본 <old-root>/sets/g4/train-g4-c.jsonl")
        s.set_defaults(func=func)
    pk = sub.choices["pick"]
    pk.add_argument("--dry-run", action="store_true", help="뽑기만 하고 파일은 안 쓴다 (요약만 화면에)")
    pk.add_argument("--klue-mrc", help="기본 <old-root>/build/g1/klue-mrc/candidates.jsonl")
    pk.add_argument("--klue-rw", help="기본 <old-root>/build/g3/klue/answered-rw.jsonl")
    pk.add_argument("--klue-check", help="기본 <old-root>/build/g3/klue/answered-rw.check.jsonl")
    pk.add_argument("--wiki", help="기본 <old-root>/build/g3/kowiki/wiki-all-train.jsonl")
    pk.add_argument("--vitc", help="기본 <old-root>/build/g1/vitaminc/candidates.jsonl")
    sub.choices["build"].add_argument("--dirs", nargs="*", default=None,
                                      help="판정 폴더 (decided.jsonl). 기본 <g4>/judge/train-g4-d")
    sub.choices["build"].add_argument("--draft", action="store_true", help="판정 전 줄을 세어서 빼고 조립")
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
