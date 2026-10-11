"""teacher — 교사 모델(nli-studio-v3, r2-02)로 후보의 거짓 음성 거르기 (설계 3-3 ① ② · 3-2 near).

score 는 판정 서버(`serve.py`, POST /judge → {"a","b","c","d","model","ms"})를 불러 점수 파일만 쓰고,
filter 는 점수 파일을 읽어 거른다. 서버가 없어도 filter 와 시험은 돈다. 캐는 모델 ≠ 학습할 모델(B1 씨앗들).
서버 기본 꼴에 기대지 않고 label_set 을 명시해("nli4") 4칸을 청한다 — 4칸으로 못 답하는 서버면 멈춘다.

    python -I teacher.py score  --cands <train-cands.jsonl> --out <teacher-scores.jsonl> [--url http://127.0.0.1:8092]
    python -I teacher.py filter --cands <train-cands.jsonl> --scores <teacher-scores.jsonl> --out <걸러진.jsonl>
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8092"
MEDIAN_SHARE = 0.95
NEAR_TOP = 0.10      # near d 후보 중 관련도(1 − p_c) 위 10% — 거짓 음성 의심
NEAR_LOW = 0.10      # 아래 10% — 대상이 다른 쉬운 c 의심
FLOOR = 0.5          # 옮길 칸 확률의 절대 바닥 — 교사가 낯선 원천에선 중앙값이 0 가까이 무너진다(10-11 46-다)
LABEL_SET = "nli4"   # 교사 점수는 d 칸이 있어야 한다
RETRIES = 2
TIMEOUT = 30


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def http_post(url, body):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/judge", data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def score_rows(cands, post, done=()):
    """후보마다 교사 확률을 받아 {id, a, b, c, d, model} 줄로. done 에 든 id 는 건너뛴다(이어 돌리기)."""
    skip = set(done)
    for row in cands:
        if row["id"] in skip:
            continue
        reply = None
        for attempt in range(RETRIES + 1):
            try:
                reply = post({"evidence": row["evidence"], "claim": row["claim"], "label_set": LABEL_SET})
                break
            except (urllib.error.URLError, OSError, ValueError) as err:
                if attempt == RETRIES:
                    raise RuntimeError(f"{row['id']} 점수 실패: {err}") from err
                time.sleep(1.0 + attempt)
        if reply.get("label_set") != LABEL_SET or "d" not in reply:
            raise RuntimeError("교사 응답에 d 칸이 없다 — --compat3 서버가 아닌 4칸 서버를 띄운다")
        yield {"id": row["id"], **{k: float(reply[k]) for k in ("a", "b", "c", "d")}, "model": reply.get("model", "")}


def cmd_score(a):
    cands = read_jsonl(a.cands)
    done = [r["id"] for r in read_jsonl(a.out)] if os.path.exists(a.out) else []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    count = 0
    with open(a.out, "a", encoding="utf-8", newline="\n") as f:
        for row in score_rows(cands, lambda body: http_post(a.url, body), done):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
    print(json.dumps({"scored": count, "skipped": len(done), "out": a.out}, ensure_ascii=False))


def source_of(row):
    return row.get("domain") or row.get("origin") or ""


def medians(cands, scores):
    """원천마다 a 후보의 p_a 중앙값, b 후보의 p_b 중앙값."""
    vals = collections.defaultdict(list)
    for row in cands:
        s = scores.get(row["id"])
        if s and row.get("hint") in ("a", "b"):
            vals[(source_of(row), row["hint"])].append(s[row["hint"]])
    return {key: statistics.median(v) for key, v in vals.items() if v}


def filter_rows(cands, scores, share=MEDIAN_SHARE, near_top=NEAR_TOP, near_low=NEAR_LOW, floor=FLOOR):
    """→ (남길 줄, 뺀 줄).

    ② d 후보의 p_a(p_b) 가 같은 원천 a(b) 후보 중앙값 × share 이상이고 floor 이상이면 hint 를 a(b) 로 돌린다(버리지 않음).
    near 꼴 d 후보는 원천마다 관련도(1 − p_c) 순위의 위 · 아래 끝을 빼고 가운데만 남긴다.
    """
    med = medians(cands, scores)
    kept, dropped = [], []
    near = collections.defaultdict(list)
    for row in cands:
        s = scores.get(row["id"])
        out = dict(row)
        if s:
            out["teacher"] = {k: s[k] for k in ("a", "b", "c", "d")}
        if not s or row.get("hint") != "d":
            kept.append(out)
            continue
        moved = ""
        best = 0.0
        for side in ("a", "b"):
            m = med.get((source_of(row), side))
            if m is None or m <= 0 or s[side] < floor:
                continue
            ratio = s[side] / (m * share)
            if ratio >= 1.0 and ratio > best:
                moved, best = side, ratio
        if moved:
            out["hint_before"] = "d"
            out["hint"] = moved
            out["teacher_rule"] = f"d→{moved}"
            kept.append(out)
        elif row.get("form") == "near":
            near[source_of(row)].append(out)
        else:
            kept.append(out)
    for rows in near.values():
        rows.sort(key=lambda r: (-(1.0 - r["teacher"]["c"]), r["id"]))
        n = len(rows)
        top, low = int(n * near_top), int(n * near_low)
        for i, row in enumerate(rows):
            if i < top:
                dropped.append(dict(row, teacher_rule="near-top"))
            elif i >= n - low:
                dropped.append(dict(row, teacher_rule="near-low"))
            else:
                kept.append(row)
    return kept, dropped


def cmd_filter(a):
    cands = read_jsonl(a.cands)
    scores = {r["id"]: r for r in read_jsonl(a.scores)}
    missing = sum(1 for r in cands if r["id"] not in scores)
    kept, dropped = filter_rows(cands, scores, a.share, a.near_top, a.near_low, a.floor)
    write_jsonl(a.out, kept)
    write_jsonl(os.path.splitext(a.out)[0] + ".dropped.jsonl", dropped)
    rules = collections.Counter(r.get("teacher_rule", "") for r in kept + dropped)
    rules.pop("", None)
    print(json.dumps({"kept": len(kept), "dropped": len(dropped), "no_score": missing, "rules": dict(rules)},
                     ensure_ascii=False))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="판정 서버로 교사 점수 매기기 (이어 돌리기 됨)")
    s.add_argument("--cands", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--url", default=DEFAULT_URL)
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("filter", help="점수 파일로 거짓 음성 · near 끝 거르기")
    s.add_argument("--cands", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--share", type=float, default=MEDIAN_SHARE)
    s.add_argument("--near-top", type=float, default=NEAR_TOP)
    s.add_argument("--near-low", type=float, default=NEAR_LOW)
    s.add_argument("--floor", type=float, default=FLOOR, help="옮길 칸 확률의 절대 바닥 (기본 0.5)")
    s.set_defaults(func=cmd_filter)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
