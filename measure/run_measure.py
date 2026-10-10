"""K 48 · game90 재기 — 모델 표 · 사다리 표 · p95 를 한 번에 낸다 (설계 6절).

    python -I run_measure.py --prereg <git 밖>/prereg.json --set game90=<game90.jsonl> \
        --url http://127.0.0.1:8092 --out-dir <git 밖>/nli1

- ② 모델 표 : 서버에 쌍마다 묻고 최고 확률 답 그대로(Unsure 무시) → 정확도 · 갈래 재현 · 거짓 지지(참고)
- ③ 사다리 표 : `mem judge support --file … --stage rules,nli` → 확신 답의 거짓 지지 · 보류 비율 · 단별 수
  (mem 이 PATH 에 없으면 모델 표만 낸다. mem 의 nli_url · 문턱이 prereg 와 다르면 재지 않는다)
- p95 : 모델 표를 묻는 같은 HTTP 왕복
prereg 의 셋 SHA · 모델 SHA 가 다르면 재지 않는다. once 셋은 결과 파일이 이미 있으면 다시 재지 않는다.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

from data_prep import LETTERS, sha256_file, want_label  # noqa: E402

SUPPORT, CONTRADICT = 0, 1
MEM_LETTERS = {"A": 0, "B": 1, "C": 2}


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_meta" not in row:
                rows.append(row)
    return rows


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def http_json(url, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def gold_labels(rows):
    """쌍의 want → 칸 번호. 모르는 want 가 하나라도 있으면 재지 않는다."""
    golds = []
    for r in rows:
        gold = want_label(r.get("want", ""))
        if gold is None:
            raise SystemExit(f"want 를 못 읽는다: {r.get('id', '')} {r.get('want')!r}")
        golds.append(gold)
    return golds


def check_ladder_rows(rows, expected, stderr):
    """mem judge 줄 수가 쌍 수와 다르면 SystemExit — 모자란 채 zip 하면 거짓 지지를 덜 센다."""
    if len(rows) != expected:
        raise SystemExit(f"mem judge 줄 {len(rows)} ≠ 쌍 {expected} — 재지 않는다. stderr: {stderr[-500:]}")


def model_table(golds, probs):
    """② 모델 표. golds 는 칸 번호, probs 는 [a, b, c]. 문턱 없이 최고 확률 답."""
    confusion = [[0, 0, 0] for _ in range(3)]
    for gold, p in zip(golds, probs):
        confusion[gold][max(range(3), key=lambda i: p[i])] += 1
    recall = {}
    for i in range(3):
        total = sum(confusion[i])
        recall[LETTERS[i]] = round(confusion[i][i] / total, 4) if total else None
    n = len(golds)
    return {"n": n, "accuracy": round(sum(confusion[i][i] for i in range(3)) / n, 4) if n else None,
            "recall": recall, "false_support_no_threshold": confusion[1][0] + confusion[2][0],
            "confusion": confusion}


def ladder_table(golds, ladder_rows, model_probs):
    """③ 사다리 표. ladder_rows 는 mem judge --file 의 줄 (letter · unsure · stage)."""
    false_support = unsure = 0
    stages = {}
    missed_by_rules = hit_by_model = 0
    for gold, row, p in zip(golds, ladder_rows, model_probs):
        stage = row.get("stage") or ("error" if row.get("problem") else "?")
        stages[stage] = stages.get(stage, 0) + 1
        if row.get("unsure") or row.get("problem"):
            unsure += 1
        elif MEM_LETTERS.get(row.get("letter")) == SUPPORT and gold != SUPPORT:
            false_support += 1
        if gold == CONTRADICT and stage != "rules":
            missed_by_rules += 1
            hit_by_model += 1 if max(range(3), key=lambda i: p[i]) == CONTRADICT else 0
    n = len(golds)
    return {"n": n, "false_support": false_support, "unsure": unsure,
            "unsure_rate": round(unsure / n, 4) if n else None, "stages": stages,
            "contradict_not_by_rules": missed_by_rules,
            "contradict_not_by_rules_model_recall": round(hit_by_model / missed_by_rules, 4) if missed_by_rules else None}


def verdict(criteria, model, ladder, p95_ms):
    """자 판정. 사다리 표가 없으면 그 자는 None(못 잼)이고 통과로 치지 않는다."""
    checks = {
        "accuracy": model["accuracy"] >= criteria["accuracy_min"],
        "contradict_recall": (model["recall"]["b"] or 0) >= criteria["contradict_recall_min"],
        "p95": p95_ms <= criteria["p95_ms_max"],
        "false_support": None, "unsure": None,
    }
    if ladder is not None:
        checks["false_support"] = ladder["false_support"] <= criteria["false_support_max"]
        checks["unsure"] = ladder["unsure_rate"] <= criteria["unsure_max"]
    passed = all(v is True for v in checks.values())
    only_p95 = not checks["p95"] and all(v is True for k, v in checks.items() if k != "p95")
    return {"checks": checks, "passed": passed, "next": "pass" if passed else ("p95_only" if only_p95 else "fail")}


def run_ladder(mem, path, url, prereg, expected):
    """mem judge 로 사다리 표 줄을 얻는다. mem 설정이 prereg 와 다르면 SystemExit."""
    conf = json.loads(subprocess.run([mem, "judge", "config", "--json"], capture_output=True, text=True,
                                     encoding="utf-8", check=True).stdout)
    want = {"nli_url": url.rstrip("/"), "nli_sure": prereg["nli_sure"], "nli_sure_support": prereg["nli_sure_support"]}
    got = {"nli_url": (conf.get("nli_url") or "").rstrip("/"), "nli_sure": conf.get("nli_sure"),
           "nli_sure_support": conf.get("nli_sure_support")}
    if got != want:
        raise SystemExit(f"mem llm.toml 이 prereg 와 다르다 — 재지 않는다. mem {got} · prereg {want}")
    proc = subprocess.run([mem, "judge", "support", "--file", path, "--stage", "rules,nli"],
                          capture_output=True, text=True, encoding="utf-8")
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    check_ladder_rows(rows, expected, proc.stderr.strip())
    return rows, proc.stderr.strip()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NLI 측정 (모델 표 · 사다리 표 · p95)")
    p.add_argument("--prereg", required=True)
    p.add_argument("--set", dest="sets", action="append", required=True, help="이름=jsonl (prereg 에 있는 이름)")
    p.add_argument("--url", default="http://127.0.0.1:8092")
    p.add_argument("--mem", default="mem", help="mem 실행 파일 (없으면 사다리 표 생략)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def measure_set(a, prereg, name, path):
    info = prereg["sets"].get(name)
    if info is None:
        raise SystemExit(f"prereg 에 없는 셋: {name}")
    if sha256_file(path) != info["sha256"]:
        raise SystemExit(f"{name} SHA 가 prereg 와 다르다 — 재지 않는다")
    result_path = os.path.join(a.out_dir, f"result-{name}.json")
    if info["once"] and os.path.exists(result_path):
        raise SystemExit(f"{name} 은 한 번만 잰다 — 이미 있다: {result_path}")
    health = http_json(a.url.rstrip("/") + "/health")
    if health.get("model") != prereg["model"]["weights_sha256"][:8]:
        raise SystemExit(f"서버 모델 {health.get('model')} 이 prereg 모델과 다르다")

    rows = read_rows(path)
    golds = gold_labels(rows)
    probs, lat = [], []
    http_json(a.url.rstrip("/") + "/judge", {"evidence": "데우기", "claim": "데우기"})
    for r in rows:
        start = time.perf_counter()
        body = http_json(a.url.rstrip("/") + "/judge", {"evidence": r["evidence"], "claim": r["claim"]})
        lat.append((time.perf_counter() - start) * 1000)
        probs.append([body["a"], body["b"], body["c"]])
    model = model_table(golds, probs)
    ladder, ladder_rows, ladder_note = None, [], "mem 없음 — 사다리 표 생략"
    mem = shutil.which(a.mem)
    if mem:
        ladder_rows, ladder_note = run_ladder(mem, path, a.url, prereg, len(rows))
        ladder = ladder_table(golds, ladder_rows, probs)
    p95 = pct(lat, 0.95)
    result = {"set": name, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": prereg["model"],
              "model_table": model, "ladder_table": ladder, "ladder_note": ladder_note,
              "http_ms": {"p50": round(pct(lat, 0.5), 1), "p95": round(p95, 1), "max": round(max(lat), 1)}}
    if info["judge"]:
        result["verdict"] = verdict(prereg["criteria"], model, ladder, p95)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"pairs-{name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i, (r, p) in enumerate(zip(rows, probs)):
            line = {"id": r.get("id", ""), "want": r["want"], "probs": p, "ms": round(lat[i], 1)}
            if ladder_rows:
                line["ladder"] = {k: ladder_rows[i].get(k) for k in ("letter", "unsure", "stage", "problem")}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(result_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return result


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    with open(a.prereg, encoding="utf-8") as f:
        prereg = json.load(f)
    for spec in a.sets:
        name, path = spec.split("=", 1)
        result = measure_set(a, prereg, name, path)
        print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
