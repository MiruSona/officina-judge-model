"""K 48 · game90 · blind 재기 — 모델 표 · 확신 표 · 사다리 표 · p95 를 한 번에 낸다 (설계 6절 · 판정 넷 확장 설계 4-2).

    python -I run_measure.py --prereg <git 밖>/prereg.json --set blind=<blind.jsonl> \
        --url http://127.0.0.1:8092 --out-dir <git 밖>/nli2

- ② 모델 표 : 서버에 쌍마다 묻고 최고 확률 답 그대로(Unsure 무시) → 정확도 · 3칸으로 접은 정확도 · 갈래 재현 · 거짓 지지(참고)
- 확신 표 : 같은 확률에 prereg 문턱(nli_sure_support · nli_sure · nli_sure_related)을 걸어 확신 거짓 지지 · 보류 · d 확신 정밀.
  Go 판정기와 같은 「모른다」 선을 쓴다 (train/calibrate.py). mem 없이 서버 확률만으로 잰다
- ③ 사다리 표 : `mem judge support --file … --stage rules,nli` → 확신 답의 거짓 지지 · 보류 비율 · 단별 수
  (mem 이 PATH 에 없으면 모델 표만 낸다. mem 의 nli_url · 문턱이 prereg 와 다르면 재지 않는다)
- p95 : 모델 표를 묻는 같은 HTTP 왕복
응답 칸은 서버 /health 의 letters 를 따른다 (없으면 옛 a·b·c). 4칸 prereg 면 서버도 4칸이어야 한다.
/judge 에는 label_set 을 명시해 청한다 — prereg 칸 수가 3이면 nli3, 4면 nli4. 응답 label_set 이 다르면 멈춘다.
판정 : 3칸 prereg(옛 자)는 모델 표 + 사다리 표, 4칸 prereg(넷 확장 자)는 모델 표 + 확신 표로 한다.
접기 오판(fold40)은 기억 짝 단위라 여기서 안 잰다 — measure/run_fold.py 가 따로 판정한다. 넷 확장 판의 통과는 둘 다 통과다.
prereg 의 셋 SHA · 모델 SHA 가 다르면 재지 않는다. once 셋은 결과 파일이 이미 있으면 다시 재지 않는다.
측정 셋에는 want `cd` 쌍을 넣지 않는다 (있으면 멈춘다).
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

from calibrate import answer  # noqa: E402
from data_prep import CD, LETTERS, sha256_file, strip_tags, want_label  # noqa: E402

SUPPORT, CONTRADICT, RELATED = 0, 1, 3
MEM_LETTERS = {"A": 0, "B": 1, "C": 2, "D": 3}


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
    """쌍의 want → 칸 번호. 모르는 want 나 cd 가 하나라도 있으면 재지 않는다."""
    golds = []
    for r in rows:
        gold = want_label(r.get("want", ""))
        if gold is None:
            raise SystemExit(f"want 를 못 읽는다: {r.get('id', '')} {r.get('want')!r}")
        if gold == CD:
            raise SystemExit(f"측정 셋에 cd 쌍이 있다: {r.get('id', '')} — cd 는 train 에만 쓴다")
        golds.append(gold)
    return golds


def check_ladder_rows(rows, expected, stderr):
    """mem judge 줄 수가 쌍 수와 다르면 SystemExit — 모자란 채 zip 하면 거짓 지지를 덜 센다."""
    if len(rows) != expected:
        raise SystemExit(f"mem judge 줄 {len(rows)} ≠ 쌍 {expected} — 재지 않는다. stderr: {stderr[-500:]}")


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def fold3(label):
    return 2 if label == RELATED else label


def model_table(golds, probs):
    """② 모델 표. golds 는 칸 번호, probs 는 [a, b, c(, d)]. 문턱 없이 최고 확률 답.

    accuracy3 = c·d 를 한 칸으로 합친 정확도 (3칸이면 accuracy 와 같다). 3칸 서버에 d 정답이 오면 그 쌍은
    accuracy 에서 늘 틀림이고 accuracy3 에서는 c 면 맞힘이다.
    """
    n_cols = len(probs[0]) if probs else 3
    size = max([n_cols] + [g + 1 for g in golds])
    confusion = [[0] * size for _ in range(size)]
    folded = 0
    for gold, p in zip(golds, probs):
        pred = argmax(p)
        confusion[gold][pred] += 1
        folded += 1 if fold3(pred) == fold3(gold) else 0
    recall = {}
    for i in range(size):
        total = sum(confusion[i])
        recall[LETTERS[i]] = round(confusion[i][i] / total, 4) if total else None
    n = len(golds)
    return {"n": n, "labels": n_cols,
            "accuracy": round(sum(confusion[i][i] for i in range(size)) / n, 4) if n else None,
            "accuracy3": round(folded / n, 4) if n else None,
            "recall": recall, "false_support_no_threshold": sum(confusion[i][0] for i in range(1, size)),
            "confusion": confusion}


def sure_table(golds, probs, sure, sure_support, sure_related=None):
    """확신 표. 문턱을 건 답 — 확신 거짓 지지(정답이 지지 아닌데 확신 A) · 보류 · 확신 D 정밀.

    「확신」은 Go 판정기와 같은 셈이다 : 기본 「모른다」(최고 < 0.40 · 1·2등 차 < 0.01)가 아니고,
    지지는 sure_support · d 는 sure_related(없으면 sure) · 나머지는 sure 이상.
    """
    false_support = unsure = rel_hit = rel_total = 0
    for gold, p in zip(golds, probs):
        top, prob, base_unsure = answer(p)
        if top == SUPPORT:
            line = sure_support
        elif top == RELATED and sure_related is not None:
            line = sure_related
        else:
            line = sure
        if base_unsure or prob < line:
            unsure += 1
            continue
        if top == SUPPORT and gold != SUPPORT:
            false_support += 1
        if top == RELATED:
            rel_total += 1
            rel_hit += 1 if gold == RELATED else 0
    n = len(golds)
    non_support = sum(1 for g in golds if g != SUPPORT)
    return {"n": n, "false_support": false_support, "non_support": non_support,
            "false_support_share": round(false_support / non_support, 4) if non_support else None,
            "unsure": unsure, "unsure_rate": round(unsure / n, 4) if n else None,
            "related_sure": rel_total, "related_sure_hit": rel_hit,
            "related_sure_precision": round(rel_hit / rel_total, 4) if rel_total else None}


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
            hit_by_model += 1 if argmax(p) == CONTRADICT else 0
    n = len(golds)
    return {"n": n, "false_support": false_support, "unsure": unsure,
            "unsure_rate": round(unsure / n, 4) if n else None, "stages": stages,
            "contradict_not_by_rules": missed_by_rules,
            "contradict_not_by_rules_model_recall": round(hit_by_model / missed_by_rules, 4) if missed_by_rules else None}


def is_v2(criteria):
    """넷 확장 자(4칸)인가 — d 재현 자가 있으면 그렇다."""
    return "related_recall_min" in criteria


def at_least(value, floor):
    return value is not None and value >= floor


def at_most(value, ceiling):
    return value is not None and value <= ceiling


def verdict(criteria, model, ladder, p95_ms, sure=None):
    """자 판정. 못 잰 자는 None 이고 통과로 치지 않는다.

    옛 자(3칸) : 모델 표 + 사다리 표. 넷 확장 자(4칸) : 모델 표 + 확신 표 (접기 오판은 run_fold.py).
    """
    if is_v2(criteria):
        checks = {
            "accuracy3": at_least(model["accuracy3"], criteria["accuracy3_min"]),
            "contradict_recall": at_least(model["recall"].get("b"), criteria["contradict_recall_min"]),
            "related_recall": at_least(model["recall"].get("d"), criteria["related_recall_min"]),
            "p95": p95_ms <= criteria["p95_ms_max"],
            "false_support": None, "unsure": None, "related_sure_precision": None,
        }
        if sure is not None:
            checks["false_support"] = at_most(sure["false_support_share"], criteria["false_support_share_max"])
            checks["unsure"] = at_most(sure["unsure_rate"], criteria["unsure_max"])
            checks["related_sure_precision"] = at_least(sure["related_sure_precision"],
                                                        criteria["related_sure_precision_min"])
    else:
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
    nxt = "pass" if passed else ("p95_only" if only_p95 else "fail")
    out = {"checks": checks, "passed": passed, "next": nxt}
    if is_v2(criteria):
        if nxt == "fail":
            # 옛 일 자가 다 통과고 d 자만 미달이면 --compat3 로 꽂는 길 (설계 4-2 「미달이면」)
            old_ok = all(checks[k] is True for k in ("accuracy3", "contradict_recall", "false_support", "unsure", "p95"))
            out["next"] = "fail_related" if old_ok else "fail_old"
        out["also_needs"] = "fold40 — measure/run_fold.py (접기 오판)"
    return out


def run_ladder(mem, path, url, prereg, expected):
    """mem judge 로 사다리 표 줄을 얻는다. mem 설정이 prereg 와 다르면 SystemExit."""
    conf = json.loads(subprocess.run([mem, "judge", "config", "--json"], capture_output=True, text=True,
                                     encoding="utf-8", check=True).stdout)
    want = {"nli_url": url.rstrip("/"), "nli_sure": prereg["nli_sure"], "nli_sure_support": prereg["nli_sure_support"]}
    got = {"nli_url": (conf.get("nli_url") or "").rstrip("/"), "nli_sure": conf.get("nli_sure"),
           "nli_sure_support": conf.get("nli_sure_support")}
    if "nli_sure_related" in prereg:
        # 4칸 prereg 면 d 확정 문턱도 같아야 한다 (3칸 prereg 엔 없다)
        want["nli_sure_related"] = prereg["nli_sure_related"]
        got["nli_sure_related"] = conf.get("nli_sure_related")
    if got != want:
        raise SystemExit(f"mem llm.toml 이 prereg 와 다르다 — 재지 않는다. mem {got} · prereg {want}")
    proc = subprocess.run([mem, "judge", "support", "--file", path, "--stage", "rules,nli"],
                          capture_output=True, text=True, encoding="utf-8")
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    check_ladder_rows(rows, expected, proc.stderr.strip())
    return rows, proc.stderr.strip()


def server_letters(health, prereg):
    """서버 응답 칸. prereg 에 letters 가 있으면 서버와 같아야 한다 (4칸 prereg 를 3칸 서버로 재지 않게)."""
    letters = health.get("letters") or LETTERS[:3]
    want = prereg.get("letters")
    if want and list(want) != list(letters):
        raise SystemExit(f"서버 칸 {letters} 이 prereg 칸 {want} 과 다르다 — --compat3 로 띄웠나 본다")
    return letters


def prereg_label_set(prereg):
    """prereg 칸 수로 청할 칸을 고른다. (칸 목록, label_set). letters 가 없는 옛 prereg 는 3칸이다."""
    letters = list(prereg.get("letters") or LETTERS[:3])
    return letters, {3: "nli3", 4: "nli4"}[len(letters)]


def ask_judge(url, evidence, claim, label_set):
    """label_set 을 명시해 /judge 를 부른다. 서버가 청한 칸으로 못 답하면 멈춘다."""
    body = http_json(url.rstrip("/") + "/judge", {"evidence": evidence, "claim": claim, "label_set": label_set})
    if body.get("label_set") != label_set:
        raise SystemExit(f"서버 칸 {body.get('label_set')} 이 청한 칸 {label_set} 과 다르다 — --compat3 로 띄웠나 본다")
    return body


def check_server(a, prereg):
    """서버 /health 를 보고 모델 SHA · 칸을 prereg 와 견준다. 응답 칸 목록을 돌려준다."""
    health = http_json(a.url.rstrip("/") + "/health")
    if health.get("model") != prereg["model"]["weights_sha256"][:8]:
        raise SystemExit(f"서버 모델 {health.get('model')} 이 prereg 모델과 다르다")
    return server_letters(health, prereg)


def check_set(a, prereg, name, path):
    """prereg 의 셋 정보를 보고 SHA · once 를 확인한다. (셋 정보, 결과 파일 경로)."""
    info = prereg["sets"].get(name)
    if info is None:
        raise SystemExit(f"prereg 에 없는 셋: {name}")
    if sha256_file(path) != info["sha256"]:
        raise SystemExit(f"{name} SHA 가 prereg 와 다르다 — 재지 않는다")
    result_path = os.path.join(a.out_dir, f"result-{name}.json")
    if info["once"] and os.path.exists(result_path):
        raise SystemExit(f"{name} 은 한 번만 잰다 — 이미 있다: {result_path}")
    return info, result_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NLI 측정 (모델 표 · 확신 표 · 사다리 표 · p95)")
    p.add_argument("--prereg", required=True)
    p.add_argument("--set", dest="sets", action="append", required=True, help="이름=jsonl (prereg 에 있는 이름)")
    p.add_argument("--url", default="http://127.0.0.1:8092")
    p.add_argument("--mem", default="mem", help="mem 실행 파일 (없으면 사다리 표 생략)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def measure_set(a, prereg, name, path):
    info, result_path = check_set(a, prereg, name, path)
    check_server(a, prereg)
    letters, label_set = prereg_label_set(prereg)

    rows = read_rows(path)
    golds = gold_labels(rows)
    probs, lat = [], []
    ask_judge(a.url, "데우기", "데우기", label_set)
    for r in rows:
        start = time.perf_counter()
        # 꼬리표는 손님처럼 떼고 보낸다 (data_prep.strip_tags)
        body = ask_judge(a.url, strip_tags(r["evidence"]), strip_tags(r["claim"]), label_set)
        lat.append((time.perf_counter() - start) * 1000)
        probs.append([body[letter] for letter in letters])
    model = model_table(golds, probs)
    sure = sure_table(golds, probs, prereg["nli_sure"], prereg["nli_sure_support"], prereg.get("nli_sure_related"))
    ladder, ladder_rows, ladder_note = None, [], "mem 없음 — 사다리 표 생략"
    mem = shutil.which(a.mem)
    if mem:
        ladder_rows, ladder_note = run_ladder(mem, path, a.url, prereg, len(rows))
        ladder = ladder_table(golds, ladder_rows, probs)
    p95 = pct(lat, 0.95)
    result = {"set": name, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": prereg["model"],
              "letters": letters, "model_table": model, "sure_table": sure,
              "ladder_table": ladder, "ladder_note": ladder_note,
              "http_ms": {"p50": round(pct(lat, 0.5), 1), "p95": round(p95, 1), "max": round(max(lat), 1)}}
    if info["judge"]:
        result["verdict"] = verdict(prereg["criteria"], model, ladder, p95, sure)
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
