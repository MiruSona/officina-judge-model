"""원본 NLI 자료 → 공통 jsonl · KorNLI · MNLI · SNLI 바퀴별 표본 · 토큰 길이 재기 (설계 2절).

    python -I data_prep.py klue   --in klue-nli-v1.1_train.json --out data/prep/klue-train.jsonl
    python -I data_prep.py klue   --in klue-nli-v1.1_dev.json   --out data/prep/klue-dev.jsonl --split dev
    python -I data_prep.py kornli --mnli multinli.train.ko.tsv --snli snli_1.0_train.ko.tsv \
                                  --out-dir data/prep --per-epoch 50000 --epochs 2 --seed 1
    python -I data_prep.py mnli   --in multinli_1.0_train.jsonl --out-dir data/prep --per-epoch 50000 --epochs 2
    python -I data_prep.py snli   --in snli_1.0_train.jsonl     --out-dir data/prep --per-epoch 25000 --epochs 2
    python -I data_prep.py eval-en --src mnli --in multinli_1.0_dev_matched.jsonl \
                                  --out data/prep/mnli-devm-eval2000.jsonl --limit 2000   # 평가 전용
    python -I data_prep.py measure --tokenizer hf/xlm-roberta-large \
                                  --set klue=data/prep/klue-train.jsonl --set k48=pairs.jsonl \
                                  --group base=klue,kornli --group serve=mem,k48,game90 --out lengths.json

공통 줄 꼴 : {"id", "premise", "hypothesis", "label": 0|1|2, "src", "kind"?}. 칸 차례는 a·b·c 로 고정이고
**라벨 바꿔 끼우기는 이 파일의 표(아래 상수)에서만 한다** (설계 2-3).
비상업(NC) 자료는 받지 않는다 — KorNLI dev/test(XNLI 번역) · XNLI · ANLI 는 경로 이름으로 거른다.
영어(MNLI · SNLI)는 학습 셋만 쓴다 — 이름에 dev·test 가 든 파일도 거른다. 라벨 `-`(사람 의견 갈림) 줄은 뺀다.
`eval-en` 만 dev 이름을 받는다 — 평가 전용 셋을 만들고 학습 표본(`-e<n>.jsonl`)은 안 쓴다. NC 거름은 그대로다.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys

# 출력 칸 차례 : 0 = A 지지 · 1 = B 반대 · 2 = C 무관 (서버 응답 a·b·c 와 같은 차례)
LABELS = ["support", "contradict", "unrelated"]
LETTERS = ["a", "b", "c"]
# KLUE-NLI 라벨 id (0 entailment · 1 neutral · 2 contradiction) → 출력 칸
KLUE_ID_TO_LABEL = {0: 0, 2: 1, 1: 2}
# 글 라벨 (KLUE gold_label · KorNLI gold_label) → 출력 칸
TEXT_TO_LABEL = {"entailment": 0, "contradiction": 1, "neutral": 2}
# mem 쌍 want (이름 또는 글자) → 출력 칸
WANT_TO_LABEL = {"support": 0, "contradict": 1, "unrelated": 2, "a": 0, "b": 1, "c": 2}
# model.json 에 적는 원본 매핑 (칸 차례대로)
SOURCE_LABELS = ["entailment", "contradiction", "neutral"]

# 자료 출처별 라이선스 · 비상업 여부 (train_info.json · model.json 의 data 칸)
LICENSES = {
    "klue": ("CC BY-SA 4.0", False),
    "kornli-mnli": ("CC BY-SA 4.0", False),
    "kornli-snli": ("CC BY-SA 4.0", False),
    "mnli": ("OANC · CC BY-SA 3.0 · CC BY 3.0 · 미국 공유 저작물 (MultiNLI 1.0)", False),
    "snli": ("CC BY-SA 4.0", False),
    "mem": ("private (studio memory, not for release)", False),
    "smoke": ("synthetic", False),
}

# 경로 이름에 이 낱말이 들면 NC 자료로 보고 거부한다
NC_WORDS = ("xnli", "anli")
# 학습 셋만 받는 자료(KorNLI · MNLI · SNLI)는 이름에 이 낱말이 들어도 거부한다
TRAIN_ONLY_WORDS = ("dev", "test")

MAX_LEN_CANDIDATES = [128, 192, 256, 320]
COVER = 0.95
MNLI_SHARE = 0.7


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def want_label(want):
    """mem 쌍 want → 칸 번호. 모르는 값이면 None."""
    return WANT_TO_LABEL.get(str(want).strip().lower())


def normalize(row):
    """공통 꼴이나 mem 쌍 꼴(evidence·claim·want) 줄을 공통 꼴로. 라벨을 못 읽으면 label None."""
    if "premise" in row:
        out = dict(row)
        out.setdefault("src", "unknown")
        return out
    out = {"id": row.get("id", ""), "premise": row["evidence"], "hypothesis": row["claim"],
           "label": want_label(row.get("want", "")), "src": "mem"}
    if "kind" in row:
        out["kind"] = row["kind"]
    return out


def read_pairs(path):
    """학습·평가용 쌍 파일을 공통 꼴로 읽는다. `_meta` 줄은 건너뛴다."""
    rows = []
    for row in read_jsonl(path):
        if "_meta" in row:
            continue
        rows.append(normalize(row))
    return rows


def check_allowed(path, train_only):
    """NC 자료 경로면 SystemExit. train_only(KorNLI · MNLI · SNLI)면 이름에 dev·test 가 들어도 거부한다."""
    name = os.path.basename(path).lower()
    words = NC_WORDS
    if train_only:
        words = NC_WORDS + TRAIN_ONLY_WORDS
    for word in words:
        if word in name:
            raise SystemExit(f"비상업(NC) 자료로 보이는 경로라 거부한다: {path} ('{word}')")


def klue_records(data):
    """KLUE 원본 꼴 셋을 받는다 : KLUE 저장소 JSON 목록 · datasets-server {"rows":[{"row":…}]} · 줄마다 dict."""
    if isinstance(data, dict) and "rows" in data:
        return [r.get("row", r) for r in data["rows"]]
    if isinstance(data, list):
        return data
    raise SystemExit("KLUE 원본 꼴을 모른다 — JSON 목록이나 {\"rows\": …} 여야 한다")


def klue_label(rec):
    """KLUE 레코드 라벨 → 칸 번호. 라벨 없음(-1·빈 값)은 None."""
    if "gold_label" in rec:
        return TEXT_TO_LABEL.get(str(rec["gold_label"]).strip().lower())
    label = rec.get("label")
    if isinstance(label, int):
        return KLUE_ID_TO_LABEL.get(label)
    return None


def load_klue_file(path):
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    with open(path, encoding="utf-8") as f:
        return klue_records(json.load(f))


def convert_klue(paths):
    """KLUE 파일들 → (공통 줄, 버린 수)."""
    rows, dropped = [], 0
    for path in paths:
        check_allowed(path, train_only=False)
        for rec in load_klue_file(path):
            label = klue_label(rec)
            if label is None:
                dropped += 1
                continue
            rows.append({"id": rec.get("guid", f"klue-{len(rows)}"), "premise": rec["premise"],
                         "hypothesis": rec["hypothesis"], "label": label, "src": "klue"})
    return rows, dropped


def read_kornli_tsv(path, src):
    """KorNLI 학습 TSV (sentence1 · sentence2 · gold_label) → (공통 줄, 버린 수). 행 번호는 id 에 남긴다."""
    check_allowed(path, train_only=True)
    rows, dropped = [], 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for n, rec in enumerate(reader):
            if len(rec) < len(header):
                dropped += 1
                continue
            label = TEXT_TO_LABEL.get(rec[col["gold_label"]].strip().lower())
            if label is None:
                dropped += 1
                continue
            rows.append({"id": f"{src}-{n}", "premise": rec[col["sentence1"]],
                         "hypothesis": rec[col["sentence2"]], "label": label, "src": src})
    return rows, dropped


def read_nli_jsonl(path, src, train_only=True):
    """MNLI · SNLI 원본 jsonl (sentence1 · sentence2 · gold_label · pairID) → (공통 줄, 버린 수).

    gold_label 이 `-` 이거나 문장이 빈 줄은 버린다. id 는 `<src>-<pairID>` (없으면 행 번호).
    train_only=False 는 평가 전용(eval-en)에서만 쓴다 — dev·test 이름을 받되 NC 이름은 여전히 거부한다.
    """
    check_allowed(path, train_only=train_only)
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = TEXT_TO_LABEL.get(str(rec.get("gold_label", "")).strip().lower())
            premise, hypothesis = rec.get("sentence1") or "", rec.get("sentence2") or ""
            if label is None or not premise.strip() or not hypothesis.strip():
                dropped += 1
                continue
            rows.append({"id": f"{src}-{rec.get('pairID', n)}", "premise": premise, "hypothesis": hypothesis,
                         "label": label, "src": src})
    return rows, dropped


def split_quota(total, parts):
    """total 을 parts 칸에 고르게 나눈다. 나머지는 앞 칸부터 하나씩."""
    base, rest = divmod(total, parts)
    return [base + (1 if i < rest else 0) for i in range(parts)]


def sample_kornli(by_src, per_epoch, epochs, seed, mnli_share=MNLI_SHARE):
    """바퀴마다 다른 표본을 뽑는다 — MNLI 7 : SNLI 3, 그 안에서 라벨 1/3 씩, 바퀴끼리 겹침 없음.

    by_src = {"kornli-mnli": [줄…], "kornli-snli": [줄…]}. 바퀴마다 줄 목록을 돌려준다.
    """
    mnli_n = round(per_epoch * mnli_share)
    return sample_balanced(by_src, {"kornli-mnli": mnli_n, "kornli-snli": per_epoch - mnli_n}, epochs, seed)


def sample_balanced(by_src, want, epochs, seed):
    """출처별 want[src] 쌍씩, 그 안에서 라벨 1/3 씩, 바퀴끼리 겹침 없이 바퀴마다 줄 목록을 돌려준다."""
    pools = {}
    for src_i, src in enumerate(sorted(want)):
        for label in range(3):
            pool = [r for r in by_src.get(src, []) if r["label"] == label]
            random.Random(seed * 100 + src_i * 10 + label).shuffle(pool)
            pools[(src, label)] = pool
    epochs_rows = []
    for epoch in range(epochs):
        rows = []
        for src in sorted(want):
            for label, n in enumerate(split_quota(want[src], 3)):
                pool = pools[(src, label)]
                start = epoch * n
                if start + n > len(pool):
                    raise SystemExit(f"{src} 라벨 {LABELS[label]} 표본이 모자란다: {len(pool)} < {start + n}")
                rows.extend(pool[start:start + n])
        random.Random(seed * 1000 + epoch).shuffle(rows)
        epochs_rows.append(rows)
    return epochs_rows


def percentile(sorted_vals, q):
    """가장 가까운 순위 백분위 (sorted_vals 는 오름차순)."""
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[k]


def choose_max_len(lengths, candidates=MAX_LEN_CANDIDATES, cover=COVER):
    """cover 이상을 자르지 않는 가장 작은 후보. 다 못 덮으면 가장 큰 후보 (320 위로는 안 간다)."""
    if not lengths:
        return candidates[0]
    for cand in candidates:
        kept = sum(1 for n in lengths if n <= cand)
        if kept / len(lengths) >= cover:
            return cand
    return candidates[-1]


def length_stats(lengths):
    vals = sorted(lengths)
    return {"n": len(vals), "p50": percentile(vals, 0.50), "p95": percentile(vals, 0.95),
            "p99": percentile(vals, 0.99), "max": vals[-1] if vals else 0,
            "max_len": choose_max_len(vals),
            "over": {str(c): sum(1 for n in vals if n > c) for c in MAX_LEN_CANDIDATES}}


def load_tokenizer(path):
    """tokenizers.Tokenizer 를 돌려준다. tokenizer.json 이 있으면 그것, 없으면 transformers 로 읽는다."""
    from tokenizers import Tokenizer

    direct = path
    if os.path.isdir(path):
        direct = os.path.join(path, "tokenizer.json")
    if os.path.isfile(direct):
        return Tokenizer.from_file(direct)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, use_fast=True).backend_tokenizer


def text_pair(row):
    if "premise" in row:
        return row["premise"], row["hypothesis"]
    return row["evidence"], row["claim"]


def cmd_klue(a):
    rows, dropped = convert_klue(a.inputs)
    write_jsonl(a.out, rows)
    by_label = [sum(1 for r in rows if r["label"] == i) for i in range(3)]
    print(json.dumps({"out": a.out, "split": a.split, "rows": len(rows), "dropped_no_label": dropped,
                      "by_label": dict(zip(LABELS, by_label))}, ensure_ascii=False))


def write_epochs(out_dir, prefix, epochs_rows, note):
    """바퀴별 표본을 <prefix>-e<n>.jsonl 로, 뽑은 id 목록을 <prefix>-sample-<seed>.txt 로 쓴다. 목록 경로를 돌려준다."""
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, f"{prefix}-sample-{note['seed']}.txt")
    with open(list_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# epoch\tid  (" + " · ".join(f"{k} {v}" for k, v in note.items()) + ")\n")
        for epoch, rows in enumerate(epochs_rows, 1):
            write_jsonl(os.path.join(out_dir, f"{prefix}-e{epoch}.jsonl"), rows)
            for r in rows:
                f.write(f"{epoch}\t{r['id']}\n")
    return list_path


def cmd_kornli(a):
    by_src, dropped = {}, {}
    for src, path in (("kornli-mnli", a.mnli), ("kornli-snli", a.snli)):
        by_src[src], dropped[src] = read_kornli_tsv(path, src)
    epochs_rows = sample_kornli(by_src, a.per_epoch, a.epochs, a.seed, a.mnli_share)
    list_path = write_epochs(a.out_dir, "kornli", epochs_rows,
                             {"seed": a.seed, "per_epoch": a.per_epoch, "mnli_share": a.mnli_share})
    print(json.dumps({"out_dir": a.out_dir, "epochs": len(epochs_rows), "per_epoch": a.per_epoch,
                      "pool": {k: len(v) for k, v in by_src.items()}, "dropped": dropped,
                      "sample_list": list_path}, ensure_ascii=False))


def cmd_english(a):
    """mnli · snli — 원본 학습 jsonl 하나 → 바퀴별 표본 (라벨 1/3 씩 · 바퀴끼리 겹침 없음)."""
    rows, dropped = read_nli_jsonl(a.input, a.cmd)
    epochs_rows = sample_balanced({a.cmd: rows}, {a.cmd: a.per_epoch}, a.epochs, a.seed)
    list_path = write_epochs(a.out_dir, a.cmd, epochs_rows, {"seed": a.seed, "per_epoch": a.per_epoch})
    by_label = [sum(1 for r in rows if r["label"] == i) for i in range(3)]
    print(json.dumps({"out_dir": a.out_dir, "epochs": len(epochs_rows), "per_epoch": a.per_epoch,
                      "pool": len(rows), "dropped": dropped, "by_label": dict(zip(LABELS, by_label)),
                      "sample_list": list_path}, ensure_ascii=False))


def cmd_eval_en(a):
    """eval-en — MNLI · SNLI dev/test 원본 → 평가 전용 공통 jsonl (앞에서부터 limit 줄, 0 = 전부)."""
    rows, dropped = read_nli_jsonl(a.input, a.src, train_only=False)
    if a.limit:
        rows = rows[:a.limit]
    write_jsonl(a.out, rows)
    by_label = [sum(1 for r in rows if r["label"] == i) for i in range(3)]
    print(json.dumps({"out": a.out, "src": a.src, "rows": len(rows), "dropped": dropped,
                      "by_label": dict(zip(LABELS, by_label)), "use": "eval only"}, ensure_ascii=False))


def cmd_measure(a):
    tokenizer = load_tokenizer(a.tokenizer)
    lengths = {}
    for spec in a.sets:
        name, path = spec.split("=", 1)
        rows = [r for r in read_jsonl(path) if "_meta" not in r]
        if a.sample and len(rows) > a.sample:
            rows = random.Random(a.seed).sample(rows, a.sample)
        lengths[name] = [length_for(tokenizer, r) for r in rows]
    out = {"tokenizer": a.tokenizer, "candidates": MAX_LEN_CANDIDATES, "cover": COVER,
           "sets": {name: length_stats(vals) for name, vals in lengths.items()}, "groups": {}}
    for spec in a.groups:
        name, members = spec.split("=", 1)
        merged = []
        for member in members.split(","):
            merged.extend(lengths.get(member, []))
        out["groups"][name] = length_stats(merged)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    print(text)


def length_for(tokenizer, row):
    # pairenc 는 툴 뿌리에 있다 (python -I 는 스크립트 폴더도 sys.path 에 안 넣는다)
    from pairenc import pair_length

    premise, hypothesis = text_pair(row)
    return pair_length(tokenizer, premise, hypothesis)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JudgeModel 자료 준비")
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("klue", help="KLUE-NLI 원본 → 공통 jsonl")
    k.add_argument("--in", dest="inputs", action="append", required=True, help="원본 JSON(여러 번 가능)")
    k.add_argument("--out", required=True)
    k.add_argument("--split", choices=["train", "dev"], default="train")
    n = sub.add_parser("kornli", help="KorNLI 학습 TSV → 바퀴별 표본 jsonl")
    n.add_argument("--mnli", required=True, help="multinli.train.ko.tsv")
    n.add_argument("--snli", required=True, help="snli_1.0_train.ko.tsv")
    n.add_argument("--out-dir", required=True)
    n.add_argument("--per-epoch", type=int, default=50000)
    n.add_argument("--epochs", type=int, default=2)
    n.add_argument("--seed", type=int, default=1)
    n.add_argument("--mnli-share", type=float, default=MNLI_SHARE)
    for name, per_epoch, file_name in (("mnli", 50000, "multinli_1.0_train.jsonl"),
                                       ("snli", 25000, "snli_1.0_train.jsonl")):
        e = sub.add_parser(name, help=f"{name.upper()} 원본 학습 jsonl → 바퀴별 표본 jsonl")
        e.add_argument("--in", dest="input", required=True, help=file_name)
        e.add_argument("--out-dir", required=True)
        e.add_argument("--per-epoch", type=int, default=per_epoch)
        e.add_argument("--epochs", type=int, default=2)
        e.add_argument("--seed", type=int, default=1)
    v = sub.add_parser("eval-en", help="MNLI · SNLI dev/test 원본 → 평가 전용 jsonl (학습에 쓰지 않는다)")
    v.add_argument("--src", choices=["mnli", "snli"], required=True)
    v.add_argument("--in", dest="input", required=True, help="multinli_1.0_dev_matched.jsonl 등")
    v.add_argument("--out", required=True)
    v.add_argument("--limit", type=int, default=0, help="앞에서부터 이만큼 (0 = 전부)")
    m = sub.add_parser("measure", help="쌍 토큰 길이 분포 · max_len 고르기")
    m.add_argument("--tokenizer", required=True, help="HF 모델 폴더 · tokenizer.json · 모델 id")
    m.add_argument("--set", dest="sets", action="append", required=True, help="이름=jsonl 경로")
    m.add_argument("--group", dest="groups", action="append", default=[], help="이름=셋,셋 (합쳐서 max_len)")
    m.add_argument("--sample", type=int, default=10000, help="셋이 이보다 크면 표본만 잰다 (0 = 전부)")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", default="")
    return p.parse_args(argv)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = parse_args()
    if a.cmd == "klue":
        cmd_klue(a)
    elif a.cmd == "kornli":
        cmd_kornli(a)
    elif a.cmd in ("mnli", "snli"):
        cmd_english(a)
    elif a.cmd == "eval-en":
        cmd_eval_en(a)
    else:
        cmd_measure(a)


if __name__ == "__main__":
    main()
