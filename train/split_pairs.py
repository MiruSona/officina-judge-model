"""실물 쌍을 기억 덩이 단위로 train/dev 다시 가르기 (설계 4-1).

    python -I split_pairs.py --in train.jsonl --in dev.jsonl \
        --exclude-ids exclude-k.txt --exclude-pairs k48.jsonl --exclude-pairs game90.jsonl \
        --out-dir <git 밖 폴더>

쌍마다 `src_ids`(또는 `src` 를 + 로 가른 것)가 있어야 한다. 기억 하나라도 같이 쓰는 쌍은 한 덩이다.
덩이를 섞어 dev 를 약 15% 뽑고, 갈래(want)별 dev 몫이 띠(13~17%) 안이고 반대 dev 가 40 이상일 때까지
씨앗을 1씩 올려 다시 뽑는다. 결과는 train ∩ dev 기억 0 을 검사한 뒤에만 쓴다. 이 스크립트는 mem 경로를 모른다.
갈래는 넷(a 지지 · b 반대 · c 무관 · d 비슷한 계열)이다. want `cd`(「c 나 d」, 라벨러가 못 고른 쌍)는
dev 에 안 넣는다 — 그 쌍이 든 덩이는 늘 train 쪽이고 띠 검사에서도 뺀다 (판정 넷 확장 설계 1-1 · 3-3).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys


# want 글자(a·b·c·d)도 이름으로 센다 — 반대 수(min_contradict)는 이름으로 본다
WANT_NAMES = {"a": "support", "b": "contradict", "c": "unrelated", "d": "related"}
# dev 에 안 넣는 want — train 에서만 합친 칸 손실로 쓴다
TRAIN_ONLY_WANTS = frozenset({"cd"})


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if "_meta" not in r]


def want_name(row):
    want = str(row.get("want", "")).strip().lower()
    return WANT_NAMES.get(want, want)


def mem_ids(row):
    """쌍이 쓴 기억 id 목록."""
    if "src_ids" in row:
        return list(row["src_ids"])
    src = row.get("src", "")
    if not src:
        return []
    return [s for s in str(src).split("+") if s]


def read_id_list(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def drop_excluded(rows, banned_ids, banned_texts):
    """금지 기억을 쓰거나 금지 쌍과 글이 같은 쌍을 뺀다. (남은 줄, 뺀 수)."""
    kept, dropped = [], 0
    for row in rows:
        if banned_ids.intersection(mem_ids(row)) or (row["evidence"], row["claim"]) in banned_texts:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def make_groups(rows):
    """합집합 찾기로 기억을 같이 쓰는 쌍을 묶는다. 덩이마다 줄 번호 목록 (차례 고정)."""
    parent = {}
    for row in rows:
        ids = mem_ids(row)
        for i in ids:
            parent.setdefault(i, i)
        for i in ids[1:]:
            a, b = find(parent, ids[0]), find(parent, i)
            if a != b:
                parent[b] = a
    groups = collections.defaultdict(list)
    for n, row in enumerate(rows):
        ids = mem_ids(row)
        key = find(parent, ids[0]) if ids else f"_solo{n}"
        groups[key].append(n)
    return [groups[k] for k in sorted(groups)]


def try_split(rows, groups, seed, dev_frac, band, min_contradict, max_group, pinned=frozenset()):
    """씨앗 하나로 가르기. 띠를 만족하면 (train 번호, dev 번호), 아니면 None.
    pinned 줄 번호가 든 덩이는 dev 로 안 뽑는다 (늘 train 쪽)."""
    order = [g for g in groups if len(g) <= max_group]
    order = [g for g in order if not pinned.intersection(g)]
    random.Random(seed).shuffle(order)
    target = round(len(rows) * dev_frac)
    dev = []
    for g in order:
        if len(dev) >= target:
            break
        dev.extend(g)
    want_all = collections.Counter(want_name(row) for row in rows)
    want_dev = collections.Counter(want_name(rows[n]) for n in dev)
    for want, total in want_all.items():
        if want in TRAIN_ONLY_WANTS:
            continue
        share = want_dev[want] / total
        if share < band[0] or share > band[1]:
            return None
    if want_dev["contradict"] < min_contradict:
        return None
    dev_set = set(dev)
    train = [n for n in range(len(rows)) if n not in dev_set]
    return train, sorted(dev)


def pinned_rows(rows, pin_ids):
    """pin_ids 기억을 하나라도 쓰는 줄 번호. 그 덩이는 dev 로 못 간다."""
    return frozenset(n for n, row in enumerate(rows) if pin_ids.intersection(mem_ids(row)))


def train_only_rows(rows):
    """want 가 cd 같은 train 전용인 줄 번호. 그 덩이도 dev 로 못 간다."""
    return frozenset(n for n, row in enumerate(rows) if want_name(row) in TRAIN_ONLY_WANTS)


def split(rows, seed=1, dev_frac=0.15, band=(0.13, 0.17), min_contradict=40, max_tries=2000, pin_ids=frozenset()):
    """(train 줄, dev 줄, 쓴 씨앗, 큰 덩이 수). 못 가르면 SystemExit.
    pin_ids 기억을 쓰는 덩이는 늘 train 쪽에 둔다 (예 : blind 에 못 넣는 game90 기억)."""
    groups = make_groups(rows)
    pinned = pinned_rows(rows, set(pin_ids)) | train_only_rows(rows)
    max_group = int(len(rows) * dev_frac)
    big = sum(1 for g in groups if len(g) > max_group)
    for s in range(seed, seed + max_tries):
        got = try_split(rows, groups, s, dev_frac, band, min_contradict, max_group, pinned)
        if got is None:
            continue
        train_n, dev_n = got
        return [rows[n] for n in train_n], [rows[n] for n in dev_n], s, big
    raise SystemExit(f"씨앗 {seed}~{seed + max_tries - 1} 에서 띠 {band} · 반대 {min_contradict} 를 못 맞췄다")


def overlap(train, dev):
    train_ids = set()
    for row in train:
        train_ids.update(mem_ids(row))
    dev_ids = set()
    for row in dev:
        dev_ids.update(mem_ids(row))
    return train_ids & dev_ids


def stats(rows):
    return {"rows": len(rows),
            "want": dict(sorted(collections.Counter(want_name(r) for r in rows).items())),
            "kind": dict(sorted(collections.Counter(r.get("kind", "-") for r in rows).items()))}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="쌍을 기억 덩이 단위로 train/dev 가르기")
    p.add_argument("--in", dest="inputs", action="append", required=True, help="쌍 jsonl (여러 번)")
    p.add_argument("--exclude-ids", action="append", default=[], help="뺄 기억 id 목록 txt")
    p.add_argument("--exclude-pairs", action="append", default=[],
                   help="이 쌍 파일이 쓴 기억·같은 글을 뺀다 (K 48 · game90)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--band", default="0.13,0.17", help="갈래별 dev 몫 띠")
    p.add_argument("--min-contradict", type=int, default=40)
    p.add_argument("--max-tries", type=int, default=2000)
    p.add_argument("--pin-ids", action="append", default=[],
                   help="id 목록 파일 경로 (한 줄에 기억 id 하나 · 여러 번) — 그 id 를 쓰는 덩이는 늘 train 쪽 (빼지는 않는다)")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    rows = []
    for path in a.inputs:
        rows.extend(read_jsonl(path))
    banned_ids, banned_texts = set(), set()
    for path in a.exclude_ids:
        banned_ids |= read_id_list(path)
    for path in a.exclude_pairs:
        for row in read_jsonl(path):
            banned_ids.update(mem_ids(row))
            banned_texts.add((row["evidence"], row["claim"]))
    pool, dropped = drop_excluded(rows, banned_ids, banned_texts)
    band = tuple(float(x) for x in a.band.split(","))
    pin_ids = set()
    for path in a.pin_ids:
        pin_ids |= read_id_list(path)
    train, dev, used_seed, big = split(pool, a.seed, a.dev_frac, band, a.min_contradict, a.max_tries, pin_ids)
    pinned_in_dev = [r["id"] for r in dev if pin_ids.intersection(mem_ids(r))]
    if pinned_in_dev:
        raise SystemExit(f"pin 기억을 쓰는 쌍이 dev 에 들어갔다 : {pinned_in_dev[:5]} — 쓰지 않는다")
    train_only_in_dev = [r.get("id", "") for r in dev if want_name(r) in TRAIN_ONLY_WANTS]
    if train_only_in_dev:
        raise SystemExit(f"cd 쌍이 dev 에 들어갔다 : {train_only_in_dev[:5]} — 쓰지 않는다")
    shared = overlap(train, dev)
    if shared:
        raise SystemExit(f"train·dev 가 기억 {len(shared)}개를 같이 쓴다 — 쓰지 않는다")

    os.makedirs(a.out_dir, exist_ok=True)
    out_train = os.path.join(a.out_dir, "train.jsonl")
    out_dev = os.path.join(a.out_dir, "dev.jsonl")
    write_jsonl(out_train, train)
    write_jsonl(out_dev, dev)
    report = {"seed": used_seed, "pool": len(rows), "excluded": dropped, "big_groups_in_train": big,
              "memory_overlap": 0, "train": stats(train), "dev": stats(dev),
              "pinned_ids": len(pin_ids), "pinned_rows": len(pinned_rows(pool, pin_ids)),
              "sha256": {"train.jsonl": sha256_file(out_train), "dev.jsonl": sha256_file(out_dev)},
              "inputs": [os.path.basename(p) for p in a.inputs]}
    with open(os.path.join(a.out_dir, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# split_pairs — 기억 덩이 단위 train/dev (train ∩ dev 기억 0)\n")
        f.write(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
