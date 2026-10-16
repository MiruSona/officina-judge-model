"""rulers — 학습 전 자 ① 주장만 보는 모델 · ③ 다양성 · ② 출처 판별기(진단만).

비공개 설계 3-4. 패키지 없이 순수 파이썬으로 짰다(글자 n묶음 + 로지스틱 회귀).
쌍 파일 줄은 `evidence` · `claim` · 칸(`--label-key`, 기본 `want`, 없으면 `label` · `hint`) · `group` 을 읽는다.

    python -I rulers.py claim-only --synth <Ⅱ 라벨 쌍.jsonl> --real <dev-cmp.jsonl>
    python -I rulers.py diversity  --synth <Ⅱ 라벨 쌍.jsonl> --real <dev-cmp.jsonl>
    python -I rulers.py source     --synth <ㄴ 합성 쌍.jsonl> --real <dev-cmp.jsonl> --diagnostic [--field evidence]
"""
import argparse
import collections
import json
import math
import random
import sys
import zlib

SEED = 20261011
CLAIM_ONLY_GAP = 0.15
DIVERSITY_RATIO = 1.5
DIVERSITY_SAMPLE = 300
SIM_PAIRS = 2000
VALID_SHARE = 0.2
LABEL_NAMES = {"support": "a", "contradict": "b", "unrelated": "c", "related": "d"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_of(row, key):
    """칸 읽기 — 옛 꼴(support · related …)도 a ~ d 로 맞춘다. cd 는 d 아님으로 친다."""
    for k in (key, "want", "label", "hint"):
        value = row.get(k)
        if value:
            return LABEL_NAMES.get(value, value)
    return ""


def char_ngrams(text, sizes=(2, 3)):
    clean = " ".join(str(text).lower().split())
    out = []
    for n in sizes:
        out.extend(clean[i:i + n] for i in range(len(clean) - n + 1))
    return out


def features(text):
    """글자 2~3 묶음 → {특징: 값} (있음 1, 길이로 나눠 √ 정규화)."""
    grams = set(char_ngrams(text))
    if not grams:
        return {}
    w = 1.0 / math.sqrt(len(grams))
    return {g: w for g in grams}


class Logistic:
    """희소 특징 로지스틱 회귀 — SGD · L2. 작은 자료용."""

    def __init__(self, epochs=5, lr=0.2, l2=1e-4, seed=SEED):
        self.epochs, self.lr, self.l2, self.seed = epochs, lr, l2, seed
        self.w = collections.defaultdict(float)
        self.bias = 0.0

    def score(self, x):
        z = self.bias + sum(self.w.get(k, 0.0) * v for k, v in x.items())
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, xs, ys):
        rng = random.Random(self.seed)
        order = list(range(len(xs)))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            lr = self.lr / (1.0 + epoch)
            for i in order:
                err = self.score(xs[i]) - ys[i]
                self.bias -= lr * err
                for k, v in xs[i].items():
                    self.w[k] -= lr * (err * v + self.l2 * self.w[k])
        return self

    def accuracy(self, xs, ys):
        if not xs:
            return None
        return sum(1 for x, y in zip(xs, ys) if (self.score(x) >= 0.5) == bool(y)) / len(xs)


def group_split(rows, share, seed):
    """열쇠 묶음(group) 단위로 검증 몫을 뗀다 — 같은 묶음의 쌍이 양쪽에 안 걸치게."""
    def key(row):
        g = row.get("group") or row.get("id") or row.get("claim", "")
        return zlib.crc32(f"{seed}:{g}".encode("utf-8")) % 1000
    train = [r for r in rows if key(r) >= share * 1000]
    valid = [r for r in rows if key(r) < share * 1000]
    return train, valid


def claim_only(synth, real, label_key="want", seed=SEED):
    """자 ① : 주장만으로 「d 냐 아니냐」. Ⅱ 검증 몫 정확도와 실물 정확도의 차가 15%p 아래여야 한다."""
    train, valid = group_split(synth, VALID_SHARE, seed)
    def xy(rows):
        return [features(r["claim"]) for r in rows], [1 if label_of(r, label_key) == "d" else 0 for r in rows]
    xt, yt = xy(train)
    model = Logistic(seed=seed).fit(xt, yt)
    xv, yv = xy(valid)
    xr, yr = xy(real)
    acc_valid, acc_real = model.accuracy(xv, yv), model.accuracy(xr, yr)
    gap = None
    if acc_valid is not None and acc_real is not None:
        gap = abs(acc_valid - acc_real)
    major = max(sum(yv), len(yv) - sum(yv)) / len(yv) if yv else None
    return {"ruler": "claim-only", "train": len(train), "valid": len(valid), "real": len(real),
            "acc_valid": acc_valid, "acc_real": acc_real, "gap": gap, "valid_majority": major,
            "pass": gap is not None and gap < CLAIM_ONLY_GAP}


def repeat_rate(texts):
    """글자 3묶음 반복률 = 1 − (서로 다른 3묶음 수 / 전체 3묶음 수). 높을수록 같은 말이 되풀이된다."""
    grams = [g for t in texts for g in char_ngrams(t, (3,))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def tfidf_mean_sim(texts, pairs, seed):
    """글자 3묶음 TF-IDF 벡터로 무작위 쌍 코사인 닮음 평균."""
    docs = [collections.Counter(char_ngrams(t, (3,))) for t in texts]
    if len(docs) < 2:
        return 0.0
    df = collections.Counter(g for d in docs for g in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = {g: tf * math.log((1 + n) / (1 + df[g])) + tf for g, tf in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({g: x / norm for g, x in v.items()})
    rng = random.Random(seed)
    total, count = 0.0, 0
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2)
        small, big = sorted((vecs[i], vecs[j]), key=len)
        total += sum(x * big.get(g, 0.0) for g, x in small.items())
        count += 1
    return total / count


def diversity(synth, real, label_key="want", field="claim", sample=DIVERSITY_SAMPLE, seed=SEED):
    """자 ③ : 칸마다 같은 수를 뽑아(크기 맞춤) 반복률 · 쌍 닮음을 실물과 견준다. 둘 다 실물의 1.5배 이하."""
    def by_label(rows):
        out = collections.defaultdict(list)
        for r in rows:
            out[label_of(r, label_key)].append(text_of(r, field))
        return out
    s_rows, r_rows = by_label(synth), by_label(real)
    rng = random.Random(seed)
    table, ok = {}, True
    for label in sorted(set(s_rows) & set(r_rows)):
        if not label:
            continue
        n = min(sample, len(s_rows[label]), len(r_rows[label]))
        if n < 2:
            continue
        s_pick, r_pick = rng.sample(s_rows[label], n), rng.sample(r_rows[label], n)
        rep_s, rep_r = repeat_rate(s_pick), repeat_rate(r_pick)
        sim_s = tfidf_mean_sim(s_pick, SIM_PAIRS, seed)
        sim_r = tfidf_mean_sim(r_pick, SIM_PAIRS, seed)
        rep_ratio, sim_ratio = ratio(rep_s, rep_r), ratio(sim_s, sim_r)
        passed = rep_ratio <= DIVERSITY_RATIO and sim_ratio <= DIVERSITY_RATIO
        ok = ok and passed
        table[label] = {"n": n, "repeat_synth": round(rep_s, 4), "repeat_real": round(rep_r, 4),
                        "repeat_ratio": rep_ratio, "sim_synth": round(sim_s, 4), "sim_real": round(sim_r, 4),
                        "sim_ratio": sim_ratio, "pass": passed}
    return {"ruler": "diversity", "field": field, "labels": table, "pass": ok and bool(table)}


def ratio(synth_value, real_value):
    """실물이 0 이면 Ⅱ 도 0 일 때만 1 로 친다 (0 으로 나누기 대신)."""
    if real_value > 0:
        return synth_value / real_value
    return 1.0 if synth_value <= 0 else math.inf


def text_of(row, field):
    if field == "both":
        return f"{row.get('evidence', '')} {row.get('claim', '')}"
    return row.get(field, "")


SOURCE_FOLDS = 5


def tfidf_vectors(texts, sizes=(3,)):
    """글자 n묶음 TF-IDF (길이 1 로 정규화) — numpy 없이 dict 로."""
    docs = [collections.Counter(char_ngrams(t, sizes)) for t in texts]
    n = len(docs)
    df = collections.Counter(g for d in docs for g in d)
    vecs = []
    for d in docs:
        v = {g: (1 + math.log(tf)) * math.log((1 + n) / (1 + df[g])) for g, tf in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({g: x / norm for g, x in v.items()})
    return vecs


class NaiveBayes:
    """다항 나이브 베이즈 — TF-IDF 무게를 낱개 수 대신 쓴다. 두 칸 수를 맞춰 넣으니 앞선 몫은 같다고 둔다."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.logp = {}

    def fit(self, xs, ys):
        sums = {0: collections.defaultdict(float), 1: collections.defaultdict(float)}
        for x, y in zip(xs, ys):
            for g, v in x.items():
                sums[y][g] += v
        vocab = set(sums[0]) | set(sums[1])
        for y in (0, 1):
            total = sum(sums[y].values()) + self.alpha * (len(vocab) + 1)
            self.logp[y] = ({g: math.log((sums[y].get(g, 0.0) + self.alpha) / total) for g in vocab},
                            math.log(self.alpha / total))
        return self

    def predict(self, x):
        def score(y):
            table, unseen = self.logp[y]
            return sum(v * table.get(g, unseen) for g, v in x.items())
        return 1 if score(1) >= score(0) else 0


def source_probe(synth, real, field="evidence", folds=SOURCE_FOLDS, seed=SEED):
    """자 ② (진단만) : 근거 글자 3묶음 TF-IDF + 나이브 베이즈로 「합성이냐 실물이냐」를 5겹으로 맞혀 본다.

    두 쪽 수를 작은 쪽에 맞춰 뽑아 우연 정확도를 0.5 로 둔다. 겹은 열쇠 묶음(group) 단위로 나눠
    같은 씨앗에서 나온 합성 넷이 학습 · 검증 양쪽에 걸치지 않게 한다. 0.5 에 가까울수록 출처 말투가 안 샌다.
    IDF 는 전체 글로 한 번 셈한다(라벨을 안 보는 셈이라 새는 것이 작다).
    """
    rng = random.Random(seed)
    n = min(len(synth), len(real))
    if n < folds:
        return {"ruler": "source", "diagnostic": True, "n_each": n, "acc": None,
                "note": "쌍이 너무 적다 — 관문 아님"}
    rows = [(r, 1) for r in rng.sample(synth, n)] + [(r, 0) for r in rng.sample(real, n)]
    vecs = tfidf_vectors([text_of(r, field) for r, _ in rows])

    def fold_of(row):
        g = row.get("group") or row.get("synth_from") or row.get("id") or text_of(row, field)
        return zlib.crc32(f"{seed}:{g}".encode("utf-8")) % folds
    fold_ids = [fold_of(r) for r, _ in rows]
    accs, right, total = [], 0, 0
    for k in range(folds):
        tr = [i for i, f in enumerate(fold_ids) if f != k]
        va = [i for i, f in enumerate(fold_ids) if f == k]
        if not tr or not va:
            continue
        model = NaiveBayes().fit([vecs[i] for i in tr], [rows[i][1] for i in tr])
        hit = sum(1 for i in va if model.predict(vecs[i]) == rows[i][1])
        accs.append(round(hit / len(va), 4))
        right += hit
        total += len(va)
    return {"ruler": "source", "diagnostic": True, "field": field, "n_each": n, "folds": accs,
            "acc": round(right / total, 4) if total else None, "chance": 0.5,
            "note": "관문 아님 — 0.5 에 가까울수록 합성과 실물이 안 갈린다 (ㄴ 진단용)"}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("claim-only", "diversity", "source"):
        s = sub.add_parser(name)
        s.add_argument("--synth", required=True, help="Ⅱ(또는 ㄱ) 라벨 붙은 쌍")
        s.add_argument("--real", required=True, help="dev-cmp 실물 쌍")
        s.add_argument("--label-key", default="want")
        s.add_argument("--seed", type=int, default=SEED)
        if name == "diversity":
            s.add_argument("--field", choices=("claim", "evidence", "both"), default="claim")
        if name == "source":
            s.add_argument("--diagnostic", action="store_true", help="진단 몫임을 알고 돌린다 (빠지면 안 돈다)")
            s.add_argument("--field", choices=("evidence", "claim", "both"), default="evidence")
            s.add_argument("--folds", type=int, default=SOURCE_FOLDS)
    a = p.parse_args(argv)
    if getattr(a, "folds", 2) < 2:
        sys.exit("--folds 는 2 이상")
    synth, real = read_jsonl(a.synth), read_jsonl(a.real)
    if a.cmd == "claim-only":
        res = claim_only(synth, real, a.label_key, a.seed)
    elif a.cmd == "diversity":
        res = diversity(synth, real, a.label_key, a.field, seed=a.seed)
    else:
        if not a.diagnostic:
            sys.exit("source 는 진단용이다 — --diagnostic 을 붙여야 돈다 (관문 아님)")
        res = source_probe(synth, real, a.field, a.folds, a.seed)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if res.get("pass") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
