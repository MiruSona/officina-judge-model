"""bm25 — 열쇠 묶음 안 근접 순위 (설계 3-2 「near」 꼴 · 근접 c).

토큰은 「공백으로 끊은 낱말 + 낱말 안 글자 2묶음」이다. 한국어는 조사가 붙어도
2묶음이 겹쳐 잡히고, 영어는 낱말이 그대로 잡힌다. 두 말을 같은 함수로 다룬다.
새 패키지(rank_bm25 등)를 쓰지 않으려고 직접 짰다.
"""
import collections
import math
import re

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏ가-힣ㄱ-ㆎ]+")

NEAR_LO = 10
NEAR_HI = 60


def tokenize(text):
    """낱말(소문자) + 낱말 안 글자 2묶음."""
    words = _WORD.findall(text.lower())
    out = list(words)
    for word in words:
        if len(word) >= 3:
            out.extend(word[i:i + 2] for i in range(len(word) - 1))
    return out


class BM25:
    """문서 목록 하나에 대한 BM25 점수 · 순위."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.size = len(docs)
        self.lengths = []
        self.postings = collections.defaultdict(list)
        for index, doc in enumerate(docs):
            counts = collections.Counter(tokenize(doc))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((index, tf))
        self.avg_len = (sum(self.lengths) / self.size) if self.size else 0.0

    def _idf(self, term):
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def scores(self, query):
        """질의 글 → {문서 번호: 점수}. 점수 0 인 문서는 빠진다."""
        out = collections.defaultdict(float)
        for term, qtf in collections.Counter(tokenize(query)).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for index, tf in postings:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avg_len or 1.0))
                out[index] += qtf * idf * tf * (self.k1 + 1) / norm
        return out

    def rank(self, query, exclude=()):
        """점수 높은 차례의 문서 번호 목록 (점수 0 · exclude 는 뺀다, 같은 점수는 번호 차례)."""
        skip = set(exclude)
        scored = [(-score, index) for index, score in self.scores(query).items() if index not in skip]
        scored.sort()
        return [index for _, index in scored]

    def near(self, query, exclude=(), lo=NEAR_LO, hi=NEAR_HI):
        """순위 lo~hi 위(1부터 셈)의 문서 번호 — 설계 3-2 의 근접 띠."""
        ranked = self.rank(query, exclude)
        return ranked[lo - 1:hi]
