# train-g2-b — v1.2 B 판 학습 층 (VitaminC 대조쌍)

비공개 설계 5-1 · 5-2 · 7절. 라벨 기준 support-v6. 본문은 git 밖 `~/.judgemodel/` 에만 있다. 여기에는 수와 명령만 적는다.
한국어 D 층은 비웠다(결정 2026-10-12, 설계 8절 ⑦).

## 0. 결론

- **400 짝 = 800쌍 (a 400 · b 400)을 얼렸다.** 후보 650 짝 → 판정 → 짝 규칙. 설계 어림 약 450 짝보다 50 짝 적다.
- 한정어 · 수 · 서수 주장 몫은 **66% (265 / 400 짝)**. 후보 때 55% 였는데, 낱말 하나 바꿈 꼴이 더 많이 버려져 올랐다.
- 판정 κ 전체 .928 · c/d κ .757 (관문 .60). 함정 286 / 290. 겹침 · scrub 0.

## 1. 후보 고른 규칙 (`build/pick_contrast.py pick`)

| 규칙 | 값 |
| --- | --- |
| 입력 | `build/g2/reserve/train-layer-cands.jsonl` 의 VitaminC 대조쌍 31,714 짝 (A-1 이 예약 · 금지를 이미 뺀 학습 층) |
| 짝 | 같은 `pair`(같은 판 고침 · 같은 주장)의 a 줄 · b 줄을 짝째 고른다 |
| 묶음 | 위키 문서(`group`) 하나에 한 짝 → 650 문서 |
| 금지 | 얼린 셋 셋(dev-cal-g2 · dev-cmp-g2 · blind-g2)의 묶음 · 출처 id · (근거, 주장) · 한쪽 글 + `blind-g2.forbid.json` 의 묶음 · 출처 id · 쌍 열쇠. 걸린 짝 0 |
| 꼴 `qual` | 주장에 less · more · fewer · than · before · after · over · under(진단 1-4 표) · 수 · 서수가 있음 — 몫 0.55 먼저 |
| 꼴 `swap` | 나머지 중 b 근거와 주장이 낱말 하나 바꿈(`rule_filter` ③) — 나머지 몫 |
| 꼴 `other` | 그 밖 — 안 썼다 (swap 이 모자라지 않음) |
| 씨앗 | 고르기 20261301 · 묶음 20261302 |

| | 풀 (짝) | 고름 (짝) | 남김 (짝) | 남김률 |
| --- | --- | --- | --- | --- |
| qual | 19,680 | 358 | 265 | .74 |
| swap | 9,644 | 292 | 135 | .46 |
| other | 2,390 | 0 | — | — |
| 합 | 31,714 | 650 | 400 | .62 |

- 고른 qual 358 짝의 주장 : 한정어 낱말 252 · 수만 74 · 서수 32.

## 2. 판정

- 1,300쌍 + 함정 144 → 100쌍 묶음 15 × j1 · j2 = 30판 + 셋째 2판(t01 14쌍 · t02 43쌍). Opus 판정 **32판 · 2,945건**(함정 포함 : 1,444 × 2 + 14 + 43).
- 함정 정답 286 / 290 (판마다 .89 ~ 1.00). 답 없음 0. 다시 돌린 판 0.
- **함정 t0046 을 관문에서 뺐다**(`merge --exempt-trap t0046`). 풀 라벨 b 인데 서로 다른 판정자 셋(b11-j1 · b11-j2 · t02-j3)이 모두 d 로 봤다. build/README 가 이미 「애매한 함정」으로 적은 것이다. 빼지 않으면 t02-j3 이 3/4 로 버려진다. A-2 의 t0007 은 이번 묶음에 안 뽑혀 뺄 것이 없었다.
- κ : 전체 **.928** (1,300) · c/d κ **.757** (102). 첫 두 판 일치 1,248 / 1,300 (96%).
- 정함 : 합의 1,248 · 셋째 46 · cd-split 3 · 버림 3 → a 733 · b 438 · d 111 · c 10 · cd 5.

hint(뽑은 판) × 판정 칸

| hint \ 판정 | a | b | c · cd | d | 버림 |
| --- | --- | --- | --- | --- | --- |
| a (SUPPORTS 판) | 617 | 14 | 2 | 17 | 0 |
| b (REFUTES 판) | 116 | 424 | 13 | 94 | 3 |

- REFUTES 판의 b 남김은 424 / 650 (65%). 진단 1-4 의 한정어 74% · 그 밖 40% 와 비슷한 갈림이다.

## 3. 짝 규칙 (`pick_contrast.py keep-pairs`)

a 줄이 a, b 줄이 b 로 **둘 다** 남은 짝만 넣는다. 하나라도 어긋나면 짝째 버린다.

| 까닭 | 짝 |
| --- | --- |
| 남김 | **400** |
| c · d · cd 가 하나라도 나옴 | 122 |
| 두 줄 다 a (b 판이 a 로) | 114 |
| 두 줄 다 b (a 판이 b 로) | 10 |
| 칸이 거꾸로 (b · a) | 1 |
| 한쪽 버림 (셋 다 다름) | 3 |
| 합 | 650 |

- 「두 줄 다 a」 114 짝은 판정 자체는 맞을 수 있다(바뀐 근거가 주장을 여전히 받침). 대조쌍이 아니라서 뺐다. 넣으면 a 만 228쌍 늘어 b 비율이 떨어진다.

## 4. 셋 · 해시 · 검사

| 무엇 | 값 |
| --- | --- |
| 파일 | `~/.judgemodel/sets/g2/train-g2-b.jsonl` — 800줄 · a 400 · b 400 · 꼴 qual 265 · swap 135 짝 · label_guide support-v6 · status agree 790 · third 10 |
| sha256 | `3b61037a974a880596bbff99d61ff075af468de6cffb7d6f6aa2a1d8e1f9aeee` (`FROZEN-train-g2-b.sha256`, `freeze.py check` ok) |
| `check-g2 --train <train-g2-b>` | 학습 × 평가 예약 후보(묶음 · 글 · 출처 id) · 학습 × 옛 셋(글 · 기억 id · g1 평가 묶음 · use1 쌍 · use1 글 · 잣대) 모두 0 (exit 0) |
| 얼린 셋 셋 + blind-g2.forbid.json 다시 보기 | 묶음 · 글 · 한쪽 글 0 (blind 보강이 끝난 00:07 판 파일 기준) |
| `scrub.py check` | 0 건 (800줄) |
| 학습 b 비율 (어림) | B2 767 / 4,401 (17%) → +400 / +800 → 1,167 / 5,201 (22%) |

## 5. 명령

`build/` 에서 돈다. `<g2>` = `~/.judgemodel/build/g2`, `<S>` = `~/.judgemodel/sets/g2`.

```
python -I pick_contrast.py pick --cands <g2>/reserve/train-layer-cands.jsonl \
    --sets <S>/dev-cal-g2.jsonl <S>/dev-cmp-g2.jsonl <S>/blind-g2.jsonl --forbid <S>/blind-g2.forbid.json \
    --out <g2>/train-g2-b/cands.jsonl --n 650 --qual-share 0.55
python -I make_sets.py check-g2 --build <g2> --train <g2>/train-g2-b/cands.jsonl
python -I label_batches.py make --cands <g2>/train-g2-b/cands.jsonl --out <g2>/train-g2-b --mode train \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261302
#   판정자 안내 : <g2>/train-g2-b/JUDGE-README.txt (Opus, 한 판에 묶음 하나)
python -I label_batches.py merge --out <g2>/train-g2-b --exempt-trap t0046
python -I label_batches.py kappa --out <g2>/train-g2-b
python -I label_batches.py decide --out <g2>/train-g2-b --mode train ; third ; (j3 판정) ; merge ; decide
python -I label_batches.py assemble --mode train --set train-g2-b-all --dirs <g2>/train-g2-b --dest <g2>/train-g2-b/assembled
python -I pick_contrast.py keep-pairs --set <g2>/train-g2-b/assembled/train-g2-b-all.jsonl --cands <g2>/train-g2-b/cands.jsonl --out <S>/train-g2-b.jsonl
python -I ../train/freeze.py write --pairs <S>/train-g2-b.jsonl --out <S>/FROZEN-train-g2-b.sha256
python -I make_sets.py check-g2 --build <g2> --train <S>/train-g2-b.jsonl
python -I scrub.py check --in <S>/train-g2-b.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
```

- `check-g2 --train` 은 이번에 더한 옵션이다(기본은 예약 때의 `train-layer-cands.jsonl`).
- 다음(B4) : B2 처방 그대로 `--train` 에 이 파일 하나 더(설계 5-2).

## 6. B4h 절반 (`pick_contrast.py halve`, 2026-10-13)

설계 5-3 의 대체 판 층이다. B4 짝 비교가 「지킴 깨짐」(d 아래끝 −.041)이라 한 번만 돌린다. 연구 기록은 공개하지 않는다.

- 원천이 VitaminC 하나라, 「원천별」은 고른 꼴 `pick`(qual · swap) 칸으로 나눴다. 칸마다 짝 수의 절반(내림)을 짝째(a · b 같이) 뽑는다.
- 씨앗 **20261303** (고르기 20261301 · 묶음 20261302 다음). 줄 차례는 원래 파일 그대로.

| 칸 | 짝 | 남김 |
| --- | --- | --- |
| qual | 265 | 132 |
| swap | 135 | 67 |
| 합 | 400 | **199 짝 = 398쌍 (a 199 · b 199)** |

| 셋 | sha256 |
| --- | --- |
| `train-g2-b-half.jsonl` | `76c183b562aeedb2b1132458b38061c0764b38d4999a234e951100fcfa53a9d8` (`FROZEN-train-g2-b-half.sha256`) |

| 검사 | 결과 |
| --- | --- |
| `check-g2 --train <S>/train-g2-b-half.jsonl` | 학습 × 평가 예약 · 학습 × 옛 셋 모두 0 (exit 0, train_rows 398) |
| `scrub.py check` | 0 건 (398줄) |

```
python -I pick_contrast.py halve --set <S>/train-g2-b.jsonl --out <S>/train-g2-b-half.jsonl
python -I ../train/freeze.py write --pairs <S>/train-g2-b-half.jsonl --out <S>/FROZEN-train-g2-b-half.sha256
```
