# g4 한국어 평가 몫 — 뽑기 · 판정 (v1.2 K2 판 흐름 3단계)

비공개 설계 1절 3단계 · 3-1 · 7절 4b. 원천 자리는 비공개 작업 기록(2-2절)에 있다.
쌍 본문은 git 밖 `~/.judgemodel/build/g4/` 에만 있다. 여기에는 수 · 경로 · 명령만 적는다. 예약(`reserve-g4`)은 다음 단계 몫이다.

**결론 (2026-10-14) :** 한국어 평가 몫 **410쌍**(위키 180 · BoolQ 100 · KLUE 130)을 새로 뽑아 v6 두 판(Opus)으로 판정했다. 첫 두 판 일치 **97.8%**(401/410), 갈린 9쌍은 셋째 판이 모두 정했다(버림 0). 확정 칸은 **b 300 · a 93 · d 17 · c 0**. reserve-g4 를 시험 자리(스크래치)에서 돌려 보니 ko b 목표(위키 110 · BoolQ 90 · KLUE 50)와 c 목표를 다 채웠고, 넘김 0 · 겹침 검사 0 이다.

## 1. 뽑기

### 1-1. 몫 셈 (목표 ÷ g3 남김률 + dev-cmp c 몫)

reserve-g4 는 blind-g4 ko b 를 먼저 채우고, **남은 묶음**에서 dev-cmp-g4 near-x c 를 만든다(`make_sets.reserve_g4`). 그래서 판정 끝 줄이 「blind b 목표 + dev-cmp c 근거 줄 수」만큼 있어야 한다.

| 원천 | blind b 목표 | 이미 판정 끝 (g4 금지 뒤) | g3 남김률 (hint b → b) | dev-cmp c 몫 (남은 줄) | 새로 뽑은 수 |
| --- | --- | --- | --- | --- | --- |
| 위키 | 110 | 0 | .815 | 43 | **b 180** (110 ÷ .815 ≈ 135 + c 몫 45) |
| BoolQ | 90 | b 84 | .893 | 54 | **b 70 · a 30** (모자란 b 6 + 넘김 받을 여유 + c 몫) |
| KLUE | 50 | b 19 · a 11 | .884 | 43 | **b 90 · a 40** ((50 − 19) ÷ .884 ≈ 35 + c 몫. a 는 같은 문단 짝) |

- 위키는 문서당 b 2 · flag 상한 20%(`wiki_pick.pick_rows`), BoolQ · KLUE 는 문단당 b 2 · a 는 같은 문단 먼저(`public_pick.pick_rows`). 씨앗 **20261402**.
- 「이미 판정 끝」 수는 g3 판정 파일에 G4 금지(`G4Forbid`, 평가 쪽)를 걸어 다시 셌다. 조사 문서 2-2 와 같다(BoolQ b 84 · KLUE b 19 · a 11 · 위키 0).

### 1-2. 뽑기 전 남은 줄 (g3 가 뽑은 줄 · G4 금지를 뺀 뒤)

| 원천 | 원본 | g3 뽑음 뺀 뒤 | G4 금지로 빠짐 | **가용** | 조사 문서 2-2 |
| --- | --- | --- | --- | --- | --- |
| 위키 `wiki-all-eval` | 1,005 | 691 | 292 (group_g3 · g3_reserved 292 · blind_g3_forbid 156 · para_g3 70 · src_g3 38, 겹쳐 셈) | **b 399 · 273 문서** | b 409 · 277 문서 |
| BoolQ dev · test 평서문 | 2,102 → 준비 2,084 (평서문 검사 2 · near_train 16) | 1,634 | 11 (para_g3_train 6 · para_g3 5) | **a 870 · b 753** | 약 1,640 |
| KLUE `answered-rw` heldout | 5,672 → 준비 1,162 (평서문 검사 128 · 맞은편 split · 겹침 문단 4,382) | 1,007 | 37 | **a 489 · b 481 · 482 문단** | 약 1,150 |

- KLUE 가 조사 어림보다 약 180 적은 것은 평서문 검사 떨어짐(짝 64)을 조사가 안 뺐기 때문이다. 필요한 몫(130)의 7배가 넘어 멈춤 조건이 아니다.
- `scrub.py check` — 뽑은 410줄 **0건**.

## 2. 판정

### 2-1. 묶음

`label_batches.py make --mode eval` · 100쌍 묶음 · 함정 10% · 판정자 둘. 판정자는 `JUDGE-README.txt` 와 자기 묶음 파일만 본다(원천 · hint · 순위 숨김). 라벨 기준 `../Docs/Guide/라벨기준support-v6.md`.

| 자리 | 값 |
| --- | --- |
| 묶음 | b01~b04 100줄 · b05 56줄 (실제 410 + 함정 46) × j1 · j2 |
| 셋째 묶음 | t01 10줄 (갈린 9 + 함정 1) × j3 |
| 함정 정답률 | 판 11개 모두 통과 (1.0 여덟 · 0.9 둘 · t01 1/1) · 답 없음 0 |
| Opus 판정자 | **11개** (첫 두 판 10개 동시 · 셋째 1개). 한 판 1~2분 |
| 판정 줄 (함정 포함) | **922** (912 + 10). 실제 쌍 판정 829 = 쌍당 2.02 |
| 어림 대비 | 이 단계 몫 어림 410 × 2.3 ≈ 943 → 0.98배 (멈춤 선 1.5배) |

### 2-2. 일치 · 갈림

| 원천 (domain) | 첫 두 판 일치 |
| --- | --- |
| kowiki | .978 |
| kobest-boolq | .980 |
| klue-answered | .977 |
| **전체** | **.978 (401/410) · κ .946** |

- **갈림 9쌍 = 2.2%** (멈춤 선 15%). 셋째 판으로 9쌍 모두 다수가 생겨 버린 줄은 0 이다.
- `label_batches kappa` 의 c/d κ 칸은 c/d 표가 12개뿐이라 값이 안 나온다(`cd: null` → pass false). 학습 묶음용 관문이라 이 판에는 안 건다.
- label_batches 판정에는 noise 칸이 없다. g3 의 `wiki_pick judge` 는 noise 를 따로 버렸는데(위키 평가 몫 11/314), 이번 판은 깨진 줄도 a · b · c · d 중 하나로 들어갔다. 다음 단계 조립 뒤 gold-g4 에서 본다.

### 2-3. 칸별 결과 (새 410줄)

| 원천 | hint | 줄 | a | b | c | d | 남김 | hint 와 같음 (남김률) | g3 값 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 위키 | b | 180 | 23 | **151** | 0 | 6 | 180 | **.839** | .815 |
| BoolQ | b | 70 | 0 | **68** | 0 | 2 | 70 | **.971** | .893 |
| BoolQ | a | 30 | **30** | 0 | 0 | 0 | 30 | **1.0** | .967 |
| KLUE | b | 90 | 1 | **81** | 0 | 8 | 90 | **.900** | .884 |
| KLUE | a | 40 | **39** | 0 | 0 | 1 | 40 | **.975** | .967 |
| **합** | | **410** | 93 | 300 | 0 | 17 | **410** | | |

- 남김률은 모두 g3 값 이상이다. 위키 hint b 중 a 23(13%)은 g3(13%)와 같은 꼴(표기 고침 · 품는 값)로 본다(판정자 까닭은 `votes.jsonl` 에만 있다).

## 3. reserve-g4 가 읽는 파일

`make_sets.G4_INPUTS` 이름 그대로다. 옛 g3 판정 줄 뒤에 새 줄을 이었고 id 겹침 0 을 확인했다. 새 줄은 `g4_pick: "eval-ko"` · `judge.via: "label_batches g4/eval-ko"` 로 가린다.

| 파일 (`~/.judgemodel/build/g4/`) | 줄 | 옛 g3 줄 | 새 줄 | keep |
| --- | --- | --- | --- | --- |
| `kowiki/wiki-labeled-eval.jsonl` | 481 | 301 | 180 | 481 (finalize 꼴 — keep 줄만, `hint` = 확정 칸, `hint_pick` = 뽑을 때 hint) |
| `boolq/judged-eval.jsonl` | 550 | 450 | 100 | 543 (g3 버림 7) |
| `klue/judged-eval.jsonl` | 285 | 155 | 130 | 283 (g3 버림 2) |

### 3-1. 미리 돌려 본 reserve-g4 (스크래치 자리 · 정식 예약 아님)

세 파일을 스크래치로 복사해 `make_sets.py reserve-g4 --build <스크래치>` 를 한 번 돌렸다. `~/.judgemodel/build/g4/reserve/` 는 만들지 않았다.

| 자리 | 값 |
| --- | --- |
| blind-g4 ko b | 위키 111 / 110 · BoolQ 90 / 90 · KLUE 50 / 50 — 넘김(shift) 0 · 기다림(pending) 0 |
| blind-g4 ko a | 위키 1 · KLUE 21 · BoolQ 0 = **22** |
| blind-g4 ko c (near-x) | 위키 54 · BoolQ 54 · KLUE 43 |
| dev-cmp-g4 새 c | BoolQ 54 · 위키 43 · KLUE 43 (모두 목표대로) |
| `check_g4` | 모든 칸 0 |
| `min_groups_preview` (판정 앞) | b@en 290/200 · **a 311/380** · c 265/260 · b@ko 238/220 |

- **a 묶음이 모자랄 수 있다.** 설계 3-1 은 ko a 180 을 어림했지만 reserve-g4 는 a 를 「b 묶음 안 같은 문단 a」만 가져간다. BoolQ 는 문단마다 질문이 하나라 같은 문단 a 가 없고(g3 8-1), 위키 a 는 판정이 b → a 로 바꾼 줄뿐이라 ko a 는 22 에 그쳤다. a 판정을 더 해도 이 규칙으로는 blind 에 안 들어간다. 다음 단계(4 · 7)의 `min_groups` 검사에서 걸릴 수 있어 메인에 넘겼다.

## 4. 명령

```
# 뽑기 (git 밖 손 스크립트 — wiki_pick · public_pick · make_sets 함수만 부른다)
python -I ~/.judgemodel/build/g4/prep/pick_g4.py --jm-build <JudgeModel>/build avail
python -I ~/.judgemodel/build/g4/prep/pick_g4.py --jm-build <JudgeModel>/build pick \
    --wiki-b 180 --boolq-b 70 --boolq-a 30 --klue-b 90 --klue-a 40          # 씨앗 20261402 → prep/pick-eval-g4.jsonl
python -I scrub.py check --in ~/.judgemodel/build/g4/prep/pick-eval-g4.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
# 판정
python -I label_batches.py make --cands ~/.judgemodel/build/g4/prep/pick-eval-g4.jsonl --out ~/.judgemodel/build/g4/eval-ko --mode eval \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261402
python -I label_batches.py merge  --out ~/.judgemodel/build/g4/eval-ko
python -I label_batches.py decide --out ~/.judgemodel/build/g4/eval-ko --mode eval
python -I label_batches.py third  --out ~/.judgemodel/build/g4/eval-ko      # 셋째 판 뒤 merge · decide 를 한 번 더
# reserve-g4 가 읽는 이름으로 (옛 g3 판정 줄 + 새 줄)
python -I ~/.judgemodel/build/g4/prep/finalize_g4.py
```

- 손 스크립트 둘(`pick_g4.py` · `finalize_g4.py`)과 요약(`pick-eval-g4.summary.json` · `finalize-g4.summary.json`)은 `~/.judgemodel/build/g4/prep/` 에 있다.

## 5. 학습 기계 동기화

학습 기계의 사본(`<작업 뿌리>/src/`)이 저장소 판과 어긋나 두 번 멈췄다(10-13 compare · 10-14 gates). 그래서 학습 기계 일(6단계)을 시작하기 전에 **네 폴더(`measure` · `build` · `train` · `tests`)를 통째로 sha 대조**한다.

- 10-14 맞춤 : 작업 트리 판(미커밋 포함) 107 파일을 복사 → 다른 파일 0 · 학습 기계 에만 있는 파일 0. 학습 기계 시험 588 통과 · skip 35 (작업 PC 32 보다 셋 많음 — 작업 PC 에만 있는 옛 r2 판 pred · calibration 이 없어서다).
- 아래 `<학습 기계>` 자리에 그 기계의 ssh 별칭을, `<작업 뿌리>` 에 학습 작업 폴더(`JT_ROOT`)를 넣는다.
- 저장소 뿌리에서 Git Bash 로 돌린다. 끝에 `같음` 이 찍혀야 통과다. 줄이 나오면 그 파일을 다시 복사한다.

```bash
diff <(git ls-files -co --exclude-standard -z measure build train tests | LC_ALL=C sort -z | xargs -0 sha256sum | sed 's/ \*/  /') <(ssh <학습 기계> "cd <작업 뿌리>/src && find measure build train tests -type f -not -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum") && echo 같음
```

- 복사는 `git ls-files -co --exclude-standard -z measure build train tests > <목록>` 뒤 `tar -cf - --null -T <목록> | ssh <학습 기계> "cd <작업 뿌리>/src && tar -xf -"` 로 한다. 쌍 · 셋 · 모델은 gitignore 라 목록에 안 들어간다.
- 이 파일(`sets-g4.md`)도 네 폴더 안이라, 고치면 다시 복사해야 대조가 맞는다.

## 6. blind-g4 예약 (`reserve-g4`, 2026-10-14)

설계 1절 4단계 · 3-1 · 3-4. 정식 예약이다 — `~/.judgemodel/build/g4/reserve/` 에 네 파일을 썼다.

**결론 :** exit 0 · `check_g4` 42칸 모두 0 · 넘김(ko b) 0 · 기다림(pending) 0. 판정 앞 묶음 미리보기는 b@en **290**/200 · a **411**/380 · c **301**/260 · b@ko **250**/220 로 네 칸 다 넘었다. 후보는 blind-g4 **1,843줄** · dev-cmp-g4 새 c **124줄**.

### 6-1. 목표 손질 — dev-cmp-g4 위키 c 몫을 BoolQ 로

- 위키 평가 몫이 blind-g4 b · c 에 먼저 쓰여, dev-cmp-g4 위키 c 근거 묶음이 **24/43** 뿐이다(3-1 미리 돌림과 같음).
- 모자란 19 를 BoolQ 로 넘겨 dev-cmp 한국어 c 합을 지키려고 `--targets` 로 dev-cmp-g4 c 를 BoolQ **73** · 위키 **24** · KLUE 43 으로 줬다(메인 결정). 목표 파일 `~/.judgemodel/build/g4/prep/targets-g4c.json`. 얕은 합침이라 c 칸 전체(blind-g4 몫 포함 · 기본값 그대로)를 같이 적었다.
- **BoolQ 는 57 까지만 찼다**(남은 묶음이 모자람). 그래서 dev-cmp-g4 새 c 는 **124**(BoolQ 57 · 위키 24 · KLUE 43)로, 기본 목표의 121 보다 3 많다. 설계 3-4 의 약 130 에 6 모자란다.
- 같은 목표로 스크래치에서 먼저 돌린 판과 두 후보 파일이 바이트 같다(씨앗 20261401).

### 6-2. 셋 × 원천 × 칸 (판정 앞 후보)

| 셋 | 원천 | 줄 | a · b · c | 묶음 | 목표 대비 |
| --- | --- | --- | --- | --- | --- |
| blind-g4 | 위키 | 186 | 22 · 110 · 54 | 131 | b 110/110 · a 22/45 · c 54/54 |
| blind-g4 | KLUE | 117 | 24 · 50 · 43 | 53 | b 50/50 · a 24/45 · c 43/43 |
| blind-g4 | BoolQ | 174 | 30 · 90 · 54 | 120 | b 90/90 · a 30/134 (위키 23 · KLUE 21 넘겨 받음) · c 54/54 |
| blind-g4 | VitaminC | 1,366 | 640 · 576 · 150 | 336 | b 쪽 290 · a 335/335 · 같은 쪽 c 75/75 · near-x c 75/75 |
| dev-cmp-g4 | BoolQ | 57 | 0 · 0 · 57 | 57 | c 57/73 |
| dev-cmp-g4 | 위키 | 24 | 0 · 0 · 24 | 24 | c 24/24 |
| dev-cmp-g4 | KLUE | 43 | 0 · 0 · 43 | 43 | c 43/43 |

- 한국어 a · b 줄(276)은 판정이 끝난 줄(`judge` 칸)이라 다시 판정하지 않는다. 판정할 줄은 VitaminC 1,366 + 한국어 near-x c 151 + dev-cmp-g4 c 124 = **1,641**.
- 넘김은 a 칸에만 있다 : `a:wiki->boolq` 23 · `a:klue->boolq` 21. BoolQ 는 문단마다 질문이 하나라 같은 문단 a 가 거의 없어 a 104 가 모자랐고, 모자란 a 묶음은 VitaminC a 335 가 메운다(설계 기본값).
- `blind-g4.forbid.json` : 묶음 640 · 출처 id 1,699 · 후보 id 1,843 · 쌍 열쇠 1,843. 얼릴 때 `sets/g4/` 로 복사한다(4e).
- `scrub.py check` — 두 후보 파일 0 건.

### 6-3. 금지로 빠진 줄 (`dropped_eval`, 까닭 겹쳐 셈)

| 원천 | 까닭 |
| --- | --- |
| 위키 | g3 예약 · 묶음 · 출처 · 글 · 문단 각 301 · blind-g3 금지 166 |
| KLUE | 위 다섯 각 123 · blind-g3 금지 66 · keep false 2 |
| BoolQ | 위 다섯 각 357(문단 358) · blind-g3 금지 186 · keep false 7 · train-g3-ko 문단 1 |
| VitaminC heldout | g1 묶음 4,253 · g3 묶음 3,311 · g2 예약 2,318 · 옛 셋 문장 1,938 · 옛 셋 글 1,214 · blind-g2 금지 1,200 · g3 글 1,047 · g3 예약 · blind-g3 금지 각 938 · g3 출처 589 |

## 7. blind-g4 · dev-cmp-g4 평가 판정 (2026-10-14)

**결론 :** 새 줄 1,641쌍을 v6 두 판(Opus)으로 판정했다. 첫 두 판 일치 **97.0%**(1,592/1,641) — 갈린 49쌍은 모두 blind-g4 VitaminC 줄이고, 셋째 판이 45쌍을 정하고 **4쌍은 버렸다**(셋이 다 다름). 함정은 판 39개 모두 통과(최저 .8). 그러나 판정 뒤 blind-g4 **c 묶음이 248 로 260 에 12 모자라** 멈췄다(8절).

### 7-1. 묶음

`label_batches.py make --mode eval` · 100줄 묶음 · 함정 10% · 판정자 둘. 4b 의 `eval-ko/` 와 따로, `assemble_g4` 가 기본으로 읽는 자리 `<g4>/eval-public/<셋>/` 에 셋마다 폴더를 뒀다. `JUDGE-README.txt` 는 `eval-ko/` 것에서 경로 · 이름만 바꿨다(`make` 가 만들지 않는다). 판정자는 이 안내 · 문서 둘 · 자기 묶음만 본다(원천 · hint · 뽑은 꼴 · BM25 순위는 `map/` · `prep/` 에만).

| 자리 | blind-g4 | dev-cmp-g4 |
| --- | --- | --- |
| 판정 대상 | 1,517 (`prep/judge-cands.jsonl`) | 124 |
| 묶음 | b01~b16 100줄 · b17 86줄 (함정 169) | b01 100줄 · b02 38줄 (함정 14) |
| 씨앗 | 20261441 | 20261442 |
| 셋째 묶음 | t01 54줄 (갈린 49 + 함정 5) × j3 | 없음 |
| 함정 정답 | 판 35개 · 333/343 (최저 .8 — b09 두 판) | 판 4개 · 26/28 |
| 답 없음 | 0 | 0 |

- Opus 판정자 **21개**(첫 두 판 20개 — 한 개가 묶음 1~2개 · 동시 18개까지 · 한 판 1.5~3.5분, 셋째 1개). 벽시계는 셋째 판까지 약 9분.
- 판정 줄(함정 포함) **3,702** — 실제 쌍 판정 3,331 = 쌍당 2.03. 어림(약 1,700쌍 × 2.3 ≈ 3,900)의 0.95배, 멈춤 선(1.5배) 아래.
- 판정자 셋이 금지된 `cd` 를 한 번 썼다고 스스로 알렸다(모두 자기 묶음만 열었다고 함). 판정 결과에는 손대지 않았다.

### 7-2. 일치 · 갈림

| 원천 | 첫 두 판 일치 |
| --- | --- |
| vitaminc | .964 (갈림 49) |
| kowiki · kobest-boolq · klue-answered (near-x c) | 1.0 |
| dev-cmp-g4 셋 원천 | 1.0 |
| **전체** | **.970 · 갈림 3.0%** (멈춤 선 15%) |

### 7-3. 칸별 남김률 (hint → 확정 칸)

| 셋 | 원천 · 꼴 | hint | 줄 | a | b | c | d | 버림 | 남김률 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind-g4 | VitaminC labeled | a | 640 | **610** | 11 | 1 | 17 | 1 | **.953** |
| blind-g4 | VitaminC labeled | b | 576 | 94 | **365** | 10 | 104 | 3 | **.634** |
| blind-g4 | VitaminC near-x | c | 75 | 1 | 1 | **69** | 4 | 0 | **.920** |
| blind-g4 | VitaminC same-page | c | 75 | 1 | 2 | **24** | 48 | 0 | **.320** |
| blind-g4 | 위키 · BoolQ · KLUE near-x | c | 151 | 0 | 0 | **151** | 0 | 0 | **1.0** |
| dev-cmp-g4 | BoolQ · 위키 · KLUE near-x | c | 124 | 0 | 0 | **124** | 0 | 0 | **1.0** |

- **VitaminC 같은 쪽 c 가 .32 다.** 같은 위키 쪽의 다른 주장이라 판정자 둘이 대부분 「같은 대상 · 다른 사실」(d)로 봤다. 설계 3-1 은 이 꼴도 c 로 셈했다(어림 c 140 중 70).
- VitaminC b 는 .634 로 g3(.52)보다 높고 설계 어림(.78)보다 낮다. 빠진 b 는 a 94 · d 104.
- 판정 뒤 blind-g4 : 1,839쌍(a 782 · b 629 · c 255 · d 173) · ko b 250. dev-cmp-g4 : 2,343쌍(g3 이음 2,219 + 새 124) · ko c **471**(347 → 471).

## 8. 판정 뒤 묶음 수 — c 모자람으로 멈춤 (2026-10-14)

`assemble_g4.py assemble --draft` 둘 → `check --draft` 를 `~/.judgemodel/sets/g4-draft/` 에서 돌렸다(얼리지 않음 · `sets/g4/` 는 안 만듦). 겹침 8칸 0 · scrub 0. train-g4-c 는 아직 없어 학습 층 검사는 train-g3-ko 2,644 만 봤다.

| 자 | 판정 앞 (예약) | **판정 뒤** | min_groups | |
| --- | --- | --- | --- | --- |
| b@en | 290 | **212** | 200 | 넘음 (+12) |
| a | 411 | **405** (en 329 · ko 76) | 380 | 넘음 (+25) |
| c | 301 | **248** (en 97 · ko 151) | 260 | **12 모자람** |
| b@ko | 250 | **250** | 220 | 넘음 (+30) |

- **멈춤 조건(판정 뒤 min_groups 모자람)에 걸렸다.** `--force` 재예약은 메인 결정이라 하지 않았다. 4e 의 `assemble_g4 check`(draft 아님)는 지금 상태로는 exit 1 이다.
- **메울 길 (스크래치에서 확인만 함)** : 목표의 blind-g4 c 중 VitaminC near-x 를 75 → 95 로 올려 스크래치에서 돌리면, 옛 후보 1,843 · 124줄은 id · 내용이 **하나도 안 바뀌고** blind-g4 에 near-x c 20줄만 더해진다(check 0 · 미리보기 c 321). near-x 남김률 .92 면 판정 뒤 c 약 266 이다. 여유를 더 두려면 105(약 +28 → 약 276). 더한 20~30줄만 판정하면 된다(묶음 1개 × 2).
- b@en 은 +12 로 여유가 작다. 판정을 바꿀 일은 없어 얼린 뒤에도 그대로다.

### 8-1. 재예약으로 c 를 메움 (2026-10-14, 메인 결정)

**결론 :** blind-g4 c 의 VitaminC near-x 목표를 75 → **105** 로 올려 `reserve-g4 --force` 로 다시 예약했다. 옛 후보 줄은 하나도 안 바뀌었고 near-x c **30줄**만 더해졌다. 30줄은 두 판 모두 c 로 일치했다. 판정 뒤 c 묶음은 **277** 로 260 을 넘는다.

| 자 | 재예약 전 판정 뒤 | **재예약 뒤 판정 뒤** | min_groups |
| --- | --- | --- | --- |
| b@en | 212 | **212** | 200 |
| a | 405 | **405** | 380 |
| c | 248 | **277** | 260 |
| b@ko | 250 | **250** | 220 |

- **옛 줄 대조 :** 옛 파일은 `--force` 가 `<이름>.20261014-114018.<확장자>` 로 돌려 뒀다. 옛 blind-g4 1,843줄 · dev-cmp-g4 124줄은 새 파일에서 id 가 다 있고, 내용이 바뀐 줄 0 · 빠진 줄 0 이다. 옛 줄만 옛 차례로 이어 붙인 sha256 이 옛 파일 sha256 과 같다(bb3de23c… · 3361f2b2…). dev-cmp-g4 후보 파일은 바이트까지 같다.
- 새 출력 : blind-g4 **1,873줄** · 미리보기 c 331 · `check_g4` 0 · notice 없음. 목표 파일 `prep/targets-g4c2.json`. sha256 : `blind-g4-cands.jsonl` 850b2ecc… · `blind-g4.forbid.json` 3a0eb0da… · `reserve-g4.json` bedfb76a…
- **판정 :** 새 30줄만 `<g4>/eval-public/blind-g4-add/` 에서 묶음 하나(33줄, 함정 3, 씨앗 20261443) × 판정자 둘(Opus 2개)로 판정했다. 일치 30/30 · 모두 c · 함정 3/3 · 3/3 · 셋째 없음. 판정 줄 66.
- **합치기 :** `blind-g4-add/decided.jsonl` 30줄을 `blind-g4/decided.jsonl` 끝에 이었다(1,517 → 1,547). 옛 1,517줄은 바이트 같고, 합치기 전 판은 `decided.pre-add.jsonl` 로 남겼다. **`blind-g4` 폴더에서 `decide` 를 다시 돌리면 이 30줄이 빠진다** — 그때는 다시 잇거나 `--dirs` 에 두 폴더를 준다(합친 뒤 두 폴더를 같이 주면 id 가 겹쳐 멈춘다).
- 미리 조립(`sets/g4-draft/`) : blind-g4 1,869쌍(a 782 · b 629 · c 285 · d 173) · dev-cmp-g4 2,343쌍(그대로). 겹침 8칸 0 · scrub 0 · min_groups 모자람 없음.

## 9. 명령 (6 ~ 8절)

```
# 예약 (목표 파일 = 기본값 + dev-cmp-g4 c 를 BoolQ 73 · 위키 24 · KLUE 43 으로)
python -I make_sets.py reserve-g4 --build ~/.judgemodel/build/g4 --targets ~/.judgemodel/build/g4/prep/targets-g4c.json
python -I scrub.py check --in ~/.judgemodel/build/g4/reserve/<셋>-cands.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
# 판정 대상 = 예약 후보 중 judge 칸이 없는 줄 → <g4>/eval-public/<셋>/prep/judge-cands.jsonl (손 한 줄)
python -I label_batches.py make --cands <g4>/eval-public/blind-g4/prep/judge-cands.jsonl --out <g4>/eval-public/blind-g4 --mode eval \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261441      # dev-cmp-g4 는 20261442
python -I label_batches.py merge  --out <g4>/eval-public/<셋>
python -I label_batches.py decide --out <g4>/eval-public/<셋> --mode eval
python -I label_batches.py third  --out <g4>/eval-public/blind-g4        # 셋째 판 뒤 merge · decide 한 번 더
# 판정 뒤 미리 조립 (얼리지 않음)
python -I assemble_g4.py assemble --set dev-cmp-g4|blind-g4 --dest ~/.judgemodel/sets/g4-draft --draft
python -I assemble_g4.py check --dest ~/.judgemodel/sets/g4-draft --draft
```

- 예약 출력 sha256 : `blind-g4-cands.jsonl` bb3de23c… · `dev-cmp-g4-cands.jsonl` 3361f2b2… · `blind-g4.forbid.json` 53397113… · `reserve-g4.json` 266d4f0b…
- 재예약(8-1) : `reserve-g4 --build ~/.judgemodel/build/g4 --targets ~/.judgemodel/build/g4/prep/targets-g4c2.json --force` → 새 줄만 `<g4>/eval-public/blind-g4-add/prep/judge-cands.jsonl` → `label_batches make`(씨앗 20261443) · `merge` · `decide` → 그 `decided.jsonl` 을 `blind-g4/decided.jsonl` 끝에 잇기 → 위 미리 조립 두 줄.

## 10. c 층 train-g4-c (설계 1절 6단계 · 4절 · 7절 4d, 2026-10-14)

**결론 :** 학습 쪽 묶음에서 c 후보 **1,440쌍**(near-x 1,000 · c 짝 440)을 뽑아 v6 두 판(Opus)으로 판정했다. 첫 두 판 일치 **99.4%**(1,431/1,440) · 갈린 9쌍은 셋째가 정함. 확정 c **1,432** · d 2 · cd 5 · 버림 1. 원천 · 꼴 목표를 다 채워 **train-g4-c 1,300쌍** · **train-g4-c-h 1,100쌍**을 조립했다(겹침 · 금지 · scrub 0). 진단 확인(4-5)에서 K 의 c 재현은 **.647**(150쌍)로 .80 아래 — 「글 갈래 지름길」 가설대로다. 얼리기(`train/freeze.py write`)는 4e 몫이라 안 했다.

### 10-1. 뽑기 (`layer_g4c.py pick`)

- 금지는 도구 기본값이 정식 예약 자리를 읽는다(`make_guard`) : `reserve/blind-g4-cands.jsonl` · `dev-cmp-g4-cands.jsonl`(평가 후보 겹침) · `reserve/blind-g4.forbid.json` · `sets/g3/blind-g3.forbid.json` · `G4Forbid`(학습 몫) · 옛 평가 문단 색인 · scrub 이름 둘.
- **KLUE 원천 줄을 바꿨다.** 기본 풀(`g3/klue/klue-labeled-train.jsonl`, 356줄 · 210 문단)은 train-g3-ko KLUE 와 문단이 거의 같아 near-x 가 **166/220** 에서 멈췄다. 설계 4-1 의 「안 쓴 학습 쪽 묶음」인 `g3/klue/answered-rw.jsonl` 학습 쪽 4,334줄을 `public_pick.prepare(train)`(평서문 `claim_rewrite` · 평서문 검사 떨어진 짝 106 뺌)로 바꾼 **4,228줄 · 1,984 문단**을 `--pool klue=` 로 줬다(손 스크립트 `prep/klue_pool_g4c.py` → `prep/klue-train-pool.jsonl`). 답 있는 질문뿐이라 「답 없는 질문」은 0 이다. BoolQ · 위키는 기본 풀로 다 찼다.

| 원천 | train-g3-ko 줄 (금지 뒤) | 풀 줄 (금지 뒤) | b 주장 | c 짝 | near-x | near-x 중 train-g3-ko 문단 | 금지로 빠짐 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BoolQ | 733 | 1,017 | 350 | **220** | **390** | 390 | 풀 옛 g3 문단 1 |
| 위키 | 945 | 1,027 | 500 | **110** | **390** | 389 | 0 |
| KLUE | 308 | 4,226 | 187 | **110** | **220** | 186 | 풀 평가 문단 2 |
| **합** | | | | **440** | **1,000** | 965 | |

- `recheck` 0 · scrub 0(1,440줄) · 금지 걸림 0. 규칙 대조(손 검사) : 근거 묶음당 최대 2 · 주장 한 번 · near-x 주장 중 train-g3-ko b 주장 0 · c 짝 440 모두 짝 b 줄(`pair`)과 주장이 같고 원천도 같음 · 원천 섞임 0 · 위키 근거 500/500 「제목 : 문장」 · BM25 순위 10~60 · `split: train` · `origin: g4-train` · `set: train-g4-c`.
- 씨앗 20261402(도구 기본). `pick.jsonl` sha256 ff6df6ff… · `klue-train-pool.jsonl` 1e3694ec…

### 10-2. 판정

`label_batches.py make --mode train` · 100줄 묶음 · 함정 10% · 판정자 둘 · 씨앗 20261444. 자리 `<g4>/judge/train-g4-c/`(build 기본값). `JUDGE-README.txt` 는 `eval-ko/` 것에서 경로 · 이름과 cd 줄(학습 묶음은 Q1 을 못 고를 때만 cd)만 바꿨다.

| 자리 | 값 |
| --- | --- |
| 묶음 | b01~b16 100줄 (실제 1,440 + 함정 160) × j1 · j2 · 셋째 t01 10줄 (갈린 9 + 함정 1) × j3 |
| Opus 판정자 | **17개** (첫 두 판 16개 동시 — 한 개가 같은 판정자 칸의 묶음 2개 · 한 판 약 2분, 셋째 1개). 벽시계 판정 약 4분 |
| 판정 줄 (함정 포함) | **3,210** · 실제 쌍 판정 2,889 = 쌍당 2.01. 어림(1,440 × 2.3 ≈ 3,300)의 0.97배 |
| 함정 | 판 33개 모두 남김 · 313/321 (최저 .8) · 답 없음 0 |
| 첫 두 판 일치 | **.994** (BoolQ .998 · 위키 .984 · KLUE 1.0) · 갈림 0.6% (멈춤 선 15%) |

| 원천 · 꼴 | 후보 | c (일치 · 셋째) | d | cd | 버림 | c 남김률 |
| --- | --- | --- | --- | --- | --- | --- |
| BoolQ c 짝 | 220 | 220 | 0 | 0 | 0 | 1.0 |
| BoolQ near-x | 390 | 389 | 1 (셋째) | 0 | 0 | .997 |
| 위키 c 짝 | 110 | 110 | 0 | 0 | 0 | 1.0 |
| 위키 near-x | 390 | 384 (378 · 6) | 0 | 5 (일치 4 · cd 갈림 1) | 1 (셋 다 다름) | .985 |
| KLUE c 짝 | 110 | 110 | 0 | 0 | 0 | 1.0 |
| KLUE near-x | 220 | 219 | 1 | 0 | 0 | .995 |
| **합** | **1,440** | **1,432** | 2 | 5 | 1 | **.994** |

- 위키 cd 5 는 판정자들이 「주장에 주어 · 곳 이름이 없어 같은 대상인지 못 고름」이라 적은 꼴(넓이 · 인구 값만 있는 위키 문장)이다. 정함 규칙(4-2)대로 c 가 아니라 버렸다.
- **도구 함정 — `layer_g4c build` 가 train 판정의 `cd` 칸에서 멈춘다.** `label_batches decide --mode train` 은 일치한 cd 를 `status: agree · label: cd` 로 쓰는데, build 가 부르는 `assemble_g3.resolve` 는 KEEP 상태의 모르는 칸을 「평가 셋에 모르는 칸 'cd'」로 멈춘다. 저장소 코드는 안 고쳤다. git 밖 사본 `<g4>/judge/train-g4-c-nocd/decided.jsonl`(cd 5줄만 `status: drop` 으로 바꿈)을 `--dirs` 로 줘서 조립했다. cd 는 c 가 아니라 어차피 빠지는 줄이라 고른 줄은 같다(c 칸 목록 · 씨앗 섞기가 같음). 원본 `decided.jsonl` 은 그대로다.

### 10-3. 조립 (`layer_g4c.py build`)

| 원천 | c 짝 (목표) | near-x (목표) | train-g4-c | train-g4-c-h (c 짝 절반) |
| --- | --- | --- | --- | --- |
| BoolQ | 200 (200) | 350 (350) | 550 | 450 |
| 위키 | 100 (100) | 350 (350) | 450 | 400 |
| KLUE | 100 (100) | 200 (200) | 300 | 250 |
| **합** | **400** | **900** | **1,300** | **1,100** |

- 모자람 0 · `recheck` 0 · scrub 0(두 파일). 칸은 모두 c(`want` · `hint`) · 상태 agree 1,295 · third 5 · 근거 묶음당 최대 2 · 주장 겹침 0. -h 는 train-g4-c 의 부분집합이고 near-x 900 이 같다.
- 같은 입력으로 build 를 메모리에서 다시 돌려 두 파일 바이트가 같았다(씨앗 20261402).
- 출력은 도구 기본 자리 `~/.judgemodel/build/g4/train-g4-c/` : `cands.jsonl`(train-g4-c) sha256 **271546a1526422f8…** · `cands-h.jsonl`(train-g4-c-h) **191bfb2df63b734f…** · `summary.json`. `sets/g4/` 로 얼리는 것은 4e 몫이다.

| 4-4 비율 (한 바퀴 우리 층, replay 뺌) | K | **K2** | K2-h |
| --- | --- | --- | --- |
| 우리 층 | 7,045 | **8,345** | 8,145 |
| 그중 c | 1,437 (20.4%) | **2,737 (32.8%)** | 2,537 (31.1%) |
| 새 층 둘 (train-g3-ko + c 층) 의 c 몫 | 1 / 2,644 | **1,301 / 3,944 = 33.0%** | 1,101 / 3,744 = 29.4% |
| replay 몫 (2,965 / 바퀴) | 29.6% | 26.2% | 26.7% |

- 설계 4-4 어림(8,345 · 32.8% · 33%)과 같다.

### 10-4. 진단 확인 (4-5)

train-g4-c near-x c 에서 원천별 무작위 50(씨앗 20261445, 합 150)을 **K = `nli-studio-v6`(run_id g1-K-s5, ONNX sha256 597c80d1… 대조함)** 로 작업 PC 에서 ONNX 그대로 쟀다(`serve.OnnxNLI`, 서빙 venv, 25초). 8092 서버(v4)는 안 썼다. 재현은 4칸 argmax 가 c 인 몫이다. 라벨은 판정 뒤 확정 c. 설계 4-5 는 K-s1~5 다섯 판이지만 이번엔 출시 후보로 내보낸 s5 한 판만 쟀다(다섯 ckpt 는 학습 기계 에만 있다).

| 원천 | 쌍 | **c 재현** | 예측 a · b · c · d |
| --- | --- | --- | --- |
| BoolQ | 50 | **.64** | 4 · 1 · 32 · 13 |
| 위키 | 50 | **.66** | 0 · 2 · 33 · 15 |
| KLUE | 50 | **.64** | 2 · 11 · 32 · 5 |
| **합** | 150 | **.647** | 6 · 14 · 97 · 33 |

- **.80 아래 → 가설대로다**(.90 이상이면 멈춤). 놓친 c 는 주로 d(33)로 가고, KLUE 는 b(11)도 많다. 150쌍은 층에 그대로 있다.
- 쌍별 예측은 git 밖 `prep/diag-g4c.jsonl`(sha256 02575fb7…) · 요약 `prep/diag-g4c.summary.json`. 손 스크립트 `prep/diag_g4c.py`.

### 10-5. 명령

```
# 원천 줄 (KLUE 만 바꿈) · 뽑기
python -I ~/.judgemodel/build/g4/prep/klue_pool_g4c.py --jm-build <JudgeModel>/build      # → prep/klue-train-pool.jsonl
python -I layer_g4c.py pick --pool klue=~/.judgemodel/build/g4/prep/klue-train-pool.jsonl   # (첫 판을 날짜 꼬리로 돌려 두려 --force)
python -I scrub.py check --in ~/.judgemodel/build/g4/train-g4-c/pick.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
# 판정
python -I label_batches.py make --cands ~/.judgemodel/build/g4/train-g4-c/pick.jsonl --out ~/.judgemodel/build/g4/judge/train-g4-c --mode train \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261444
python -I label_batches.py merge  --out ~/.judgemodel/build/g4/judge/train-g4-c
python -I label_batches.py decide --out ~/.judgemodel/build/g4/judge/train-g4-c --mode train
python -I label_batches.py third  --out ~/.judgemodel/build/g4/judge/train-g4-c       # 셋째 판 뒤 merge · decide 한 번 더
# cd 줄을 버림으로 바꾼 git 밖 사본 → 조립
#   judge/train-g4-c-nocd/decided.jsonl = decided.jsonl 에서 label cd 인 줄만 status drop (손 한 줄)
python -I layer_g4c.py build --dirs ~/.judgemodel/build/g4/judge/train-g4-c-nocd
# 진단 확인 (서빙 venv — onnxruntime · tokenizers)
<venv-serve>/python -I ~/.judgemodel/build/g4/prep/diag_g4c.py --jm <JudgeModel>
```

## 11. 조립 · 정답 확인 · 얼림 (설계 1절 7단계 · 3-5 · 7절 4e, 2026-10-14)

**결론 :** 두 셋을 본 조립(draft 아님)했다. dev-cmp-g4 **2,343쌍** · blind-g4 **1,869쌍**이고 8-1 의 미리 조립과 바이트까지 같다. 겹침 8칸 0 · scrub 0 · `min_groups` 네 칸 다 넘음(check exit 0). 정답 확인 gold-g4 는 **60/60 = 1.0**(b 30/30 · c 30/30)으로 관문 .85 를 넘었다. 그래서 셋 둘과 학습 층 둘을 `~/.judgemodel/sets/g4/` 에 얼렸다. **blind-g4 쌍 본문은 이 뒤로 열지 않는다**(한 판 · 한 번 규칙).

### 11-1. 셋 수

| 셋 | 쌍 | g3 이음 · 새 | a · b · c · d | 버림 | 중복 뺌 | 대기 |
| --- | --- | --- | --- | --- | --- | --- |
| dev-cmp-g4 | **2,343** | 2,219 · 124 | 513 · 571 · 753 · 506 | 0 | 0 | 0 |
| blind-g4 | **1,869** | 0 · 1,869 | 782 · 629 · 285 · 173 | 4 (VitaminC) | 0 | 0 |

- blind-g4 는 예약 후보 1,873줄에서 판정 버림 4줄(VitaminC)을 뺀 수다. dev-cmp-g4 는 dev-cmp-g3 2,219줄 + 새 c 124줄이다.
- 4c 미리 조립(`sets/g4-draft/`)과 두 파일 sha256 이 같다 — 수가 다를 까닭이 없다. `g4-draft/` 는 그대로 뒀다.

| 셋 × lang (a · b · c · d) | en | ko | mix |
| --- | --- | --- | --- |
| dev-cmp-g4 | 188 · 276 · 278 · 226 | 308 · 284 · **471** · 278 | 17 · 11 · 4 · 2 |
| blind-g4 | 706 · 379 · 134 · 173 | 76 · 250 · 151 · 0 | — |

| 새 줄 원천 × 칸 (a · b · c · d) | VitaminC en | BoolQ | 위키 | KLUE |
| --- | --- | --- | --- | --- |
| dev-cmp-g4 | — | 0 · 0 · 57 · 0 | 0 · 0 · 24 · 0 | 0 · 0 · 43 · 0 |
| blind-g4 | 706 · 379 · 134 · 173 | 30 · 90 · 54 · 0 | 22 · 110 · 54 · 0 | 24 · 50 · 43 · 0 |

- dev-cmp-g4 ko c 는 347 → **471**(+124 = BoolQ 57 · 위키 24 · KLUE 43). 설계 3-4 어림(약 477)과 6 차이다.

### 11-2. 검사 (`check`, draft 아님)

| 검사 | 결과 |
| --- | --- |
| 셋 사이(dev-cmp-g4 · blind-g4 · dev-cal-g3 · blind-g3) 글 · 묶음 · 쌍 열쇠 · 출처 id | 0 · 0 · 0 · 0 |
| 학습 층 5,044줄(train-g4-c 1,300 + train-g4-c-h 1,100 + train-g3-ko 2,644) × 셋 같은 넷 | 0 · 0 · 0 · 0 |
| scrub | dev-cmp-g4 0 · blind-g4 0 |
| `min_groups` (blind-g4) | b@en **212**/200 · a **405**/380 · c **277**/260 · b@ko **250**/220 — 모자람 없음 |
| 금지 목록 대조 (손 스크립트, 묶음 · 출처 id · 줄 id · 쌍 열쇠) | 학습 층 셋 × blind-g4.forbid · blind-g3.forbid 0 · dev-cmp-g4 새 줄 × 둘 0 · blind-g4 × blind-g3.forbid 0 |

- 학습 층 둘(train-g4-c · -h)은 `--train` 에 경로를 더 주어 넣었다(기본값은 train-g4-c 와 train-g3-ko 뿐). -h 는 train-g4-c 의 부분집합이라 결과는 같다.
- 얼린 뒤 `sets/g4/` 의 학습 층 사본으로 `check` 를 한 번 더 돌려 같은 값(exit 0)을 봤다.

### 11-3. 정답 확인 (gold-g4, 설계 3-4 ⑧)

| 항목 | 값 |
| --- | --- |
| 뽑기 | blind-g4 ko b 250 · ko c 151 을 id 차례로 늘어놓고 씨앗 **20261451** 로 b 30 · c 30 (why=random). 원천 b : 위키 11 · BoolQ 13 · KLUE 6 / c : BoolQ 15 · 위키 7 · KLUE 8 |
| 묶음 | 신중 판(careful) · v6 기준 · 함정 g1 풀 · 씨앗 **20261452**. 4묶음(20 · 20 · 20 · 7) × 판정자 둘(Opus 2개, 셋 판정자와 다른 새 판정자) · 함정 7 |
| 함정 | 8판 모두 1.0 (b01 2/2 · b02 2/2 · b03 2/2 · b04 1/1) · 답 없음 0 |
| 정함 (`decide --mode eval`) | 합의 60 · 갈림 0 · 셋째 0 |
| 「낮음」 | 5 — **모두 함정 줄**. 실제 60쌍은 0 |
| **dev = gold** | **b 30/30 = 1.0 · c 30/30 = 1.0 · 합 60/60 = 1.0** (관문 .85) |
| 깨진 줄 (4b 「noise 칸 없음」) | **0** — 60쌍 모두 판정자 둘이 셋 칸과 같은 칸을 줬다 |

- 자리 : `~/.judgemodel/build/g4/gold-g4/` — `gold-cands.jsonl` · `batches/` · `map/` · `answers/` · `decided.jsonl` · `JUDGE-README.txt`(gold-g3 것에서 경로만 바꿈).
- 셋 줄에 앞뒤 문맥 칸이 없어 `before` · `after` 는 g3 와 같이 「없음」 으로 넣었다.
- 표본이 ko b · ko c 뿐이라 이 값은 「한국어 b · c 라벨의 정밀」이다. 다른 칸으로 빠진 b · c 는 못 본다.

### 11-4. 얼림 · 해시

| 파일 (`~/.judgemodel/sets/g4/`) | 쌍 | sha256 |
| --- | --- | --- |
| `dev-cmp-g4.jsonl` | 2,343 | `fa2f81cb4226475aebb825d61af8f36efc0be695ef9d32e78427d4ea0b1ae59b` |
| `blind-g4.jsonl` | 1,869 | `c6d796d16a7b240146497871684fad385fec5f120f84bc980aadf335b04fc8dc` |
| `train-g4-c.jsonl` (학습 층) | 1,300 | `271546a1526422f856fbcf18f9cef208b3ed8be2dce32bdfb8db8cbc8bf3b057` |
| `train-g4-c-h.jsonl` (학습 층) | 1,100 | `191bfb2df63b734fc869f194d7287884d85ea6ff19bda70b83ac1af38005650f` |

- 넷 다 `FROZEN-<이름>.sha256` 을 쓰고 `freeze.py check` ok. 학습 층 둘은 `build/g4/train-g4-c/cands.jsonl` · `cands-h.jsonl` 을 그대로 복사했다(10-3 해시와 같음).
- 금지 목록 `blind-g4.forbid.json`(sha256 `3a0eb0da…`)을 `build/g4/reserve/` 에서 `sets/g4/` 로 복사했다. 앞으로 모든 학습 · 합성 · 덧학습 금지 목록에 이 파일을 더한다.
- dev-cal 은 g3 그대로다(설계 9-2 ⑤) — `sets/g3/dev-cal-g3.jsonl` `7f9a2f1d…`.

### 11-5. 명령

```
python -I assemble_g4.py assemble --set dev-cmp-g4          # blind-g4 도 같다, 기본 --dest ~/.judgemodel/sets/g4
python -I assemble_g4.py check --train train-g4-c/cands.jsonl train-g4-c/cands-h.jsonl ~/.judgemodel/sets/g3/train-g3-ko.jsonl
#   정답 확인 : blind-g4 ko b 250 · ko c 151 → 각 무작위 30 (씨앗 20261451) → <g4>/gold-g4/gold-cands.jsonl (why=random, before · after 「없음」, 손 스크립트)
G=~/.judgemodel/build/g4/gold-g4
python -I label_batches.py make --cands $G/gold-cands.jsonl --out $G --mode eval --style careful \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261452
python -I label_batches.py merge --out $G
python -I label_batches.py decide --out $G --mode eval
python -I label_batches.py gold-score --dev-from $G/gold-cands.jsonl --gold $G
cp ~/.judgemodel/build/g4/train-g4-c/cands.jsonl   ~/.judgemodel/sets/g4/train-g4-c.jsonl
cp ~/.judgemodel/build/g4/train-g4-c/cands-h.jsonl ~/.judgemodel/sets/g4/train-g4-c-h.jsonl
cp ~/.judgemodel/build/g4/reserve/blind-g4.forbid.json ~/.judgemodel/sets/g4/
python -I assemble_g4.py freeze --set dev-cmp-g4            # blind-g4 도 같다
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g4/train-g4-c.jsonl   --out ~/.judgemodel/sets/g4/FROZEN-train-g4-c.sha256
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g4/train-g4-c-h.jsonl --out ~/.judgemodel/sets/g4/FROZEN-train-g4-c-h.sha256
```

## 12. 규약 protocol-g4 (설계 3-5 ⑤ · 6-4, 2026-10-14)

**결론 :** `measure/protocol-g4.draft.json` 의 빈 해시 넷을 얼린 해시로 채워 **`measure/protocol-g4.json`**(sha256 `a3572d2ae5a9cfd3c29102a2e640ad06f3d67f4a79b7967f39916e4f47632255`)을 썼다. 초안과 다른 곳은 해시 네 줄뿐이다. 초안은 그대로 뒀다. 첫 짝 비교 전에 커밋하고(사용자 승인), 그 뒤로는 안 고친다(hash_rule).

| 칸 | 값 |
| --- | --- |
| `sets.dev-cal-g3` | `7f9a2f1d…` — protocol-g3 · FROZEN-dev-cal-g3 와 같음 (도구가 대조) |
| `sets.dev-cmp-g4` | `fa2f81cb…` |
| `sets.blind-g4` | `c6d796d1…` |
| `train_sets.train-g4-c` · `train-g4-c-h` | `271546a1…` · `191bfb2d…` (`train-g3-ko` 는 g3 해시 그대로) |

- 검증(읽기만) : `compare.read_protocol` 로 읽힘 · TODO 0 · 빈 해시 0 · `gates.check_release_hashes` 가 blind-g4 해시는 해시 줄을 지나고 틀린 해시는 거부.
- 설계 6-4 대조 : `compare_on` dev-cmp-g4 · `calibrate_on` dev-cal-g3 · `release_pick` c_recall · ko · dev-cmp-g4 · `pair_gate` K → K2 · target `c_recall@ko` +.03 · 아래끝 > 0 · guards 9줄(g3 8줄 + `b_recall@ko` ≥ −.03) · alarms 4줄(g3 3줄 + `d_recall@ko` ≥ −.05) · `release_gate.set` blind-g4 · `min_groups` {b@en 200 · a 380 · c 260 · b@ko 220} · `group_cap` {vitaminc 2} · 나머지 release_gate 자 · `ref_model` · `bootstrap` · `seeds` · `majority` 는 g3 와 같음(다른 것은 note 글 둘뿐) — 모두 맞다.

```
python -I assemble_g4.py protocol --draft ../measure/protocol-g4.draft.json --out ../measure/protocol-g4.json
```

## 13. 학습 기계 6단계 머리 검사 · 기준선 (2026-10-14)

**결론 :** 학습 기계 사본을 맞추고(다른 파일 0 · 시험 589 통과 · skip 35) 셋 셋을 옮겨 해시를 맞춘 뒤, 기준선 K 를 dev-cmp-g4 로 다시 쟀다. **ko c 짝 반폭 0.0115 ≤ .04 → 판별 가능.** 이어 K2 학습 다섯을 떼어 띄웠다. 표는 `baseline-g4.md`.

| 검사 | 결과 |
| --- | --- |
| 사본 대조(5절 명령) | 처음엔 4줄 다름(`build/layer_g4c.py` · `measure/sets-g4.md` · `tests/test_layer_g4c.py` · `measure/protocol-g4.json` 없음) → 5절 tar 로 복사 → `같음` |
| 학습 기계 시험 | `python3 -I -m unittest discover -s tests` — 589 OK · skip 35 (pytest 는 학습 기계에 없다) |
| 셋 | `mem/g4/` 에 dev-cmp-g4 · train-g4-c · train-g4-c-h + FROZEN 셋 복사 → `freeze.py check` ok. dev-cal-g3 · train-g3-ko 는 그대로 ok. **blind-g4 는 복사 안 함**(7단계 몫) |
| 해시 | dev-cmp-g4 `fa2f81cb…` · train-g4-c `271546a1…` · train-g4-c-h `191bfb2d…` · dev-cal-g3 `7f9a2f1d…` · train-g3-ko `4a1025aa…` · 규약 `a3572d2a…` |
| llama · 다른 GPU 작업 | `inactive` · 꺼짐 |
| 메모리 · 디스크 | 가용 59GB · 스왑 80MB · 디스크 여유 399GB (K2 다섯 ckpt 약 45GB) |
| K-s1~s5 ckpt | `runs/g1-K-s{1..5}/ckpt-best` 있음 → K 재학습 안 함 |

| 기준선 K (dev-cmp-g4) | 씨앗 평균 [95%] | 짝 반폭 어림 |
| --- | --- | --- |
| **ko c** (471쌍) | **0.7563** [0.7166, 0.7939] | **0.0115** |
| ko b (284) | 0.8028 [0.7516, 0.8521] | 0.0138 |
| ko d (278) | 0.4899 [0.4290, 0.5458] | 0.0198 |
| 전체 c · b · d · acc3 | 0.7615 · 0.7867 · 0.4858 · 0.7786 | — · 0.0114 · 0.0163 · 0.0053 |

- K2 학습 : 학습 기계 `logs/g4-K2.sh` (= `g3-K.sh` 의 K 처방 + `--train mem/g4/train-g4-c.jsonl`, 씨앗 1..5 순차 → 다섯 다 `runs/g1-K2-sN/g4/` 에 dev-cal-g3 · dev-cmp-g4 · 잣대 · 곁 자 평가). 짝 비교는 안 한다. 표식 `runs/k2/DONE` · `runs/k2/FAILED`, 씨앗별 로그 `runs/k2/g1-K2-sN.log`. 걸음 2,828 (설계 어림 2,830).

```bash
# 학습 기계 (작업 뿌리 JT_ROOT)
logs/g4-K-base.sh                                   # 기준선 — runs/g4-K-base/baseline.{json,md}
nohup setsid logs/g4-K2.sh > runs/k2/k2.out 2>&1 < /dev/null &   # K2 다섯
```

## 14. 짝 비교 K → K2 (2026-10-14)

- **지킴 깨짐** — 목표 ko c +.117 [+.089, +.149] 는 넘었으나 지킴 b_recall@en · b_recall · d_recall 셋이 깨졌다(ko b 는 지킴). 표 · 명령은 비공개 작업 기록에 있다.

## 15. K2-h 학습 시작 (설계 6-3, 2026-10-14)

**결론 :** 14절 짝 비교에서 지킴이 깨져 미리 정한 대체 판 K2-h 를 띄웠다. K2 와 다른 것은 `--train mem/g4/train-g4-c.jsonl` → **`mem/g4/train-g4-c-h.jsonl`**(1,100 · `191bfb2d…` · freeze ok) 하나다. 처방 · 씨앗 1..5 순차 · g4 평가(dev-cal-g3 · dev-cmp-g4 · 잣대 · 곁 자 · calibration)는 K2 와 같다.

| 항목 | 값 |
| --- | --- |
| 스크립트 | 학습 기계 `logs/g4-K2h.sh` (`g4-K2.sh` 를 sed 로 옮김 — 다른 줄은 c 층 파일 · 판 이름 `g1-K2h-sN` · 표식 자리 `runs/k2h/` · stage 파일뿐) |
| 머리 검사 | llama inactive · 다른 GPU 작업 꺼짐 · jt- 컨테이너 없음 · 가용 59GB · 스왑 80MB · 디스크 여유 357GB |
| 시작 | 14:25:31 · 걸음 2,780 (설계 어림 2,780) · 첫 판 200걸음에서 가용 43GB · GTT 14.5GB · 스왑 +0 |
| 표식 | `runs/k2h/DONE` · `runs/k2h/FAILED` · 씨앗별 로그 `runs/k2h/g1-K2h-sN.log` · 전체 `runs/k2h/k2h.out` |
| K2 실측 (눈금) | 학습 판당 약 22분 · 평가 판당 2분 15초 · 합 2시간 2분 (12:14:51 → 14:17:02) |
| 예상 끝 | 약 16:25 |

```bash
# 학습 기계 (작업 뿌리 JT_ROOT)
nohup setsid logs/g4-K2h.sh > runs/k2h/k2h.out 2>&1 < /dev/null &
```

## 16. K3 d 층 뽑기 (K3 설계 1절 2단계 · 4-1 · 4-2 · 4-4, 2026-10-14)

**결론 :** `layer_g4d.py pick` 으로 d 후보 **1,150쌍**을 뽑았다(ko 750 · en 400). 꼴 다섯이 다 목표를 채웠고 두 언어의 겹침 구간 몫(25/25/30/20)도 다 찼다. `recheck` 0 · scrub 0. 판정 묶음(판정자 둘 몫)까지 만들었고 판정은 아직 안 했다.

| 꼴 | 풀 (0~.1 / .1~.2 / .2~.35 / .35~) | 뽑음 (같은 구간) | 합 / 목표 |
| --- | --- | --- | --- |
| klue-removed | 142 / 170 / 232 / 164 | 94 / 93 / 98 / 75 | 360 / 360 (d 짝 35 / 35) |
| wiki-other | 17,376 / 3,437 / 1,763 / 2,912 | 94 / 94 / 127 / 75 | 390 / 390 |
| vitc-same-page | 15,944 / 13,871 / 11,664 / 5,141 | 34 / 33 / 67 / 26 | 160 / 160 |
| vitc-near | 4,185 / 3,788 / 2,870 / 955 | 33 / 33 / 27 / 27 | 120 / 120 |
| vitc-same-doc | 2,109 / 3,508 / 8,160 / 19,634 | 33 / 34 / 26 / 27 | 120 / 120 |

- 구간 몫 : ko 188 / 187 / 225 / 150 · en 100 / 100 / 120 / 80. 넘침 판 앞 모자람 0.
- 원천에서 뺀 줄 : 위키 같은 고침 줄기 3,352쌍 · **위키 겹침 .7 이상 4,192쌍**(같은 문장의 다른 판, 리뷰 10-14) · KLUE 답 글이 근거에 남음 47 · 평서문 검사 탈락 53 · 평서문 짝 없는 removed 3,178.
- 금지에 걸린 줄 (꺼낼 때) : 옛 g1 묶음 15 · 평가 문단 8 · 옛 셋 글 4 · train-g4-c 쌍 열쇠 0.
- 판정 묶음 : `label_batches.py make --mode train` · 100줄 · 함정 10% · 씨앗 20261445 → **13묶음**(b01~b12 100줄 · b13 78줄) × j1 · j2 = 실제 1,150 + 함정 128. 안내 `JUDGE-README.txt` 는 c 층 것에서 경로 · 이름만 바꿨다.
- 판정이 `<g4>/train-g4-d/` 에 쌓이므로 조립은 `layer_g4d.py build --dirs ~/.judgemodel/build/g4/train-g4-d` 로 한다(기본 `--dirs` 는 `judge/train-g4-d`).
- sha256 : `pick.jsonl` `2483043a…` · `pick-summary.json` `0fa9e9c4…` · `map/map.jsonl` `3d868997…`.

```
python -I layer_g4d.py pick                          # → ~/.judgemodel/build/g4/train-g4-d/pick.jsonl · pick-summary.json
python -I scrub.py check --in ~/.judgemodel/build/g4/train-g4-d/pick.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
python -I label_batches.py make --cands ~/.judgemodel/build/g4/train-g4-d/pick.jsonl --out ~/.judgemodel/build/g4/train-g4-d --mode train \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261445
```

## 17. K3 d 층 판정 — 첫 두 판 일치 .899 로 멈춤 → (나) 재판정 (K3 설계 4-3 · 8-1, 2026-10-14)

**결론 :** d 후보 1,150쌍을 v6 두 판(Opus 판정자 8개, 묶음 3~4개씩)으로 판정했다. 함정은 26판 모두 남김(251/256, 최저 .8) · 답 없음 0. **첫 두 판 일치 .899(1,034/1,150)** 로 설계 4-3 문턱 .90 바로 아래라 **셋째 판정 · 조립 전에 멈췄다**(8-1). 갈림 116 중 **VitaminC c/d 갈림이 73** 이고 그중 70 은 j1 이 c 다 — 영문 근거가 「It」·「the film」처럼 이름 없이 말할 때 j1 판정자(b01~b07)는 「같은 대상인지 글로 모름 → c」, j2 는 「같은 자리면 같은 대상 → d」로 갈랐다. 판정 실수가 아니라 기준 v6 이 이 꼴을 정하지 않은 데서 온 갈림이다.

| 꼴 | 쌍 | 첫 두 판 일치 | 일치 d | 갈림 (그중 c/d) |
| --- | --- | --- | --- | --- |
| klue-removed | 360 | .989 | 333 | 4 (2) |
| wiki-other | 390 | .933 | 179 | 26 (10) |
| vitc-same-page | 160 | .806 | 91 | 31 (27) |
| vitc-near | 120 | .742 | 67 | 31 (29) |
| vitc-same-doc | 120 | .800 | 70 | 24 (17) |
| **합** | **1,150** | **.899** | **740** | **116** |

- 일치 줄의 칸 : d 740 · a 122 · b 104 · c 67 · cd 1. 묶음별 일치 .811~.986(b09~b13 은 VitaminC c/d 갈림 0~1).
- 판정 줄(함정 포함) 2,556 · 어림 2,600 의 0.98배.
- **조립 전 어림** : 일치 d 만으로 klue 333 / 250 · same-page 91 / 100 · near 67 / 75 · same-doc 70 / 75 는 80% 선을 넘는다. **wiki 179 / 250 은 80%(200) 아래**다 — 갈림 26 이 셋째에서 21 넘게 d 가 돼야 닿는다. 묶음당 2 상한 전 셈이다.
- 다음은 메인 결정 : (가) .899 를 받아 셋째로 진행 (나) 영문 대명사 근거 꼴을 기준에 적고 VitaminC 몫만 다시 판정 (다) 그 꼴을 셋째에 맡김. wiki 모자람은 어느 쪽이든 따로 정한다(8-1 : 후보 더 뽑아 그 몫만 판정).
- 자리 `~/.judgemodel/build/g4/train-g4-d/` : `votes.jsonl` sha256 `09c113c9…` · `decided.jsonl` `9b946125…`(셋째 전) · `merge-report.json` · `decide-summary.json`. `third` 는 아직 안 돌렸다(map 그대로).

```
python -I label_batches.py merge  --out ~/.judgemodel/build/g4/train-g4-d
python -I label_batches.py decide --out ~/.judgemodel/build/g4/train-g4-d --mode train
```

### 17-1. 메인 결정 (나) — 규칙 보충 · VitaminC 다시 판정 · 위키 추가 뽑기

- **규칙 보충** : `Docs/Guide/라벨기준support-v6.md` **3-2절** 「보충 2026-10-14 — 대상을 이름 없이 가리키는 근거」. 다른 대상이라는 단서(이름 · 시기 · 장소 어긋남)가 없으면 같은 대상으로 보고 a · b · d 를 가린다. c 는 그 단서가 있을 때만. 판 이름은 v6 그대로다. 문서 끝(4절 뒤)이 아니라 3절 안에 둔 까닭 : `label_batches.guide_sections` 가 1~3절만 떼어 묶음 txt 에 싣기 때문이다.
- **위키 추가 뽑기** : 손 스크립트 `prep/wiki_add_g4d.py`(sha256 `12dfe628…`). 처음 pick.jsonl 의 묶음(group) · 주장 · 쌍 열쇠를 위키 풀에서 미리 빼고(묶음 겹침 18,838 줄 뺌 → 풀 6,650), `layer_g4d.pick()` 을 targets `{wiki-other: 160, 나머지 0}` · 씨앗 **20261404** 로 그대로 불렀다. id 는 1001~1160(꼬리 쌍 해시 그대로) · `pick_add: wiki-add-1`. 구간 몫 40 / 40 / 48 / 32(같은 25/25/30/20) · 금지 걸림 0 · recheck 0 · scrub 0. 출력 `train-g4-d-add/pick.jsonl` sha256 `87496619…`. **처음 `pick.jsonl` 은 `2483043a…` 그대로다**(전후 대조).
- **다시 판정 묶음** : VitaminC 400(처음 pick 에서 꼴 vitc-*) + 위키 추가 160 = 560쌍 → `train-g4-d-re/cands-re.jsonl`(sha256 `6a044934…`) · `label_batches make --mode train` · 씨앗 20261446 · 7묶음(b01~b06 100줄 · b07 22줄) × 두 판 = 실제 560 + 함정 62. 새 Opus 판정자 6개(메인 이름 j3 · j4, 폴더 안 파일 이름은 도구 규칙대로 j1 · j2). 묶음 txt · `JUDGE-README.txt` 에 3-2절이 들었다.
- **옛 VitaminC 판정 보존** : `train-g4-d/answers-v1/` 에 26파일 통째 복사. `answers/` 는 그대로 뒀다 — 줄을 빼면 merge 의 「답 없음」 관문이 판을 버린다. 대신 처음 폴더 `decided.jsonl` 의 VitaminC 줄은 조립에서 뺀다(아래 `build_all_g4d.py`). 처음 폴더 셋째(`third`)는 VitaminC 줄을 `replaced-re` 로 막고 KLUE · 위키 갈림 30 만 보냈다.

| 다시 판정 (보충 3-2) | 쌍 | 첫 두 판 일치 | 갈림 (c/d) |
| --- | --- | --- | --- |
| vitc-same-page | 160 | .938 | 10 (6) |
| vitc-near | 120 | .933 | 8 (5) |
| vitc-same-doc | 120 | .967 | 4 (1) |
| **VitaminC 합** | **400** | **.945** (처음 .785) | 22 (12) |
| 위키 추가 | 160 | .919 | 13 (4) |

- 함정 : 다시 판정 14판 모두 남김 124/128(최저 .9) · 셋째 4/4. VitaminC 문턱 .90 을 넘어 진행했다.
- **층 전체 첫 두 판 일치(바뀐 뒤)** : KLUE 356/360 .989 · 위키 364/390 .933 · 위키 추가 147/160 .919 · VitaminC 378/400 .945 → **1,245/1,310 = .950**.
- 판정 줄(함정 포함) : 첫 판 2,556 + 셋째 33 + 다시 판정 1,244 + 셋째 39 = **3,872** — 어림 2,600 의 1.49배(버린 VitaminC 첫 판 800줄 포함). 1.5배 바로 아래.

## 18. K3 d 층 확정 — train-g4-d 750 (K3 설계 4-3 · 4-5, 2026-10-14)

**결론 :** 셋째 판정까지 끝내고 `layer_g4d.build()` 로 **train-g4-d 750쌍**을 조립했다. 꼴 다섯이 다 목표를 채웠다(모자람 0). recheck 0 · scrub 0 · 근거 묶음당 최대 2 · 주장 겹침 0. 얼리기는 4-5 몫이라 안 했다.

| 셋째 | 갈림 | 정함 (셋 중 둘) | 버림 (셋 다 다름) |
| --- | --- | --- | --- |
| 처음 폴더 t01 (KLUE · 위키) | 30 | 29 (d 4) | 1 |
| 다시 판정 t01 (VitaminC · 위키 추가) | 35 | 34 | 1 |

| 꼴 | d 확정 (일치 + 셋째) | 뽑음 / 목표 | 겹침 구간 (0~.1 / .1~.2 / .2~.35 / .35~) |
| --- | --- | --- | --- |
| klue-removed | 333 | **250 / 250** | 57 / 69 / 67 / 57 |
| wiki-other (처음 183 + 추가 69) | 252 | **250 / 250** | 97 / 73 / 66 / 14 |
| vitc-same-page | 121 | **100 / 100** | 23 / 20 / 48 / 9 |
| vitc-near | 96 | **75 / 75** | 22 / 24 / 16 / 13 |
| vitc-same-doc | 87 | **75 / 75** | 24 / 24 / 14 / 13 |
| **합** | **889** | **750 / 750** | |

- 후보 1,310 의 칸 : d 889 · a 187 · b 146 · c 85 · 버림 3(cd 1 포함, `_drop_boundary`). c 로 정한 줄은 c 층에 안 더한다(설계 4-3).
- 위키는 d 252 로 목표 250 을 겨우 넘었다 — 추가 160 중 d 는 69(.43)뿐이다. 남김률이 낮은 꼴이라 다음 판에서 위키 d 를 늘리려면 더 넉넉히 뽑는다.
- 줄 : 상태 agree 740 · third 10 · `label_guide: support-v6` · ko 500 · en 250 · 위키 추가 줄 69.
- **조립 방법** : `layer_g4d build` 는 pick.jsonl 하나만 받아서 손 스크립트 `prep/build_all_g4d.py`(sha256 `3194ce76…`)로 `layer_g4d.build()` 를 그대로 불렀다. 후보 = 처음 pick 1,150 + 추가 160, 판정 = 처음 `decided.jsonl` 에서 VitaminC 뺀 줄 + `train-g4-d-re/decided.jsonl`. 금지 recheck 는 `cmd_build` 와 같다. 같은 입력으로 메모리에서 다시 돌려 바이트가 같았다(씨앗 20261403 기본).
- 출력 `~/.judgemodel/build/g4/train-g4-d/` : **`cands.jsonl` sha256 `739072b92c9824ff…` (750줄)** · `summary.json` `c7d63304…`. 판정 : `train-g4-d/decided.jsonl` `0ef1c936…` · `train-g4-d-re/decided.jsonl` `ed288916…`.

```
# 위키 추가 뽑기 → train-g4-d-add/pick.jsonl
python -I ~/.judgemodel/build/g4/prep/wiki_add_g4d.py --jm-build <JudgeModel>/build          # --n 160 --seed 20261404
python -I scrub.py check --in ~/.judgemodel/build/g4/train-g4-d-add/pick.jsonl --names … --game-names …
# 다시 판정 (cands-re = 처음 pick 의 vitc-* 400 + 추가 160)
python -I label_batches.py make --cands ~/.judgemodel/build/g4/train-g4-d-re/cands-re.jsonl --out ~/.judgemodel/build/g4/train-g4-d-re --mode train \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261446
python -I label_batches.py merge|decide|third --out ~/.judgemodel/build/g4/train-g4-d-re (--mode train)      # 셋째 뒤 merge · decide 한 번 더
# 처음 폴더 셋째 — decided.jsonl 의 vitc 줄을 status replaced-re 로 바꾼 뒤 (손 한 줄)
python -I label_batches.py third --out ~/.judgemodel/build/g4/train-g4-d      # 뒤 merge · decide --mode train
# 조립
python -I ~/.judgemodel/build/g4/prep/build_all_g4d.py --jm-build <JudgeModel>/build
python -I scrub.py check --in ~/.judgemodel/build/g4/train-g4-d/cands.jsonl --names … --game-names …
```

## 19. K3 d 층 검사 · 얼림 · 규약 protocol-g5 (K3 설계 4-4 · 4-5 · 6-3, 2026-10-14)

**결론 :** train-g4-d 750쌍을 더한 `check` 가 겹침 8칸 0 · scrub 0 · `min_groups` 모자람 없음(exit 0)이다. train-g4-d 를 `~/.judgemodel/sets/g4/` 에 얼렸고 앞선 넷은 바이트 그대로다. `measure/protocol-g5.draft.json` 의 빈 자리(train-g4-d 해시 하나)를 채워 **`measure/protocol-g5.json`**(sha256 `6422e1e229d0039d7b4774619b57dfadce3a817a29c8fc3c1800b598318ccd4f`)을 썼다. 평가 셋이 안 바뀌어 정답 확인(gold)은 이번에 안 했다. protocol-g4.json 은 그대로다(`a3572d2a…`).

### 19-1. 검사

| 검사 | 결과 |
| --- | --- |
| `check` 기본값 — 학습 층 5,794줄(train-g3-ko 2,644 + train-g4-c 1,300 + -h 1,100 + train-g4-d 750) | 셋 사이 글 · 묶음 · 쌍 열쇠 · 출처 id 0 · 0 · 0 · 0 / 학습 층 × 셋(dev-cmp-g4 · blind-g4 · dev-cal-g3 · blind-g3) 0 · 0 · 0 · 0 |
| scrub | dev-cmp-g4 0 · blind-g4 0 · train-g4-d 0 (`scrub.py check`, 750줄) |
| `min_groups` (blind-g4) | b@en 212 · a 405 · c 277 · b@ko 250 — 모자람 없음 (11-2 와 같음) |
| train-g4-d × train-g4-c · -h · train-g3-ko 쌍 열쇠 · 줄 id (손 스크립트) | 0 · 0 · 0 (문단이 같은 것은 설계 4-4 대로 셈하지 않음) |
| train-g4-d × blind-g4.forbid · blind-g3.forbid (묶음 · 출처 id · 줄 id · 쌍 열쇠, 손 스크립트) | 0 · 0 |

- 얼린 뒤 `sets/g4/` 사본 넷 + train-g3-ko 로 `check` 를 한 번 더 돌려 같은 값(exit 0)을 봤다.

### 19-2. 얼림 · 해시

| 파일 (`~/.judgemodel/sets/g4/`) | 쌍 | sha256 | |
| --- | --- | --- | --- |
| `dev-cmp-g4.jsonl` | 2,343 | `fa2f81cb4226475aebb825d61af8f36efc0be695ef9d32e78427d4ea0b1ae59b` | 그대로 (check ok) |
| `blind-g4.jsonl` | 1,869 | `c6d796d16a7b240146497871684fad385fec5f120f84bc980aadf335b04fc8dc` | 그대로 (check ok) |
| `train-g4-c.jsonl` | 1,300 | `271546a1526422f856fbcf18f9cef208b3ed8be2dce32bdfb8db8cbc8bf3b057` | 그대로 (check ok) |
| `train-g4-c-h.jsonl` | 1,100 | `191bfb2df63b734fc869f194d7287884d85ea6ff19bda70b83ac1af38005650f` | 그대로 (check ok) |
| **`train-g4-d.jsonl`** | **750** (d 750 · ko 500 · en 250) | **`739072b92c9824ffab193c73528bcabcd8efd4fb89ec261bf4ae2dbd647ebfb9`** | 새로 얼림 |

- train-g4-d 는 `build/g4/train-g4-d/cands.jsonl` 을 그대로 복사했다(18절 해시와 같음).

### 19-3. protocol-g5.json

- 초안과 다른 곳은 `train_sets.train-g4-d` 한 줄뿐이다(diff 확인). 초안은 그대로 뒀다(`fca9f067…`).
- 검증(읽기만) : `compare.read_protocol` 로 읽힘 · TODO 0 · `gates.check_release_hashes` 가 blind-g4 해시는 해시 줄을 지나고 틀린 해시는 거부.
- 설계 6-3 대조 (protocol-g4 와 견줌) : `sets` · `compare_on` · `calibrate_on` · `release_pick` · `bootstrap` · `seeds` · `majority` · `release_gate` 전부 같음 · `train_sets` 는 g4 셋 그대로 + train-g4-d · `pair_gate` K → K3 · target · guards 9줄 같음 · alarms 6줄(g4 4줄 + `c_recall@ko` ≥ +.06 · `d_recall@en` ≥ −.05) · report_only 는 g4 것 다 + 새 4줄(K2 → K3 차 · 원천 × 꼴별 d 등) · verdicts 「지킴 깨짐」 = 접는다(대체 판 없음). 그 밖에 다른 칸은 name · design · note 글뿐 — 모두 맞다.
- 첫 짝 비교 전에 커밋(사용자 승인), 그 뒤로는 안 고친다(hash_rule).

### 19-4. 명령

```
python -I assemble_g4.py check                                   # 기본 --train = train-g4-c · cands-h · train-g4-d · train-g3-ko
python -I scrub.py check --in ~/.judgemodel/build/g4/train-g4-d/cands.jsonl --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
#   train-g4-d × train-g4-c 쌍 열쇠 · × blind-g4/g3.forbid : 손 스크립트 (make_sets.sha_key)
cp ~/.judgemodel/build/g4/train-g4-d/cands.jsonl ~/.judgemodel/sets/g4/train-g4-d.jsonl
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g4/train-g4-d.jsonl --out ~/.judgemodel/sets/g4/FROZEN-train-g4-d.sha256
python -I ../train/freeze.py check --pairs ~/.judgemodel/sets/g4/<이름>.jsonl --frozen ~/.judgemodel/sets/g4/FROZEN-<이름>.sha256   # 다섯 다
python -I assemble_g4.py protocol --draft ../measure/protocol-g5.draft.json --out ../measure/protocol-g5.json
```

## 20. 학습 기계 동기화 — K3 (K3 설계 1절 5단계, 2026-10-14)

**결론 :** 학습 기계 사본을 다시 맞췄다(다른 파일 0 · 시험 623 OK · skip 35). train-g4-d 를 `mem/g4/` 로 옮겨 `freeze.py check` 여섯이 다 ok 다. blind-g4 는 여전히 안 옮겼다(7단계 몫).

| 검사 | 결과 |
| --- | --- |
| 사본 대조(5절 명령) | 처음 14파일 다름 — 새 파일 `build/layer_g4d.py` · `measure/protocol-g5{,.draft}.json` · `tests/test_layer_g4d.py`, 바뀐 파일 `build/{README.md,assemble_g4.py}` · `measure/{README.md,prereg.py,sets-g4.md}` · `tests/{test_assemble_g4,test_compare,test_four_labels}.py` → 5절 tar 로 복사 → `같음` |
| 학습 기계 시험 | `python3 -I -m unittest discover -s tests` — 623 OK · skip 35 |
| 셋 | `mem/g4/train-g4-d.jsonl` + FROZEN 복사 · sha `739072b9…` · `freeze.py check` ok : dev-cmp-g4 · train-g4-c · train-g4-c-h · train-g4-d · dev-cal-g3 · train-g3-ko |
| 규약 | `src/measure/protocol-g5.json` `6422e1e2…` (19-3 과 같음) |
| llama · 다른 GPU 작업 · 컨테이너 | inactive · 꺼짐 · jt- 없음 |
| 메모리 · 디스크 | 가용 59GB · 스왑 80MB · 디스크 여유 315GB (K2-h 판 하나 8.4GB) |

## 21. K3 진단 확인 · 학습 시작 (K3 설계 6-1 · 6-2, 2026-10-14)

**결론 :** 진단 확인에서 K2-s1 의 d 재현이 **.167**(멈춤 문턱 .70 아래)이라 d 층이 무너진 꼴을 맞게 겨눴다. c 파일은 K2-h 짝 비교 ⑧ 대로 **train-g4-c-h** 로 정해(비공개 작업 기록 2절) K3 다섯을 띄웠다. 규약 protocol-g5 는 안 고쳤다.

### 21-1. 진단 확인 (train-g4-d 꼴 다섯 × 무작위 30, 씨앗 20261471, 층에서 안 뺌)

| 꼴 | 쌍 | K-s1 d 재현 | K-s1 c 몫 | K2-s1 d 재현 | K2-s1 c 몫 | K2-s1 a · b 몫 |
| --- | --- | --- | --- | --- | --- | --- |
| klue-removed | 30 | .733 | .033 | .300 | .500 | .000 · .200 |
| vitc-near | 30 | .567 | .267 | .033 | .767 | .000 · .200 |
| vitc-same-doc | 30 | .567 | .200 | .233 | .600 | .067 · .100 |
| vitc-same-page | 30 | .567 | .200 | .167 | .800 | .000 · .033 |
| wiki-other | 30 | .533 | .367 | .100 | .900 | .000 · .000 |
| **합** | **150** | **.593** | .213 | **.167** | **.713** | .013 · .107 |

- 기대(K .4~.6 · K2 < .5)와 맞다. K2 가 잃은 d 는 거의 다 c 로 갔다(c 몫 .21 → .71).
- 자리 : 학습 기계 `runs/g4-K3-diag/` (diag150.jsonl sha `101500e6…` · `eval-{K,K2}-s1/` · diag.md · diag_tab.py). 평가 인자는 dev-cmp 평가와 같고 `--protocol src/measure/protocol-g5.json`.

### 21-2. K3 학습

| 항목 | 값 |
| --- | --- |
| 스크립트 | 학습 기계 `logs/g4-K3.sh` (`g4-K2h.sh` 를 sed 로 옮김 — 더한 것은 `--train mem/g4/train-g4-d.jsonl` · 규약 `protocol-g5.json`(sha 박음) · 머리 검사 파일 · freeze 줄에 train-g4-d, 그 밖은 판 이름 `g1-K3-sN` · 표식 자리 `runs/k3/` · stage 파일) |
| `--train` | train-r2 · train-g1-arch · train-g3-ko · **train-g4-c-h** · **train-g4-d** (+ replay 같음) |
| 머리 검사 | llama inactive · 다른 GPU 작업 꺼짐 · 가용 59GB · 스왑 80MB · freeze 여섯 ok |
| 시작 | 16:33:08 · 걸음 **2,968** (설계 어림 3,015 는 c 파일 train-g4-c 기준. c-h 라 200쌍 적음) · 200걸음에서 가용 43GB · GTT 14.2GB · 스왑 +0 |
| 표식 | `runs/k3/DONE` · `runs/k3/FAILED` · 씨앗별 로그 `runs/k3/g1-K3-sN.log` · 전체 `runs/k3/k3.out` |
| 예상 끝 | 약 18:40 (K2-h 실측 2시간 × 걸음 1.07) |

```bash
# 학습 기계 (작업 뿌리 JT_ROOT)
nohup setsid logs/g4-K3.sh > runs/k3/k3.out 2>&1 < /dev/null &
```

## 22. 짝 비교 K → K3 (K3 설계 9절, 2026-10-14)

**결론 :** **지킴 깨짐 → 접음.** 목표 `c_recall@ko` +.119 [+.091, +.150] 는 넘었고 d 는 +.208 로 되살아났지만, en b −.209 · 전체 b −.112 · ko b −.022(아래끝 −.042)가 깨졌다. v1.2 멈춤 · 8092 v1.1.0 유지 · prereg · blind-g4 안 함. 경보 여섯은 모두 조용. 결과 문서는 공개하지 않는다.

| 항목 | 값 |
| --- | --- |
| 코드 · 규약 · 셋 | compare.py `b55ba7a8…` · protocol-g5.json `6422e1e2…` · dev-cmp-g4 `fa2f81cb…` (저장소 판과 같음) |
| 학습 머리 | K3 다섯 2,968걸음 끝 · exit=0 done · Traceback · nan · FAILED 0 · `runs/k3/DONE` 18:38:16 |
| 첫 실행 | exit 2 거부 — K · K2 summary 의 protocol_sha256 이 g4(`a3572d2a…`) |
| 다시 찍기 | `runs/g4-K3-eval/restamp/g1-{K,K2}-sN/g4/` — pred · calibration · eval-side 는 원본 링크, summary 의 protocol_sha256 만 g5 로(옛 값 · 까닭 칸 덧붙임). 원본 · src 안 건드림 · GPU 안 씀 |
| 맞춤 확인 | 다시 찍은 K → K2 (`sanity-K-K2`) 가 14절 · K2 문서 숫자와 같음 |
| 출력 sha | compare.json `8642052a695465b29795688d1265a9619eb82e1b98b4bb5b8f99cdf99e789c3b` · compare.md `46866d3d…` · compare-K2.json `44773f8af372c6285db0c7b17fedf2d226fcd246bbb3d8fcad61b74fe0c824f4` · compare-K2.md `687113ab…` · sanity-K-K2.json `fd1a3db6…` |
| 자리 | 학습 기계 `runs/g4-K3-eval/` · 작업 PC `~/.judgemodel/runs/g4-K3/eval/` |

```bash
# 학습 기계 (작업 뿌리 JT_ROOT), R=runs/g4-K3-eval/restamp
python3 -I src/measure/compare.py --protocol src/measure/protocol-g5.json --set mem/g4/dev-cmp-g4.jsonl \
  --base K=$R/g1-K-s1/g4/eval-dev-cmp-g4,…(s5) --new K3=runs/g1-K3-s1/g4/eval-dev-cmp-g4,…(s5) \
  --by domain,lang --out runs/g4-K3-eval/compare.json --md runs/g4-K3-eval/compare.md
# 보고만 --base K2=$R/g1-K2-s* → compare-K2 · 맞춤 확인 --base K=$R --new K2=$R → sanity-K-K2
```

## 23. K4 원천 균형 층 train-g4-bal 조립 · 검사 · 얼림 · protocol-g6 (원천 균형 판 설계 1절 · 4절 · 5-1 6단계 · 5-6, 2026-10-14)

**결론 :** train-g4-bal **2,914쌍**(a 384 · b 409 · c 1,332 · d 789 · ko 1,608 · en 1,304 · mix 2)을 조립했다. 목표 3,222 의 90%다. 빈칸 검사 K 처방 + 새 층 `zero` **0** · `thin` 1(mit-cs b 3) · 겹침 8칸 0 · recheck 0 · scrub 0 · exit 0. `~/.judgemodel/sets/g4/` 에 얼렸고 앞선 다섯은 바이트 그대로다. `protocol-g6.json` 의 빈 자리(train-g4-bal 해시 하나)를 채웠다.

### 23-1. 원천 × 칸 채움

| 원천 | 칸 | 목표 | 채움 | 모자람 | 어디서 |
| --- | --- | --- | --- | --- | --- |
| vitaminc | a · b · c · d | 200 · 320 · 100 · 180 | 200 · 320 · 100 · 180 | 0 | v6 (c 는 v6 43 · v5 57) |
| klue-mrc | a · b · c · d | 90 · 37 · 132 · 243 | 90 · **32** · 132 · 243 | b 5 | a v6 7 · v5 83 / b v6 10 · v5 22 / c v6 10 · v5 122 / d v6 |
| proposals | a · b · c · d | 50 · 13 · 310 · 130 | 50 · **10** · 310 · 130 | b 3 | v5 (다시 판정한 줄만) |
| kobest-boolq | c · d | 300 · 67 | 300 · **59** | d 8 | c v6 / d v6 1 · 새 판정 58 |
| kowiki | c · d | 200 · 100 | 200 · 100 | 0 | v6 |
| klue-answered | c · d | 250 · 60 | 250 · 60 | 0 | c v6 / d v6 29 · 새 판정 31 |
| mit-cs | b · c · d | 60 · 40 · 40 | **3** · 40 · **17** | b 57 · d 23 | 새 판정 (d 는 v6 3 더함) |
| archive | a · b · d | 100 · 100 · 100 | **44** · **44** · **0** | 56 · 56 · 100 | 씨앗 짝(b · a) 44 |

- 새 판정에서 다른 칸으로 정해진 줄이 많다 : mit-rev(b 후보) 87 중 b 3 · a 84, mit-other(d) 75 중 c 59 · d 14, boolq-removed(d) 130 중 d 58 · a 31 · b 36. 칸 옮김(moved) 271줄.
- archive 씨앗 : 판정 216줄 중 짝 규칙(짝의 모든 줄 판정 = hint)으로 남은 줄 208. 그중 b 짝 a 가 있는 씨앗 44 만 짝째 들고, a 줄이 없는 b 씨앗 120 은 「한 줄 묶음」이라 빠진다. 씨앗 d 는 메인 결정으로 안 넣었다.
- 짝 묶음 234 통째 · 짝 일부 171줄 · 한 줄 2,275. 근거 묶음 × 칸 최대 2. 빠진 까닭 : K 처방과 같은 쌍 139 · 평가 글 32 · 평가 문단 12 · 쌍 열쇠 겹침 9 · 그 밖 금지 7.

### 23-2. 검사

| 검사 | 결과 |
| --- | --- |
| `cells` (train-r2 · train-g1-arch · train-g3-ko + train-g4-bal, dev-cmp-g4) | `zero` 0 · `thin_cells` mit-cs b (3 / dev 31) · `dev_missing` mit-cs c · d · seed a · b · d |
| `check --recipe` (같은 넷) | 학습 8,708줄(기본 `--train` 5,794 + train-g4-bal) · 겹침 8칸 0 · `min_groups` 모자람 없음 · exit 0 · `overlap_skipped_base` train-r2 · train-g1-arch |
| scrub (`scrub.py check`, 2,914줄) | 0 |
| 학습 입력 꼴 | `data_prep.read_pairs` 로 2,914줄 다 라벨 읽힘(evidence · claim · want 만 씀, `v5_want` · `relabel` · `keep` 같은 칸은 안 읽음) |

- 얼린 뒤 `sets/g4/` 사본으로 `cells` · `check` 를 한 번 더 돌려 같은 값(exit 0)을 봤다.

### 23-3. 얼림 · protocol-g6

| 파일 | 쌍 | sha256 |
| --- | --- | --- |
| `sets/g4/train-g4-bal.jsonl` (`build/g4/train-g4-bal/cands.jsonl` 그대로 복사) | 2,914 | `27729dad0fbeba1edc96d023226171b0fabe4f62e384f21a15967848b42794ca` |
| `measure/protocol-g6.json` | — | `6d5d1ccc0d3688140649ba5f07935e307cc141a3614ea53034b4964b17b68a7b` |

- `freeze.py check` 여섯(dev-cmp-g4 · blind-g4 · train-g4-c · -c-h · -d · -bal) ok.
- protocol-g6 : 초안과 다른 곳은 `train_sets.train-g4-bal` 한 줄뿐(diff 확인, 초안 `0120f7ca…` 그대로). `compare.read_protocol` 로 읽힘 · TODO 0. protocol-g5 와 견주면 sets · compare_on · calibrate_on · release_pick · bootstrap · seeds · majority · release_gate · pair_gate.target · guards(9줄) 같음, base K → new K4 · alarms 8줄 · `cell_check` 새 칸 — 설계 5-6 표와 맞다.

### 23-4. 학습 층 비율 (한 바퀴 · replay 뺌, 실물)

| | K | **K4** |
| --- | --- | --- |
| 쌍 | 7,045 | **9,959** |
| a · b · c · d · cd | 2,207 · 2,124 · 1,437 · 948 · 329 | **2,591 · 2,533 · 2,769 · 1,737 · 329** |
| replay 몫 (2,965 / 바퀴) | 29.6% | **22.9%** |

### 23-5. 명령

```
# 씨앗 판정 → keep 칸 (git 밖 손 스크립트 — seed_to_pairs.judge_rows 의 짝 규칙을 newjudge/decided.jsonl 로)
#   seed-pairs.jsonl + newjudge/decided.jsonl → <g4>/train-g4-bal/seed-judged.jsonl
B=~/.judgemodel/build; S=~/.judgemodel/sets; G=$B/g4
python -I layer_g4bal.py build --g4 $G \
  --v6 $S/g2/train-g2-b.jsonl $S/g4/train-g4-c.jsonl $S/g4/train-g4-d.jsonl \
       $G/train-g4-d-re@$G/train-g4-d-re/cands-re.jsonl $G/train-g4-d@$G/train-g4-d/pick.jsonl \
       $B/g2/train-g2-b@$B/g2/train-g2-b/cands.jsonl $G/judge/train-g4-c@$G/train-g4-c/pick.jsonl \
       $B/g3/boolq/boolq-labeled-train.jsonl $B/g3/kowiki/wiki-labeled-train.jsonl $B/g3/klue/klue-labeled-train.jsonl \
       $B/g3/mit/judged.jsonl $B/g3/seed/judged.jsonl \
  --v5-dirs $G/train-g4-bal/v5sample $G/train-g4-bal/v5re --dirs $G/train-g4-bal/newjudge \
  --seed-pairs $G/train-g4-bal/seed-judged.jsonl
R="$S/2026-10-10/build/train-r2.jsonl $S/g1/train-g1-arch.jsonl $S/g3/train-g3-ko.jsonl $S/g4/train-g4-bal.jsonl"
python -I assemble_g4.py cells --recipe $R --dev $S/g4/dev-cmp-g4.jsonl
python -I assemble_g4.py check --recipe $R
cp $G/train-g4-bal/cands.jsonl $S/g4/train-g4-bal.jsonl
python -I ../train/freeze.py write --pairs $S/g4/train-g4-bal.jsonl --out $S/g4/FROZEN-train-g4-bal.sha256
python -I assemble_g4.py protocol --draft ../measure/protocol-g6.draft.json --out ../measure/protocol-g6.json
```

- 경로는 Git Bash 에서 `C:/…` 꼴로 준다 — `~/…@~/…` 꼴은 `@` 뒤가 안 풀려 「없는 파일」로 멈춘다.
- 나3 은 `train-g4-d-re` 를 `train-g4-d` 보다 앞에 둔다 — 같은 id 는 앞 것이 남아(`same_id_earlier`) VitaminC 다시 판정 줄이 쓰인다.

- **K4 결과(10-15)** : 짝 비교 K → K4 지킴 깨짐(b_recall@ko −.019 [−.0394, 0]) → 접음 · 8092 v1.1.0 유지 (결과 문서는 공개하지 않는다).
