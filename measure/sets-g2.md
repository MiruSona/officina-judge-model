# g2 b 자 — 후보 · 예약 · 겹침 (v1.2 B 판 단계 A-1)

비공개 설계 3절 · 5-1 · 7절. 후보 · 예약 본문은 git 밖 `~/.judgemodel/build/g2/` 에만 있다. 여기에는 수와 명령만 적는다.
해시 · 얼림 · 판정은 아직 안 했다(A-2 몫).

## 0. 결론

- **평가 몫 예약은 b 기대 목표를 채웠다.** dev-cmp-g2 +181 · dev-cal-g2 +47 · blind-g2 +250 (기대 b = 후보 × 남김률).
- **한국어 b 가 크게 모자라다.** 목표 160 중 기대 37 만 찼다. 모자란 125 는 설계 3-1 대로 같은 셋의 VitaminC 몫으로 한 번 넘겼다(shift).
- 그래서 원천 비율이 52 : 35 : 13 이 아니라 dev-cmp 81 : 7 : 11 · blind 78 : 8 : 14 쯤이다(기대 b 기준).
- **학습 층의 한국어(D)는 거의 0 이다.** 평가 몫을 먼저 예약하니 한국어 고침이 평가에 다 들어갔다. 학습 층은 VitaminC 대조쌍 31,714 짝뿐이다.
- 겹침 검사 0 건 · scrub 0 건.

## 1. 원천별 후보

| 원천 | 후보 | 뽑은 법 | 비고 |
| --- | --- | --- | --- |
| `vitaminc-heldout` | 8,368 (b) | validation + test 의 실제 판 고침 REFUTES 전부 | 평가 몫 전용 |
| `vitaminc-train` | 64,000 줄 = 32,000 짝 (a · b) | train 실제 판 고침에서 같은 판 고침 · 같은 주장에 SUPPORTS 근거와 REFUTES 근거가 다 있는 짝 | 학습 층 전용. 한쪽만 남은 짝 189 버림 |
| `ko-revision` | 447 (수 바뀜 198) | 사설 저장소 A · 그 서브모듈 · 사설 저장소 B md 의 한 줄 고침(-1 +1, 닮음 ≥ .6, 15자 이상) | 이름 바꾸기 277 · 덜 닮음 140 · 짧음 110 · Memory 등 폴더 427 거름 |
| `ko-supersede` | 214 (짝 107 × 요약 · 제목) | 두 mem 저장소의 `superseded_by` 짝 | 잣대 기억에 걸린 짝 3 뺌 |
| `klue` | 22,406 (그중 평가 쓸 것 1,779) | g1 `klue-mrc` 후보를 그대로 옮김 | 평가는 heldout 의 「답 없는 질문」 줄만(다시 쓸 때 가짜 답을 넣어 b 로) |

- 한 줄 고침 짝 : 근거 = 고친 뒤 줄, 주장 = 고치기 전 줄. 표 줄은 칸을 「 · 」로 이었다.
- **이름 바꾸기 거르기** (`mine_archive.py` `is_rename`) : 수는 그대로이고 바뀐 낱말이 다 영문 식별자 꼴이면서, ① 경로 표지(`/` · `\` · 확장자)가 있거나 ② 같은 「옛 → 새」 바꿈이 저장소 전체에서 3번 넘게 나오면 뺀다. 가장 많이 걸린 바꿈은 툴 이름 · 하네스 경로 바꾸기였다.
- 진단의 1,637쌍보다 적다. Memory · Tools · `.claude` 폴더를 뺐고, 목록 거르기 대신 길이 · 코드 덩이 거르기만 했다.

## 2. 예약 (평가 몫 먼저)

남김률(어림) : VitaminC .52 · 한국어 고침 .16 · 덮은 결정 .30 · KLUE .22(g1 dev 의 「답 없는 질문」 → b 비율). 후보 = b 목표 ÷ 남김률이며, 목표의 1.6배 아래로는 안 내려간다.

| 셋 | 원천 | b 목표 | 후보 | 묶음 | 기대 b |
| --- | --- | --- | --- | --- | --- |
| dev-cmp-g2 | 한국어 고침 | 60 | 84 | 25 | 13.4 |
| | KLUE | 20 | 91 | 91 | 20.0 |
| | VitaminC | 100 + 넘김 47 | 284 | 103 | 147.7 |
| dev-cal-g2 | 한국어 고침 | 15 | 21 | 8 | 3.4 |
| | KLUE | 5 | 23 | 23 | 5.1 |
| | VitaminC | 25 + 넘김 12 | 73 | 25 | 38.0 |
| blind-g2 | 한국어 고침 | 85 | 120 | 33 | 19.2 |
| | KLUE | 35 | 160 | 157 | 35.2 |
| | VitaminC | 130 + 넘김 66 | 377 | 135 | 196.0 |
| **합** | | **475** | **1,233** | | **478** |

- 덮은 결정(`ko-supersede`)은 평가 몫에 하나도 못 들어갔다. 기억 204줄이 옛 셋에 이미 쓰였고, 나머지도 옛 셋 문장과 같거나 g1 평가 묶음에 걸렸다.
- 한국어는 셋마다 「채움률이 가장 낮은 셋이 다음 묶음을 가져가는」 차례로 나눴다. 모자라도 dev-cmp : dev-cal : blind 비율이 목표 비율(60 : 15 : 85)대로 남는다.
- 묶음은 통째로 한 셋에 간다(문서 · 위키 문서 · 기억 덩이). 한 묶음에서 고르는 줄 상한은 VitaminC 4 · KLUE 3 · 한국어 고침 12다.
- 출처 id(커밋 + 바뀐 낱말 · 기억 id · 판 고침 id)와 (근거, 주장) 글도 한 셋만 가진다. 한 커밋이 두 문서에서 같은 값을 고친 쌍이 dev 와 blind 로 갈라지지 않게 하려는 것이다.

학습 층 (예약 밖 · 금지 밖)

| 원천 | 줄 | b hint | 묶음 |
| --- | --- | --- | --- |
| VitaminC 대조쌍 | 63,428 (31,714 짝) | 31,714 | 11,231 |
| 한국어 고침 | 1 | 1 | 1 |
| 덮은 결정 | 0 | 0 | 0 |

- 설계 5-1 은 대조쌍 약 650 짝 · 한국어 약 700 을 기대했다. 대조쌍 650 짝 고르기(한정어 · 수 · 서수 주장 반 넘게)는 B+D 단계 몫이다. 한국어 D 는 원천이 없어 메인 판단이 필요하다.

## 3. 빼는 것 (금지 목록)

| 무엇 | 평가 몫 | 학습 층 |
| --- | --- | --- |
| 옛 셋과 같은 (근거, 주장) — `make_sets.OLD_SETS` + g1 셋 넷 + g1 학습 셋 셋 (train-r2 · train-g1-II · train-g1-arch · train-g1-syn 포함) | 뺌 | 뺌 |
| 근거나 주장 한쪽이 옛 셋 문장과 같음 | 뺌 | — |
| 옛 셋에 쓴 기억 id | 뺌 | 뺌 |
| g1 평가 셋(dev-cal · dev-cmp · blind-g1 · ood-g1)에 든 묶음 | 뺌 | 뺌 |
| g1 학습 셋에 든 묶음 | 문장으로만 가름 (위 둘째 줄) | — |
| use1 얼린 쌍 열쇠 (`use1/EXCLUDE.sha256list`, 2,439) · use1 셋 문장 | 뺌 | 뺌 |
| 잣대 3쌍의 기억 6개 (`build/g2/forbid-ruler-ids.txt`) | 뺌 | 뺌 |
| g1 KLUE 다시쓰기에 쓴 줄 | 뺌 | — |
| VitaminC train 에도 나오는 위키 문서 | 뺌 | — |
| g2 평가 몫의 묶음 · 출처 id · 글 | — | 뺌 (대조쌍은 짝째) |

- **g1 학습 묶음을 평가에서 문서째 막지 않은 까닭** : 한국어 문서 거의 모두가 train-g1-arch 묶음이라 막으면 한국어 평가 b 가 0 이 된다. 대신 문장이 같으면 뺀다. g1 예약(`reserve.json` · `reserve-archive.json`)도 안 쓴다 — 쓰지 않은 묶음까지 넓게 잡혀 있어서다(g1 아카이브 학습 후보의 `--reserve-used-only` 와 같은 생각).

평가 몫에서 빠진 까닭별 수 (한 줄이 까닭 여럿을 가질 수 있다)

| 원천 | g1 평가 묶음 | 옛 셋 글 | 옛 셋 문장 | 옛 기억 | use1 문장 | 그 밖 |
| --- | --- | --- | --- | --- | --- | --- |
| VitaminC heldout | 458 | 144 | 135 | — | — | — |
| 한국어 고침 | 156 | — | 5 | 9 | — | — |
| 덮은 결정 | 157 | 20 | 116 | 204 | 26 | — |
| KLUE | 213 | — | 178 | — | — | g1 다시쓰기 272 |

## 4. 겹침 · 누출 검사

| 검사 | 결과 |
| --- | --- |
| `check-g2` — 셋 사이(묶음 · 글 · 출처 id) · 평가 × 학습 층 · 둘 × 옛 셋(글 · 문장 · 기억 id · g1 묶음 · use1 · 잣대) | **20칸 모두 0** (exit 0) |
| g1 꼴 `check` — 새 평가 후보 셋 + 학습 층 × g1 예약 · 옛 셋 | group 0 · text 0 · text_old 0 · mem_id_old 0 (exit 0) |
| `scrub.py check` — 셋 파일 셋 + 학습 층 | 0 건 (넷 다 exit 0) |

## 5. 명령

`build/` 에서 돈다. `<repo-a>` · `<repo-b>` · `<store>` 자리는 자기 저장소 · 기억 저장소 경로로 채운다. `revision` · `supersede` 는 사설 메모 저장소용 선택 기능이다.

```
python -I mine_public.py vitaminc-heldout --raw ~/.judgemodel/raw/2026-10-11/vitaminc --out ~/.judgemodel/build/g2
python -I mine_public.py vitaminc-train   --raw ~/.judgemodel/raw/2026-10-11/vitaminc --out ~/.judgemodel/build/g2
python -I mine_archive.py revision --repo <repo-a>=<저장소 A> --repo <repo-b>=<저장소 B> … \
    --out ~/.judgemodel/build/g2 --names ~/.judgemodel/build/g1/names.txt --game-names ~/.judgemodel/build/g1/game-names.txt
python -I mine_archive.py supersede --store <repo-a>=<store A> --store <repo-b>=<store B> \
    --forbid ~/.judgemodel/build/g2/forbid-ruler-ids.txt --out ~/.judgemodel/build/g2 --names … --game-names …
cp ~/.judgemodel/build/g1/klue-mrc/{candidates.jsonl,groups.json} ~/.judgemodel/build/g2/klue/
python -I make_sets.py reserve-g2 --build ~/.judgemodel/build/g2      # → reserve/reserve-g2.json · <셋>-b-cands.jsonl · train-layer-cands.jsonl
python -I make_sets.py check-g2   --build ~/.judgemodel/build/g2
python -I scrub.py check --in ~/.judgemodel/build/g2/reserve/<파일> --names … --game-names …
```

- 씨앗 : 예약 20261201. 같은 입력이면 같은 예약이 나온다.
- 다음(A-2) : 예약 후보에 a · c · d 짝을 붙일지 정하고 → v6 판정 → 조립 → 해시 · `protocol-g2.json`.

## 6. A-2 결과 — dev-cal-g2 · dev-cmp-g2 판정 · 조립 · 얼림 (2026-10-13)

결론 : **두 셋을 얼렸다. b 목표는 넘겼지만 한국어 b 는 목표에 크게 못 미친다.** dev-cmp-g2 b 370 (목표 약 320) · dev-cal-g2 b 125 (약 100). 새 b 의 원천은 VitaminC 가 많고, KLUE 「답 없는 질문」은 거의 다 d 로 갔다(b 1+1).
정답 확인 60쌍 일치 1.00 (관문 .85). 겹침 · scrub 0 건.

### 6-1. 셋 구성

| 셋 | 쌍 | a | b | c | d | 해시 (sha256) |
| --- | --- | --- | --- | --- | --- | --- |
| dev-cal-g2 | 566 | 118 | 125 | 201 | 122 | `7f6f78064d2f1c1140526a0d97f68225e9d0633cb0b9624d745f79bac34c5713` |
| dev-cmp-g2 | 1,880 | 412 | 370 | 629 | 469 | `9571f4d9f849124b473615d910044d27c2de933eb0cf0f0c59b074c8590cd532` |

- 셋 = g1 옛 쌍 + 새 쌍. 줄의 `g2_label` 칸이 라벨 출처를 적는다 : `v6-new`(새 쌍) · `v6-relabel`(규칙에 걸려 v6 로 다시 판정, `v5_want` · `rule` 남김) · `v5-carry`(규칙에 안 걸려 v5 그대로, `label_guide: support-v5` 그대로).
- 수 : dev-cal-g2 = carry 353 · relabel 104 · new 109 / dev-cmp-g2 = carry 1,173 · relabel 275 · new 432. 옛 쌍 1개(dev-cmp, 셋 다 다름)는 버렸다.
- `FROZEN-<셋>.sha256` 은 `~/.judgemodel/sets/g2/` 에 셋 파일과 같이 있다.

새 쌍만 본 칸 · 원천별 b (목표는 2절 예약 표의 b 목표 · 기대 b)

| 셋 | 원천 | 판정 쌍 | 새 b | 남김률 (어림 → 실측, 예약 후보 대비) | b 목표 / 기대 |
| --- | --- | --- | --- | --- | --- |
| dev-cmp-g2 | VitaminC | 282 | 204 | .52 → .72 | 147 / 147.7 |
| | 한국어 고침 | 84 | 29 | .16 → .35 | 60 / 13.4 |
| | KLUE | 66 (다시 쓰기 skip 25 뺌) | 1 | .22 → .01 | 20 / 20.0 |
| dev-cal-g2 | VitaminC | 73 | 60 | .52 → .82 | 37 / 38.0 |
| | 한국어 고침 | 21 | 13 | .16 → .62 | 15 / 3.4 |
| | KLUE | 15 (skip 8 뺌) | 1 | .22 → .04 | 5 / 5.1 |

- 새 쌍 칸 : dev-cmp a 96 · b 234 · c 5 · d 97 / dev-cal a 12 · b 74 · c 5 · d 18.
- 셋 전체 b 의 원천 : dev-cmp-g2 VitaminC 267 (72%) · mem 51 · 한국어 고침 32 · KLUE 19 · 제안서 1 → 한국어 b 102 (28%). 설계 목표 52 : 35 : 13 보다 VitaminC 로 기운다.
- **KLUE 가 b 를 거의 못 냈다.** 가짜 답 넣기(지문에 없는 값)는 v6 에서 「근거가 말하지 않음」 → d 가 된다. g1 의 .22 는 「전제 어긋남」 꼴에서 나왔는데, 다시 쓰기 규칙이 그 꼴을 skip 으로 뺀다(114 중 22). 한국어 b 를 늘리려면 KLUE 다시 쓰기 규칙을 바꾸는 판단이 필요하다.

### 6-2. 판정 품질

| 폴더 | 쌍 | 판 (남김/전부) | 함정 정답 | κ 전체 · c/d (n) | 정함 |
| --- | --- | --- | --- | --- | --- |
| relabel-rest (옛 쌍 180) | 180 | 5 / 6 | 38/41 | .934 · .832 (47) | 합의 172 · 셋째 7 · 버림 1 |
| dev-cal-g2 | 109 | 5 / 5 | 22/25 | .769 · **.588** (21) | 합의 96 · 셋째 13 |
| dev-cmp-g2 | 434 | 11 / 11 | 99/99 | .890 · .752 (95) | 합의 405 · 셋째 27 · 버림 2 |
| 새 쌍 합 (dev-cal + dev-cmp) | 543 | | | .870 · .700 (116) | |

- relabel-rest 셋째 판(t01-j3)은 함정 0/1 로 버리고 한 번 다시 돌렸다(r2, 1/1).
- dev-cal-g2 c/d κ .588 은 0.60 아래다. c/d 쌍이 21 뿐이라 흔들림이 크고, 두 셋을 합치면 .700 이다. 기준 글은 v6 로 정해져 있어 고치지 않았다(메인 판단 몫).
- 원천별 첫 두 판 일치 : dev-cmp VitaminC .926 · 한국어 고침 .964 · KLUE .924 / dev-cal .89 · .905 · .80.

정답 확인 (gold-g2, 설계 3-3)

| 항목 | 값 |
| --- | --- |
| 뽑기 | 새 쌍 중 b 로 정해진 308 에서 무작위 60 (씨앗 20261224) — VitaminC 50 · 한국어 고침 10 · KLUE 0 |
| 판정 | 신중 판정자(`--style careful`, support-v6) 20쌍 묶음 4 × 둘 |
| 함정 | b01 두 판이 같은 함정 t0007(풀 라벨 a)을 둘 다 d 로 봐 관문에서만 뺐다(`merge --exempt-trap t0007`). 나머지 12/12 |
| 정함 | 합의 60 · 모두 b |
| **dev = gold (random 층)** | **60/60 = 1.00** (관문 .85) |

- 표본이 b 쌍뿐이라 이 값은 「b 라벨의 정밀」이다. b 를 놓친 쌍(a · d 로 간 b)은 이 확인이 못 본다.

### 6-3. 옛 쌍 v5 → v6 옮김 (규칙에 걸린 380, C 단계 200 + A-2 180)

| v5 \ v6 | a | b | c | d | 버림 | 합 |
| --- | --- | --- | --- | --- | --- | --- |
| a | 182 | 1 | 0 | 4 | 0 | 187 |
| b | 1 | 87 | 1 | 6 | 0 | 95 |
| c | 0 | 4 | 13 | 3 | 1 | 21 |
| d | 3 | 1 | 1 | 72 | 0 | 77 |

- 380 중 바뀐 것 25 (6.6%) · 버림 1. 걸린 규칙 : ① 137 · ② 17 · ③ 220 · ④ 6 (`rule_filter.py pick`, cap 없음).
- 셋별 : dev-cal 104 (C 52 · A-2 52) · dev-cmp 276 (C 148 · A-2 128).

### 6-4. 검사

| 검사 | 결과 |
| --- | --- |
| `check-g2` (예약 후보 20칸) | 0 (exit 0) |
| g1 꼴 `check --exclude-relabel` — 새 쌍 541 (다시 쓴 KLUE 주장 포함) × g1 예약 · 옛 셋 | group 0 · text 0 · text_old 0 · mem_id_old 0 (exit 0) |
| 조립 — 셋 안 같은 id · 같은 (근거, 주장) | 0 |
| `scrub.py check` — 두 셋 파일 · 두 후보 파일 | 0 건 (넷 다 exit 0) |

### 6-5. 명령

`build/` 에서 돈다. 묶음 · 답 · 셋 본문은 git 밖이다.

```
# 옛 쌍 380 다시 뽑기 → C 의 200 을 뺀 180
python -I rule_filter.py pick --sets ~/.judgemodel/sets/g1/dev-cal.jsonl ~/.judgemodel/sets/g1/dev-cmp.jsonl --out <g2>/eval-public/rule-picked-380.jsonl
# KLUE 다시 쓰기 (Opus, g1 rewrite-reserved 와 같은 규칙) → 붙이기
python -I make_sets.py attach-rewrite --cands <g2>/reserve/<셋>-b-cands.jsonl --rewrite <g2>/klue-rewrite/rewrite-<셋>.jsonl --out <g2>/klue-rewrite/<셋>-klue.jsonl
# 묶음 (씨앗 relabel-rest 20261221 · dev-cal-g2 20261222 · dev-cmp-g2 20261223 · gold 20261224)
python -I label_batches.py make --cands <g2>/eval-public/<셋>-cands.jsonl --out <g2>/eval-public/<셋> --mode eval \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed <씨앗>
python -I label_batches.py merge|decide --mode eval|third|kappa --out <g2>/eval-public/<셋>
# 정답 확인
python -I label_batches.py make --cands <g2>/gold-g2/gold-cands.jsonl --out <g2>/gold-g2 --mode eval --style careful --guide … --traps … --seed 20261224
python -I label_batches.py merge --out <g2>/gold-g2 --exempt-trap t0007
python -I label_batches.py gold-score --out <g2>/eval-public/dev-cal-g2 <g2>/eval-public/dev-cmp-g2 --gold <g2>/gold-g2
# 조립 : 새 쌍 assemble → 옛 셋과 잇기(스크래치 스크립트, 위 6-1 의 g2_label 규칙) → 얼림
python -I label_batches.py assemble --set <셋> --dirs <g2>/eval-public/<셋> --dest <g2>/assembled-new
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g2/<셋>.jsonl --out ~/.judgemodel/sets/g2/FROZEN-<셋>.sha256
```

- 옛 쌍 잇기는 `assemble` 이 못 한다(묶음 폴더만 읽고, 한 셋 안 라벨 기준이 섞이면 멈춘다). 그래서 git 밖 스크래치 스크립트로 이었다. 다시 할 일이 생기면 `assemble --base <옛 셋> --relabel <옮김 표>` 같은 옵션으로 옮길 만하다.
- 판정 안내 : `<g2>/eval-public/JUDGE-README.txt` · `<g2>/gold-g2/JUDGE-README.txt`.

## blind-g2 결과 (v1.2 B 판 · 판정 v6 · 조립 · 얼림, 2026-10-13)

blind 는 사전 등록 뒤 한 판만 한 번 잰다(사전 등록 규칙). 이번 일은 만들고 얼리기까지다. 모델로 재지 않았다.

### 셋 구성

| 칸 | a | b | c | d | 합 |
| --- | --- | --- | --- | --- | --- |
| 쌍 수 | 128 | **336** | 21 | 153 | **638** |

| 원천 | 쌍 | b | b 목표(예약) | 기대 b(예약) |
| --- | --- | --- | --- | --- |
| VitaminC heldout | 375 | 263 | 130 + 넘김 66 | 196 |
| 한국어 고침 | 119 | 52 | 85 | 19 |
| KLUE (가짜 답) | 144 | 21 | 35 | 35 |
| **합** | 638 | **336** (목표 250 의 134%) | 250 | 250 |

- b 원천 비율은 78 : 15 : 6 (VitaminC : 한국어 : KLUE). 한국어 고침의 b 남김률이 어림 .16 보다 훨씬 높았고(.44), KLUE 는 낮았다(.15).
- 언어 : en 383 · ko 235 · mix 20.

### 판정

- 후보 657 → KLUE 다시 쓰기에서 16 skip(지문에 답 있음 7 · 전제 어긋남 9) → 641쌍 · 8묶음(100쌍 · 함정 10%) × j1 · j2 = 16판 + 셋째 1판(43쌍).
- 함정 정답률 : 판마다 .80 ~ 1.00 (17판 모두 관문 통과, 다시 돌린 판 0) · 답 없음 0.
- κ : 전체 **.894** (641) · c/d κ .872 (164). 첫 두 판 일치 598 / 641 (93%) — 원천별 VitaminC .947 · 한국어 .925 · KLUE .903.
- 셋째가 가름 41 · 버림(셋 다 다름) 2 · 같은 (근거, 주장) 1 은 `--dedupe-same` 로 하나로(같은 칸 b).
- KLUE 다시 쓰기는 g1 길(`attach-rewrite`)과 같다 : 질문 + 지문에 없는 가짜 답 → 평서문(Opus 2판, 80줄씩). 가짜 답이 지문에 글자로 든 줄 0 · 물음 꼴 0 · 길이 밖 0.

### 정답 확인 (신중 판정자)

- 얼린 셋 b 쌍에서 무작위 60(씨앗 20261242 · VitaminC 45 · 한국어 10 · KLUE 5) → `--style careful` 4묶음 × j1 · j2 (함정 정답률 모두 1.0).
- 두 신중 판정이 같은 55쌍 중 54쌍이 b → **일치 .982** (관문 .85 통과). 갈린 5쌍은 셋째를 안 돌렸다 — 다 어긋나도 54/60 = .90 이라 관문 결과가 안 바뀐다.

### 해시 · 겹침

| 무엇 | 값 |
| --- | --- |
| `blind-g2.jsonl` sha256 | `1705b7990bffe85df51a18f3d5f5479246d276e7035b3b45a48272729a637390` |
| `FROZEN-blind-g2.sha256` | `freeze.py check` ok |
| `check-g2` (예약 후보 20칸) | 모두 0 (exit 0) |
| 얼린 셋(KLUE 주장 포함) × 같은 금지 규칙 · dev-cmp-g2 · dev-cal-g2 · 학습 층 글 겹침 | 모두 0 (git 밖 스크래치 검사) |
| `scrub.py check` | 0 건 (638줄) |
| `label_guide` | support-v6 |

- 금지 목록 : `~/.judgemodel/sets/g2/blind-g2.forbid.json` — 판정자에게 보인 후보 657줄 전부(KLUE skip 줄 포함)에서 뽑은 기억 id 1 · 문서 경로 33 · 커밋 54 · 묶음 325 · 출처 id 681 · 후보 id 657 · 쌍 열쇠 640. 학습 · 합성 · 덧학습 금지 목록에 이 파일을 더한다(설계 7절).
- protocol-g2 의 `TODO-blind-g2` 자리에 위 sha256 을 넣는다(이 판은 protocol-g2.json 을 안 건드렸다).

### 명령 (blind-g2)

```
# KLUE 다시 쓰기 : <g2>/eval-public/blind-g2/prep/klue-rw/in-N.jsonl → Opus → out-N.jsonl → rewrite-blind-g2.jsonl
python -I make_sets.py attach-rewrite --cands <g2>/reserve/blind-g2-b-cands.jsonl --rewrite <prep>/klue-rw/rewrite-blind-g2.jsonl --out <prep>/klue-attached.jsonl
#   cands.jsonl = 예약 후보 중 KLUE 아닌 줄 + klue-attached
python -I label_batches.py make --cands <g2>/eval-public/blind-g2/cands.jsonl --out <g2>/eval-public/blind-g2 --mode eval \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261241
python -I label_batches.py merge|kappa --out <g2>/eval-public/blind-g2
python -I label_batches.py decide --out <g2>/eval-public/blind-g2 --mode eval ; third ; merge ; decide
python -I label_batches.py assemble --set blind-g2 --dirs <g2>/eval-public/blind-g2 --dest ~/.judgemodel/sets/g2 --dedupe-same
#   정답 확인 : b 무작위 60 → <g2>/eval-public/blind-g2/gold/gold-cands.jsonl (why=random)
python -I label_batches.py make --cands <gold>/gold-cands.jsonl --out <gold> --mode eval --style careful --guide … --traps … --seed 20261243
python -I label_batches.py gold-score --out <g2>/eval-public/blind-g2 --gold <gold>
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g2/blind-g2.jsonl --out ~/.judgemodel/sets/g2/FROZEN-blind-g2.sha256
python -I make_sets.py check-g2 --build ~/.judgemodel/build/g2
python -I scrub.py check --in ~/.judgemodel/sets/g2/blind-g2.jsonl --names … --game-names …
```

- 판정 안내 : `<g2>/eval-public/blind-g2/JUDGE-README.txt` · `<g2>/eval-public/blind-g2/gold/JUDGE-README.txt`. 씨앗 : 묶음 20261241 · 신중 20261243 · 무작위 뽑기 20261242.

### blind-g2 보강 — a · c 더해 다시 얼림 (2026-10-13)

- **까닭 :** 처음 얼린 638쌍은 c 21 · a 128 이었다. c 21 로는 릴리스 관문의 c 재현(v1.1.0 대비 −3%p)을 잴 수 없고 3칸 정확도도 소음만 잰다. 아직 한 번도 안 쟀고 사전 등록 전이라 a · c 를 더해 다시 얼렸다.
- **해시 :** 보강 전 `1705b7990bffe85df51a18f3d5f5479246d276e7035b3b45a48272729a637390` → 보강 후 `c015b039b7c7ecd4f2aa631d0d06ac5eda954734d0460f4aea87fbce5ad8e5e2`. 옛 셋 · 해시 · 금지 목록은 `<g2>/eval-public/blind-g2/prep/*.pre-ac.*` 에 남겼다.
- **새 후보 352쌍** (blind-g2 예약 묶음 안에서만 뽑음, 씨앗 20261244). 예약 파일 `reserve/blind-g2-b-cands.jsonl` 에 더했고(657 → 1009줄) `check-g2` 를 다시 돌려 0 을 봤다.

| 원천 | 꼴 | 수 |
| --- | --- | --- |
| VitaminC heldout | a — SUPPORTS | 110 |
| VitaminC heldout | c — 같은 쪽 다른 주장 | 60 |
| VitaminC heldout | c — 다른 묶음 근거 × 주장 (xscope 무관, BM25 10~60위) | 60 |
| 한국어 고침 | c — 다른 문서 근거 × 주장 | 40 |
| KLUE | a — 물음 + 참 답을 평서문으로 (Opus 다시 쓰기 42, 건너뜀 0) | 42 |
| KLUE | c — 다른 문서 근거 × 주장 | 40 |

- **판정 :** v6 · 4묶음 × j1 · j2 (함정 39) + 세 번째 판정 15쌍. 8판 모두 통과(함정 .89~1.0, 답 없음 0). κ 전체 **.930** · c/d **.861**. 일치 337 · 세 번째 15 · 버림 0. 새 쌍 칸 : a 143 · b 4 · c 159 · d 46.
- **보강 뒤 셋 990쌍 :** a **271** · b **340** · c **180** · d **199** (b 는 새 후보에서 판정자가 b 로 본 4쌍만 늘었다).

| 원천 | a | b | c | d | 합 |
| --- | --- | --- | --- | --- | --- |
| VitaminC | 172 | 267 | 85 | 81 | 605 |
| KLUE | 32 | 21 | 55 | 118 | 226 |
| 한국어 고침 | 67 | 52 | 40 | 0 | 159 |

| 검사 | 결과 |
| --- | --- |
| `freeze.py check` | ok |
| `check-g2` (예약 후보 1009) | 모두 0 (exit 0) |
| 얼린 셋 × dev-cmp-g2 · dev-cal-g2 · 학습 층 글 겹침 | 0 |
| `scrub.py check` — 셋 990줄 · 새 후보 352줄 | 0 건 |
| `blind-g2.forbid.json` 다시 만듦 | 후보 id 1009 · 쌍 열쇠 992 · 묶음 325 |

```
python -I label_batches.py make --cands <g2>/eval-public/blind-g2/prep/ac-cands.jsonl --out <g2>/eval-public/blind-g2/ac --mode eval \
    --guide ../Docs/Guide/라벨기준support-v6.md --traps ~/.judgemodel/build/g1/traps.jsonl --seed 20261245
python -I label_batches.py merge|kappa|decide --mode eval|third --out <g2>/eval-public/blind-g2/ac
python -I label_batches.py assemble --set blind-g2 --dirs <g2>/eval-public/blind-g2 <g2>/eval-public/blind-g2/ac --dest ~/.judgemodel/sets/g2 --dedupe-same
python -I ../train/freeze.py write --pairs ~/.judgemodel/sets/g2/blind-g2.jsonl --out ~/.judgemodel/sets/g2/FROZEN-blind-g2.sha256
```
