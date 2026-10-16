# 모델 · 자료 출처와 라이선스

- **코드** : 이 저장소의 코드는 **MIT** 다 (`LICENSE`).
- **배포하는 가중치** : 공개 자료로만 학습한 판이며 **CC BY-SA 4.0** 으로 낸다. 학습 자료 대부분이 CC BY-SA 라서다. 아래 표의 출처를 같이 적는다.
- **비공개 자료로 덧학습한 모델은 배포하지 않는다.**
- **비상업(NC) 자료는 쓰지 않는다.** `model.json` 의 `nc` 가 참이거나 `publishable` 이 거짓이면 내놓지 않는다.

## 1. 몸통 모델

| 이름 | 출처 | 라이선스 | 어디에 쓰나 |
| --- | --- | --- | --- |
| `FacebookAI/xlm-roberta-large` (revision `c23d21b`) | Meta AI · Hugging Face | MIT | 한 · 영 판 몸통 |
| `klue/roberta-large` | KLUE 벤치마크 팀 · Hugging Face | CC BY-SA 4.0 | 첫 한국어 판 몸통 (지금은 안 씀) |
| `mmetamong/ko-decision-roberta-large-klue` | Hugging Face | CC BY-SA 4.0 | 견줌 측정에만. 학습에는 안 씀 |

## 2. 기반 학습 — 공개 NLI 자료

`train/data_prep.py` 의 `LICENSES` 표와 같다. 모델을 내보내면 `model.json` 의 `data` 칸에 자료마다 라이선스 · `nc` 가 적힌다.

| 이름 | 출처 | 라이선스 | 어디에 쓰나 |
| --- | --- | --- | --- |
| KLUE-NLI v1.1 학습 · dev | KLUE 벤치마크 | CC BY-SA 4.0 | 학습 · dev |
| KorNLI 학습 (`multinli.train.ko.tsv` · `snli_1.0_train.ko.tsv`) | kakaobrain | CC BY-SA 4.0 | 학습 (바퀴마다 표본) |
| MultiNLI 1.0 학습 | NYU (Williams 등 2018) | 대부분 OANC · 소설 일부 CC BY-SA 3.0 · CC BY 3.0 · 나머지 미국 공유 저작물 | 학습 (바퀴마다 표본) |
| MultiNLI dev matched | 같음 | 같음 | 영어 확인용 평가에만 |
| SNLI 1.0 학습 | Stanford NLP (Bowman 등 2015) | CC BY-SA 4.0 | 학습 (바퀴마다 표본) |

MultiNLI · SNLI · KorNLI 는 학습 셋만 받는다. 이름에 `dev` · `test` 가 든 파일은 `data_prep.py` 가 거부한다 (위 MultiNLI dev matched 평가용은 따로 받는다).

## 3. 덧학습 · 평가 쌍 — 공개 원천

`build/` 가 원천마다 (근거, 주장) 후보를 뽑는다. 줄마다 `lic` 칸에 원천 라이선스가 붙는다.

| 이름 | 출처 | 라이선스 | 어디에 쓰나 |
| --- | --- | --- | --- |
| VitaminC | Schuster 등 2021 (위키백과 고치기 전후 문장) | CC BY-SA 3.0 | 영어 학습 · 평가 쌍 |
| KLUE-MRC | KLUE 벤치마크 | CC BY-SA 4.0 | 한국어 학습 · 평가 쌍 (지문 + 질문을 평서문으로 바꾼 주장) |
| KoBEST BoolQ | KoBEST | CC BY-SA 4.0 | 한국어 학습 · 평가 쌍 |
| 한국어 위키백과 개정 이력 (덤프) | 위키미디어 재단 | CC BY-SA 4.0 | 한국어 학습 · 평가 쌍 (고치기 전후 문장) |
| MIT 라이선스 한국어 공개 저장소의 커밋 이력 | GitHub 공개 저장소 | MIT | 한국어 학습 · 평가 쌍 |
| 공개 Jira 이슈 덤프 | 공개 Jira 자료 모음 | CC BY 4.0 | 영어 쌍 (이슈 링크 · 중복) |
| Python PEP | python/peps | CC0 · 퍼블릭 도메인 (그 표시가 있는 문서만) | 영어 쌍 (설계 제안서) |
| Kubernetes KEP | kubernetes/enhancements | Apache-2.0 | 영어 쌍 (설계 제안서) |
| Swift Evolution | swiftlang/swift-evolution | Apache-2.0 | 영어 쌍 (설계 제안서) |
| Go proposal | golang/proposal | BSD-3-Clause | 영어 쌍 (설계 제안서) |
| MiniCheck | Tang 등 2024 | MIT | 영어 쌍 (첫 판 후보) |
| WiCE | Kamoi 등 2023 | ODC-BY · CC BY-SA | 학습 밖 평가(ood)에만 |
| SufficientFacts | Atanasova 등 2022 | MIT | 학습 밖 평가(ood)에만 |
| ContractNLI | Koreeda · Manning 2021 | CC BY 4.0 | 학습 밖 평가(ood)에만 |
| 리브레 위키 | librewiki.net | CC BY-SA 3.0 | 조사만 했다. 아직 안 씀 (`Docs/Research/2026-10-13-리브레위키실물조사.md`) |

**라벨 만들기에 LLM 을 썼다.** 위 원천 쌍의 라벨 판정과 일부 주장 문장 다시 쓰기(질문 → 평서문 등)는 LLM(Anthropic Claude)이 했다. 판정자 여럿이 갈린 쌍은 빼거나 사람 확인 목록으로 돌렸다. 라벨 기준은 `Docs/Guide/라벨기준support-v6.md`.

## 4. 비공개 자료

| 이름 | 라이선스 | 어디에 쓰나 |
| --- | --- | --- |
| 비공개 메모 · 기록 쌍, 비공개 저장소의 개정 이력 | 비공개 | 덧학습 · 측정에만. **이 자료로 덧학습한 모델은 배포하지 않는다** |

## 5. 안 쓰는 자료

| 이름 | 라이선스 | 까닭 |
| --- | --- | --- |
| KorNLI dev · test (XNLI 번역) | **CC BY-NC-SA 4.0** | 비상업. `data_prep.py` 가 경로 이름으로 거부 |
| XNLI · ANLI | 비상업 | 비상업. `data_prep.py` 가 경로 이름으로 거부 |
| 나무위키 | CC BY-NC-SA 2.0 KR | 비상업 |
| MultiNLI · SNLI test | 원본과 같음 | 학습 셋만 쓴다 |

## 6. 배포 가중치에 적을 출처

공개 가중치는 **CC BY-SA 4.0** 으로 내고, 그 판이 배운 자료의 출처(위 2절 · 3절)를 모델 카드에 적는다.
몸통 `xlm-roberta-large` 는 MIT 지만 자료(KLUE · KorNLI · SNLI · KLUE-MRC · 위키백과의 BY-SA 4.0, MultiNLI 일부 · VitaminC 의 BY-SA 3.0)가 BY-SA 라 가중치도 BY-SA 를 따른다.
MultiNLI 의 OANC · CC BY 3.0, Jira · ContractNLI 의 CC BY 4.0 출처 표시도 같이 적는다.
