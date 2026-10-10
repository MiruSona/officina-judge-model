# 모델 · 자료 출처와 라이선스

| 무엇 | 출처 | 라이선스 | 쓰나 |
| --- | --- | --- | --- |
| `FacebookAI/xlm-roberta-large` (한·영 판 몸통, revision `c23d21b`) | Meta AI · Hugging Face | MIT | 씀 (2026-10-09 R0 부터) |
| `klue/roberta-large` (첫 판 몸통) | KLUE 벤치마크 팀 · Hugging Face | CC BY-SA 4.0 | `nli-base-v1` · `nli-studio-v1` 에 씀 |
| `mmetamong/ko-decision-roberta-large-klue` | Hugging Face | CC BY-SA 4.0 | 0단계 견줌만 · 지름길이면 인코더 시작점 |
| KLUE-NLI v1.1 학습 · dev | KLUE 벤치마크 | CC BY-SA 4.0 | 씀 |
| KorNLI 학습 (`multinli.train.ko.tsv` · `snli_1.0_train.ko.tsv`) | kakaobrain | CC BY-SA 4.0 | 씀 (바퀴마다 표본) |
| MultiNLI 1.0 학습 (`multinli_1.0_train.jsonl`) | NYU (Williams 등 2018) | 대부분 OANC · 소설 일부 CC BY-SA 3.0 · CC BY 3.0 · 나머지 미국 공유 저작물 | 씀 (바퀴마다 표본) |
| SNLI 1.0 학습 (`snli_1.0_train.jsonl`) | Stanford NLP (Bowman 등 2015) | CC BY-SA 4.0 | 씀 (바퀴마다 표본) |
| MultiNLI · SNLI dev · test | 같음 | 같음 | 안 씀 — 학습 셋만 (`data_prep.py` 가 이름으로 거부) |
| KorNLI dev · test (XNLI 번역) | kakaobrain | **CC BY-NC-SA 4.0** | **안 씀** — `data_prep.py` 가 경로 이름으로 거부 |
| XNLI · ANLI | — | 비상업 | **안 씀** |
| 기억 쌍 (스튜디오) | 스튜디오 기억 저장소 | 비공개 | 스튜디오 모델에만 · 밖으로 안 나감 |

- 기본 모델(`nli-base-v1`) 가중치를 내놓을 때는 **CC BY-SA 4.0** 으로 위 출처를 적는다.
- 한·영 판 기본 모델도 같다 — 몸통은 MIT 지만 자료(KLUE · KorNLI · SNLI 의 BY-SA 4.0, MultiNLI 일부의 BY-SA 3.0)가 BY-SA 라 **CC BY-SA 4.0** 으로 내놓고 MultiNLI 의 OANC · CC BY 3.0 출처도 같이 적는다. 내놓는 길(릴리스 첨부 등)은 공개 전환 때 따로 정한다.
- `model.json` 의 `data` 칸에 가르친 자료마다 라이선스 · `nc` · `private` 가 적힌다. `publishable` 이 거짓이면 내놓지 않는다.
- 코드의 라이선스는 저장소를 공개할 때 정한다.
