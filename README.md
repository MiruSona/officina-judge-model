# JudgeModel

> 옛 이름은 JudgeTool 이었다 (2026-10-10 에 JudgeModel 로 바꿈 — 만든 것이 모델이라서. 독립 저장소 분리 전).

(근거, 주장) 한 쌍을 읽고 **지지 · 반대 · 무관** 확률을 내는 한국어 NLI 분류기 툴이다.
**다음 판부터 한·영** — 한국어와 영어만 함께 학습한다(다국어 아님, 사용자 2026-10-09).
학습 · 측정 · 서빙(HTTP)을 이 폴더가 다 갖는다. 다른 툴은 HTTP 응답 꼴만 따른다.

설계는 `Docs/Design/2026-10-09-길1한국어NLI분류기설계.md`.
**판정 넷 확장(다음 판, 2026-10-10)** : 무관을 c 무관 · d 비슷한 계열로 쪼갠 네 칸 모델(`nli-v2`)과 양방향 접기 — 설계 `Docs/Design/2026-10-10-판정넷확장설계.md`, 라벨 기준 `Docs/Guide/라벨기준support-v4.md`. 학습 · 내보내기 · 서버 코드는 들어갔고(15 소단계), 네 칸 모델은 아직 없다.

## 상태 : 길 1 blind 통과 — 꽂힘 · 상시 (2026-10-09)

모델 `nli-studio-v2` 가 얼린 blind 332쌍에서 자 다섯을 다 넘었다. 한 번만 쟀고, 잰 뒤 문턱 · 온도 · 모델을 다시 고르지 않았다.

| 자 | 값 | 기준 |
| --- | --- | --- |
| 정확도 | **0.892** | ≥ 0.85 |
| 반대 재현 | **0.882** | ≥ 0.75 |
| 확신 거짓 지지 | **2** | ≤ 3 |
| 보류 | **13.6%** | ≤ 30% |
| p95 | **414ms** | ≤ 1,000ms |

- 모델 : `nli-studio-v2` = `xlm-roberta-large` 한·영 몸통 위에 스튜디오 쌍을 덧학습한 것 (보강 4바퀴째 · 온도 2.15). 밖으로 안 나간다.
- 약한 자리 셋 : **num(숫자) 7/14** — 지지를 반대로 본다 · **mix(한·영 섞임) 지지 재현 0.643** · **실물 supersede(옛 결정을 새 결정이 덮음)는 확인 못 함** — blind 의 supersede 는 합성뿐이다.
- 앞 판 : 첫 모델 `nli-studio-v1`(klue 몸통)은 game90 에서 정확도 0.748 · 반대 재현 0.533 으로 미달해 보류했었다.

사용자 결정 (2026-10-09)

| # | 정한 것 |
| --- | --- |
| ① | 손님 쪽 `nli_url` 을 상시로 켜고, 쓰면서 피드백을 받아 보완한다 — **끝**(로그온 등록만 사용자 몫) |
| ② | 독립 저장소로 분리한다 — **끝**(2026-10-10, 공개 저장소 `officina-judge-model`) |
| ③ | 보강 내내 틀린 25쌍의 라벨은 사용자가 직접 판별한다 |
| ④ | 안 쓰는 검사점(`r1-01·02·03·05`)은 지운다 — **끝** |

다음 판과 남은 일은 `Docs/Todo/진행상황.md`. 기본 모델 가중치는 git 밖이다.

| 결과 문서 | 무엇 |
| --- | --- |
| `Docs/Research/2026-10-09-N1-GPU벤치.md` | 미니PC GPU 학습 벤치 |
| `Docs/Research/2026-10-09-N5N6-0단계와기반학습.md` | ko-decision 무학습 견줌 · 기반 모델 학습 |
| `Docs/Research/2026-10-09-N7-덧학습.md` | 스튜디오 쌍 덧학습 |
| `Docs/Research/2026-10-09-N8N10-ONNX와측정.md` | ONNX 일치 · p95 · game90 측정(미달) |
| `Docs/Research/2026-10-09-R0-기반모델과영어자료조사.md` | 한·영 몸통 · 영어 자료 고르기 |
| `Docs/Research/2026-10-09-R0-새판정셋.md` | 새 판정 셋 dev 329 · blind 332 얼리기 |
| `Docs/Research/2026-10-09-R0-기반학습v2.md` | 한·영 기반 모델 `nli-base-v2` 학습 |
| `Docs/Research/2026-10-09-R1-합성supersede.md` | 합성 supersede 쌍 만들기 |
| `Docs/Research/2026-10-09-R3-학습쌍둥이.md` | 한·영 학습 쌍둥이 |
| `Docs/Research/2026-10-09-길1보강반복.md` | 보강 5바퀴 기록 · 반복 마감 |
| `Docs/Research/2026-10-09-R5-blind측정.md` | **blind 332 측정(통과)** |

## 끝점 (다른 툴과의 약속)

```
POST /judge  {"evidence": "...", "claim": "..."}
  → 200 {"a": 지지, "b": 반대, "c": 무관, "model": "<가중치 SHA 앞 8자>", "ms": 12.3}
GET  /health → {"ok": true, "model": "...", "backend": "onnx", "name": "nli-studio-v2"}
```

- 127.0.0.1 에만 붙는다 · 기본 포트 **8092** · 한 번에 한 판 · 몸통 1MB 상한(413) · 꼴이 틀리면 400.
- 확률은 온도를 건 뒤 값이다. 확신 문턱(`nli_sure` · `nli_sure_support`)은 손님 쪽이 건다 — 출발값은 `model.json` 의 `scores.thresholds`.

**네 칸 모델 (`nli-v2`, 판정 넷 확장)** — 서버는 `model.json` 의 칸 수를 읽어 꼴을 고른다. 옛 3칸 `model.json` 은 위 꼴 그대로 낸다.

```
POST /judge  → 200 {"a": 지지, "b": 반대, "c": 무관, "d": 비슷한 계열, "model": "...", "ms": 12.3}
GET  /health → {..., "letters": ["a","b","c","d"], "protocol": "nli-v2"}
```

- `serve.py --compat3` : d 를 c 에 더해 옛 3칸 꼴(`nli-v1`)로 낸다. 네 칸 모델을 옛 손님에게 붙일 때 쓴다.
- `model.json` 에 `labels` · `letters` · `protocol`(`nli-v1`/`nli-v2`) · `label_guide`(`support-v3`/`support-v4`)가 붙는다. 네 칸이면 문턱에 `nli_sure_related`(확신 d)가 더 있다.

### 1.1.0 응답 꼴 — `label_set` opt-in (2026-10-11)

**요청에 `label_set` 을 안 실으면 응답은 옛 꼴 그대로다(바이트까지).** 실은 손님에게만 세 칸이 더 붙는다. 설계는 `Docs/Design/2026-10-11-d자료층재구축설계.md` 6절.

```
POST /judge {"evidence": "...", "claim": "...", "label_set": "nli4"}
  → 200 {"a", "b", "c", "d", "model", "ms",
         "labels": ["support","contradict","unrelated","related"], "model_version": "1.1.0", "label_set": "nli4"}
GET  /info  → {"api_version": "1.1.0", "model_version", "model", "name", "protocol", "label_guide",
               "default_label_set", "label_sets": {"nli3": {letters, labels}, "nli4": {...}}, "id2label"}
```

| 요청 `label_set` | 4칸 모델 | 3칸 모델 · `--compat3` |
| --- | --- | --- |
| 없음 (또는 `null`) | 옛 꼴 그대로 (`a b c d model ms`) | 옛 꼴 그대로 (`a b c model ms`) |
| `"nli3"` | d 를 c 에 더한 3칸 + 세 칸 · `label_set: "nli3"` | 3칸 + 세 칸 · `"nli3"` |
| `"nli4"` | 4칸 + 세 칸 · `"nli4"` | **3칸 + 세 칸 · `"nli3"`** — 에러 아님. 손님은 응답의 `label_set` 을 읽어 c·d 를 못 가른다는 것을 안다 |
| 그 밖 (`"nli5"` · 숫자 · 목록) | 400 `{"error": "unknown label_set"}` | 같음 |

- `/health` 는 안 바뀐다.
- `model_version` 은 `model.json` 의 `model_version`(`export_onnx.py --model-version 1.1.0` 이 적는다)이고, 없으면 `/health` 의 `model`(가중치 SHA 앞 8자)과 같은 값이다. `model.json` 의 `label_set`(`nli3`/`nli4`)은 칸 수와 어긋나면 서버가 안 뜬다. 옛 `model.json`(두 칸 없음)은 그대로 읽힌다.
- 판 번호 `1.1.0` 은 `serve.py` 의 `API_VERSION` 이다. 기본 응답을 4칸으로 바꾸는 날이 2.0.0 이다.
- 저장소 안 호출자(`build/teacher.py` · `measure/run_fold.py` · `measure/run_measure.py` · `check_onnx.py`)는 `label_set` 을 명시해 보낸다.

#### 카드 — v1.1 한계와 성적

**v1.1.0 은 「3칸 개선판」이다 (사용자 결정 2026-10-11).** 모델은 `nli-studio-v4`(학습 판 `g1-B2-s3` · 가중치 SHA `14d0d520…` · 온도 2.35). 3칸(지지 · 반대 · 무관) 일은 v1.0.0(`nli-studio-v2`)보다 같은 쌍에서 분명히 낫다. **릴리스 자 여덟 중 셋은 미달이다** — 숨기지 않고 아래에 적는다.

- **실서버(8092)는 `--compat3` 로 띄워 응답 꼴이 v1.0.0 과 같다(3칸).** 옛 손님은 그대로 쓴다. 옵션 없이 띄우면 기본이 4칸(nli4, `a b c d model ms`)이고, `label_set: "nli3"` 로 3칸을 받을 수 있다(위 표).
- 띄우는 스크립트는 `run/serve-studio.ps1`(기본 v4 · `--compat3`). 되돌리기 : `.\run\serve-studio.ps1 -Model ~\.judgemodel\models\nli-studio-v2 -Compat3:$false`
- **4칸(`label_set: "nli4"`)은 실험 기능 · 목표 미달이다.** d 재현이 .46 으로 목표 .70 에 크게 못 미치고, 확신 d 문턱을 못 찾아(`nli_sure_related` .99) **d 답은 늘 보류(hold)로 간다.** d 를 믿고 자동 처리하지 않는다.
- d 70% · b 80% 는 v1.2 과제다.

- d(비슷한 계열)는 「우리 도메인(개발 메모 · 설계 문서 · git 이력) + 공개 원천(이슈 · 설계 제안서 · 위키 · 뉴스 지문)」 안에서 쟀다. 손님 실물로는 안 쟀다.
- 영어 d 는 번역하지 않은 영어 원천에서, 한국어 d 는 KLUE-MRC 와 우리 아카이브에서 왔다.
- 라벨은 Opus 판정자 합의다. 사람 감사는 갈린 쌍 목록으로 미뤘다.
- 학습 자료 라이선스 : VitaminC CC BY-SA 3.0 · KLUE-MRC CC BY-SA 4.0 · Jira CC BY 4.0 · PEP CC0 · KEP · Swift Apache-2.0 · Go BSD-3 · MiniCheck MIT.

blind-g1 1,118쌍(a 221 · b 66 · c 566 · d 265) · 한 번만 잼 · 2026-10-11

| 자 | 값 | 기준 | 판정 |
| --- | --- | --- | --- |
| 3칸 접은 정확도 | **.8148** | ≥ .85 | **미달** |
| v1.0.0 대비 3칸 정확도 | +.0501 [+.0301, +.0706] · McNemar p 1e-6 (v1.0.0 .7648) | 95% 아래끝 ≥ −.02 | 통과 |
| b(반대) 재현 | .7576 | ≥ .75 | 통과 |
| 확신 거짓 지지 | 1 / 897 (0.11%) | ≤ 1.5% | 통과 |
| 보류 | **.4159** | ≤ .30 | **미달** |
| d 재현 | **.4604** | ≥ .70 | **미달** |
| d 확신 정밀 | 잴 수 없음 (확신 d 0개) | ≥ .80 | — |
| p95 | 281.9ms | ≤ 1,000ms | 통과 |

- 칸별 재현 : a .557 · b .758 · c .839 · d .460. 문턱 없는 거짓 지지는 v1.0.0 70 → 38.
- 보류 42% 의 뿌리는 d 다. 온도를 건 d 확률이 가장 높아도 .671 이라 확신 d 문턱(.99)을 넘는 답이 없다.
- 도메인별 · 언어별 d 재현 : 이번 판정에서 나눠 재지 않았다(v1.2 때 잰다).

| ood-g1 320쌍 (판정 밖, 참고) | 값 |
| --- | --- |
| 3칸 접은 정확도 | .5156 (v1.0.0 .4844 · 차 +.0312 [−.0031, +.0656]) |
| 보류 | .7031 |
| d 재현 | .4902 |

- 낯선 글에선 70% 를 사람에게 넘긴다. 학습에 없는 도메인에서 쓸 때는 보류가 많다고 보고 쓴다.

### 꼬리표 떼기 (손님과의 약속, 2026-10-11)

mem 기억 본문은 `왜 :` · `무엇을 :` 같은 꼬리표로 줄을 시작하는 일이 많다(판정 셋 v3 쌍 문장의 약 25%). 꼬리표는 뜻이 없는 틀이라 **보내기 전에 뗀다.**

- 규칙 : 줄마다 머리의 `^\s*(근거|왜|무엇을|무엇|이후|다음에|다음|까닭|내용|결론|주의|정한 것|확인|어떻게|무엇이|무슨 일이)\s*[:：]\s*` 를 지운다(여러 줄 모드). 줄 가운데 것은 안 건드린다.
- 이 툴 안 : `train/data_prep.py` 의 `strip_tags` 한 곳. mem 쌍 꼴(evidence·claim)을 읽을 때 학습 · 평가 · `check_onnx` 가 떼고, `measure/run_measure.py` · `run_fold.py` 는 서버에 보내기 전에 뗀다. 셋 파일은 안 고친다(SHA 유지).
- 서버(`serve.py`)는 안 뗀다. **손님(AIMemoryTool)도 보내기 전에 같은 규칙으로 뗀다(10-11 구현, Go `StripTags`).** 낱말 목록·꼴을 바꾸면 두 쪽을 같이 바꾼다.
- 머리의 `^\s*` 가 앞 빈 줄의 줄바꿈까지 먹는 것은 의도다 — 손님 `StripTags` 도 같게 동작한다.

## 세 자리

| 자리 | 무엇 | 어디 |
| --- | --- | --- |
| 이 툴 (공개 예정) | 코드 · 문서 · 시험 · 기본 모델 `nli-base-v1` 의 `model.json` | git |
| 기본 모델 가중치 | 공개 NLI 자료로만 가르친 `nli-base-v1` | git 밖 (`~/.judgemodel/models/`) |
| 스튜디오 모델 | 기억 쌍으로 덧학습한 `nli-studio-v1` | git 밖 · **밖으로 안 나간다** |

가중치 · 자료 · 판 기록은 git 에 넣지 않는다 (`.gitignore`).

## 까는 법 (서빙만)

```
python -m venv venv-serve
venv-serve/Scripts/pip install -r requirements.txt     # onnxruntime · tokenizers · numpy
python -I serve.py --model ~/.judgemodel/models/nli-studio-v1 --port 8092 --threads 4
```

모델 자리는 `--model` 또는 환경 변수 `JUDGEMODEL_MODEL_DIR`. 모델 폴더에는 `model.onnx` · `tokenizer/tokenizer.json` · `model.json` · `calibration.json` 이 있다. 가중치가 2GB 를 넘는 몸통(XLM-R large)은 `model.onnx.data` 가 더 붙는다 — 같은 폴더에 같이 둔다. 모델 폴더를 옮길 때도 `model.onnx.data` 를 함께 옮긴다. 이 판에서 `model.json` 의 `onnx_sha256` 은 그래프(`model.onnx`)만 가리키고, 가중치 파일은 `onnx_data_sha256` 이 따로 가리킨다.
`--backend torch` 는 물러설 길이다 (`export_onnx.py --with-torch` 로 `torch/` 를 같이 내보냈을 때).

메인 PC 에서는 `run/serve-studio.ps1` 로 띄운다. 기본 `nli-studio-v4`(v1.1.0 · `--compat3`)를 8092 에 띄우고, 이미 떠 있으면 `/health` 만 찍는다. 로그는 `~/.judgemodel/logs/serve-studio.{log,err}`.
로그온 때 저절로 뜨게 하려면 `run/serve-studio.ps1 -Register` 를 한 번 돌린다 (작업 이름 `JudgeModel-NLI` · 내릴 땐 `-Unregister`).

## 폴더 지도

| 경로 | 하는 일 |
| --- | --- |
| `pairenc.py` | 쌍 인코딩 한 자리 (토크나이저의 쌍 틀 그대로 — XLM-R `<s> 근거 </s></s> 주장 </s>` · klue `[CLS] 근거 [SEP] 주장 [SEP]` · 근거만 자름 · 특수 토큰 수는 토크나이저에서 셈). 학습 · 평가 · 서버가 같이 쓴다 |
| `serve.py` | 판정 서버 |
| `export_onnx.py` · `check_onnx.py` | ONNX 내보내기 · `model.json` 쓰기 / 토큰·확률·p95·int8 일치 검사 |
| `train/data_prep.py` | KLUE-NLI · KorNLI · MNLI · SNLI 원본 → 공통 jsonl · 바퀴별 표본 · 토큰 길이 재기(`measure`). **라벨 매핑은 여기 한 곳** |
| `train/train_nli.py` | 학습 고리 (기반 · 덧학습 공용 · `--smoke` · 네 칸으로 머리 늘리기 `--grow-head 4`) |
| `train/heads.py` | torch 없이 도는 칸 계산 (클래스 가중치 · 평가 지표 · 합친 칸 손실 기준값) |
| `train/eval_nli.py` · `train/calibrate.py` | 평가(쌍별 로짓) · 온도와 문턱 둘 |
| `train/split_pairs.py` | 쌍을 기억 덩이 단위로 train/dev 가르기 |
| `train/freeze.py` · `train/bench.py` | 판정 셋 얼리기 · GPU 벤치 |
| `measure/` | `prereg.py`(사전 등록) · `run_measure.py`(모델 표 · 사다리 표 · p95) · `run_fold.py`(fold40 접기 오판, 네 칸) |
| `run/` | 미니PC 감싸개 `podman-train.sh`(GPU) · `cpu-train.sh`(CPU) · 감시 고리 `watch-lib.sh` |
| `run/serve-studio.ps1` | 메인 PC 판정 서버 띄우기 · 로그온 작업 등록(`-Register`)·해제(`-Unregister`) |
| `tests/` | 내려받기 없이 도는 시험 |

## 학습 차례 (미니PC)

미니PC 자리는 `~/judgetool/` 하나다 — `src/`(이 폴더 사본) · `hf/` · `data/` · `mem/` · `runs/` · `export/` · `logs/`.
감싸개는 그 자리를 작업 폴더로 쓰므로 인자의 상대 경로는 `~/judgetool` 기준이다.
감싸개는 llama 서버를 내리거나 올리지 않는다. 큰 학습 판 앞뒤의 `systemctl --user stop/start llama` 는 사람이 한다 — llama 가 떠 있으면 스왑이 밀려 감시가 멈춘다.

1. **기계별 값** : `~/judgetool/env.sh` (git 밖, 처음 한 번 만든다). 게임 서버를 지켜야 하면 그 실행 파일 이름을 적는다 — 비면 게임 서버 검사를 건너뛴다.
   ```
   export JUDGEMODEL_GAME_PROC=<게임 서버 실행 파일 이름>
   export JT_VENV=<CPU 길에 쓸 torch venv>      # cpu-train.sh 기본은 ~/judgetool/venv
   ```
2. **자료** : `train/data_prep.py klue` (학습 · dev) → `kornli` (바퀴마다 5만 표본) → `mnli` (바퀴마다 5만) · `snli` (바퀴마다 2.5만) → `measure` 로 `max_len` 고르기 · 영어 확인용은 `eval-en`(MNLI dev matched → 평가 전용, 학습엔 안 씀). 한 : 영 = 1 : 1 (`Docs/Research/2026-10-09-R0-기반모델과영어자료조사.md`).
3. **기반 학습** : `run/podman-train.sh -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 -e HF_HUB_OFFLINE=1 --name base-2 train/train_nli.py --device cuda --dtype bf16 --init hf/xlm-roberta-large …` → `eval_nli.py --max-len 320` · `calibrate.py --temperature-only` → `export_onnx.py --name nli-base-v2 --max-len 320` (한·영 판 실측은 `Docs/Research/2026-10-09-R0-기반학습v2.md`).
4. **덧학습** : `split_pairs.py` 로 dev 다시 가르기 → `train_nli.py --stage studio --init runs/base-2/ckpt-best --replay … --mix-klue 1:1 --class-weight --select last --eval-every 0` → `eval_nli.py`(dev) → `calibrate.py` → `export_onnx.py --name nli-studio-v2 --parent …/nli-base-v2/model.json`. `--select last` 는 마지막 바퀴를 `ckpt-best` 로 낸다(덧학습 기본). `studio` 는 거짓 지지 개수를 먼저 견줘 덜 배운 이른 바퀴를 고르는 함정이 있다 (`Docs/Research/2026-10-09-길1보강반복.md`). `export_onnx.py` 는 `--body-name` 이 없으면 `--parent` 의 몸통을 물려받고, 둘 다 없으면 멈춘다 — 기반 판은 `--body-name FacebookAI/xlm-roberta-large` 를 꼭 준다.
5. **꽂기 전** : 메인 PC 로 모델 폴더를 옮겨 `serve.py` 를 띄우고 `check_onnx.py --url …` 통과.
6. **측정** : `freeze.py check` → `measure/prereg.py` (한 번만, 덮지 않음) → `measure/run_measure.py`.
   - **실사용 측정 use1** (mem 이 실제로 묻는 자리에서 v1.0.0 · v1.1.0 견주기, 2026-10-12) : `measure/README.md` 7항.

명령마다 인자는 각 파일 머리 주석에 있다.

**네 칸 덧학습 (`nli-studio-v3`, 판정 넷 확장)** — 위 4단계에 `--grow-head 4` 만 더한다. 3칸 `runs/base-2/ckpt-best` 의 a·b·c 줄을 그대로 옮기고 d 줄은 c 줄 + 작은 잡음으로 시작한다.

- train 쌍은 `support-v4` 라벨(a·b·c·d · 못 고른 `cd`). `cd` 쌍과 replay 의 공개 NLI 중립은 「c 또는 d」로 가르친다 — 손실 −log(p_c+p_d). replay 의 `src` 가 공개 출처 이름(klue · kornli-* · mnli · snli)이어야 이렇게 간다. 로그의 「replay 중립 … 합친 칸 손실」 줄로 확인한다.
- `calibrate.py` 는 네 칸이면 확신 거짓 지지 몫을 2% → 1% 로 조이고, `nli_sure_related` 를 dev 확신 d 정밀도 ≥ 0.80 인 가장 작은 값(0.50~0.99)으로 고른다.
- 측정 자의 정본은 `measure/protocol-g1.json` 의 `release_gate`(prereg.py 가 읽음). `prereg-v2.example.json` 은 꼴 보기용. blind 는 `run_measure.py`, fold40 은 `run_fold.py` 로 잰다.
- 명령 줄 보기는 `run/podman-train.sh` 머리 주석의 r2-01 줄.

| 사용법 | 하는 일 |
| --- | --- |
| `train/soup.py --out <폴더> <검사점> <검사점> …` | 같은 시작에서 갈라진 검사점들의 가중치를 단순 평균해 새 검사점으로 (model soup). 원본 목록은 `soup.json` |
| `train_nli.py --save-every-epoch` | 바퀴 끝마다 `ckpt-e<N>` 을 따로 남긴다 — 바퀴별로 따로 재거나 soup 재료로 쓴다 |
| `train_nli.py --replay-neutral {cd,c}` | 네 칸 학습에서 replay 공개 중립을 무엇으로 배우나. `cd` = 「c 또는 d」 합친 칸(기본) · `c` = c 로 고정 |
| `eval_nli.py --public-neutral {cd,c}` | 공개 중립을 채점할 칸. 학습의 `--replay-neutral` 과 같게 준다(기본 `cd`) |

### 감시와 멈춤

| 조건 | 누가 보나 | 처리 |
| --- | --- | --- |
| 손실 NaN·inf 3 연속 | 학습 스크립트 | 멈춤 (`nan`, 종료 코드 4) |
| 스왑 +256MB | 스크립트 · 감싸개 | 검사점 저장 뒤 멈춤 |
| 게임 서버가 새로 뜸 · 떠 있는데 가용 < 10GB | 감싸개 (컨테이너 안에선 못 봄) | 멈춤 파일 → 스크립트가 검사점 저장 뒤 멈춤 · 유예 120초 넘으면 죽임 |
| 심장 박동 60초 끊김 | 감싸개 | 바로 죽임 · 안 죽으면 `GPU_STUCK` 줄 (재부팅은 사람) |
| large 무너짐 (2,000 스텝 dev < 0.50) | 스크립트 | 씨앗 +1 로 두 번 → lr 5e-6 한 번 |

## 시험

```
python -m unittest discover -s tests          # pytest 로도 돈다
```

torch · transformers 가 없으면 `--smoke` 와 학습 조각 시험은 건너뛴다. `tokenizers` 가 없으면 쌍 인코딩 시험도 건너뛴다.

메인 PC 에서는 onnxruntime · tokenizers 가 든 venv(`venv-serve`)로 돌리면 143 판 중 건너뜀 16 이다 (2026-10-10 판정 넷 확장 뒤). `JUDGEMODEL_TEST_MODEL=<모델 폴더>` 를 주면 실제 모델 시험까지 돌아 건너뜀이 하나 준다. 메인 PC 에는 torch 가 든 venv 가 없어 학습 조각 시험은 미니PC 에서만 돈다.

## 라이선스

이 툴의 코드는 MIT (`LICENSE`). 학습 자료·모델 가중치의 출처와 라이선스 표는 `LICENSES.md`. 요점 : 한·영 판 몸통 `xlm-roberta-large` 는 MIT, 학습 자료는 BY-SA(KLUE · KorNLI · SNLI 4.0 · MultiNLI 일부 3.0)라 기본 모델 가중치를 내놓을 때 **CC BY-SA 4.0** 으로 출처를 적는다. 비상업(NC) 자료는 쓰지 않는다 — `model.json` 의 `nc` 가 참이면 공개 금지.
