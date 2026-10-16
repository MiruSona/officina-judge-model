# JudgeModel

> **English summary.** JudgeModel is a Korean–English NLI (natural language inference) classifier.
> Given an *evidence* text and a *claim*, it returns calibrated probabilities that the evidence
> **supports** the claim, **contradicts** it, is **unrelated**, or (in 4-label models) is **related but not decisive**.
> The repository holds the training pipeline (XLM-RoBERTa large fine-tuning), pre-registered measurement tools,
> ONNX export, and a small local HTTP server (`serve.py`, `POST /judge`). Code is MIT.
> Public weights are not released yet; when they are, they will be trained only on public data and released under CC BY-SA 4.0.
> Documentation below is in Korean.

## 1. 소개

(근거, 주장) 한 쌍을 읽고 「근거가 주장을 **지지**하나 · **반대**하나 · **무관**한가」를 확률로 내는 분류기다.
네 칸 모델은 무관을 둘로 나눠 「같은 주제지만 판가름은 안 되는 **비슷한 계열**」 칸을 더 낸다.
한국어와 영어 두 말을 함께 배운다.

이런 데 쓴다.

- 메모 · 설계 문서 · 기록 안에서 「새 글이 옛 글과 어긋나나」를 자동으로 걸러 낼 때
- 요약 · 답변이 근거 글에 실제로 있는 말인지 확인할 때

학습 · 측정 · 서빙(HTTP)을 이 저장소가 다 갖는다. 다른 프로그램은 HTTP 응답 꼴만 따르면 된다.

```
POST /judge {"evidence": "회의는 3시로 미뤄졌다.", "claim": "회의는 2시에 열린다."}
→ 200 {"a": 0.03, "b": 0.94, "c": 0.02, "d": 0.01, "model": "1a2b3c4d", "ms": 41.7}
```

`a` 지지 · `b` 반대 · `c` 무관 · `d` 비슷한 계열. 넷의 합은 1 이다. (숫자는 꼴 보기용 예시다.)

## 2. 빠른 시작

### 2-1. 설치

서빙에는 `onnxruntime` · `tokenizers` · `numpy` 셋만 든다.

```
python -m venv venv-serve
venv-serve/Scripts/pip install -r requirements.txt      # Linux · macOS 는 venv-serve/bin/pip
```

학습까지 하려면 `requirements-train.txt`(torch · transformers · onnx 등)도 깐다.

### 2-2. 가중치 받기

**공개 가중치는 준비 중이다 — 올라오면 여기에 내려받는 명령을 적는다.**

모델 폴더는 이렇게 생겼다. 직접 학습했다면 `export_onnx.py` 가 이 꼴로 낸다(5절).

| 파일 | 무엇 |
| --- | --- |
| `model.onnx` | 계산 그래프 |
| `model.onnx.data` | 가중치. 몸통이 2GB 를 넘으면(XLM-R large) 그래프와 따로 나온다. **늘 `model.onnx` 와 같은 폴더에 같이 둔다** |
| `tokenizer/tokenizer.json` | 토크나이저 |
| `model.json` | 이름 · 칸 · 최대 길이 · 학습 자료와 라이선스 · 파일 SHA · 추천 문턱 |
| `calibration.json` | 온도 보정 값 |

`model.json` 의 `onnx_sha256` 은 그래프만, `onnx_data_sha256` 은 가중치 파일을 가리킨다.

### 2-3. 띄우기

```
python -I serve.py --model ~/.judgemodel/models/<모델 이름> --port 8092 --threads 4
```

- 모델 자리는 `--model` 또는 환경 변수 `JUDGEMODEL_MODEL_DIR`.
- `--backend torch` 는 물러설 길이다 (`export_onnx.py --with-torch` 로 `torch/` 를 같이 내보냈을 때만).
- Windows 에서는 `run/serve.ps1` 로 띄워도 된다. 창 없이 뜨고, 그 포트에 이미 떠 있으면 `/health` 만 찍는다. 로그는 `~/.judgemodel/logs/serve.{log,err}`.
  - `-Model` 은 꼭 준다. 이름이면 `~/.judgemodel/models/<이름>` 을, 경로(`\` 나 `/` 가 든 값)면 그 폴더를 그대로 쓴다. 포트는 `-Port`(기본 8092).
  - `-Python` 은 서빙 venv 의 python 이다. 안 주면 저장소 안 `venv-serve` 를 쓴다.
  - 기본은 `--compat3` 를 붙여 3칸 꼴로 답한다. `-NoCompat3` 를 주면 모델 칸 그대로(4칸) 답한다.
  - `-Register` 는 Windows 로그온 때 지금 준 인자 그대로 자동으로 뜨게 작업 스케줄러에 올린다(작업 이름 `JudgeModel-NLI`). `-Unregister` 로 내린다.

### 2-4. 물어보기

```
curl -s http://127.0.0.1:8092/health
curl -s -X POST http://127.0.0.1:8092/judge \
     -H "Content-Type: application/json" \
     -d '{"evidence": "The release was moved to Friday.", "claim": "The release ships on Monday."}'
```

## 3. API

서버는 **127.0.0.1 에만** 붙는다. 기본 포트 8092. 판정은 한 번에 하나씩 돈다.

### 3-1. `POST /judge`

요청 몸통은 JSON 이다.

| 칸 | 꼴 | 뜻 |
| --- | --- | --- |
| `evidence` | 글 (필수) | 근거 |
| `claim` | 글 (필수) | 주장 |
| `label_set` | `"nli3"` · `"nli4"` (선택) | 응답 칸 세트를 고른다. 아래 3-2 |

응답 칸.

| 칸 | 뜻 |
| --- | --- |
| `a` · `b` · `c` | 지지 · 반대 · 무관 확률 |
| `d` | 비슷한 계열 확률 (네 칸 모델만) |
| `model` | 가중치 SHA-256 앞 8자 |
| `ms` | 판정에 든 시간 (밀리초) |

- 확률은 온도를 건 뒤 값이다.
- 「확신해도 되나」 문턱은 서버가 걸지 않는다. 클라이언트가 건다. 출발값은 `model.json` 의 `scores.thresholds`
  (`nli_sure_support` · `nli_sure` · 네 칸이면 `nli_sure_related`)에 있다. 문턱 아래면 「보류」로 두고 사람에게 넘기는 것을 권한다.

### 3-2. `label_set` 고르기 (opt-in)

**`label_set` 을 안 실으면 응답은 위 꼴 그대로다.** 실으면 칸 세 개(`labels` · `model_version` · `label_set`)가 더 붙는다.

```
POST /judge {"evidence": "...", "claim": "...", "label_set": "nli4"}
→ 200 {"a", "b", "c", "d", "model", "ms",
       "labels": ["support", "contradict", "unrelated", "related"], "model_version": "1.1.0", "label_set": "nli4"}
```

| 요청 `label_set` | 네 칸 모델 | 세 칸 모델 · `--compat3` |
| --- | --- | --- |
| 없음 · `null` | `a b c d model ms` | `a b c model ms` |
| `"nli3"` | d 를 c 에 더한 세 칸 + 세 칸 · `label_set: "nli3"` | 세 칸 + 세 칸 · `"nli3"` |
| `"nli4"` | 네 칸 + 세 칸 · `"nli4"` | **세 칸 + 세 칸 · `"nli3"`** — 에러가 아니다. 클라이언트는 응답의 `label_set` 을 보고 c · d 를 못 가른다는 것을 안다 |
| 그 밖 | 400 `{"error": "unknown label_set"}` | 같음 |

- `model_version` 은 `model.json` 의 `model_version` 이다. 없으면 `model` 과 같은 값이다.
- 응답 꼴 판 번호는 `serve.py` 의 `API_VERSION`(지금 `1.1.0`)이다. 기본 응답을 네 칸으로 바꾸는 날이 2.0.0 이다.

### 3-3. `--compat3`

네 칸 모델을 띄우되 d 를 c 에 더해 세 칸 꼴로 답한다. 세 칸만 아는 옛 클라이언트에 네 칸 모델을 붙일 때 쓴다.

```
python -I serve.py --model <네 칸 모델 폴더> --port 8092 --compat3
```

### 3-4. `GET /health` · `GET /info`

```
GET /health → {"ok": true, "model": "<SHA 앞 8자>", "backend": "onnx", "name": "<모델 이름>",
               "letters": ["a","b","c","d"], "protocol": "nli-v2"}
GET /info   → {"api_version", "model_version", "model", "name", "protocol", "label_guide",
               "default_label_set", "label_sets": {"nli3": {letters, labels}, "nli4": {...}}, "id2label"}
```

- `/health` 의 `letters` · `protocol` 은 `model.json` 에 칸 정보가 있는 모델만 붙는다. `protocol` 은 세 칸 `nli-v1` · 네 칸 `nli-v2`.
- `/info` 는 이 서버가 답할 수 있는 칸 세트와 라벨 이름을 알린다.

### 3-5. 상한과 오류

| 경우 | 응답 |
| --- | --- |
| 몸통이 1MB 를 넘음 | 413 `{"error": "body size"}` |
| `Content-Length` 없음 | 400 |
| JSON 이 아님 | 400 `{"error": "bad json"}` |
| `evidence` · `claim` 이 글이 아님 | 400 |
| 모르는 `label_set` | 400 |
| 없는 경로 | 404 |

몸통을 덜 보내고 멈춘 연결은 10초 뒤 끊는다.

## 4. 모델

| 항목 | 값 |
| --- | --- |
| 몸통 | `FacebookAI/xlm-roberta-large` (MIT) |
| 쌍 인코딩 | `<s> 근거 </s></s> 주장 </s>` · 최대 320 토큰 · 넘치면 근거만 자른다 (`pairenc.py` 한 곳) |
| 칸 | 세 칸(지지 · 반대 · 무관) 또는 네 칸(+ 비슷한 계열) |
| 보정 | dev 셋에서 고른 온도 하나 (`train/calibrate.py`) |
| 실행 | ONNX Runtime CPU |

**라벨 기준.** 어떤 쌍을 무슨 칸으로 보나는 문서 하나가 정한다.
최신은 [`Docs/Guide/라벨기준support-v6.md`](Docs/Guide/라벨기준support-v6.md) 이고,
[`v4`](Docs/Guide/라벨기준support-v4.md) · [`v5`](Docs/Guide/라벨기준support-v5.md) 는 앞 판이다.
모델이 어느 기준으로 배웠는지는 `model.json` 의 `label_guide` 에 적힌다.

### 4-1. 모델 카드

**공개 가중치를 올릴 때 채운다.** 칸은 아래 넷이다.

| 칸 | 담을 것 |
| --- | --- |
| 학습 자료 | 원천 · 양 · 말 비율 · 라이선스 (`LICENSES.md` 와 같게) |
| 성적 | 사전 등록한 측정 셋에서 잰 값과 그 셋의 출처 |
| 한계 | 약한 자리 (숫자 바뀜 · 말 섞임 · 학습에 없는 분야 등) |
| 의도된 용도 | 쓸 자리와 쓰면 안 되는 자리 |

비공개 자료로 덧학습한 판의 성적은 여기 적지 않는다. 남이 같은 자료로 다시 잴 수 없어서다.

## 5. 학습 · 측정 재현

작업 자리는 `~/.judgemodel/` 아래(`hf/` 몸통 · `data/` 자료 · `runs/` 학습 판 · `export/` · `models/`)다.
학습 감싸개(`run/*.sh`)로 돌리는 학습 기계는 따로 `$JT_ROOT` 아래를 쓴다(5-1 끝).
가중치 · 자료 · 판 기록은 git 에 넣지 않는다. 명령마다 인자는 각 파일 머리 주석에 다 적혀 있다.

### 5-1. 학습 차례

1. **자료 준비** — `train/data_prep.py` 가 공개 NLI 원본(KLUE-NLI · KorNLI · MultiNLI · SNLI)을 공통 jsonl 로 바꾸고, 바퀴마다 표본을 뽑는다. **라벨 매핑은 여기 한 곳**에 있다. 비상업 자료는 경로 이름으로 거부한다.
   그 밖의 원천(위키 개정 이력 · 설계 제안서 · 이슈 등)은 `build/` 가 후보를 뽑고 라벨 묶음을 만든다(`build/README.md`).
2. **학습** — `train/train_nli.py`. 기반 학습과 덧학습을 같이 쓴다. `--grow-head 4` 로 세 칸 머리를 네 칸으로 늘린다. `--smoke` 는 내려받기 없이 작은 무작위 모델로 끝까지 도는지만 본다.
3. **평가** — `train/eval_nli.py` 가 쌍마다 로짓 · 확률을 남긴다.
4. **보정** — `train/calibrate.py` 가 dev 로짓으로 온도와 확신 문턱을 고른다 → `calibration.json`.
5. **내보내기** — `export_onnx.py` 가 서빙 모델 폴더(2-2 표)를 만든다. 기반 판은 `--body-name FacebookAI/xlm-roberta-large` 를 꼭 주고, 덧학습 판은 `--parent <기반 model.json>` 으로 몸통 정보를 물려받는다.
6. **일치 검사** — `check_onnx.py` 가 토큰 · 확률(torch 판과 ONNX 판 차 ≤ 1e-3) · p95 를 확인한다. 다 통과해야 그 모델을 쓴다. `--url` 을 주면 띄운 서버까지 검사한다.

**학습 감싸개.** GPU 기계에서 컨테이너로 돌리는 감싸개는 `run/podman-train.sh`(CPU 판 `run/cpu-train.sh`)다.
작업 뿌리는 `JT_ROOT`(기본 `~/judgemodel-work`)이고, 그 아래 `src/`(이 저장소 사본) · `hf/`(내려받은 몸통) · `runs/`(검사점) · `logs/`(로그 · 심장 박동 · 멈춤 파일)를 둔다.
기계마다 다른 값은 git 밖 `$JT_ROOT/env.sh` 에 적으면 감싸개가 읽는다 — `WATCH_PROC`(같은 기계에서 메모리를 나눠 쓰는 다른 GPU 작업의 실행 파일 이름, 비우면 그 검사를 건너뛴다) · `JT_VENV`(CPU 판이 쓸 venv, 기본 `$JT_ROOT/venv`).
멈추면 로그에 까닭이 남는다. 심장 박동 끊김만 바로 죽이고, 나머지는 검사점을 저장한 뒤 끝낸다.

| 까닭 | 언제 |
| --- | --- |
| `nan` | 손실 NaN · inf 가 3번 잇달아 나옴 |
| `swap` | 시작 때보다 스왑이 256MB 넘게 늚 |
| `stall` | 심장 박동이 60초 넘게 끊김 (시작 뒤 180초 안에 한 번도 없어도) |
| `watch_proc` | 학습 중에 `WATCH_PROC` 프로세스가 새로 뜸 |
| `memory` | `WATCH_PROC` 가 떠 있는 동안 가용 메모리가 10GB 아래로 내려감 |

그 밖의 도구.

| 도구 | 하는 일 |
| --- | --- |
| `train/split_pairs.py` | 쌍을 묶음 단위로 train / dev 가르기 (같은 글이 양쪽에 새지 않게) |
| `train/freeze.py` | 측정 셋을 SHA 로 얼리기 · 확인 |
| `train/soup.py` | 같은 시작에서 갈라진 검사점 가중치를 평균해 새 검사점 만들기 |
| `train/heads.py` | torch 없이 도는 칸 계산 (클래스 가중치 · 지표) |

### 5-2. 측정 (`measure/`)

측정은 **잰 뒤에 자를 옮기지 않게** 사전 등록부터 한다.

- `protocol-*.json` — 측정 규약. 셋 SHA · 통과 기준(`release_gate`)을 담고, 해시로 고정된 자 역할을 한다.
- `prereg.py` — 재기 직전에 모델 · 셋 · 문턱 · 온도 · 자를 `prereg.json` 에 박제한다. 이미 있으면 덮지 않는다.
- `run_measure.py` — 띄운 서버에 쌍마다 묻고 정확도 · 칸별 재현 · 확신 거짓 지지 · 보류 · p95 를 낸다. prereg 의 SHA 가 다르면 재지 않는다.
- `run_fold.py` — 네 칸을 세 칸으로 접었을 때 생기는 오판을 따로 잰다.
- `gates.py` · `compare.py` — 앞 판과 짝지어 견주고 통과 · 미달을 판정한다.

자세한 쓰임은 `measure/README.md`.

## 6. 저장소 구성

| 경로 | 하는 일 |
| --- | --- |
| `serve.py` | 판정 서버 |
| `pairenc.py` | 쌍 인코딩 한 자리. 학습 · 평가 · 서버가 같이 쓴다 |
| `export_onnx.py` · `check_onnx.py` | ONNX 내보내기 · 일치 검사 |
| `train/` | 자료 준비 · 학습 · 평가 · 보정 · 셋 얼리기 |
| `build/` | 공개 원천에서 쌍 후보 뽑기 · 라벨 묶음 · 셋 조립 |
| `measure/` | 사전 등록 · 측정 · 판정 규약 |
| `run/` | 학습 감싸개(podman · CPU) · Windows 서버 띄우기 |
| `tests/` | 내려받기 없이 도는 시험 |
| `Docs/Guide/` | 라벨 기준 · 학습 판 규칙 |
| `Docs/Research/` | 자료 원천 · 측정 방법 조사 |

## 7. 시험

```
python -I -m pytest tests -q              # pytest 가 깔려 있으면
python -I -m unittest discover -s tests   # 추가 패키지 없이
```

- torch · transformers 가 없으면 학습 조각 시험은 건너뛴다. `tokenizers` 가 없으면 쌍 인코딩 시험도 건너뛴다.
- `JUDGEMODEL_TEST_MODEL=<모델 폴더>` 를 주면 실제 모델로 도는 시험까지 돈다.

## 8. 라이선스

- 코드 : **MIT** (`LICENSE`).
- 학습 자료 · 몸통 모델의 출처와 라이선스 : [`LICENSES.md`](LICENSES.md).
- 배포하는 가중치는 공개 자료로만 학습한 판이며 **CC BY-SA 4.0** 으로 낸다. 비상업(NC) 자료는 쓰지 않는다.
