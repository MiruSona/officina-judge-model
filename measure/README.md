# measure — 자 (짝 비교 · 관문)

규약은 `protocol-g1.json` 하나다. 이 파일의 sha256 이 모든 결과에 `protocol_sha256` 으로 붙는다. 고치려면 새 판 파일(`protocol-g2.json`)을 만든다. 설계 `../Docs/Design/2026-10-11-d자료층재구축설계.md` 2-3 · 6-2.

1. **돌리기 (미니PC)** : `run/seed3.sh <조건> -- <train_nli 인자> --deterministic` 를 `G1_SEEDS="1 2 3 4 5"` 로 → 판 `g1-<조건>-s1`~`s5`. 규약 씨앗은 5 이고 학습은 결정 모드다(45-라 : 같은 씨앗이 비트 단위로 재현됨 · 학습 약 +20% 느림). 판마다 `eval-dev-cal` · `eval-dev-cmp` · 곁 자 평가, `calibration.json`(dev-cal 로), `eval-side.txt` 가 생긴다. 쌍을 더해 견줄 땐 `--replay-n` 을 기준 판 값에 묶는다.
   - **평가만 다시** : 학습만 한 판(`G1_TRAIN_ONLY=1`)은 `G1_EVAL_ONLY=1 run/seed3.sh <조건> --` 로 `runs/<판>/ckpt-best` 를 그대로 잰다 (재실행 판은 `G1_SEEDS=1 G1_SUFFIX=r` 를 같이).
   - **기준선 자 상태** : `python -I measure/baseline.py --protocol … --set <dev-cmp-g1.jsonl> --runs B0=<판1>/eval-dev-cmp,… --rerun <판1r>/eval-dev-cmp --out b.json --md b.md` → 판별 지표 · 씨앗 짝 반폭(3%p 가리나) · 재실행 차 · 곁 자. 거부 규칙은 compare 와 같다.
2. **짝 비교** : `python -I measure/compare.py --protocol measure/protocol-g1.json --set <dev-cmp-g1.jsonl> --base B0=<판1>/eval-dev-cmp,<판2>/…,<판3>/… --new B1=… --by domain,lang --out c.json --md c.md` → 먹힘 · 같다 · 지킴 깨짐 · 보류.
3. **릴리스 관문** : `python -I measure/gates.py release --protocol … --set <blind-g1.jsonl> --pred <판>/eval-blind --calibration <판>/calibration.json --ref-pred <v1.0.0 pred> --p95-ms <값>`. `gates.py pair …` 는 compare 와 같다.
   - **사전 등록** : `measure/prereg.py` 도 4칸 모델의 자를 `--protocol`(기본 `measure/protocol-g1.json`)의 `release_gate` 에서 읽어 `criteria` · `release_gate` · `protocol_sha256` 으로 박제한다. 코드에 자 숫자를 두지 않는다. fold40 을 관문에 넣으려면 새 규약에 `fold_wrong_max` 를 넣는다(g1 에는 없음).
4. **거부(exit 2)** : 판의 `summary.json` 에 `set_sha256` · `protocol_sha256` 이 없거나 다를 때, calibration 이 잴 셋으로 골라졌을 때(`calib_set_sha256` · 옛 `fitted_on.sha256`). 옛 판은 `eval_nli.py --protocol` 로 다시 잰다.
5. **옛 고리 관문** : `python -I measure/gates.py legacy <diag.json>…` — dev 8 + 곁 3(옆 `eval-side.txt`).
6. 시험 : `python -m unittest discover -s tests -p "test_compare.py"` · `"test_measure_gates.py"`.
