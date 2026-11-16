# CLEVRER seRNN — 동적 모델 (PyTorch)

팀 `동적_모델/` 폴더(TensorFlow)를 PyTorch로 옮긴 것이다. 입력 처리·질문 인코딩·채점·결과 기록은 팀 코드와 같은 형식이고,
**바꿔 가며 연구하는 부분은 `rnn.py` 한 파일**이다.

| 파일 | 역할 | 손대나? |
|---|---|---|
| `perception.py` | 영상 → 물체 6칸 × 37특징 × 25 frame (CLEVRER 주석 사용) | 아니오 |
| `questions.py` | 질문 → program 단어 개수, 정답 23종 | 아니오 |
| `dataset.py` | 캐시 생성·불러오기, 배치 | 아니오 |
| `model.py` | 입력 잡음, 읽기층(23칸), 답 후보 제한, 공간 벌점 연결 | 아니오 |
| **`rnn.py`** | **순환층(RecurrentCore) + 공간 벌점 `penalty`** | **여기만** |
| `layout.py` | 3D 격자에서 출발하는 좌표: position / swap / both / relayout | 필요할 때 |
| `topology.py` | 연결 마스크: full, gateway, bridges, **A/B2/B1 × full/rand** (`gateway_structure.py`) | 필요할 때 |
| `metrics.py`, `results.py` | 채점, 구조 지표, JSON·JSONL | 아니오 |
| `run.py`, `config/` | 실험 실행 | 설정만 |

## 데이터 (직접 넣는다)
`config/base.json`의 `paths.clevrer_root`(기본 `data/clevrer/`) 아래에 다음을 둔다.
`questions/train.json`, `questions/validation.json`, `annotations/annotation_00000-01000/ … annotation_14000-15000/`.
분할은 영상 번호 기준 **학습 0–1999 / 검증 2000–2499 / 평가 2500–2999**다 (`data.*_scenes`). 검증으로 epoch를 고르고, 평가는 실행마다 한 번 채점한다.

## 실행
```bash
pip install torch numpy scipy
python run.py --exp config/e00_smoke.json                 # 점검 (장면 2%, 2 epoch)
python run.py --exp config/e01_example_dynamic.json       # position / swap / relayout × seed 3개
python run.py --set model.topology=A_full --set layout.dynamic.method=swap --name a_swap   # 한 번만
python run.py --overview                                  # results/RESULTS.md 다시 만들기
```
- 결과: `results/runs/<run_id>.json`(1회 전체), `results/ledger.jsonl`(한 줄씩 누적), `results/weights/*.pt`, `results/coords/*.npy`(최종 좌표).
- 값 바꾸기: 점으로 이은 키 (`layout.dynamic.method`, `model.topology`, `reg.gamma`, `prune.method` …). 전체 목록은 팀 README 4장과 같다.

## 동적 ↔ 정적 사이 (swap)
- 격자에서 출발해 학습으로 구조를 만든다: `layout.dynamic.method=position` (또는 `swap`, `both`).
- 이상적인 구조(`model.topology=A_full` 등)를 먼저 고정하고 그 위에서 자리만 바꾼다: `model.topology=A_full` + `layout.dynamic.method=swap` 또는 `relayout`.
- 구조 마스크의 모듈 구분은 뉴런 번호(출발 격자 위치) 기준이라, 자리 교환 뒤에도 연결은 뉴런에 붙어 있다.

## `rnn.py` 규칙
`RecurrentCore(cfg, d_in, seed, in_mask)`: `core(x[B,T,d_in]) → h_T[B,units]`, `core.effective_recurrent()`(방향 있는 i→j 행렬), `core.rec_mask`(0/1 버퍼, run.py가 채움).
이 세 가지만 지키면 앞뒤는 그대로 돈다.
