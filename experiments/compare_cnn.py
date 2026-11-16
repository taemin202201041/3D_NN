"""CNN (ResNet-50) input, team setting (descriptive only): word2vec mean (e28b) vs embedding learned on this dataset (e34).
Accuracy (mean +- sd over seeds), descriptive sub-types, trainable parameters, wall-clock seconds of the runs (measured while other runs were also running). read-only."""
import glob
import json

import numpy as np

EXPS = [("w2v 평균", "e28b_team_r50w2v"), ("학습형 임베딩", "e34_cnn_embed")]
VARS = [("고정", "fixed"), ("position", "position"), ("swap", "swap"), ("both", "both"), ("relayout", "relayout"), ("LSTM 코어", "lstm"),
        ("선형 RNN", "linear_rnn"), ("패널티 없는 ReLU", "relu_nopen"), ("질문만", "question_only")]
SUB = [("개수", "desc_count"), ("존재", "desc_exist"), ("색", "desc_query_color"), ("모양", "desc_query_shape"), ("재질", "desc_query_material")]


def load(e, v):
    out = []
    for f in glob.glob(f"results/runs/{e}__{v}__s*.json"):
        r = json.load(open(f, encoding="utf-8"))
        if "metrics" in r:
            out.append(r)
    return out


def ms(x):
    x = [a for a in x if a is not None]
    return (np.mean(x), np.std(x, ddof=1) if len(x) > 1 else 0.0, len(x)) if x else (None, None, 0)


print("## 서술형 정확도 (평가 2500-2999, 시드 평균 ± 표준편차)\n")
print("| 모델 | " + " | ".join(n for n, _ in EXPS) + " | 차이 (학습형 - w2v) |")
print("|---|---|---|---|")
for vn, v in VARS:
    cells, means = [], []
    for _, e in EXPS:
        m, s, n = ms([r["metrics"]["test"]["descriptive_per_option"] for r in load(e, v)])
        cells.append("-" if m is None else f"{m:.3f} ± {s:.3f} (n={n})")
        means.append(m)
    d = "-" if None in means else f"{means[1] - means[0]:+.3f}"
    print(f"| {vn} | " + " | ".join(cells) + f" | {d} |")

for vn, v in [("고정", "fixed"), ("LSTM 코어", "lstm"), ("질문만", "question_only")]:
    print(f"\n## 하위 유형별 - {vn}\n")
    print("| 질문 방식 | " + " | ".join(n for n, _ in SUB) + " |")
    print("|---|" + "---|" * len(SUB))
    for en, e in EXPS:
        rs = load(e, v)
        print(f"| {en} | " + " | ".join(("-" if ms([r['metrics']['test'].get(k) for r in rs])[0] is None else f"{ms([r['metrics']['test'].get(k) for r in rs])[0]:.3f}") for _, k in SUB) + " |")

print("\n## 파라미터와 학습 시간 (고정 seRNN)\n")
print("| 질문 방식 | 학습 파라미터 | 학습 시간(초, 동시 실행 중 측정) | 최고 에포크 | 학습 정확도 | 평가 정확도 |")
print("|---|---|---|---|---|---|")
for en, e in EXPS:
    rs = load(e, "fixed")
    if rs:
        be = np.mean([r["metrics"]["best_epoch"] for r in rs])
        tr = np.mean([r["history"]["accuracy"][int(r["metrics"]["best_epoch"]) - 1] for r in rs])
        print(f"| {en} | {rs[0]['model']['trainable_params']:,} | {np.mean([r['seconds'] for r in rs]):.0f} | {be:.0f} | {tr:.3f} | {np.mean([r['metrics']['test']['acc_all'] for r in rs]):.3f} |")
