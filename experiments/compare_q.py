"""Compare the three question encodings on the annotation input, team setting (descriptive only):
e31_ann_w2v (word2vec mean, general dictionary) / e33_ann_embed (embedding learned on this dataset) / e32_ann_prog2 (parser program).
Prints markdown tables: overall + descriptive sub-types per variant, plus per-variant differences.  read-only."""
import glob
import json

import numpy as np

EXPS = [("w2v 평균", "e31_ann_w2v"), ("학습형 임베딩", "e33_ann_embed"), ("program2", "e32_ann_prog2")]
SUB = [("개수", "desc_count"), ("존재", "desc_exist"), ("색", "desc_query_color"), ("모양", "desc_query_shape"), ("재질", "desc_query_material")]
VARS = [("고정", "fixed_full"), ("position", "position_full"), ("swap", "swap_full"), ("relayout", "relayout_full"),
        ("고정 + B1", "fixed_B1"), ("position + B1", "position_B1"), ("LSTM", "lstm"), ("선형 RNN", "linear_rnn"),
        ("패널티 없는 ReLU", "relu_nopen"), ("질문만", "question_only")]


def load(exp, var):
    rs = []
    for f in glob.glob(f"results/runs/{exp}__{var}__s*.json"):
        r = json.load(open(f, encoding="utf-8"))
        if "metrics" in r:
            rs.append(r["metrics"]["test"])
    return rs


def ms(rs, key):
    x = [r[key] for r in rs if r.get(key) is not None]
    return (np.mean(x), np.std(x, ddof=1) if len(x) > 1 else 0.0, len(x)) if x else (None, None, 0)


print("## 서술형 정확도 (평가 2500-2999, 시드 평균 ± 표준편차)\n")
print("| 모델 | " + " | ".join(n for n, _ in EXPS) + " |")
print("|---|" + "---|" * len(EXPS))
for vn, v in VARS:
    cells = []
    for _, e in EXPS:
        m, s, n = ms(load(e, v), "descriptive_per_option")
        cells.append("-" if m is None else f"{m:.3f} ± {s:.3f} (n={n})")
    print(f"| {vn} | " + " | ".join(cells) + " |")

for vn, v in [("고정", "fixed_full"), ("position", "position_full"), ("질문만", "question_only")]:
    print(f"\n## 하위 유형별 - {vn}\n")
    print("| 질문 방식 | " + " | ".join(n for n, _ in SUB) + " |")
    print("|---|" + "---|" * len(SUB))
    for en, e in EXPS:
        rs = load(e, v)
        print(f"| {en} | " + " | ".join(("-" if ms(rs, k)[0] is None else f"{ms(rs, k)[0]:.3f}") for _, k in SUB) + " |")
