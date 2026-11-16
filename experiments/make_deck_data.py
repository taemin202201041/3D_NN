"""Collect the numbers for the presentation (deck_data.json). Reads finished runs only (results/runs). read-only otherwise.
Usage: python make_deck_data.py"""
import collections
import glob
import json

import numpy as np

import visualize as V

RUNS = {}
for f in glob.glob("results/runs/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    if "metrics" in r:
        RUNS.setdefault((r["experiment"], f.replace("\\", "/").split("__")[1]), []).append(r)


def ms(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)


def movement(r):
    try:
        run, c1, w = V.load_run(r["run_id"])
        c0 = V.grid_coords(r["config"])
        return float(np.linalg.norm(c1 - c0, axis=1).mean())
    except SystemExit:
        return None


out = {}
# 1. dynamic movement on the team setting (descriptive only, ResNet18 + bag of words, train 0-1999 / val 2000-2499 / test 2500-2999), e28a (open_epoch 1)
rows = []
for label, name in [("고정", "fixed"), ("position", "position"), ("swap", "swap"), ("both", "both"), ("relayout", "relayout"),
                    ("LSTM", "lstm"), ("선형 RNN", "linear_rnn"), ("패널티 없는 ReLU", "relu_nopen"), ("질문만 (영상 끔)", "question_only")]:
    rs = RUNS.get(("e28a_team_r18bow", name), [])
    st = lambda k: ms([(r.get("structure") or {}).get(k) for r in rs])[0]
    rows.append({"name": label, "acc": ms([r["metrics"]["test"]["acc_all"] for r in rs]), "move": ms([movement(r) for r in rs])[0] if name in ("position", "swap", "both", "relayout") else None,
                 "wire": st("wiring_cost"), "Q": st("Q"), "sigma": st("sigma"), "edges": st("edges_active_offdiag"),
                 "params": rs[0]["model"]["trainable_params"] if rs else None})
out["dyn_team"] = rows

# 2. structure masks + position movement (e02)
rows = []
for label, name in [("구조 없음 (전 결합)", "position_full"), ("A (게이트웨이 4)", "position_A_full"), ("B2 (게이트웨이 2)", "position_B2_full"), ("B1 (게이트웨이 1)", "position_B1_full")]:
    rs = RUNS.get(("e02_requested", name), [])
    st = lambda k: ms([(r.get("structure") or {}).get(k) for r in rs])[0]
    rows.append({"name": label, "Q": st("Q"), "sigma": st("sigma"), "edges": st("edges_active_offdiag"), "wire": st("wiring_cost"),
                 "acc": ms([r["metrics"]["test"]["acc_all"] for r in rs]), "params": rs[0]["model"]["trainable_params"] if rs else None})
out["structure"] = rows

# 3. CNN input: question only vs video, per descriptive sub-type (e28a: ResNet18 + bag of words)
sub = ["desc_count", "desc_exist", "desc_query_color", "desc_query_shape", "desc_query_material"]
rows = []
for label, name in [("질문만 (영상 끔)", "question_only"), ("영상 포함 (고정 seRNN)", "fixed"), ("영상 포함 (LSTM)", "lstm")]:
    rs = RUNS.get(("e28a_team_r18bow", name), [])
    rows.append({"name": label, "n": len(rs), "all": ms([r["metrics"]["test"]["acc_all"] for r in rs]), **{s: ms([r["metrics"]["test"].get(s) for r in rs])[0] for s in sub}})
out["cnn_subtype"] = rows
base = RUNS.get(("e28a_team_r18bow", "fixed"), [None])[0]
out["majority"] = (base or {}).get("metrics", {}).get("baselines")

# 4. e28 models on the team setting (CNN)
rows = []
for exp, tag in (("e28a_team_r18bow", "ResNet18 + 단어 묶음"), ("e28b_team_r50w2v", "ResNet-50 + word2vec")):
    for name in ["fixed", "position", "swap", "both", "relayout", "lstm", "linear_rnn", "relu_nopen", "question_only"]:
        rs = RUNS.get((exp, name), [])
        rows.append({"input": tag, "name": name, "acc": ms([r["metrics"]["test"]["acc_all"] for r in rs])})
out["e28"] = rows

# 5. e30 (annotation input, diagnostic)
rows = []
keys = ["acc_all", "descriptive_per_option", "explanatory_per_question", "predictive_per_question", "counterfactual_per_question"]
for (e, name), rs in sorted(RUNS.items()):
    if e != "e30_ann_structure":
        continue
    st = lambda k: ms([(r.get("structure") or {}).get(k) for r in rs])[0]
    rows.append({"name": name, "n": len(rs), **{k: ms([r["metrics"]["test"].get(k) for r in rs]) for k in keys},
                 "Q": st("Q"), "sigma": st("sigma"), "edges": st("edges_active_offdiag"), "move": ms([movement(r) for r in rs])[0],
                 "params": rs[0]["model"]["trainable_params"]})
out["e30"] = rows

# 6. 15000-video annotation results (e25) and the 2000-video controls (e24)
rows = []
for e in ("e24_controls", "e25_scale"):
    for (ee, name), rs in sorted(RUNS.items()):
        if ee != e:
            continue
        rows.append({"exp": e, "name": name, "n": len(rs), **{k: ms([r["metrics"]["test"].get(k) for r in rs]) for k in keys}})
out["scale"] = rows
json.dump(out, open("deck_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(k, len(v) if isinstance(v, list) else v)
