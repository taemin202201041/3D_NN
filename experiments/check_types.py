"""Which question types does each experiment train / test on?  (from the saved run configs and the test sample counts). read-only."""
import collections
import glob
import json

seen = {}
for f in sorted(glob.glob("results/runs/*.json")):
    r = json.load(open(f, encoding="utf-8"))
    if "metrics" not in r or not r["metrics"].get("test"):
        continue
    e = r["experiment"]
    if e in seen:
        continue
    t = r["config"]["data"]["question_types"]
    seen[e] = (t, r["metrics"]["test"].get("n_samples"), r["config"]["data"]["train_scenes"])
for e in ["e24_controls", "e25_scale", "e28a_team_r18bow", "e28b_team_r50w2v", "e30_ann_structure", "e31_ann_w2v", "e32_ann_prog2", "e33_ann_embed", "e34_cnn_embed", "e35_slots"]:
    if e in seen:
        t, n, tr = seen[e]
        print(f"{e:20s} types={'+'.join(x[:4] for x in t):26s} test samples={n}  train scenes={tr}")
