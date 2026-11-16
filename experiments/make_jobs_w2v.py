"""jobs_w2v.txt = the e31_ann_w2v runs (annotation + word2vec mean, descriptive only, team split) that are not finished yet.
config/e31_ann_w2v.json is created if missing (copy of e30 with word2vec_mean, descriptive only)."""
import glob
import json
import os

if not os.path.exists("config/e31_ann_w2v.json"):
    c = json.load(open("config/e30_ann_structure.json", encoding="utf-8"))
    c["experiment"] = "e31_ann_w2v"
    c["description"] = "[진단용] 주석 입력 + word2vec 평균 질문. 팀 설정(서술형만, 학습 0-1999 / 검증 2000-2499 / 평가 2500-2999). e30(program 질문)과 같은 14가지 비교. 이동은 첫 에포크부터."
    st = c["set"]
    st["data.question_encoding"] = "word2vec_mean"
    st["data.question_types"] = ["descriptive"]
    for k in ("model.answer_mask", "train.select"):
        st.pop(k, None)
    json.dump(c, open("config/e31_ann_w2v.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
c = json.load(open("config/e31_ann_w2v.json", encoding="utf-8"))
names = [v["name"] for v in c["variants"]]
done = set()
for f in glob.glob("results/runs/e31_ann_w2v__*.json"):
    r = json.load(open(f, encoding="utf-8"))
    if "metrics" in r:
        parts = f.replace("\\", "/").split("__")
        done.add((parts[1], parts[2]))
jobs = [f"9 e31_ann_w2v {n} {s}" for s in (1, 2, 3) for n in names if (n, f"s{s}") not in done]
open("jobs_w2v.txt", "w").write("\n".join(jobs) + "\n")
print("e31 finished", len(done), "to run", len(jobs))
