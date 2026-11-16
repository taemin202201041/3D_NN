"""Share of descriptive test-scene questions (scenes 2500-2999) that the program2 parser (fitted on scenes 0-1999) cannot read, per sub-type. read-only."""
import collections
import json

from qparser import TemplateParser

Q = "../data_large_v1/data_large_v1/cache/official_questions_train.json"
vs = json.load(open(Q, encoding="utf-8"))
fit = []
for v in vs:
    if 0 <= v["scene_index"] < 2000:
        for q in v["questions"]:
            fit.append((q["question_type"], q["question"], q["program"]))
parser = TemplateParser().fit(fit)
tot, miss, wrong = collections.Counter(), collections.Counter(), collections.Counter()
for v in vs:
    if 2500 <= v["scene_index"] < 3000:
        for q in v["questions"]:
            if q["question_type"] != "descriptive":
                continue
            k = q["question_subtype"]
            p = parser.parse("descriptive", q["question"])
            tot[k] += 1
            if p is None:
                miss[k] += 1
            elif p != q["program"]:
                wrong[k] += 1
print(f"{'sub-type':18s} {'n':>6s} {'unparsed':>9s} {'parsed but different from official':>36s}")
for k in tot:
    print(f"{k:18s} {tot[k]:6d} {miss[k] / tot[k]:9.3f} {wrong[k] / tot[k]:36.3f}")
n, m, w = sum(tot.values()), sum(miss.values()), sum(wrong.values())
print(f"{'ALL':18s} {n:6d} {m / n:9.3f} {w / n:36.3f}")
