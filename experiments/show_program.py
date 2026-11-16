"""Print one descriptive and one explanatory question with its `program` from the official question file. read-only."""
import json

qs = json.load(open("../data_large_v1/data_large_v1/cache/official_questions_train.json", encoding="utf-8"))
seen = set()
for v in qs[:200]:
    for q in v["questions"]:
        t = q["question_type"]
        if t in seen or t not in ("descriptive", "explanatory"):
            continue
        seen.add(t)
        print("TYPE:", t)
        print("question:", q["question"])
        print("program :", q["program"])
        if "choices" in q:
            c = q["choices"][0]
            print("choice  :", c["choice"], "->", c["program"])
        print()
    if len(seen) == 2:
        break
