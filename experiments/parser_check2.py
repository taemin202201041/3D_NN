"""Can a rule-free 'template lookup' reproduce the official program from the question text alone?  (no learning, no annotation, read-only)
Idea: CLEVRER questions are generated from templates.  Replace colour / shape / material words by placeholders in the text and in the program,
learn text-template -> program-template from the TRAIN questions, apply it to VALIDATION questions and fill the words back in order of appearance.
Reports: share of validation questions (and choices) whose template was seen, and exact-match of the rebuilt program."""
import collections
import json
import re

Q = "../data_large_v1/data_large_v1/cache/"
tr = json.load(open(Q + "official_questions_train.json", encoding="utf-8"))
va = json.load(open(Q + "official_questions_validation.json", encoding="utf-8"))
COLORS = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
MATS = ["rubber", "metal"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "last"]
CLASS = {**{c: "<C>" for c in COLORS}, **{s: "<S>" for s in SHAPES}, **{m: "<M>" for m in MATS}, **{o: "<O>" for o in ORD}}


def words(text):
    return re.findall(r"[a-z]+|\d+", text.lower())


def abstract(tokens):
    """-> (template tokens, list of filled words in order)"""
    out, fill = [], []
    for t in tokens:
        if t in CLASS:
            out.append(CLASS[t]); fill.append(t)
        else:
            out.append(t)
    return tuple(out), fill


def refill(template, fill):
    it = iter(fill)
    res = []
    for t in template:
        if t in ("<C>", "<S>", "<M>", "<O>"):
            res.append(next(it, "?"))
        else:
            res.append(t)
    return res


def items(videos):
    for v in videos:
        for q in v["questions"]:
            yield q["question_type"], "q", q["question"], q["program"]
            for c in q.get("choices", []):
                yield q["question_type"], "c", c["choice"], c["program"]


table = collections.defaultdict(collections.Counter)
for typ, kind, text, prog in items(tr):
    tt, fill_t = abstract(words(text))
    pt, fill_p = abstract(prog)
    if fill_t == fill_p or sorted(fill_t) == sorted(fill_p):          # same words, maybe other order: keep the order of the program's own fill
        table[tt][(pt, tuple(fill_p) == tuple(fill_t))] += 1
print("distinct text templates in train:", len(table))

stat = collections.defaultdict(lambda: [0, 0, 0])   # type/kind -> [n, seen, exact]
for typ, kind, text, prog in items(va):
    tt, fill_t = abstract(words(text))
    s = stat[(typ, kind)]
    s[0] += 1
    if tt in table:
        s[1] += 1
        (pt, same_order), _ = table[tt].most_common(1)[0]
        pred = refill(pt, fill_t)
        if pred == list(prog):
            s[2] += 1
print(f"{'type':16s} {'kind':4s} {'n':>7s} {'template seen':>14s} {'exact program':>14s}")
tot = [0, 0, 0]
for (typ, kind), (n, seen, ex) in sorted(stat.items()):
    print(f"{typ:16s} {kind:4s} {n:7d} {seen / n:14.3f} {ex / n:14.3f}")
    tot = [tot[0] + n, tot[1] + seen, tot[2] + ex]
print(f"{'ALL':21s} {tot[0]:7d} {tot[1] / tot[0]:14.3f} {tot[2] / tot[0]:14.3f}")
