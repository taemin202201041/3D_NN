"""Template lookup parser v2: the program may repeat a word (e.g. the same colour twice), so each program placeholder is stored as an INDEX into the
text's list of attribute words.  Learned only from the train questions' (text, program) pairs; applied to validation. read-only."""
import collections
import json
import re

Q = "../data_large_v1/data_large_v1/cache/"
tr = json.load(open(Q + "official_questions_train.json", encoding="utf-8"))
va = json.load(open(Q + "official_questions_validation.json", encoding="utf-8"))
CLS = {}
for c in ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]:
    CLS[c] = "<C>"
for s in ["cube", "sphere", "cylinder"]:
    CLS[s] = "<S>"
for m in ["rubber", "metal"]:
    CLS[m] = "<M>"


def words(t):
    return re.findall(r"[a-z]+|\d+", t.lower())


def split(tokens):
    tpl, fill = [], []
    for t in tokens:
        if t in CLS:
            tpl.append(CLS[t]); fill.append(t)
        else:
            tpl.append(t)
    return tuple(tpl), fill


def items(videos):
    for v in videos:
        for q in v["questions"]:
            yield q["question_type"], "q", q["question"], q["program"]
            for c in q.get("choices", []):
                yield q["question_type"], "c", c["choice"], c["program"]


table = collections.defaultdict(collections.Counter)
skipped = 0
for typ, kind, text, prog in items(tr):
    tt, ft = split(words(text))
    pt, fp = split(prog)
    try:
        idx = tuple(ft.index(w) for w in fp)                  # each program attribute word = which attribute word of the text
    except ValueError:
        skipped += 1
        continue
    table[tt][(pt, idx)] += 1
print("train pairs not expressible by index:", skipped, " templates:", len(table))

stat = collections.defaultdict(lambda: [0, 0, 0])
for typ, kind, text, prog in items(va):
    tt, ft = split(words(text))
    s = stat[(typ, kind)]
    s[0] += 1
    if tt in table:
        s[1] += 1
        (pt, idx), _ = table[tt].most_common(1)[0]
        it = iter(idx)
        pred = [ft[next(it)] if (p in ("<C>", "<S>", "<M>")) and next_ok(ft, it) else p for p in pt] if False else None
        out, ok = [], True
        k = 0
        for p in pt:
            if p in ("<C>", "<S>", "<M>"):
                if k >= len(idx) or idx[k] >= len(ft):
                    ok = False
                    break
                out.append(ft[idx[k]]); k += 1
            else:
                out.append(p)
        if ok and out == list(prog):
            s[2] += 1
print(f"{'type':16s} {'kind':4s} {'n':>7s} {'seen':>8s} {'exact':>8s}")
tot = [0, 0, 0]
for (typ, kind), (n, seen, ex) in sorted(stat.items()):
    print(f"{typ:16s} {kind:4s} {n:7d} {seen / n:8.3f} {ex / n:8.3f}")
    tot = [tot[0] + n, tot[1] + seen, tot[2] + ex]
print(f"{'ALL':21s} {tot[0]:7d} {tot[1] / tot[0]:8.3f} {tot[2] / tot[0]:8.3f}")
