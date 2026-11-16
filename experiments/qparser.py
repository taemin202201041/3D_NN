"""program2: a question parser that needs NO learning and NO annotation - a template lookup built from the (question text, program) pairs of the TRAIN questions.

CLEVRER questions are generated from templates.  Colour / shape / material words are replaced by placeholders (<C> <S> <M>) in the text and in the program;
text-template -> program-template is stored.  A new question is parsed by looking up its text template and filling the words back.
Two ways to fill the placeholders (both are tried, the better one per question type first):
  by order  : the i-th placeholder of the program = the i-th attribute word of the text        (exact for explanatory / predictive / counterfactual)
  by index  : each program placeholder = which attribute word of the text (handles a repeated word) (better for descriptive)
Unknown template -> None (the question gets an empty program; only the question-type one-hot remains).
"""
from __future__ import annotations

import collections
import re

CLS = {}
for _c in ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]:
    CLS[_c] = "<C>"
for _s in ["cube", "sphere", "cylinder"]:
    CLS[_s] = "<S>"
for _m in ["rubber", "metal"]:
    CLS[_m] = "<M>"
HOLES = ("<C>", "<S>", "<M>")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


def _split(tokens):
    tpl, fill = [], []
    for t in tokens:
        if t in CLS:
            tpl.append(CLS[t]); fill.append(t)
        else:
            tpl.append(t)
    return tuple(tpl), fill


class TemplateParser:
    def __init__(self):
        self.by_order = collections.defaultdict(collections.Counter)
        self.by_index = collections.defaultdict(collections.Counter)

    def fit(self, pairs):
        """pairs: iterable of (question_type, text, program)"""
        for qtype, text, prog in pairs:
            tt, ft = _split(words(text))
            pt, fp = _split(prog)
            key = (qtype == "descriptive", tt)
            if fp == ft:
                self.by_order[key][pt] += 1
            try:
                self.by_index[key][(pt, tuple(ft.index(w) for w in fp))] += 1
            except ValueError:
                pass
        return self

    @staticmethod
    def _fill_order(pt, ft):
        it = iter(ft)
        out = []
        for p in pt:
            if p in HOLES:
                w = next(it, None)
                if w is None:
                    return None
                out.append(w)
            else:
                out.append(p)
        return out

    @staticmethod
    def _fill_index(pt, idx, ft):
        out, k = [], 0
        for p in pt:
            if p in HOLES:
                if k >= len(idx) or idx[k] >= len(ft):
                    return None
                out.append(ft[idx[k]]); k += 1
            else:
                out.append(p)
        return out

    def parse(self, qtype: str, text: str):
        tt, ft = _split(words(text))
        key = (qtype == "descriptive", tt)
        order = [("index", self.by_index), ("order", self.by_order)] if qtype == "descriptive" else [("order", self.by_order), ("index", self.by_index)]
        for kind, table in order:
            if table.get(key):                                           # (a failed fit attempt can leave an empty Counter behind)
                top = table[key].most_common(1)[0][0]
                res = self._fill_index(top[0], top[1], ft) if kind == "index" else self._fill_order(top, ft)
                if res is not None:
                    return res
        return None


def parse_videos(videos: list[dict], parser: TemplateParser):
    """returns a deep-enough copy of the official question entries whose `program` fields are PARSED from the text (official programs are dropped)."""
    out, miss, n = [], 0, 0
    for v in videos:
        qs = []
        for q in v["questions"]:
            q2 = {k: x for k, x in q.items() if k not in ("program", "choices")}
            p = parser.parse(q["question_type"], q["question"])
            n += 1; miss += p is None
            q2["program"] = p or []
            if "choices" in q:
                cs = []
                for c in q["choices"]:
                    c2 = {k: x for k, x in c.items() if k != "program"}
                    pc = parser.parse(q["question_type"], c["choice"])
                    n += 1; miss += pc is None
                    c2["program"] = pc or []
                    cs.append(c2)
                q2["choices"] = cs
            qs.append(q2)
        v2 = {k: x for k, x in v.items() if k != "questions"}
        v2["questions"] = qs
        out.append(v2)
    return out, miss, n
