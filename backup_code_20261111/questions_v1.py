"""Question / answer encoding.

Question input: bag of words of the question TEXT (data_large_v1 has no programs).
    [question word counts (V)] + [choice word counts (V)] + [question type one-hot (4)]
Words not in the training vocabulary are ignored.

Answer output: 23 classes. Multiple-choice questions become one sample per choice (correct/wrong).
Answer groups say which classes are possible for a sample (used by model.answer_mask):
    0 count -> 0..5 | 1 exist -> yes/no | 2 color | 3 shape | 4 material | 5 choice -> correct/wrong
"""

from __future__ import annotations

import re

import numpy as np

from perception import COLORS, MATERIALS, SHAPES

QTYPES = ["descriptive", "explanatory", "predictive", "counterfactual"]
SUBTYPES = ["count", "exist", "query_color", "query_shape", "query_material"]
CLASSES = ["yes", "no", "0", "1", "2", "3", "4", "5"] + SHAPES + COLORS + MATERIALS + ["correct", "wrong"]
N_CLASSES = len(CLASSES)  # 23
GROUPS = ["count", "exist", "color", "shape", "material", "choice"]
_GROUP_CLASSES = {
    "count": ["0", "1", "2", "3", "4", "5"],
    "exist": ["yes", "no"],
    "color": COLORS,
    "shape": SHAPES,
    "material": MATERIALS,
    "choice": ["correct", "wrong"],
}
ALLOWED = np.zeros((len(GROUPS), N_CLASSES), np.float32)
for _g, _names in _GROUP_CLASSES.items():
    for _n in _names:
        ALLOWED[GROUPS.index(_g), CLASSES.index(_n)] = 1.0
_SUB2GROUP = {"count": 0, "exist": 1, "query_color": 2, "query_shape": 3, "query_material": 4}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


def build_vocab(question_files: list[list[dict]]) -> list[str]:
    vocab = set()
    for videos in question_files:
        for v in videos:
            for q in v["questions"]:
                vocab.update(words(q["question"]))
                for c in q.get("choices", []):
                    vocab.update(words(c["choice"]))
    return sorted(vocab)


def encode_videos(videos: list[dict], vocab: list[str], row_of: dict[int, int], w2v: dict | None = None) -> dict[str, np.ndarray]:
    """All samples of the given question-file entries as flat arrays."""
    tok = {t: i for i, t in enumerate(vocab)}
    v_n = len(vocab)
    cols = {k: [] for k in ["scene", "q", "y", "qtype", "subtype", "agroup", "qkey", "choice"]}
    dim = len(next(iter(w2v.values()))) if w2v else 0

    def mean_vec(ws):                                  # paper: average of pretrained word vectors (words without a vector are skipped)
        vs = [w2v[t] for t in ws if t in w2v]
        return np.mean(vs, 0).astype(np.float32) if vs else np.zeros(dim, np.float32)

    for v in videos:
        s = row_of[v["scene_index"]]                    # row in the feature table (0..2999)
        sid = v["scene_index"]
        for q in v["questions"]:
            base = np.zeros(2 * v_n + len(QTYPES), np.uint8) if not w2v else None
            if not w2v:
                for t in words(q["question"]):
                    if t in tok:
                        base[tok[t]] += 1
            ti = QTYPES.index(q["question_type"])
            if not w2v:
                base[2 * v_n + ti] = 1
            qkey = sid * 1000 + q["question_id"]
            if ti == 0:
                rows = [(mean_vec(words(q["question"])) if w2v else base, CLASSES.index(q["answer"]), SUBTYPES.index(q["question_subtype"]),
                         _SUB2GROUP[q["question_subtype"]], -1)]
            else:
                rows = []
                for c in q["choices"]:
                    if w2v:
                        b = mean_vec(words(q["question"]) + words(c["choice"]))      # one sample per (question, choice) pair
                    else:
                        b = base.copy()
                        for t in words(c["choice"]):
                            if t in tok:
                                b[v_n + tok[t]] += 1
                    rows.append((b, CLASSES.index(c["answer"]), -1, 5, c["choice_id"]))
            for b, y, sub, grp, ch in rows:
                cols["scene"].append(s); cols["q"].append(b); cols["y"].append(y)
                cols["qtype"].append(ti); cols["subtype"].append(sub); cols["agroup"].append(grp)
                cols["qkey"].append(qkey); cols["choice"].append(ch)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["q"] = np.stack(cols["q"]).astype(np.float32 if w2v else np.uint8)
    for k in ["scene", "y", "qtype", "subtype", "agroup", "choice"]:
        out[k] = out[k].astype(np.int32)
    out["qkey"] = out["qkey"].astype(np.int64)
    return out
