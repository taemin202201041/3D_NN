"""GoogleNews word2vec (300-d, as in the CLEVRER paper) rows for every word of the question / choice texts, WITHOUT unpickling.

The 182 MB .model pickle is only *scanned* with pickletools.genops (an opcode reader that executes nothing); the order of its
unicode strings is the word -> row mapping. Vectors come from the 3.6 GB .vectors.npy via HTTP range requests (300 floats each).
The mapping is checked with king - man + woman ~ queen before anything is written.

Output (experiments/data/clevrer_paper/):  w2v_words.json (list of words)   w2v.npy float32 [n_words, 300]  (zero row = word not in GoogleNews)
Usage: python make_w2v.py [data_large_v1 dir]
"""
import ast
import json
import pickletools
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from questions import words  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data_large_v1" / "data_large_v1"
OUT = HERE / "data" / "clevrer_paper"
OUT.mkdir(parents=True, exist_ok=True)
REPO = "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/"
STR_OPS = {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"}

vocab = set()
for sp in ("train", "validation"):
    for v in json.loads((ROOT / "questions" / f"{sp}.json").read_text(encoding="utf-8")):
        for q in v["questions"]:
            vocab.update(words(q["question"]))
            for c in q.get("choices", []):
                vocab.update(words(c["choice"]))
vocab = sorted(vocab)
print("words in questions:", len(vocab))

mp = OUT / "_w2v.model"
if not mp.exists():
    urllib.request.urlretrieve(REPO + "word2vec-google-news-300.model", mp)
strs = []
with open(mp, "rb") as f:
    for op, arg, pos in pickletools.genops(f):
        if op.name in STR_OPS:
            strs.append(arg)
start = strs.index("</s>")
assert strs[start + 1:start + 3] == ["count", "index"] and strs[start + 3:start + 5] == ["in", "for"], strs[start:start + 6]
keys = ["</s>"] + strs[start + 3:]          # keys[i] = word of row i (validated below)
k2i = {}
for i, w in enumerate(keys):
    k2i.setdefault(w, i)


def rng(a, b):
    req = urllib.request.Request(REPO + "word2vec-google-news-300.model.vectors.npy", headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120).read()


head = rng(0, 255)
assert head[:6] == b"\x93NUMPY"
hl = int.from_bytes(head[8:10], "little") + 10
meta = ast.literal_eval(head[10:hl].decode("latin1").strip())
assert meta["descr"] == "<f4" and not meta["fortran_order"], meta
rows, dim = meta["shape"]
assert len(keys) >= rows, (len(keys), rows)


def vec(w):
    j = k2i[w]
    assert j < rows
    return np.frombuffer(rng(hl + j * dim * 4, hl + (j + 1) * dim * 4 - 1), "<f4")


unit = lambda v: v / np.linalg.norm(v)
k, m, w_, q = (unit(vec(x)) for x in ("king", "man", "woman", "queen"))
target = unit(k - m + w_)
sims = {c: float(target @ unit(vec(c))) for c in ("queen", "car", "table", "purple", "cube", "green")}
print("king-man+woman vs", {c: round(s, 3) for c, s in sims.items()})
assert sims["queen"] == max(sims.values()) and sims["queen"] > 0.5, "index mapping looks wrong, aborting"

W = np.zeros((len(vocab), dim), "float32")
missing = []
for i, w in enumerate(vocab):
    if w in k2i and k2i[w] < rows:
        W[i] = vec(w)
    else:
        missing.append(w)
np.save(OUT / "w2v.npy", W)
(OUT / "w2v_words.json").write_text(json.dumps(vocab), encoding="utf-8")
mp.unlink()
print("found", len(vocab) - len(missing), "missing", len(missing), missing)
