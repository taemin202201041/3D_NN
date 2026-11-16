"""Remove the files and the ledger line of one smoke run (1-epoch check). Usage: python clean_smoke.py <experiment prefix>  (only my own check runs)"""
import glob
import json
import os
import sys

exp = sys.argv[1]
for f in glob.glob(f"results/runs/{exp}__*.json"):
    r = json.load(open(f, encoding="utf-8"))
    rid = r["run_id"]
    for p in (f, f"results/coords/{rid}.npy", f"results/weights/{rid}.weights.pt"):
        if os.path.exists(p):
            os.remove(p)
            print("removed", p)
lines = [l for l in open("results/ledger.jsonl", encoding="utf-8") if exp not in l]
open("results/ledger.jsonl", "w", encoding="utf-8").writelines(lines)
print("ledger lines left", len(lines))
