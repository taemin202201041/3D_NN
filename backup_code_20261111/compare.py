"""Compare an experiment with the baselines (e03_baselines: lstm, linear_rnn) from results/ledger.jsonl.

    python compare.py e02_requested            # every condition of that experiment + the baselines, seed mean +- sd
    python compare.py e05_mine --baseline e03_baselines

Only runs with the same data splits / question types / epochs / batch as the baselines are comparable; a mismatch is flagged.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ledger = Path(__file__).resolve().parent / "results" / "ledger.jsonl"
exp = sys.argv[1] if len(sys.argv) > 1 else "e02_requested"
base = sys.argv[sys.argv.index("--baseline") + 1] if "--baseline" in sys.argv else "e03_baselines"
rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("status") == "ok" and r["experiment"] in (exp, base)]
latest = {r["run_id"]: r for r in rows}


def key(r):
    c = r["config"]
    return json.dumps([c["data"]["train_scenes"], c["data"]["val_scenes"], c["data"]["test_scenes"], c["data"]["question_types"],
                       c["train"]["epochs"], c["train"]["batch"]], sort_keys=True)


groups = defaultdict(list)
for r in latest.values():
    groups[(r["experiment"], r["name"] if "name" in r else r["run_id"].split("__")[1])].append(r)
bkeys = {key(r) for (e, _), rs in groups.items() if e == base for r in rs}
ms = lambda a: f"{np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}" if len(a) > 1 else f"{a[0]:.4f}"
print(f"{'condition':26s} {'n':>2s} {'test acc':>16s} {'val acc':>16s} {'params':>8s} {'wiring':>8s} {'Q':>6s} {'sigma':>6s}  same-setting-as-baseline")
for (e, name), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != base, kv[0])):
    t = [r["metrics"]["test"]["acc_all"] for r in rs]
    v = [r["metrics"]["val"]["acc_all"] for r in rs]
    st = lambda k: np.mean([r["structure"].get(k, np.nan) for r in rs])
    ok = "yes" if all(key(r) in bkeys for r in rs) else "NO"
    tag = "[base] " if e == base else ""
    print(f"{tag + name:26s} {len(rs):2d} {ms(t):>16s} {ms(v):>16s} {rs[0]['model']['trainable_params']:8d} "
          f"{st('wiring_cost'):8.1f} {st('Q'):6.3f} {st('sigma'):6.2f}  {ok}")
