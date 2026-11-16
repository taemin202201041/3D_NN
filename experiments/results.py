"""Result files: one JSON per run + an append-only JSONL ledger + a readable overview.

    results/runs/<run_id>.json   full record of one run
    results/ledger.jsonl         one line per run, never rewritten (history of every experiment)
    results/RESULTS.md           table rebuilt from the ledger: what changed -> what happened
    results/weights/<run_id>.weights.pt
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

SCHEMA = "clevrer-sernn/v1"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(base: dict, cfg: dict) -> dict:
    fb, fc = flatten(base), flatten(cfg)
    skip = ("experiment", "paths.", "name", "run_id", "note")
    return {k: v for k, v in fc.items() if fb.get(k) != v and not k.startswith(skip)}


def config_hash(cfg: dict) -> str:
    """Hash of everything that defines a run. Keys added to base.json later are dropped while they hold
    their 'off' value, so run_ids of earlier runs stay valid (ledger skip keeps working)."""
    import copy
    c = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in ("experiment", "paths", "name", "run_id", "note")}
    m = c.get("model", {})
    if m.get("topology", "full") == "full":
        m.pop("topology", None)
        m.pop("topology_cfg", None)
    if m.get("cell", "rnn") == "rnn":
        m.pop("cell", None)
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]


def git_commit(here: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
        return "unavailable: " + ("dubious ownership (git safe.directory)" if "dubious ownership" in r.stderr else "git error")
    except Exception as e:
        return f"unavailable: {e!r}"[:100]


def code_hash(here: Path) -> str:
    """sha1 of every .py file in the model folder: identifies the code even when git is unavailable."""
    h = hashlib.sha1()
    for f in sorted(here.glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:10]


def versions() -> dict:
    import numpy as np
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}


def save(results_dir: Path, record: dict) -> Path:
    runs = results_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(results_dir / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    overview(results_dir)
    return path


def done_ids(results_dir: Path) -> set:
    p = results_dir / "ledger.jsonl"
    if not p.exists():
        return set()
    ids = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok":
                ids.add(r["run_id"])
    return ids


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def overview(results_dir: Path) -> None:
    p = results_dir / "ledger.jsonl"
    if not p.exists():
        return
    import numpy as np
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in rows:
        latest[r["run_id"]] = r
    sub = ["desc_count", "desc_exist", "desc_query_color", "desc_query_shape", "desc_query_material"]
    mc = ["explanatory_per_question", "predictive_per_question", "counterfactual_per_question"]
    has_mc = any(k in ((r.get("metrics") or {}).get("test") or {}) for r in latest.values() for k in mc)
    test_cols = ["acc_all"] + sub + (mc if has_mc else [])
    names = {"acc_all": "test 전체", "desc_count": "개수", "desc_exist": "존재", "desc_query_color": "색",
             "desc_query_shape": "모양", "desc_query_material": "재질", "explanatory_per_question": "설명(문제)",
             "predictive_per_question": "예측(문제)", "counterfactual_per_question": "반사실(문제)"}
    head = ["run_id", "상태", "바꾼 값", "best epoch", "val 전체"] + [names[c] for c in test_cols] + ["활성 엣지", "배선 비용", "초"]
    lines = ["# 실험 결과 한눈에 보기", "",
             f"자동 생성 ({time.strftime('%Y-%m-%d %H:%M')}). 원본은 `ledger.jsonl`과 `runs/*.json`이다. "
             "정확도는 validation으로 고른 epoch의 값이고, test는 같은 가중치로 잰 값이다.", "",
             "## 실행별", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in latest.values():
        m = r.get("metrics", {}) or {}
        t = m.get("test", {}) or {}
        st = r.get("structure", {}) or {}
        ch = ", ".join(f"{k}={v}" for k, v in r.get("changed", {}).items()) or "(기준)"
        cells = [r["run_id"], r.get("status", ""), ch, _fmt(m.get("best_epoch")),
                 _fmt((m.get("val") or {}).get("acc_all"))] + [_fmt(t.get(c)) for c in test_cols] + [
                 _fmt(st.get("edges_active")), _fmt(st.get("wiring_cost")), _fmt(r.get("seconds"))]
        lines.append("| " + " | ".join(cells) + " |")
    groups = {}
    for r in latest.values():
        if r.get("status") != "ok":
            continue
        key = (r["experiment"], json.dumps({k: v for k, v in r.get("changed", {}).items() if k != "seed"},
                                           sort_keys=True, ensure_ascii=False))
        groups.setdefault(key, []).append(r)
    ghead = ["실험", "조건", "n", "val 전체"] + [names[c] for c in test_cols]
    lines += ["", "## 조건별 (seed 평균 ± SD)", "", "| " + " | ".join(ghead) + " |", "|" + "---|" * len(ghead)]

    def ms(rs, split, key):
        v = np.array([((r["metrics"].get(split) or {}).get(key, np.nan)) for r in rs], float)
        v = v[~np.isnan(v)]
        return "" if not len(v) else (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v[0]:.4f}")
    for (exp, cond), rs in sorted(groups.items()):
        lines.append("| " + " | ".join([exp, cond, str(len(rs)), ms(rs, "val", "acc_all")]
                                       + [ms(rs, "test", c) for c in test_cols]) + " |")
    (results_dir / "RESULTS.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
