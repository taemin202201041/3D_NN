"""Entry point (PyTorch): input -> training -> answer scoring -> JSON / JSONL.

    python run.py --exp config/e01_example_dynamic.json            run every variant x seed of an experiment
    python run.py --exp config/e01_example_dynamic.json --dry-run  list the runs without training
    python run.py --set train.epochs=2 --set data.subsample=0.02 --name smoke   one run from base.json
    python run.py --overview                                       rebuild results/RESULTS.md from the ledger

Experiment file:
    {"experiment": "e01", "description": "...", "set": {"train.epochs": 30},
     "variants": [{"name": "d3", "layout.dynamic.method": "swap"}, ...], "seeds": [1, 2, 3]}
Runs already in the ledger with status ok are skipped unless --rerun.
Splits (data.*_scenes): train 0-1999, validation 2000-2499 (epoch selection), test 2500-2999 (scored once per run).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import results as R  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def set_path(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_base() -> dict:
    return json.loads((HERE / "config" / "base.json").read_text(encoding="utf-8"))


def expand(args):
    base = load_base()
    exp = (json.loads(Path(args.exp).read_text(encoding="utf-8")) if args.exp else
           {"experiment": args.name or "adhoc", "variants": [{"name": args.name or "adhoc"}], "seeds": [base["seed"]]})
    common = dict(exp.get("set", {}))
    for s in args.set:
        k, v = s.split("=", 1)
        common[k] = parse_value(v)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else exp.get("seeds", [base["seed"]])
    runs = []
    for seed in seeds:
        for var in exp["variants"]:
            if args.only and var.get("name") not in args.only.split(","):
                continue
            cfg = copy.deepcopy(base)
            cfg["experiment"] = exp["experiment"]
            for k, v in {**common, **{k: v for k, v in var.items() if k != "name"}}.items():
                set_path(cfg, k, v)
            cfg["seed"] = seed
            cfg["name"] = var.get("name", "run")
            cfg["run_id"] = f"{exp['experiment']}__{cfg['name']}__s{seed}__{R.config_hash(cfg)}"
            runs.append(cfg)
    return runs, base


# ---------------------------------------------------------------------- training / evaluation
@torch.no_grad()
def evaluate(model, vt, split):
    """predictions, logits, mean task loss, penalty on one split"""
    from dataset import batches
    model.eval()
    logits, ys = [], []
    for x, y in batches(vt, split, 1024, False, device=dev):
        logits.append(model(*x))
        ys.append(y)
    lg, y = torch.cat(logits), torch.cat(ys)
    task = torch.nn.functional.cross_entropy(lg, y)
    return lg.argmax(-1).cpu().numpy(), lg.cpu().numpy(), float(task), float(model.spatial_penalty())


def set_rewire(model, keep_ratio, rewire, rng):
    """SET: drop the weakest `rewire` share of active edges and regrow as many at random."""
    core = model.core
    m = core.rec_mask.detach().cpu().numpy().ravel().copy()
    w = core.recurrent_kernel.detach().cpu().numpy().ravel().copy()
    act, ina = np.flatnonzero(m > 0), np.flatnonzero(m == 0)
    n = max(1, int(round(len(act) * rewire)))
    drop = act[np.argsort(np.abs(w[act]), kind="stable")[:n]]
    grow = rng.choice(ina, size=min(n, len(ina)), replace=False)
    scale = float(np.std(w[act])) or 0.05
    m[drop], m[grow] = 0.0, 1.0
    w[drop], w[grow] = 0.0, rng.normal(0, scale, size=len(grow))
    shape = tuple(core.rec_mask.shape)
    with torch.no_grad():
        core.rec_mask.copy_(torch.tensor(m.reshape(shape)))
        core.recurrent_kernel.copy_(torch.tensor(w.reshape(shape)))


def fit(model, vt, data, tr, epochs, geom, nan_epochs, set_cfg=None):
    from dataset import batches
    from metrics import scores
    s = data["splits"]
    opt = torch.optim.Adam(list(model.parameters()), lr=tr["lr"], weight_decay=float(tr.get("weight_decay", 0.0)))
    keep = {"best_acc": -1.0, "best_epoch": 0, "state": None, "best_logs": {}, "coords": None}
    hist = {k: [] for k in ("loss", "task_loss", "penalty", "accuracy", "val_loss", "val_task_loss", "val_penalty", "val_accuracy")}
    rng = np.random.default_rng(int(model.cfg["seed"]))
    for ep in range(epochs):
        model.train()
        tot = {"loss": 0.0, "task_loss": 0.0, "penalty": 0.0, "n": 0, "ok": 0}
        for x, y in batches(vt, s["train"], tr["batch"], True, seed=int(model.cfg["seed"]) * 1000 + ep + len(hist["loss"]), device=dev):
            logits = model(*x)
            task = torch.nn.functional.cross_entropy(logits, y)
            pen = model.spatial_penalty()
            loss = task + pen + geom.extra_loss()
            opt.zero_grad()
            loss.backward()
            for p in model.parameters():                      # per-tensor norm clipping (Keras clipnorm semantics)
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], tr["clipnorm"])
            opt.step()
            geom.step()
            b = len(y)
            tot["loss"] += float((task + pen).detach()) * b; tot["task_loss"] += float(task.detach()) * b; tot["penalty"] += float(pen.detach()) * b
            tot["n"] += b; tot["ok"] += int((logits.argmax(-1) == y).sum())
        pv, lv, vt_loss, vpen = evaluate(model, vt, s["val"])
        val_acc = float((pv == s["val"]["y"]).mean())
        logs = {"loss": tot["loss"] / tot["n"], "task_loss": tot["task_loss"] / tot["n"], "penalty": tot["penalty"] / tot["n"],
                "accuracy": tot["ok"] / tot["n"], "val_loss": vt_loss + vpen, "val_task_loss": vt_loss, "val_penalty": vpen,
                "val_accuracy": val_acc}
        if any(not np.isfinite(v) for v in logs.values()):
            nan_epochs.append(ep + 1)
            break
        for k, v in logs.items():
            hist[k].append(round(v, 5))
        geom.on_epoch_end(ep)
        sel_val = val_acc
        if tr.get("select", "val_best") == "val_mean":     # mean of descriptive acc and per-question accs (MC options are noisy)
            sc = scores(pv, s["val"])
            parts = [sc[k] for k in ("descriptive_per_option", "explanatory_per_question", "predictive_per_question", "counterfactual_per_question") if k in sc]
            sel_val = float(np.mean(parts)) if parts else val_acc
        if sel_val > keep["best_acc"]:
            keep.update(best_acc=sel_val, best_epoch=ep + 1, best_logs=dict(logs), coords=geom.coords_numpy(),
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        print("  epoch %d: " % (ep + 1) + " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)
        if set_cfg:
            set_rewire(model, rng=rng, **{k: set_cfg[k] for k in ("keep_ratio", "rewire")}) if ep + 1 < epochs else None
    if keep["state"] is not None:
        model.load_state_dict(keep["state"])
        geom.set_coords(keep["coords"])
    return keep, hist


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(HERE))
    except ValueError:
        return str(p)


def resolve_from_run(spec: str, cfg: dict, results_dir: Path) -> str:
    """'match:<experiment>' -> the ok dense run of that experiment with the same seed and layout.dim"""
    if not spec.startswith("match:"):
        return spec
    exp, hits = spec.split(":", 1)[1], []
    for line in (results_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        c = r.get("config", {})
        if (r.get("status") == "ok" and r.get("experiment") == exp and c.get("seed") == cfg["seed"]
                and c["layout"]["dim"] == cfg["layout"]["dim"] and not c["model"]["autapse_only"] and c["prune"]["method"] == "none"):
            hits.append(r["run_id"])
    if not hits:
        raise FileNotFoundError(f"no dense run for {spec} seed={cfg['seed']} dim={cfg['layout']['dim']}")
    return hits[-1]


def prune_mask(w_eff: np.ndarray, method: str, keep_ratio: float, rng) -> np.ndarray:
    n = w_eff.size
    k = int(round(keep_ratio * n))
    flat = np.zeros(n, np.float32)
    if method in ("magnitude", "set"):
        idx = np.argsort(-np.abs(w_eff).ravel(), kind="stable")[:k]
    elif method == "random":
        idx = rng.choice(n, size=k, replace=False)
    else:
        raise NotImplementedError(f"prune.method={method} is not implemented yet")
    flat[idx] = 1.0
    return flat.reshape(w_eff.shape)


def loss_profile(logits: np.ndarray, y: np.ndarray) -> dict:
    """Per-sample cross-entropy summary (a few exploding samples can make the MEAN loss huge while accuracy is fine)."""
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    ce = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))
    return {"median": round(float(np.median(ce)), 5), "mean": round(float(ce.mean()), 5), "frac_gt10": round(float((ce > 10).mean()), 6),
            "max": round(float(ce.max()), 3), "max_abs_logit": round(float(np.abs(logits[logits > -1e8]).max()), 3)}


# ---------------------------------------------------------------------- one run
def run_one(cfg: dict, base: dict, data: dict, vt) -> dict:
    from layout import Geometry
    from metrics import baselines, scores, structure
    from model import SeRNN
    from questions import N_CLASSES
    from rnn import communicability_np
    from topology import topology_mask

    t0 = time.time()
    seed = int(cfg["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    results_dir = HERE / cfg["paths"]["results_dir"]
    tr, m, pr = cfg["train"], cfg["model"], cfg["prune"]
    s = data["splits"]
    geom = Geometry(cfg, dev)
    steps, d_video = data["video"].shape[1], data["video"].shape[2]
    model = SeRNN(cfg, geom, d_video, s["train"]["q"].shape[1], seed).to(dev)
    base_mask, topo_info = topology_mask(m.get("topology", "full"), geom.coords0, m.get("topology_cfg", {}), seed)
    if m["autapse_only"]:
        base_mask = np.eye(m["units"], dtype=np.float32)
    model.core.rec_mask.copy_(torch.tensor(base_mask))
    nan_epochs: list[int] = []
    history, prune_info = {}, {"method": pr["method"]}

    if pr["method"] != "none" and pr.get("from_run"):
        src = resolve_from_run(pr["from_run"], cfg, results_dir)
        ck = torch.load(results_dir / "weights" / f"{src}.weights.pt", map_location=dev)
        model.load_state_dict(ck["model"]); geom.set_coords(ck["coords"])
        model.core.rec_mask.copy_(torch.tensor(base_mask))
        keep = {"best_epoch": 0, "best_logs": {}}
        prune_info["dense_source"] = src
    else:
        print(f"[{cfg['run_id']}] training {tr['epochs']} epochs", flush=True)
        keep, history = fit(model, vt, data, tr, tr["epochs"], geom, nan_epochs)

    if pr["method"] != "none" and not nan_epochs:
        pre = scores(evaluate(model, vt, s["val"])[0], s["val"])
        mask = prune_mask(model.core.effective_recurrent().detach().cpu().numpy(), pr["method"], pr["keep_ratio"],
                          np.random.default_rng(seed)) * base_mask
        model.core.rec_mask.copy_(torch.tensor(mask))
        prune_info.update(keep_ratio=pr["keep_ratio"], edges_kept=int(mask.sum()), val_before_prune=pre["acc_all"],
                          val_after_prune_no_retrain=scores(evaluate(model, vt, s["val"])[0], s["val"])["acc_all"],
                          retrain_epochs=pr["retrain_epochs"], schedule=pr["schedule"])
        if pr["retrain_epochs"] > 0:
            set_cfg = dict(keep_ratio=pr["keep_ratio"], rewire=pr["set_rewire"]) if pr["method"] == "set" else None
            print(f"[{cfg['run_id']}] retraining {pr['retrain_epochs']} epochs after pruning", flush=True)
            keep, h2 = fit(model, vt, data, tr, pr["retrain_epochs"], geom, nan_epochs, set_cfg)
            history = {**{f"dense_{k}": v for k, v in history.items()}, **h2}

    vp, vl, v_task, v_pen = evaluate(model, vt, s["val"])
    val_scores = scores(vp, s["val"])
    profile = {"val": loss_profile(vl, s["val"]["y"])}
    test_scores, t_task, t_pen = {}, None, None
    if cfg["eval"]["test"]:
        tp, tl, t_task, t_pen = evaluate(model, vt, s["test"])
        test_scores = scores(tp, s["test"])
        profile["test"] = loss_profile(tl, s["test"]["y"])
    w_eff = model.core.effective_recurrent().detach().cpu().numpy()
    struct = structure(w_eff, geom.distance_numpy(), cfg["eval"]["edge_threshold"], lambda a: communicability_np(a, cfg["reg"]["eps"]))
    (results_dir / "weights").mkdir(parents=True, exist_ok=True)
    wpath = results_dir / "weights" / f"{cfg['run_id']}.weights.pt"
    torch.save({"model": model.state_dict(), "coords": geom.coords_numpy()}, wpath)
    cpath = results_dir / "coords" / f"{cfg['run_id']}.npy"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(cpath, geom.coords_numpy())
    files = {"weights": _rel(wpath), "coords": _rel(cpath)}

    bl = keep.get("best_logs", {})
    n_params = int(sum(p.numel() for p in model.parameters()))
    d = cfg["data"]
    record = {
        "schema": R.SCHEMA, "run_id": cfg["run_id"], "experiment": cfg["experiment"], "team": cfg["team"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": R.git_commit(HERE), "code_hash": R.code_hash(HERE),
        "config_hash": R.config_hash(cfg), "status": "nan" if nan_epochs else "ok",
        "changed": R.diff(base, cfg), "config": cfg,
        "data": {"perception": d["perception"], "encoder": d["encoder"],
                 "splits": {k: d[f"{k}_scenes"] for k in ("train", "val", "test")},
                 "n_scenes": {k: s[k]["n_scenes"] for k in s}, "n_samples": {k: int(len(s[k]["y"])) for k in s},
                 "question_types": d["question_types"], "question_encoding": d["question_encoding"],
                 "question_at": d["question_at"], "steps": int(steps), "frame_feature_dim": int(d_video),
                 "input_dim": int(d_video + s["train"]["q"].shape[1]), "encoder_settings": data["meta"].get("encoder_settings"),
                 "scene_ids": {k: data["meta"]["scenes"][d[f"{k}_scenes"][0]:d[f"{k}_scenes"][1]] for k in ("train", "val", "test")}},
        "model": {"units": m["units"], "layout": geom.summary(), "activation": m["activation"], "init": m["init"],
                  "lags": m["lags"], "leak_alpha": m["leak_alpha"], "io": m["io"], "autapse_only": m["autapse_only"],
                  "answer_mask": m["answer_mask"], "topology": topo_info, "edges_allowed": int(base_mask.sum()),
                  "directed_edges": True, "trainable_params": n_params, "reg": cfg["reg"], "optimizer": tr["optimizer"],
                  "lr": tr["lr"], "clipnorm": tr["clipnorm"], "batch": tr["batch"], "epochs": tr["epochs"],
                  "input_noise": m["input_noise"]},
        "prune": prune_info,
        "metrics": {
            "best_epoch": keep.get("best_epoch"), "epochs_run": len(history.get("loss", [])), "nan_epochs": nan_epochs,
            "loss": {"train_task": bl.get("task_loss"), "train_penalty": bl.get("penalty"), "train_total": bl.get("loss"),
                     "val_task": v_task, "val_penalty": v_pen, "test_task": t_task, "per_sample": profile},
            "val": val_scores, "test": test_scores,
            "baselines": {"val": baselines(s["train"], s["val"], N_CLASSES),
                          "test": baselines(s["train"], s["test"], N_CLASSES) if test_scores else {}}},
        "structure": struct, "history": history, "files": files,
        "seconds": round(time.time() - t0, 1), "versions": R.versions(),
        "test_evaluated": bool(test_scores), "note": cfg.get("note", ""),
    }
    record = json.loads(json.dumps(record, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    R.save(results_dir, record)
    print(f"[{cfg['run_id']}] done in {record['seconds']}s  val={val_scores['acc_all']}  test={test_scores.get('acc_all')}", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp"); ap.add_argument("--set", action="append", default=[]); ap.add_argument("--name")
    ap.add_argument("--only"); ap.add_argument("--seeds"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true"); ap.add_argument("--overview", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="run at most N pending runs, then exit")
    args = ap.parse_args()
    base = load_base()
    results_dir = HERE / base["paths"]["results_dir"]
    if args.overview:
        R.overview(results_dir)
        return
    runs, base = expand(args)
    done = set() if args.rerun else R.done_ids(results_dir)
    todo = [c for c in runs if c["run_id"] not in done]
    if args.max_runs:
        todo = todo[: args.max_runs]
    for c in runs:
        print(("skip " if c["run_id"] in done else "run  ") + c["run_id"], flush=True)
    if args.dry_run or not todo:
        return
    from dataset import load

    tensors, failed = {}, 0
    for cfg in todo:
        data = load(cfg, HERE)                      # cached in-process after the first call
        if id(data["video"]) not in tensors:
            tensors.clear()
            tensors[id(data["video"])] = torch.from_numpy(data["video"]).to(dev)
        try:
            run_one(cfg, base, data, tensors[id(data["video"])])
        except Exception as e:                      # keep the queue going; record the failure
            import traceback
            traceback.print_exc()
            R.save(results_dir, {"schema": R.SCHEMA, "run_id": cfg["run_id"], "experiment": cfg["experiment"], "team": cfg["team"],
                                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "failed", "changed": R.diff(base, cfg),
                                 "config": cfg, "metrics": {}, "structure": {}, "note": repr(e)[:2000]})
            failed += 1
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
