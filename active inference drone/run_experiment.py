#!/usr/bin/env python3
"""Run the preregistered V5 constitutive-boundary benchmark."""

from __future__ import annotations

import argparse, csv, json, platform, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from active_drone.agent import CONDITIONS
from active_drone.graphics import architecture_figure, benchmark_figure, role_posterior_figure, triad_figure
from active_drone.simulation import make_system, run_mission
from active_drone.world import TASKS


def mean_se(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return (float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0)


def run_one(spec):
    condition, task, seed = spec
    world, drone = make_system(seed, condition, task)
    return run_mission(world, drone)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def mediation(rows: list[dict], bootstrap: int = 1000) -> dict:
    data = [r for r in rows if r["condition"] in ("constitutive", "diagnostic_only")]
    x = np.array([r["condition"] == "constitutive" for r in data], float)
    m = np.array([r["architecture_score"] for r in data], float)
    y = np.array([r["viability"] for r in data], float)
    task = np.array([[r["task"] == "stabilize", r["task"] == "transform"] for r in data], float)
    def fit(idx):
        xm = np.column_stack([np.ones(len(idx)), x[idx], task[idx]])
        a = np.linalg.lstsq(xm, m[idx], rcond=None)[0][1]
        xy = np.column_stack([np.ones(len(idx)), x[idx], m[idx], task[idx]])
        beta = np.linalg.lstsq(xy, y[idx], rcond=None)[0]
        return float(a), float(beta[2]), float(beta[1])
    idx = np.arange(len(data)); a, b, direct = fit(idx)
    rng = np.random.default_rng(5505); indirect = []
    seeds = sorted(set(r["seed"] for r in data))
    for _ in range(bootstrap):
        sampled = rng.choice(seeds, len(seeds), replace=True)
        boot = np.concatenate([[i for i, r in enumerate(data) if r["seed"] == s] for s in sampled])
        aa, bb, _ = fit(boot); indirect.append(aa*bb)
    lo, hi = np.quantile(indirect, [.025, .975])
    return {"a_constitutive_to_architecture": a, "b_architecture_to_viability": b,
            "direct_effect": direct, "indirect_effect": a*b,
            "indirect_bootstrap_95_low": float(lo), "indirect_bootstrap_95_high": float(hi)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.quick: args.seeds = min(args.seeds, 4); args.seed_offset = 0

    missions = []; rows = []; examples = {}
    specs = [(condition, task, args.seed_offset+k) for condition in CONDITIONS for task in TASKS for k in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed = pool.map(run_one, specs, chunksize=1)
            results = list(completed)
    else:
        results = [run_one(spec) for spec in specs]
    for (condition, task, seed), result in zip(specs, results):
        missions.append(result)
        rows.append({"condition": condition, "task": task, "seed": seed,
                     "success": result.success, "collision": result.collision,
                     "viability": result.viability, "mean_gap": result.mean_gap,
                     "final_gap": result.final_gap, "architecture_score": result.architecture_score,
                     "boundary_precision": result.boundary_precision,
                     "role_accuracy": result.role_accuracy, "blanket_accuracy": result.blanket_accuracy,
                     "exact_partition": result.exact_partition, "causal_changes": result.causal_changes,
                     "recovery_delay": result.recovery_delay})
        if condition == "constitutive" and task not in examples and result.success:
            examples[task] = result

    summary = []
    for condition in CONDITIONS:
        for task in TASKS:
            subset = [r for r in rows if r["condition"] == condition and r["task"] == task]
            row = {"condition": condition, "task": task, "missions": len(subset)}
            for metric in ("success", "collision", "viability", "mean_gap", "final_gap", "architecture_score",
                           "boundary_precision", "role_accuracy", "blanket_accuracy", "exact_partition", "causal_changes", "recovery_delay"):
                mean, se = mean_se([r[metric] for r in subset])
                key = {"success": "success_rate", "collision": "collision_rate"}.get(metric, metric)
                row[key], row[key+"_se"] = mean, se
            summary.append(row)

    paired = []
    for task in TASKS:
        for metric in ("success", "viability", "final_gap", "architecture_score"):
            delta = []
            for k in range(args.seeds):
                seed = args.seed_offset+k
                full = next(r[metric] for r in rows if r["condition"] == "constitutive" and r["task"] == task and r["seed"] == seed)
                diag = next(r[metric] for r in rows if r["condition"] == "diagnostic_only" and r["task"] == task and r["seed"] == seed)
                delta.append(full-diag)
            mean, se = mean_se(delta); paired.append({"task": task, "metric": metric, "constitutive_minus_diagnostic": mean, "se": se})

    write_csv(args.output/"mission_level.csv", rows); write_csv(args.output/"summary.csv", summary); write_csv(args.output/"paired_effects.csv", paired)
    med = mediation(rows, bootstrap=250 if args.quick else 2000)
    (args.output/"mediation.json").write_text(json.dumps(med, indent=2)+"\n")
    meta = {"version": "0.5.0", "seeds": args.seeds, "seed_offset": args.seed_offset,
            "tasks": TASKS, "conditions": CONDITIONS, "structured_partitions": 2520,
            "primary_comparison": "constitutive vs diagnostic_only", "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "quick": args.quick}
    (args.output/"run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    architecture_figure(args.output/"v5_architecture.png")
    benchmark_figure(summary, args.output/"v5_benchmark.png")
    if len(examples) == 3:
        triad_figure(examples, args.output/"construct_stabilize_transform.png")
        role_posterior_figure(examples["transform"], args.output/"four_role_posterior.png")
    print(json.dumps({"summary": summary, "paired": paired, "mediation": med}, indent=2))


if __name__ == "__main__":
    main()
