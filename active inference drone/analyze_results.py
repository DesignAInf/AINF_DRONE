#!/usr/bin/env python3
"""Frozen statistical analysis and extended publication figures for V5.

Author: Luca M. Possati
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from active_drone.agent import CONDITIONS
from active_drone.graphics import BLUE, CYAN, GREEN, GREY, NAVY, ORANGE, PURPLE, RED, style
from active_drone.world import TASKS

DISPLAY = {
    "constitutive": "Constitutive",
    "diagnostic_only": "Diagnostic only",
    "factorized": "Factorized",
    "random_design": "Random design",
    "passive": "Passive",
    "oracle": "Oracle",
}
COLORS = {
    "constitutive": BLUE,
    "diagnostic_only": RED,
    "factorized": PURPLE,
    "random_design": CYAN,
    "passive": GREY,
    "oracle": GREEN,
}


def read_rows(path: Path) -> list[dict]:
    out = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out.append({k: (v if k in ("condition", "task") else int(v) if k == "seed" else float(v))
                        for k, v in row.items()})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def paired_statistics(rows: list[dict], bootstrap: int = 10_000) -> list[dict]:
    rng = np.random.default_rng(5506)
    metrics = ("success", "viability", "final_gap", "architecture_score",
               "boundary_precision", "role_accuracy", "blanket_accuracy", "exact_partition")
    comparators = tuple(c for c in CONDITIONS if c != "constitutive")
    results = []
    for task in TASKS:
        for comparator in comparators:
            seeds = sorted({int(r["seed"]) for r in rows if r["task"] == task})
            for metric in metrics:
                delta = []
                for seed in seeds:
                    full = next(r[metric] for r in rows if r["condition"] == "constitutive" and r["task"] == task and r["seed"] == seed)
                    comp = next(r[metric] for r in rows if r["condition"] == comparator and r["task"] == task and r["seed"] == seed)
                    delta.append(full - comp)
                d = np.asarray(delta, float)
                samples = d[rng.integers(0, len(d), (bootstrap, len(d)))].mean(1)
                sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                results.append({
                    "task": task,
                    "comparison": f"constitutive - {comparator}",
                    "metric": metric,
                    "mean_difference": float(d.mean()),
                    "standard_error": float(sd / np.sqrt(len(d))),
                    "bootstrap_95_low": float(np.quantile(samples, .025)),
                    "bootstrap_95_high": float(np.quantile(samples, .975)),
                    "paired_dz": float(d.mean() / sd) if sd > 1e-12 else float("nan"),
                    "n_pairs": len(d),
                })
    return results


def ablation_matrix(summary: list[dict], output: Path) -> None:
    style(); fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2), constrained_layout=True)
    specs = (("success_rate", "Task success", 100, "%"),
             ("architecture_score", "Architecture score", 1, ""),
             ("viability", "Viability", 1, ""))
    for ax, (metric, title, scale, suffix) in zip(axes, specs):
        matrix = np.array([[next(r[metric] for r in summary if r["condition"] == c and r["task"] == t)
                            for t in TASKS] for c in CONDITIONS]) * scale
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100 if scale == 100 else 1, aspect="auto")
        ax.set_xticks(range(3), [t.capitalize() for t in TASKS])
        ax.set_yticks(range(len(CONDITIONS)), [DISPLAY[c] for c in CONDITIONS])
        ax.set_title(title)
        for i in range(len(CONDITIONS)):
            for j in range(3):
                value = matrix[i, j]
                color = "white" if value > (58 if scale == 100 else .58) else NAVY
                label = f"{value:.0f}{suffix}" if scale == 100 else f"{value:.3f}"
                ax.text(j, i, label, ha="center", va="center", color=color, fontweight="bold", fontsize=9)
        cb = fig.colorbar(im, ax=ax, shrink=.78, pad=.02)
        cb.set_label("Percent" if scale == 100 else "Normalized score")
    fig.suptitle("Full ablation matrix · 20 paired seeds per cell", fontsize=17, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def causal_chain(rows: list[dict], output: Path) -> None:
    style(); fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        seeds = sorted({int(r["seed"]) for r in rows if r["task"] == task})
        for seed in seeds:
            diag = next(r for r in rows if r["condition"] == "diagnostic_only" and r["task"] == task and r["seed"] == seed)
            full = next(r for r in rows if r["condition"] == "constitutive" and r["task"] == task and r["seed"] == seed)
            ax.add_patch(FancyArrowPatch((diag["architecture_score"], diag["viability"]),
                                         (full["architecture_score"], full["viability"]),
                                         arrowstyle="-|>", mutation_scale=8, lw=.9, color="#94A3B8", alpha=.55))
            ax.scatter(diag["architecture_score"], diag["viability"], s=23, color=RED, alpha=.78)
            ax.scatter(full["architecture_score"], full["viability"], s=28, color=BLUE, alpha=.88)
        ax.set(xlim=(-.03, 1.03), ylim=(0, 1.03), xlabel="Final architecture score",
               ylabel="Viability", title=task.capitalize())
        ax.grid(True); ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, label="Diagnostic only"),
               plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Constitutive")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Paired causal contrast: realized architecture and viability", fontsize=17, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def inference_action(summary: list[dict], output: Path) -> None:
    style(); chosen = ("constitutive", "diagnostic_only", "factorized", "random_design", "oracle")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    metrics = (("exact_partition", "Exact final four-role partition", 100, "%"),
               ("causal_changes", "Effective causal changes", 1, "actions"),
               ("final_gap", "Final path-factorization gap", 1, r"$\Delta_{fac}$"))
    x = np.arange(3); width = .16
    for ax, (metric, title, scale, ylabel) in zip(axes, metrics):
        for k, condition in enumerate(chosen):
            values = [next(r[metric] for r in summary if r["condition"] == condition and r["task"] == task)*scale for task in TASKS]
            errors = [next(r[metric+"_se"] for r in summary if r["condition"] == condition and r["task"] == task)*scale for task in TASKS]
            ax.bar(x+(k-2)*width, values, width, yerr=errors, capsize=2, color=COLORS[condition], alpha=.9,
                   label=DISPLAY[condition])
        ax.set_xticks(x, [t.capitalize() for t in TASKS]); ax.set_title(title); ax.set_ylabel(ylabel)
        ax.grid(axis="y"); ax.spines[["top", "right"]].set_visible(False)
        if metric == "exact_partition": ax.set_ylim(0, 105)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Inference quality is distinct from constitutive efficacy", fontsize=17, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def inversion_levels(output: Path) -> None:
    style(); fig, ax = plt.subplots(figsize=(15.5, 6.2)); ax.set(xlim=(0, 15.5), ylim=(0, 6.2)); ax.axis("off")
    cards = [(.45, 1.15, 4.25, "1 · Detection", "Infer an existing boundary", r"$q_t(M)\rightarrow \mathcal{C}$", GREY),
             (5.62, 1.15, 4.25, "2 · Tracking", "Follow exogenous change", r"$\omega_t:\mathcal{C}_t\rightarrow\mathcal{C}_{t+1}$", PURPLE),
             (10.79, 1.15, 4.25, "3 · Constitution", "Design the realized boundary", r"$d_t:\mathcal{C}_t\rightarrow\mathcal{C}_{t+1}$", GREEN)]
    for x, y, w, title, subtitle, formula, color in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, 3.45, boxstyle="round,pad=.16,rounding_size=.16",
                                    facecolor="white", edgecolor=color, linewidth=2.3))
        ax.add_patch(FancyBboxPatch((x, y+2.65), w, .8, boxstyle="round,pad=.16,rounding_size=.16",
                                    facecolor=color, edgecolor=color))
        ax.text(x+w/2, y+3.05, title, ha="center", va="center", color="white", fontweight="bold", fontsize=15)
        ax.text(x+w/2, y+2.08, subtitle, ha="center", va="center", color=NAVY, fontsize=12, fontweight="bold")
        ax.text(x+w/2, y+1.25, formula, ha="center", va="center", color=color, fontsize=14)
        ax.text(x+w/2, y+.48, "belief changes" if x < 10 else "world and belief change",
                ha="center", va="center", color=GREY, fontsize=10)
    ax.text(7.75, 5.55, "Three claims that must not be conflated", ha="center", fontsize=20, fontweight="bold")
    ax.text(7.75, .45, r"V5 tests constitution by requiring $\mathcal{T}(\mathcal{C}_t,d_t,\omega_t)\neq\mathcal{T}(\mathcal{C}_t,d_t',\omega_t)$",
            ha="center", fontsize=12, color=NAVY)
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def efe_decomposition(output: Path) -> None:
    style(); fig, ax = plt.subplots(figsize=(15.5, 7.2)); ax.set(xlim=(0, 15.5), ylim=(0, 7.2)); ax.axis("off")
    terms = [(.5, 4.55, 2.5, "Pragmatic risk", r"$R_t(u)$", RED),
             (3.5, 4.55, 2.5, "Ambiguity", r"$\alpha\mathcal{A}_t(u)$", ORANGE),
             (6.5, 4.55, 2.5, "Structure", r"$-\hat p_k\beta_sV_s(d)$", CYAN),
             (9.5, 4.55, 2.5, "Precision", r"$-\hat p_k\beta_pV_p(d)$", GREEN),
             (12.5, 4.55, 2.5, "Information", r"$-\beta_I\mathrm{IG}_t(d)$", PURPLE)]
    for x, y, w, label, formula, color in terms:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.35, boxstyle="round,pad=.1,rounding_size=.1",
                                    facecolor=color, edgecolor="none", alpha=.95))
        ax.text(x+w/2, y+.87, label, ha="center", color="white", fontweight="bold", fontsize=11)
        ax.text(x+w/2, y+.38, formula, ha="center", color="white", fontsize=11)
        ax.add_patch(FancyArrowPatch((x+w/2, y-.05), (7.75, 3.27), arrowstyle="-|>", mutation_scale=13,
                                     lw=1.6, color=color))
    ax.add_patch(FancyBboxPatch((5.35, 1.82), 4.8, 1.35, boxstyle="round,pad=.13,rounding_size=.13",
                                facecolor=BLUE, edgecolor="none"))
    ax.text(7.75, 2.72, r"Joint EFE $G_t(u,d)$", ha="center", color="white", fontsize=15, fontweight="bold")
    ax.text(7.75, 2.2, r"enumerate control $\times$ design", ha="center", color="white", fontsize=11)
    ax.add_patch(FancyArrowPatch((7.75, 1.78), (7.75, .82), arrowstyle="-|>", mutation_scale=15, lw=2, color=NAVY))
    ax.text(7.75, .45, r"$q(\pi)\propto\exp[-G_t(\pi)/\gamma]$, $\pi=(u,d)$", ha="center", fontsize=14, fontweight="bold")
    ax.text(7.75, 6.75, "One policy posterior couples flight control and boundary design", ha="center", fontsize=20, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def latex_table(summary: list[dict], path: Path) -> None:
    lines = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.7pt}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Condition & Task & Success & Viability & $Q_{arch}$ & Final $\Delta_{fac}$ & Role acc. & Exact \\",
             r"\midrule"]
    for ci, condition in enumerate(CONDITIONS):
        for task in TASKS:
            row = next(r for r in summary if r["condition"] == condition and r["task"] == task)
            lines.append(f"{DISPLAY[condition]} & {task.capitalize()} & {100*row['success_rate']:.1f}\\% & "
                         f"{row['viability']:.3f} & {row['architecture_score']:.3f} & {row['final_gap']:.3f} & "
                         f"{100*row['role_accuracy']:.1f}\\% & {100*row['exact_partition']:.1f}\\% \\\\")
        if ci < len(CONDITIONS)-1: lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Complete frozen benchmark. Values are means over 20 paired seeds.}",
              r"\label{tab:full-results}", r"\end{table}"]
    path.write_text("\n".join(lines)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/extended"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.results/"mission_level.csv")
    summary = read_rows(args.results/"summary.csv")
    stats = paired_statistics(rows)
    write_csv(args.output/"paired_statistics_extended.csv", stats)
    latex_table(summary, args.output/"full_results_table.tex")
    ablation_matrix(summary, args.output/"ablation_matrix.png")
    causal_chain(rows, args.output/"paired_causal_chain.png")
    inference_action(summary, args.output/"inference_action_diagnostics.png")
    inversion_levels(args.output/"inversion_levels.png")
    efe_decomposition(args.output/"joint_efe_decomposition.png")
    metadata = {"author": "Luca M. Possati", "source": "frozen V5 benchmark",
                "missions": len(rows), "paired_seeds": len(set(r["seed"] for r in rows)),
                "bootstrap_samples": 10_000, "rng_seed": 5506}
    (args.output/"analysis_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
