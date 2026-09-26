"""Publication graphics for the constitutive V5 benchmark."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#102A43"; BLUE = "#1769E0"; CYAN = "#00A6A6"; GREEN = "#168B5B"
ORANGE = "#F28E2B"; RED = "#D64545"; PURPLE = "#7A5AF8"; GREY = "#66788A"
ROLE_COLORS = [PURPLE, CYAN, GREEN, ORANGE]


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.edgecolor": "#CBD5E1", "axes.labelcolor": NAVY,
                         "text.color": NAVY, "xtick.color": GREY, "ytick.color": GREY,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "grid.color": "#E7EDF3", "legend.frameon": False})


def architecture_figure(path: Path) -> None:
    style(); fig, ax = plt.subplots(figsize=(15.5, 8.2)); ax.set(xlim=(0, 15.5), ylim=(0, 8.2)); ax.axis("off")
    # Upper track: ontology. Lower track: inference. Keeping them distinct is
    # the central conceptual repair from V4.
    top = [(0.6, 5.0, 2.5, "Physical organization\n$C_t$", ORANGE),
           (4.0, 5.0, 2.5, "Constitutive action\n$d_t$", GREEN),
           (7.4, 5.0, 2.5, "New organization\n$C_{t+1}$", ORANGE),
           (10.8, 5.0, 3.1, r"Path-law factorization"+"\n"+r"$\Delta_{fac}(C_{t+1})$", CYAN)]
    bottom = [(0.6, 1.6, 2.5, "Observed paths\n$x_{0:t}$", BLUE),
              (4.0, 1.6, 2.5, r"Learned dynamics"+"\n"+r"$q(A,\Phi)$", PURPLE),
              (7.4, 1.6, 2.5, "Joint posterior\n$q(M_t)$", PURPLE),
              (10.8, 1.6, 3.1, r"EFE-motivated design"+"\n"+r"$d_t\sim q(d)$", BLUE)]
    for x, y, w, label, color in top + bottom:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.45, boxstyle="round,pad=.12,rounding_size=.12",
                                    facecolor=color, edgecolor="none", alpha=.96))
        ax.text(x+w/2, y+.725, label, ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    def arrow(a, b, color=NAVY, rad=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=2,
                                     color=color, connectionstyle=f"arc3,rad={rad}"))
    for row in (top, bottom):
        for left, right in zip(row[:-1], row[1:]):
            arrow((left[0]+left[2]+.08, left[1]+.725), (right[0]-.08, right[1]+.725))
    arrow((12.35, 3.05), (5.25, 4.9), GREEN, -.12)
    arrow((8.65, 4.9), (1.85, 3.05), GREY, -.10)
    ax.text(7.75, 7.75, "V6 · Inversion-guided boundary design", ha="center", fontsize=21, fontweight="bold")
    ax.text(7.75, 7.25, "Beliefs track the boundary; design actions make a different boundary empirically true",
            ha="center", color=GREY, fontsize=12)
    ax.text(7.75, .72, r"Discriminating condition: $\partial C_{t+1}/\partial d_t \ne 0$", ha="center", fontsize=14,
            fontweight="bold", color=GREEN)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def benchmark_figure(summary: list[dict], path: Path) -> None:
    style(); tasks = ["construct", "stabilize", "transform"]
    conditions = ["constitutive", "diagnostic_only", "factorized", "random_design", "passive", "oracle"]
    colors = [BLUE, RED, PURPLE, CYAN, GREY, GREEN]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), constrained_layout=True)
    metrics = [("success_rate", "Task success", 100, "%"), ("architecture_score", "Final architecture score", 1, ""),
               ("viability", "Viability", 1, "")]
    x = np.arange(len(tasks)); width = .125
    for ax, (key, title, scale, suffix) in zip(axes, metrics):
        for k, (condition, color) in enumerate(zip(conditions, colors)):
            vals = [next(r[key] for r in summary if r["condition"] == condition and r["task"] == task)*scale for task in tasks]
            errs = [next(r[key+"_se"] for r in summary if r["condition"] == condition and r["task"] == task)*scale for task in tasks]
            ax.bar(x+(k-2.5)*width, vals, width, yerr=errs, capsize=2, color=color, alpha=.90,
                   label=condition.replace("_", " "))
        ax.set_xticks(x, [t.capitalize() for t in tasks]); ax.set_title(title); ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 105 if scale == 100 else 1.08); ax.set_ylabel(suffix if suffix else "Normalized score")
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")
    fig.suptitle("Constitutive boundary design versus diagnostic-only inversion · mean ± seed-level SE",
                 fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def triad_figure(examples: dict[str, object], path: Path) -> None:
    style(); fig, axes = plt.subplots(3, 2, figsize=(15.5, 11), constrained_layout=True)
    for row, task in enumerate(("construct", "stabilize", "transform")):
        mission = examples[task]; rec = mission.records; t = np.arange(len(rec))
        gap = np.array([r["factorization_gap"] for r in rec]); est = np.array([r["estimated_gap"] for r in rec])
        y = np.array([r["y"] for r in rec]); qs = np.array([r["sensory_precision"] for r in rec]); qa = np.array([r["active_precision"] for r in rec])
        changed = np.array([r["design_changed_causal_matrix"] for r in rec], bool)
        gap_norm = gap / max(float(np.max(gap)), 1e-9)
        est_norm = est / max(float(np.max(est)), 1e-9)
        ax = axes[row, 0]; ax.plot(t, gap_norm, color=RED, lw=2.2, label="True path-gap (normalized)")
        ax.plot(t, est_norm, color=PURPLE, lw=1.5, alpha=.8, label="Posterior leak energy (normalized)")
        ax.scatter(t[changed], gap_norm[changed], color=GREEN, marker="D", s=38, zorder=3, label="Causal redesign")
        ax.set(title=f"{task.capitalize()}: causal factorization", ylabel="Normalized diagnostic"); ax.grid(True)
        if row == 0: ax.legend(ncol=3, fontsize=8)
        ax = axes[row, 1]; ax.plot(t, y, color=BLUE, lw=2, label="Cross-track state")
        ax.plot(t, qs, color=CYAN, lw=1.4, label="Sensory precision"); ax.plot(t, qa, color=GREEN, lw=1.4, label="Active precision")
        ax.axhline(2.35, color=RED, ls="--", lw=1); ax.axhline(-2.35, color=RED, ls="--", lw=1)
        ax.set(title=f"{task.capitalize()}: viability consequences", ylabel="State / precision"); ax.grid(True)
        if row == 0: ax.legend(ncol=3, fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("Time step")
    fig.suptitle("Construct · stabilize · transform: interventions alter the process, not only its description",
                 fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def role_posterior_figure(mission, path: Path) -> None:
    style(); rec = mission.records; marg = np.asarray([r["role_marginals"] for r in rec])
    truth = np.asarray([r["roles"] for r in rec])
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    for role, ax in enumerate(axes):
        im = ax.imshow(marg[:, :, role].T, aspect="auto", vmin=0, vmax=1, cmap="Blues",
                       extent=[0, len(rec)-1, 7.5, -.5])
        ax.contour((truth == role).T, levels=[.5], colors=[ROLE_COLORS[role]], linewidths=1.3,
                   extent=[0, len(rec)-1, 7.5, -.5])
        ax.set_ylabel(f"{['I','S','A','E'][role]} node"); ax.set_yticks(range(8))
    axes[-1].set_xlabel("Time step"); fig.colorbar(im, ax=axes, shrink=.65, label="Marginal role probability")
    fig.suptitle("Full structured posterior over all four blanket roles", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
