"""
All paper figures and tables, produced from results/results.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.experiments import SWEEPS

# Style settings for all figures
mpl.rcParams.update({
    "figure.dpi":   110,
    "savefig.dpi":  160,
    "font.size":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

METHOD_ORDER = ["Pearson", "MI", "GENIE3", "PC", "GES", "NOTEARS"]

METHOD_COLORS = {
    "Pearson":  "#2E86AB",  # correlation-class: blue
    "MI":       "#6FB1D0",
    "GENIE3":   "#F5A623",  # tree: orange
    "PC":       "#C23B22",  # causal: reds
    "GES":      "#8B1E3F",
    "NOTEARS":  "#4B0D33",
}

PATHOLOGY_ORDER = list(SWEEPS.keys())
FAMILY_ORDER = ["correlation", "tree", "causal"]
FAMILY_COLORS = {"correlation": "#2E86AB", "tree": "#F5A623", "causal": "#8B1E3F"}
FAMILY_LABELS  = {"correlation": "Correlation", "tree": "Tree", "causal": "Causal"}

PATHOLOGY_LABELS = {
    "dropout":          "Dropout",
    "n_confounders":    "Confounders",
    "mix_ratio":        "Cell-Type Mix Ratio",
    "feedback":         "Feedback",
    "density":          "Network Density",
    "n_cells":          "Number of Cells",
    "pseudotime_drift": "Pseudotime Drift",
}

PATHOLOGY_DESCRIPTIONS = {
    "dropout":          "(Technical Zero-Inflation, Missingness)",
    "n_confounders":    "(Unobserved Latents Driving Gene Groups)",
    "mix_ratio":        "(Cell-Type Mixing)",
    "feedback":         "(Back-Edge Probability, Cycles)",
    "density":          "(Edge Density of the Ground-Truth DAG)",
    "n_cells":          "(Sample Size)",
    "pseudotime_drift": "(Non-Stationarity Along Pseudotime)",
}

METHOD_TO_FAMILY = {
    "Pearson": "correlation",
    "MI": "correlation",
    "GENIE3": "tree",
    "PC": "causal",
    "GES": "causal",
    "NOTEARS": "causal",
}
ERROR_COLORS = {
    "TRUE": "#2a9d8f",
    "REVERSED": "#f4a261",
    "CONFOUNDED": "#e76f51",
    "SPURIOUS": "#8d2e3d",
    "MISSED": "#bbbbbb",
}

# Helpers
def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- std of `metric` across seeds, per (pathology, level, method)."""
    g = df.groupby(["pathology", "level", "method"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out

def _metric_label(metric: str) -> str:
    if metric == "auprc_dir":
        return "AUPRC (Directed)"
    return "AUPRC (Undirected)"

def _extreme_level(pathology: str):
    """Return the most extreme sweep level as defined in SWEEPS order."""
    return SWEEPS[pathology]["levels"][-1]

def _family_aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- std of `metric` across seeds, per (pathology, level, family)."""
    df = df.copy()
    df["family"] = df["method"].map(METHOD_TO_FAMILY)
    agg = (df.groupby(["pathology", "level", "family"])[metric]
             .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return agg

def _save_figure(fig: plt.Figure, out_path: Path, **savefig_kwargs) -> None:
    """Save the requested figure plus a sibling PDF for paper use."""
    out_path = Path(out_path)
    fig.savefig(out_path, **savefig_kwargs)
    pdf_path = out_path.with_suffix(".pdf")
    if pdf_path != out_path:
        fig.savefig(pdf_path, **savefig_kwargs)

# Headline figure: degradation curves
def plot_headline_grid(df: pd.DataFrame, out_path: Path,
                       metric: str = "auprc_und") -> None:
    """
    One panel per pathology. Each panel shows degradation curves
    (AUPRC vs pathology level) for every method.
    """
    agg = _aggregate(df, metric)
    ylabel = _metric_label(metric)
    n_pathologies = len(PATHOLOGY_ORDER)
    ncols = 4
    nrows = int(np.ceil(n_pathologies / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.flatten()
    for ax, pathology in zip(axes, PATHOLOGY_ORDER):
        sub = agg[agg["pathology"] == pathology]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("level")
            if len(m) == 0:
                continue
            x = m["level"].values
            y = m["mean"].values
            err = m["sem"].values
            ax.plot(x, y, "-o", color=METHOD_COLORS[method], label=method,
                    linewidth=1.5, markersize=4)
            ax.fill_between(x, y - err, y + err, color=METHOD_COLORS[method],
                            alpha=0.12, linewidth=0)
        ax.set_title(f"{PATHOLOGY_LABELS[pathology]}\n{PATHOLOGY_DESCRIPTIONS[pathology]}",
                     fontsize=9)
        ax.set_xlabel(PATHOLOGY_LABELS[pathology])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
    # legend in last empty axis
    for ax in axes[len(PATHOLOGY_ORDER):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o",
                          linestyle="-", label=m) for m in METHOD_ORDER]
    axes[-1].legend(handles=handles, loc="center", fontsize=10,
                    title="Method", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(f"Degradation Curves: {ylabel} as Each Pathology Intensifies",
                 fontsize=11)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

# Error-type decomposition at the hardest level of each pathology
def plot_error_decomp(df: pd.DataFrame, out_path: Path) -> None:
    """
    Stacked bars of TRUE / REVERSED / CONFOUNDED / SPURIOUS / MISSED per
    method, at the *highest* level of each pathology.
    """
    # For each pathology, grab the hardest level
    hardest = {p: max(s["levels"]) for p, s in SWEEPS.items()}
    n_p = len(PATHOLOGY_ORDER)
    ncols = 4
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows))
    axes = axes.flatten()
    for ax, pathology in zip(axes, PATHOLOGY_ORDER):
        level = hardest[pathology]
        sub = df[(df["pathology"] == pathology) & (df["level"] == level)]
        means = sub.groupby("method")[["err_TRUE", "err_REVERSED",
                                       "err_CONFOUNDED", "err_SPURIOUS",
                                       "err_MISSED"]].mean()
        means = means.reindex([m for m in METHOD_ORDER if m in means.index])
        x = np.arange(len(means))
        bottom = np.zeros(len(means))
        for cat in ["TRUE", "REVERSED", "CONFOUNDED", "SPURIOUS", "MISSED"]:
            vals = means[f"err_{cat}"].values
            ax.bar(x, vals, bottom=bottom, color=ERROR_COLORS[cat], label=cat,
                   width=0.75, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{PATHOLOGY_LABELS[pathology]} = {level}", fontsize=9)
        ax.set_ylabel("Edges (Count)")
        ax.grid(axis="x", alpha=0)
    for ax in axes[len(PATHOLOGY_ORDER):]:
        ax.axis("off")
    handles = [mpl.patches.Patch(color=c, label=k.capitalize()) for k, c in ERROR_COLORS.items()]
    axes[-1].legend(handles=handles, loc="center", title="Edge Type", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Error-Type Decomposition at the Hardest Level of Each Pathology",
                 fontsize=11)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

# Plot correlation vs. causal method families across pathologies
def plot_corr_vs_causal(df: pd.DataFrame, out_path: Path,
                        metric: str = "auprc_und") -> None:
    """
    Aggregate methods into families (correlation, tree, causal) and plot mean +/- sem of the
    chosen metric across pathology levels, faceted by pathology.
    """
    agg = _family_aggregate(df, metric)
    ylabel = _metric_label(metric)
    n_p = len(PATHOLOGY_ORDER)
    ncols = 4
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.flatten()
    for ax, pathology in zip(axes, PATHOLOGY_ORDER):
        sub = agg[agg["pathology"] == pathology]
        for family in FAMILY_ORDER:
            m = sub[sub["family"] == family].sort_values("level")
            if len(m) == 0:
                continue
            ax.plot(m["level"], m["mean"], "-o",
                    color=FAMILY_COLORS[family], label=family,
                    linewidth=2, markersize=4)
            ax.fill_between(m["level"], m["mean"] - m["sem"],
                            m["mean"] + m["sem"], color=FAMILY_COLORS[family],
                            alpha=0.15)
        ax.set_title(PATHOLOGY_LABELS[pathology], fontsize=9)
        ax.set_xlabel(PATHOLOGY_LABELS[pathology])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
    for ax in axes[len(PATHOLOGY_ORDER):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=FAMILY_COLORS[f], marker="o",
                          linestyle="-", label=FAMILY_LABELS[f])
               for f in FAMILY_ORDER]
    axes[-1].legend(handles=handles, loc="center", title="Method Class",
                    fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(f"Correlation vs. Causal Method Families Across Pathologies",
                 fontsize=11)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

# Runtime comparison across methods
def plot_runtime(df: pd.DataFrame, out_path: Path) -> None:
    """Mean runtime per method, faceted by pathology."""
    agg = df.groupby("method")["runtime_s"].agg(["mean", "std"]).reset_index()
    agg = agg.set_index("method").reindex(METHOD_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=agg["std"],
           color=[METHOD_COLORS[m] for m in agg["method"]],
           edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["method"], rotation=0)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Mean Runtime per Call (Averaged Across All Experiments)")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

def plot_error_decomp_normalized(df: pd.DataFrame, out_path: Path) -> None:
    """Normalized error-type composition at the most extreme level."""
    extreme = {p: _extreme_level(p) for p in PATHOLOGY_ORDER}
    n_p = len(PATHOLOGY_ORDER)
    ncols = 4
    nrows = int(np.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows))
    axes = axes.flatten()
    for ax, pathology in zip(axes, PATHOLOGY_ORDER):
        level = extreme[pathology]
        sub = df[(df["pathology"] == pathology) & (df["level"] == level)]
        means = sub.groupby("method")[["err_TRUE", "err_REVERSED",
                                       "err_CONFOUNDED", "err_SPURIOUS",
                                       "err_MISSED"]].mean()
        means = means.reindex([m for m in METHOD_ORDER if m in means.index])
        normalized = means.div(means.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        x = np.arange(len(normalized))
        bottom = np.zeros(len(normalized))
        for cat in ["TRUE", "REVERSED", "CONFOUNDED", "SPURIOUS", "MISSED"]:
            vals = normalized[f"err_{cat}"].values
            ax.bar(x, vals, bottom=bottom, color=ERROR_COLORS[cat], label=cat,
                   width=0.75, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(normalized.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{PATHOLOGY_LABELS[pathology]} = {level}", fontsize=9)
        ax.set_ylabel("Error Share")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", alpha=0)
    for ax in axes[len(PATHOLOGY_ORDER):]:
        ax.axis("off")
    handles = [mpl.patches.Patch(color=c, label=k.capitalize()) for k, c in ERROR_COLORS.items()]
    axes[-1].legend(handles=handles, loc="center", title="Edge Type", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Normalized Error-Type Composition at the Most Extreme Level",
                 fontsize=11)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

def plot_pareto_accuracy_runtime(df: pd.DataFrame, out_path: Path) -> None:
    """Pareto-style scatter of accuracy vs runtime."""
    agg = (df.groupby("method")
             .agg(
                 runtime_mean=("runtime_s", "mean"),
                 runtime_std=("runtime_s", "std"),
                 auprc_und_mean=("auprc_und", "mean"),
                 auprc_und_std=("auprc_und", "std"),
                 auprc_dir_mean=("auprc_dir", "mean"),
                 auprc_dir_std=("auprc_dir", "std"),
             )
             .reset_index())
    agg = agg.set_index("method").reindex(METHOD_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for _, row in agg.iterrows():
        method = row["method"]
        color = METHOD_COLORS[method]
        ax.errorbar(
            row["runtime_mean"], row["auprc_und_mean"],
            yerr=row["auprc_und_std"],
            fmt="o", color=color, markersize=7,
            elinewidth=1.2, capsize=3, alpha=0.95,
        )
        ax.errorbar(
            row["runtime_mean"], row["auprc_dir_mean"],
            yerr=row["auprc_dir_std"],
            fmt="s", color=color, markersize=7,
            elinewidth=1.2, capsize=3, alpha=0.70,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Runtime per Call (s, Log Scale)")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.02)
    method_handles = [mpl.patches.Patch(color=METHOD_COLORS[m], label=m)
                      for m in METHOD_ORDER]
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=7, label="Undirected"),
        plt.Line2D([0], [0], marker="s", color="gray", linestyle="None",
                   markersize=7, label="Directed"),
    ]
    fig.legend(
        handles=method_handles + shape_handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Pareto View of Accuracy vs. Runtime Across All Experiments",
                 fontsize=11)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

def tab1_summary(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Headline table: undirected + directed AUPRC at baseline level vs. hardest level per pathology, per method."""
    rows = []
    for pathology, spec in SWEEPS.items():
        baseline_level = spec["levels"][0]
        hardest_level = spec["levels"][-1]
        for method in METHOD_ORDER:
            base = df[
                (df.pathology == pathology)
                & (df.level == baseline_level)
                & (df.method == method)
            ]
            hard = df[
                (df.pathology == pathology)
                & (df.level == hardest_level)
                & (df.method == method)
            ]
            if len(base) == 0 or len(hard) == 0:
                continue
            rows.append({
                "pathology":            pathology,
                "method":               method,
                "baseline_auprc_und":   round(float(base["auprc_und"].mean()), 3),
                "hardest_auprc_und":    round(float(hard["auprc_und"].mean()), 3),
                "delta_auprc_und":      round(float(hard["auprc_und"].mean() - base["auprc_und"].mean()), 3),
                "baseline_auprc_dir":   round(float(base["auprc_dir"].mean()), 3),
                "hardest_auprc_dir":    round(float(hard["auprc_dir"].mean()), 3),
                "delta_auprc_dir":      round(float(hard["auprc_dir"].mean() - base["auprc_dir"].mean()), 3),
            })
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    return t

def tab2_breakdown(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Undirected + directed AUPRC mean +/- std at every (pathology, level, method)."""
    t = (df.groupby(["pathology", "level", "method"])
           .agg(
               auprc_und_mean=("auprc_und", "mean"),
               auprc_und_std=("auprc_und", "std"),
               auprc_dir_mean=("auprc_dir", "mean"),
               auprc_dir_std=("auprc_dir", "std"),
               count=("auprc_und", "count"),
           )
           .reset_index())
    for col in ["auprc_und_mean", "auprc_und_std",
                "auprc_dir_mean", "auprc_dir_std"]:
        t[col] = t[col].round(3)
    t.to_csv(out_path, index=False)
    return t

def plot_nonlinear_comparison(
    df_lin: pd.DataFrame,
    df_nonlin: pd.DataFrame,
    out_path: str | Path,
    metric: str = "auprc_und",
) -> None:
    """Degradation curves for linear (solid) and nonlinear (dashed) SCMs overlaid.

    Layout: 2 rows × 4 columns (7 pathology panels + 1 legend panel).
    Both SCMs share the same colour scheme; solid = linear, dashed = nonlinear.
    This compact layout fits within a standard paper column width.
    """
    out_path = Path(out_path)
    metric_label = _metric_label(metric)
    agg_lin    = _aggregate(df_lin,    metric)
    agg_nonlin = _aggregate(df_nonlin, metric)
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.8 * nrows),
                             sharey=True)
    axes = axes.flatten()
    for idx, pathology in enumerate(PATHOLOGY_ORDER):
        ax = axes[idx]
        levels = SWEEPS[pathology]["levels"]
        x = list(range(len(levels)))
        for method in METHOD_ORDER:
            color = METHOD_COLORS[method]
            m_lin = agg_lin[(agg_lin["pathology"] == pathology) &
                            (agg_lin["method"] == method)].sort_values("level")
            m_nonlin = agg_nonlin[(agg_nonlin["pathology"] == pathology) &
                                  (agg_nonlin["method"] == method)].sort_values("level")
            if not m_lin.empty:
                ax.plot(x, m_lin["mean"].values, "-o",
                        color=color, linewidth=1.4, markersize=3.5, alpha=0.95)
            if not m_nonlin.empty:
                ax.plot(x, m_nonlin["mean"].values, "--o",
                        color=color, linewidth=1.4, markersize=3.5, alpha=0.60)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lv) for lv in levels], fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(PATHOLOGY_LABELS[pathology], fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel(metric_label, fontsize=8)
    # Legend panel (8th slot)
    legend_ax = axes[len(PATHOLOGY_ORDER)]
    legend_ax.axis("off")
    method_handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m],
                                  linewidth=1.8, marker="o", markersize=4,
                                  label=m) for m in METHOD_ORDER]
    scm_handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.8,
                   linestyle="-",  label="Linear SCM"),
        plt.Line2D([0], [0], color="gray", linewidth=1.8,
                   linestyle="--", label="Nonlinear SCM (Tanh)"),
    ]
    legend_ax.legend(handles=method_handles + scm_handles,
                     loc="center", fontsize=8.5, frameon=False,
                     title="Method / SCM Type", title_fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(f"Linear vs. Nonlinear SCM: {metric_label} Degradation Curves",
                 fontsize=10)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)


# Interaction sweep figures (dropout x confounders x density)
def _interaction_aggregate(df_int: pd.DataFrame,
                           metric: str = "auprc_und") -> pd.DataFrame:
    """Mean over seeds at every (method, dropout, n_confounders, density)."""
    g = (df_int.groupby(["method", "dropout", "n_confounders", "density"])[metric]
                .mean().reset_index())
    return g

def plot_interaction_surfaces(df_int: pd.DataFrame, out_path: Path,
                              metric: str = "auprc_und") -> None:
    """Per-method failure surfaces over (dropout, n_confounders), faceted by density."""
    out_path = Path(out_path)
    agg = _interaction_aggregate(df_int, metric)
    drops = sorted(df_int["dropout"].unique())
    confs = sorted(df_int["n_confounders"].unique())
    dens  = sorted(df_int["density"].unique())
    methods = [m for m in METHOD_ORDER if m in df_int["method"].unique()]
    nrows, ncols = len(methods), len(dens)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.65 * ncols + 1.3, 1.55 * nrows + 0.8),
                             squeeze=False)
    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    for i, method in enumerate(methods):
        for j, rho in enumerate(dens):
            ax = axes[i, j]
            mat = np.full((len(drops), len(confs)), np.nan)
            for di, d in enumerate(drops):
                for ci, c in enumerate(confs):
                    sub = agg[(agg["method"] == method) &
                              (agg["dropout"] == d) &
                              (agg["n_confounders"] == c) &
                              (agg["density"] == rho)]
                    if len(sub):
                        mat[di, ci] = float(sub.iloc[0][metric])
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                           origin="lower")
            for di in range(len(drops)):
                for ci in range(len(confs)):
                    if np.isnan(mat[di, ci]):
                        continue
                    val = mat[di, ci]
                    txt_color = "white" if val < 0.5 else "black"
                    ax.text(ci, di, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color=txt_color)
            ax.set_xticks(range(len(confs)))
            ax.set_yticks(range(len(drops)))
            if i == nrows - 1:
                ax.set_xticklabels([str(c) for c in confs], fontsize=7)
                ax.set_xlabel("Confounders", fontsize=7.5)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels([f"{d:g}" for d in drops], fontsize=7)
                ax.set_ylabel(f"{method}\nDropout",
                              fontsize=8, color=METHOD_COLORS[method])
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(f"Density = {rho:g}", fontsize=8.5)
            ax.grid(False)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.08,
                        wspace=0.10, hspace=0.10)
    sp = fig.subplotpars
    cbar_ax = fig.add_axes([sp.right + 0.015, sp.bottom,
                            0.014, sp.top - sp.bottom])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(_metric_label(metric), fontsize=8.5)
    fig.suptitle("Joint Pathology Failure Surfaces (Dropout x Confounders, "
                 "Grouped by Density)", fontsize=10, y=0.97)
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

def plot_interaction_winner_map(df_int: pd.DataFrame, out_path: Path,
                                metric: str = "auprc_und") -> None:
    """Regime map: best method per (dropout, n_confounders) cell, per density slice."""
    out_path = Path(out_path)
    agg = _interaction_aggregate(df_int, metric)
    drops = sorted(df_int["dropout"].unique())
    confs = sorted(df_int["n_confounders"].unique())
    dens  = sorted(df_int["density"].unique())
    methods = [m for m in METHOD_ORDER if m in df_int["method"].unique()]
    cmap = mpl.colors.ListedColormap([METHOD_COLORS[m] for m in methods])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(methods) + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(1, len(dens),
                             figsize=(3.4 * len(dens) + 1.0, 3.8),
                             squeeze=False)
    axes = axes[0]
    for j, rho in enumerate(dens):
        ax = axes[j]
        win_idx = np.full((len(drops), len(confs)), np.nan)
        annot = [["" for _ in confs] for _ in drops]
        for di, d in enumerate(drops):
            for ci, c in enumerate(confs):
                sub = agg[(agg["dropout"] == d) &
                          (agg["n_confounders"] == c) &
                          (agg["density"] == rho)]
                if len(sub) == 0:
                    continue
                best = sub.sort_values([metric, "method"],
                                       ascending=[False, True]).iloc[0]
                m = best["method"]
                if m not in methods:
                    continue
                win_idx[di, ci] = methods.index(m)
                annot[di][ci] = f"{m}\n{best[metric]:.2f}"
        ax.imshow(np.ma.masked_invalid(win_idx), cmap=cmap, norm=norm,
                  aspect="auto", origin="lower")
        for di in range(len(drops)):
            for ci in range(len(confs)):
                if np.isnan(win_idx[di, ci]):
                    continue
                m = annot[di][ci].split("\n", 1)[0]
                tc = "white" if m in {"PC", "GES", "NOTEARS"} else "black"
                ax.text(ci, di, annot[di][ci], ha="center", va="center",
                        fontsize=8.5, color=tc)
        ax.set_xticks(range(len(confs)))
        ax.set_xticklabels([str(c) for c in confs], fontsize=8.5)
        ax.set_xlabel("Confounders", fontsize=9.5)
        ax.set_yticks(range(len(drops)))
        if j == 0:
            ax.set_yticklabels([f"{d:g}" for d in drops], fontsize=8.5)
            ax.set_ylabel("Dropout", fontsize=9.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"Density = {rho:g}", fontsize=10.5)
        ax.grid(False)
    handles = [mpl.patches.Patch(color=METHOD_COLORS[m], label=m)
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Best Method per (Dropout, Confounders) Interaction Regime, Grouped by Density",
                 fontsize=11, y=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)

def build_all(df: pd.DataFrame, fig_dir: str = "figures",
              tab_dir: str = "tables",
              df_nonlin: pd.DataFrame | None = None,
              df_interaction: pd.DataFrame | None = None) -> None:
    fig_dir = Path(fig_dir)
    tab_dir = Path(tab_dir)
    extra_dir = fig_dir / "additional"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    extra_dir.mkdir(parents=True, exist_ok=True)

    print("Building headline grid...")
    plot_headline_grid(df, fig_dir / "headline_grid.png")
    print("Building headline grid (directed)...")
    plot_headline_grid(df, fig_dir / "headline_grid_directed.png",
                       metric="auprc_dir")
    print("Building error decomposition...")
    plot_error_decomp(df, fig_dir / "error_decomp.png")
    print("Building correlation vs causal...")
    plot_corr_vs_causal(df, fig_dir / "corr_vs_causal.png")
    print("Building correlation vs causal (directed)...")
    plot_corr_vs_causal(df, fig_dir / "corr_vs_causal_directed.png",
                        metric="auprc_dir")
    print("Building runtime...")
    plot_runtime(df, fig_dir / "runtime.png")
    print("Building normalized error decomposition...")
    plot_error_decomp_normalized(df, extra_dir / "error_decomp_normalized.png")
    print("Building pareto accuracy vs runtime...")
    plot_pareto_accuracy_runtime(df, extra_dir / "pareto_accuracy_runtime.png")
    print("Building table 1 (summary)...")
    tab1_summary(df, tab_dir / "tab1_summary.csv")
    print("Building table 2 (breakdown)...")
    tab2_breakdown(df, tab_dir / "tab2_breakdown.csv")
    if df_nonlin is not None:
        print("Building nonlinear comparison...")
        plot_nonlinear_comparison(df, df_nonlin,
                                 extra_dir / "nonlinear_comparison.png")
        print("Building nonlinear comparison (directed)...")
        plot_nonlinear_comparison(df, df_nonlin,
                                 extra_dir / "nonlinear_comparison_directed.png",
                                 metric="auprc_dir")
    if df_interaction is not None:
        print("Building interaction surfaces...")
        plot_interaction_surfaces(df_interaction,
                                  fig_dir / "interaction_surfaces.png")
        print("Building interaction surfaces (directed)...")
        plot_interaction_surfaces(df_interaction,
                                  extra_dir / "interaction_surfaces_directed.png",
                                  metric="auprc_dir")
        print("Building interaction winner map...")
        plot_interaction_winner_map(df_interaction,
                                    fig_dir / "interaction_winner_map.png")
    print("Done.")
