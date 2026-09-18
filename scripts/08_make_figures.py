"""
08_make_figures.py

Builds the three figures cited in Section 6.2 and Section 6.4 of the paper,
from the real sweep, bootstrap, and cross-trace output files already on disk
(processed/sweep_results.csv, processed/bootstrap_raw.csv,
processed/crosstrace_helios_comparisons.csv). No synthetic or illustrative
data; every point plotted is a real simulation result.

Figure 1: average JCT vs. cluster size across the full 19-point sweep, for
the four policies discussed in the text (FIFO, Backfill_FIFO, RF_SRPT_proxy,
PredSched_LLM). Left panel: full sweep, log-scale y-axis, since FIFO's
collapse spans roughly a 9x range while the other three stay in a narrow
band. Right panel: linear-scale zoom on the contended region (150 to 64
hosts) where the policies actually separate.

Figure 2: bootstrap mean average JCT with 95% CI, at 120 hosts and 64 hosts,
for the same four policies, as a forest-style error-bar plot (300-replicate
job-level bootstrap, Section 6.2).

Figure 3: cross-trace check on all four Helios clusters at their constrained
(feasibility-floor) point (Section 6.4): backfilling vs. FIFO, and
prediction-informed ordering vs. backfilling-only, each as a paired
bootstrap mean difference with 95% CI (40-replicate bootstrap per cluster).

Title and axis-label text uses black as the main font color throughout;
color is reserved for the data-encoding markers/lines only.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED = "processed"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

TEXT_COLOR = "#000000"

PURPLE = "#5B2A86"
PURPLE_LIGHT = "#9B6FC4"
PURPLE_DARK = "#3A1B5C"
GREY = "#8A8A8A"
ACCENT = "#C2185B"

POLICY_STYLE = {
    "FIFO": {"color": GREY, "marker": "o", "linestyle": "--"},
    "Backfill_FIFO": {"color": PURPLE_LIGHT, "marker": "s", "linestyle": "-"},
    "RF_SRPT_proxy": {"color": PURPLE, "marker": "^", "linestyle": "-"},
    "PredSched_LLM": {"color": ACCENT, "marker": "D", "linestyle": "-"},
}
POLICY_LABEL = {
    "FIFO": "FIFO",
    "Backfill_FIFO": "Backfill_FIFO",
    "RF_SRPT_proxy": "RF_SRPT_proxy",
    "PredSched_LLM": "PredSched_LLM",
}


def fig1_sweep_curve():
    df = pd.read_csv(os.path.join(PROCESSED, "sweep_results.csv"))
    policies = ["FIFO", "PredSched_LLM", "Backfill_FIFO", "RF_SRPT_proxy"]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)

    for pol in policies:
        sub = df[df["policy"] == pol].sort_values("n_hosts")
        style = POLICY_STYLE[pol]
        ax_full.plot(sub["n_hosts"], sub["avg_jct_min"], label=POLICY_LABEL[pol],
                     color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=5, alpha=0.95)

        zoom_sub = sub[sub["n_hosts"] <= 150]
        ax_zoom.plot(zoom_sub["n_hosts"], zoom_sub["avg_jct_min"],
                     color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=6, alpha=0.95)

    ax_full.set_yscale("log")
    ax_full.invert_xaxis()
    ax_full.set_xlabel("Cluster size (hosts, 8 GPUs/host)", color=TEXT_COLOR)
    ax_full.set_ylabel("Average job completion time (min, log scale)", color=TEXT_COLOR)
    ax_full.set_title("Full sweep (615 to 64 hosts)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_full.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_full.spines["top"].set_visible(False)
    ax_full.spines["right"].set_visible(False)
    ax_full.legend(frameon=False, loc="upper left", labelcolor=TEXT_COLOR)

    ax_zoom.invert_xaxis()
    ax_zoom.set_xlabel("Cluster size (hosts)", color=TEXT_COLOR)
    ax_zoom.set_ylabel("Average JCT (min, linear)", color=TEXT_COLOR)
    ax_zoom.set_title("Zoom: contended region (150 to 64 hosts)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax_zoom.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_zoom.spines["top"].set_visible(False)
    ax_zoom.spines["right"].set_visible(False)

    fig.suptitle("Average JCT across the full cluster-size sweep\n(Alibaba 2023 trace, 656-job test set)",
                  color=TEXT_COLOR, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIGDIR, "fig1_sweep_curve.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig2_bootstrap_ci():
    df = pd.read_csv(os.path.join(PROCESSED, "bootstrap_raw.csv"))
    policies = ["FIFO", "Backfill_FIFO", "RF_SRPT_proxy", "PredSched_LLM"]
    host_configs = sorted(df["n_hosts"].unique(), reverse=True)
    n_replicates = df["boot_rep"].max() + 1

    fig, axes = plt.subplots(1, len(host_configs), figsize=(9.5, 4.2), dpi=200, sharey=False)
    if len(host_configs) == 1:
        axes = [axes]

    for ax, n_hosts in zip(axes, host_configs):
        sub_hosts = df[df["n_hosts"] == n_hosts]
        means, los, his, colors, labels = [], [], [], [], []
        for pol in policies:
            vals = sub_hosts[sub_hosts["policy"] == pol]["avg_jct_min"].values
            mean = np.mean(vals)
            lo = np.percentile(vals, 2.5)
            hi = np.percentile(vals, 97.5)
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)
            colors.append(POLICY_STYLE[pol]["color"])
            labels.append(POLICY_LABEL[pol])

        ypos = np.arange(len(policies))[::-1]
        ax.errorbar(means, ypos, xerr=[los, his], fmt="none", ecolor=PURPLE_DARK,
                     elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
        ax.scatter(means, ypos, c=colors, s=70, zorder=3, edgecolor=PURPLE_DARK, linewidth=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, color=TEXT_COLOR)
        ax.set_xlabel("Average JCT (min)", color=TEXT_COLOR)
        ax.set_title(f"{n_hosts} hosts", color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Bootstrap mean avg JCT with 95% CI ({n_replicates}-replicate job-level bootstrap)",
                  color=TEXT_COLOR, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, "fig2_bootstrap_ci.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig3_crosstrace_helios():
    comp = pd.read_csv(os.path.join(PROCESSED, "crosstrace_helios_comparisons.csv"))
    # each cluster has two rows (its constrained feasibility-floor host count,
    # and a larger uncontended host count); keep only the constrained one
    floor_rows = comp.loc[comp.groupby("cluster")["n_hosts"].idxmin()]
    cluster_order = ["Earth", "Saturn", "Uranus", "Venus"]
    floor_rows = floor_rows.set_index("cluster").loc[cluster_order].reset_index()
    cluster_colors = [PURPLE_DARK, PURPLE_LIGHT, ACCENT, PURPLE]

    fig, (ax_bf, ax_pred) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    ypos = np.arange(len(cluster_order))[::-1]
    labels = [f"{c} ({int(h)} hosts)" for c, h in zip(floor_rows["cluster"], floor_rows["n_hosts"])]

    means_bf = floor_rows["Backfill_minus_FIFO_mean"].values
    los_bf = means_bf - floor_rows["Backfill_minus_FIFO_ci_lo"].values
    his_bf = floor_rows["Backfill_minus_FIFO_ci_hi"].values - means_bf
    ax_bf.errorbar(means_bf, ypos, xerr=[los_bf, his_bf], fmt="none", ecolor=PURPLE_DARK,
                   elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
    ax_bf.scatter(means_bf, ypos, c=cluster_colors, s=90, zorder=3, edgecolor=PURPLE_DARK, linewidth=0.6)
    ax_bf.axvline(0, color="#AAAAAA", linestyle=":", linewidth=1.0, zorder=1)
    ax_bf.set_yticks(ypos)
    ax_bf.set_yticklabels(labels, color=TEXT_COLOR)
    ax_bf.set_xlabel("Backfill_FIFO minus FIFO,\navg JCT diff (min)", color=TEXT_COLOR)
    ax_bf.set_title("Backfilling vs. FIFO", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax_bf.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_bf.spines["top"].set_visible(False)
    ax_bf.spines["right"].set_visible(False)

    means_pr = floor_rows["RF_SRPT_minus_Backfill_mean"].values
    los_pr = means_pr - floor_rows["RF_SRPT_minus_Backfill_ci_lo"].values
    his_pr = floor_rows["RF_SRPT_minus_Backfill_ci_hi"].values - means_pr
    ax_pred.errorbar(means_pr, ypos, xerr=[los_pr, his_pr], fmt="none", ecolor=PURPLE_DARK,
                     elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
    ax_pred.scatter(means_pr, ypos, c=cluster_colors, s=90, zorder=3, edgecolor=PURPLE_DARK, linewidth=0.6)
    ax_pred.axvline(0, color="#AAAAAA", linestyle=":", linewidth=1.0, zorder=1)
    ax_pred.set_yticks(ypos)
    ax_pred.set_yticklabels(labels, color=TEXT_COLOR)
    ax_pred.set_xlabel("RF_SRPT_proxy minus Backfill_FIFO,\navg JCT diff (min)", color=TEXT_COLOR)
    ax_pred.set_title("Prediction vs. backfilling-only", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax_pred.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)

    fig.suptitle(
        "Cross-trace check: Helios clusters at their constrained (feasibility-floor) point\n"
        "(40-replicate bootstrap per cluster; negative = left side has lower/better JCT)",
        color=TEXT_COLOR, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = os.path.join(FIGDIR, "fig3_crosstrace_helios.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig1_sweep_curve()
    fig2_bootstrap_ci()
    fig3_crosstrace_helios()
