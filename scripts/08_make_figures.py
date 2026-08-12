"""
08_make_figures.py

Builds the two figures cited in Section 6.2 of the paper, from the real
sweep and bootstrap output files already on disk (processed/sweep_results.csv,
processed/bootstrap_raw.csv). No synthetic or illustrative data; every point
plotted is a real simulation result.

Figure 1: average JCT vs. cluster size across the full 19-point sweep, for
the four policies discussed in the text (FIFO, Backfill_FIFO, RF_SRPT_proxy,
PredSched_LLM). Log-scale y-axis, since FIFO's collapse spans roughly a 9x
range while the other three stay in a narrow band.

Figure 2: bootstrap mean average JCT with 95% CI, at 120 hosts and 64 hosts,
for the same four policies, as a forest-style error-bar plot.
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
    policies = ["FIFO", "Backfill_FIFO", "RF_SRPT_proxy", "PredSched_LLM"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    for pol in policies:
        sub = df[df["policy"] == pol].sort_values("n_hosts")
        style = POLICY_STYLE[pol]
        ax.plot(sub["n_hosts"], sub["avg_jct_min"], label=POLICY_LABEL[pol],
                color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                linewidth=1.8, markersize=5, alpha=0.95)

    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Cluster size (hosts, 8 GPUs/host)")
    ax.set_ylabel("Average job completion time (min, log scale)")
    ax.set_title("Average JCT across the full cluster-size sweep\n(Alibaba 2023 trace, 656-job test set)",
                  color=PURPLE_DARK, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_sweep_curve.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig2_bootstrap_ci():
    df = pd.read_csv(os.path.join(PROCESSED, "bootstrap_raw.csv"))
    policies = ["FIFO", "Backfill_FIFO", "RF_SRPT_proxy", "PredSched_LLM"]
    host_configs = sorted(df["n_hosts"].unique(), reverse=True)

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
        ax.set_yticklabels(labels)
        ax.set_xlabel("Average JCT (min)")
        ax.set_title(f"{n_hosts} hosts", color=PURPLE_DARK, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Bootstrap mean avg JCT with 95% CI (60-replicate job-level bootstrap)",
                  color=PURPLE_DARK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, "fig2_bootstrap_ci.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig1_sweep_curve()
    fig2_bootstrap_ci()
