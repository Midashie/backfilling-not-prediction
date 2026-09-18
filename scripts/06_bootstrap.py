"""
06_bootstrap.py

Job-level bootstrap to put a confidence interval on the policy differences
reported in 03_simulate.py and 05_sweep.py, instead of the single
deterministic run those scripts report. Method: resample 656 jobs with
replacement from the real 656-job test set (each resampled job keeps its
real arrival time, true duration, resource footprint, and predictions),
re-sort by arrival time, and re-run the full discrete-event simulation.
Repeated 300 times per cluster size (raised from an initial 60 after that
smaller run flagged a tail-sensitivity issue at the 64-host config, where
the point estimate fell just outside its own reported 95% CI, a sign 60
replicates was not enough to pin down that boundary). This is a standard
job-level bootstrap for trace-driven simulation; it treats jobs as the
resampling unit, which is an approximation (real job arrivals are not
fully independent), stated here rather than left implicit.

Two cluster sizes are bootstrapped: 120 hosts (the original headline
comparison point) and 64 hosts (the extreme-contention point the sweep in
05_sweep.py found, where strict FIFO collapses catastrophically and every
backfilling policy, prediction-informed or not, stays in a tight band).
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

PROCESSED = "processed"
N_BOOT = 300
HOST_CONFIGS = [120, 64]


def resample_jobs(jobs, rng):
    idx = rng.integers(0, len(jobs), size=len(jobs))
    resampled = [jobs[i] for i in idx]
    resampled.sort(key=lambda j: j["arrival_min"])
    return resampled


def main():
    jobs, t0 = sim03.load_jobs()
    rng = np.random.default_rng(12345)

    all_rows = []
    for n_hosts in HOST_CONFIGS:
        hosts = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        print(f"\n=== Bootstrapping at {n_hosts} hosts, {N_BOOT} replicates ===")
        for b in range(N_BOOT):
            boot_jobs = resample_jobs(jobs, rng)
            results = sim03.run_all_policies(boot_jobs, [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts])
            for r in results:
                r["n_hosts"] = n_hosts
                r["boot_rep"] = b
            all_rows.extend(results)
        print(f"  done")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(PROCESSED, "bootstrap_raw.csv"), index=False)

    summary_rows = []
    for n_hosts in HOST_CONFIGS:
        sub = df[df["n_hosts"] == n_hosts]
        for policy in sub["policy"].unique():
            vals = sub[sub["policy"] == policy]["avg_jct_min"].values
            summary_rows.append({
                "n_hosts": n_hosts, "policy": policy,
                "mean": np.mean(vals), "std": np.std(vals),
                "ci_2.5%": np.percentile(vals, 2.5), "ci_97.5%": np.percentile(vals, 97.5),
            })
    summary = pd.DataFrame(summary_rows)
    print("\n=== Bootstrap summary: avg JCT mean and 95% CI ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(PROCESSED, "bootstrap_summary.csv"), index=False)

    # Paired comparisons per replicate: does PredSched_LLM beat RF_SRPT_proxy, and
    # does RF_SRPT_proxy (prediction) beat Backfill_FIFO (backfill only, no prediction)?
    # (column names below were briefly "Pollux" during an earlier version of this
    # script, before Backfill_FIFO was renamed away from that name -- see 03_simulate.py;
    # fixed here so the shipped CSV cannot contradict the paper's own Section 5.2 text.)
    comparisons = []
    for n_hosts in HOST_CONFIGS:
        sub = df[df["n_hosts"] == n_hosts]
        piv = sub.pivot(index="boot_rep", columns="policy", values="avg_jct_min")
        pred_vs_rf = piv["PredSched_LLM"] - piv["RF_SRPT_proxy"]
        rf_vs_backfill = piv["RF_SRPT_proxy"] - piv["Backfill_FIFO"]
        backfill_vs_fifo = piv["Backfill_FIFO"] - piv["FIFO"]
        comparisons.append({
            "n_hosts": n_hosts,
            "PredSched_minus_RF_SRPT_mean": pred_vs_rf.mean(), "PredSched_minus_RF_SRPT_ci": (pred_vs_rf.quantile(0.025), pred_vs_rf.quantile(0.975)),
            "pct_reps_PredSched_beats_RF_SRPT": float((pred_vs_rf < 0).mean() * 100),
            "RF_SRPT_minus_Backfill_mean": rf_vs_backfill.mean(), "RF_SRPT_minus_Backfill_ci": (rf_vs_backfill.quantile(0.025), rf_vs_backfill.quantile(0.975)),
            "pct_reps_RF_SRPT_beats_Backfill": float((rf_vs_backfill < 0).mean() * 100),
            "Backfill_minus_FIFO_mean": backfill_vs_fifo.mean(), "Backfill_minus_FIFO_ci": (backfill_vs_fifo.quantile(0.025), backfill_vs_fifo.quantile(0.975)),
            "pct_reps_Backfill_beats_FIFO": float((backfill_vs_fifo < 0).mean() * 100),
        })
    comp_df = pd.DataFrame(comparisons)
    print("\n=== Paired bootstrap comparisons (negative = left side has lower/better JCT) ===")
    for row in comparisons:
        print(json.dumps(row, indent=2, default=str))
    comp_df.to_csv(os.path.join(PROCESSED, "bootstrap_comparisons.csv"), index=False)
    print(f"\nWrote bootstrap_raw.csv, bootstrap_summary.csv, bootstrap_comparisons.csv")


if __name__ == "__main__":
    main()
