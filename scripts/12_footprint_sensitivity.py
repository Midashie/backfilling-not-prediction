"""
12_footprint_sensitivity.py

Added in response to Array Reviewer 1, point 5: "The trace does not contain
an explicit requested-resource field, and the manuscript substitutes the
realized worker allocation... A sensitivity analysis excluding jobs for
which realized allocation may plausibly differ from the original request
would help determine whether this assumption affects queue feasibility or
placement outcomes."

Operationalization, stated plainly because it is the only one this trace
actually supports: the only signal available for "realized allocation may
differ from what was requested" is job kind. ElasticBatchJob is the one
kind in this test set whose name itself implies elastic (resizable at
runtime) resource allocation; PyTorchJob (92.4% of the test set) and TFJob
are not documented as elastic in this trace and have no comparable signal
suggesting their realized worker count would diverge from what was
requested. This is a narrow, honestly-bounded test: ElasticBatchJob is only
5 of 656 test-set jobs (0.76%), so this check has limited power to detect
an effect even if one existed, and that limitation is reported rather than
hidden. A stronger version of this check is not possible without a
requested-resource field this trace does not provide.

Method: rerun the same 300-replicate job-level bootstrap as 06_bootstrap.py,
at the same two host configs (120, 64), on the 651-job test set with the
5 ElasticBatchJob jobs removed, and compare the two headline bootstrap
comparisons (Backfill_FIFO vs. FIFO; RF_SRPT_proxy vs. Backfill_FIFO)
against the full-656-job results already in bootstrap_comparisons.csv. This
is a fresh, independently-seeded bootstrap (not paired against the original
per-replicate, since the job set itself is now a different population, 651
vs. 656 jobs, so index-level pairing is not meaningful here); the question
this answers is whether the AGGREGATE conclusion changes, not whether any
single replicate's outcome shifts.
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
    print(f"Full test set: {len(jobs)} jobs")

    preds = pd.read_parquet(os.path.join(PROCESSED, "predictor_test_predictions.parquet"))
    elastic_ids = set(preds.loc[preds["kind"] == "ElasticBatchJob", "id"].astype(int))
    print(f"Excluding {len(elastic_ids)} ElasticBatchJob jobs (kind == 'ElasticBatchJob'): {sorted(elastic_ids)}")

    filtered_jobs = [j for j in jobs if j["id"] not in elastic_ids]
    print(f"Filtered test set: {len(filtered_jobs)} jobs "
          f"({len(jobs) - len(filtered_jobs)} removed, "
          f"{100 * (len(jobs) - len(filtered_jobs)) / len(jobs):.2f}% of the test set)")
    assert len(jobs) - len(filtered_jobs) == len(elastic_ids)

    rng = np.random.default_rng(20260823)  # independent seed; different job population, no pairing intended

    all_rows = []
    for n_hosts in HOST_CONFIGS:
        hosts = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        print(f"\n=== Footprint-sensitivity bootstrapping at {n_hosts} hosts, {N_BOOT} replicates "
              f"({len(filtered_jobs)}-job filtered test set) ===")
        for b in range(N_BOOT):
            boot_jobs = resample_jobs(filtered_jobs, rng)
            hosts_copy = [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts]
            results = sim03.run_all_policies(boot_jobs, hosts_copy)
            for r in results:
                r["n_hosts"] = n_hosts
                r["boot_rep"] = b
            all_rows.extend(results)
        print("  done")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(PROCESSED, "footprint_sensitivity_bootstrap_raw.csv"), index=False)

    comparisons = []
    for n_hosts in HOST_CONFIGS:
        sub = df[df["n_hosts"] == n_hosts]
        piv = sub.pivot(index="boot_rep", columns="policy", values="avg_jct_min")
        backfill_vs_fifo = piv["Backfill_FIFO"] - piv["FIFO"]
        rf_vs_backfill = piv["RF_SRPT_proxy"] - piv["Backfill_FIFO"]
        comparisons.append({
            "n_hosts": n_hosts,
            "n_jobs": len(filtered_jobs),
            "Backfill_minus_FIFO_mean": float(backfill_vs_fifo.mean()),
            "Backfill_minus_FIFO_ci": (float(backfill_vs_fifo.quantile(0.025)), float(backfill_vs_fifo.quantile(0.975))),
            "pct_reps_Backfill_beats_FIFO": float((backfill_vs_fifo < 0).mean() * 100),
            "RF_SRPT_minus_Backfill_mean": float(rf_vs_backfill.mean()),
            "RF_SRPT_minus_Backfill_ci": (float(rf_vs_backfill.quantile(0.025)), float(rf_vs_backfill.quantile(0.975))),
            "pct_reps_RF_SRPT_beats_Backfill": float((rf_vs_backfill < 0).mean() * 100),
        })

    comp_df = pd.DataFrame(comparisons)
    print("\n=== Footprint-sensitivity paired comparisons (651-job filtered test set, negative = left side better) ===")
    for row in comparisons:
        print(json.dumps(row, indent=2, default=str))
    comp_df.to_csv(os.path.join(PROCESSED, "footprint_sensitivity_comparisons.csv"), index=False)

    print("\n=== For comparison, the full-656-job results already in bootstrap_comparisons.csv ===")
    orig = pd.read_csv(os.path.join(PROCESSED, "bootstrap_comparisons.csv"))
    print(orig[["n_hosts", "Backfill_minus_FIFO_mean", "Backfill_minus_FIFO_ci",
                "pct_reps_Backfill_beats_FIFO", "RF_SRPT_minus_Backfill_mean",
                "RF_SRPT_minus_Backfill_ci", "pct_reps_RF_SRPT_beats_Backfill"]].to_string(index=False))

    print("\nWrote footprint_sensitivity_bootstrap_raw.csv, footprint_sensitivity_comparisons.csv")


if __name__ == "__main__":
    main()
