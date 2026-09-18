"""
10_oracle_bootstrap.py

Added in response to Array reviewer feedback (Reviewer 1, point 6; Reviewer 4,
"Impact of Poor Prediction Accuracy on Scheduling Outcomes"): both reviewers
independently ask the same question. RF_SRPT_proxy's ordering is driven by a
predictor with a 130.7-minute MAE against a 151.9-minute mean true duration
(Section 6.1). Is prediction failing to beat backfilling because the concept
of prediction-informed ordering does not help on this workload, or simply
because this specific predictor is too weak to test that concept fairly?

This script answers that by adding one more policy, Oracle_SRPT: identical to
RF_SRPT_proxy in every respect (ascending-duration ordering, first-fit
placement, backfilling enabled) except that the ordering key is the job's
TRUE recorded duration (true_duration, from jobs_clean.parquet) instead of
the trained predictor's output. This is a theoretical upper bound: no real
predictor can do better than perfect foreknowledge. If Oracle_SRPT still adds
little over Backfill_FIFO, that is strong evidence the ceiling on prediction's
value here is set by the workload, not by predictor quality. If Oracle_SRPT
clears a bar RF_SRPT_proxy does not, that points the other way, at predictor
quality specifically.

Reproducibility note: to make the Oracle_SRPT bootstrap directly, per-replicate
comparable to the existing Backfill_FIFO and RF_SRPT_proxy bootstrap already
in processed/bootstrap_raw.csv, this script reproduces the EXACT same
job-resampling sequence as 06_bootstrap.py: same jobs list construction, same
np.random.default_rng(12345) seed, same host_configs order ([120, 64]), same
300 replicates, same resample_jobs() calls in the same order. That means
boot_rep b at a given n_hosts here is the identical resampled job set as
boot_rep b in bootstrap_raw.csv, so Oracle_SRPT can be validly paired against
the policies already bootstrapped there without re-running them.

Only the Oracle_SRPT policy is simulated here (not the other five), since
those already exist in bootstrap_raw.csv under the identical resample
sequence -- re-running them would waste compute and, done wrong, would risk
a subtly different resample sequence that breaks the pairing.
"""

import json
import os

import numpy as np
import pandas as pd

import importlib.util

spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

PROCESSED = "processed"
N_BOOT = 300
HOST_CONFIGS = [120, 64]


def oracle_srpt_key(j):
    return j["true_duration"]


def resample_jobs(jobs, rng):
    # identical to 06_bootstrap.py's resample_jobs -- must not diverge, since
    # the whole point is byte-for-byte reproducing that script's rng draws.
    idx = rng.integers(0, len(jobs), size=len(jobs))
    resampled = [jobs[i] for i in idx]
    resampled.sort(key=lambda j: j["arrival_min"])
    return resampled


def main():
    jobs, t0 = sim03.load_jobs()
    rng = np.random.default_rng(12345)  # same seed as 06_bootstrap.py

    all_rows = []
    for n_hosts in HOST_CONFIGS:
        hosts_template = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        print(f"\n=== Oracle_SRPT bootstrapping at {n_hosts} hosts, {N_BOOT} replicates ===")
        for b in range(N_BOOT):
            boot_jobs = resample_jobs(jobs, rng)
            hosts = [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts_template]
            r = sim03.run_policy(boot_jobs, hosts, oracle_srpt_key, sim03.first_fit_place,
                                  allow_backfill=True, name="Oracle_SRPT")
            r["n_hosts"] = n_hosts
            r["boot_rep"] = b
            all_rows.append(r)
        print("  done")

    oracle_df = pd.DataFrame(all_rows)
    oracle_df.to_csv(os.path.join(PROCESSED, "oracle_bootstrap_raw.csv"), index=False)

    # Sanity check the pairing assumption before trusting any comparison: the
    # existing bootstrap_raw.csv's per-replicate job counts should line up with
    # Oracle_SRPT's, since they came from the same resampled job set.
    existing = pd.read_csv(os.path.join(PROCESSED, "bootstrap_raw.csv"))

    summary_rows = []
    comparisons = []
    for n_hosts in HOST_CONFIGS:
        osub = oracle_df[oracle_df["n_hosts"] == n_hosts].set_index("boot_rep")
        esub = existing[existing["n_hosts"] == n_hosts]

        # pairing sanity check: n_jobs_completed for Oracle_SRPT must match
        # Backfill_FIFO's per replicate, since both backfill and both see the
        # identical resampled job set (no jobs should be structurally
        # unplaceable under one policy and not the other at the same n_hosts).
        bf = esub[esub["policy"] == "Backfill_FIFO"].set_index("boot_rep")
        mismatch = (osub["n_jobs_completed"] != bf["n_jobs_completed"]).sum()
        if mismatch:
            raise RuntimeError(
                f"Pairing check FAILED at n_hosts={n_hosts}: {mismatch} replicates have a "
                f"different completed-job count between Oracle_SRPT and Backfill_FIFO. "
                f"The resample sequences do not match; do not trust the paired comparison."
            )
        print(f"Pairing check passed at n_hosts={n_hosts}: job-set alignment confirmed across {N_BOOT} replicates.")

        vals = osub["avg_jct_min"].values
        summary_rows.append({
            "n_hosts": n_hosts, "policy": "Oracle_SRPT",
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "ci_2.5%": float(np.percentile(vals, 2.5)), "ci_97.5%": float(np.percentile(vals, 97.5)),
        })

        rf = esub[esub["policy"] == "RF_SRPT_proxy"].set_index("boot_rep")

        oracle_vs_backfill = osub["avg_jct_min"] - bf["avg_jct_min"]
        rf_vs_oracle = rf["avg_jct_min"] - osub["avg_jct_min"]

        comparisons.append({
            "n_hosts": n_hosts,
            "Oracle_minus_Backfill_mean": float(oracle_vs_backfill.mean()),
            "Oracle_minus_Backfill_ci": (float(oracle_vs_backfill.quantile(0.025)), float(oracle_vs_backfill.quantile(0.975))),
            "pct_reps_Oracle_beats_Backfill": float((oracle_vs_backfill < 0).mean() * 100),
            "RF_SRPT_minus_Oracle_mean": float(rf_vs_oracle.mean()),
            "RF_SRPT_minus_Oracle_ci": (float(rf_vs_oracle.quantile(0.025)), float(rf_vs_oracle.quantile(0.975))),
            "pct_reps_RF_SRPT_beats_Oracle": float((rf_vs_oracle < 0).mean() * 100),
        })

    summary = pd.DataFrame(summary_rows)
    print("\n=== Oracle_SRPT bootstrap summary: avg JCT mean and 95% CI ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(PROCESSED, "oracle_bootstrap_summary.csv"), index=False)

    comp_df = pd.DataFrame(comparisons)
    print("\n=== Oracle_SRPT paired comparisons (negative = left side has lower/better JCT) ===")
    for row in comparisons:
        print(json.dumps(row, indent=2, default=str))
    comp_df.to_csv(os.path.join(PROCESSED, "oracle_bootstrap_comparisons.csv"), index=False)
    print(f"\nWrote oracle_bootstrap_raw.csv, oracle_bootstrap_summary.csv, oracle_bootstrap_comparisons.csv")


if __name__ == "__main__":
    main()
