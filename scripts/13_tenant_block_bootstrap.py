"""
13_tenant_block_bootstrap.py

Added in response to Array Reviewer 1, point 3: "Individual jobs are
resampled as if they were independent observations, even though production
workloads often contain bursty arrivals, repeated tenant behavior, related
training runs, and temporal dependence... A block bootstrap over time
windows, tenant-level resampling, or both would provide a much stronger
test of whether the scheduling differences survive realistic workload
dependence."

06_bootstrap.py resamples individual jobs with replacement, which treats
each job as an independent draw. If jobs from the same tenant are
correlated (similar durations, correlated arrival timing, repeated training
runs), that independence assumption understates the true uncertainty:
job-level resampling can make the bootstrap distribution look tighter than
it really is.

This script instead resamples at the TENANT level: the 656-job test set
has 17 distinct tenant_id values (attached here from jobs_clean.parquet,
which 03_simulate.py's job dicts do not otherwise carry). Each replicate
draws 17 tenants WITH replacement and takes the union of every job
belonging to each drawn tenant (a tenant drawn twice contributes its jobs
twice), which preserves whatever within-tenant correlation exists instead
of breaking it apart. This is a real, if imperfect, test: with only 17
tenants to draw from, this is a small number of clusters for a block
bootstrap (a acknowledged limitation, stated here rather than hidden), and
because tenants vary enormously in job count (the largest carries 154 of
656 jobs, the smallest a handful), replicate-to-replicate job counts vary a
lot more here than in the job-level bootstrap, itself a feature of this
design, not a bug: it is exactly the extra variability the reviewer is
asking whether the paper's conclusions survive.

Only the three policies needed for the two headline comparisons are run
here (FIFO, Backfill_FIFO, RF_SRPT_proxy), not all six, to keep runtime
in check; this check is about whether the CONCLUSIONS survive
tenant-clustered resampling, not about re-deriving every comparison in the
paper under it.
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


def load_jobs_with_tenant():
    jobs, t0 = sim03.load_jobs()
    clean = pd.read_parquet(os.path.join(PROCESSED, "jobs_clean.parquet"))
    id_to_tenant = dict(zip(clean["id"].astype(int), clean["tenant_id"]))
    for j in jobs:
        j["tenant_id"] = id_to_tenant[j["id"]]
    return jobs, t0


def build_tenant_index(jobs):
    tenant_to_jobs = {}
    for j in jobs:
        tenant_to_jobs.setdefault(j["tenant_id"], []).append(j)
    return tenant_to_jobs


def resample_tenant_block(tenant_ids, tenant_to_jobs, rng):
    drawn = rng.choice(tenant_ids, size=len(tenant_ids), replace=True)
    pool = []
    for t in drawn:
        pool.extend(tenant_to_jobs[t])
    pool.sort(key=lambda j: j["arrival_min"])
    return pool


def main():
    jobs, t0 = load_jobs_with_tenant()
    tenant_to_jobs = build_tenant_index(jobs)
    tenant_ids = sorted(tenant_to_jobs.keys())
    n_tenants = len(tenant_ids)
    tenant_sizes = {t: len(v) for t, v in tenant_to_jobs.items()}
    print(f"Full test set: {len(jobs)} jobs across {n_tenants} tenants")
    print(f"Tenant sizes (jobs per tenant): {sorted(tenant_sizes.values(), reverse=True)}")

    rng = np.random.default_rng(20260823)

    all_rows = []
    pool_sizes = {n_hosts: [] for n_hosts in HOST_CONFIGS}
    for n_hosts in HOST_CONFIGS:
        hosts_template = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        print(f"\n=== Tenant-block bootstrapping at {n_hosts} hosts, {N_BOOT} replicates ===")
        for b in range(N_BOOT):
            boot_jobs = resample_tenant_block(tenant_ids, tenant_to_jobs, rng)
            pool_sizes[n_hosts].append(len(boot_jobs))
            for order_key, place_fn, allow_backfill, name in [
                (sim03.fifo_key, sim03.first_fit_place, False, "FIFO"),
                (sim03.fifo_key, sim03.first_fit_place, True, "Backfill_FIFO"),
                (sim03.panel_srpt_key, sim03.first_fit_place, True, "RF_SRPT_proxy"),
            ]:
                hosts = [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts_template]
                r = sim03.run_policy(boot_jobs, hosts, order_key, place_fn, allow_backfill, name)
                r["n_hosts"] = n_hosts
                r["boot_rep"] = b
                all_rows.append(r)
        print("  done "
              f"(replicate job-count range: {min(pool_sizes[n_hosts])}-{max(pool_sizes[n_hosts])}, "
              f"vs. 656 in the original job-level bootstrap)")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(PROCESSED, "tenant_block_bootstrap_raw.csv"), index=False)

    comparisons = []
    for n_hosts in HOST_CONFIGS:
        sub = df[df["n_hosts"] == n_hosts]
        piv = sub.pivot(index="boot_rep", columns="policy", values="avg_jct_min")
        backfill_vs_fifo = piv["Backfill_FIFO"] - piv["FIFO"]
        rf_vs_backfill = piv["RF_SRPT_proxy"] - piv["Backfill_FIFO"]
        comparisons.append({
            "n_hosts": n_hosts,
            "min_pool_size": int(min(pool_sizes[n_hosts])),
            "max_pool_size": int(max(pool_sizes[n_hosts])),
            "mean_pool_size": float(np.mean(pool_sizes[n_hosts])),
            "Backfill_minus_FIFO_mean": float(backfill_vs_fifo.mean()),
            "Backfill_minus_FIFO_ci": (float(backfill_vs_fifo.quantile(0.025)), float(backfill_vs_fifo.quantile(0.975))),
            "pct_reps_Backfill_beats_FIFO": float((backfill_vs_fifo < 0).mean() * 100),
            "RF_SRPT_minus_Backfill_mean": float(rf_vs_backfill.mean()),
            "RF_SRPT_minus_Backfill_ci": (float(rf_vs_backfill.quantile(0.025)), float(rf_vs_backfill.quantile(0.975))),
            "pct_reps_RF_SRPT_beats_Backfill": float((rf_vs_backfill < 0).mean() * 100),
        })

    comp_df = pd.DataFrame(comparisons)
    print("\n=== Tenant-block bootstrap comparisons (negative = left side better) ===")
    for row in comparisons:
        print(json.dumps(row, indent=2, default=str))
    comp_df.to_csv(os.path.join(PROCESSED, "tenant_block_bootstrap_comparisons.csv"), index=False)

    print("\n=== For comparison, the original job-level bootstrap (bootstrap_comparisons.csv) ===")
    orig = pd.read_csv(os.path.join(PROCESSED, "bootstrap_comparisons.csv"))
    print(orig[["n_hosts", "Backfill_minus_FIFO_mean", "Backfill_minus_FIFO_ci",
                "pct_reps_Backfill_beats_FIFO", "RF_SRPT_minus_Backfill_mean",
                "RF_SRPT_minus_Backfill_ci", "pct_reps_RF_SRPT_beats_Backfill"]].to_string(index=False))

    print("\nWrote tenant_block_bootstrap_raw.csv, tenant_block_bootstrap_comparisons.csv")


if __name__ == "__main__":
    main()
