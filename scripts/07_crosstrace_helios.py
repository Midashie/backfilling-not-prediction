"""
07_crosstrace_helios.py

Cross-trace validation of the paper's central finding (backfilling, not
prediction-informed ordering, drives nearly all scheduling-policy benefit)
on the Helios trace (S-Lab-System-Group/HeliosData, SC '21, CC-BY-4.0),
https://github.com/S-Lab-System-Group/HeliosData. This is the same trace
Hu et al. 2021 (reference [6], the QSSF paper this study's QSSF_proxy
baseline is built from) used in their original evaluation, so this is a
direct check on the source authors' own cluster, not an arbitrary third
trace.

Four independent clusters are provided: Earth, Saturn, Uranus, Venus. All
four are run, at a fixed, identical calendar window chosen once in
advance (2020-05-04 to 2020-05-18, 14 days) rather than tuned per cluster,
to avoid any appearance of window-shopping for a favorable result. This
window was chosen only by checking that all four clusters have non-trivial
job volume in it; no simulation was run before the window was fixed.

Real, verified facts about this trace, stated here rather than assumed:
  - Schema (from cluster_log.csv): job_id, user, vc, gpu_num, cpu_num,
    node_num, state, submit_time, start_time, end_time, duration (sec),
    queue. No per-host machine assignment and no rack/topology data are
    published, unlike the Alibaba trace's topo.csv.
  - Roughly half of logged jobs across all four clusters have gpu_num=0
    (CPU-only jobs sharing GPU-provisioned nodes). These are excluded
    here, since this paper's scope is GPU scheduling; this CPU-only
    share is itself much larger than in the Alibaba trace, which is a
    real difference between the two clusters worth stating, not
    smoothing over.
  - Among GPU jobs with node_num>0, gpu_num/node_num takes value 8 for
    only 4-17% of jobs depending on cluster; most jobs use 1 GPU on a
    shared node (matches this study's existing sharing model: hosts have
    8-GPU capacity, gpu_per_worker can be < 8, multiple jobs/workers
    share a host's remaining free capacity).
  - Real total GPU capacity per cluster during the chosen window (from
    cluster_gpu_number.csv, stable across the window): Earth 856,
    Saturn 2072, Uranus 2136, Venus 968.

What is NOT real and must not be read as such: since Helios's public
release has no host-level topology, a synthetic host pool is built per
cluster (N_hosts = round(real total capacity / 8), 8-GPU capacity per
host, matching the Alibaba host pool's already-documented 8-GPU
simplification) and hosts are partitioned round-robin into synthetic
racks of ~8 hosts each purely so the placement-aware policies
(Wind_proxy's Hilbert curve, PredSched_LLM's rack-aware placement) have
a structure to operate over. This means: the ordering-discipline finding
(backfilling vs. prediction) is tested with real topology-independent
fidelity here, but the placement-specific comparison (PredSched_LLM vs.
RF_SRPT_proxy) on Helios is testing placement-algorithm behavior against
a synthetic, not observed, topology, and must be read with that caveat
in the paper.

Predictor adaptation: Helios has no "kind" (job-architecture) field like
Alibaba's PyTorchJob/TFJob/ElasticBatchJob, so the per-kind selection
layer is replaced with a per-vc (virtual cluster / queue) selection
layer, the closest available categorical grouping. The History
estimator's fallback chain becomes (user, vc) -> vc -> global, replacing
Alibaba's (tenant_id, model) -> model -> global. Both are documented
substitutions of the closest available analogous keys, not claims of
equivalence.

Scope note: given four clusters must be run, the sweep here uses 7 host
counts per cluster (vs. 19 for the primary Alibaba analysis) and the
bootstrap uses 40 replicates at 2 load levels per cluster (vs. 60 for
the primary analysis's 2 configs, now being separately raised to ~300
in the main pipeline). This is a real reduction in statistical
resolution relative to the primary analysis and is reported as such,
not hidden.
"""

import json
import math
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

import importlib.util
spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

DATA_DIR = "data/helios"
PROCESSED = "processed"
CLUSTERS = ["Earth", "Saturn", "Uranus", "Venus"]
WIN_START = pd.Timestamp("2020-05-04")
WIN_END = pd.Timestamp("2020-05-18")
HISTORY_K = 10
N_BOOT = 40
SWEEP_POINTS = 7


def mape(y_true, y_pred, eps=1.0):
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100.0)


def chrono_split(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def history_predict(df, target, key1="user", key2="vc", k=HISTORY_K):
    values = df[target].values
    k1 = df[key1].values
    k2 = df[key2].values
    n = len(df)
    preds = np.full(n, np.nan)
    key_hist = defaultdict(lambda: deque(maxlen=k))
    key2_hist = defaultdict(lambda: deque(maxlen=k))
    global_hist = deque(maxlen=k)
    for i in range(n):
        key = (k1[i], k2[i])
        if len(key_hist[key]) > 0:
            preds[i] = float(np.median(key_hist[key]))
        elif len(key2_hist[k2[i]]) > 0:
            preds[i] = float(np.median(key2_hist[k2[i]]))
        elif len(global_hist) > 0:
            preds[i] = float(np.median(global_hist))
        else:
            preds[i] = 0.0
        key_hist[key].append(values[i])
        key2_hist[k2[i]].append(values[i])
        global_hist.append(values[i])
    return preds


def load_clean_cluster(cluster):
    df = pd.read_csv(os.path.join(DATA_DIR, cluster, "cluster_log.csv"),
                      parse_dates=["submit_time", "start_time", "end_time"])
    total_raw = len(df)
    df = df[(df["submit_time"] >= WIN_START) & (df["submit_time"] < WIN_END)]
    in_window = len(df)
    df = df[df["state"] == "COMPLETED"]
    completed = len(df)
    df = df[df["gpu_num"] > 0]
    gpu_jobs = len(df)
    df = df[df["node_num"] > 0]
    df = df.sort_values("submit_time").reset_index(drop=True)

    df["duration_min"] = df["duration"] / 60.0
    df["n_workers"] = df["node_num"].astype(int)
    df["total_gpu"] = df["gpu_num"].astype(float)
    df["gpu_per_worker"] = np.ceil(df["total_gpu"] / df["n_workers"]).clip(lower=1).astype(int)
    df["submit_hour"] = df["submit_time"].dt.hour
    df["submit_dow"] = df["submit_time"].dt.dayofweek

    print(f"  {cluster}: raw={total_raw} in_window={in_window} completed={completed} "
          f"gpu_jobs={gpu_jobs} final={len(df)}")
    return df


def train_predictor(df, cluster):
    train, val, test = chrono_split(df)
    hist_preds = history_predict(df, "duration_min")
    df = df.copy()
    df["hist_pred"] = hist_preds

    cat_features = ["vc"]
    num_features = ["gpu_num", "cpu_num", "node_num", "submit_hour", "submit_dow"]
    all_features = cat_features + num_features

    X_train = train[all_features].copy()
    X_val = val[all_features].copy()
    X_test = test[all_features].copy()
    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        dtype = pd.CategoricalDtype(categories=X_train[c].cat.categories)
        X_val[c] = X_val[c].astype(dtype)
        X_test[c] = X_test[c].astype(dtype)

    y_train = np.log1p(train["duration_min"].values)
    model = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=0, verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_features)

    gbm_pred_val = np.clip(np.expm1(model.predict(X_val)), 0, None)
    gbm_pred_test = np.clip(np.expm1(model.predict(X_test)), 0, None)
    hist_pred_val = df.loc[val.index, "hist_pred"].values
    hist_pred_test = df.loc[test.index, "hist_pred"].values
    y_val_true = val["duration_min"].values
    y_test_true = test["duration_min"].values

    # per-vc selection layer (closest available analog to Alibaba's per-kind layer)
    vcs = sorted(df["vc"].unique())
    val_vc = val["vc"].values
    test_vc = test["vc"].values
    selection = {}
    for v in vcs:
        vmask = val_vc == v
        if vmask.sum() == 0:
            selection[v] = "GBM"
            continue
        mae_gbm = mean_absolute_error(y_val_true[vmask], gbm_pred_val[vmask])
        mae_hist = mean_absolute_error(y_val_true[vmask], hist_pred_val[vmask])
        selection[v] = "GBM" if mae_gbm <= mae_hist else "History"

    combined_test = np.where(
        np.isin(test_vc, [v for v in vcs if selection[v] == "GBM"]),
        gbm_pred_test, hist_pred_test,
    )

    results = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "gbm_test_mae": mean_absolute_error(y_test_true, gbm_pred_test),
        "gbm_test_mape": mape(y_test_true, gbm_pred_test),
        "history_test_mae": mean_absolute_error(y_test_true, hist_pred_test),
        "history_test_mape": mape(y_test_true, hist_pred_test),
        "selected_test_mae": mean_absolute_error(y_test_true, combined_test),
        "selected_test_mape": mape(y_test_true, combined_test),
        "n_vc_groups": len(vcs),
        "n_vc_using_gbm": sum(1 for v in vcs if selection[v] == "GBM"),
    }

    test_out = test.copy()
    test_out["pred_duration_min"] = np.clip(combined_test, 0.01, None)
    test_out["pred_duration_min_hist_only"] = np.clip(hist_pred_test, 0.01, None)
    test_out["pred_duration_min_gbm_only"] = np.clip(gbm_pred_test, 0.01, None)
    return results, test_out


def build_jobs(test_out):
    t0 = test_out["submit_time"].min()
    test_out = test_out.sort_values("submit_time").reset_index(drop=True)
    jobs = []
    for i, row in test_out.iterrows():
        jobs.append({
            "id": i,
            "arrival_min": (row["submit_time"] - t0).total_seconds() / 60.0,
            "true_duration": max(float(row["duration_min"]), 0.01),
            "pred_duration_panel": float(row["pred_duration_min"]),
            "pred_duration_hist_only": float(row["pred_duration_min_hist_only"]),
            "pred_duration_gbm_only": float(row["pred_duration_min_gbm_only"]),
            "n_workers": int(row["n_workers"]),
            "gpu_per_worker": int(row["gpu_per_worker"]),
        })
    return jobs


def build_host_template(real_capacity_gpus, hosts_per_rack=8):
    n_hosts = max(1, round(real_capacity_gpus / 8))
    n_racks = max(1, math.ceil(n_hosts / hosts_per_rack))
    hosts = []
    for i in range(n_hosts):
        rack = f"SynRack_{i % n_racks}"
        hosts.append(sim03.Host(i, rack, 8))
    return hosts, n_hosts


def sample_hosts(template, n_hosts, seed=0):
    if n_hosts >= len(template):
        chosen = template
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(template), size=n_hosts, replace=False)
        chosen = [template[i] for i in sorted(idx)]
    return [sim03.Host(h.host_id, h.rack, h.capacity) for h in chosen]


def resample_jobs(jobs, rng):
    idx = rng.integers(0, len(jobs), size=len(jobs))
    resampled = [jobs[i] for i in idx]
    resampled.sort(key=lambda j: j["arrival_min"])
    return resampled


BOOTSTRAP_JOB_CAP = 700


def subsample_for_bootstrap(jobs, cap=BOOTSTRAP_JOB_CAP, seed=99):
    """
    The deterministic sweep runs on the full test-window job set (real,
    unmodified). The bootstrap, run 40x per host-count point, is not
    computationally tractable at full scale for the larger Helios clusters:
    at severe contention (the sweep floor, where host count equals the
    largest job's worker requirement), the backfill scan's cost grows much
    faster than linearly in pending-queue size, and Earth/Saturn/Uranus all
    have 3-6x more test-window jobs than the Alibaba test set this method
    was originally sized for. To keep the bootstrap tractable, it draws one
    fixed random subsample of at most `cap` jobs (seed fixed, re-sorted by
    arrival time) per cluster before resampling with replacement; this
    subsample is reused across both bootstrapped host-count points for
    that cluster. This is a real, documented scope reduction on the
    bootstrap step only, not applied to the sweep, and is reported as such.
    """
    if len(jobs) <= cap:
        return jobs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(jobs), size=cap, replace=False)
    sub = [jobs[i] for i in sorted(idx)]
    return sub


def main():
    all_predictor_results = {}
    all_sweep_rows = []
    all_bootstrap_rows = []
    all_comparisons = []

    print("=== Loading and cleaning Helios clusters (window 2020-05-04 to 2020-05-18) ===")
    for cluster in CLUSTERS:
        df = load_clean_cluster(cluster)
        max_workers_needed = df["n_workers"].max()

        pred_results, test_out = train_predictor(df, cluster)
        all_predictor_results[cluster] = pred_results
        print(f"  {cluster} predictor: GBM MAE={pred_results['gbm_test_mae']:.2f} "
              f"History MAE={pred_results['history_test_mae']:.2f} "
              f"Selected MAE={pred_results['selected_test_mae']:.2f}")

        jobs = build_jobs(test_out)

        g = pd.read_csv(os.path.join(DATA_DIR, cluster, "cluster_gpu_number.csv"), parse_dates=["date"])
        gwin = g[(g["date"] >= WIN_START) & (g["date"] < WIN_END)]
        real_capacity = float(gwin["total"].median())
        host_template, n_hosts_full = build_host_template(real_capacity)
        print(f"  {cluster}: real capacity {real_capacity:.0f} GPUs -> {n_hosts_full} synthetic hosts, "
              f"max_workers_needed={max_workers_needed}, test_jobs={len(jobs)}")

        floor = max(max_workers_needed, 1)
        if floor >= n_hosts_full:
            host_counts = [n_hosts_full]
        else:
            host_counts = sorted(set(np.linspace(floor, n_hosts_full, SWEEP_POINTS).astype(int).tolist()), reverse=True)

        print(f"  {cluster} sweep host counts: {host_counts}")
        for n_hosts in host_counts:
            hosts = sample_hosts(host_template, n_hosts, seed=0)
            cap = sum(h.capacity for h in hosts)
            results = sim03.run_all_policies(jobs, hosts)
            for r in results:
                r["cluster"] = cluster
                r["n_hosts"] = n_hosts
                r["total_capacity"] = cap
                if r["n_jobs_completed"] != len(jobs):
                    print(f"    WARNING: {r['policy']} at n_hosts={n_hosts} completed "
                          f"{r['n_jobs_completed']}/{len(jobs)}")
            all_sweep_rows.extend(results)

        jobs_boot_base = subsample_for_bootstrap(jobs)
        max_workers_boot = max(j["n_workers"] for j in jobs_boot_base)
        print(f"  {cluster} bootstrap job pool: {len(jobs_boot_base)} of {len(jobs)} test jobs "
              f"(subsampled={len(jobs_boot_base) < len(jobs)}), max_workers_needed_in_pool={max_workers_boot}")

        # IMPORTANT: the constrained bootstrap point must be the feasibility
        # floor for the ACTUAL bootstrap population (jobs_boot_base), not the
        # floor inherited from the full, unsubsampled sweep population. Those
        # can differ a lot: subsampling 700 jobs at random can (and for
        # Earth, does) drop the rare job(s) that set the full population's
        # floor, so reusing the full-population floor host count leaves the
        # subsampled workload with far more headroom than intended and shows
        # no contention at all, not because contention doesn't exist but
        # because the wrong cluster size was tested. Same principle as
        # 05_sweep.py's floor assertion, applied to this population.
        boot_points = sorted(set([host_counts[0], max_workers_boot]))
        rng = np.random.default_rng(20260811)
        for n_hosts in boot_points:
            print(f"  {cluster} bootstrapping at {n_hosts} hosts, {N_BOOT} reps")
            hosts_template_run = sample_hosts(host_template, n_hosts, seed=0)
            rows = []
            for b in range(N_BOOT):
                boot_jobs = resample_jobs(jobs_boot_base, rng)
                hosts_b = [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts_template_run]
                results = sim03.run_all_policies(boot_jobs, hosts_b)
                for r in results:
                    r["cluster"] = cluster
                    r["n_hosts"] = n_hosts
                    r["boot_rep"] = b
                    if r["n_jobs_completed"] != len(boot_jobs):
                        print(f"    WARNING: {r['policy']} at n_hosts={n_hosts} completed "
                              f"{r['n_jobs_completed']}/{len(boot_jobs)} in bootstrap rep {b}")
                rows.extend(results)
            all_bootstrap_rows.extend(rows)

            boot_df = pd.DataFrame(rows)
            piv = boot_df.pivot(index="boot_rep", columns="policy", values="avg_jct_min")
            pred_vs_rf = piv["PredSched_LLM"] - piv["RF_SRPT_proxy"]
            rf_vs_backfill = piv["RF_SRPT_proxy"] - piv["Backfill_FIFO"]
            backfill_vs_fifo = piv["Backfill_FIFO"] - piv["FIFO"]
            all_comparisons.append({
                "cluster": cluster, "n_hosts": n_hosts,
                "PredSched_minus_RF_SRPT_mean": pred_vs_rf.mean(),
                "PredSched_minus_RF_SRPT_ci_lo": pred_vs_rf.quantile(0.025), "PredSched_minus_RF_SRPT_ci_hi": pred_vs_rf.quantile(0.975),
                "pct_reps_PredSched_beats_RF_SRPT": float((pred_vs_rf < 0).mean() * 100),
                "RF_SRPT_minus_Backfill_mean": rf_vs_backfill.mean(),
                "RF_SRPT_minus_Backfill_ci_lo": rf_vs_backfill.quantile(0.025), "RF_SRPT_minus_Backfill_ci_hi": rf_vs_backfill.quantile(0.975),
                "pct_reps_RF_SRPT_beats_Backfill": float((rf_vs_backfill < 0).mean() * 100),
                "Backfill_minus_FIFO_mean": backfill_vs_fifo.mean(),
                "Backfill_minus_FIFO_ci_lo": backfill_vs_fifo.quantile(0.025), "Backfill_minus_FIFO_ci_hi": backfill_vs_fifo.quantile(0.975),
                "pct_reps_Backfill_beats_FIFO": float((backfill_vs_fifo < 0).mean() * 100),
            })

    with open(os.path.join(PROCESSED, "crosstrace_helios_predictor_results.json"), "w") as f:
        json.dump(all_predictor_results, f, indent=2, default=str)
    pd.DataFrame(all_sweep_rows).to_csv(os.path.join(PROCESSED, "crosstrace_helios_sweep.csv"), index=False)
    pd.DataFrame(all_bootstrap_rows).to_csv(os.path.join(PROCESSED, "crosstrace_helios_bootstrap_raw.csv"), index=False)
    comp_df = pd.DataFrame(all_comparisons)
    comp_df.to_csv(os.path.join(PROCESSED, "crosstrace_helios_comparisons.csv"), index=False)

    print("\n=== Cross-trace paired bootstrap comparisons (all 4 Helios clusters) ===")
    print(comp_df.to_string(index=False))
    print("\nWrote crosstrace_helios_predictor_results.json, crosstrace_helios_sweep.csv, "
          "crosstrace_helios_bootstrap_raw.csv, crosstrace_helios_comparisons.csv")


if __name__ == "__main__":
    main()
