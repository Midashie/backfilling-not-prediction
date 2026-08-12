"""
01_build_dataset.py

Builds a clean, leakage-safe, per-job modeling dataset from the raw
Alibaba 2023 GPU cluster trace (job.csv, worker.csv, topo.csv).

Source: https://github.com/alibaba/alibaba-lingjun-dataset-2023

Design decisions (documented here so they can be cited directly in the
paper's Section 3 / 4.1 / 5.1):

1. Population: only jobs that (a) are not still Running at trace-export
   time (status != 'Running', 9 jobs excluded — right-censored, no
   observed duration), (b) actually reached the Running state at least
   once (gmt_job_running not null, 803 jobs excluded — these failed or
   were stopped before ever being scheduled, so they have no duration
   to predict), and (c) have a resolvable end time (gmt_job_finished,
   falling back to gmt_job_stopped; 48 further exclusions after (a)/(b)
   already remove most of these).

2. Anomalous record: job id=8169 (job_name dlc1uyze9wefdjps) has
   gmt_job_running (2023-07-05 15:18) dated *before* gmt_job_submitted
   (2023-07-17 02:53) by about 11.5 days. This is the single cause of
   both the reported negative queue_delay_min outlier (-16535 min) and
   an equally anomalous ~16539-minute duration if computed naively. No
   other row in the 5,180-row trace exhibits this. job_restart_times is
   NaN for every row in this trace (the column carries no information
   in this dataset), so we cannot use it to explain the anomaly as a
   resubmission artifact. Given it is an isolated single-record clock
   inconsistency (1 of 5,180 rows, 0.02%) with no way to determine which
   timestamp is correct, this job is excluded from the modeling dataset
   entirely rather than guessed at or silently kept.

3. Feature/target split (the core leakage-avoidance decision):

   IMPORTANT CORRECTION vs. the original Section 3 draft language: the
   draft's formal model described *both* duration d_j and resource
   demand r_j as "unobserved-at-submission" quantities to be predicted.
   That is not how this trace, or a real GPU cluster scheduler, actually
   works. A scheduler cannot place a job without knowing how many
   GPUs/workers it needs; that footprint is declared by the submitter
   at job-creation time (it is baked into the PyTorchJob/TFJob spec).
   job.csv does not export that declared value as a separate column,
   but worker.csv's realized allocation (total_gpu, n_workers) is the
   closest available proxy for it, and for these rigid (non-elastic)
   job kinds the realized allocation matches the declared spec by
   construction. Treating total_gpu/n_workers as an unknown prediction
   target (as an earlier draft of this pipeline did) would make the
   simulator unable to place jobs at all, which is not how any
   scheduler in the literature this paper cites (Hu et al. 2021,
   Luo et al. 2025) is built: they all assume resource footprint is
   known at admission, and predict only job duration/length to drive
   SRPT-style ordering. This pipeline now follows that same, more
   defensible convention. Section 3 of the paper will be corrected to
   match (resource footprint moves from r_j into m_j; only duration
   d_j remains the predicted quantity).

   FEATURES (m_j) — everything known at the moment the job is admitted:
     kind, model, namespace, tenant_id, group_id, user_id, workspace_id,
     priority, resource_level, job_max_running_time_minutes (a
     user-declared cap, not an outcome), is_enable_gpu_topo_aware,
     total_gpu, n_workers, distinct_hosts (the declared resource
     footprint, proxied by worker.csv's realized allocation as
     explained above), and calendar features derived from
     gmt_job_submitted (hour of day, day of week).
   TARGET (d_j) — only known after the job runs to completion:
     run_duration_min (= end_time - gmt_job_running). This is the sole
     quantity the prediction module estimates, consistent with how
     SRPT-family schedulers in the cited prior work operate.
   EXCLUDED (outcome / leakage fields, never used as features):
     status, reason, reason_code, sub_status, gmt_job_running,
     gmt_job_stopped, gmt_job_finished, gmt_modified, job_restart_times
     (all-NaN in this trace), and all of worker.csv/topo.csv beyond
     what is needed to compute the targets and simulator placement.

4. Topology join: 86.3% of distinct worker host_ips in worker.csv match
   an ip in topo.csv; the remaining 13.7% have no topology record and
   are treated as "unknown rack" in the simulator's fragmentation-aware
   placement logic, not dropped.

Outputs (written to processed/):
  - jobs_clean.parquet   : one row per modeling-eligible job, features + targets
  - jobs_excluded.csv    : jobs dropped, with the exact reason for each
  - topo_lookup.csv      : host_ip -> DSW/PSW/ASW rack mapping
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

TS_COLS = ["gmt_job_submitted", "gmt_job_running", "gmt_job_stopped", "gmt_job_finished"]

ANOMALOUS_JOB_IDS = {8169}  # dlc1uyze9wefdjps — gmt_job_running predates gmt_job_submitted


def load_job_table():
    df = pd.read_csv(os.path.join(DATA_DIR, "job.csv"), low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for c in TS_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["end_time"] = df["gmt_job_finished"].fillna(df["gmt_job_stopped"])
    return df


def load_worker_resources():
    w = pd.read_csv(os.path.join(DATA_DIR, "worker.csv"), low_memory=False)

    def gpu_count(res_json):
        if pd.isna(res_json):
            return 0.0
        try:
            d = json.loads(res_json)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        v = d.get("nvidia.com/gpu")
        if v is None:
            return 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0

    w["gpu_per_worker"] = w["RES"].apply(gpu_count)
    agg = w.groupby("job_name").agg(
        n_workers=("worker_name", "nunique"),
        total_gpu=("gpu_per_worker", "sum"),
        distinct_hosts=("host_ip", "nunique"),
    ).reset_index()
    # host list per job, needed later for topology-aware simulation replay
    hosts_per_job = w.groupby("job_name")["host_ip"].apply(lambda s: sorted(set(s.dropna()))).reset_index()
    hosts_per_job.columns = ["job_name", "host_ips"]
    agg = agg.merge(hosts_per_job, on="job_name", how="left")
    return agg


def load_topo():
    t = pd.read_csv(os.path.join(DATA_DIR, "topo.csv"), low_memory=False)
    t = t.rename(columns={"ip": "host_ip"})
    return t


def main():
    jobs = load_job_table()
    worker_agg = load_worker_resources()
    topo = load_topo()

    total_jobs = len(jobs)
    exclusions = []

    still_running = jobs["status"] == "Running"
    exclusions.append(jobs.loc[still_running, ["id", "job_name"]].assign(reason="status=Running (censored, no observed duration)"))

    never_scheduled = jobs["gmt_job_running"].isna() & ~still_running
    exclusions.append(jobs.loc[never_scheduled, ["id", "job_name"]].assign(reason="gmt_job_running missing (never entered Running state)"))

    remaining_mask = ~still_running & ~never_scheduled
    no_end_time = remaining_mask & jobs["end_time"].isna()
    exclusions.append(jobs.loc[no_end_time, ["id", "job_name"]].assign(reason="end_time unresolvable (both gmt_job_finished and gmt_job_stopped missing)"))

    remaining_mask = remaining_mask & ~no_end_time
    is_anomalous = jobs["id"].isin(ANOMALOUS_JOB_IDS)
    exclusions.append(jobs.loc[remaining_mask & is_anomalous, ["id", "job_name"]].assign(
        reason="anomalous timestamp: gmt_job_running predates gmt_job_submitted by ~11.5 days (single corrupted record)"
    ))

    keep_mask = remaining_mask & ~is_anomalous
    clean = jobs.loc[keep_mask].copy()

    clean["run_duration_min"] = (clean["end_time"] - clean["gmt_job_running"]).dt.total_seconds() / 60.0
    clean["queue_delay_min"] = (clean["gmt_job_running"] - clean["gmt_job_submitted"]).dt.total_seconds() / 60.0

    assert (clean["run_duration_min"] < 0).sum() == 0, "unexpected negative duration after cleaning"
    assert (clean["queue_delay_min"] < 0).sum() == 0, "unexpected negative queue delay after cleaning"

    clean = clean.merge(worker_agg, on="job_name", how="left")
    # jobs that reached Running but somehow have no worker rows: treat resource demand as 0/unknown, flag them
    clean["has_worker_records"] = clean["n_workers"].notna()
    clean["n_workers"] = clean["n_workers"].fillna(0).astype(int)
    clean["total_gpu"] = clean["total_gpu"].fillna(0.0)
    clean["distinct_hosts"] = clean["distinct_hosts"].fillna(0).astype(int)

    clean["submit_hour"] = clean["gmt_job_submitted"].dt.hour
    clean["submit_dow"] = clean["gmt_job_submitted"].dt.dayofweek

    feature_cols = [
        "id", "job_name",
        "kind", "model", "namespace", "tenant_id", "group_id", "user_id", "workspace_id",
        "priority", "resource_level", "job_max_running_time_minutes", "is_enable_gpu_topo_aware",
        "total_gpu", "n_workers", "distinct_hosts",
        "submit_hour", "submit_dow", "gmt_job_submitted",
    ]
    target_cols = ["run_duration_min", "queue_delay_min"]
    meta_cols = ["status", "reason_code", "has_worker_records", "host_ips", "gmt_job_running", "end_time"]

    final_cols = feature_cols + target_cols + meta_cols
    final = clean[final_cols].sort_values("gmt_job_submitted").reset_index(drop=True)

    final.to_parquet(os.path.join(OUT_DIR, "jobs_clean.parquet"), index=False)

    excl_df = pd.concat(exclusions, ignore_index=True)
    excl_df.to_csv(os.path.join(OUT_DIR, "jobs_excluded.csv"), index=False)

    topo.to_csv(os.path.join(OUT_DIR, "topo_lookup.csv"), index=False)

    print(f"Total jobs in raw trace: {total_jobs}")
    print(f"Excluded: {len(excl_df)}")
    print(excl_df["reason"].value_counts().to_string())
    print(f"Modeling population (jobs_clean.parquet): {len(final)}")
    print(f"  jobs with no worker.csv records despite reaching Running: {(~final['has_worker_records']).sum()}")
    print(f"Time span: {final['gmt_job_submitted'].min()} to {final['gmt_job_submitted'].max()}")


if __name__ == "__main__":
    main()
