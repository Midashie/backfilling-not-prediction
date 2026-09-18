"""
09_peak_demand_check.py

Computes the two peak-concurrent-GPU-demand figures cited in Section 5.1
and Section 5.3 of the paper, both derived from the same cleaned dataset
(processed/jobs_clean.parquet) built by 01_build_dataset.py, using a
sweep-line (interval-stabbing) pass over each job's [gmt_job_running,
end_time) interval weighted by its total_gpu allocation.

Two distinct populations are reported, because they answer two distinct
questions and are both cited in the paper:

1. full_cluster_peak_gpu: peak concurrent GPU allocation across EVERY
   job in the cleaned dataset (all 4,367 jobs) whose running interval
   overlaps the test window at all, restricted to timestamps that fall
   inside the window. This is the real, observed demand on the actual
   production cluster during that calendar span, independent of how
   this paper split the data into train/val/test. Cited in Section 5.1.

2. test_set_replay_peak_gpu: peak concurrent GPU allocation among only
   the 656 chronological test-set jobs, using their own true arrival
   and duration timeline. This is the demand the simulator in Section
   5.3 actually replays (the simulator never models the other ~3,700
   train/val jobs running concurrently), so it is the figure that
   directly explains why the full 615-host/4,920-GPU cluster run showed
   zero contention. Cited in Section 5.3.

Output: processed/peak_demand_check.json
"""

import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "processed"


def sweep_line_peak(jobs, clip_start=None, clip_end=None):
    """Peak concurrent total_gpu over a set of jobs' [gmt_job_running, end_time)
    intervals. If clip_start/clip_end given, only timestamps within that
    range are considered when checking for a new peak (but a job's full
    interval, even outside the clip range, still contributes to the running
    total at any point within the range -- i.e. jobs already running when
    the window opens are correctly counted)."""
    events = []
    for _, r in jobs.iterrows():
        if pd.isna(r["gmt_job_running"]) or pd.isna(r["end_time"]):
            continue
        events.append((r["gmt_job_running"], r["total_gpu"]))
        events.append((r["end_time"], -r["total_gpu"]))
    if not events:
        return 0.0, None
    times = np.array([e[0] for e in events])
    deltas = np.array([e[1] for e in events])
    order = np.argsort(times)
    cum = 0.0
    peak = 0.0
    peak_time = None
    for idx in order:
        cum += deltas[idx]
        t = times[idx]
        if clip_start is not None and t < clip_start:
            continue
        if clip_end is not None and t > clip_end:
            continue
        if cum > peak:
            peak = cum
            peak_time = t
    return peak, peak_time


def main():
    df = pd.read_parquet(os.path.join(OUT_DIR, "jobs_clean.parquet"))
    df = df.sort_values("gmt_job_submitted").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    test = df.iloc[n_train + n_val:].copy()

    test_start = test["gmt_job_submitted"].min()
    test_end = test["gmt_job_submitted"].max()

    # Population 1: every cleaned job overlapping the test window, real cluster demand
    overlap = df[(df["gmt_job_running"] < test_end) & (df["end_time"] > test_start)].copy()
    full_peak, full_peak_time = sweep_line_peak(overlap, clip_start=test_start, clip_end=test_end)

    # Population 2: only the 656 replayed test-set jobs, their own timeline
    test_peak, test_peak_time = sweep_line_peak(test)

    result = {
        "test_window_start": str(test_start),
        "test_window_end": str(test_end),
        "n_test_set_jobs": len(test),
        "n_jobs_overlapping_window_full_population": len(overlap),
        "full_cluster_peak_gpu": full_peak,
        "full_cluster_peak_gpu_timestamp": str(full_peak_time),
        "test_set_replay_peak_gpu": test_peak,
        "test_set_replay_peak_gpu_timestamp": str(test_peak_time),
        "nominal_cluster_capacity_gpu": 4920,
    }
    with open(os.path.join(OUT_DIR, "peak_demand_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
