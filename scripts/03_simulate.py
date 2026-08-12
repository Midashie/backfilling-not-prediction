"""
03_simulate.py

Trace-driven discrete-event simulator (Section 5.3) that replays the
chronologically held-out test-set jobs (the same 656 jobs the stage-1
predictor was evaluated on) against six scheduling policies on a shared
simulated cluster.

IMPORTANT, stated plainly: none of the five baselines below are exact
reproductions of the cited papers' full algorithms. Each is a
simplified reimplementation of that paper's *core scheduling idea*,
built from what is actually derivable from this trace and from the
level of detail available in the literature review (not from those
papers' source code, which is not public for most of them). Every
simplification is documented at the policy definition below and must
be carried into the paper's Limitations section verbatim, not smoothed
over.

Simulated cluster: 615 real GPU-bearing hosts (host_ip values from
worker.csv with observed max GPU > 0), each given a capacity of 8 GPUs
(the modal true capacity: 596 of these 615 hosts, 96.9%, were observed
running exactly 8 GPUs at some point; using 8 uniformly for the
remaining hosts is a documented simplification, not a measured fact for
every host). Rack topology (ASW/PSW/DSW) is attached from topo.csv;
589 of 615 hosts (95.8%) have a topology match, the rest are placed in
an UNKNOWN_RACK bucket.

Policies:
  FIFO        - strict arrival-order queue, head-of-line blocking (no
                backfilling: if the earliest-arrived job can't be
                placed, later jobs wait too, even if they'd fit). The
                deliberately weak baseline every scheduling paper compares
                against.
  QSSF        - proxy for Hu et al. (SC 2021, Helios)'s Quasi-Shortest-
                Service-First. Ordering = ascending predicted duration
                using ONLY the causal rolling-median ("History")
                estimator (a lightweight, training-free predictor, in
                the spirit of Hu et al.'s "quasi" estimate), with
                backfilling. Placement = first-fit.
  RF_SRPT     - proxy for Luo et al. (INFOCOM 2025). Ordering =
                ascending predicted duration using the full trained
                per-job-type-selected GBM/History panel (the stronger,
                trained predictor), with backfilling. Placement =
                first-fit. Differs from QSSF only in predictor strength,
                which is the axis Luo et al.'s paper argues matters.
  BACKFILL_FIFO - NOT a Pollux reimplementation, deliberately renamed away
                from an earlier version of this script that called it
                "Pollux_proxy". Pollux's actual mechanism (goodput-optimized
                elastic batch-size/allocation adaptation) requires a
                per-job scaling-efficiency curve that is not present in,
                and cannot be derived from, this trace, so no real
                Pollux comparison is possible here. What this policy
                actually is: FIFO ordering with backfilling enabled,
                i.e. FIFO minus the head-of-line-blocking weakness, and
                nothing else. It turns out to be the single most
                important baseline in this study (see Section 6), so
                calling it by a borrowed name would have been actively
                misleading rather than a minor simplification.
  WIND        - proxy for Wind (EuroSys 2026). Ordering = FIFO with
                backfilling (Wind's paper does not propose a novel
                queue-ordering discipline; its contribution is
                placement). Placement = a real Hilbert-curve packing:
                hosts are mapped to a 1D order via a 2D Hilbert curve
                over (rack index, free-capacity bucket), and each job's
                workers are placed by scanning hosts in that order and
                taking the first hosts with enough free capacity. This
                concentrates fragmentation instead of spreading it.
  PREDSCHED   - the proposed method. Ordering = same predicted-duration
                SRPT as RF_SRPT (full trained panel), with backfilling.
                Placement = rack-aware: for each job, prefer a single
                rack (ASW) that alone has enough free capacity across
                >= n_workers hosts; if none exists, greedily fill from
                the rack with the most free capacity first, minimizing
                the number of distinct racks touched.

Metrics per policy: average JCT, p95 JCT, makespan, mean GPU
utilization over the simulated window, fragmentation rate (mean, over
completed placement decisions, of stranded-capacity fraction: hosts
touched whose remaining free capacity after placement was insufficient
for the modal 1-GPU job but nonzero), and job admission/backfill
counts.
"""

import heapq
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

PROCESSED = "processed"


# ---------------------------------------------------------------- Hilbert curve

def xy2d(order, x, y):
    """Standard xy-to-distance Hilbert curve mapping, order = bits per axis."""
    rx = ry = d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


# ---------------------------------------------------------------- Cluster model

class Host:
    __slots__ = ("host_id", "rack", "capacity", "free")

    def __init__(self, host_id, rack, capacity):
        self.host_id = host_id
        self.rack = rack
        self.capacity = capacity
        self.free = capacity


def load_hosts(n_hosts=None, seed=0):
    hp = pd.read_csv(os.path.join(PROCESSED, "host_pool.csv"))
    if n_hosts is not None and n_hosts < len(hp):
        hp = hp.sample(n=n_hosts, random_state=seed).reset_index(drop=True)
    hosts = [Host(i, row["ASW"], int(row["capacity"])) for i, row in hp.iterrows()]
    return hosts


def load_jobs():
    preds = pd.read_parquet(os.path.join(PROCESSED, "predictor_test_predictions.parquet"))
    full = pd.read_parquet(os.path.join(PROCESSED, "jobs_clean.parquet"))
    pred_cols = ["id", "pred_run_duration_min", "pred_run_duration_min_hist_only", "pred_run_duration_min_gbm_only"]
    df = full.merge(preds[pred_cols], on="id", how="inner")
    df = df.sort_values("gmt_job_submitted").reset_index(drop=True)
    t0 = df["gmt_job_submitted"].min()
    df["arrival_min"] = (df["gmt_job_submitted"] - t0).dt.total_seconds() / 60.0

    jobs = []
    for _, row in df.iterrows():
        n_workers = max(int(row["n_workers"]), 0)
        total_gpu = float(row["total_gpu"])
        if n_workers > 0:
            gpu_per_worker = max(1, math.ceil(total_gpu / n_workers)) if total_gpu > 0 else 0
        else:
            # the 2 jobs that reached Running but have no worker.csv records at all:
            # treat as a single zero-footprint placeholder worker so they still occupy
            # a queue slot and complete, without claiming any GPU capacity.
            n_workers = 1
            gpu_per_worker = 0
        true_duration = max(float(row["run_duration_min"]), 0.01)  # avoid zero-duration events colliding
        jobs.append({
            "id": int(row["id"]),
            "job_name": row["job_name"],
            "arrival_min": float(row["arrival_min"]),
            "true_duration": true_duration,
            "pred_duration_panel": max(float(row["pred_run_duration_min"]), 0.01),
            "pred_duration_hist_only": max(float(row["pred_run_duration_min_hist_only"]), 0.01),
            "pred_duration_gbm_only": max(float(row["pred_run_duration_min_gbm_only"]), 0.01),
            "n_workers": n_workers,
            "gpu_per_worker": gpu_per_worker,
        })
    return jobs, t0


# ---------------------------------------------------------------- Placement strategies

def first_fit_place(job, hosts):
    need = job["n_workers"]
    gpw = job["gpu_per_worker"]
    if gpw == 0:
        return hosts[:need] if len(hosts) >= need else None
    chosen = []
    for h in hosts:
        if h.free >= gpw:
            chosen.append(h)
            if len(chosen) == need:
                return chosen
    return None


def rack_aware_place(job, hosts, hosts_by_rack):
    need = job["n_workers"]
    gpw = job["gpu_per_worker"]
    if gpw == 0:
        return hosts[:need] if len(hosts) >= need else None
    # try single rack that alone satisfies the job
    for rack, hlist in hosts_by_rack.items():
        capable = [h for h in hlist if h.free >= gpw]
        if len(capable) >= need:
            return capable[:need]
    # fallback: fill from racks with most total free capacity first, minimizing racks touched
    rack_order = sorted(hosts_by_rack.items(), key=lambda kv: -sum(h.free for h in kv[1]))
    chosen = []
    for rack, hlist in rack_order:
        for h in hlist:
            if h.free >= gpw:
                chosen.append(h)
                if len(chosen) == need:
                    return chosen
    return None


def build_hilbert_order(hosts, rack_index):
    # 2D coordinate: (rack index, free-capacity bucket), mapped through an 8-bit Hilbert curve
    order_bits = 8
    side = 1 << order_bits
    max_rack = max(rack_index.values()) + 1 if rack_index else 1

    def coord(h):
        x = int(rack_index[h.rack] / max(max_rack, 1) * (side - 1))
        y = int(h.free / max(h.capacity, 1) * (side - 1))
        return x, y

    keyed = []
    for h in hosts:
        x, y = coord(h)
        keyed.append((xy2d(order_bits, x, y), h))
    keyed.sort(key=lambda t: t[0])
    return [h for _, h in keyed]


def hilbert_place(job, hosts, rack_index):
    need = job["n_workers"]
    gpw = job["gpu_per_worker"]
    ordered = build_hilbert_order(hosts, rack_index)
    if gpw == 0:
        return ordered[:need] if len(ordered) >= need else None
    chosen = []
    for h in ordered:
        if h.free >= gpw:
            chosen.append(h)
            if len(chosen) == need:
                return chosen
    return None


# ---------------------------------------------------------------- Simulator core

def run_policy(jobs, hosts_template, order_key, place_fn, allow_backfill, name):
    hosts = [Host(h.host_id, h.rack, h.capacity) for h in hosts_template]
    hosts_by_id = {h.host_id: h for h in hosts}
    rack_index = {r: i for i, r in enumerate(sorted(set(h.rack for h in hosts)))}

    pending = []  # list of job dicts, arrived, not yet started
    running = []  # heap of (end_time, job_id, list_of_host_ids, gpu_per_worker)
    completed = []  # dicts with jct, start_time etc.

    arrivals = sorted(jobs, key=lambda j: j["arrival_min"])
    arr_idx = 0
    n = len(arrivals)
    clock = 0.0
    stranded_events = []  # fragmentation bookkeeping: per successful placement, host free-after fractions
    backfill_count = 0
    total_admitted = 0

    def hosts_ordered_for_ff():
        return sorted(hosts, key=lambda h: h.host_id)

    def try_schedule(now):
        nonlocal backfill_count, total_admitted
        if not pending:
            return
        pending.sort(key=order_key)
        still_pending = []
        placed_any = True
        i = 0
        first = True
        while i < len(pending):
            job = pending[i]
            if place_fn.__name__ == "first_fit_place":
                chosen = first_fit_place(job, hosts_ordered_for_ff())
            elif place_fn.__name__ == "rack_aware_place":
                hbr = defaultdict(list)
                for h in hosts:
                    hbr[h.rack].append(h)
                chosen = rack_aware_place(job, hosts, hbr)
            else:  # hilbert
                chosen = hilbert_place(job, hosts, rack_index)

            if chosen:
                for h in chosen:
                    frac_before = h.free / h.capacity if h.capacity else 0
                    h.free -= job["gpu_per_worker"]
                    frac_after = h.free / h.capacity if h.capacity else 0
                    stranded = 1.0 if (0 < h.free < job["gpu_per_worker"]) else 0.0
                    stranded_events.append(stranded)
                end_time = now + job["true_duration"]
                heapq.heappush(running, (end_time, job["id"], [h.host_id for h in chosen], job["gpu_per_worker"]))
                completed.append({"id": job["id"], "arrival": job["arrival_min"], "start": now,
                                   "end": end_time, "jct": end_time - job["arrival_min"]})
                total_admitted += 1
                if not first:
                    backfill_count += 1
                i += 1
                first = True if not allow_backfill else first
            else:
                if not allow_backfill:
                    # strict FIFO: head blocks, stop trying the rest
                    still_pending.extend(pending[i:])
                    break
                else:
                    still_pending.append(job)
                    first = False
                    i += 1
        pending[:] = still_pending

    util_samples = []  # (time, used_fraction) sampled at every event
    last_time = 0.0

    while arr_idx < n or running:
        next_arrival_t = arrivals[arr_idx]["arrival_min"] if arr_idx < n else math.inf
        next_dep_t = running[0][0] if running else math.inf
        t = min(next_arrival_t, next_dep_t)

        used = sum(h.capacity - h.free for h in hosts)
        total_cap = sum(h.capacity for h in hosts)
        util_samples.append((t - last_time, used / total_cap if total_cap else 0))
        last_time = t
        clock = t

        # process all departures at this time
        while running and running[0][0] <= t + 1e-9:
            end_time, jid, host_ids, gpw = heapq.heappop(running)
            for hid in host_ids:
                hosts_by_id[hid].free += gpw

        # process all arrivals at this time
        while arr_idx < n and arrivals[arr_idx]["arrival_min"] <= t + 1e-9:
            pending.append(arrivals[arr_idx])
            arr_idx += 1

        try_schedule(t)

    makespan = max((c["end"] for c in completed), default=0.0)
    jcts = np.array([c["jct"] for c in completed])
    weighted_util = sum(w * u for w, u in util_samples) / sum(w for w, u in util_samples) if util_samples else 0.0
    frag_rate = float(np.mean(stranded_events)) if stranded_events else 0.0

    return {
        "policy": name,
        "n_jobs_completed": len(completed),
        "avg_jct_min": float(np.mean(jcts)) if len(jcts) else None,
        "p95_jct_min": float(np.percentile(jcts, 95)) if len(jcts) else None,
        "makespan_min": float(makespan),
        "mean_gpu_utilization": float(weighted_util),
        "fragmentation_rate": frag_rate,
        "backfill_admissions": backfill_count,
    }


def fifo_key(j):
    return j["arrival_min"]


def hist_srpt_key(j):
    return j["pred_duration_hist_only"]


def panel_srpt_key(j):
    return j["pred_duration_panel"]


def run_all_policies(jobs, hosts):
    results = []
    results.append(run_policy(jobs, hosts, fifo_key, first_fit_place, allow_backfill=False, name="FIFO"))
    results.append(run_policy(jobs, hosts, hist_srpt_key, first_fit_place, allow_backfill=True, name="QSSF_proxy"))
    results.append(run_policy(jobs, hosts, panel_srpt_key, first_fit_place, allow_backfill=True, name="RF_SRPT_proxy"))
    results.append(run_policy(jobs, hosts, fifo_key, first_fit_place, allow_backfill=True, name="Backfill_FIFO"))
    results.append(run_policy(jobs, hosts, fifo_key, hilbert_place, allow_backfill=True, name="Wind_proxy"))
    results.append(run_policy(jobs, hosts, panel_srpt_key, rack_aware_place, allow_backfill=True, name="PredSched_LLM"))
    return results


def main():
    jobs, t0 = load_jobs()

    # Real peak concurrent GPU demand actually observed in this trace during the
    # test window (2023-07-28 05:37 to 2023-07-30 22:50), computed directly from
    # worker.csv pod start/finish timestamps: 1537 GPUs. A cluster sized well
    # above that (the full 615-host / 4920-GPU pool used across the whole 16-day
    # trace) shows essentially no contention in just a 656-job / 3-day slice, so
    # every scheduling policy collapses to "start immediately on arrival" and
    # produces identical metrics -- that null result is reported first, honestly,
    # rather than hidden. Two more load levels are then run at cluster sizes
    # deliberately set at and below the observed peak demand, to test whether
    # policy differences appear once queueing pressure exists.
    configs = [
        ("oversized_615host", 615),
        ("tight_200host", 200),
        ("constrained_120host", 120),
    ]

    all_rows = []
    for label, n_hosts in configs:
        hosts = load_hosts(n_hosts=n_hosts)
        cap = sum(h.capacity for h in hosts)
        print(f"\n=== Config: {label} ({len(hosts)} hosts, {cap} GPU capacity) ===")
        results = run_all_policies(jobs, hosts)
        for r in results:
            r["config"] = label
            r["n_hosts"] = len(hosts)
            r["total_capacity"] = cap
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        all_rows.extend(results)

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(os.path.join(PROCESSED, "simulation_results.csv"), index=False)
    with open(os.path.join(PROCESSED, "simulation_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {os.path.join(PROCESSED, 'simulation_results.csv')}")


if __name__ == "__main__":
    main()
