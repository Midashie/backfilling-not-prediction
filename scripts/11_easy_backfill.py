"""
11_easy_backfill.py

Added in response to Array Reviewer 1, point 7: "An oracle SRPT baseline
and at least one additional well-established backfilling scheduler would
help separate implementation artifacts from scheduling principles." The
oracle-SRPT half is 10_oracle_bootstrap.py; this script is the second half.

Backfill_FIFO (03_simulate.py) is what the scheduling literature calls
AGGRESSIVE backfilling: any later-queued job that currently fits is started
immediately, with no guarantee that this cannot delay the job at the head of
the queue indefinitely. This script adds EASY backfilling (Lifka 1995;
Mu'alem & Feitelson 2001), the best-known alternative and the specific
mechanism most GPU/HPC scheduling papers mean by "backfilling" when they
cite prior art: the head-of-queue job is given a reservation, the earliest
future time its placement becomes feasible given currently running jobs'
estimated completion times, and a later job may only jump ahead of it if
doing so is guaranteed not to delay that reservation (a sufficient
condition used here: the candidate job's own estimated completion must be
no later than the reservation time; if it finishes by then, whatever
capacity it used is guaranteed free again in time, regardless of which
specific hosts were involved).

Estimated completion time, for the purpose of computing and honoring the
reservation, uses each job's panel-predicted duration (pred_duration_panel,
the same trained predictor used by RF_SRPT_proxy), NOT its true duration.
This mirrors how a real scheduler works, it does not know the future, only
an estimate, and it is a deliberate, stated choice: the trace's own
job_max_running_time_minutes field is not usable for this (75th percentile
is 0 across the full dataset, and the nonzero tail contains implausible
values up to 6.2e8 minutes, evidently placeholder/sentinel data rather than
real user-declared limits), so no other runtime-estimate field exists to
use instead. The DISCRETE-EVENT SIMULATION ITSELF still advances on each
job's true duration; only EASY's internal scheduling decisions reason from
the (possibly wrong) predicted duration, exactly as a live scheduler would.
This also means EASY's reservation can, in principle, be honored or missed
depending on how good the predictor turns out to be, a realistic property
of the actual algorithm, not a simulation artifact.

Everything else, arrival order as priority, first-fit placement, cluster
and job data, is identical to Backfill_FIFO, so any difference between the
two isolates the aggressive-vs-conservative backfilling design choice
itself, not an unrelated confound.
"""

import heapq
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

PROCESSED = "processed"
N_BOOT = 300
HOST_CONFIGS = [120, 64]


def _feasible(job, free_map):
    """Would `job` fit right now, given a {host_id: free_capacity} snapshot?
    Mirrors 03_simulate.first_fit_place's logic exactly, but against a plain
    dict snapshot instead of live Host objects, so it can be run against
    hypothetical future capacity without touching real simulator state."""
    need = job["n_workers"]
    gpw = job["gpu_per_worker"]
    if gpw == 0:
        return True
    count = 0
    for hid in free_map:
        if free_map[hid] >= gpw:
            count += 1
            if count >= need:
                return True
    return False


def compute_reservation_time(head_job, hosts, running_list, now):
    """Earliest time >= now at which head_job's placement becomes feasible,
    given hosts' CURRENT free capacity plus the estimated (predicted-
    duration) release of every currently running job, released in
    ascending order of estimated completion. Caller only invokes this when
    head_job does NOT already fit now.

    A running job's estimated completion (est_end = its own start time +
    its panel-predicted duration) can already be in the past relative to
    `now`, if the predictor underestimated that job's true duration -- the
    job is still actually running (true_end has not been reached yet, or
    it would already have departed), but the ESTIMATE the scheduler is
    working from says it should be done by now. Given this predictor's
    130.7-minute MAE against a 151.9-minute mean duration (Section 6.1),
    this is not a rare edge case: at every one of the 71 head-blocked
    scheduling events observed on the real 656-job test set at 120 hosts,
    every single currently running job was already "overdue" against its
    own estimate. A live scheduler cannot say an overdue job's remaining
    time is exactly zero (every job still takes SOME further, if brief,
    time to actually stop and release its resources); each release event
    is therefore assigned a value strictly after the previous one, walking
    forward from `now` by a fixed small increment when an estimate is
    already stale, so the reservation clock advances monotonically instead
    of collapsing onto `now` itself (which would make the candidate-
    admission test below impossible to ever satisfy, since a job that
    starts now always completes strictly after now)."""
    EPS_MIN = 0.01  # matches this codebase's own zero-duration floor convention (03_simulate.py load_jobs)
    free_map = {h.host_id: h.free for h in hosts}
    events = sorted(running_list, key=lambda r: r[4])
    clock = now
    for (_true_end, _jid, host_ids, gpw, est_end) in events:
        clock = max(est_end, clock + EPS_MIN)
        for hid in host_ids:
            free_map[hid] = free_map.get(hid, 0) + gpw
        if _feasible(head_job, free_map):
            return clock
    return math.inf  # should not happen once all running jobs have drained


def run_easy_backfill(jobs, hosts_template, name="EASY_Backfill"):
    hosts = [sim03.Host(h.host_id, h.rack, h.capacity) for h in hosts_template]
    hosts_by_id = {h.host_id: h for h in hosts}

    pending = []
    running = []  # heap of (true_end_time, job_id, host_ids, gpu_per_worker, est_end_time)
    completed = []

    arrivals = sorted(jobs, key=lambda j: j["arrival_min"])
    arr_idx = 0
    n = len(arrivals)
    stranded_events = []
    backfill_count = 0

    def hosts_ordered_for_ff():
        return sorted(hosts, key=lambda h: h.host_id)

    def place_job(job, chosen, now):
        for h in chosen:
            frac_before = h.free / h.capacity if h.capacity else 0
            h.free -= job["gpu_per_worker"]
            stranded = 1.0 if (0 < h.free < job["gpu_per_worker"]) else 0.0
            stranded_events.append(stranded)
        true_end = now + job["true_duration"]
        est_end = now + job["pred_duration_panel"]
        heapq.heappush(running, (true_end, job["id"], [h.host_id for h in chosen], job["gpu_per_worker"], est_end))
        completed.append({"id": job["id"], "arrival": job["arrival_min"], "start": now,
                           "end": true_end, "jct": true_end - job["arrival_min"]})

    def try_schedule(now):
        nonlocal backfill_count
        if not pending:
            return
        pending.sort(key=lambda j: j["arrival_min"])

        # keep placing new heads while possible
        while pending:
            head = pending[0]
            chosen = sim03.first_fit_place(head, hosts_ordered_for_ff())
            if chosen is None:
                break
            place_job(head, chosen, now)
            pending.pop(0)

        if not pending:
            return

        head = pending[0]
        reservation_time = compute_reservation_time(head, hosts, running, now)

        i = 1
        while i < len(pending):
            job = pending[i]
            chosen = sim03.first_fit_place(job, hosts_ordered_for_ff())
            if chosen is not None:
                est_completion = now + job["pred_duration_panel"]
                if est_completion <= reservation_time + 1e-9:
                    place_job(job, chosen, now)
                    backfill_count += 1
                    del pending[i]
                    continue  # re-check the same index, list has shifted
            i += 1

    util_samples = []
    last_time = 0.0

    while arr_idx < n or running:
        next_arrival_t = arrivals[arr_idx]["arrival_min"] if arr_idx < n else math.inf
        next_dep_t = running[0][0] if running else math.inf
        t = min(next_arrival_t, next_dep_t)

        used = sum(h.capacity - h.free for h in hosts)
        total_cap = sum(h.capacity for h in hosts)
        util_samples.append((t - last_time, used / total_cap if total_cap else 0))
        last_time = t

        while running and running[0][0] <= t + 1e-9:
            true_end, jid, host_ids, gpw, est_end = heapq.heappop(running)
            for hid in host_ids:
                hosts_by_id[hid].free += gpw

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


def _sanity_check():
    """Fast checks against the real (non-bootstrapped) 656-job test set at
    120 hosts before spending compute on a 300-replicate bootstrap.

    EASY's head-placement loop is mechanically identical to Backfill_FIFO's;
    backfilling can only ADD placements on top of that, using free capacity
    the head could not have used anyway, so EASY_Backfill can never do
    WORSE than strict FIFO, only the same or better (weak inequality, not
    strict: with this trace's real predictor, see below, EASY finds ZERO
    safe backfill opportunities and degenerates exactly to FIFO's number).

    That zero-backfill result on the real predictor is real, not a sign of
    a bug: this predictor's 130.7-minute MAE against a 151.9-minute mean
    duration (Section 6.1) means that at every one of the 71 head-blocked
    scheduling events on this trace, every currently running job is already
    "overdue" against its own predicted completion, which collapses EASY's
    reservation clock to just after `now`, a window essentially no
    candidate job can fit inside. To confirm this is a predictor-quality
    finding and not an implementation defect, the same run is repeated here
    with each job's TRUE duration substituted for its predicted duration
    (an oracle-EASY diagnostic, not part of the reported results): under
    perfect duration knowledge, EASY_Backfill must find a non-zero number
    of safe backfill opportunities and land strictly between FIFO and the
    aggressive Backfill_FIFO. If it does not, that means the algorithm
    itself is broken, not merely fed a bad predictor."""
    jobs, _ = sim03.load_jobs()
    hosts = sim03.load_hosts(n_hosts=120, seed=0)
    r = run_easy_backfill(jobs, hosts)
    assert r["n_jobs_completed"] == len(jobs), f"lost jobs: {r['n_jobs_completed']} != {len(jobs)}"
    assert r["avg_jct_min"] is not None and r["avg_jct_min"] > 0
    assert math.isfinite(r["makespan_min"]) and r["makespan_min"] > 0
    print("Sanity check (120 hosts, real 656-job test set, single run, real predictor):")
    print(json.dumps(r, indent=2))

    hosts_bf = sim03.load_hosts(n_hosts=120, seed=0)
    fifo_result = sim03.run_policy(jobs, hosts_bf, sim03.fifo_key, sim03.first_fit_place, allow_backfill=False, name="FIFO")
    hosts_bf2 = sim03.load_hosts(n_hosts=120, seed=0)
    agg_result = sim03.run_policy(jobs, hosts_bf2, sim03.fifo_key, sim03.first_fit_place, allow_backfill=True, name="Backfill_FIFO")
    print(f"FIFO avg_jct_min:          {fifo_result['avg_jct_min']:.2f}")
    print(f"Backfill_FIFO avg_jct_min: {agg_result['avg_jct_min']:.2f}")
    print(f"EASY_Backfill avg_jct_min: {r['avg_jct_min']:.2f}  (backfill_admissions={r['backfill_admissions']})")
    assert r["avg_jct_min"] <= fifo_result["avg_jct_min"] + 1e-6, (
        "EASY_Backfill did WORSE than strict FIFO -- this should be structurally impossible "
        "(backfilling can only add placements, never remove them); implementation bug."
    )

    # Oracle-EASY diagnostic: same algorithm, perfect duration knowledge instead
    # of the real predictor. Must find real backfill opportunities and land
    # strictly between FIFO and Backfill_FIFO, or the algorithm itself is broken.
    jobs_oracle = [dict(j, pred_duration_panel=j["true_duration"]) for j in jobs]
    hosts_oracle = sim03.load_hosts(n_hosts=120, seed=0)
    r_oracle = run_easy_backfill(jobs_oracle, hosts_oracle, name="EASY_Backfill_oracle_diagnostic")
    print(f"[diagnostic only, not a reported result] EASY with perfect duration "
          f"knowledge: avg_jct_min={r_oracle['avg_jct_min']:.2f}, "
          f"backfill_admissions={r_oracle['backfill_admissions']}")
    assert r_oracle["backfill_admissions"] > 0, (
        "EASY found zero backfill opportunities even with PERFECT duration knowledge -- "
        "this indicates a genuine implementation bug (with the real predictor, zero is "
        "expected and correct; with perfect knowledge, it is not)."
    )
    assert agg_result["avg_jct_min"] <= r_oracle["avg_jct_min"] <= fifo_result["avg_jct_min"] + 1e-6, (
        f"Oracle-EASY avg_jct_min ({r_oracle['avg_jct_min']:.2f}) is not between "
        f"Backfill_FIFO ({agg_result['avg_jct_min']:.2f}) and FIFO ({fifo_result['avg_jct_min']:.2f}) "
        f"as expected for a conservative backfilling variant with perfect information."
    )
    print("Sanity check passed: EASY_Backfill never beats-worse-than-FIFO, and the "
          "algorithm itself is confirmed correct via the perfect-information diagnostic.")


def resample_jobs(jobs, rng):
    idx = rng.integers(0, len(jobs), size=len(jobs))
    resampled = [jobs[i] for i in idx]
    resampled.sort(key=lambda j: j["arrival_min"])
    return resampled


def main():
    _sanity_check()

    jobs, t0 = sim03.load_jobs()
    rng = np.random.default_rng(12345)  # same seed/order as 06_bootstrap.py

    all_rows = []
    for n_hosts in HOST_CONFIGS:
        hosts_template = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        print(f"\n=== EASY_Backfill bootstrapping at {n_hosts} hosts, {N_BOOT} replicates ===")
        for b in range(N_BOOT):
            boot_jobs = resample_jobs(jobs, rng)
            r = run_easy_backfill(boot_jobs, hosts_template)
            r["n_hosts"] = n_hosts
            r["boot_rep"] = b
            all_rows.append(r)
        print("  done")

    easy_df = pd.DataFrame(all_rows)
    easy_df.to_csv(os.path.join(PROCESSED, "easy_backfill_bootstrap_raw.csv"), index=False)

    existing = pd.read_csv(os.path.join(PROCESSED, "bootstrap_raw.csv"))

    summary_rows = []
    comparisons = []
    for n_hosts in HOST_CONFIGS:
        esub = easy_df[easy_df["n_hosts"] == n_hosts].set_index("boot_rep")
        exist_sub = existing[existing["n_hosts"] == n_hosts]

        bf = exist_sub[exist_sub["policy"] == "Backfill_FIFO"].set_index("boot_rep")
        mismatch = (esub["n_jobs_completed"] != bf["n_jobs_completed"]).sum()
        if mismatch:
            raise RuntimeError(
                f"Pairing check FAILED at n_hosts={n_hosts}: {mismatch} replicates have a "
                f"different completed-job count between EASY_Backfill and Backfill_FIFO."
            )
        print(f"Pairing check passed at n_hosts={n_hosts}: job-set alignment confirmed across {N_BOOT} replicates.")

        vals = esub["avg_jct_min"].values
        summary_rows.append({
            "n_hosts": n_hosts, "policy": "EASY_Backfill",
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "ci_2.5%": float(np.percentile(vals, 2.5)), "ci_97.5%": float(np.percentile(vals, 97.5)),
        })

        fifo = exist_sub[exist_sub["policy"] == "FIFO"].set_index("boot_rep")
        easy_vs_backfill = esub["avg_jct_min"] - bf["avg_jct_min"]
        easy_vs_fifo = esub["avg_jct_min"] - fifo["avg_jct_min"]

        comparisons.append({
            "n_hosts": n_hosts,
            "EASY_minus_Backfill_mean": float(easy_vs_backfill.mean()),
            "EASY_minus_Backfill_ci": (float(easy_vs_backfill.quantile(0.025)), float(easy_vs_backfill.quantile(0.975))),
            "pct_reps_EASY_beats_Backfill": float((easy_vs_backfill < 0).mean() * 100),
            "EASY_minus_FIFO_mean": float(easy_vs_fifo.mean()),
            "EASY_minus_FIFO_ci": (float(easy_vs_fifo.quantile(0.025)), float(easy_vs_fifo.quantile(0.975))),
            "pct_reps_EASY_beats_FIFO": float((easy_vs_fifo < 0).mean() * 100),
        })

    summary = pd.DataFrame(summary_rows)
    print("\n=== EASY_Backfill bootstrap summary: avg JCT mean and 95% CI ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(PROCESSED, "easy_backfill_bootstrap_summary.csv"), index=False)

    comp_df = pd.DataFrame(comparisons)
    print("\n=== EASY_Backfill paired comparisons (negative = left side has lower/better JCT) ===")
    for row in comparisons:
        print(json.dumps(row, indent=2, default=str))
    comp_df.to_csv(os.path.join(PROCESSED, "easy_backfill_bootstrap_comparisons.csv"), index=False)
    print("\nWrote easy_backfill_bootstrap_raw.csv, easy_backfill_bootstrap_summary.csv, easy_backfill_bootstrap_comparisons.csv")


if __name__ == "__main__":
    main()
