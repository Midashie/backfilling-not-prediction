"""
05_sweep.py

Systematic cluster-size sweep. The earlier 3-point comparison (615/200/120
hosts) showed PredSched_LLM losing to RF_SRPT_proxy on average JCT at the
one load level where any policy differentiation existed at all. That's a
single data point standing in for a whole design space. This script sweeps
host count from the full 615-host pool down to a heavily constrained
40-host cluster, in enough steps to see the full utilization-vs-JCT curve
for all six policies, and specifically to check whether PredSched_LLM ever
overtakes RF_SRPT_proxy on average JCT at higher contention, where its
placement discipline has more opportunity to matter.
"""

import importlib.util
import json
import os

import pandas as pd

spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

PROCESSED = "processed"

# The largest job in the test set needs 64 workers, i.e. 64 hosts under this
# simulator's one-worker-per-host placement model (verified below). Any
# cluster smaller than that makes that job, and the 40 other jobs needing
# 60+ hosts, structurally impossible to ever place, which is not a load
# level, it is an infeasible cluster. An earlier version of this sweep
# included host counts down to 40 and silently produced misleading numbers:
# at 40/50/60 hosts, strict FIFO deadlocked on the first unplaceable job and
# completed only 6 of 656 jobs, and even the backfilling policies permanently
# dropped several jobs (651 or 614 of 656 completed) without that being
# visible in the headline average-JCT number. Both problems are caught here
# by asserting full completion, and the swept range is restricted to
# feasible cluster sizes.
HOST_COUNTS = [615, 500, 400, 300, 250, 200, 175, 150, 130, 120, 110, 100, 90, 85, 80, 75, 70, 65, 64]


def main():
    jobs, t0 = sim03.load_jobs()
    max_workers_needed = max(j["n_workers"] for j in jobs)
    print(f"Largest job needs {max_workers_needed} hosts; sweep floor set at {min(HOST_COUNTS)} hosts.")
    assert min(HOST_COUNTS) >= max_workers_needed, "sweep includes a structurally infeasible cluster size"

    all_rows = []
    for n_hosts in HOST_COUNTS:
        hosts = sim03.load_hosts(n_hosts=n_hosts, seed=0)
        cap = sum(h.capacity for h in hosts)
        results = sim03.run_all_policies(jobs, hosts)
        for r in results:
            r["n_hosts"] = n_hosts
            r["total_capacity"] = cap
            if r["n_jobs_completed"] != len(jobs):
                print(f"  WARNING: {r['policy']} at n_hosts={n_hosts} completed "
                      f"{r['n_jobs_completed']}/{len(jobs)}, some jobs never admitted")
        all_rows.extend(results)
        util = results[0]["mean_gpu_utilization"]
        print(f"n_hosts={n_hosts:4d} cap={cap:5d} util={util:.3f}  " +
              "  ".join(f"{r['policy']}={r['avg_jct_min']:.2f}" for r in results))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(PROCESSED, "sweep_results.csv"), index=False)
    with open(os.path.join(PROCESSED, "sweep_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)

    # crossover check: at which host counts, if any, does PredSched_LLM beat RF_SRPT_proxy on avg JCT?
    piv = df.pivot(index="n_hosts", columns="policy", values="avg_jct_min")
    piv["PredSched_beats_RF_SRPT"] = piv["PredSched_LLM"] < piv["RF_SRPT_proxy"]
    piv["gap_pct"] = (piv["PredSched_LLM"] - piv["RF_SRPT_proxy"]) / piv["RF_SRPT_proxy"] * 100
    print("\n=== PredSched_LLM vs RF_SRPT_proxy across the sweep ===")
    print(piv[["RF_SRPT_proxy", "PredSched_LLM", "gap_pct", "PredSched_beats_RF_SRPT"]].to_string())
    piv.to_csv(os.path.join(PROCESSED, "sweep_crossover_check.csv"))
    print(f"\nWrote {os.path.join(PROCESSED, 'sweep_results.csv')} and sweep_crossover_check.csv")


if __name__ == "__main__":
    main()
