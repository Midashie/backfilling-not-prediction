"""
04_ablations.py

Two ablations on the constrained_120host config (the only one of the
three load levels in 03_simulate.py where scheduling policy produces a
measurable difference; see that file's docstring for why the two
higher-capacity configs show no policy differentiation).

Ablation 1 -- model-selection layer, on vs off:
  "on"  = RF_SRPT_proxy as already defined (ordering by the per-job-type
          selected GBM/History panel prediction).
  "off" = the same SRPT ordering and first-fit placement, but always
          using the plain global GBM prediction (no per-type selection
          against the History candidate).
  This isolates the effect of the model-selection layer from
  everything else (same placement, same predictor training data).

Ablation 2 -- fragmentation-aware placement, on vs off:
  This one does not need a new simulation run: it is exactly the
  existing RF_SRPT_proxy (first-fit placement) vs PredSched_LLM
  (rack-aware placement) comparison in 03_simulate.py's results, since
  both use the identical predicted-duration SRPT ordering and differ
  only in placement. Pulled from simulation_results.csv here rather
  than recomputed, to avoid a second, possibly-inconsistent run.
"""

import json
import os

import pandas as pd

import importlib.util

spec = importlib.util.spec_from_file_location("sim03", os.path.join(os.path.dirname(__file__), "03_simulate.py"))
sim03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim03)

PROCESSED = "processed"


def gbm_only_key(j):
    return j["pred_duration_gbm_only"]


def main():
    jobs, t0 = sim03.load_jobs()
    hosts = sim03.load_hosts(n_hosts=120)

    on_result = sim03.run_policy(jobs, hosts, sim03.panel_srpt_key, sim03.first_fit_place,
                                  allow_backfill=True, name="model_selection_ON (RF_SRPT_proxy)")
    off_result = sim03.run_policy(jobs, hosts, gbm_only_key, sim03.first_fit_place,
                                   allow_backfill=True, name="model_selection_OFF (GBM-only SRPT)")

    ablation1 = pd.DataFrame([on_result, off_result])
    print("=== Ablation 1: model-selection layer (constrained_120host) ===")
    print(ablation1.to_string(index=False))

    sim_results = pd.read_csv(os.path.join(PROCESSED, "simulation_results.csv"))
    sub = sim_results[(sim_results["config"] == "constrained_120host") &
                       (sim_results["policy"].isin(["RF_SRPT_proxy", "PredSched_LLM"]))]
    print("\n=== Ablation 2: fragmentation-aware placement (constrained_120host) ===")
    print(sub[["policy", "avg_jct_min", "p95_jct_min", "mean_gpu_utilization", "fragmentation_rate"]].to_string(index=False))

    out = {
        "ablation_1_model_selection": ablation1.to_dict(orient="records"),
        "ablation_2_fragmentation_placement": sub.to_dict(orient="records"),
    }
    with open(os.path.join(PROCESSED, "ablation_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {os.path.join(PROCESSED, 'ablation_results.json')}")


if __name__ == "__main__":
    main()
