# Backfilling Explains Most Scheduling Gains, While Prediction Benefits Are Trace-Dependent (code release)

Code accompanying the paper *"Backfilling Explains Most Scheduling Gains,
While Prediction Benefits Are Trace-Dependent: A Cluster-Size Sweep and
Cross-Trace Bootstrap of Forecast-Driven GPU Scheduling in the LLM Era"*
(Don Harl C. Malabanan, Aboitiz School of Innovation, Technology, And
Entrepreneurship, Asian Institute of Management).

This repository contains the full pipeline used to produce every number and
figure in the paper: dataset cleaning, duration prediction, a trace-driven
discrete-event scheduling simulator, ablations, a cluster-size sweep, a
job-level bootstrap for confidence intervals, a cross-trace validation on a
second public dataset, and the figure-generation script. Nothing in the
paper's Results section is illustrative; every reported number comes from
running this code against the datasets below.

## What this study found

Stated here as directly as it is stated in the paper: the proposed
prediction-informed and fragmentation-aware scheduling method
(`PredSched_LLM`) does not show a statistically distinguishable
job-completion-time advantage over simpler baselines on either trace tested.
The real, large, bootstrap-confirmed and cross-trace-confirmed effect is
backfilling (removing head-of-line blocking) versus strict FIFO, not
prediction-informed ordering and not fragmentation-aware placement. See
Section 6 and Section 8 of the paper for the full, non-simplified account,
including where the two traces disagree (prediction shows a real directional
signal on the Helios trace that it does not show on the Alibaba trace). An
oracle-duration upper-bound check (`scripts/10_oracle_bootstrap.py`) rules
out a natural objection, that the trained predictor is simply too weak to
test this fairly: even a scheduler ordered on each job's true, already-known
duration does not clear backfilling's bar either, so the shortfall is not
primarily a predictor-quality artifact. A second check
(`scripts/11_easy_backfill.py`) rules out a different objection, that the
backfilling effect itself is specific to `Backfill_FIFO`'s aggressive
implementation: EASY-style conservative backfilling also beats strict FIFO
substantially, confirming backfilling as a concept is not an implementation
artifact, though it captures much less of the benefit than the aggressive
variant under this trace's poor duration predictions, since conservative
backfilling (unlike aggressive) depends on those estimates being reliable.
A resource-footprint sensitivity check (`scripts/12_footprint_sensitivity.py`)
finds both headline comparisons essentially unchanged when the 5 jobs whose
kind implies runtime-resizable allocation are removed. A tenant-level block
bootstrap (`scripts/13_tenant_block_bootstrap.py`), resampling the test
set's 17 tenants instead of individual jobs to test whether the original
bootstrap's job-independence assumption matters, finds the backfilling
effect weakens (70.0% and 90.3% of replicates favor it, at 120 and 64 hosts
respectively, down from 97.0% and 99.7% under job-level resampling) but
does not disappear, while the prediction-versus-backfilling null result
stays just as null. The qualitative conclusion survives; the certainty
attached to the backfilling effect specifically is more modest than the
job-level bootstrap's headline numbers alone would suggest.

## Pipeline

Run in order from the repository root:

| Script | Purpose | Output |
|---|---|---|
| `scripts/01_build_dataset.py` | Cleans the raw Alibaba trace into a leakage-safe modeling dataset | `processed/jobs_clean.parquet`, `jobs_excluded.csv`, `topo_lookup.csv` |
| `scripts/02_train_predictor.py` | Trains the duration predictor (GBM + causal History estimator, per-kind selection) | `processed/predictor_results.json`, `predictor_test_predictions.parquet` |
| `scripts/03_simulate.py` | Trace-driven discrete-event simulator; six scheduling policies | `processed/simulation_results.csv/json` |
| `scripts/04_ablations.py` | Model-selection-layer and placement-constraint ablations | `processed/ablation_results.json` |
| `scripts/05_sweep.py` | 19-point cluster-size sweep, 64 to 615 hosts | `processed/sweep_results.csv/json`, `sweep_crossover_check.csv` |
| `scripts/06_bootstrap.py` | 300-replicate job-level bootstrap at 120 and 64 hosts | `processed/bootstrap_raw.csv`, `bootstrap_summary.csv`, `bootstrap_comparisons.csv` |
| `scripts/07_crosstrace_helios.py` | Cross-trace validation on all 4 Helios clusters, bootstrapped at each cluster's constrained floor, one added intermediate matched-contention point, and full capacity | `processed/crosstrace_helios_*.csv/json` |
| `scripts/08_make_figures.py` | Builds the figures referenced in Section 6.2 and Section 6.4 | `figures/fig1_sweep_curve.png`, `figures/fig2_bootstrap_ci.png`, `figures/fig3_crosstrace_helios.png` |
| `scripts/09_peak_demand_check.py` | Computes the two peak-concurrent-GPU-demand figures cited in Section 5.1 (real, whole-cluster peak) and Section 5.3 (peak demand of the 656 replayed test-set jobs alone) | `processed/peak_demand_check.json` |
| `scripts/10_oracle_bootstrap.py` | Oracle-duration upper-bound check (Section 6.2): reruns the same 300-replicate bootstrap at 120 and 64 hosts with an `Oracle_SRPT` policy ordered on true, already-known job duration instead of a prediction, to separate predictor quality from the value of prediction-informed ordering itself. Reuses the exact resample sequence from `06_bootstrap.py` so results are directly paired against `Backfill_FIFO` and `RF_SRPT_proxy` there. | `processed/oracle_bootstrap_raw.csv`, `oracle_bootstrap_summary.csv`, `oracle_bootstrap_comparisons.csv` |
| `scripts/11_easy_backfill.py` | Additional established-scheduler check (Section 6.2): adds `EASY_Backfill`, the reservation-based conservative backfilling algorithm (Lifka 1995; Mu'alem & Feitelson 2001), to test whether the backfilling effect is specific to `Backfill_FIFO`'s aggressive implementation. Includes a built-in sanity check (a perfect-information diagnostic run) confirming the algorithm itself is correct independent of predictor quality. Reuses the same resample sequence as `06_bootstrap.py`. | `processed/easy_backfill_bootstrap_raw.csv`, `easy_backfill_bootstrap_summary.csv`, `easy_backfill_bootstrap_comparisons.csv` |
| `scripts/12_footprint_sensitivity.py` | Resource-footprint sensitivity check (Section 7 Limitations): reruns the same 300-replicate, six-policy bootstrap at 120 and 64 hosts with the 5 ElasticBatchJob jobs (the only kind whose name implies runtime-resizable allocation) removed from the 656-job test set, to test whether the two headline comparisons are sensitive to treating realized worker allocation as a proxy for requested allocation. | `processed/footprint_sensitivity_bootstrap_raw.csv`, `footprint_sensitivity_comparisons.csv` |
| `scripts/13_tenant_block_bootstrap.py` | Tenant-level block bootstrap (Section 7 Limitations): tests whether 06_bootstrap.py's job-level independence assumption matters by resampling the test set's 17 tenants with replacement instead of individual jobs, preserving within-tenant correlation. Only the three policies needed for the two headline comparisons (FIFO, Backfill_FIFO, RF_SRPT_proxy) are run, to keep runtime reasonable. | `processed/tenant_block_bootstrap_raw.csv`, `tenant_block_bootstrap_comparisons.csv` |
| `scripts/14_predictor_ranking_metrics.py` | Predictor ranking-quality check (Section 6.1): computes Spearman rank correlation, Kendall's tau, and pairwise ordering accuracy from `predictor_test_predictions.parquet`, the property an SRPT-style scheduler actually needs, alongside the MAE/MAPE already reported. | `processed/predictor_ranking_metrics.json` |

`processed/` in this repository already contains the exact output files used
to write the paper, so you can inspect results without rerunning anything.
To reproduce from scratch, delete `processed/*` and run the scripts in
order; they read each other's outputs (e.g. `03_simulate.py` requires
`02_train_predictor.py`'s output to already exist).

## Data

Raw trace data is not bundled in this repository (the Alibaba trace's
`job.csv`/`worker.csv` and the Helios trace together are several hundred MB).
See `data/README.md` for exact download sources, licenses, and the expected
directory layout (`data/job.csv`, `data/worker.csv`, `data/topo.csv`,
`data/helios/{Earth,Saturn,Uranus,Venus}/cluster_log.csv` etc.).

## Requirements

Python 3.11+, see `requirements.txt`. Key dependencies: `pandas`, `numpy`,
`lightgbm`, `scikit-learn`, `matplotlib`, `scipy`.

```bash
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_train_predictor.py
python scripts/03_simulate.py
python scripts/04_ablations.py
python scripts/05_sweep.py
python scripts/06_bootstrap.py
python scripts/07_crosstrace_helios.py
python scripts/08_make_figures.py
python scripts/09_peak_demand_check.py
python scripts/10_oracle_bootstrap.py
python scripts/11_easy_backfill.py
python scripts/12_footprint_sensitivity.py
python scripts/13_tenant_block_bootstrap.py
python scripts/14_predictor_ranking_metrics.py
```

## Known limitations (see paper Section 7 for the full list)

- The Alibaba host pool's 8-GPU-per-host capacity is a documented
  simplification (true for 96.9% of observed hosts, not all).
- The Helios cross-trace check uses a synthetic host pool (no host-level
  topology is published for that trace), and at severe-contention cluster
  sizes that pool is smaller than one synthetic rack, which structurally
  prevents the rack-aware placement policy from differing from plain
  first-fit. The placement-specific comparison on Helios should be read
  with that caveat; it did not receive a clean test.
- `Backfill_FIFO` is FIFO-with-backfilling, not a Pollux reimplementation;
  see Section 5.2 of the paper for why it is named this way.
- Baselines (`QSSF_proxy`, `RF_SRPT_proxy`, `Wind_proxy`) are reimplemented
  from each paper's published description, not from released source code
  (none of the three publish an implementation).

## Citation

If you use this code, please cite the paper (full reference in the paper's
front matter) and the underlying trace datasets (see `data/README.md`).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
This license covers the code only; the datasets in `data/` are governed by
their own, separate licenses (see `data/README.md`).
