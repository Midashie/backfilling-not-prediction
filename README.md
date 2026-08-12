# Prediction-Informed Scheduling for LLM-Era GPU Clusters — code release

Code accompanying the paper *"Prediction-Informed Scheduling for LLM-Era GPU
Clusters: Revisiting Forecast-Driven Job Scheduling on a Modern Production
Trace"* (Don Harl C. Malabanan, Aboitiz School of Innovation, Technology, And
Entrepreneurship, Asian Institute of Management).

This repository contains the full pipeline used to produce every number and
figure in the paper: dataset cleaning, duration prediction, a trace-driven
discrete-event scheduling simulator, ablations, a cluster-size sweep, a
job-level bootstrap for confidence intervals, a cross-trace validation on a
second public dataset, and the figure-generation script. Nothing in the
paper's Results section is illustrative; every reported number comes from
running this code against the datasets below.

## Findings of the Paper

Stated here as directly as it is stated in the paper: the proposed
prediction-informed and fragmentation-aware scheduling method
(`PredSched_LLM`) does not show a statistically robust job-completion-time
advantage over simpler baselines on either trace tested. The real, robust,
bootstrap-confirmed and cross-trace-confirmed effect is backfilling
(removing head-of-line blocking) versus strict FIFO, not prediction-informed
ordering and not fragmentation-aware placement. See Section 6 and Section 8
of the paper for the full, non-simplified account, including where the two
traces disagree (prediction shows a real directional signal on the Helios
trace that it does not show on the Alibaba trace).

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
| `scripts/07_crosstrace_helios.py` | Cross-trace validation on all 4 Helios clusters | `processed/crosstrace_helios_*.csv/json` |
| `scripts/08_make_figures.py` | Builds the two figures referenced in Section 6.2 | `figures/fig1_sweep_curve.png`, `figures/fig2_bootstrap_ci.png` |

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
`lightgbm`, `scikit-learn`, `matplotlib`.

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
