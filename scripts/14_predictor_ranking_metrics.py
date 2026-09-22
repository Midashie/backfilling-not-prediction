"""
14_predictor_ranking_metrics.py

Reviewer 4, Round 3, Comment 8: MAE/MAPE (Table 1, from
02_train_predictor.py) describe average deviation but not whether the
predictor orders jobs correctly, the property an SRPT-family scheduler
actually needs. This script computes ranking-quality metrics directly
from processed/predictor_test_predictions.parquet (written by
02_train_predictor.py; run that script first if this file is missing):

  - Spearman rank correlation (true duration vs. predicted duration)
  - Kendall's tau
  - Pairwise ordering accuracy: the fraction of all non-tied job pairs
    (i, j) in the 656-job test set for which sign(pred_i - pred_j)
    matches sign(true_i - true_j). Pairs tied on true duration are
    excluded (they carry no ordering information to get right or wrong).

Computed for the selected panel predictor (the one actually used to
drive the simulator's PredSched_LLM/RF_SRPT_proxy/QSSF_proxy policies)
and, for comparison, the two component predictors (History-only,
GBM-only) reported alongside it in Table 1.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

IN_PATH = "processed/predictor_test_predictions.parquet"
OUT_PATH = "processed/predictor_ranking_metrics.json"


def pairwise_accuracy(y_true, y_pred):
    n = len(y_true)
    iu = np.triu_indices(n, k=1)
    true_sign = np.sign(y_true[:, None] - y_true[None, :])[iu]
    pred_sign = np.sign(y_pred[:, None] - y_pred[None, :])[iu]
    mask = true_sign != 0  # exclude true ties: no ordering information
    concordant = int((true_sign[mask] == pred_sign[mask]).sum())
    total = int(mask.sum())
    return concordant / total, concordant, total


def main():
    df = pd.read_parquet(IN_PATH)
    y = df["run_duration_min"].values
    cols = {
        "selected_panel": "pred_run_duration_min",
        "history_only": "pred_run_duration_min_hist_only",
        "gbm_only": "pred_run_duration_min_gbm_only",
    }

    results = {"n_test_jobs": len(df)}
    for label, col in cols.items():
        p = df[col].values
        rho, rho_p = stats.spearmanr(y, p)
        tau, tau_p = stats.kendalltau(y, p)
        acc, concordant, total = pairwise_accuracy(y, p)
        results[label] = {
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "kendall_tau": float(tau), "kendall_p": float(tau_p),
            "pairwise_ordering_accuracy_pct": float(acc * 100.0),
            "concordant_pairs": concordant, "non_tied_pairs": total,
        }
        print(f"{label}: Spearman={rho:.4f} Kendall={tau:.4f} "
              f"pairwise_accuracy={acc*100:.2f}% ({concordant}/{total} non-tied pairs)")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
