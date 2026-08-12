"""
02_train_predictor.py

Stage 1 of PredSched-LLM: the prediction module.

Predicts, from submission-time features, the single quantity a
SRPT-family scheduler needs and cannot otherwise know: run_duration_min
(d_j, true wall-clock duration once the job starts running). Resource
footprint (total_gpu, n_workers) is treated as a known input feature,
not a prediction target -- see the corrected discussion in
01_build_dataset.py for why (a scheduler must already know a job's
GPU/worker footprint to place it; only duration is genuinely unknown
at admission time, matching how Hu et al. 2021 and Luo et al. 2025
formulate the same problem).

Per Section 4.1 of the paper draft, this is a panel of two candidate
model families with a per-job-type selection layer:

  Candidate A ("GBM"): LightGBM gradient-boosted-tree regressor over the
    submission-time feature set (categorical + numeric), trained on
    log1p(target) to handle the heavy right skew, predictions
    expm1-transformed back.

  Candidate B ("History"): a lightweight, causal sequence-based
    estimator. For a job of a given (tenant_id, model) pair, it predicts
    the target as the rolling median of that same target over the K
    most recent prior jobs (strictly earlier in submission order) from
    the same tenant_id+model pair. Falls back to the rolling median over
    the same `model` type, then to the global rolling median, when a
    tenant+model or model history is not yet available (cold start).
    This is intentionally simple (no neural sequence model): the trace
    has no sub-job telemetry time series, only a submission-ordered
    sequence of jobs, so a rolling-statistic estimator is the honest
    "sequence model" that real data in this trace supports.

Model-selection layer: for each job `kind` (PyTorchJob / TFJob /
ElasticBatchJob) and each target, both candidates are scored on the
validation split by MAE; the candidate with the lower validation MAE is
selected for that (kind, target) pair and used to produce test-set
predictions. This selection is exactly what the "model-selection layer
on/off" ablation toggles (off = always use the single global GBM
candidate for every job).

Train/validation/test split is strictly chronological on
gmt_job_submitted (70/15/15) to avoid any future-leaks-into-past
evaluation.
"""

import json
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

IN_PATH = "processed/jobs_clean.parquet"
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["run_duration_min"]
CAT_FEATURES = ["kind", "model", "namespace", "tenant_id", "user_id", "workspace_id"]
NUM_FEATURES = [
    "priority", "job_max_running_time_minutes", "is_enable_gpu_topo_aware", "submit_hour", "submit_dow",
    "total_gpu", "n_workers", "distinct_hosts",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
HISTORY_K = 10


def chrono_split(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def mape(y_true, y_pred, eps=1.0):
    # eps guards against division blow-up for near-zero durations/GPU counts,
    # which are common in this trace (median duration is 4 minutes).
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100.0)


def history_predict(full_df, target, tenant_col="tenant_id", type_col="model", k=HISTORY_K):
    """
    Causal rolling-median predictor. full_df must already be sorted by
    gmt_job_submitted ascending. Returns an array aligned to full_df.index
    where prediction for row i only uses rows strictly before i.
    """
    values = full_df[target].values
    tenant = full_df[tenant_col].values
    jtype = full_df[type_col].values
    n = len(full_df)
    preds = np.full(n, np.nan)

    from collections import defaultdict, deque
    tenant_type_hist = defaultdict(lambda: deque(maxlen=k))
    type_hist = defaultdict(lambda: deque(maxlen=k))
    global_hist = deque(maxlen=k)

    for i in range(n):
        key = (tenant[i], jtype[i])
        if len(tenant_type_hist[key]) > 0:
            preds[i] = float(np.median(tenant_type_hist[key]))
        elif len(type_hist[jtype[i]]) > 0:
            preds[i] = float(np.median(type_hist[jtype[i]]))
        elif len(global_hist) > 0:
            preds[i] = float(np.median(global_hist))
        else:
            preds[i] = 0.0
        tenant_type_hist[key].append(values[i])
        type_hist[jtype[i]].append(values[i])
        global_hist.append(values[i])
    return preds


def main():
    df = pd.read_parquet(IN_PATH).sort_values("gmt_job_submitted").reset_index(drop=True)
    train, val, test = chrono_split(df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Train window: {train['gmt_job_submitted'].min()} .. {train['gmt_job_submitted'].max()}")
    print(f"Val window:   {val['gmt_job_submitted'].min()} .. {val['gmt_job_submitted'].max()}")
    print(f"Test window:  {test['gmt_job_submitted'].min()} .. {test['gmt_job_submitted'].max()}")

    # history predictor is computed once over the full chronological sequence
    # (causal by construction: row i only sees rows < i), then sliced by split.
    hist_preds = {}
    for target in TARGETS:
        hist_preds[target] = history_predict(df, target)
    df_hist = df.copy()
    for target in TARGETS:
        df_hist[f"hist_pred_{target}"] = hist_preds[target]

    results = {}
    selection = {}  # (kind, target) -> "GBM" or "History"
    kinds = sorted(df["kind"].unique())

    final_preds = {t: np.zeros(len(test)) for t in TARGETS}
    gbm_only_preds = {}
    hist_only_preds = {}

    for target in TARGETS:
        X_train = train[ALL_FEATURES].copy()
        X_val = val[ALL_FEATURES].copy()
        X_test = test[ALL_FEATURES].copy()
        cat_categories = {}
        for c in CAT_FEATURES:
            X_train[c] = X_train[c].astype("category")
            cat_categories[c] = X_train[c].cat.categories
            dtype = pd.CategoricalDtype(categories=cat_categories[c])
            X_val[c] = X_val[c].astype(dtype)
            X_test[c] = X_test[c].astype(dtype)

        y_train = np.log1p(train[target].values)
        model = LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=0, verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

        gbm_pred_val = np.clip(np.expm1(model.predict(X_val)), 0, None)
        gbm_pred_test = np.clip(np.expm1(model.predict(X_test)), 0, None)

        hist_pred_val = df_hist.loc[val.index, f"hist_pred_{target}"].values
        hist_pred_test = df_hist.loc[test.index, f"hist_pred_{target}"].values

        y_val_true = val[target].values
        y_test_true = test[target].values

        # per-kind model selection on validation MAE
        val_kind = val["kind"].values
        test_kind = test["kind"].values
        for k in kinds:
            vmask = val_kind == k
            if vmask.sum() == 0:
                selection[(k, target)] = "GBM"
                continue
            mae_gbm = mean_absolute_error(y_val_true[vmask], gbm_pred_val[vmask])
            mae_hist = mean_absolute_error(y_val_true[vmask], hist_pred_val[vmask])
            selection[(k, target)] = "GBM" if mae_gbm <= mae_hist else "History"

        # apply selection to test set
        combined_test = np.where(
            np.isin(test_kind, [k for k in kinds if selection[(k, target)] == "GBM"]),
            gbm_pred_test, hist_pred_test,
        )
        final_preds[target] = combined_test
        gbm_only_preds[target] = gbm_pred_test
        hist_only_preds[target] = hist_pred_test

        results[target] = {
            "gbm_test_mae": mean_absolute_error(y_test_true, gbm_pred_test),
            "gbm_test_mape": mape(y_test_true, gbm_pred_test),
            "history_test_mae": mean_absolute_error(y_test_true, hist_pred_test),
            "history_test_mape": mape(y_test_true, hist_pred_test),
            "selected_test_mae": mean_absolute_error(y_test_true, combined_test),
            "selected_test_mape": mape(y_test_true, combined_test),
            "per_kind_selection": {k: selection[(k, target)] for k in kinds},
        }

    print(json.dumps(results, indent=2, default=str))

    with open(os.path.join(OUT_DIR, "predictor_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    test_out = test[["id", "job_name", "kind"] + TARGETS].copy()
    for t in TARGETS:
        test_out[f"pred_{t}"] = final_preds[t]
        test_out[f"pred_{t}_hist_only"] = hist_only_preds[t]
        test_out[f"pred_{t}_gbm_only"] = gbm_only_preds[t]
    test_out.to_parquet(os.path.join(OUT_DIR, "predictor_test_predictions.parquet"), index=False)
    print(f"\nWrote {os.path.join(OUT_DIR, 'predictor_results.json')}")
    print(f"Wrote {os.path.join(OUT_DIR, 'predictor_test_predictions.parquet')}")


if __name__ == "__main__":
    main()
