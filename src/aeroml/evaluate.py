# -*- coding: utf-8 -*-
"""
@file evaluate.py
@description Model evaluation and metric computation utilities
@module aeroml
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hardcoded fallback metrics for Baseline v1 ensemble
BASELINE_METRICS_FALLBACK = {
    "LDMax": {
        "MAE": 5.347763538360596,
        "RMSE": 16.63013442067169,
        "R2": 0.9058724641799927
    },
    "ClMax": {
        "MAE": 0.04613952338695526,
        "RMSE": 0.12314485251908894,
        "R2": 0.8681695461273193
    },
    "CdMin": {
        "MAE": 0.026333842426538467,
        "RMSE": 0.045418272011022415,
        "R2": 0.7169221639633179,
        "MedianAE": 0.012432217597961426,
        "Within10Pct": 0.5433972707725715,
        "Within25Pct": 0.785238328401422,
        "Within50Pct": 0.8968769292998106
    }
}


def regression_report(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def collect_metrics(y_true, y_pred):
    metrics = {
        "LDMax": regression_report(y_true[:, 0], y_pred[:, 0]),
        "ClMax": regression_report(y_true[:, 1], y_pred[:, 1]),
        "CdMin": regression_report(y_true[:, 2], y_pred[:, 2]),
    }
    cd_rel_err = np.abs((y_pred[:, 2] - y_true[:, 2]) / np.clip(y_true[:, 2], 1e-8, None))
    metrics["CdMin"]["MedianAE"] = float(np.median(np.abs(y_pred[:, 2] - y_true[:, 2])))
    metrics["CdMin"]["Within10Pct"] = float((cd_rel_err <= 0.10).mean())
    metrics["CdMin"]["Within25Pct"] = float((cd_rel_err <= 0.25).mean())
    metrics["CdMin"]["Within50Pct"] = float((cd_rel_err <= 0.50).mean())
    return metrics
