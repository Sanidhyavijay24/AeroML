# -*- coding: utf-8 -*-
"""
@file forward.py
@description Forward prediction pipeline runtime for airfoil performance estimation
@module aeroml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

import aeroml.data as data
import aeroml.features as features


def find_artifact(filename: str, search_roots: list[Path] | None = None) -> Path:
    search_roots = search_roots or [data.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(filename))
        if matches:
            matches.sort(key=lambda p: len(str(p)))
            return matches[0]
    raise FileNotFoundError(f"Could not find artifact: {filename}")


class ForwardV3Predictor:
    def __init__(self, search_roots: list[Path] | None = None):
        self.search_roots = search_roots or [data.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        metrics_path = find_artifact("aeroml_xfoil_forward_v3_ensemble_metrics.json", self.search_roots)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.chosen_variant = metrics["chosen_variant"]

        model_paths = []
        for seed in [42, 52, 62]:
            filename = f"aeroml_xfoil_forward_v3_{self.chosen_variant}_seed{seed}.keras"
            model_paths.append(find_artifact(filename, self.search_roots))

        X_profile, X_scalar, X_flow, y_targets, meta = data.build_or_load_cached_dataset()
        split_manifest = pd.read_csv(find_artifact("aeroml_xfoil_split_manifest.csv", self.search_roots))
        train_idx, val_idx, test_idx = data.materialize_indices(meta, split_manifest)

        self.profile_scaler, _, _, _ = data.fit_transform_standard(
            X_profile[train_idx], X_profile[val_idx], X_profile[test_idx]
        )
        self.scalar_scaler, _, _, _ = data.fit_transform_standard(
            X_scalar[train_idx], X_scalar[val_idx], X_scalar[test_idx]
        )
        self.flow_scaler, _, _, _ = data.fit_transform_standard(
            X_flow[train_idx], X_flow[val_idx], X_flow[test_idx]
        )

        y_train_raw = y_targets[train_idx]
        self.ld_scaler = StandardScaler().fit(y_train_raw[:, [0]])
        self.cl_scaler = StandardScaler().fit(y_train_raw[:, [1]])
        self.cd_scaler = StandardScaler().fit(np.log(y_train_raw[:, [2]]))

        self.models = [keras.models.load_model(path, compile=False) for path in model_paths]

        # Wrapping each model call in tf.function traces the graph once and reuses it,
        # instead of paying Python/Keras dispatch overhead on every single call. The
        # trace is keyed on input shape, so as long as callers use a stable batch size
        # (e.g. always calling _predict_batch with the same population size within one
        # search run) this only pays the tracing cost once per shape, not once per call.
        self._compiled_calls = [
            tf.function(lambda inputs, m=model: m(inputs, training=False), reduce_retracing=True)
            for model in self.models
        ]

    def _predict_batch(
        self,
        profile_features: np.ndarray,
        scalar_features: np.ndarray,
        re_value: float,
        mach_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched counterpart to _predict_inputs. Runs P samples through the ensemble
        in a single call per model instead of one call per sample, which is what makes
        the reverse search fast: the bottleneck was never the model's FLOPs, it was
        thousands of single-sample Python/TF dispatches.

        Returns (mean_pred, std_pred), each shaped (P, 3) in [LDMax, ClMax, CdMin] order.
        """
        profile = np.asarray(profile_features, dtype=np.float32)
        scalar = np.asarray(scalar_features, dtype=np.float32)
        n = profile.shape[0]

        flow_row = features.build_flow_features(re_value, mach_value).astype(np.float32)
        flow = np.tile(flow_row, (n, 1))

        profile_scaled = self.profile_scaler.transform(profile).astype(np.float32)
        scalar_scaled = self.scalar_scaler.transform(scalar).astype(np.float32)
        flow_scaled = self.flow_scaler.transform(flow).astype(np.float32)

        inputs = {
            "profile": tf.constant(profile_scaled),
            "scalar": tf.constant(scalar_scaled),
            "flow": tf.constant(flow_scaled),
        }

        preds = []
        for call_fn in self._compiled_calls:
            pred_scaled = call_fn(inputs)
            pred_scaled = {key: value.numpy() for key, value in pred_scaled.items()}
            pred, _ = features.decode_predictions(pred_scaled, self.ld_scaler, self.cl_scaler, self.cd_scaler)
            preds.append(pred)  # (P, 3)

        preds = np.stack(preds, axis=0)  # (n_models, P, 3)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        return mean_pred, std_pred

    def _predict_inputs(
        self,
        profile_features: np.ndarray,
        scalar_features: np.ndarray,
        re_value: float,
        mach_value: float,
    ) -> dict[str, Any]:
        profile = np.asarray(profile_features, dtype=np.float32).reshape(1, -1)
        scalar = np.asarray(scalar_features, dtype=np.float32).reshape(1, -1)
        flow = features.build_flow_features(re_value, mach_value).reshape(1, -1).astype(np.float32)

        profile_scaled = self.profile_scaler.transform(profile).astype(np.float32)
        scalar_scaled = self.scalar_scaler.transform(scalar).astype(np.float32)
        flow_scaled = self.flow_scaler.transform(flow).astype(np.float32)

        preds = []
        for model in self.models:
            pred_scaled = model(
                {"profile": profile_scaled, "scalar": scalar_scaled, "flow": flow_scaled},
                training=False,
            )
            pred_scaled = {key: value.numpy() for key, value in pred_scaled.items()}
            pred, _ = features.decode_predictions(pred_scaled, self.ld_scaler, self.cl_scaler, self.cd_scaler)
            preds.append(pred[0])

        preds = np.asarray(preds, dtype=np.float64)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        return {
            "predictions": {
                "LDMax": float(mean_pred[0]),
                "ClMax": float(mean_pred[1]),
                "CdMin": float(mean_pred[2]),
            },
            "uncertainty": {
                "LDMax_std": float(std_pred[0]),
                "ClMax_std": float(std_pred[1]),
                "CdMin_std": float(std_pred[2]),
                "CdMin_rel_std": float(std_pred[2] / max(mean_pred[2], 1e-6)),
            },
            "ensemble_predictions": preds,
        }

    def predict_from_dat_file(self, dat_path: str | Path, re_value: float, mach_value: float) -> dict[str, Any]:
        geom = features.geometry_representation(Path(dat_path))
        if geom is None:
            raise ValueError(f"Could not parse a valid airfoil geometry from {dat_path}")

        result = self._predict_inputs(geom["profile"], geom["scalar"], re_value, mach_value)
        result["geometry"] = {
            "fingerprint": geom["fingerprint"],
            "profile_features": geom["profile"],
            "scalar_features": geom["scalar"],
        }
        return result
