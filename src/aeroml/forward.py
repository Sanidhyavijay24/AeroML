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

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

import aeroml.data as data
import aeroml.features as features


class ArrayScaler:
    """Lightweight scaler mimicking StandardScaler interface with pre-computed mean and scale."""
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) * self.scale_) + self.mean_


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

        # Check for pre-computed scalers
        scalers_path = None
        try:
            scalers_path = find_artifact("aeroml_forward_scalers.npz", self.search_roots)
        except FileNotFoundError:
            pass

        # Check for ONNX models if onnxruntime is available
        onnx_model_paths = []
        if HAS_ORT:
            try:
                for seed in [42, 52, 62]:
                    fname = f"aeroml_xfoil_forward_v3_{self.chosen_variant}_seed{seed}.onnx"
                    onnx_model_paths.append(find_artifact(fname, self.search_roots))
            except FileNotFoundError:
                onnx_model_paths = []

        if scalers_path is not None and len(onnx_model_paths) == 3:
            # Fast, low-memory ONNX path: avoids loading 45MB CSV & TensorFlow runtime
            self.is_onnx = True
            scalers_data = np.load(scalers_path)
            self.profile_scaler = ArrayScaler(scalers_data["profile_mean"], scalers_data["profile_scale"])
            self.scalar_scaler = ArrayScaler(scalers_data["scalar_mean"], scalers_data["scalar_scale"])
            self.flow_scaler = ArrayScaler(scalers_data["flow_mean"], scalers_data["flow_scale"])
            self.ld_scaler = ArrayScaler(scalers_data["ld_mean"], scalers_data["ld_scale"])
            self.cl_scaler = ArrayScaler(scalers_data["cl_mean"], scalers_data["cl_scale"])
            self.cd_scaler = ArrayScaler(scalers_data["cd_mean"], scalers_data["cd_scale"])

            # Configure single-threaded ONNX session options for minimal memory
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.ort_sessions = [
                ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
                for p in onnx_model_paths
            ]
            self.ort_output_names = [
                [o.name for o in sess.get_outputs()] for sess in self.ort_sessions
            ]
            self.models = self.ort_sessions
            return

        # Fallback to Keras/TensorFlow model loading
        self.is_onnx = False
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from tensorflow import keras

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
        self._compiled_calls = [
            tf.function(lambda inputs, m=model: m(inputs, training=False), reduce_retracing=True)
            for model in self.models
        ]

        # Garbage collect massive data loaders immediately to prevent OOM
        del X_profile, X_scalar, X_flow, y_targets, meta
        import gc
        gc.collect()

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

        if self.is_onnx:
            input_feed = {
                "profile": profile_scaled,
                "scalar": scalar_scaled,
                "flow": flow_scaled,
            }
            preds = []
            for sess, out_names in zip(self.ort_sessions, self.ort_output_names):
                raw_outputs = sess.run(None, input_feed)
                pred_scaled = dict(zip(out_names, raw_outputs))
                pred, _ = features.decode_predictions(pred_scaled, self.ld_scaler, self.cl_scaler, self.cd_scaler)
                preds.append(pred)  # (P, 3)

            preds = np.stack(preds, axis=0)  # (n_models, P, 3)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)
            return mean_pred, std_pred

        import tensorflow as tf
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

        if self.is_onnx:
            input_feed = {
                "profile": profile_scaled,
                "scalar": scalar_scaled,
                "flow": flow_scaled,
            }
            preds = []
            for sess, out_names in zip(self.ort_sessions, self.ort_output_names):
                raw_outputs = sess.run(None, input_feed)
                pred_scaled = dict(zip(out_names, raw_outputs))
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
