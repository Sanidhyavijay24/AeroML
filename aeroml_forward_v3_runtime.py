from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import aeroml_notebook_common as common

common.add_tf_helpers(globals())


def find_artifact(filename: str, search_roots: list[Path] | None = None) -> Path:
    search_roots = search_roots or [common.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
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
        self.search_roots = search_roots or [common.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        metrics_path = find_artifact("aeroml_xfoil_forward_v3_ensemble_metrics.json", self.search_roots)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.chosen_variant = metrics["chosen_variant"]

        model_paths = []
        for seed in [42, 52, 62]:
            filename = f"aeroml_xfoil_forward_v3_{self.chosen_variant}_seed{seed}.keras"
            model_paths.append(find_artifact(filename, self.search_roots))

        X_profile, X_scalar, X_flow, y_targets, meta = common.build_or_load_cached_dataset()
        split_manifest = pd.read_csv(find_artifact("aeroml_xfoil_split_manifest.csv", self.search_roots))
        train_idx, val_idx, test_idx = common.materialize_indices(meta, split_manifest)

        self.profile_scaler, _, _, _ = common.fit_transform_standard(
            X_profile[train_idx], X_profile[val_idx], X_profile[test_idx]
        )
        self.scalar_scaler, _, _, _ = common.fit_transform_standard(
            X_scalar[train_idx], X_scalar[val_idx], X_scalar[test_idx]
        )
        self.flow_scaler, _, _, _ = common.fit_transform_standard(
            X_flow[train_idx], X_flow[val_idx], X_flow[test_idx]
        )

        y_train_raw = y_targets[train_idx]
        self.ld_scaler = common.StandardScaler().fit(y_train_raw[:, [0]])
        self.cl_scaler = common.StandardScaler().fit(y_train_raw[:, [1]])
        self.cd_scaler = common.StandardScaler().fit(np.log(y_train_raw[:, [2]]))

        self.models = [keras.models.load_model(path, compile=False) for path in model_paths]

    def _predict_inputs(
        self,
        profile_features: np.ndarray,
        scalar_features: np.ndarray,
        re_value: float,
        mach_value: float,
    ) -> dict[str, Any]:
        profile = np.asarray(profile_features, dtype=np.float32).reshape(1, -1)
        scalar = np.asarray(scalar_features, dtype=np.float32).reshape(1, -1)
        flow = common.build_flow_features(re_value, mach_value).reshape(1, -1).astype(np.float32)

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
            pred, _ = common.decode_predictions(pred_scaled, self.ld_scaler, self.cl_scaler, self.cd_scaler)
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
        geom = common.geometry_representation(Path(dat_path))
        if geom is None:
            raise ValueError(f"Could not parse a valid airfoil geometry from {dat_path}")

        result = self._predict_inputs(geom["profile"], geom["scalar"], re_value, mach_value)
        result["geometry"] = {
            "fingerprint": geom["fingerprint"],
            "profile_features": geom["profile"],
            "scalar_features": geom["scalar"],
        }
        return result
