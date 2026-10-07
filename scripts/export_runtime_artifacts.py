# -*- coding: utf-8 -*-
"""
@file export_runtime_artifacts.py
@description Export Keras ensemble models to ONNX and pre-compute scalers/PCA artifacts for low-memory runtime
@module scripts
"""

import json
import sys
from pathlib import Path

# Bootstrap local src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import aeroml.data as data
import aeroml.features as features


def export_models_to_onnx(output_dir: Path):
    print("--- Exporting Keras Ensemble to ONNX ---")
    seeds = [42, 52, 62]
    variant = "cd_loss_only"
    dummy_input = [
        np.zeros((1, 640), dtype=np.float32),
        np.zeros((1, 16), dtype=np.float32),
        np.zeros((1, 5), dtype=np.float32),
    ]

    for seed in seeds:
        keras_path = output_dir / f"aeroml_xfoil_forward_v3_{variant}_seed{seed}.keras"
        onnx_path = output_dir / f"aeroml_xfoil_forward_v3_{variant}_seed{seed}.onnx"
        print(f"Loading {keras_path.name}...")
        model = keras.models.load_model(keras_path, compile=False)
        # Call model to build state
        model(dummy_input)
        print(f"Exporting to {onnx_path.name}...")
        model.export(onnx_path, format="onnx")
        print(f"Successfully exported {onnx_path.name} ({onnx_path.stat().st_size / 1024:.1f} KB)")


def export_scalers_and_geometry_space(output_dir: Path):
    print("\n--- Pre-computing and Exporting Scalers and Geometry Space ---")
    X_profile, X_scalar, X_flow, y_targets, meta = data.build_or_load_cached_dataset()
    split_manifest = pd.read_csv(Path("Data_Cache") / "aeroml_xfoil_split_manifest.csv")
    train_idx, val_idx, test_idx = data.materialize_indices(meta, split_manifest)

    # 1. Forward Scalers
    print("Computing StandardScaler parameters...")
    profile_scaler = StandardScaler().fit(X_profile[train_idx])
    scalar_scaler = StandardScaler().fit(X_scalar[train_idx])
    flow_scaler = StandardScaler().fit(X_flow[train_idx])

    y_train_raw = y_targets[train_idx]
    ld_scaler = StandardScaler().fit(y_train_raw[:, [0]])
    cl_scaler = StandardScaler().fit(y_train_raw[:, [1]])
    cd_scaler = StandardScaler().fit(np.log(y_train_raw[:, [2]]))

    scalers_npz_path = output_dir / "aeroml_forward_scalers.npz"
    np.savez_compressed(
        scalers_npz_path,
        profile_mean=profile_scaler.mean_.astype(np.float32),
        profile_scale=profile_scaler.scale_.astype(np.float32),
        scalar_mean=scalar_scaler.mean_.astype(np.float32),
        scalar_scale=scalar_scaler.scale_.astype(np.float32),
        flow_mean=flow_scaler.mean_.astype(np.float32),
        flow_scale=flow_scaler.scale_.astype(np.float32),
        ld_mean=ld_scaler.mean_.astype(np.float32),
        ld_scale=ld_scaler.scale_.astype(np.float32),
        cl_mean=cl_scaler.mean_.astype(np.float32),
        cl_scale=cl_scaler.scale_.astype(np.float32),
        cd_mean=cd_scaler.mean_.astype(np.float32),
        cd_scale=cd_scaler.scale_.astype(np.float32),
    )
    print(f"Saved scalers to {scalers_npz_path.name} ({scalers_npz_path.stat().st_size / 1024:.1f} KB)")

    # 2. Reverse Geometry Space & PCA
    print("Fitting PCA and extracting latent bounds...")
    n_stations = features.N_STATIONS
    thickness_train = X_profile[train_idx, :n_stations]
    camber_train = X_profile[train_idx, n_stations : 2 * n_stations]
    shape_train = np.concatenate([thickness_train, camber_train], axis=1).astype(np.float32)

    pca = PCA(n_components=12, random_state=data.RANDOM_STATE)
    z_train = pca.fit_transform(shape_train)

    latent_low = np.quantile(z_train, 0.01, axis=0)
    latent_high = np.quantile(z_train, 0.99, axis=0)
    latent_span = np.maximum(latent_high - latent_low, 1e-6)

    max_thickness_train = thickness_train.max(axis=1)
    max_camber_train = np.abs(camber_train).max(axis=1)
    te_thickness_train = thickness_train[:, -1]

    geom_limits = {
        "max_thickness_min": float(np.quantile(max_thickness_train, 0.001)),
        "max_thickness_max": float(np.quantile(max_thickness_train, 0.999)),
        "max_camber_max": float(np.quantile(max_camber_train, 0.999)),
        "te_thickness_min": float(np.quantile(te_thickness_train, 0.001)),
        "te_thickness_max": float(np.quantile(te_thickness_train, 0.999)),
    }

    ld_scale = float(np.std(y_train_raw[:, 0]))
    cl_scale = float(np.std(y_train_raw[:, 1]))
    cd_log_scale = float(np.std(np.log(y_train_raw[:, 2])))

    # Minimal train_meta columns needed for reverse local_flow_pool
    train_meta_slice = meta.iloc[train_idx][["Re", "Mach", "LDMax", "ClMax", "CdMin"]].copy()

    geom_npz_path = output_dir / "aeroml_reverse_geometry_space.npz"
    np.savez_compressed(
        geom_npz_path,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        z_train=z_train.astype(np.float32),
        latent_low=latent_low.astype(np.float64),
        latent_high=latent_high.astype(np.float64),
        latent_span=latent_span.astype(np.float64),
        train_re=train_meta_slice["Re"].to_numpy(dtype=np.float64),
        train_mach=train_meta_slice["Mach"].to_numpy(dtype=np.float64),
        train_ldmax=train_meta_slice["LDMax"].to_numpy(dtype=np.float64),
        train_clmax=train_meta_slice["ClMax"].to_numpy(dtype=np.float64),
        train_cdmin=train_meta_slice["CdMin"].to_numpy(dtype=np.float64),
        scales=np.array([ld_scale, cl_scale, cd_log_scale], dtype=np.float64),
    )
    print(f"Saved geometry space to {geom_npz_path.name} ({geom_npz_path.stat().st_size / 1024:.1f} KB)")

    limits_json_path = output_dir / "aeroml_reverse_geom_limits.json"
    with open(limits_json_path, "w", encoding="utf-8") as f:
        json.dump(geom_limits, f, indent=2)
    print(f"Saved geometric limits to {limits_json_path.name}")


if __name__ == "__main__":
    out_dir = Path("Forward_outputs")
    export_models_to_onnx(out_dir)
    export_scalers_and_geometry_space(out_dir)
    print("\n--- All Runtime Artifacts Exported Successfully! ---")
