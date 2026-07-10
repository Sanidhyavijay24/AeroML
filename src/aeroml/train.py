# -*- coding: utf-8 -*-
"""
@file train.py
@description Standalone parameterized training logic for the forward ensemble model
@module aeroml
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks

import aeroml.data as data
import aeroml.features as features
from aeroml.models import set_all_seeds, build_forward_model
from aeroml.evaluate import collect_metrics, BASELINE_METRICS_FALLBACK


def make_variant_weights(variant_name, y_raw, mach_values, cd_q10, cd_q25):
    """Compute sample weights and loss weights depending on the variant."""
    n = len(y_raw)
    weights = {name: np.ones(n, dtype="float32") for name in ["ldmax", "clmax", "cdmin_log"]}
    loss_weights = {"ldmax": 1.0, "clmax": 1.0, "cdmin_log": 1.0}

    if variant_name == "cd_loss_only":
        loss_weights["cdmin_log"] = 1.12

    elif variant_name == "low_drag_only":
        cd_w = weights["cdmin_log"].copy()
        cd_w[y_raw[:, 2] <= cd_q25] = 1.30
        cd_w[y_raw[:, 2] <= cd_q10] = 1.65
        weights["cdmin_log"] = cd_w
        loss_weights["cdmin_log"] = 1.05

    elif variant_name == "low_drag_plus_mach05":
        cd_w = weights["cdmin_log"].copy()
        cd_w[y_raw[:, 2] <= cd_q25] = 1.35
        cd_w[y_raw[:, 2] <= cd_q10] = 1.75
        cd_w[mach_values >= 0.49] *= 1.20
        weights["cdmin_log"] = cd_w
        loss_weights["cdmin_log"] = 1.05

    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return weights, loss_weights


def train_forward_ensemble(
    seeds: list[int],
    variant: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str
) -> dict:
    """Runs the parameterized training pipeline for the forward models ensemble."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading cached dataset...")
    X_profile, X_scalar, X_flow, y_targets, meta = data.build_or_load_cached_dataset()

    print("Loading split manifest...")
    split_manifest = data.build_or_load_split_manifest(meta)
    train_idx, val_idx, test_idx = data.materialize_indices(meta, split_manifest)

    # Extract raw subsets
    Xp_train, Xp_val, Xp_test = X_profile[train_idx], X_profile[val_idx], X_profile[test_idx]
    Xs_train, Xs_val, Xs_test = X_scalar[train_idx], X_scalar[val_idx], X_scalar[test_idx]
    Xf_train, Xf_val, Xf_test = X_flow[train_idx], X_flow[val_idx], X_flow[test_idx]

    y_train_raw = y_targets[train_idx]
    y_val_raw = y_targets[val_idx]
    y_test_raw = y_targets[test_idx]

    # Compute scale parameters
    print("Fitting input feature scalers...")
    profile_scaler, Xp_train_sc, Xp_val_sc, Xp_test_sc = data.fit_transform_standard(Xp_train, Xp_val, Xp_test)
    scalar_scaler, Xs_train_sc, Xs_val_sc, Xs_test_sc = data.fit_transform_standard(Xs_train, Xs_val, Xs_test)
    flow_scaler, Xf_train_sc, Xf_val_sc, Xf_test_sc = data.fit_transform_standard(Xf_train, Xf_val, Xf_test)

    # Fit target scalers
    print("Fitting target scaling parameters...")
    ld_scaler = StandardScaler().fit(y_train_raw[:, [0]])
    cl_scaler = StandardScaler().fit(y_train_raw[:, [1]])
    cd_scaler = StandardScaler().fit(np.log(y_train_raw[:, [2]]))

    y_train_ld = ld_scaler.transform(y_train_raw[:, [0]]).astype("float32")
    y_val_ld = ld_scaler.transform(y_val_raw[:, [0]]).astype("float32")

    y_train_cl = cl_scaler.transform(y_train_raw[:, [1]]).astype("float32")
    y_val_cl = cl_scaler.transform(y_val_raw[:, [1]]).astype("float32")

    y_train_cd_log = cd_scaler.transform(np.log(y_train_raw[:, [2]])).astype("float32")
    y_val_cd_log = cd_scaler.transform(np.log(y_val_raw[:, [2]])).astype("float32")

    # Quantiles for low-drag thresholds
    cd_q10 = float(np.quantile(y_train_raw[:, 2], 0.10))
    cd_q25 = float(np.quantile(y_train_raw[:, 2], 0.25))

    # Build weights
    train_sw, loss_weights = make_variant_weights(variant, y_train_raw, meta.iloc[train_idx]["Mach"].to_numpy(), cd_q10, cd_q25)
    val_sw, _ = make_variant_weights(variant, y_val_raw, meta.iloc[val_idx]["Mach"].to_numpy(), cd_q10, cd_q25)

    final_runs = []
    for seed in seeds:
        print(f"\n" + "=" * 72)
        print(f"Training final v3 champion | variant={variant} | seed={seed}")
        print("=" * 72)

        set_all_seeds(seed)
        tf.keras.backend.clear_session()

        model = build_forward_model(Xp_train_sc.shape[1], Xs_train_sc.shape[1], Xf_train_sc.shape[1])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={name: tf.keras.losses.Huber(delta=1.0) for name in ["ldmax", "clmax", "cdmin_log"]},
            loss_weights=loss_weights,
            metrics={name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in ["ldmax", "clmax", "cdmin_log"]},
        )

        history = model.fit(
            x={"profile": Xp_train_sc, "scalar": Xs_train_sc, "flow": Xf_train_sc},
            y={"ldmax": y_train_ld, "clmax": y_train_cl, "cdmin_log": y_train_cd_log},
            sample_weight=train_sw,
            validation_data=(
                {"profile": Xp_val_sc, "scalar": Xs_val_sc, "flow": Xf_val_sc},
                {"ldmax": y_val_ld, "clmax": y_val_cl, "cdmin_log": y_val_cd_log},
                val_sw,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ],
            verbose=1,
        )

        # Predict
        pred_val_scaled = model.predict({"profile": Xp_val_sc, "scalar": Xs_val_sc, "flow": Xf_val_sc}, batch_size=4096, verbose=0)
        pred_test_scaled = model.predict({"profile": Xp_test_sc, "scalar": Xs_test_sc, "flow": Xf_test_sc}, batch_size=4096, verbose=0)

        y_val_pred, _ = features.decode_predictions(pred_val_scaled, ld_scaler, cl_scaler, cd_scaler)
        y_test_pred, test_cd_log = features.decode_predictions(pred_test_scaled, ld_scaler, cl_scaler, cd_scaler)

        val_metrics = collect_metrics(y_val_raw, y_val_pred)
        test_metrics = collect_metrics(y_test_raw, y_test_pred)

        low_drag_mask = y_test_raw[:, 2] <= cd_q25
        low_drag_metrics = collect_metrics(y_test_raw[low_drag_mask], y_test_pred[low_drag_mask])

        # Save model
        model_filename = f"aeroml_xfoil_forward_v3_{variant}_seed{seed}.keras"
        model.save(output_path / model_filename)

        final_runs.append({
            "seed": seed,
            "variant": variant,
            "best_val_loss": float(min(history.history["val_loss"])),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "low_drag_metrics": low_drag_metrics,
            "test_pred": y_test_pred,
            "test_cd_log": test_cd_log,
        })

    # Compile seed metric rows
    seed_metric_rows = []
    for run in final_runs:
        seed_metric_rows.append({
            "seed": run["seed"],
            "variant": run["variant"],
            "best_val_loss": run["best_val_loss"],
            "LDMax_R2": run["test_metrics"]["LDMax"]["R2"],
            "ClMax_R2": run["test_metrics"]["ClMax"]["R2"],
            "CdMin_R2": run["test_metrics"]["CdMin"]["R2"],
            "CdMin_MAE": run["test_metrics"]["CdMin"]["MAE"],
            "CdMin_MedianAE": run["test_metrics"]["CdMin"]["MedianAE"],
            "CdMin_Within25Pct": run["test_metrics"]["CdMin"]["Within25Pct"],
            "LowDrag_CdMin_Within25Pct": run["low_drag_metrics"]["CdMin"]["Within25Pct"],
        })

    seed_metrics_df = pd.DataFrame(seed_metric_rows).sort_values("best_val_loss").reset_index(drop=True)
    seed_metrics_df.to_csv(output_path / f"aeroml_xfoil_forward_v3_seed_metrics.csv", index=False)
    print("\nChampion variant seed metrics:")
    print(seed_metrics_df.to_string(index=False))

    best_run = min(final_runs, key=lambda item: item["best_val_loss"])

    # Compute ensemble predictions
    ensemble_ld = np.mean(np.stack([run["test_pred"][:, 0] for run in final_runs], axis=0), axis=0)
    ensemble_cl = np.mean(np.stack([run["test_pred"][:, 1] for run in final_runs], axis=0), axis=0)
    ensemble_cd = np.exp(np.mean(np.stack([run["test_cd_log"] for run in final_runs], axis=0), axis=0))
    ensemble_pred = np.column_stack([ensemble_ld, ensemble_cl, ensemble_cd])

    ensemble_metrics = collect_metrics(y_test_raw, ensemble_pred)
    low_drag_mask = y_test_raw[:, 2] <= cd_q25
    low_drag_metrics = collect_metrics(y_test_raw[low_drag_mask], ensemble_pred[low_drag_mask])

    stability = {
        "LDMax_R2_std": float(seed_metrics_df["LDMax_R2"].std(ddof=0)),
        "ClMax_R2_std": float(seed_metrics_df["ClMax_R2"].std(ddof=0)),
        "CdMin_R2_std": float(seed_metrics_df["CdMin_R2"].std(ddof=0)),
        "LowDrag_CdMin_Within25Pct_min": float(seed_metrics_df["LowDrag_CdMin_Within25Pct"].min()),
    }

    stability_ok = (
        stability["LDMax_R2_std"] <= 0.02
        and stability["ClMax_R2_std"] <= 0.02
        and stability["CdMin_R2_std"] <= 0.03
        and stability["LowDrag_CdMin_Within25Pct_min"] >= 0.75
    )

    gate_primary = (
        ensemble_metrics["LDMax"]["R2"] >= 0.90
        and ensemble_metrics["ClMax"]["R2"] >= 0.86
        and (
            ensemble_metrics["CdMin"]["R2"] >= 0.75
            or (
                ensemble_metrics["CdMin"]["MedianAE"] <= 0.012
                and ensemble_metrics["CdMin"]["Within25Pct"] >= 0.80
            )
        )
    )

    baseline_metrics = BASELINE_METRICS_FALLBACK

    metrics_payload = {
        "chosen_variant": variant,
        "baseline_metrics": baseline_metrics,
        "best_seed": int(best_run["seed"]),
        "best_seed_metrics": best_run["test_metrics"],
        "ensemble_metrics": ensemble_metrics,
        "low_drag_ensemble_metrics": low_drag_metrics,
        "stability": stability,
        "gate_primary": bool(gate_primary),
        "stability_ok": bool(stability_ok),
        "gate_pass": bool(gate_primary and stability_ok),
    }

    # Write ensemble metrics json
    metrics_json_path = output_path / f"aeroml_xfoil_forward_v3_ensemble_metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Save ensemble predictions csv
    ensemble_frame = meta.iloc[test_idx][["example_name", "fingerprint", "Re", "Mach", "duplicate_rows", "duplicate_names"]].copy()
    ensemble_frame["LDMax_true"] = y_test_raw[:, 0]
    ensemble_frame["LDMax_pred"] = ensemble_pred[:, 0]
    ensemble_frame["ClMax_true"] = y_test_raw[:, 1]
    ensemble_frame["ClMax_pred"] = ensemble_pred[:, 1]
    ensemble_frame["CdMin_true"] = y_test_raw[:, 2]
    ensemble_frame["CdMin_pred"] = ensemble_pred[:, 2]
    ensemble_frame.to_csv(output_path / f"aeroml_xfoil_forward_v3_ensemble_predictions.csv", index=False)

    print(f"\nSaved ensemble metrics to: {metrics_json_path}")
    return metrics_payload
