# -*- coding: utf-8 -*-
"""
@file data.py
@description Dataset loading, splitting, and scaling helpers
@module aeroml
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from aeroml.features import geometry_representation, build_flow_features

warnings.filterwarnings("ignore")

SOURCE_NAME = "XFOIL ncrit=9"
RANDOM_STATE = 42

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

if (WORK_DIR / "Data_Cache").exists():
    CACHE_DIR = WORK_DIR / "Data_Cache"
else:
    CACHE_DIR = WORK_DIR

CACHE_DATA_PATH = CACHE_DIR / "aeroml_xfoil_n9_dataset.npz"
CACHE_META_PATH = CACHE_DIR / "aeroml_xfoil_n9_meta.csv"
SPLIT_MANIFEST_PATH = CACHE_DIR / "aeroml_xfoil_split_manifest.csv"


def discover_data_dir():
    explicit = os.environ.get("AEROML_DATA_DIR")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    search_roots = [Path("/kaggle/input"), Path.cwd()]
    candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_dir():
                continue

            dat_count = len(list(path.glob("*.dat")))
            if dat_count < 100:
                continue

            pkl_count = len(list(path.glob("*.pkl")))
            if dat_count == pkl_count and pkl_count > 0:
                candidates.append((dat_count, path))

    if not candidates:
        return Path("AeroML_Data")

    candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return candidates[0][1]


DATA_DIR = discover_data_dir()


def build_or_load_cached_dataset():
    if CACHE_DATA_PATH.exists() and CACHE_META_PATH.exists():
        cached = np.load(CACHE_DATA_PATH, allow_pickle=True)
        meta = pd.read_csv(CACHE_META_PATH)
        return (
            cached["X_profile"].astype(np.float32),
            cached["X_scalar"].astype(np.float32),
            cached["X_flow"].astype(np.float32),
            cached["y_targets"].astype(np.float32),
            meta,
        )

    dat_paths = sorted(DATA_DIR.glob("*.dat"))
    geom_by_fp = {}
    rows = []

    for dat_path in tqdm(dat_paths, desc="Parsing airfoils"):
        name = dat_path.stem
        pkl_path = DATA_DIR / f"{name}.pkl"
        if not pkl_path.exists():
            continue

        geom = geometry_representation(dat_path)
        if geom is None:
            continue

        geom_by_fp[geom["fingerprint"]] = geom

        try:
            df = pd.read_pickle(pkl_path)
        except Exception as exc:
            print(f"Skipping {pkl_path.name}: {exc}")
            continue

        required_cols = {"datasource", "Re", "Mach", "LDMax", "ClMax", "CdMin"}
        if not required_cols.issubset(df.columns):
            continue

        df = df.loc[df["datasource"].astype(str) == SOURCE_NAME, ["Re", "Mach", "LDMax", "ClMax", "CdMin"]].copy()
        if df.empty:
            continue

        for col in ["Re", "Mach", "LDMax", "ClMax", "CdMin"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([-99, -99.0], np.nan).dropna()
        df = df[
            (df["ClMax"] > 0.0)
            & (df["ClMax"] < 5.0)
            & (df["CdMin"] > 0.0)
            & (df["CdMin"] < 1.0)
            & (df["LDMax"] > 0.0)
            & (df["LDMax"] < 500.0)
        ].copy()
        if df.empty:
            continue

        df = df.groupby(["Re", "Mach"], as_index=False)[["LDMax", "ClMax", "CdMin"]].median()
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "name": name,
                    "fingerprint": geom["fingerprint"],
                    "Re": float(row.Re),
                    "Mach": float(row.Mach),
                    "LDMax": float(row.LDMax),
                    "ClMax": float(row.ClMax),
                    "CdMin": float(row.CdMin),
                }
            )

    if not rows:
        raise RuntimeError("No XFOIL rows survived the filtering step.")

    raw_meta = pd.DataFrame(rows)
    meta = (
        raw_meta.groupby(["fingerprint", "Re", "Mach"], as_index=False)
        .agg(
            LDMax=("LDMax", "median"),
            ClMax=("ClMax", "median"),
            CdMin=("CdMin", "median"),
            duplicate_rows=("name", "size"),
            duplicate_names=("name", "nunique"),
            example_name=("name", "first"),
        )
    )

    X_profile = np.stack([geom_by_fp[fp]["profile"] for fp in meta["fingerprint"]], axis=0).astype(np.float32)
    X_scalar = np.stack([geom_by_fp[fp]["scalar"] for fp in meta["fingerprint"]], axis=0).astype(np.float32)
    X_flow = np.stack(
        [build_flow_features(re_val, mach_val) for re_val, mach_val in zip(meta["Re"], meta["Mach"])],
        axis=0,
    ).astype(np.float32)
    y_targets = meta[["LDMax", "ClMax", "CdMin"]].to_numpy(dtype=np.float32)

    np.savez_compressed(
        CACHE_DATA_PATH,
        X_profile=X_profile,
        X_scalar=X_scalar,
        X_flow=X_flow,
        y_targets=y_targets,
    )
    meta.to_csv(CACHE_META_PATH, index=False)
    return X_profile, X_scalar, X_flow, y_targets, meta


def build_or_load_split_manifest(meta, random_state=RANDOM_STATE):
    if SPLIT_MANIFEST_PATH.exists():
        manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
        if set(manifest["split"]) == {"train", "val", "test"}:
            return manifest

    sample_index = np.arange(len(meta))
    groups = meta["fingerprint"].to_numpy()

    outer_split = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=random_state)
    train_val_idx, test_idx = next(outer_split.split(sample_index, groups=groups))

    inner_split = GroupShuffleSplit(n_splits=1, test_size=0.111111, random_state=random_state)
    train_rel_idx, val_rel_idx = next(inner_split.split(train_val_idx, groups=groups[train_val_idx]))
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    split_labels = np.full(len(meta), "unassigned", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    manifest = (
        pd.DataFrame({"fingerprint": meta["fingerprint"], "split": split_labels})
        .drop_duplicates()
        .sort_values(["split", "fingerprint"])
        .reset_index(drop=True)
    )
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
    return manifest


def materialize_indices(meta, manifest):
    split_map = dict(zip(manifest["fingerprint"], manifest["split"]))
    split_series = meta["fingerprint"].map(split_map)
    train_idx = np.flatnonzero(split_series.to_numpy() == "train")
    val_idx = np.flatnonzero(split_series.to_numpy() == "val")
    test_idx = np.flatnonzero(split_series.to_numpy() == "test")
    return train_idx, val_idx, test_idx


def fit_transform_standard(train_array, val_array, test_array):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_array).astype(np.float32)
    val_scaled = scaler.transform(val_array).astype(np.float32)
    test_scaled = scaler.transform(test_array).astype(np.float32)
    return scaler, train_scaled, val_scaled, test_scaled
