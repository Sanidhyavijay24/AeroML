import hashlib
import json
import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SOURCE_NAME = "XFOIL ncrit=9"
N_STATIONS = 160
FINGERPRINT_DECIMALS = 5
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
        raise FileNotFoundError(
            "Could not auto-discover the AeroML dataset. "
            "Set AEROML_DATA_DIR manually if needed."
        )

    candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return candidates[0][1]


DATA_DIR = discover_data_dir()


def read_dat_file(path):
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 20:
        return None

    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(coords, axis=0)) > 1e-12, axis=1)
    coords = coords[keep]
    return coords if len(coords) >= 20 else None


def normalize_coords(coords):
    coords = np.asarray(coords, dtype=np.float64).copy()
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    chord = x_max - x_min
    if chord <= 1e-8:
        return None

    coords[:, 0] = (coords[:, 0] - x_min) / chord
    coords[:, 1] = coords[:, 1] / chord
    return coords


def split_upper_lower(coords):
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1]
    lower = coords[le_idx:]

    if len(upper) < 5 or len(lower) < 5:
        return None, None

    if upper[0, 0] < upper[-1, 0]:
        upper = upper[::-1]
    if lower[0, 0] > lower[-1, 0]:
        lower = lower[::-1]

    return upper, lower


def prepare_surface_for_interp(surface):
    surface = np.asarray(surface, dtype=np.float64)
    order = np.argsort(surface[:, 0])
    surface = surface[order]

    rounded_x = np.round(surface[:, 0], 10)
    _, unique_idx = np.unique(rounded_x, return_index=True)
    surface = surface[np.sort(unique_idx)]
    return surface if len(surface) >= 5 else None


def cosine_spacing(n_stations):
    beta = np.linspace(0.0, np.pi, n_stations)
    return 0.5 * (1.0 - np.cos(beta))


def estimate_le_radius(x_grid, thickness):
    nose_x = x_grid[1:6]
    nose_t = thickness[1:6]
    radius = 0.5 * (nose_t ** 2) / np.clip(nose_x, 1e-6, None)
    return float(np.median(radius))


def geometry_representation(dat_path, n_stations=N_STATIONS):
    coords = read_dat_file(dat_path)
    if coords is None:
        return None

    coords = normalize_coords(coords)
    if coords is None:
        return None

    upper, lower = split_upper_lower(coords)
    if upper is None or lower is None:
        return None

    upper = prepare_surface_for_interp(upper)
    lower = prepare_surface_for_interp(lower)
    if upper is None or lower is None:
        return None

    x_grid = cosine_spacing(n_stations)
    y_upper = np.interp(x_grid, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_grid, lower[:, 0], lower[:, 1])

    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)
    dyu_dx = np.gradient(y_upper, x_grid)
    dyl_dx = np.gradient(y_lower, x_grid)

    curv_upper = np.gradient(dyu_dx, x_grid) / np.maximum((1.0 + dyu_dx**2) ** 1.5, 1e-6)
    curv_lower = np.gradient(dyl_dx, x_grid) / np.maximum((1.0 + dyl_dx**2) ** 1.5, 1e-6)

    max_t_idx = int(np.argmax(thickness))
    max_c_idx = int(np.argmax(np.abs(camber)))

    scalar_features = np.array(
        [
            thickness[max_t_idx],
            x_grid[max_t_idx],
            camber[max_c_idx],
            x_grid[max_c_idx],
            np.max(camber),
            np.min(camber),
            estimate_le_radius(x_grid, thickness),
            thickness[-1],
            math.degrees(math.atan(dyu_dx[-1]) - math.atan(dyl_dx[-1])),
            np.trapz(thickness, x_grid),
            np.sum(np.sqrt(np.diff(x_grid) ** 2 + np.diff(y_upper) ** 2)),
            np.sum(np.sqrt(np.diff(x_grid) ** 2 + np.diff(y_lower) ** 2)),
            np.mean(np.abs(curv_upper)),
            np.mean(np.abs(curv_lower)),
            np.max(np.abs(curv_upper)),
            np.max(np.abs(curv_lower)),
        ],
        dtype=np.float32,
    )

    profile_features = np.concatenate([thickness, camber, dyu_dx, dyl_dx], axis=0).astype(np.float32)
    fingerprint_payload = np.round(np.concatenate([y_upper, y_lower]), FINGERPRINT_DECIMALS).astype(np.float32)
    fingerprint = hashlib.sha1(fingerprint_payload.tobytes()).hexdigest()

    return {
        "profile": profile_features,
        "scalar": scalar_features,
        "fingerprint": fingerprint,
    }


def build_flow_features(re_value, mach_value):
    re_value = float(re_value)
    mach_value = float(mach_value)
    return np.array(
        [
            np.log10(re_value),
            mach_value,
            mach_value**2,
            1.0 / np.sqrt(re_value),
            1.0 / np.sqrt(max(1.0 - mach_value**2, 1e-6)),
        ],
        dtype=np.float32,
    )


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


def regression_report(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_tf_helpers(namespace):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    namespace["tf"] = tf
    namespace["keras"] = keras
    namespace["layers"] = layers

    def set_all_seeds(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)

    def dense_block(x, units, dropout):
        x = layers.Dense(units, kernel_initializer="he_normal")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("swish")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        return x

    def build_forward_model(profile_dim, scalar_dim, flow_dim):
        profile_in = layers.Input(shape=(profile_dim,), name="profile")
        scalar_in = layers.Input(shape=(scalar_dim,), name="scalar")
        flow_in = layers.Input(shape=(flow_dim,), name="flow")

        p = layers.GaussianNoise(0.01)(profile_in)
        p = dense_block(p, 512, 0.10)
        p = dense_block(p, 256, 0.10)
        p = dense_block(p, 128, 0.05)

        s = dense_block(scalar_in, 64, 0.05)
        s = dense_block(s, 32, 0.00)

        f = dense_block(flow_in, 64, 0.05)
        f = dense_block(f, 32, 0.00)

        x = layers.Concatenate()([p, s, f])
        x = dense_block(x, 256, 0.10)
        x = dense_block(x, 128, 0.05)
        shared = dense_block(x, 64, 0.00)

        ld_head = dense_block(shared, 32, 0.00)
        cl_head = dense_block(shared, 32, 0.00)
        cd_head = dense_block(shared, 32, 0.00)

        outputs = {
            "ldmax": layers.Dense(1, name="ldmax")(ld_head),
            "clmax": layers.Dense(1, name="clmax")(cl_head),
            "cdmin_log": layers.Dense(1, name="cdmin_log")(cd_head),
        }

        return keras.Model(
            inputs=[profile_in, scalar_in, flow_in],
            outputs=outputs,
            name="AeroML_XFOIL_Forward_MLP",
        )

    namespace["set_all_seeds"] = set_all_seeds
    namespace["build_forward_model"] = build_forward_model


def decode_predictions(pred_scaled, ld_scaler, cl_scaler, cd_scaler):
    ld_pred = ld_scaler.inverse_transform(pred_scaled["ldmax"]).ravel()
    cl_pred = cl_scaler.inverse_transform(pred_scaled["clmax"]).ravel()
    cd_log = cd_scaler.inverse_transform(pred_scaled["cdmin_log"]).ravel()
    cd_pred = np.exp(cd_log)
    return np.column_stack([ld_pred, cl_pred, cd_pred]), cd_log


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
