# -*- coding: utf-8 -*-
"""
@file features.py
@description Geometry parsing and feature engineering helpers
@module aeroml
"""

import math
import hashlib
import numpy as np

# np.trapz was removed in NumPy 2.0 (renamed to np.trapezoid). This keeps the codebase
# working on both old and new NumPy without pinning a version.
_trapz = getattr(np, "trapezoid", None) or np.trapz

N_STATIONS = 160
FINGERPRINT_DECIMALS = 5


def cosine_spacing(n_stations):
    beta = np.linspace(0.0, np.pi, n_stations)
    return 0.5 * (1.0 - np.cos(beta))


def estimate_le_radius(x_grid, thickness):
    nose_x = x_grid[1:6]
    nose_t = thickness[1:6]
    radius = 0.5 * (nose_t ** 2) / np.clip(nose_x, 1e-6, None)
    return float(np.median(radius))


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
            _trapz(thickness, x_grid),
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


def decode_predictions(pred_scaled, ld_scaler, cl_scaler, cd_scaler):
    ld_pred = ld_scaler.inverse_transform(pred_scaled["ldmax"]).ravel()
    cl_pred = cl_scaler.inverse_transform(pred_scaled["clmax"]).ravel()
    cd_log = cd_scaler.inverse_transform(pred_scaled["cdmin_log"]).ravel()
    cd_pred = np.exp(cd_log)
    return np.column_stack([ld_pred, cl_pred, cd_pred]), cd_log


# Mach grid constants derived from training data distributions.
# See "low-drag CdMin gap" section in README.md for context.
KNOWN_MACH_VALUES = [0.0, 0.10, 0.25, 0.50]
MACH_EXTRAPOLATION_THRESHOLD = 0.02


def mach_extrapolation_distance(mach_value: float) -> float:
    """
    Calculate the absolute distance from the nearest Mach value in the training dataset.

    :param mach_value: The user-specified Mach number.
    :return: The minimal distance from the known Mach values.
    """
    return min(abs(mach_value - m) for m in KNOWN_MACH_VALUES)

