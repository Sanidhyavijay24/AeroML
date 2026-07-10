# -*- coding: utf-8 -*-
"""
@file reverse.py
@description Reverse airfoil design using latent search/optimization
@module aeroml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import aeroml.data as data
import aeroml.features as features
from aeroml.forward import ForwardV3Predictor, find_artifact

_trapz = features._trapz


class ReverseV3Designer:
    def __init__(self, search_roots: list[Path] | None = None):
        self.search_roots = search_roots or [data.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
        self.forward = ForwardV3Predictor(search_roots=self.search_roots)
        self._load_geometry_space()

        self.local_re_log_tol = 0.18
        self.local_mach_tol = 0.10
        self.local_pool_min = 80
        self.init_pool_size = 24
        self.target_weights = {"LDMax": 1.0, "ClMax": 1.0, "CdMin": 1.15}
        self.flow_seed_weights = {"re_log": 2.50, "mach": 5.00}
        self.disagreement_penalty = 0.25
        self.geometry_penalty_weight = 10.0
        self.cd_rel_std_penalty = 0.35
        self.max_cd_rel_std = 0.60
        self.max_ldmax_std_norm = 0.18
        self.max_clmax_std_norm = 0.18

    def _load_geometry_space(self) -> None:
        X_profile, X_scalar, X_flow, y_targets, meta = data.build_or_load_cached_dataset()
        split_manifest = pd.read_csv(find_artifact("aeroml_xfoil_split_manifest.csv", self.search_roots))
        train_idx, _, _ = data.materialize_indices(meta, split_manifest)

        self.meta = meta
        self.train_meta = meta.iloc[train_idx].reset_index(drop=True)
        self.y_train_raw = y_targets[train_idx]

        self.n_stations = features.N_STATIONS
        self.x_grid = features.cosine_spacing(self.n_stations)
        self.rng = np.random.default_rng(data.RANDOM_STATE)

        thickness_train = X_profile[train_idx, : self.n_stations]
        camber_train = X_profile[train_idx, self.n_stations : 2 * self.n_stations]
        shape_train = np.concatenate([thickness_train, camber_train], axis=1).astype(np.float32)

        self.pca = PCA(n_components=12, random_state=data.RANDOM_STATE)
        self.z_train = self.pca.fit_transform(shape_train)
        self.latent_low = np.quantile(self.z_train, 0.01, axis=0)
        self.latent_high = np.quantile(self.z_train, 0.99, axis=0)
        self.latent_span = np.maximum(self.latent_high - self.latent_low, 1e-6)
        self.latent_bounds = list(zip(self.latent_low, self.latent_high))

        max_thickness_train = thickness_train.max(axis=1)
        max_camber_train = np.abs(camber_train).max(axis=1)
        te_thickness_train = thickness_train[:, -1]
        self.geom_limits = {
            "max_thickness_min": float(np.quantile(max_thickness_train, 0.001)),
            "max_thickness_max": float(np.quantile(max_thickness_train, 0.999)),
            "max_camber_max": float(np.quantile(max_camber_train, 0.999)),
            "te_thickness_min": float(np.quantile(te_thickness_train, 0.001)),
            "te_thickness_max": float(np.quantile(te_thickness_train, 0.999)),
        }

        self.ld_scale = float(np.std(self.y_train_raw[:, 0]))
        self.cl_scale = float(np.std(self.y_train_raw[:, 1]))
        self.cd_log_scale = float(np.std(np.log(self.y_train_raw[:, 2])))

    def shape_from_latent(self, z: np.ndarray) -> dict[str, Any]:
        clipped = np.clip(np.asarray(z, dtype=np.float64), self.latent_low, self.latent_high)
        shape = self.pca.inverse_transform(clipped.reshape(1, -1))[0]
        thickness = shape[: self.n_stations]
        camber = shape[self.n_stations :]
        y_upper = camber + 0.5 * thickness
        y_lower = camber - 0.5 * thickness
        dyu_dx = np.gradient(y_upper, self.x_grid)
        dyl_dx = np.gradient(y_lower, self.x_grid)
        return {
            "latent": clipped,
            "thickness": thickness,
            "camber": camber,
            "y_upper": y_upper,
            "y_lower": y_lower,
            "dyu_dx": dyu_dx,
            "dyl_dx": dyl_dx,
        }

    def scalar_from_surfaces(self, thickness, camber, y_upper, y_lower, dyu_dx, dyl_dx):
        curv_upper = np.gradient(dyu_dx, self.x_grid) / np.maximum((1.0 + dyu_dx**2) ** 1.5, 1e-6)
        curv_lower = np.gradient(dyl_dx, self.x_grid) / np.maximum((1.0 + dyl_dx**2) ** 1.5, 1e-6)
        max_t_idx = int(np.argmax(thickness))
        max_c_idx = int(np.argmax(np.abs(camber)))
        return np.array(
            [
                thickness[max_t_idx],
                self.x_grid[max_t_idx],
                camber[max_c_idx],
                self.x_grid[max_c_idx],
                np.max(camber),
                np.min(camber),
                features.estimate_le_radius(self.x_grid, thickness),
                thickness[-1],
                np.degrees(np.arctan(dyu_dx[-1]) - np.arctan(dyl_dx[-1])),
                _trapz(thickness, self.x_grid),
                np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_upper) ** 2)),
                np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_lower) ** 2)),
                np.mean(np.abs(curv_upper)),
                np.mean(np.abs(curv_lower)),
                np.max(np.abs(curv_upper)),
                np.max(np.abs(curv_lower)),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Batched versions of shape_from_latent / scalar_from_surfaces /
    # geometry_penalty / objective. These are the actual speed fix: the
    # original reverse search called the forward ensemble on a batch of 1,
    # one restart and one finite-difference perturbation at a time, which
    # meant thousands of separate Python/TF calls per search. Everything
    # below does the identical math, but on a population of latent vectors
    # at once, so a full optimization step is a handful of batched calls
    # instead of thousands of single-sample ones.
    # ------------------------------------------------------------------ #

    def shape_from_latent_batch(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        clipped = np.clip(Z, self.latent_low, self.latent_high)
        shape = self.pca.inverse_transform(clipped)
        thickness = shape[:, : self.n_stations]
        camber = shape[:, self.n_stations :]
        y_upper = camber + 0.5 * thickness
        y_lower = camber - 0.5 * thickness
        dyu_dx = np.gradient(y_upper, self.x_grid, axis=1)
        dyl_dx = np.gradient(y_lower, self.x_grid, axis=1)
        return {
            "latent": clipped,
            "thickness": thickness,
            "camber": camber,
            "y_upper": y_upper,
            "y_lower": y_lower,
            "dyu_dx": dyu_dx,
            "dyl_dx": dyl_dx,
        }

    def _estimate_le_radius_batch(self, thickness: np.ndarray) -> np.ndarray:
        nose_x = self.x_grid[1:6]
        nose_t = thickness[:, 1:6]
        radius = 0.5 * (nose_t**2) / np.clip(nose_x, 1e-6, None)
        return np.median(radius, axis=1)

    def scalar_from_surfaces_batch(self, thickness, camber, y_upper, y_lower, dyu_dx, dyl_dx) -> np.ndarray:
        curv_upper = np.gradient(dyu_dx, self.x_grid, axis=1) / np.maximum((1.0 + dyu_dx**2) ** 1.5, 1e-6)
        curv_lower = np.gradient(dyl_dx, self.x_grid, axis=1) / np.maximum((1.0 + dyl_dx**2) ** 1.5, 1e-6)
        max_t_idx = np.argmax(thickness, axis=1)
        max_c_idx = np.argmax(np.abs(camber), axis=1)
        rows = np.arange(thickness.shape[0])

        te_angle = np.degrees(np.arctan(dyu_dx[:, -1]) - np.arctan(dyl_dx[:, -1]))
        arc_upper = np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_upper, axis=1) ** 2), axis=1)
        arc_lower = np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_lower, axis=1) ** 2), axis=1)

        return np.stack(
            [
                thickness[rows, max_t_idx],
                self.x_grid[max_t_idx],
                camber[rows, max_c_idx],
                self.x_grid[max_c_idx],
                camber.max(axis=1),
                camber.min(axis=1),
                self._estimate_le_radius_batch(thickness),
                thickness[:, -1],
                te_angle,
                _trapz(thickness, self.x_grid, axis=1),
                arc_upper,
                arc_lower,
                np.mean(np.abs(curv_upper), axis=1),
                np.mean(np.abs(curv_lower), axis=1),
                np.max(np.abs(curv_upper), axis=1),
                np.max(np.abs(curv_lower), axis=1),
            ],
            axis=1,
        ).astype(np.float32)

    def geometry_penalty_batch(self, thickness: np.ndarray, camber: np.ndarray) -> np.ndarray:
        max_thick = thickness.max(axis=1)
        min_thick = thickness.min(axis=1)
        max_camber_abs = np.abs(camber).max(axis=1)
        te_thick = thickness[:, -1]
        w = self.geometry_penalty_weight

        penalty = np.zeros(thickness.shape[0], dtype=np.float64)
        penalty += w * np.where(min_thick < -1e-4, np.abs(min_thick), 0.0)
        penalty += w * np.clip(self.geom_limits["max_thickness_min"] - max_thick, 0.0, None)
        penalty += w * np.clip(max_thick - self.geom_limits["max_thickness_max"], 0.0, None)
        penalty += w * np.clip(max_camber_abs - self.geom_limits["max_camber_max"], 0.0, None)
        penalty += w * np.clip(self.geom_limits["te_thickness_min"] - te_thick, 0.0, None)
        penalty += w * np.clip(te_thick - self.geom_limits["te_thickness_max"], 0.0, None)
        return penalty

    def objective_batch(self, Z: np.ndarray, target: dict[str, float], flow: dict[str, float]) -> np.ndarray:
        surf = self.shape_from_latent_batch(Z)
        scalar = self.scalar_from_surfaces_batch(
            surf["thickness"], surf["camber"], surf["y_upper"], surf["y_lower"], surf["dyu_dx"], surf["dyl_dx"]
        )
        profile = np.concatenate(
            [surf["thickness"], surf["camber"], surf["dyu_dx"], surf["dyl_dx"]], axis=1
        ).astype(np.float32)

        mean_pred, std_pred = self.forward._predict_batch(profile, scalar, flow["Re"], flow["Mach"])
        ld_pred, cl_pred, cd_pred = mean_pred[:, 0], mean_pred[:, 1], mean_pred[:, 2]
        ld_std, cl_std, cd_std = std_pred[:, 0], std_pred[:, 1], std_pred[:, 2]
        cd_rel_std = cd_std / np.maximum(cd_pred, 1e-6)

        ld_term = self.target_weights["LDMax"] * ((ld_pred - target["LDMax"]) / max(self.ld_scale, 1e-6)) ** 2
        cl_term = self.target_weights["ClMax"] * ((cl_pred - target["ClMax"]) / max(self.cl_scale, 1e-6)) ** 2
        cd_term = self.target_weights["CdMin"] * (
            (np.log(np.clip(cd_pred, 1e-8, None)) - np.log(target["CdMin"])) / max(self.cd_log_scale, 1e-6)
        ) ** 2
        disagreement = self.disagreement_penalty * (
            (ld_std / max(self.ld_scale, 1e-6))
            + (cl_std / max(self.cl_scale, 1e-6))
            + (cd_std / max(target["CdMin"], 1e-6))
        )
        uncertainty_penalty = self.cd_rel_std_penalty * np.clip(cd_rel_std - 0.25, 0.0, None)
        geom_penalty = self.geometry_penalty_batch(surf["thickness"], surf["camber"])

        return ld_term + cl_term + cd_term + disagreement + uncertainty_penalty + geom_penalty

    def _batched_gradient(self, Z: np.ndarray, target: dict[str, float], flow: dict[str, float], fd_eps: float = 0.02):
        """Forward-difference gradient of objective_batch w.r.t. every latent vector in
        Z, computed with exactly two batched objective_batch calls regardless of how
        many restarts or dimensions there are: one for the base points, one covering
        every (restart, dimension) perturbation stacked into a single batch. This is
        the same finite-difference approximation scipy's L-BFGS-B was already using
        under the hood when no jac was supplied -- the only thing that changes is that
        it's now one big batched model call instead of (dims+1) separate ones."""
        P, D = Z.shape
        base_obj = self.objective_batch(Z, target, flow)

        step = fd_eps * self.latent_span  # (D,)
        Z_pert = np.repeat(Z[:, None, :], D, axis=1)  # (P, D, D)
        diag_idx = np.arange(D)
        Z_pert[:, diag_idx, diag_idx] += step[diag_idx]
        Z_pert_flat = np.clip(Z_pert.reshape(P * D, D), self.latent_low, self.latent_high)

        pert_obj = self.objective_batch(Z_pert_flat, target, flow).reshape(P, D)
        grad = (pert_obj - base_obj[:, None]) / step[None, :]
        return grad, base_obj

    def _batched_latent_search(
        self,
        Z0: np.ndarray,
        target: dict[str, float],
        flow: dict[str, float],
        maxiter: int = 50,
        lr_frac: float = 0.04,
        fd_eps: float = 0.02,
        patience: int = 6,
        tol: float = 1e-5,
        grad_clip: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs every restart in Z0 as one batched Adam-style optimization instead of
        looping scipy.optimize.minimize per restart. lr is scaled per latent dimension
        by that dimension's span, since PCA components can have very different scales
        and we don't have a Hessian estimate (the way L-BFGS-B implicitly builds one)
        to normalize against automatically.

        Adam steps aren't monotonic -- a restart can overshoot and land somewhere worse
        than it started, especially this close to the geometry-penalty boundary. So
        instead of trusting the final iterate, we track the best point *seen* for each
        restart independently and return that. A restart can therefore never end up
        reported worse than its own starting point."""
        Z = np.clip(np.asarray(Z0, dtype=np.float64), self.latent_low, self.latent_high)
        m = np.zeros_like(Z)
        v = np.zeros_like(Z)
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        lr_vec = lr_frac * self.latent_span

        best_obj = self.objective_batch(Z, target, flow)
        best_Z = Z.copy()

        best_mean_obj = float(np.mean(best_obj))
        stall_count = 0

        for t in range(1, maxiter + 1):
            grad, _ = self._batched_gradient(Z, target, flow, fd_eps=fd_eps)

            # Clip per-restart gradient norm (in span-normalized units) so a single bad
            # finite-difference estimate near a penalty boundary can't blow up the step.
            grad_units = grad / self.latent_span[None, :]
            norm = np.linalg.norm(grad_units, axis=1, keepdims=True)
            scale = np.minimum(1.0, grad_clip / np.maximum(norm, 1e-8))
            grad = grad * scale

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            Z = Z - lr_vec[None, :] * m_hat / (np.sqrt(v_hat) + adam_eps)
            Z = np.clip(Z, self.latent_low, self.latent_high)

            cur_obj = self.objective_batch(Z, target, flow)
            improved = cur_obj < best_obj
            best_obj = np.where(improved, cur_obj, best_obj)
            best_Z = np.where(improved[:, None], Z, best_Z)

            mean_obj = float(np.mean(cur_obj))
            if best_mean_obj - mean_obj < tol:
                stall_count += 1
                if stall_count >= patience:
                    break
            else:
                stall_count = 0
            best_mean_obj = min(best_mean_obj, mean_obj)

        return best_Z, best_obj

    def predict_candidate(self, z: np.ndarray, flow: dict[str, float]) -> dict[str, Any]:
        surf = self.shape_from_latent(z)
        profile = np.concatenate([surf["thickness"], surf["camber"], surf["dyu_dx"], surf["dyl_dx"]], axis=0).astype(np.float32)
        scalar = self.scalar_from_surfaces(
            surf["thickness"], surf["camber"], surf["y_upper"], surf["y_lower"], surf["dyu_dx"], surf["dyl_dx"]
        )

        result = self.forward._predict_inputs(profile, scalar, flow["Re"], flow["Mach"])
        predictions = result["predictions"]
        uncertainty = result["uncertainty"]
        return {
            **surf,
            "predictions": predictions,
            "uncertainty": uncertainty,
            "profile": profile,
            "scalar": scalar,
            "passes_uncertainty": bool(
                (uncertainty["CdMin_rel_std"] <= self.max_cd_rel_std)
                and (uncertainty["LDMax_std"] / max(self.ld_scale, 1e-6) <= self.max_ldmax_std_norm)
                and (uncertainty["ClMax_std"] / max(self.cl_scale, 1e-6) <= self.max_clmax_std_norm)
            ),
        }

    def geometry_penalty(self, candidate: dict[str, Any]) -> float:
        thickness = candidate["thickness"]
        camber = candidate["camber"]
        penalty = 0.0
        if thickness.min() < -1e-4:
            penalty += self.geometry_penalty_weight * abs(float(thickness.min()))
        if thickness.max() < self.geom_limits["max_thickness_min"]:
            penalty += self.geometry_penalty_weight * (self.geom_limits["max_thickness_min"] - float(thickness.max()))
        if thickness.max() > self.geom_limits["max_thickness_max"]:
            penalty += self.geometry_penalty_weight * (float(thickness.max()) - self.geom_limits["max_thickness_max"])
        if np.abs(camber).max() > self.geom_limits["max_camber_max"]:
            penalty += self.geometry_penalty_weight * (float(np.abs(camber).max()) - self.geom_limits["max_camber_max"])
        if thickness[-1] < self.geom_limits["te_thickness_min"]:
            penalty += self.geometry_penalty_weight * (self.geom_limits["te_thickness_min"] - float(thickness[-1]))
        if thickness[-1] > self.geom_limits["te_thickness_max"]:
            penalty += self.geometry_penalty_weight * (float(thickness[-1]) - self.geom_limits["te_thickness_max"])
        return float(penalty)

    def objective(self, z: np.ndarray, target: dict[str, float], flow: dict[str, float]) -> float:
        candidate = self.predict_candidate(z, flow)
        preds = candidate["predictions"]
        unc = candidate["uncertainty"]

        ld_term = self.target_weights["LDMax"] * ((preds["LDMax"] - target["LDMax"]) / max(self.ld_scale, 1e-6)) ** 2
        cl_term = self.target_weights["ClMax"] * ((preds["ClMax"] - target["ClMax"]) / max(self.cl_scale, 1e-6)) ** 2
        cd_term = self.target_weights["CdMin"] * (
            (np.log(max(preds["CdMin"], 1e-8)) - np.log(target["CdMin"])) / max(self.cd_log_scale, 1e-6)
        ) ** 2
        disagreement = self.disagreement_penalty * (
            (unc["LDMax_std"] / max(self.ld_scale, 1e-6))
            + (unc["ClMax_std"] / max(self.cl_scale, 1e-6))
            + (unc["CdMin_std"] / max(target["CdMin"], 1e-6))
        )
        uncertainty_penalty = self.cd_rel_std_penalty * max(unc["CdMin_rel_std"] - 0.25, 0.0)
        return float(ld_term + cl_term + cd_term + disagreement + uncertainty_penalty + self.geometry_penalty(candidate))

    def flow_distance_frame(self, frame: pd.DataFrame, flow: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "re_log_abs": np.abs(np.log10(frame["Re"]) - np.log10(flow["Re"])),
                "mach_abs": np.abs(frame["Mach"] - flow["Mach"]),
            },
            index=frame.index,
        )

    def local_flow_pool(self, flow: dict[str, float]) -> pd.DataFrame:
        flow_dist = self.flow_distance_frame(self.train_meta, flow)
        local_mask = (
            (flow_dist["re_log_abs"] <= self.local_re_log_tol)
            & (flow_dist["mach_abs"] <= self.local_mach_tol)
        )
        local = self.train_meta.loc[local_mask].copy()
        local_dist = flow_dist.loc[local_mask].copy()
        if len(local) < self.local_pool_min:
            ranked_idx = (
                flow_dist.assign(
                    flow_rank=self.flow_seed_weights["re_log"] * flow_dist["re_log_abs"]
                    + self.flow_seed_weights["mach"] * flow_dist["mach_abs"]
                )
                .sort_values("flow_rank")
                .head(self.local_pool_min)
                .index
            )
            local = self.train_meta.loc[ranked_idx].copy()
            local_dist = flow_dist.loc[ranked_idx].copy()

        local["re_log_abs"] = local_dist["re_log_abs"]
        local["mach_abs"] = local_dist["mach_abs"]
        local["flow_rank"] = (
            self.flow_seed_weights["re_log"] * local["re_log_abs"]
            + self.flow_seed_weights["mach"] * local["mach_abs"]
        )
        return local.sort_values(["flow_rank", "re_log_abs", "mach_abs"]).copy()

    def feasibility_summary(self, local_pool: pd.DataFrame, target: dict[str, float]) -> dict[str, Any]:
        q05 = local_pool[["LDMax", "ClMax", "CdMin"]].quantile(0.05)
        q95 = local_pool[["LDMax", "ClMax", "CdMin"]].quantile(0.95)
        min_v = local_pool[["LDMax", "ClMax", "CdMin"]].min()
        max_v = local_pool[["LDMax", "ClMax", "CdMin"]].max()
        return {
            "count": int(len(local_pool)),
            "local_re_range": [float(local_pool["Re"].min()), float(local_pool["Re"].max())],
            "local_mach_values": sorted(float(v) for v in local_pool["Mach"].unique()),
            "target_within_local_5_95": {
                "LDMax": bool(q05["LDMax"] <= target["LDMax"] <= q95["LDMax"]),
                "ClMax": bool(q05["ClMax"] <= target["ClMax"] <= q95["ClMax"]),
                "CdMin": bool(q05["CdMin"] <= target["CdMin"] <= q95["CdMin"]),
            },
            "target_within_local_min_max": {
                "LDMax": bool(min_v["LDMax"] <= target["LDMax"] <= max_v["LDMax"]),
                "ClMax": bool(min_v["ClMax"] <= target["ClMax"] <= max_v["ClMax"]),
                "CdMin": bool(min_v["CdMin"] <= target["CdMin"] <= max_v["CdMin"]),
            },
        }

    def build_seed_pool(self, local_pool: pd.DataFrame, target: dict[str, float], flow: dict[str, float], init_pool_size: int = 24) -> pd.DataFrame:
        score = (
            self.target_weights["LDMax"] * np.abs((local_pool["LDMax"] - target["LDMax"]) / max(self.ld_scale, 1e-6))
            + self.target_weights["ClMax"] * np.abs((local_pool["ClMax"] - target["ClMax"]) / max(self.cl_scale, 1e-6))
            + self.target_weights["CdMin"] * np.abs((np.log(local_pool["CdMin"]) - np.log(target["CdMin"])) / max(self.cd_log_scale, 1e-6))
            + self.flow_seed_weights["re_log"] * local_pool["re_log_abs"]
            + self.flow_seed_weights["mach"] * local_pool["mach_abs"]
        )
        return local_pool.assign(init_score=score).sort_values(["init_score", "flow_rank"]).head(init_pool_size).copy()

    def _summarize_candidate(self, label: str, candidate: dict[str, Any], objective_value: float) -> dict[str, Any]:
        preds = candidate["predictions"]
        unc = candidate["uncertainty"]
        return {
            "label": label,
            "objective": float(objective_value),
            "LDMax_pred": float(preds["LDMax"]),
            "ClMax_pred": float(preds["ClMax"]),
            "CdMin_pred": float(preds["CdMin"]),
            "LDMax_std": float(unc["LDMax_std"]),
            "ClMax_std": float(unc["ClMax_std"]),
            "CdMin_std": float(unc["CdMin_std"]),
            "CdMin_rel_std": float(unc["CdMin_rel_std"]),
            "passes_uncertainty": bool(candidate["passes_uncertainty"]),
            "geometry": {
                "x": self.x_grid.copy(),
                "y_upper": candidate["y_upper"].copy(),
                "y_lower": candidate["y_lower"].copy(),
                "thickness": candidate["thickness"].copy(),
                "camber": candidate["camber"].copy(),
            },
            "latent": candidate["latent"].copy(),
        }

    def run_reverse_search(self, target: dict[str, float], flow: dict[str, float], n_restarts: int = 8, opt_maxiter: int = 35) -> dict[str, Any]:
        local_pool = self.local_flow_pool(flow)
        feasibility = self.feasibility_summary(local_pool, target)
        init_pool = self.build_seed_pool(local_pool, target, flow)

        seed_items = []
        elite_count = min(4, len(init_pool))
        elite_pool = init_pool.head(elite_count)
        elite_indices = elite_pool.index.to_numpy()
        elite_z = self.z_train[elite_indices]
        for rank, (row, z0) in enumerate(zip(elite_pool.itertuples(index=False), elite_z), start=1):
            seed_items.append((f"elite_{rank}", z0))
            if len(seed_items) < n_restarts:
                z_jitter = np.clip(z0 + self.rng.normal(scale=0.06 * self.latent_span, size=z0.shape), self.latent_low, self.latent_high)
                seed_items.append((f"jitter_{rank}", z_jitter))

        while len(seed_items) < n_restarts:
            row = local_pool.sample(n=1, random_state=int(self.rng.integers(0, 1_000_000))).iloc[0]
            seed_items.append((f"local_random_{len(seed_items)+1}", self.z_train[int(row.name)]))

        labels = [label for label, _ in seed_items[:n_restarts]]
        Z0 = np.stack([z0 for _, z0 in seed_items[:n_restarts]], axis=0)
        Z_final, final_obj = self._batched_latent_search(Z0, target, flow, maxiter=opt_maxiter)

        raw_results = []
        for label, z_opt, obj_val in zip(labels, Z_final, final_obj):
            candidate = self.predict_candidate(z_opt, flow)
            summary = self._summarize_candidate(label, candidate, float(obj_val))
            summary["success"] = True
            summary["message"] = "batched-adam"
            summary["target_gap"] = float(
                abs(summary["LDMax_pred"] - target["LDMax"]) / max(target["LDMax"], 1e-6)
                + abs(summary["ClMax_pred"] - target["ClMax"]) / max(target["ClMax"], 1e-6)
                + abs(summary["CdMin_pred"] - target["CdMin"]) / max(target["CdMin"], 1e-6)
            )
            raw_results.append(summary)

        raw_results.sort(key=lambda item: (not item["passes_uncertainty"], item["target_gap"], item["objective"], item["CdMin_rel_std"]))
        return {
            "feasibility": feasibility,
            "local_pool": init_pool,
            "candidates": raw_results,
        }

    def refine_candidate(
        self,
        candidate: dict[str, Any],
        target: dict[str, float],
        flow: dict[str, float],
        refinement_restarts: int = 6,
        opt_maxiter: int = 60,
    ) -> list[dict[str, Any]]:
        base_latent = np.asarray(candidate["latent"], dtype=np.float64)
        jitter_scales = [0.00, 0.015, 0.030, 0.050, 0.075, 0.100][:refinement_restarts]
        seed_latents = []
        for i, scale in enumerate(jitter_scales, start=1):
            if scale == 0.0:
                seed_latents.append((f"base_{i}", np.clip(base_latent, self.latent_low, self.latent_high)))
            else:
                z0 = np.clip(base_latent + self.rng.normal(scale=scale * self.latent_span, size=base_latent.shape), self.latent_low, self.latent_high)
                seed_latents.append((f"jitter_{i}", z0))

        labels = [label for label, _ in seed_latents]
        Z0 = np.stack([z0 for _, z0 in seed_latents], axis=0)
        Z_final, final_obj = self._batched_latent_search(Z0, target, flow, maxiter=opt_maxiter)

        results = []
        for label, z_opt, obj_val in zip(labels, Z_final, final_obj):
            refined = self.predict_candidate(z_opt, flow)
            summary = self._summarize_candidate(label, refined, float(obj_val))
            summary["success"] = True
            summary["message"] = "batched-adam"
            summary["target_gap"] = float(
                abs(summary["LDMax_pred"] - target["LDMax"]) / max(target["LDMax"], 1e-6)
                + abs(summary["ClMax_pred"] - target["ClMax"]) / max(target["ClMax"], 1e-6)
                + abs(summary["CdMin_pred"] - target["CdMin"]) / max(target["CdMin"], 1e-6)
            )
            results.append(summary)

        results.sort(key=lambda item: (not item["passes_uncertainty"], item["target_gap"], item["objective"], item["CdMin_rel_std"]))
        return results
