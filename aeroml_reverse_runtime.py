from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA

import aeroml_notebook_common as common
from aeroml_forward_v3_runtime import ForwardV3Predictor, find_artifact

common.add_tf_helpers(globals())


class ReverseV3Designer:
    def __init__(self, search_roots: list[Path] | None = None):
        self.search_roots = search_roots or [common.WORK_DIR, Path.cwd(), Path("/kaggle/input")]
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
        X_profile, X_scalar, X_flow, y_targets, meta = common.build_or_load_cached_dataset()
        split_manifest = pd.read_csv(find_artifact("aeroml_xfoil_split_manifest.csv", self.search_roots))
        train_idx, _, _ = common.materialize_indices(meta, split_manifest)

        self.meta = meta
        self.train_meta = meta.iloc[train_idx].reset_index(drop=True)
        self.y_train_raw = y_targets[train_idx]

        self.n_stations = common.N_STATIONS
        self.x_grid = common.cosine_spacing(self.n_stations)
        self.rng = np.random.default_rng(common.RANDOM_STATE)

        thickness_train = X_profile[train_idx, : self.n_stations]
        camber_train = X_profile[train_idx, self.n_stations : 2 * self.n_stations]
        shape_train = np.concatenate([thickness_train, camber_train], axis=1).astype(np.float32)

        self.pca = PCA(n_components=12, random_state=common.RANDOM_STATE)
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
                common.estimate_le_radius(self.x_grid, thickness),
                thickness[-1],
                np.degrees(np.arctan(dyu_dx[-1]) - np.arctan(dyl_dx[-1])),
                np.trapz(thickness, self.x_grid),
                np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_upper) ** 2)),
                np.sum(np.sqrt(np.diff(self.x_grid) ** 2 + np.diff(y_lower) ** 2)),
                np.mean(np.abs(curv_upper)),
                np.mean(np.abs(curv_lower)),
                np.max(np.abs(curv_upper)),
                np.max(np.abs(curv_lower)),
            ],
            dtype=np.float32,
        )

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
        target_gap = None
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

        raw_results = []
        for label, z0 in seed_items[:n_restarts]:
            opt = minimize(self.objective, x0=z0, args=(target, flow), method="L-BFGS-B", bounds=self.latent_bounds, options={"maxiter": opt_maxiter})
            candidate = self.predict_candidate(opt.x, flow)
            summary = self._summarize_candidate(label, candidate, float(opt.fun))
            summary["success"] = bool(opt.success)
            summary["message"] = str(opt.message)
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

        results = []
        for label, z0 in seed_latents:
            opt = minimize(self.objective, x0=z0, args=(target, flow), method="L-BFGS-B", bounds=self.latent_bounds, options={"maxiter": opt_maxiter})
            refined = self.predict_candidate(opt.x, flow)
            summary = self._summarize_candidate(label, refined, float(opt.fun))
            summary["success"] = bool(opt.success)
            summary["message"] = str(opt.message)
            summary["target_gap"] = float(
                abs(summary["LDMax_pred"] - target["LDMax"]) / max(target["LDMax"], 1e-6)
                + abs(summary["ClMax_pred"] - target["ClMax"]) / max(target["ClMax"], 1e-6)
                + abs(summary["CdMin_pred"] - target["CdMin"]) / max(target["CdMin"], 1e-6)
            )
            results.append(summary)

        results.sort(key=lambda item: (not item["passes_uncertainty"], item["target_gap"], item["objective"], item["CdMin_rel_std"]))
        return results
