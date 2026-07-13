# -*- coding: utf-8 -*-
"""
@file api_bridge.py
@description API Bridge CLI utility to execute predictions and optimizations returning JSON output
@module scripts
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['absl_minloglevel'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Bootstrap local src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeroml.forward import ForwardV3Predictor
from aeroml.reverse import ReverseV3Designer
import aeroml.features as features


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to support Numpy arrays and float32 types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def handle_predict(args):
    predictor = ForwardV3Predictor()
    dat_path = Path(args.file)
    
    geom = features.geometry_representation(dat_path)
    if geom is None:
        print(json.dumps({"error": f"Could not parse valid coordinates from {args.file}"}))
        sys.exit(1)
        
    res = predictor._predict_inputs(geom["profile"], geom["scalar"], args.re, args.mach)
    
    # Reconstruct surface coordinates for plotting
    coords = features.read_dat_file(dat_path)
    if coords is None:
        print(json.dumps({"error": "Failed to read coordinates"}))
        sys.exit(1)
    coords = features.normalize_coords(coords)
    upper, lower = features.split_upper_lower(coords)
    upper = features.prepare_surface_for_interp(upper)
    lower = features.prepare_surface_for_interp(lower)
    x_grid = features.cosine_spacing(features.N_STATIONS)
    y_upper = np.interp(x_grid, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_grid, lower[:, 0], lower[:, 1])
    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)

    # Pack up all predictions, uncertainties, and geometry coordinate arrays
    payload = {
        "predictions": res["predictions"],
        "uncertainty": res["uncertainty"],
        "geometry": {
            "fingerprint": geom["fingerprint"],
            "x": x_grid,
            "y_upper": y_upper,
            "y_lower": y_lower,
            "thickness": thickness,
            "camber": camber
        },
        "mach_warning": {
            "extrapolated": features.mach_extrapolation_distance(args.mach) > features.MACH_EXTRAPOLATION_THRESHOLD,
            "nearest_known_mach": min(features.KNOWN_MACH_VALUES, key=lambda m: abs(m - args.mach)),
            "distance": features.mach_extrapolation_distance(args.mach),
        }
    }
    
    print(json.dumps(payload, cls=NumpyEncoder))


def handle_optimize(args):
    designer = ReverseV3Designer()
    target = {"LDMax": args.ldmax, "ClMax": args.clmax, "CdMin": args.cdmin}
    flow = {"Re": args.re, "Mach": args.mach}
    
    results = designer.run_reverse_search(
        target=target,
        flow=flow,
        n_restarts=args.restarts,
        opt_maxiter=args.maxiter
    )
    
    # Extract structural candidate data
    candidates_payload = []
    for c in results["candidates"]:
        candidates_payload.append({
            "label": c["label"],
            "objective": float(c["objective"]),
            "passes_uncertainty": bool(c["passes_uncertainty"]),
            "predictions": {
                "LDMax": float(c["LDMax_pred"]),
                "ClMax": float(c["ClMax_pred"]),
                "CdMin": float(c["CdMin_pred"])
            },
            "uncertainty": {
                "LDMax_std": float(c["LDMax_std"]),
                "ClMax_std": float(c["ClMax_std"]),
                "CdMin_std": float(c["CdMin_std"]),
                "CdMin_rel_std": float(c["CdMin_rel_std"])
            },
            "geometry": {
                "x": c["geometry"]["x"],
                "y_upper": c["geometry"]["y_upper"],
                "y_lower": c["geometry"]["y_lower"],
                "thickness": c["geometry"]["thickness"],
                "camber": c["geometry"]["camber"]
            }
        })
        
    payload = {
        "feasibility": results["feasibility"],
        "candidates": candidates_payload,
        "mach_warning": {
            "extrapolated": features.mach_extrapolation_distance(args.mach) > features.MACH_EXTRAPOLATION_THRESHOLD,
            "nearest_known_mach": min(features.KNOWN_MACH_VALUES, key=lambda m: abs(m - args.mach)),
            "distance": features.mach_extrapolation_distance(args.mach),
        }
    }
    
    print(json.dumps(payload, cls=NumpyEncoder))


def main():
    parser = argparse.ArgumentParser(description="AeroML Backend API Process Bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Predict Subcommand
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--file", type=str, required=True, help="Path to dat airfoil file")
    predict_parser.add_argument("--re", type=float, required=True, help="Reynolds number")
    predict_parser.add_argument("--mach", type=float, required=True, help="Mach number")
    
    # Optimize Subcommand
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--ldmax", type=float, required=True)
    optimize_parser.add_argument("--clmax", type=float, required=True)
    optimize_parser.add_argument("--cdmin", type=float, required=True)
    optimize_parser.add_argument("--re", type=float, required=True)
    optimize_parser.add_argument("--mach", type=float, required=True)
    optimize_parser.add_argument("--restarts", type=int, default=8)
    optimize_parser.add_argument("--maxiter", type=int, default=35)
    
    args = parser.parse_args()
    
    if args.command == "predict":
        handle_predict(args)
    elif args.command == "optimize":
        handle_optimize(args)


if __name__ == "__main__":
    main()
