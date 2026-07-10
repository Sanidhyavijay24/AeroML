# -*- coding: utf-8 -*-
"""
@file run_reverse.py
@description CLI script to run reverse design optimization
@module scripts
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Bootstrap local src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeroml.reverse import ReverseV3Designer


def dat_format_geometry(x, y_upper, y_lower) -> str:
    """Assembles coordinates into standard wrap-around TE -> LE -> TE tab-separated format."""
    x_upper_rev = x[::-1]
    y_upper_rev = y_upper[::-1]
    
    x_all = np.concatenate([x_upper_rev, x[1:]])
    y_all = np.concatenate([y_upper_rev, y_lower[1:]])
    
    lines = []
    for xi, yi in zip(x_all, y_all):
        lines.append(f"{xi:.8f}\t{yi:.8f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="AeroML Reverse Airfoil Design Optimization CLI")
    parser.add_argument("--ldmax", type=float, required=True, help="Target Cl/Cd Max (LDMax)")
    parser.add_argument("--clmax", type=float, required=True, help="Target Cl Max (ClMax)")
    parser.add_argument("--cdmin", type=float, required=True, help="Target Cd Min (CdMin)")
    parser.add_argument("--re", type=float, required=True, help="Operating Reynolds Number (Re)")
    parser.add_argument("--mach", type=float, required=True, help="Operating Mach Number (Mach)")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="airfoil_design.dat",
        help="Output path for the generated .dat coordinates (default: airfoil_design.dat)"
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=8,
        help="Number of initialization seed restarts for the latent search (default: 8)"
    )
    parser.add_argument(
        "--opt-maxiter",
        type=int,
        default=35,
        help="Max optimization iterations per restart (default: 35)"
    )

    args = parser.parse_args()

    target = {"LDMax": args.ldmax, "ClMax": args.clmax, "CdMin": args.cdmin}
    flow = {"Re": args.re, "Mach": args.mach}

    print("Initializing Reverse Designer...")
    designer = ReverseV3Designer()

    print(f"\nRunning reverse design search for:")
    print(f"  Targets: LDMax={args.ldmax}, ClMax={args.clmax}, CdMin={args.cdmin}")
    print(f"  Flow conditions: Re={args.re}, Mach={args.mach}")
    print(f"  Optimizer: n_restarts={args.n_restarts}, maxiter={args.opt_maxiter}")

    results = designer.run_reverse_search(
        target=target,
        flow=flow,
        n_restarts=args.n_restarts,
        opt_maxiter=args.opt_maxiter
    )

    feasibility = results["feasibility"]
    candidates = results["candidates"]

    print("\n--- Feasibility Summary ---")
    print(f"Local flow pool count: {feasibility['count']} airfoils")
    print(f"Local Re range:       [{feasibility['local_re_range'][0]:.1f}, {feasibility['local_re_range'][1]:.1f}]")
    print(f"Local Mach values:     {feasibility['local_mach_values']}")
    print("Targets within local flow pool bounds:")
    for metric, within in feasibility["target_within_local_min_max"].items():
        print(f"  {metric:<6}: {within} (5-95% quantile: {feasibility['target_within_local_5_95'][metric]})")

    if not candidates:
        print("\n[Error] No candidates returned from search.")
        return

    best = candidates[0]
    print("\n--- Best Candidate Found ---")
    print(f"Label:               {best['label']}")
    print(f"Objective Value:     {best['objective']:.6f}")
    print(f"Passes Uncertainty:  {best['passes_uncertainty']}")
    print("Predicted Performance:")
    print(f"  LDMax:             {best['LDMax_pred']:.4f}  (std: {best['LDMax_std']:.4f})")
    print(f"  ClMax:             {best['ClMax_pred']:.4f}  (std: {best['ClMax_std']:.4f})")
    print(f"  CdMin:             {best['CdMin_pred']:.6f}  (std: {best['CdMin_std']:.6f}, rel_std: {best['CdMin_rel_std']*100:.1f}%)")

    # Write .dat file
    geom = best["geometry"]
    dat_content = dat_format_geometry(geom["x"], geom["y_upper"], geom["y_lower"])
    output_path = Path(args.output)
    output_path.write_text(dat_content, encoding="utf-8")
    print(f"\n[Success] Airfoil geometry successfully written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
