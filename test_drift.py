import sys
import numpy as np
import pandas as pd
from aeroml_forward_v3_runtime import ForwardV3Predictor
import aeroml_notebook_common as common

def run_test():
    print("Loading predictor...")
    predictor = ForwardV3Predictor()
    
    # Check what artifacts were loaded
    print(f"Chosen variant: {predictor.chosen_variant}")
    print(f"Profile scaler mean sum: {predictor.profile_scaler.mean_.sum():.4f}")
    
    # Load dataset to get exact cached features for these airfoils
    X_profile, X_scalar, X_flow, y_targets, meta = common.build_or_load_cached_dataset()
    
    cases = [
        {"name": "sd7062", "Re": 2e6, "Mach": 0.10, "exp_ld": 142.05, "exp_cl": 1.8204, "exp_cd": 0.08815},
        {"name": "sg6041", "Re": 2e6, "Mach": 0.10, "exp_ld": 106.67, "exp_cl": 1.6226, "exp_cd": 0.10757},
        {"name": "clarkyh", "Re": 4e6, "Mach": 0.10, "exp_ld": 117.73, "exp_cl": 1.6671, "exp_cd": 0.10019},
    ]
    
    for case in cases:
        print(f"\n--- Testing base case: {case['name']} Re={case['Re']} Mach={case['Mach']} ---")
        
        # 1. Find it in meta
        mask = (meta["example_name"] == case["name"]) & (np.isclose(meta["Re"], case["Re"])) & (np.isclose(meta["Mach"], case["Mach"]))
        idxs = np.flatnonzero(mask)
        
        if len(idxs) == 0:
            print(f"Could not find exact match in meta for {case['name']}")
            print("Trying just by name...")
            name_mask = meta["example_name"] == case["name"]
            idxs = np.flatnonzero(name_mask)
            if len(idxs) > 0:
                print(f"Found it at different Re/Mach: Re={meta['Re'].iloc[idxs[0]]}, Mach={meta['Mach'].iloc[idxs[0]]}")
                continue
            else:
                print("Could not find at all.")
                continue
                
        idx = idxs[0]
        y_true = y_targets[idx]
        print(f"Meta expected values: LDMax={y_true[0]:.4f}, ClMax={y_true[1]:.4f}, CdMin={y_true[2]:.4f}")
        print(f"Manual expected (from prompt): LDMax={case['exp_ld']:.4f}, ClMax={case['exp_cl']:.4f}, CdMin={case['exp_cd']:.4f}")
        
        # 2. Run prediction using cached features
        prof_feat = X_profile[idx]
        scal_feat = X_scalar[idx]
        
        res_cached = predictor._predict_inputs(prof_feat, scal_feat, case['Re'], case['Mach'])
        p_c = res_cached["predictions"]
        print(f"Direct from Dataset Features -> LDMax={p_c['LDMax']:.4f}, ClMax={p_c['ClMax']:.4f}, CdMin={p_c['CdMin']:.4f}")
        
        # 3. Simulate getting it from dat file
        # We need the dat file path
        # Try to find dat file in Data folder
        dat_path = common.DATA_DIR / f"{case['name']}.dat"
        if dat_path.exists():
            geom = common.geometry_representation(dat_path)
            prof_dat = geom["profile"]
            scal_dat = geom["scalar"]
            
            # Check diff between geom and cached
            prof_diff = np.abs(prof_feat - prof_dat).max()
            scal_diff = np.abs(scal_feat - scal_dat).max()
            print(f"Max feature diff vs cache: Profile={prof_diff:.2e}, Scalar={scal_diff:.2e}")
            
            res_dat = predictor._predict_inputs(prof_dat, scal_dat, case['Re'], case['Mach'])
            p_d = res_dat["predictions"]
            print(f"From raw .dat processing   -> LDMax={p_d['LDMax']:.4f}, ClMax={p_d['ClMax']:.4f}, CdMin={p_d['CdMin']:.4f}")
        else:
            print(f"Dat file not found: {dat_path}")

if __name__ == '__main__':
    run_test()
