# AeroML Project Context

## Project Goal
AeroML is an airfoil ML project with two core capabilities:

1. **Forward prediction**
   - Input: airfoil geometry + flow condition (`Re`, `Mach`)
   - Output: `LDMax`, `ClMax`, `CdMin`

2. **Reverse design**
   - Input: desired `LDMax`, `ClMax`, `CdMin` + flow condition (`Re`, `Mach`)
   - Output: candidate airfoil geometry that should match the requested aerodynamic behavior

The long-term vision is a usable airfoil design workflow where a user can either:
- analyze an existing airfoil, or
- request target aerodynamic performance and get a candidate airfoil back

---

## Dataset

### Raw dataset
- Location used locally: `C:\Users\sanid\Documents\Playground\AeroML_Data`
- Contains:
  - `6351` `.dat` airfoil geometry files
  - `6351` `.pkl` aerodynamic data files
- Total size: about `3.5 GB`
- Total operating-condition rows discovered from the `.pkl` files: about `1.08M`

### Dataset structure
- Each `.dat` file contains airfoil geometry
- Each `.pkl` file contains aerodynamic data for many operating conditions and sources
- Important target values:
  - `LDMax`
  - `ClMax`
  - `CdMin`
- The `.pkl` files also include other information like arrays/polars and metadata

### Important dataset finding
The dataset is **multi-source**. The same `(airfoil, Re, Mach)` can appear with multiple `datasource` values, and these sources can disagree significantly, especially for `CdMin`.

That source disagreement was the main reason the early mixed-source models struggled.

---

## Data Policy We Chose

To get a trustworthy benchmark, the project uses a clean single-source policy:

- **Canonical source:** `XFOIL ncrit=9`
- **Task definition:** generalize to unseen airfoils
- **Inputs:** geometry + `Re` + `Mach`
- **Split policy:** grouped by geometry fingerprint, not random row split
- **Geometry handling:** keep genuinely similar airfoils, but group exact duplicate geometries by fingerprint to avoid leakage
- **Target policy:** predict `log(CdMin)` internally

This was chosen because:
- mixed-source training introduced heavy label noise
- `CdMin` was especially corrupted by cross-source disagreement
- grouped splits are more honest than random row splits

---

## Forward Modeling Work Completed

### Original work
The original notebooks were:
- `airfoil-project-final.ipynb`
- `aeroml-clmax.ipynb`
- `aeroml-cdmin.ipynb`

Early findings:
- plain MLP outperformed the CNN-MLP hybrid
- `LDMax` was relatively strong
- `ClMax` was acceptable
- `CdMin` was poor
- training separate target-specific notebooks did not beat the clean shared baseline

### New forward notebook series created
These notebooks were created in this project:

- `aeroml_xfoil_forward_policy.ipynb`
- `aeroml_xfoil_forward_diagnostics.ipynb`
- `aeroml_xfoil_forward_v2.ipynb`
- `aeroml_xfoil_forward_v3.ipynb`
- `aeroml_xfoil_forward_v4.ipynb`

Shared helper module:
- `aeroml_notebook_common.py`

### Forward benchmark progression

#### `v1`
First clean benchmark using:
- `XFOIL ncrit=9`
- grouped split by geometry fingerprint
- engineered geometry features
- MLP-based multi-target model

This became the first trustworthy baseline.

#### Diagnostics notebook
Purpose:
- verify no split leakage
- analyze error by `Re`, `Mach`, and target slices
- inspect low-drag behavior

Important finding:
- `CdMin` remained the weakest target, especially in low-drag regions and some Mach bands

#### `v2`
Tried broader target-aware weighting.

Result:
- stable, but worse than `v1`
- conclusion: broad weighting was too blunt

#### `v3`
Focused only on `CdMin`, using a narrower weighting strategy.

Result:
- best forward model so far
- became the **current champion**

#### `v4`
Tried additional fine weight tuning.

Result:
- slightly better than `v1`
- slightly worse than `v3`
- conclusion: `v3` remains the best forward model

---

## Current Best Forward Model

### Champion model
`Forward v3` ensemble

### Best global results
Approximate held-out test metrics:

- `LDMax R² = 0.9077`
- `ClMax R² = 0.8708`
- `CdMin R² = 0.7223`

Additional `CdMin` metrics:
- `MedianAE ≈ 0.0123`
- `Within25Pct ≈ 78.85%`

### Interpretation
- Forward prediction is now good enough to be practically useful
- `CdMin` is still the weakest target
- low-drag inverse requests should still be treated carefully

### Artifact / model role
The dashboard and reverse workflow should use:
- **Forward v3 ensemble** as the main oracle

---

## Reverse Design Work Completed

### Reverse approach chosen
We did **not** use a one-shot generative model.

Instead, reverse design is:
- optimization-based
- over a low-dimensional geometry latent space
- with the forward `v3` ensemble used as the surrogate/oracle

### Why this approach
The reverse task is naturally one-to-many:
- many airfoils can produce similar aerodynamic targets
- direct generation would have been much harder to stabilize

Optimization over a constrained latent geometry space is more practical for a first working version.

### Reverse notebook progression

- `aeroml_reverse_design_optimization.ipynb`
- `aeroml_reverse_design_optimization_v2.ipynb`
- `aeroml_reverse_design_optimization_v3.ipynb`
- `aeroml_reverse_design_refinement.ipynb`

### Shared reverse pipeline concept
1. Represent airfoil geometry using thickness/camber distributions on fixed stations
2. Fit PCA on the training geometry representation
3. Optimize latent PCA coordinates instead of raw airfoil points
4. Decode latent vector back into airfoil geometry
5. Build forward-model features for that geometry
6. Run the `Forward v3` ensemble
7. Score the candidate using:
   - target match
   - ensemble disagreement / uncertainty
   - geometry plausibility penalties

---

## Reverse Progression Summary

### Reverse v1
- worked technically
- but search was slow and low confidence
- often collapsed to weak or high-uncertainty solutions

### Reverse v2
- added more diverse seeds
- added uncertainty-aware ranking
- still initialized from unrealistic flow neighborhoods in some cases

### Reverse v3
- added **flow-aware seeding**
- added local feasibility checks for requested `Re` and `Mach`
- restricted candidate pool to nearby flow conditions

This was the first reverse version that produced a genuinely promising candidate.

### Reverse refinement notebook
- takes the best `reverse v3` candidate
- reconstructs its latent representation
- runs local higher-budget optimization around it

This produced the current best reverse result.

---

## Current Best Reverse Result

### Best candidate source
From:
- `aeroml_reverse_design_refinement.ipynb`
- best candidate label: `jitter_5`

### Approximate best refined prediction
- `LDMax_pred ≈ 175.87`
- `ClMax_pred ≈ 1.909`
- `CdMin_pred ≈ 0.01966`

For a target around:
- `LDMax = 180`
- `ClMax = 1.85`
- `CdMin = 0.020`
- `Re = 2e6`
- `Mach = 0.10`

### Confidence
- passed uncertainty filters
- improved over the original reverse v3 winner

### Interpretation
The reverse pipeline is now working as a meaningful prototype, not just a proof of concept.

---

## Dashboard Direction Already Chosen

### Dashboard purpose
Showcase AeroML’s two capabilities in one app:

1. **Forward Prediction**
2. **Reverse Design**

### Chosen product direction
- **Deployment target:** local machine first
- **Audience:** research/demo users
- **Inference:** live inference from saved artifacts
- **Scope for v1:** core two flows only
- **Framework chosen:** `Streamlit`

### Planned dashboard structure

#### Forward page
- upload `.dat` file or choose known airfoil
- input `Re` and `Mach`
- run forward `v3` ensemble
- show predicted `LDMax`, `ClMax`, `CdMin`
- show uncertainty and geometry plot

#### Reverse page
- input target `LDMax`, `ClMax`, `CdMin`, `Re`, `Mach`
- choose reverse mode: `Fast Demo`, `Balanced`, or `High Quality`
- run flow-aware reverse search
- optionally refine the top candidate
- show ranked candidate list
- show geometry plots
- show target vs predicted metrics
- show uncertainty / feasibility warnings
- allow candidate CSV download

### Reverse dashboard modes
The dashboard should support three reverse-design quality/runtime modes.

#### `Fast Demo`
- fewer restarts
- fewer optimizer iterations
- no automatic refinement
- intended for fast interactive demo use

#### `Balanced`
- moderate restart count
- moderate optimizer iteration budget
- refinement optional after the first result
- intended as the default practical mode

#### `High Quality`
- larger restart count
- larger optimizer iteration budget
- allow or encourage refinement on the top candidate
- intended for the best candidate quality, even if runtime is much longer

### Why these modes are needed
- forward prediction is fast and suitable for live interaction
- reverse design is optimization-based, not a one-shot direct inverse model
- reverse search repeatedly calls the forward ensemble, so full-quality reverse runs can take much longer than forward inference
- the dashboard should therefore expose reverse design as a configurable search workflow rather than an always-instant action

---

## Files Created During This Project

### Shared code
- `aeroml_notebook_common.py`

### Forward notebooks
- `aeroml_xfoil_forward_policy.ipynb`
- `aeroml_xfoil_forward_diagnostics.ipynb`
- `aeroml_xfoil_forward_v2.ipynb`
- `aeroml_xfoil_forward_v3.ipynb`
- `aeroml_xfoil_forward_v4.ipynb`

### Reverse notebooks
- `aeroml_reverse_design_optimization.ipynb`
- `aeroml_reverse_design_optimization_v2.ipynb`
- `aeroml_reverse_design_optimization_v3.ipynb`
- `aeroml_reverse_design_refinement.ipynb`

### Notebook generator scripts
- `build_aeroml_kaggle_notebook.py`
- `build_aeroml_phase2_notebooks.py`
- `build_aeroml_v3_notebook.py`
- `build_aeroml_v4_notebook.py`
- `build_aeroml_reverse_design_notebook.py`
- `build_aeroml_reverse_design_v2_notebook.py`
- `build_aeroml_reverse_design_v3_notebook.py`
- `build_aeroml_reverse_refinement_notebook.py`

---

## What the Dashboard Implementer Should Treat as Ground Truth

### Ground-truth choices
- forward champion model = **Forward v3 ensemble**
- reverse search = **reverse v3 flow-aware search**
- local polishing = **reverse refinement notebook**
- app framework = **Streamlit**
- deployment target = **local first**

### Important limitations to preserve in UX
- `CdMin` is still the least trustworthy regime
- low-drag reverse requests should show confidence / feasibility warnings
- reverse results should not be presented as guaranteed physical truth
- app should make clear that reverse design is surrogate-guided candidate generation
- app should make clear that reverse-design runtime depends on the selected mode, and that higher-quality search can be much slower than forward prediction

---

## Recommended Near-Term Next Steps After the Dashboard

1. Add validation/comparison views:
   - refined winner
   - original reverse winner
   - nearest known real airfoils
   - side-by-side predicted metrics

2. Package the dashboard for easier sharing:
   - stable artifact layout
   - one-click local setup
   - eventual hosted deployment if needed

3. Improve reverse trustworthiness:
   - uncertainty calibration
   - plausibility checks
   - nearest-neighbor comparison
   - possibly a drag-specialist validation model later

---

## Short One-Paragraph Summary
AeroML is an airfoil ML system built around a clean `XFOIL ncrit=9` dataset policy. The current best forward model is the `Forward v3` ensemble, which predicts `LDMax`, `ClMax`, and `CdMin` from airfoil geometry plus `Re` and `Mach`. The reverse pipeline works by optimizing a PCA-based latent geometry representation using the forward ensemble as a surrogate. The best reverse result so far comes from the local refinement workflow, which improves a flow-aware reverse candidate into a strong prototype design. The next product step is a local Streamlit dashboard that exposes both forward prediction and reverse design with live inference, uncertainty, feasibility, and downloadable candidate outputs.
