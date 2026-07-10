# AeroML Project Context

## Project Overview
AeroML is an advanced airfoil design system that integrates two primary physics-informed machine learning capabilities into a unified operational workflow:
1. **Forward Prediction:** Takes an airfoil geometry and fluid dynamics operating conditions (Reynolds number, Mach number) and predicts essential aerodynamic performance targets (`LDMax`, `ClMax`, `CdMin`).
2. **Reverse Design:** Takes a set of target aerodynamic parameters and flow conditions, and returns a plausible, high-performance candidate airfoil geometry designed specifically to match those conditions using surrogate-guided search across a PCA-compressed latent representation.

## Tech Stack
- Python Environment: **Python 3.10** managed via Conda (environment: `aeroml`).
- Machine Learning: **TensorFlow 2.21.0**, **scikit-learn 1.7.2**, **SciPy 1.15.3**.
- Utilities: **NumPy 2.2.6**, **Pandas 2.3.3**, **Matplotlib 3.10.9**, **tqdm 4.68.4**.

## Architecture & Folder Structure
```
AeroML/
├── src/
│   └── aeroml/
│       ├── __init__.py      # Package version and metadata
│       ├── data.py          # Dataset loading, caching, split manifest, scaling helpers
│       ├── features.py      # Cosine spacing, LE radius, geometry representations, decoder
│       ├── models.py        # Forward ensemble model definition, Swish dense blocks, seeds
│       ├── train.py         # Standalone ensemble training function
│       ├── evaluate.py      # Metric collection, regression reports, baseline comparisons
│       ├── forward.py       # ForwardV3Predictor class definition
│       └── reverse.py       # ReverseV3Designer class definition (Adam-based batched search)
├── scripts/
│   ├── train_forward.py     # Command Line Interface to train/fine-tune the forward ensemble
│   └── run_reverse.py       # Command Line Interface to run one-off reverse design searches
├── tests/
│   ├── test_data/           # Baseline validation airfoil coordinates (.dat format)
│   └── test_forward_drift.py # Validation test script to verify predictions and check for drift
├── Data_Cache/              # Preprocessed dataset + train/val/test split manifest
├── Forward_outputs/         # Trained ensemble models + metrics
```

## Data Flow
```
[Raw .dat files] ─(geometry representation)─> [NPZ Dataset / Split manifest]
                                                      │
                                             (Standard scaling)
                                                      │
                                                      ▼
[scripts/train_forward.py] ───────────────> [Forward Ensemble (.keras)]
                                                      │
                                             (Forward predictions)
                                                      │
                                                      ▼
[scripts/run_reverse.py] ──(Batched Adam)─> [Target Airfoil Geometry (.dat)]
```

## Feature Status
- [x] **Phase 1: Fix Reverse-Search Bottleneck**
  - [x] Implement batched latent search with Adam optimizer.
  - [x] Trace forward ensemble predictions using tf.function compile.
- [x] **Phase 2: Restructure into a Local Package**
  - [x] Reorganize codebase into `src/aeroml` modules.
  - [x] Extract dataset loader/splitting to `data.py`.
  - [x] Extract geometry/spacing algorithms to `features.py`.
  - [x] Extract MLP architecture to `models.py`.
  - [x] Relocate `ForwardV3Predictor` to `forward.py`.
  - [x] Relocate `ReverseV3Designer` to `reverse.py`.
  - [x] Implement argparse-driven CLI training script `scripts/train_forward.py`.
  - [x] Implement CLI reverse search script `scripts/run_reverse.py`.
  - [x] Pin exact working package versions in `requirements.txt` and `pyproject.toml`.
  - [x] Keep backwards compatibility and clean up legacy root-level runtime scripts.

## Data Models
- **Caching Dataset:** NPZ file (`aeroml_xfoil_n9_dataset.npz`) containing:
  - `X_profile`: profile features (thickness, camber, derivatives)
  - `X_scalar`: engineered scalar coordinates
  - `X_flow`: scaled Reynolds and Mach features
  - `y_targets`: raw `[LDMax, ClMax, CdMin]` parameters.
- **Split Manifest:** CSV file containing the map between `fingerprint` and split label `[train, val, test]`.

## Open Issues / Technical Debt
- **TensorFlow Retracing Warn:** Retracing warnings are raised on native Windows if tf.functions are compiled repeatedly inside loops (managed in `ForwardV3Predictor` via pre-tracing compilation during loading).
