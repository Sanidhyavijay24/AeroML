# AeroML Project Context

## Project Overview
AeroML is an advanced airfoil design system that integrates two primary physics-informed machine learning capabilities into a unified operational workflow:
1. **Forward Prediction:** Takes an airfoil geometry and fluid dynamics operating conditions (Reynolds number, Mach number) and predicts essential aerodynamic performance targets (`LDMax`, `ClMax`, `CdMin`).
2. **Reverse Design:** Takes a set of target aerodynamic parameters and flow conditions, and returns a plausible, high-performance candidate airfoil geometry designed specifically to match those conditions using surrogate-guided search across a PCA-compressed latent representation.

## Tech Stack
- ML/AI: **TensorFlow**, **scikit-learn**, **SciPy**
- Dashboard Framework: **Streamlit**
- Language Environment: **Python 3.x (managed via Anaconda/Conda)**

## Architecture
- **Inference modules:** Reusable Python scripts (`aeroml_forward_v3_runtime.py`, `aeroml_reverse_runtime.py`) executing core optimization and inference tasks by delegating to cached pre-trained ensemble models.
- **Frontend Dashboard:** A Streamlit server (`app.py`) built to provide real-time inference routing, user parameter configuration, robust session state management, and geometry visualisations without bleeding backend notebook procedures into the interface logic.
- **Data Architecture:** The system loads pre-processed scaling files, a dataset of `.dat` geometries and their XFOIL `pkl` output parameters from `Data_Cache` and `.keras` models from `Forward_outputs`.

## Feature Status
- [x] Consolidate forward and reverse pipelines into standalone reusable runtimes.
- [x] Write missing Streamlit app (`app.py`).
- [x] Connect forward prediction interface capabilities (upload `.dat`, enter flow constraints).
- [x] Connect reverse prediction module (configurable performance search modes, geometry evaluation, results exporter).
- [x] Refine system aesthetics to match premium product expectations.
- [x] Define comprehensive project requirements in `requirements.txt`.

## Open Issues / Technical Debt
None identified directly yet.
