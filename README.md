# AeroML: Airfoil Design & Analysis Framework ✈️

AeroML is a Deep learning system that provides two major capabilities for aerodynamic design:

1. **Forward Prediction:** Instantly predicts essential aerodynamic performance targets (`LDMax`, `ClMax`, `CdMin`) by analyzing an inputted airfoil geometry alongside fluid dynamics operating conditions (Reynolds number, Mach number).
2. **Reverse Design:** An optimization-based design flow. Provide your target aerodynamic parameters and flow conditions, and the system performs a surrogate-guided search across a PCA-compressed latent representation to generate plausible, high-performance candidate airfoil geometries matching your targets.

Built with **TensorFlow / Keras**, **scikit-learn**, and packaged locally into a premium interactive dashboard with **Streamlit**.

---

## 🚀 Quickstart & Setup

To ensure all machine learning dependencies run cleanly, we recommend creating a dedicated Conda environment.

### 1. Environment Setup
Open your terminal (or **Anaconda Prompt** if on Windows) and run:

```bash
# Create a new environment with Python 3.10
conda create --name aeroml python=3.10 -y

# Activate the environment
conda activate aeroml
```

### 2. Install Dependencies
Navigate into the repository directory and install the necessary packages using the provided `requirements.txt`:

```bash
cd /path/to/AeroML
pip install -r requirements.txt
```

### 3. Run the Dashboard
Boot up the local Streamlit server:

```bash
streamlit run app.py
```

The application will launch automatically in your default web browser (typically at `http://localhost:8501`).

---

## 📂 Project Structure

- `app.py`: The main Streamlit dashboard application mapping UI to ML tasks.
- `aeroml_notebook_common.py`: Core utility scripting handles dataset loading, artifact scaling, PCA transforms, and glob-based artifact searching.
- `aeroml_forward_v3_runtime.py`: Executes predictions routing inputs to the pre-trained `Forward v3` ensemble Keras models. 
- `aeroml_reverse_runtime.py`: Handles optimization, PCA latent space bounds, feasibility checks, and reverse geometry reconstruction algorithms. 
- `requirements.txt`: Core library dependencies.
- `context.md`: Internal ledger tracking architecture decisions and technical notes.

*Note: The `Data_Cache` directory containing the pre-processed `.npz` dataset and `.csv` splits is included in this repository. This allows the Streamlit app and inference models to run instantly out-of-the-box without needing to re-parse the raw 6,000+ airfoil `.dat` geometries.*
