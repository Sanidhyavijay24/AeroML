# AeroML

AeroML is a small deep learning system for airfoil design — it can look at an airfoil and tell you how it'll perform, or you can tell it how you *want* an airfoil to perform and it'll go find a geometry that gets you there.

I built this after getting curious about whether a surrogate model could replace a chunk of the trial-and-error in airfoil design — normally you'd run XFOIL over and over, tweaking geometry by hand and waiting on simulations. AeroML instead learns the mapping between geometry + flow conditions and aerodynamic performance, then uses that learned mapping to search backwards from a target.

## What it actually does

**Forward prediction** — give it an airfoil `.dat` file plus a Reynolds number and Mach number, and it predicts:
- `LDMax` — max lift-to-drag ratio
- `ClMax` — max lift coefficient
- `CdMin` — min drag coefficient

**Reverse design** — give it target values for those same three, plus your flow conditions, and it searches a compressed geometry space to propose airfoil shapes that should hit those targets. This isn't a generative model — it's an optimizer (multi-restart L-BFGS-B) working over a PCA-compressed latent representation of airfoil geometry, using the forward model as the thing it's optimizing against. It also reports how much its own ensemble disagrees on each candidate, so you're not just getting a shape back with no sense of how confident the system actually is in it.

Both are wrapped in a Streamlit dashboard so you can actually play with it instead of digging through notebooks.

## The data

Trained on XFOIL-simulated data (ncrit=9) across 6,000+ airfoil geometries. A few decisions worth knowing about if you're digging into the code:

- The raw dataset had multiple simulation sources per airfoil that didn't always agree with each other, especially on drag — so I settled on one canonical source (XFOIL ncrit=9) rather than training on a noisy blend.
- Train/val/test splits are grouped by geometry fingerprint, not split randomly row-by-row, so near-duplicate airfoils can't leak across splits and inflate the numbers.
- `CdMin` is trained in log-space, which handles the scale of drag values a lot better than raw.

The forward model is an ensemble of 3 seeds, and on held-out test data it lands around:

| Target | R² |
|---|---|
| LDMax | ~0.91 |
| ClMax | ~0.87 |
| CdMin | ~0.72 |

`CdMin` is the harder target of the three — drag is just noisier and more sensitive to fine geometry detail than lift — and it's the one I'm still actively improving.

## Running it locally

I'd recommend a dedicated conda environment so the TF/sklearn versions don't fight with anything else on your machine:

```bash
conda create --name aeroml python=3.10 -y
conda activate aeroml
```

Then install dependencies and launch:

```bash
cd AeroML
pip install -r requirements.txt
streamlit run app.py
```

It should open at `http://localhost:8501`. First launch will take a few seconds while it loads the cached dataset and model artifacts.

## Training and Reproduction

To retrain the forward prediction ensemble model locally using your GPU, run the parameterized CLI training script:

```bash
python scripts/train_forward.py --seeds 42,52,62 --variant cd_loss_only --epochs 80
```

This script reproduces the exact ensemble training pipeline (data prep, scaling, training all ensemble seeds, computing metrics, and writing artifacts) and saves them to `Forward_outputs/`.

You can also run one-off reverse design searches directly from the CLI without opening Streamlit:

```bash
python scripts/run_reverse.py --ldmax 120 --clmax 1.4 --cdmin 0.008 --re 3000000 --mach 0.15
```

## Project layout

```
app.py                          Streamlit dashboard entrypoint — forward + reverse UI
src/aeroml/                     Core package containing modules for data processing, features, models, training, evaluation, and runtime prediction
scripts/                        CLI scripts for retraining and running one-off reverse searches
Reference_Notebooks/            Historical experimentation history (v1 -> v3 forward, reverse search notebooks), kept for reference but no longer required to run or retrain
Data_Cache/                     Preprocessed dataset + train/val/test split manifest
Forward_outputs/                Trained ensemble models + metrics
Reverse_outputs/                Saved reverse-search results from past runs
context.md                      My running notes on architecture decisions
```

The `Data_Cache` folder ships with the repo on purpose — it means you can clone this and run the dashboard immediately without re-parsing 6,000+ raw `.dat` files first.

## Tech stack

TensorFlow/Keras for the forward ensemble, scikit-learn for PCA and scaling, SciPy for the reverse optimizer, Streamlit for the dashboard.

## Where this is headed

The reverse search currently works best when your target performance is reasonably close to something in the training distribution — it gets less reliable the further out you push it, particularly for very low-drag requests, which is the next thing on my list to tighten up. Longer term I'd like to add a proper geometry validity check that re-runs candidates through XFOIL directly rather than trusting the surrogate alone.

---

Built by [Sanidhya](https://github.com/Sanidhyavijay24) — feel free to open an issue if something breaks.
