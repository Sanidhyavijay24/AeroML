# AeroML Level-Up Plan

Goal: take AeroML from "working demo with a slow reverse pass" to a project that's fast, GPU-trained end to end, cleanly structured, and has a real quantified engineering story behind it — good enough to put in front of a recruiter or post about without caveats.

This is roughly four phases. They build on each other, so do them in order — don't jump to retraining models before the reverse search is fast, or you'll be waiting 20 minutes every time you want to sanity-check a change.

---

## Phase 1 — Fix the reverse-search bottleneck

**Why first:** this is the single most visible thing about the project right now (a live demo that takes 20 minutes doesn't feel like a live demo), and it's also the best resume story in the whole repo if done properly. Everything else is incremental; this is a step-change.

**Diagnosis (already confirmed from the code):**
- `scipy.optimize.minimize` is called without a `jac`, so L-BFGS-B silently falls back to finite-difference gradients — ~13 extra objective evaluations per gradient step for your 12-dim latent space.
- Each objective evaluation calls the 3-model ensemble eagerly on a batch of 1, three separate times. TF's per-call Python↔graph dispatch overhead (~10-50ms) dominates, not the actual matrix math.
- You run this whole thing 8 times in the initial search and 6 more times in refinement, each with dozens of iterations. The evaluation count compounds fast.

**Tasks:**
1. Rewrite `objective()` and the optimization loop to run **all restarts as one batched, GPU-resident optimization** using `tf.Variable` of shape `(n_restarts, 12)` and `tf.GradientTape` for analytic gradients through the PCA-inverse → scalar-feature → forward-model pipeline. Use Adam (or a batched L-BFGS if you want to keep it closer to the current optimizer) instead of scipy's per-sample CPU loop.
   - Note: the scalar-feature computation (`scalar_from_surfaces`) currently uses non-differentiable numpy ops (`argmax`, `np.gradient`, arc-length sums). You'll need to either reimplement that block in TF ops so gradients flow through it, or keep it as a numerically-differentiated small block while batching the *model* calls — the second option is a smaller lift and still gets you most of the speedup.
2. Wrap the forward ensemble call in `@tf.function` so it's traced once instead of re-dispatched on every call.
3. Batch the finite-difference perturbations if you keep any numerical-gradient fallback — one call of shape `(13, 12)` instead of 13 sequential calls.
4. Cache the fitted PCA + scalers + geometry limits (pickle them) instead of rebuilding from the raw dataset every time `ReverseV3Designer` is instantiated.
5. Re-run the existing `feasibility_summary` / `passes_uncertainty` logic unchanged — the goal here is speed, not changing what the optimizer is optimizing for.

**Target:** get the full reverse search (init pool + optimization + refinement) down to single-digit seconds. Benchmark before/after with a wall-clock number you can actually quote.

**Deliverable:** update `app.py` to run the reverse search live by default once this lands, instead of showing a pre-cached result. Keep the cached fallback as a "instant preview" toggle if you want belt-and-suspenders for the live demo.

---

## Phase 2 — Restructure into a proper local package

**Why:** right now the *inference* path (`app.py`, the two runtime files, `aeroml_notebook_common.py`) is already clean and notebook-free — that part doesn't need dismantling. What's still notebook-only is **training**. That's the part to move to your machine and your GPU.

**Tasks:**
1. New structure:
   ```
   src/aeroml/
     data.py          # dataset loading, cleaning, split logic (pulled from aeroml_notebook_common.py)
     features.py       # geometry parsing, scalar feature engineering
     models.py          # model architectures (forward ensemble definition)
     train.py            # standalone, argparse-driven training script
     evaluate.py         # metrics computation, writes the same ensemble_metrics.json schema
     reverse.py           # the (now-fast) reverse designer
     forward.py            # forward predictor runtime
   scripts/
     train_forward.py      # CLI entrypoint: python scripts/train_forward.py --variant cd_loss_only --seed 42
     run_reverse.py         # CLI entrypoint for one-off reverse searches outside Streamlit
   app.py                    # unchanged, just updated imports
   Reference_Notebooks/       # kept as-is, explicitly labeled as exploration history, not required to reproduce results
   ```
2. Make `train_forward.py` fully parameterized (seed, architecture variant, loss weighting, epochs) so you can kick off GPU runs from the command line and log results reproducibly — this is what actually benefits from your 5050, since training touches full batches over the whole dataset repeatedly, unlike single-sample inference.
3. Pin dependency versions in `requirements.txt` (right now several are unpinned) and add a `pyproject.toml` or at minimum an `environment.yml` so "clone and run" is actually guaranteed to work a year from now.
4. Add a one-line note in the README pointing to `Reference_Notebooks/` as history, and `scripts/train_forward.py` as the canonical way to reproduce/retrain.

**Deliverable:** you can run `python scripts/train_forward.py --seed 42` on your laptop GPU and get a model out, no Kaggle notebook required.

---

## Phase 3 — Use the GPU for what it's actually good for: improving the model

Now that training is scriptable and local, this is where your 5050 earns its keep.

**Tasks:**
1. **Attack the low-drag `CdMin` weakness directly.** From the metrics you already have, `CdMin` is the weak target and low-drag rows are the weakest slice of it. Options, roughly in order of effort:
   - Stratified sampling / reweighting to oversample low-drag rows during training.
   - A quantile or focal-style loss that penalizes the model more for large errors in the low-drag regime specifically.
   - A separate small head or auxiliary model specialized for low-drag conditions, blended with the main ensemble.
2. **Broaden the ensemble** — more seeds, or a couple of different architectures (not just seed variation) now that a training run is cheap to kick off locally.
3. **Re-run the same held-out evaluation** and update `Forward_outputs/aeroml_xfoil_forward_v3_ensemble_metrics.json` (or a v5) with the new numbers, including the low-drag slice, so you have an honest before/after.
4. Once forward model v5 is meaningfully better on the low-drag slice, re-point `aeroml_forward_v3_runtime.py` (or fork it to `_v5`) at the new artifacts.

**Deliverable:** a concrete "R² on low-drag CdMin went from X to Y" number — this is the second good quantified story for your resume/LinkedIn, alongside the reverse-search speedup.

---

## Phase 4 — Polish for public consumption

**Tasks:**
1. Update the README's metrics table with the Phase 3 numbers once they land.
2. Add a small `tests/` directory with real `pytest` assertions (not just print statements like the current `test_drift.py`) and a GitHub Actions workflow that runs them on push.
3. Add a `LICENSE` file.
4. Record a 30-60 second screen capture of the dashboard doing a live reverse search in a few seconds — this is your Twitter/LinkedIn post, and it only works once Phase 1 is done.
5. Consider deploying a live version (Streamlit Community Cloud or a Hugging Face Space) so people can try it without cloning — the repo is small enough (~45MB with data included) that this is realistic.
6. Write the actual post copy once everything above lands: lead with the concrete numbers (speedup, R² improvement), not adjectives. "Reverse airfoil design from 18 minutes to 4 seconds by batching the optimizer and using analytic gradients instead of scipy's finite-difference fallback" is a much stronger LinkedIn hook than "excited to share my ML project."

---

## Suggested order

1. Phase 1 (speed) — do this first, it unblocks everything else and is the highest-leverage single change.
2. Phase 2 (restructure) — mechanical, but needed before Phase 3 so GPU training runs are reproducible and scriptable.
3. Phase 3 (model quality) — the actual ML work, now fast to iterate on locally.
4. Phase 4 (polish + publish) — last, once there are real numbers to report.

Ready to start on Phase 1 whenever you are — that's the batched-optimizer rewrite of `aeroml_reverse_runtime.py`.
