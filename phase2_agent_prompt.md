# Agent Prompt — AeroML Phase 2: Restructure into a local package

Paste this together with `AEROML_LEVELUP_PLAN.md` (attached/pasted separately) for
full context on why this matters and what Phases 1, 3, and 4 look like. This prompt
covers Phase 2 only — do not start on Phase 3 (model retraining) or Phase 4 (polish)
work even if you can see it in the plan doc.

---

## Context

This is my AeroML repo. Phase 1 (making the reverse-design search fast via a batched
optimizer) is already done and confirmed working — `aeroml_reverse_runtime.py`,
`aeroml_forward_v3_runtime.py`, and `aeroml_notebook_common.py` at the repo root are
current and correct. Do not rewrite the optimization logic, the model architecture, or
any of the math in those files. This phase is a **structural refactor**, not a
rewrite of behavior.

Right now the *inference* path (`app.py` + those three files) is already clean and
notebook-free. What's still notebook-only is **training** — the forward ensemble
(v1 through the current `cd_loss_only` variant) was trained entirely inside
`Reference_Notebooks/aeroml-v3.ipynb`. Your job is to:

1. Reorganize the repo into a proper `src/` package.
2. Extract the training pipeline out of the notebook into a standalone, parameterized
   CLI script I can run on my local GPU.
3. Leave everything about *what the code does* unchanged — this is about *where the
   code lives* and *how it's invoked*, not new functionality.

## Non-negotiables

- **`streamlit run app.py` must still work exactly as before when you're done**, with
  identical behavior in both the Forward Prediction and Reverse Design tabs. This is
  the single most important acceptance criterion.
- **Do not change any model architecture, loss function, hyperparameter, random seed,
  or optimization logic** while moving code around. If you find something in the
  notebook that looks like a bug or an inconsistency while extracting it, don't fix it
  silently — flag it to me in your summary at the end instead.
- **Do not touch, retrain, or regenerate anything in `Forward_outputs/`,
  `Reverse_outputs/`, or `Data_Cache/`.** Those are trained artifacts and cached data;
  Phase 2 is purely about code organization. The training script you write should be
  *capable* of reproducing them, but don't actually run a full training job unless I
  ask you to.
- **Do not delete `Reference_Notebooks/`.** Keep it as-is, just note in the README
  that it's historical/exploratory and no longer required to reproduce results.
- Work in small, verifiable steps. After each major move (see steps below), run the
  app and confirm it still boots and behaves correctly before moving to the next step.
  If something breaks, fix it before continuing — don't pile up unverified changes.
- If you're not sure whether two code paths (e.g. a notebook cell and
  `aeroml_notebook_common.py`) are actually doing the same thing, check carefully
  rather than assuming — subtle differences here (e.g. a different random seed, a
  different scaler fit order) would silently change model results.

## Target structure

```
src/aeroml/
  __init__.py
  data.py          # dataset loading/caching, split logic — pulled from aeroml_notebook_common.py
  features.py       # geometry parsing, scalar feature engineering — pulled from aeroml_notebook_common.py
  models.py          # model architecture definition (build_forward_model, dense_block, etc.)
  train.py            # training loop logic, callable as a function (not just a script)
  evaluate.py          # metrics computation — writes the same ensemble_metrics.json schema as today
  forward.py            # ForwardV3Predictor, currently aeroml_forward_v3_runtime.py
  reverse.py              # ReverseV3Designer, currently aeroml_reverse_runtime.py

scripts/
  train_forward.py        # CLI entrypoint, e.g.: python scripts/train_forward.py --seed 42 --variant cd_loss_only --epochs 200
  run_reverse.py            # CLI entrypoint for one-off reverse searches outside Streamlit

app.py                        # unchanged behavior, imports updated to the new module paths
Reference_Notebooks/           # untouched, README note added marking it historical
requirements.txt                # pin versions
```

## Steps, in order

1. **Set up the package skeleton.** Create `src/aeroml/` with an `__init__.py`. Decide
   and tell me whether you're using a `src/` layout with an editable install
   (`pip install -e .` + `pyproject.toml`) or just adding `src/` to the Python path —
   pick whichever is simpler given the existing `requirements.txt`-based setup, and
   explain your choice.

2. **Move shared utilities first.** Split `aeroml_notebook_common.py` into
   `src/aeroml/data.py` (dataset loading/caching: `build_or_load_cached_dataset`,
   `build_or_load_split_manifest`, `materialize_indices`, `fit_transform_standard`)
   and `src/aeroml/features.py` (geometry parsing: `geometry_representation`,
   `cosine_spacing`, `estimate_le_radius`, `build_flow_features`, the `_trapz`
   compatibility shim). Update imports everywhere that used
   `aeroml_notebook_common`. Run the app, confirm both tabs still work, before
   continuing.

3. **Move the model architecture.** Pull `build_forward_model`, `dense_block`, and
   `set_all_seeds` out of `add_tf_helpers` in the old common module into
   `src/aeroml/models.py` as normal top-level functions (no need for the
   inject-into-namespace pattern once this isn't being run inside a notebook).

4. **Move the runtime classes.** Relocate `ForwardV3Predictor` into
   `src/aeroml/forward.py` and `ReverseV3Designer` into `src/aeroml/reverse.py`,
   updating their imports to pull from `src/aeroml/data.py`,
   `src/aeroml/features.py`, and `src/aeroml/models.py` instead of the old common
   module. Update `app.py`'s imports accordingly. Run the app again, confirm both
   tabs still work.

5. **Extract training into `scripts/train_forward.py`.** Read through
   `Reference_Notebooks/aeroml-v3.ipynb` carefully and pull out the actual training
   pipeline: data prep → scaling → building the 3-seed ensemble → training each
   seed → computing the ensemble metrics (including the low-drag regime slice) →
   writing `aeroml_xfoil_forward_v3_{variant}_seed{N}.keras` and
   `aeroml_xfoil_forward_v3_ensemble_metrics.json` in the same format the current
   runtime code expects. Make it a real CLI with `argparse`:
   - `--seeds` (default `42,52,62`, comma-separated)
   - `--variant` (name used in output filenames, default `cd_loss_only`)
   - `--epochs`, `--batch-size`, `--learning-rate` (defaults matching what's in the
     notebook)
   - `--output-dir` (default `Forward_outputs/`)
   Put the actual training loop logic in `src/aeroml/train.py` as an importable
   function; keep `scripts/train_forward.py` as a thin CLI wrapper around it. Same
   for evaluation — put metric computation in `src/aeroml/evaluate.py`.

   Do not actually run a full training job as part of this step unless I ask —
   just get the script working and verify with `--epochs 1` (or similar) that it
   runs end-to-end without crashing and produces output files in the expected
   shape and schema. Report back before doing a full run.

6. **Add `scripts/run_reverse.py`.** A small CLI that takes target LDMax/ClMax/CdMin
   and Re/Mach as arguments, runs `ReverseV3Designer.run_reverse_search`, and prints
   the best candidate plus writes the resulting `.dat` geometry to a file — so I can
   run a one-off reverse search from the terminal without opening Streamlit.

7. **Pin dependencies.** Update `requirements.txt` with pinned versions matching
   what's actually installed/working right now (`pip freeze` the relevant packages).
   If you set up a `pyproject.toml` in step 1, reflect that here too.

8. **Small README update.** Add a short section pointing to
   `scripts/train_forward.py` as the way to reproduce/retrain, and note that
   `Reference_Notebooks/` is kept for history but isn't required anymore. Don't
   rewrite the rest of the README — just add this one section.

## When you're done, report back with:

- A list of every file moved, created, or deleted.
- Confirmation that `streamlit run app.py` boots and both tabs work.
- Confirmation that `python3 scripts/train_forward.py --help` runs and shows sane
  arguments.
- Anything you found in the notebook while extracting training logic that looked
  inconsistent, undocumented, or worth double-checking — don't silently resolve
  ambiguities, tell me what you found and what you assumed.
- Any place you weren't sure whether to move vs. duplicate vs. leave alone, and why
  you made the call you did.
