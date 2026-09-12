# V4: full donors, nested background, bounded relation correction

This is the candidate accepted for testing after three proposal/review rounds,
NOT an empirically validated improvement. No target weather/history is added.

## Changes

- Ordinary station-balanced MSE, with existing full-dataset weights
  N/(number of training stations * station sample count). No tail or BCE loss.
- No artificial donor dropout or station mixup. Original 11 dynamic channels,
  49 static features, missing observation masks, geometry, and causal 24h history.
- For each training target, exclude its label from BOTH background hyperparameter
  selection and final fit. Nested ridge LOO uses a fixed design and an
  unpenalized intercept; a numerical high-leverage fallback uses explicit refits.
- Background context is scaled by fixed 10 micrograms/m3, with zero center.
  Donor historical means remain legal source inputs. No target-derived mean or
  label-derived across-station center is used in background context.
- Additional attention logits are tanh(w dot z), initialized at w=0.
  Distance/static scales come from off-diagonal training-source pairs; physical
  winds use source wind standard deviations WITHOUT shifting physical zero.
  The additional logits, not the complete attention scores, are bounded +/-1.
- Background lookup and relation scales live in memory for a context and are
  recomputed on resume. No persistent background cache to mix old versions.
  Precompute runtime is printed and recorded in background audit metadata.
- TCN is still run once per batch. No claim of further speedup has been tested.
  Target-relative wind makes blindly sharing TCN outputs across targets invalid.

Legacy V2 environment options for donor dropout, tail, virtual and event losses
no longer affect V4. Model family name is internally still v2; revision is
full_donor_nested_background_bounded_relation_v4. Old checkpoints are rejected.

## Colab pilot: recommended first run

After pulling the repository, set DL_TCN_DATA_ROOT=/content/dl_tcn_data and run:

    python run_v4_validation_pilot.py

This trains group 0 inner folds 0 and 3 only, 15 epochs each by default. It does
NOT perform outer testing, 72-refit or 73-station deployment. It uses the actual
existing group splits (not a claim that every inner fold is 60/12).

Outputs/checkpoints stay in /content/DL_TCN_V4_VALIDATION_PILOT, not Drive.
The directory contains per-fold training_history.csv and
validation_station_metrics_all_epochs.csv. Temporary predictions/checkpoints
are available for comparison; Colab runtime replacement can delete /content.
Rerunning in the same runtime resumes available state, or skips completed folds.
This pilot does not calculate the full six-fold production ensemble and cannot
establish all-station performance. Compare with matched old runs, not outer scores.

## Full runner (only when intentionally starting the production workflow)

    python run_73_v4_loso.py

Default fresh root: /content/drive/MyDrive/DL_TCN_V4_FULL_DONOR_73.
Drive must already be mounted. The legacy run_73_v2_loso.py is an alias for this
current revision. An existing DL_TCN_GROUP_73_OUTPUT_ROOT environment value
overrides the default; explicitly point it at a fresh V4 directory.
Existing grouped six-fold selector + per-station 72-refit remains unchanged.
DL_TCN_MAX_NEW_STATIONS can limit completed refits per invocation, but the
preceding group selector still needs all its six inner folds. Do not confuse
that setting with the two-fold pilot.

## Local sanity

    python -m unittest discover -s tests -p "test_v*_revision.py"

Covers exact-vs-brute LOO, training-target-label perturbation, held-out/future
label exclusion, fixed background units, source-only relation scales,
station weights, finite gradients and actual two-epoch limited-batch loops.
