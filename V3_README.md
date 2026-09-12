# V3 training revision (existing V2 entry point)

Run `python run_73_v2_loso.py`. Model family remains v2; training revision is
`real_station_masked_mse_v3`. Default output is a NEW Drive directory
`/content/drive/MyDrive/DL_TCN_V3_MASKED_MSE_73`. Do not resume old V2 results.

- Physical attention uses source-scaler inverse-transformed wind in m/s. TCN
  values remain standardized. Missing wind contributes zero to the prior.
- Remove spatial mixup and its second model forward/backward path. Training
  randomly excludes 15% of donor tokens, with at least one retained; validation
  and outer inference retain their original donor pools. No synthetic labels,
  target coordinates, or target static features are created. Existing observation
  masks stay unchanged; exclusion uses attention padding. The source-fitted
  background still uses the original source pool: dropout is representation
  regularization, NOT an entirely withheld-station experiment.
- Station-balanced MSE with bounded tail weights: 1 below 25, linear rise to 3
  at 35, then capped at 3. This is concentration weighting, NOT density-estimated
  weighting. It uses training labels only. Auxiliary event loss is disabled;
  event head remains for state-dict compatibility, not deployment or training.
- Source background, splits, refit72, static49, dynamic11, and ensemble selection
  are unchanged. Legacy virtual/event loss environment options have no effect.
- No full training has been performed to establish accuracy or speed. Synthetic
  two-step AdamW test: `python tests/test_v3_revision.py`.

The output marker rejects incompatible or unversioned directories before the
runner can reuse selector locks or skip old station results. Old files are not
deleted. This change removes duplicate model work, not half of total runtime.
