# Decomposed-field DL pilot

This is the single follow-up to V8. It retains the strict 60/12 unseen-station
protocol, 49 train-normalized static features, 11 dynamic channels, 24-hour
history and 59/60 donor rules.

Prediction is explicitly split into:

1. a convex mixture of five fixed distance fields;
2. a bounded, time-invariant target-static offset;
3. a donor-history anomaly constrained toward zero station mean.

Only folds 0 and 3 are run. The exact path is benchmarked first. If the
projected two-fold maximum-20-epoch cost exceeds two A100 hours, training is
aborted automatically. Epoch 15 is recorded, convergence is checked at 18,
and epoch 20 is an upper cap rather than an automatic target.

The final gate is locked against the supplied V8 fold-0/fold-3 results:

- each fold macro RMSE improves by at least 5%;
- mean absolute bias does not worsen;
- sparse-quartile macro RMSE does not worsen;
- no validation station RMSE worsens by more than 10%.

The runner never launches a full 73-station experiment. During training only
the active resume state and best model are mirrored to Drive. At fold
completion it keeps the best model, compact histories, best station metrics,
compressed predictions and one result JSON.
