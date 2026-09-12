"""Small source-only background fits, computed once per training context."""
from __future__ import annotations

import numpy as np

PENALTIES = (0.1, 1.0, 10.0, 100.0)


def ridge_predict(x, y, query, penalty):
    design = np.column_stack([np.ones(len(x)), x])
    q = np.column_stack([np.ones(len(query)), query])
    regularizer = np.eye(design.shape[1]); regularizer[0, 0] = 0
    return q @ np.linalg.solve(design.T @ design + penalty * regularizer, design.T @ y)


def ridge_loo(x, y, penalty):
    """Exact PRESS for a FIXED design/scaler and unpenalized intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(design.shape[1]); regularizer[0, 0] = 0
    inverse_times_x = np.linalg.solve(
        design.T @ design + penalty * regularizer, design.T,
    )
    fitted = design @ (inverse_times_x @ y)
    denominator = 1 - np.einsum('ij,ji->i', design, inverse_times_x)
    result = np.empty_like(y, dtype=np.float64)
    stable = denominator > 1e-8
    result[stable] = y[stable] - (y[stable] - fitted[stable]) / denominator[stable]
    # Uncommon high-leverage cases: exact refit, never silently divide by epsilon.
    for i in np.flatnonzero(~stable):
        keep = np.arange(len(x)) != i
        result[i] = ridge_predict(x[keep], y[keep], x[[i]], penalty)[0]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite background LOO')
    return result


def idw_predictions(y, distances_m, diagonal=False):
    weights = 1 / np.maximum(np.asarray(distances_m, dtype=np.float64) / 1000, 1e-3)**2
    if diagonal:
        np.fill_diagonal(weights, 0)
    return (weights @ y) / weights.sum(axis=1)


def select_background(x, y, distances_m):
    """All selection labels are from this explicitly supplied source subset."""
    idw = idw_predictions(y, distances_m, diagonal=True)
    best = None
    for penalty in PENALTIES:
        linear = ridge_loo(x, y, penalty)
        delta = linear - idw
        denom = float(delta @ delta)
        alpha = float(np.clip((y - idw) @ delta / denom, 0, 1)) if denom > 1e-15 else 0.0
        rmse = float(np.sqrt(np.mean((alpha * linear + (1-alpha) * idw - y)**2)))
        if best is None or rmse < best[0]:
            best = (rmse, penalty, alpha)
    return best


def crossfit_background(x, observed, distances_m, source):
    source = np.asarray(source, dtype=int)
    if len(source) < 4 or len(np.unique(source)) != len(source):
        raise ValueError('Background requires at least four distinct source stations')
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(observed[source], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite background inputs')
    _, penalty, alpha = select_background(x[source], y, distances_m[np.ix_(source, source)])
    predicted = np.full(len(x), np.nan)
    held = np.setdiff1d(np.arange(len(x)), source)
    if len(held):
        predicted[held] = alpha * ridge_predict(x[source], y, x[held], penalty) + (1-alpha) * idw_predictions(y, distances_m[np.ix_(held, source)])
    selected = []
    for station in source:
        others = source[source != station]
        # i's label is absent from BOTH hyperparameter selection and fitting.
        _, lam_i, alpha_i = select_background(x[others], observed[others], distances_m[np.ix_(others, others)])
        predicted[station] = alpha_i * ridge_predict(x[others], observed[others], x[[station]], lam_i)[0] + (1-alpha_i) * idw_predictions(observed[others], distances_m[np.ix_([station], others)])[0]
        selected.append((int(station), lam_i, alpha_i))
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite cross-fitted background')
    return predicted, penalty, alpha, selected
