"""Assessment-accuracy and interval-calibration metrics (IAAO conventions)."""
from __future__ import annotations

import numpy as np


def mdape(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.median(np.abs(pred - actual) / actual))


def share_within(pred, actual, tol=0.10):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.mean(np.abs(pred - actual) / actual <= tol))


def cod(pred, actual):
    """Coefficient of dispersion of the ratio pred/actual about its median (%)."""
    ratio = np.asarray(pred, float) / np.asarray(actual, float)
    med = np.median(ratio)
    return float(100.0 * np.mean(np.abs(ratio - med)) / med)


def prd(pred, actual):
    """Price-related differential: mean ratio / dollar-weighted mean ratio.
    >1.03 regressive (low-value over-valued); <0.98 progressive."""
    pred = np.asarray(pred, float); actual = np.asarray(actual, float)
    ratio = pred / actual
    mean_ratio = float(np.mean(ratio))
    weighted = float(np.sum(pred) / np.sum(actual))
    return mean_ratio / weighted if weighted else float("nan")


def by_strata(pred, actual, k=3):
    """COD/PRD per price stratum (terciles of actual price by default)."""
    pred = np.asarray(pred, float); actual = np.asarray(actual, float)
    order = np.argsort(actual)
    out = []
    for j, idx in enumerate(np.array_split(order, k)):
        out.append({
            "stratum": j + 1,
            "n": int(len(idx)),
            "price_lo": float(actual[idx].min()),
            "price_hi": float(actual[idx].max()),
            "mdape": mdape(pred[idx], actual[idx]),
            "cod": cod(pred[idx], actual[idx]),
            "prd": prd(pred[idx], actual[idx]),
        })
    return out


def conformal_q(cal_pred, cal_actual, alpha=0.10):
    """Split-conformal half-width on the LOG scale (multiplicative interval).
    Uses the finite-sample-valid rank (ceil((n+1)(1-alpha))/n quantile)."""
    r = np.abs(np.log(np.asarray(cal_actual, float)) - np.log(np.asarray(cal_pred, float)))
    n = len(r)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, level))


def coverage(pred, actual, q):
    """Empirical coverage of the multiplicative interval pred*[e^-q, e^+q]."""
    pred = np.asarray(pred, float); actual = np.asarray(actual, float)
    lo, hi = pred * np.exp(-q), pred * np.exp(q)
    return float(np.mean((actual >= lo) & (actual <= hi)))
