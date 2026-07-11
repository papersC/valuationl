"""Calibration-vintage ablation (referee comment M4).

The headline conformal coverage (0.777) fixes the calibration set a full year
before the test year, so on a rising market the interval is stale by
construction. This ablation holds the point estimator fixed (one model trained
strictly before the calibration year) and varies ONLY the calibration vintage,
from the full prior year down to the freshest prior quarter, plus a rolling
scheme that recalibrates each test quarter on the immediately preceding quarter.
Coverage moving back toward the nominal 0.90 as the vintage freshens shows the
0.777 figure is a recalibration-cadence artefact the certificate is meant to
expose, not an irreducible failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .hedonic import fit_hedonic


def _cov(model, cal, test, alpha):
    q = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), alpha)
    cov = M.coverage(model.predict_price(test), test["price"].to_numpy(float), q)
    return q, cov


def recalibration_ablation(df, test_year, alpha=0.10, seed=0):
    cal_year = test_year - 1
    train = df[df["year"] < cal_year]
    model = fit_hedonic(train, seed=seed)
    cal_all = df[df["year"] == cal_year]
    test = df[df["year"] == test_year]
    qp_cal = pd.PeriodIndex(cal_all["quarter"], freq="Q")

    variants = {}
    # 1) annual (as in the headline): calibrate on the full prior year
    q, cov = _cov(model, cal_all, test, alpha)
    variants["annual"] = {"cal_span": f"{cal_year} (4 quarters)", "n_cal": int(len(cal_all)),
                          "half_width_log": round(q, 4), "coverage": round(cov, 4)}
    # 2) half-year: freshest two quarters of the prior year
    h2 = cal_all[qp_cal >= pd.Period(f"{cal_year}Q3", freq="Q")]
    q, cov = _cov(model, h2, test, alpha)
    variants["half_year"] = {"cal_span": f"{cal_year}H2 (2 quarters)", "n_cal": int(len(h2)),
                             "half_width_log": round(q, 4), "coverage": round(cov, 4)}
    # 3) quarterly: freshest single prior quarter
    q4 = cal_all[qp_cal == pd.Period(f"{cal_year}Q4", freq="Q")]
    q, cov = _cov(model, q4, test, alpha)
    variants["quarterly"] = {"cal_span": f"{cal_year}Q4 (1 quarter)", "n_cal": int(len(q4)),
                             "half_width_log": round(q, 4), "coverage": round(cov, 4)}

    # 4) rolling: recalibrate each test quarter on the immediately preceding quarter
    test_qs = sorted(test["quarter"].unique(), key=lambda s: pd.Period(s, freq="Q"))
    covered = 0; ntot = 0
    per_q = []
    for tq in test_qs:
        tqp = pd.Period(tq, freq="Q")
        prevp = tqp - 1
        cal = df[pd.PeriodIndex(df["quarter"], freq="Q") == prevp]
        te = test[test["quarter"] == tq]
        if len(cal) < 100 or len(te) < 20:
            continue
        q = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), alpha)
        lo = model.predict_price(te) * np.exp(-q); hi = model.predict_price(te) * np.exp(q)
        a = te["price"].to_numpy(float)
        c = int(np.sum((a >= lo) & (a <= hi)))
        covered += c; ntot += len(a)
        per_q.append({"test_quarter": tq, "cal_quarter": str(prevp),
                      "coverage": round(c / len(a), 4)})
    variants["rolling_quarterly"] = {"cal_span": "each test quarter on the prior quarter",
                                     "n_cal": None, "coverage": round(covered / ntot, 4) if ntot else None,
                                     "per_quarter": per_q}
    return {"nominal": 1 - alpha, "cal_year_base": int(cal_year), "test_year": int(test_year),
            "n_train": int(len(train)), "n_test": int(len(test)), "variants": variants}


def decision_boundary(df, test_year, alpha=0.10, seed=0, ltv=0.80):
    """How wide is the 90% interval relative to a lending decision boundary, per
    price stratum? (referee minor comment). Reports the multiplicative half-width
    and the loan-to-value band that a nominal ``ltv`` cap spans across the
    interval -- making the interval, not the point value, the decision-relevant
    object, and showing whether the low stratum carries wider relative
    uncertainty (a regressivity signal)."""
    cal_year = test_year - 1
    model = fit_hedonic(df[df["year"] < cal_year], seed=seed)
    cal = df[df["year"] == cal_year]
    test = df[df["year"] == test_year]
    a = test["price"].to_numpy(float)
    order = np.argsort(a)
    strata = np.array_split(order, 3)

    def band(q):
        # a value V has interval [V e^-q, V e^+q]; an ltv cap lends ltv*V, which is
        # ltv*e^{+q} of the conservative (lower) value -> the interval spans this LTV range
        return {"half_width_pct_up": round(float(np.exp(q) - 1) * 100, 1),
                "half_width_pct_dn": round(float(1 - np.exp(-q)) * 100, 1),
                "ltv_at_lower_bound_pct": round(float(ltv * np.exp(q)) * 100, 1)}

    # pooled
    q_all = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), alpha)
    out = {"ltv_nominal_pct": int(ltv * 100), "pooled": {"half_width_log": round(q_all, 4), **band(q_all)},
           "by_stratum": []}
    cal_a = cal["price"].to_numpy(float)
    cal_order = np.argsort(cal_a)
    cal_strata = np.array_split(cal_order, 3)
    for j, (ci, ti) in enumerate(zip(cal_strata, strata)):
        csub = cal.iloc[ci];
        q = M.conformal_q(model.predict_price(csub), csub["price"].to_numpy(float), alpha)
        out["by_stratum"].append({"stratum": j + 1, "half_width_log": round(q, 4),
                                  "price_lo": float(a[ti].min()), "price_hi": float(a[ti].max()),
                                  **band(q)})
    return out
