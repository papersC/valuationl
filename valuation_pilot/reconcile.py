"""Reconciliation and the conformal interval.

The reconciled value is a reliability-weighted convex combination of the
approach values (paper eq. 1); by convexity it lies within the approach range
(no-extrapolation invariant). The conformal half-width is the (1-alpha) empirical
quantile of supplied absolute residuals. The reliability weights are a heuristic
prior (see agents.py) to be calibrated in deployment; the conformal coverage
guarantee holds only under exchangeability of the calibration and test data
(see coverage_experiment.py for an empirical check and its failure under drift).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import Estimate


@dataclass
class Reconciled:
    value: float
    interval: tuple[float, float]
    weights: dict
    reliability_label: str
    approach_range: tuple[float, float]


def reconcile(estimates: list[Estimate], calibration_residuals, alpha: float = 0.10) -> Reconciled:
    active = [e for e in estimates if e.reliability > 0]
    rsum = sum(e.reliability for e in active)
    weights = {e.appr: e.reliability / rsum for e in active}
    value = sum(weights[e.appr] * e.value for e in active)

    vals = [e.value for e in active]
    lo_range, hi_range = min(vals), max(vals)
    assert lo_range - 1e-6 <= value <= hi_range + 1e-6, "NE violated"

    q = float(np.quantile(np.abs(np.asarray(calibration_residuals, dtype=float)), 1.0 - alpha))
    interval = (value - q, value + q)

    deep = [e for e in active if e.appr in ("SC", "Inc")]
    if len(deep) == 2:
        disagree = abs(deep[0].value - deep[1].value) / value
        label = "high" if disagree < 0.03 else ("medium" if disagree < 0.08 else "low")
    else:
        label = "low"
    return Reconciled(value, interval, {k: round(v, 3) for k, v in weights.items()},
                      label, (lo_range, hi_range))
