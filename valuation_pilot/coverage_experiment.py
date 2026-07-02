"""Empirical conformal-coverage experiment on synthetic ground truth.

Split-conformal coverage is guaranteed only under exchangeability of the
calibration and test data. Housing markets drift, so we measure coverage twice:
once with exchangeable calibration/test draws (coverage should match the nominal
level) and once with a temporal price drift injected into the test draws
(coverage should degrade). This characterizes both the method and its documented
failure mode. All values are synthetic; this measures interval CALIBRATION, not
point accuracy.
"""
from __future__ import annotations

import numpy as np

from .agents import run_all_agents
from .evidence import Evidence, EvidenceStores
from .ledger import Ledger
from .pipeline import _point

DT = "2BR-apartment"


def _stores_for(true_value: float, seed: int, size: float = 110.0, drift: float = 0.0):
    rng = np.random.default_rng(seed)
    stores = EvidenceStores.empty(Ledger())
    attrs = {"dwelling_type": DT, "built_up_area": size, "floor_level": "mid",
             "parking": 1, "tenure": "freehold"}
    stores.attrs.add(Evidence("attr:subject", "Attr", "s", "t", attrs))
    unit_true = true_value / size
    for i in range(6):
        up = unit_true * (1.0 + drift) + rng.normal(0, 0.03 * unit_true)
        stores.comps.add(Evidence(f"comp:c{i+1}", "Comp", "s", "t",
                                  {"dwelling_type": DT, "unit_price": float(up),
                                   "size": 110.0, "floor": 5, "parking": 1}))
    stores.index.add(Evidence("idx:i", "Index", DT, "t",
                              {"segment": DT, "trailing_quarter_growth": 0.0}))
    y = 0.067
    for i in range(3):
        rent = true_value * y * (1.0 + rng.normal(0, 0.03))
        stores.leases.add(Evidence(f"lease:l{i+1}", "Lease", "s", "t",
                                   {"dwelling_type": DT, "annual_rent": float(rent)}))
    stores.leases.add(Evidence("yield:y1", "Lease", DT, "t",
                               {"dwelling_type": DT, "gross_yield": y}))
    stores.attrs.add(Evidence("cost:m1", "Attr", "s", "t",
                              {"land_value": true_value * 0.45,
                               "replacement_cost": true_value * 0.60, "depreciation": 0.20}))
    return stores, attrs


def _predict(true_value: float, seed: int, drift: float = 0.0) -> float:
    stores, attrs = _stores_for(true_value, seed, drift=drift)
    return _point(run_all_agents(stores, attrs))


def run(n_cal: int = 300, n_test: int = 1000, alpha: float = 0.10,
        seed: int = 0, drift_test: float = 0.0):
    cal_true = np.random.default_rng(seed).uniform(1.5e6, 2.5e6, size=n_cal)
    cal_res = np.array([abs(_predict(tv, 10_000 + i) - tv) for i, tv in enumerate(cal_true)])
    q = float(np.quantile(cal_res, 1.0 - alpha))
    test_true = np.random.default_rng(seed + 1).uniform(1.5e6, 2.5e6, size=n_test)
    covered = sum(1 for i, tv in enumerate(test_true)
                  if _predict(tv, 500_000 + i, drift=drift_test) - q <= tv
                  <= _predict(tv, 500_000 + i, drift=drift_test) + q)
    return covered / n_test, q


if __name__ == "__main__":
    cov0, q = run(drift_test=0.0)
    covd, _ = run(drift_test=0.05)
    print(f"nominal coverage 1-alpha      = 0.90")
    print(f"empirical coverage (exchangeable) = {cov0:.3f}   (q = CU {q/1e6:.3f}M)")
    print(f"empirical coverage (+5% drift)    = {covd:.3f}")
