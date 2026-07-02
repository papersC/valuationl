"""Measured experiments over the pilot (all on synthetic evidence).

Three measurements that feed the paper's measured-results figure:

1. coverage_vs_nominal  -- empirical coverage of the split-conformal interval as
   a function of the nominal level, for an exchangeable test set and for a test
   set with an injected price drift. The exchangeable curve should track the
   diagonal (the conformal guarantee); the drifted curve should fall below it
   (the documented failure mode).
2. coverage_vs_drift    -- empirical coverage at nominal 0.90 as the injected
   drift magnitude grows from 0 to 8%.
3. audit_latency        -- measured wall-clock of the structural citation audit
   (anchoring + structured entailment, audit.audit_citation_invariant) as a
   function of the number of value-affecting claims n, demonstrating the O(n)
   cost claimed in the paper.

Everything is deterministic given the seeds. Run as a script to print a JSON
document with all three result sets:

    python -m valuation_pilot.experiments
"""
from __future__ import annotations

import json
import time

import numpy as np

from .audit import audit_citation_invariant
from .coverage_experiment import _predict
from .narrative import Claim
from .pipeline import value_property


# ---------------------------------------------------------------- coverage ---

def _calibration_residuals(n_cal: int = 300, seed: int = 0) -> np.ndarray:
    cal_true = np.random.default_rng(seed).uniform(1.5e6, 2.5e6, size=n_cal)
    return np.array([abs(_predict(tv, 10_000 + i) - tv)
                     for i, tv in enumerate(cal_true)])


def _test_errors(drift: float, n_test: int = 1000, seed: int = 0) -> np.ndarray:
    test_true = np.random.default_rng(seed + 1).uniform(1.5e6, 2.5e6, size=n_test)
    return np.array([abs(_predict(tv, 500_000 + i, drift=drift) - tv)
                     for i, tv in enumerate(test_true)])


def coverage_vs_nominal(nominals, cal_res: np.ndarray, errs: np.ndarray):
    """Empirical coverage at each nominal level, given calibration residuals
    and test absolute errors (coverage = share of |err| <= q_nominal)."""
    out = []
    for nom in nominals:
        q = float(np.quantile(cal_res, nom))
        out.append(float(np.mean(errs <= q)))
    return out


def coverage_vs_drift(drifts, cal_res: np.ndarray, nominal: float = 0.90,
                      n_test: int = 1000, seed: int = 0):
    q = float(np.quantile(cal_res, nominal))
    return [float(np.mean(_test_errors(d, n_test=n_test, seed=seed) <= q))
            for d in drifts]


# ------------------------------------------------------------ audit latency ---

def audit_latency(ns, repeats: int = 25):
    """Median wall-clock (ms) of the citation-invariant audit for a narrative
    of n value-affecting claims, built by cycling the pilot's real claims over
    the real ledger (so every timed check does real anchor resolution, hash
    recomputation, and figure recomputation)."""
    res = value_property()
    base = [c for c in res.claims]
    out = []
    for n in ns:
        claims = [Claim(base[i % len(base)].text, base[i % len(base)].cites,
                        base[i % len(base)].kind, base[i % len(base)].value)
                  for i in range(n)]
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = audit_citation_invariant(claims, res.ledger, res.subject_attrs)
            times.append((time.perf_counter() - t0) * 1e3)
            assert r.ok
        out.append(float(np.median(times)))
    return out


# ----------------------------------------------------------------- run all ---

def run_all(n_cal: int = 300, n_test: int = 1000, seed: int = 0):
    cal_res = _calibration_residuals(n_cal=n_cal, seed=seed)
    errs_exch = _test_errors(0.0, n_test=n_test, seed=seed)
    errs_drift = _test_errors(0.05, n_test=n_test, seed=seed)

    nominals = [round(x, 2) for x in np.arange(0.50, 0.99, 0.05)]
    drifts = [round(x, 3) for x in np.arange(0.0, 0.081, 0.01)]
    ns = [4, 8, 16, 24, 32, 48, 64]

    return {
        "nominals": nominals,
        "coverage_exchangeable": coverage_vs_nominal(nominals, cal_res, errs_exch),
        "coverage_drift5": coverage_vs_nominal(nominals, cal_res, errs_drift),
        "drifts": drifts,
        "coverage_at_90_vs_drift": coverage_vs_drift(drifts, cal_res,
                                                     n_test=n_test, seed=seed),
        "n_claims": ns,
        "audit_latency_ms": audit_latency(ns),
        "n_cal": n_cal, "n_test": n_test, "seed": seed,
    }


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
