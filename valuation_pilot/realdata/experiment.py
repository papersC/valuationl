"""Measured accuracy on real transactions -- the full Part B experiment.

Pipeline:
  1. Load + clean real residential resales (loader).
  2. Point accuracy under WALK-FORWARD evaluation on the final held-out year
     (expanding-window retrain each quarter; no random shuffle): MdAPE, share
     within +/-10%, COD and PRD overall / per price stratum / per quarter.
  3. Split-conformal interval calibration at nominal 90%, calibrated on the
     prior year (disjoint from training).
  4. Real drift: calibrate in year Y, test in Y+1 without recalibration.
  5. Full audit pipeline (containment, citation audit, selection audit,
     certificate replay, tamper) on 1,000 sampled real valuations, plus a
     cherry-pick attack demo and audit-latency-vs-length.

Every number is written to results/realdata_results.json; misses are reported
as misses (no tuning, no reroll).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .hedonic import fit_hedonic
from .index import QuarterlyIndex
from .loader import load_sales
from .pipeline import value_real
from ..pipeline import reproduce_from_ledger, verify

TARGETS = {
    "mdape": ("<=", 0.07),
    "within_10pct": (">=", 0.70),
    "cod": ("<=", 15.0),
    "prd_lo": (">=", 0.98), "prd_hi": ("<=", 1.03),
    "coverage_lo": (">=", 0.88), "coverage_hi": ("<=", 0.92),
}


def _quarters_of_year(df, year):
    q = sorted(df[df["year"] == year]["quarter"].unique(),
               key=lambda s: pd.Period(s, freq="Q"))
    return q


def walk_forward_accuracy(df, test_year, seed=0):
    """Expanding-window retrain per test quarter; pooled + per-quarter + strata."""
    preds, actuals, prices, quarters = [], [], [], []
    per_quarter = []
    for q in _quarters_of_year(df, test_year):
        qp = pd.Period(q, freq="Q")
        train = df[pd.PeriodIndex(df["quarter"], freq="Q") < qp]
        test = df[(df["year"] == test_year) & (df["quarter"] == q)]
        if len(train) < 500 or len(test) < 20:
            continue
        model = fit_hedonic(train, seed=seed)
        p = model.predict_price(test)
        a = test["price"].to_numpy(float)
        preds.append(p); actuals.append(a); prices.append(a); quarters += [q] * len(a)
        per_quarter.append({
            "quarter": q, "n": int(len(a)),
            "mdape": M.mdape(p, a), "within_10pct": M.share_within(p, a),
            "cod": M.cod(p, a), "prd": M.prd(p, a),
        })
    P = np.concatenate(preds); A = np.concatenate(actuals)
    return {
        "n_test": int(len(A)),
        "mdape": M.mdape(P, A),
        "within_10pct": M.share_within(P, A),
        "cod": M.cod(P, A),
        "prd": M.prd(P, A),
        "by_stratum": M.by_strata(P, A, k=3),
        "by_quarter": per_quarter,
    }


def conformal_main(df, test_year, alpha=0.10, seed=0):
    """Train < (test_year-1), calibrate on the prior year (disjoint), test the
    final year. Clean split-conformal coverage at nominal 90%."""
    cal_year = test_year - 1
    train = df[df["year"] < cal_year]
    cal = df[df["year"] == cal_year]
    test = df[df["year"] == test_year]
    model = fit_hedonic(train, seed=seed)
    q = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), alpha)
    cov = M.coverage(model.predict_price(test), test["price"].to_numpy(float), q)
    return {"nominal": 1 - alpha, "cal_year": int(cal_year), "test_year": int(test_year),
            "n_cal": int(len(cal)), "n_test": int(len(test)),
            "half_width_log": round(q, 4), "coverage": cov}


def drift_experiment(df, years, alpha=0.10, seed=0):
    """Calibrate in year Y, test Y+1 without recalibration (train strictly < Y so
    the calibration set is unseen -> valid conformal). Reports coverage per pair;
    a monotone fall with market drift is the real-data analogue of Fig. 2b."""
    out = []
    for Y in years[:-1]:
        Yn = Y + 1
        if Yn not in years:
            continue
        train = df[df["year"] < Y]
        cal = df[df["year"] == Y]
        test = df[df["year"] == Yn]
        if len(train) < 500 or len(cal) < 100 or len(test) < 100:
            continue
        model = fit_hedonic(train, seed=seed)
        q = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), alpha)
        cov = M.coverage(model.predict_price(test), test["price"].to_numpy(float), q)
        out.append({"calibrate_year": int(Y), "test_year": int(Yn),
                    "n_cal": int(len(cal)), "n_test": int(len(test)),
                    "coverage": cov})
    return out


def _comps_for(df_hist, seg, qp, k=15):
    hist = df_hist[(df_hist["segment"] == seg)
                   & (pd.PeriodIndex(df_hist["quarter"], freq="Q") < qp)]
    return hist.sort_values("sale_date").tail(k)


def pipeline_on_real(df, test_year, n_sample=1000, seed=0):
    """Run the deterministic audit pipeline on n_sample real subjects; confirm
    the structural guarantees hold on real evidence."""
    rng = np.random.default_rng(seed)
    idx_model = QuarterlyIndex(df)
    test = df[df["year"] == test_year]
    hist = df  # comparables may come from any prior quarter (temporal filter in _comps_for)
    cand = test.index.to_numpy()
    rng.shuffle(cand)

    ran = 0; contain_ok = 0; audit_ok = 0; selection_ok = 0; replay_ok = 0; verify_ok = 0
    skipped_no_comps = 0
    for i in cand:
        if ran >= n_sample:
            break
        row = df.loc[i]
        seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = _comps_for(hist, seg, qp, k=15)
        comps = comps[comps["sid"] != row["sid"]]
        if len(comps) < 3:
            skipped_no_comps += 1
            continue
        growth = idx_model.trailing_growth(seg, row["quarter"])
        res = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth)
        rep = verify(res.certificate, res.ledger)
        ran += 1
        lo, hi = res.reconciled.approach_range
        contain_ok += int(lo - 1e-6 <= res.reconciled.value <= hi + 1e-6)
        audit_ok += int(res.audit.ok)
        selection_ok += int(res.selection.ok)
        replay_ok += int(reproduce_from_ledger(res.certificate, res.ledger))
        verify_ok += int(rep.ok)
    return {
        "n_run": ran, "skipped_no_comps": int(skipped_no_comps),
        "containment_ok": contain_ok, "citation_audit_ok": audit_ok,
        "selection_audit_ok": selection_ok, "replay_ok": replay_ok,
        "verify_ok": verify_ok,
        "all_structural_hold": (contain_ok == ran and audit_ok == ran
                                and replay_ok == ran and verify_ok == ran),
    }


def attack_demo(df, test_year, seed=0):
    """A cherry-picked comparable set passes citation+replay but is caught by the
    completeness (selection) audit -- threat T6, on real evidence."""
    idx_model = QuarterlyIndex(df)
    test = df[df["year"] == test_year]
    for i in test.index:
        row = df.loc[i]; seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = _comps_for(df, seg, qp, k=15)
        comps = comps[comps["sid"] != row["sid"]]
        if len(comps) < 6:
            continue
        growth = idx_model.trailing_growth(seg, row["quarter"])
        honest = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth, attack=False)
        cherry = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth, attack=True)
        return {
            "honest_citation_ok": honest.audit.ok, "honest_selection_ok": honest.selection.ok,
            "cherry_citation_ok": cherry.audit.ok,          # passes citation invariant
            "cherry_selection_flagged": not cherry.selection.ok,   # caught by completeness
            "cherry_value_impact": round(cherry.selection.value_impact, 4),
        }
    return {}


def audit_latency(df, test_year, lengths=(4, 8, 16, 24, 32, 48, 64), reps=40, seed=0):
    """Per-claim audit cost: replicate a real certificate's claims to length n
    and time the citation-invariant audit. Expected ~linear (~0.4 ms/claim)."""
    from ..audit import audit_citation_invariant
    from ..narrative import Claim
    idx_model = QuarterlyIndex(df)
    test = df[df["year"] == test_year]
    row = None
    for i in test.index:
        r = df.loc[i]; qp = pd.Period(r["quarter"], freq="Q")
        comps = _comps_for(df, r["segment"], qp, k=15)
        comps = comps[comps["sid"] != r["sid"]]
        if len(comps) >= 4:
            row = r; break
    growth = idx_model.trailing_growth(row["segment"], row["quarter"])
    res = value_real(row["sid"], row["segment"], float(row["gross_sqft"]),
                     _comps_for(df, row["segment"], pd.Period(row["quarter"], freq="Q"), 15), growth)
    base = res.claims
    out = []
    for n in lengths:
        claims = [Claim(c.text, c.cites, c.kind, c.value)
                  for c in (base * (n // len(base) + 1))[:n]]
        t0 = time.perf_counter()
        for _ in range(reps):
            audit_citation_invariant(claims, res.ledger, res.subject_attrs)
        out.append({"n_claims": n, "ms": round((time.perf_counter() - t0) / reps * 1e3, 4)})
    slope = (out[-1]["ms"] - out[0]["ms"]) / (out[-1]["n_claims"] - out[0]["n_claims"])
    return {"points": out, "ms_per_claim": round(slope, 4)}


def run(datadir="realdata_src", out_path="results/realdata_results.json",
        n_pipeline=1000, seed=0):
    t0 = time.time()
    load = load_sales(datadir)
    df = load.df
    years = sorted(int(y) for y in df["year"].unique())
    test_year = years[-1]

    acc = walk_forward_accuracy(df, test_year, seed=seed)
    conf = conformal_main(df, test_year, seed=seed)
    drift = drift_experiment(df, years, seed=seed)
    pipe = pipeline_on_real(df, test_year, n_sample=n_pipeline, seed=seed)
    attack = attack_demo(df, test_year, seed=seed)
    latency = audit_latency(df, test_year, seed=seed)

    def score(metric, val):
        if metric == "prd":
            return bool(TARGETS["prd_lo"][1] <= val <= TARGETS["prd_hi"][1])
        if metric == "coverage":
            return bool(TARGETS["coverage_lo"][1] <= val <= TARGETS["coverage_hi"][1])
        op, tgt = TARGETS[metric]
        return bool(val <= tgt) if op == "<=" else bool(val >= tgt)

    targets_vs_achieved = {
        "mdape": {"target": "<= 0.07", "achieved": round(acc["mdape"], 4), "met": score("mdape", acc["mdape"])},
        "within_10pct": {"target": ">= 0.70", "achieved": round(acc["within_10pct"], 4), "met": score("within_10pct", acc["within_10pct"])},
        "cod": {"target": "<= 15", "achieved": round(acc["cod"], 2), "met": score("cod", acc["cod"])},
        "prd": {"target": "0.98-1.03", "achieved": round(acc["prd"], 4), "met": score("prd", acc["prd"])},
        "coverage_at_90": {"target": "0.88-0.92", "achieved": round(conf["coverage"], 4), "met": score("coverage", conf["coverage"])},
    }

    result = {
        "dataset": {
            "source": "NYC DOF annualized residential sales (open data)",
            "note": ("Dubai DLD open data was requested but is gated behind a JS-only "
                     "portal / registered API and could not be fetched autonomously; the "
                     "source-agnostic loader accepts a DLD CSV drop-in. Income and cost "
                     "approaches are n/a for this feed (no rents / no replacement-cost "
                     "fields), so the sales-comparison estimator carries the valuation."),
            "files": [{"name": n, "sha256": s, "rows_raw": r} for (n, s, r) in load.files],
            "filter_counts": load.counts_dict(),
            "n_clean": int(len(df)),
            "years": years,
            "boroughs": sorted(df["borough"].unique().tolist()),
            "segments": int(df["segment"].nunique()),
            "price_median": float(df["price"].median()),
            "unit_price_median": float(df["unit_price"].median()),
        },
        "point_accuracy_walkforward": acc,
        "interval_calibration": conf,
        "drift": drift,
        "targets_vs_achieved": targets_vs_achieved,
        "pipeline_structural": pipe,
        "selection_attack": attack,
        "audit_latency": latency,
        "config": {"test_year": test_year, "n_pipeline": pipe["n_run"], "seed": seed,
                   "model": "HistGradientBoostingRegressor(log unit-price)"},
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({
        "n_clean": r["dataset"]["n_clean"],
        "mdape": r["point_accuracy_walkforward"]["mdape"],
        "within_10pct": r["point_accuracy_walkforward"]["within_10pct"],
        "cod": r["point_accuracy_walkforward"]["cod"],
        "prd": r["point_accuracy_walkforward"]["prd"],
        "coverage": r["interval_calibration"]["coverage"],
        "drift": r["drift"],
        "pipeline": r["pipeline_structural"],
        "attack": r["selection_attack"],
        "runtime_s": r["runtime_s"],
    }, indent=2))
