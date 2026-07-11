"""Checkpointed Dubai run of the real-data experiment.

The sandbox caps each shell call at ~45 s, so this runner executes exactly one
bounded stage per invocation and persists partial results to
/tmp/dld/run_state.json. Stages replicate experiment.run() faithfully
(same code paths: fit_hedonic, metrics, value_real, verify, reproduce), with
the walk-forward, drift, and structural stages split into per-fit units.

Usage:  python -m valuation_pilot.realdata.run_dubai        # next stage
        python -m valuation_pilot.realdata.run_dubai status
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .experiment import (TARGETS, _comps_for, _quarters_of_year, attack_demo,
                         audit_latency, conformal_main)
from .hedonic import fit_hedonic
from .index import QuarterlyIndex
from .loader_dubai import load_sales_dubai
from .pipeline import value_real
from ..pipeline import reproduce_from_ledger, verify

STATE = "/tmp/dld/run_state.json"
OUT = "/tmp/dld/realdata_results_dubai.json"
SEED = 0
TEST_YEAR = 2025
N_PIPE_TOTAL = 1000
PIPE_BATCH = 250


def _load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"done_stages": [], "wf": [], "drift": [], "pipe": None, "results": {}}


def _save_state(st):
    json.dump(st, open(STATE, "w"))


def _wf_one_quarter(df, q, seed=SEED):
    qp = pd.Period(q, freq="Q")
    train = df[pd.PeriodIndex(df["quarter"], freq="Q") < qp]
    test = df[(df["year"] == TEST_YEAR) & (df["quarter"] == q)]
    model = fit_hedonic(train, seed=seed)
    p = model.predict_price(test)
    a = test["price"].to_numpy(float)
    np.savez(f"/tmp/dld/wf_{q}.npz", p=p, a=a)
    return {"quarter": q, "n": int(len(a)), "n_train": int(len(train)),
            "mdape": M.mdape(p, a), "within_10pct": M.share_within(p, a),
            "cod": M.cod(p, a), "prd": M.prd(p, a)}


def _drift_one(df, Y, seed=SEED):
    train = df[df["year"] < Y]
    cal = df[df["year"] == Y]
    test = df[df["year"] == Y + 1]
    if len(train) < 500 or len(cal) < 100 or len(test) < 100:
        return None
    model = fit_hedonic(train, seed=seed)
    q = M.conformal_q(model.predict_price(cal), cal["price"].to_numpy(float), 0.10)
    cov = M.coverage(model.predict_price(test), test["price"].to_numpy(float), q)
    return {"calibrate_year": int(Y), "test_year": int(Y + 1),
            "n_cal": int(len(cal)), "n_test": int(len(test)), "coverage": cov}


def _pipe_batch(df, st, seed=SEED):
    """Run up to PIPE_BATCH sampled valuations through the full audit pipeline."""
    if st["pipe"] is None:
        rng = np.random.default_rng(seed)
        cand = df[df["year"] == TEST_YEAR].index.to_numpy().copy()
        rng.shuffle(cand)
        st["pipe"] = {"order": cand.tolist(), "pos": 0, "ran": 0,
                      "contain_ok": 0, "audit_ok": 0, "selection_ok": 0,
                      "replay_ok": 0, "verify_ok": 0, "skipped_no_comps": 0}
    P = st["pipe"]
    idx_model = QuarterlyIndex(df)
    # per-segment cache: presorted subframes with a precomputed quarter Period,
    # so each sampled valuation slices a few hundred rows instead of scanning 158k
    qp_all = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.assign(_qp=qp_all)
    seg_groups = {s: g.sort_values("sale_date") for s, g in df.groupby("segment")}

    def _comps_fast(seg, qp, k=15):
        g = seg_groups.get(seg)
        if g is None:
            return df.iloc[0:0]
        return g[g["_qp"] < qp].tail(k)

    done_in_call = 0
    t0 = time.time()
    while P["ran"] < N_PIPE_TOTAL and P["pos"] < len(P["order"]):
        if done_in_call >= PIPE_BATCH or time.time() - t0 > 30:
            break
        i = P["order"][P["pos"]]; P["pos"] += 1
        row = df.loc[i]
        seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = _comps_fast(seg, qp, k=15)
        comps = comps[comps["sid"] != row["sid"]]
        if len(comps) < 3:
            P["skipped_no_comps"] += 1
            continue
        growth = idx_model.trailing_growth(seg, row["quarter"])
        res = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth)
        rep = verify(res.certificate, res.ledger)
        P["ran"] += 1; done_in_call += 1
        lo, hi = res.reconciled.approach_range
        P["contain_ok"] += int(lo - 1e-6 <= res.reconciled.value <= hi + 1e-6)
        P["audit_ok"] += int(res.audit.ok)
        P["selection_ok"] += int(res.selection.ok)
        P["replay_ok"] += int(reproduce_from_ledger(res.certificate, res.ledger))
        P["verify_ok"] += int(rep.ok)
    return P["ran"] >= N_PIPE_TOTAL or P["pos"] >= len(P["order"])


def finalize(df, load, st):
    wf = st["wf"]
    P_all = [np.load(f"/tmp/dld/wf_{w['quarter']}.npz") for w in wf]
    P = np.concatenate([z["p"] for z in P_all])
    A = np.concatenate([z["a"] for z in P_all])
    acc = {"n_test": int(len(A)), "mdape": M.mdape(P, A),
           "within_10pct": M.share_within(P, A), "cod": M.cod(P, A),
           "prd": M.prd(P, A), "by_stratum": M.by_strata(P, A, k=3),
           "by_quarter": [{k: v for k, v in w.items() if k != "n_train"} for w in wf]}
    conf = st["results"]["conformal"]
    drift = st["drift"]
    pipe_s = st["pipe"]
    pipe = {"n_run": pipe_s["ran"], "skipped_no_comps": pipe_s["skipped_no_comps"],
            "containment_ok": pipe_s["contain_ok"], "citation_audit_ok": pipe_s["audit_ok"],
            "selection_audit_ok": pipe_s["selection_ok"], "replay_ok": pipe_s["replay_ok"],
            "verify_ok": pipe_s["verify_ok"],
            "all_structural_hold": (pipe_s["contain_ok"] == pipe_s["ran"]
                                    and pipe_s["audit_ok"] == pipe_s["ran"]
                                    and pipe_s["replay_ok"] == pipe_s["ran"]
                                    and pipe_s["verify_ok"] == pipe_s["ran"])}

    def score(metric, val):
        if metric == "prd":
            return bool(TARGETS["prd_lo"][1] <= val <= TARGETS["prd_hi"][1])
        if metric == "coverage":
            return bool(TARGETS["coverage_lo"][1] <= val <= TARGETS["coverage_hi"][1])
        op, tgt = TARGETS[metric]
        return bool(val <= tgt) if op == "<=" else bool(val >= tgt)

    tva = {
        "mdape": {"target": "<= 0.07", "achieved": round(acc["mdape"], 4), "met": score("mdape", acc["mdape"])},
        "within_10pct": {"target": ">= 0.70", "achieved": round(acc["within_10pct"], 4), "met": score("within_10pct", acc["within_10pct"])},
        "cod": {"target": "<= 15", "achieved": round(acc["cod"], 2), "met": score("cod", acc["cod"])},
        "prd": {"target": "0.98-1.03", "achieved": round(acc["prd"], 4), "met": score("prd", acc["prd"])},
        "coverage_at_90": {"target": "0.88-0.92", "achieved": round(conf["coverage"], 4), "met": score("coverage", conf["coverage"])},
    }
    meta = json.load(open("/tmp/dld/build_meta.json"))
    df_years = sorted(int(y) for y in df["year"].unique())
    result = {
        "dataset": {
            "source": "Dubai Land Department registered sale transactions (open data, Dubai open-data platform; snapshot 2026-05-29)",
            "note": ("Existing-market (`Sell`) residential resales of the two unambiguous "
                     "dwelling classes (Flat, Villa). The bulk export is an append-log with "
                     "7 snapshot copies per transaction; the cascade dedups on the registry's "
                     "transaction_id first. The feed carries no build year, land area, rents, "
                     "or replacement-cost fields: age/land features are constant, and the "
                     "income and cost approaches are n/a for this evaluation, so the "
                     "sales-comparison estimator carries the valuation. Prices in AED."),
            "files": [{"name": n, "sha256": s, "rows_raw": r} for (n, s, r) in load.files],
            "filter_counts": load.counts_dict(),
            "n_clean": int(len(df)),
            "years": df_years,
            "boroughs": sorted(df["borough"].unique().tolist()),
            "areas": int(df["neighborhood"].nunique()),
            "projects_nonnull": int((df["zip"] != "NONE").sum()),
            "flat_share": round(float((df["category"] == "Flat").mean()), 4),
            "segments": int(df["segment"].nunique()),
            "price_median": float(df["price"].median()),
            "unit_price_median": float(df["unit_price"].median()),
            "currency": "AED",
        },
        "point_accuracy_walkforward": acc,
        "interval_calibration": conf,
        "drift": drift,
        "targets_vs_achieved": tva,
        "pipeline_structural": pipe,
        "selection_attack": st["results"]["attack"],
        "audit_latency": st["results"]["latency"],
        "config": {"test_year": TEST_YEAR, "n_pipeline": pipe["n_run"], "seed": SEED,
                   "model": "HistGradientBoostingRegressor(log unit-price)"},
        "runtime_s": st["results"].get("elapsed_s", None),
    }
    json.dump(result, open(OUT, "w"), indent=2)
    return result


def main():
    t0 = time.time()
    st = _load_state()
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps({"done": st["done_stages"],
                          "wf_done": [w["quarter"] for w in st["wf"]],
                          "drift_done": [d["calibrate_year"] for d in st["drift"]],
                          "pipe": (st["pipe"] or {}).get("ran")}))
        return

    load = load_sales_dubai()
    df = load.df

    quarters = _quarters_of_year(df, TEST_YEAR)
    for q in quarters:
        if not any(w["quarter"] == q for w in st["wf"]):
            r = _wf_one_quarter(df, q)
            st["wf"].append(r)
            _save_state(st)
            print(json.dumps({"stage": f"wf_{q}", **r})); return

    if "conformal" not in st["results"]:
        st["results"]["conformal"] = conformal_main(df, TEST_YEAR, seed=SEED)
        _save_state(st)
        print(json.dumps({"stage": "conformal", **st["results"]["conformal"]})); return

    for Y in range(2020, TEST_YEAR):
        if not any(d["calibrate_year"] == Y for d in st["drift"]):
            r = _drift_one(df, Y)
            if r:
                st["drift"].append(r)
            _save_state(st)
            print(json.dumps({"stage": f"drift_{Y}", **(r or {})})); return

    if (st["pipe"] is None) or (st["pipe"]["ran"] < N_PIPE_TOTAL
                                and st["pipe"]["pos"] < len(st["pipe"]["order"])):
        done = _pipe_batch(df, st)
        _save_state(st)
        print(json.dumps({"stage": "pipeline", "ran": st["pipe"]["ran"],
                          "skipped": st["pipe"]["skipped_no_comps"], "complete": done})); return

    if "attack" not in st["results"]:
        st["results"]["attack"] = attack_demo(df, TEST_YEAR, seed=SEED)
        _save_state(st)
        print(json.dumps({"stage": "attack", **st["results"]["attack"]})); return

    if "latency" not in st["results"]:
        st["results"]["latency"] = audit_latency(df, TEST_YEAR, seed=SEED)
        _save_state(st)
        print(json.dumps({"stage": "latency", "ms_per_claim": st["results"]["latency"]["ms_per_claim"]})); return

    res = finalize(df, load, st)
    st["done_stages"].append("finalized")
    _save_state(st)
    print(json.dumps({"stage": "FINAL", "out": OUT,
                      "mdape": res["point_accuracy_walkforward"]["mdape"],
                      "within_10pct": res["point_accuracy_walkforward"]["within_10pct"],
                      "cod": res["point_accuracy_walkforward"]["cod"],
                      "prd": res["point_accuracy_walkforward"]["prd"],
                      "coverage": res["interval_calibration"]["coverage"],
                      "pipe": res["pipeline_structural"],
                      "secs": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
