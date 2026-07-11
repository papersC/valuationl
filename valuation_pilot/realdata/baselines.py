"""Accuracy baselines for the real-data study (referee comment M3).

Three cheap, standard AVM baselines are evaluated under the *same* strict
walk-forward protocol as the hedonic sales-comparison estimator, so the headline
MdAPE can be read against an alternative rather than against the deployment bar
alone:

  * ``segment_median_wf`` -- a naive comparable: predict the subject's unit price
    as the median unit price of its market segment over the expanding training
    window, times its area.
  * ``index_baseline`` -- the registry's published segment index applied to the
    subject (the most recent segment level carried forward by trailing growth).
  * ``knn_baseline`` -- a k-nearest-neighbour comparables model on standardised
    attributes, predicting the median unit price of the k nearest prior sales.

All three use only data strictly before each test quarter (no leakage). The
point of the comparison is to show whether the pipeline's estimator is good or
the market is merely predictable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .hedonic import fit_hedonic
from .index import QuarterlyIndex


def _quarters_of_year(df, year):
    return sorted(df[df["year"] == year]["quarter"].unique(),
                 key=lambda s: pd.Period(s, freq="Q"))


def _pack(P, A, extra=None):
    d = {"n_test": int(len(A)), "mdape": M.mdape(P, A),
         "within_10pct": M.share_within(P, A), "cod": M.cod(P, A), "prd": M.prd(P, A)}
    if extra:
        d.update(extra)
    return d


def segment_median_wf(df, test_year):
    """Walk-forward segment-median comparable."""
    qp_all = pd.PeriodIndex(df["quarter"], freq="Q")
    P, A = [], []
    for q in _quarters_of_year(df, test_year):
        qpp = pd.Period(q, freq="Q")
        tr = df[qp_all < qpp]
        te = df[(df["year"] == test_year) & (df["quarter"] == q)]
        med = tr.groupby("segment")["unit_price"].median()
        gmed = float(tr["unit_price"].median())
        up = te["segment"].map(med).fillna(gmed).to_numpy(float)
        P.append(up * te["gross_sqft"].to_numpy(float))
        A.append(te["price"].to_numpy(float))
    return _pack(np.concatenate(P), np.concatenate(A))


def index_baseline(df, test_year):
    """Registry segment index applied to the subject (causal: past levels only)."""
    idx = QuarterlyIndex(df)
    te = df[df["year"] == test_year]
    lvl = te.apply(lambda r: idx.level(r["segment"], r["quarter"]), axis=1).to_numpy(float)
    g = te.apply(lambda r: idx.trailing_growth(r["segment"], r["quarter"]), axis=1).to_numpy(float)
    val = lvl * (1.0 + g) * te["gross_sqft"].to_numpy(float)
    a = te["price"].to_numpy(float)
    m = ~np.isnan(val)
    return _pack(val[m], a[m], {"n_skipped_no_index": int((~m).sum())})


def _knn_features(df, base_period, nbhd_te, global_te):
    d = pd.DataFrame(index=df.index)
    d["log_gross_sqft"] = np.log(df["gross_sqft"].clip(lower=1).to_numpy(float))
    d["res_units"] = df["res_units"].to_numpy(float)
    q = pd.PeriodIndex(df["quarter"], freq="Q")
    d["time_idx"] = np.array([(p - base_period).n for p in q], dtype=float)
    d["nbhd_te"] = df["neighborhood"].map(nbhd_te).fillna(global_te).to_numpy(float)
    d["is_villa"] = (df["category"].astype(str) == "Villa").astype(float).to_numpy()
    return d


def knn_baseline(df, test_year, k=10, seed=0):
    """Walk-forward k-NN comparables model on standardised attributes."""
    from sklearn.neighbors import NearestNeighbors
    qp_all = pd.PeriodIndex(df["quarter"], freq="Q")
    P, A = [], []
    for q in _quarters_of_year(df, test_year):
        qpp = pd.Period(q, freq="Q")
        tr = df[qp_all < qpp]
        te = df[(df["year"] == test_year) & (df["quarter"] == q)]
        base = pd.PeriodIndex(tr["quarter"], freq="Q").min()
        log_up = np.log(tr["unit_price"].clip(lower=1).to_numpy(float))
        global_te = float(np.mean(log_up))
        g = pd.DataFrame({"key": tr["neighborhood"].to_numpy(), "y": log_up}).groupby("key")["y"]
        nbhd_te = ((g.mean() * g.count() + global_te * 20.0) / (g.count() + 20.0)).to_dict()
        Xtr = _knn_features(tr, base, nbhd_te, global_te).to_numpy(float)
        Xte = _knn_features(te, base, nbhd_te, global_te).to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        up_tr = tr["unit_price"].to_numpy(float)
        nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(Xtr)
        _, ind = nn.kneighbors(Xte)
        up_pred = np.median(up_tr[ind], axis=1)
        P.append(up_pred * te["gross_sqft"].to_numpy(float))
        A.append(te["price"].to_numpy(float))
    return _pack(np.concatenate(P), np.concatenate(A), {"k": k})


def hedonic_wf(df, test_year, seed=0):
    """Recompute the pipeline's own walk-forward accuracy (reproduces the headline)."""
    qp_all = pd.PeriodIndex(df["quarter"], freq="Q")
    P, A = [], []
    for q in _quarters_of_year(df, test_year):
        qpp = pd.Period(q, freq="Q")
        tr = df[qp_all < qpp]
        te = df[(df["year"] == test_year) & (df["quarter"] == q)]
        model = fit_hedonic(tr, seed=seed)
        P.append(model.predict_price(te))
        A.append(te["price"].to_numpy(float))
    return _pack(np.concatenate(P), np.concatenate(A))
