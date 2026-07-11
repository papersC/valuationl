"""Gradient-boosted hedonic adjuster, fit on a temporal training window.

Predicts log unit-price (price per gross ft^2) from property attributes and
time; the sales-comparison value for a subject is the predicted unit price
times its area. This is the learned generalisation of the paper's
sales-comparison estimator: instead of a fixed median-of-comparables, the model
learns the comparable-implied price surface over the whole training market. It
plugs in behind the same interface (attributes in, a value out) as
`agents.sc_value_from`.

Features are all numeric so the model is robust across sklearn versions:
log gross/land area, age, residential units, a monotone quarter index, one-hot
borough and dwelling category, and a target-encoded neighborhood. The
neighborhood encoding and every fit use only the training window for a given
test period (temporal split -> no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Categorical one-hot levels default to the NYC codes; fit_hedonic derives the
# actual levels from the training frame, so any market (e.g. Dubai DLD, where
# borough is a single tier and category is Flat/Villa) plugs in unchanged.
BOROUGHS = ["1", "2", "3", "4", "5"]           # NYC borough codes (default)
CATEGORIES = ["01", "02", "03", "04", "12", "13", "15"]

NUM_BASE = ["log_gross_sqft", "log_land_sqft", "age", "res_units", "time_idx",
            "nbhd_te", "zip_te"]


def _base_period(df: pd.DataFrame) -> pd.Period:
    return pd.PeriodIndex(df["quarter"], freq="Q").min()


def _features(df: pd.DataFrame, base_period, nbhd_te: dict, zip_te: dict, global_te: float,
              boroughs=None, categories=None):
    boroughs = BOROUGHS if boroughs is None else boroughs
    categories = CATEGORIES if categories is None else categories
    d = pd.DataFrame(index=df.index)
    d["log_gross_sqft"] = np.log(df["gross_sqft"].clip(lower=1).to_numpy(float))
    d["log_land_sqft"] = np.log(df["land_sqft"].clip(lower=0).to_numpy(float) + 1.0)
    d["age"] = df["age"].to_numpy(float)
    d["res_units"] = df["res_units"].to_numpy(float)
    q = pd.PeriodIndex(df["quarter"], freq="Q")
    d["time_idx"] = np.array([(p - base_period).n for p in q], dtype=float)
    d["nbhd_te"] = df["neighborhood"].map(nbhd_te).fillna(global_te).to_numpy(float)
    d["zip_te"] = df["zip"].map(zip_te).fillna(global_te).to_numpy(float)
    for b in boroughs:
        d[f"boro_{b}"] = (df["borough"].astype(str) == b).astype(float).to_numpy()
    for c in categories:
        d[f"cat_{c}"] = (df["category"].astype(str) == c).astype(float).to_numpy()
    return d


def _columns(boroughs=None, categories=None):
    boroughs = BOROUGHS if boroughs is None else boroughs
    categories = CATEGORIES if categories is None else categories
    return NUM_BASE + [f"boro_{b}" for b in boroughs] + [f"cat_{c}" for c in categories]


@dataclass
class HedonicModel:
    model: HistGradientBoostingRegressor
    base_period: pd.Period
    nbhd_te: dict
    zip_te: dict
    global_te: float
    boroughs: list = None
    categories: list = None

    def _X(self, df):
        return _features(df, self.base_period, self.nbhd_te, self.zip_te,
                         self.global_te, self.boroughs,
                         self.categories)[_columns(self.boroughs, self.categories)]

    def predict_unit_price(self, df):
        return np.exp(self.model.predict(self._X(df)))

    def predict_price(self, df):
        return self.predict_unit_price(df) * df["gross_sqft"].to_numpy(float)


def fit_hedonic(train: pd.DataFrame, seed: int = 0) -> HedonicModel:
    base = _base_period(train)
    log_up = np.log(train["unit_price"].clip(lower=1).to_numpy(float))
    global_te = float(np.mean(log_up))
    # smoothed target encoding of neighborhood on the TRAIN fold only
    def _te(key_series, k):
        t = pd.DataFrame({"key": key_series.to_numpy(), "y": log_up})
        g = t.groupby("key")["y"]
        cnt, mean = g.count(), g.mean()
        return ((mean * cnt + global_te * k) / (cnt + k)).to_dict()

    nbhd_te = _te(train["neighborhood"], 20.0)
    zip_te = _te(train["zip"], 15.0)

    boroughs = sorted(train["borough"].astype(str).unique().tolist())
    categories = sorted(train["category"].astype(str).unique().tolist())
    X = _features(train, base, nbhd_te, zip_te, global_te,
                  boroughs, categories)[_columns(boroughs, categories)]
    y = log_up
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X, y)
    return HedonicModel(model=model, base_period=base, nbhd_te=nbhd_te,
                        zip_te=zip_te, global_te=global_te,
                        boroughs=boroughs, categories=categories)
