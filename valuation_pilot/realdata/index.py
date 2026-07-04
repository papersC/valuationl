"""Quarterly median-ratio price index per market segment.

For each segment (borough x dwelling category) the index is the median unit
price in each quarter; the trailing-quarter growth is the log-change from the
previous quarter, used exactly as the paper's `trailing_quarter_growth` to
time-adjust comparables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class QuarterlyIndex:
    def __init__(self, df: pd.DataFrame):
        # median unit price per (segment, quarter), on a monotone quarter axis
        g = (df.groupby(["segment", "quarter"])["unit_price"]
               .median().reset_index())
        g["q"] = pd.PeriodIndex(g["quarter"], freq="Q")
        self._med = {}       # (segment, Period) -> median unit price
        for _, r in g.iterrows():
            self._med[(r["segment"], r["q"])] = float(r["unit_price"])
        self._segments = sorted(df["segment"].unique())

    def trailing_growth(self, segment: str, quarter: str) -> float:
        """Log growth of the segment median from the previous quarter to the one
        BEFORE `quarter` (only past data, so it is causal for a sale in
        `quarter`)."""
        q = pd.Period(quarter, freq="Q")
        prev, prev2 = q - 1, q - 2
        a = self._med.get((segment, prev))
        b = self._med.get((segment, prev2))
        if a is None or b is None or b <= 0:
            return 0.0
        return float(np.log(a / b))

    def level(self, segment: str, quarter: str) -> float:
        q = pd.Period(quarter, freq="Q")
        for back in range(0, 8):
            v = self._med.get((segment, q - 1 - back))
            if v is not None:
                return v
        return float("nan")
