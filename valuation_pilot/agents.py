"""The three specialist valuer agents (interface I2).

Each agent implements one professionally recognised approach to value and emits
an Estimate: a value, an interval, a reliability weight in [0,1], cited
adjustments, and the evidence identifiers it used. The sales-comparison value is
a deterministic function of the cited comparables and index, so the auditor can
recompute it exactly when checking entailment (audit.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from .evidence import Evidence, EvidenceStores


@dataclass
class Adjustment:
    factor: str
    magnitude: float
    cite: str


@dataclass
class Estimate:
    appr: Literal["SC", "Inc", "Cost"]
    value: float
    interval: tuple[float, float]
    reliability: float
    adjustments: list[Adjustment] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


def sc_value_from(comps: list[Evidence], growth: float, size: float) -> float:
    """Deterministic sales-comparison value: median time-adjusted unit price x size."""
    units = np.array([c.payload["unit_price"] * (1.0 + growth) for c in comps], dtype=float)
    return float(np.median(units) * size)


class SalesComparisonAgent:
    appr = "SC"

    def estimate(self, stores: EvidenceStores, subject_attrs: dict,
                 comps: Optional[list[Evidence]] = None) -> Estimate:
        if comps is None:
            comps = stores.comps.comparables(subject_attrs["dwelling_type"])
        idx = stores.index.latest_growth()
        growth = idx.payload["trailing_quarter_growth"] if idx else 0.0
        size = subject_attrs["built_up_area"]

        value = sc_value_from(comps, growth, size)
        units = np.array([c.payload["unit_price"] * (1.0 + growth) for c in comps], dtype=float)
        eids = [c.eid for c in comps] + ([idx.eid] if idx else [])
        adjustments = [Adjustment("time_to_valuation_date", growth, idx.eid)] if idx else []

        rel_disp = float(np.std(units) / np.mean(units)) if len(units) else 1.0
        half = value * max(0.02, min(0.08, rel_disp * 1.5))
        reliability = float(np.clip(0.9 - rel_disp * 4 + 0.02 * len(comps), 0.3, 0.95))
        return Estimate("SC", value, (value - half, value + half),
                        round(reliability, 2), adjustments, eids)


class IncomeAgent:
    appr = "Inc"

    def estimate(self, stores: EvidenceStores, subject_attrs: dict) -> Estimate:
        leases = [e for e in stores.leases.leases(subject_attrs["dwelling_type"])
                  if "annual_rent" in e.payload]
        yield_ev = next((e for e in stores.leases.all() if "gross_yield" in e.payload), None)
        rents = np.array([e.payload["annual_rent"] for e in leases], dtype=float)
        market_rent = float(np.median(rents)) if len(rents) else 0.0
        gross_yield = yield_ev.payload["gross_yield"] if yield_ev else 0.065
        value = market_rent / gross_yield if gross_yield else 0.0
        eids = [e.eid for e in leases] + ([yield_ev.eid] if yield_ev else [])
        adj = [Adjustment("capitalisation_yield", gross_yield,
                          yield_ev.eid if yield_ev else "yield:y1")]
        rel_disp = float(np.std(rents) / np.mean(rents)) if len(rents) else 1.0
        half = value * max(0.03, min(0.10, rel_disp * 2 + 0.03))
        reliability = float(np.clip(0.7 - rel_disp * 3, 0.35, 0.75))
        return Estimate("Inc", value, (value - half, value + half),
                        round(reliability, 2), adj, eids)


class CostAgent:
    appr = "Cost"

    def estimate(self, stores: EvidenceStores, subject_attrs: dict) -> Estimate:
        cost_ev = next((e for e in stores.attrs.all() if "replacement_cost" in e.payload), None)
        if cost_ev is None:
            return Estimate("Cost", 0.0, (0.0, 0.0), 0.0, [], [])
        p = cost_ev.payload
        value = p["land_value"] + p["replacement_cost"] * (1.0 - p["depreciation"])
        half = value * 0.10
        return Estimate("Cost", value, (value - half, value + half), 0.15,
                        [Adjustment("depreciation", p["depreciation"], cost_ev.eid)],
                        [cost_ev.eid])


def run_all_agents(stores: EvidenceStores, subject_attrs: dict,
                   sc_comps: Optional[list[Evidence]] = None) -> list[Estimate]:
    return [
        SalesComparisonAgent().estimate(stores, subject_attrs, comps=sc_comps),
        IncomeAgent().estimate(stores, subject_attrs),
        CostAgent().estimate(stores, subject_attrs),
    ]
