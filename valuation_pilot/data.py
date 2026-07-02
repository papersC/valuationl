"""Synthetic evidence generation for the pilot.

Deterministic given a seed. All figures are illustrative currency units (CU).
A full candidate comparable pool is anchored on the ledger; a valuation records
which comparables it used, so an auditor can later check the used set against
the anchored pool for selection bias (see audit.py, threat T6).
"""
from __future__ import annotations

import numpy as np

from .evidence import Evidence, EvidenceStores
from .ledger import Ledger


def build_stores(subject: str = "unit-2BR-110",
                 dwelling_type: str = "2BR-apartment",
                 n_pool: int = 8,
                 seed: int = 7) -> tuple[EvidenceStores, dict, list[str]]:
    """Anchor synthetic evidence for one subject.

    Returns (stores, subject_attributes, candidate_pool_comp_eids).
    """
    rng = np.random.default_rng(seed)
    ledger = Ledger()
    stores = EvidenceStores.empty(ledger)

    subject_attrs = {
        "dwelling_type": dwelling_type,
        "built_up_area": 110.0,
        "floor_level": "mid",
        "parking": 1,
        "tenure": "freehold",
    }
    stores.attrs.add(Evidence("attr:subject", "Attr", subject, "2026-06", subject_attrs))

    # candidate comparable pool (all anchored); unit prices near CU 18,500/m^2
    pool_eids = []
    for i in range(n_pool):
        unit_price = float(18500.0 + rng.normal(0, 800))
        stores.comps.add(Evidence(
            eid=f"comp:c{i+1}", kind="Comp", subject=subject, ts="2026-05",
            payload={"dwelling_type": dwelling_type,
                     "unit_price": round(unit_price, 2),
                     "size": round(float(rng.uniform(100, 122)), 1),
                     "floor": int(rng.integers(2, 20)),
                     "parking": int(rng.integers(0, 2))},
        ))
        pool_eids.append(f"comp:c{i+1}")

    stores.index.add(Evidence("idx:ix7", "Index", dwelling_type, "2026-06",
                              {"segment": dwelling_type, "trailing_quarter_growth": 0.018}))

    for i in range(3):
        rent = float(140_000 + rng.normal(0, 4_000))
        stores.leases.add(Evidence(f"lease:l{i+1}", "Lease", subject, "2026-04",
                                   {"dwelling_type": dwelling_type,
                                    "annual_rent": round(rent, 2)}))
    stores.leases.add(Evidence("yield:y1", "Lease", dwelling_type, "2026-06",
                               {"dwelling_type": dwelling_type, "gross_yield": 0.067}))

    stores.attrs.add(Evidence("cost:m1", "Attr", subject, "2026-01",
                              {"land_value": 900_000.0, "replacement_cost": 1_250_000.0,
                               "depreciation": 0.20}))

    return stores, subject_attrs, pool_eids
