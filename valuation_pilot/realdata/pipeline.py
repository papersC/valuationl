"""Run the audit pipeline on ONE real subject with real comparable evidence.

Income and cost approaches are n/a for this market (the open registry feed
carries neither rents nor replacement-cost fields), so only the
sales-comparison estimator is active. The SC value uses the paper's exact
deterministic formula -- median time-adjusted comparable unit price times the
subject area -- so the auditor's structured-entailment recomputation matches
bit-for-bit; the reconciliation is therefore SC alone and the no-extrapolation
invariant holds trivially. This exercises containment, the citation invariant,
the selection (cherry-pick) audit, certificate replay, and tamper detection on
real evidence.
"""
from __future__ import annotations

import numpy as np

from ..agents import Estimate, SalesComparisonAgent
from ..audit import audit_citation_invariant, audit_selection
from ..evidence import Evidence, EvidenceStores
from ..ledger import Ledger, canonical_hash
from ..narrative import build_narrative
from ..pipeline import MODEL_VER, IDX_VER, STACK_VER, PilotResult
from ..reconcile import reconcile


def _residuals_sc(presented, growth, size, vpoint):
    return np.array([c.payload["unit_price"] * (1.0 + growth) * size - vpoint
                     for c in presented], dtype=float)


def build_real_stores(ledger, subject, segment, size, comps_rows, growth):
    """Anchor a subject, its comparable pool, and the price index on the ledger."""
    stores = EvidenceStores.empty(ledger)
    subject_attrs = {
        "dwelling_type": segment,
        "built_up_area": float(size),
        "floor_level": "na",
        "parking": 0,
        "tenure": "freehold",
    }
    stores.attrs.add(Evidence("attr:subject", "Attr", subject, "real", subject_attrs))
    pool_eids = []
    for _, c in comps_rows.iterrows():
        eid = f"comp:{c['sid']}"
        stores.comps.add(Evidence(eid, "Comp", subject, str(c["quarter"]), {
            "dwelling_type": segment,
            "unit_price": round(float(c["unit_price"]), 2),
            "size": round(float(c["gross_sqft"]), 1),
            "floor": 0, "parking": 0,
        }))
        pool_eids.append(eid)
    stores.index.add(Evidence("idx:seg", "Index", segment, "real",
                              {"segment": segment, "trailing_quarter_growth": round(float(growth), 6)}))
    return stores, subject_attrs, pool_eids


def value_real(subject, segment, size, comps_rows, growth, attack: bool = False,
               alpha: float = 0.10) -> PilotResult:
    ledger = Ledger()
    stores, subject_attrs, pool_eids = build_real_stores(
        ledger, subject, segment, size, comps_rows, growth)

    all_comps = stores.comps.comparables(segment)
    if attack:  # cherry-pick: present only the top-half priced comparables
        presented = sorted(all_comps, key=lambda c: c.payload["unit_price"],
                           reverse=True)[: max(1, len(all_comps) // 2)]
    else:
        presented = all_comps

    sc = SalesComparisonAgent().estimate(stores, subject_attrs, comps=presented)
    estimates = [sc]                       # income + cost n/a for this market
    growth_v = stores.index.latest_growth().payload["trailing_quarter_growth"]
    vpoint = sc.value
    rec = reconcile(estimates, _residuals_sc(presented, growth_v, size, vpoint), alpha)

    claims = build_narrative(estimates, rec)
    used = [c.eid for c in presented]
    audit = audit_citation_invariant(claims, ledger, subject_attrs)
    selection = audit_selection(used, ledger, segment)

    E = sorted({eid for e in estimates for eid in e.evidence_ids})
    certificate = {
        "subject": subject, "attr_hash": canonical_hash(subject_attrs),
        "evidence_hash": canonical_hash({eid: ledger.content_hash_of(eid) for eid in E}),
        "evidence_ids": E, "used_comp_eids": used,
        "candidate_pool_eids": list(pool_eids),
        "model_ver": MODEL_VER, "idx_ver": IDX_VER,
        "value": round(rec.value, 2),
        "interval": [round(rec.interval[0], 2), round(rec.interval[1], 2)],
        "reliability": rec.reliability_label,
        "claims": [{"text": c.text, "cites": c.cites, "kind": c.kind, "value": c.value} for c in claims],
        "narrative_hash": canonical_hash([c.text for c in claims]),
        "stack_ver": STACK_VER, "derived": True,
    }
    cert_txid = ledger.commit_certificate(certificate)
    return PilotResult(subject, subject_attrs, estimates, rec, claims, audit,
                       selection, certificate, cert_txid, ledger)
