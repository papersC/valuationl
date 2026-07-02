"""End-to-end pipeline: value a property, reproduce it from the ledger, verify it.

value_property() runs the full architecture and commits a certificate that
records the released claims and the comparables actually used.
reproduce_from_ledger() rebuilds the evidence from the ledger's content-addressed
store and re-runs the pinned deterministic stack -- no reseeding -- then checks
the value matches (property P2).
verify() uses only the certificate and the ledger: it checks the hash chain, the
citation invariant (anchoring + structured entailment), the selection audit
(threat T6), and tamper (T2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import Estimate, run_all_agents
from .audit import AuditResult, SelectionAudit, audit_citation_invariant, audit_selection
from .data import build_stores
from .evidence import Evidence, EvidenceStores
from .ledger import Ledger, canonical_hash
from .narrative import Claim, build_narrative
from .reconcile import Reconciled, reconcile

MODEL_VER, IDX_VER, STACK_VER = "sc-median-0.2", "index-0.1", "valuation_pilot-0.2.0"


@dataclass
class PilotResult:
    subject: str
    subject_attrs: dict
    estimates: list
    reconciled: Reconciled
    claims: list
    audit: AuditResult
    selection: SelectionAudit
    certificate: dict
    cert_txid: str
    ledger: Ledger = field(repr=False)


def _residuals(presented, growth, size, vpoint, estimates):
    r = [c.payload["unit_price"] * (1.0 + growth) * size - vpoint for c in presented]
    r += [e.value - vpoint for e in estimates if e.reliability > 0]
    return np.array(r, dtype=float)


def _point(estimates):
    active = [e for e in estimates if e.reliability > 0]
    rsum = sum(e.reliability for e in active)
    return sum((e.reliability / rsum) * e.value for e in active)


def value_property(subject: str = "unit-2BR-110", seed: int = 7, alpha: float = 0.10,
                   attack: bool = False) -> PilotResult:
    stores, subject_attrs, pool_eids = build_stores(subject=subject, seed=seed)
    ledger = stores.comps.ledger
    dt = subject_attrs["dwelling_type"]

    all_comps = stores.comps.comparables(dt)
    if attack:  # cherry-pick: present only the five highest-priced comparables
        presented = sorted(all_comps, key=lambda c: c.payload["unit_price"], reverse=True)[: max(1, len(all_comps) // 2)]
    else:
        presented = all_comps

    estimates = run_all_agents(stores, subject_attrs, sc_comps=presented)
    idx = stores.index.latest_growth()
    growth = idx.payload["trailing_quarter_growth"]
    size = subject_attrs["built_up_area"]
    vpoint = _point(estimates)
    rec = reconcile(estimates, _residuals(presented, growth, size, vpoint, estimates), alpha)

    claims = build_narrative(estimates, rec)
    used_comp_eids = [c.eid for c in presented]
    audit = audit_citation_invariant(claims, ledger, subject_attrs)
    selection = audit_selection(used_comp_eids, ledger, dt)

    E = sorted({eid for e in estimates for eid in e.evidence_ids})
    certificate = {
        "subject": subject,
        "attr_hash": canonical_hash(subject_attrs),
        "evidence_hash": canonical_hash({eid: ledger.content_hash_of(eid) for eid in E}),
        "evidence_ids": E,
        "used_comp_eids": used_comp_eids,
        "candidate_pool_eids": [e for e in pool_eids],
        "model_ver": MODEL_VER, "idx_ver": IDX_VER,
        "value": round(rec.value, 2), "interval": [round(rec.interval[0], 2), round(rec.interval[1], 2)],
        "reliability": rec.reliability_label,
        "claims": [{"text": c.text, "cites": c.cites, "kind": c.kind, "value": c.value} for c in claims],
        "narrative_hash": canonical_hash([c.text for c in claims]),
        "stack_ver": STACK_VER, "derived": True,
    }
    cert_txid = ledger.commit_certificate(certificate)
    return PilotResult(subject, subject_attrs, estimates, rec, claims, audit,
                       selection, certificate, cert_txid, ledger)


def _stores_from_ledger(ledger: Ledger) -> tuple[EvidenceStores, dict]:
    stores = EvidenceStores.empty(ledger)
    subject_attrs = None
    for eid, rec in ledger.all_evidence().items():
        ev = Evidence(eid, rec.get("kind","?"), rec.get("subject","?"), rec.get("ts","?"), rec.get("payload", {}))
        if rec["kind"] == "Comp":
            stores.comps._items.append(ev)
        elif rec["kind"] == "Lease":
            stores.leases._items.append(ev)
        elif rec["kind"] == "Index":
            stores.index._items.append(ev)
        elif rec["kind"] == "Attr":
            stores.attrs._items.append(ev)
            if eid == "attr:subject":
                subject_attrs = rec.get("payload", {})
    return stores, subject_attrs


def reproduce_from_ledger(certificate: dict, ledger: Ledger, alpha: float = 0.10) -> bool:
    """Refetch anchored evidence and recompute the value -- no reseeding (P2)."""
    stores, subject_attrs = _stores_from_ledger(ledger)
    used = [ev for ev in stores.comps.all() if ev.eid in certificate["used_comp_eids"]]
    estimates = run_all_agents(stores, subject_attrs, sc_comps=used)
    growth = stores.index.latest_growth().payload["trailing_quarter_growth"]
    size = subject_attrs["built_up_area"]
    vpoint = _point(estimates)
    rec = reconcile(estimates, _residuals(used, growth, size, vpoint, estimates), alpha)
    return abs(round(rec.value, 2) - certificate["value"]) < 1e-6


@dataclass
class VerifyReport:
    chain_ok: bool
    audit: AuditResult
    selection: SelectionAudit
    tampered_eids: list
    reproduced: bool

    @property
    def ok(self) -> bool:
        return (self.chain_ok and self.audit.ok and self.selection.ok
                and not self.tampered_eids and self.reproduced)


def verify(certificate: dict, ledger: Ledger) -> VerifyReport:
    """Independent verification using only the certificate and the ledger (P3)."""
    _, subject_attrs = _stores_from_ledger(ledger)
    claims = [Claim(c["text"], c["cites"], c["kind"], c["value"]) for c in certificate["claims"]]
    audit = audit_citation_invariant(claims, ledger, subject_attrs)
    dt = subject_attrs["dwelling_type"]
    selection = audit_selection(certificate["used_comp_eids"], ledger, dt)
    tampered = [eid for eid in certificate["evidence_ids"] if not ledger.recompute_matches(eid)]
    try:
        reproduced = reproduce_from_ledger(certificate, ledger)
    except Exception:
        reproduced = False
    return VerifyReport(ledger.verify_chain(), audit, selection, tampered, reproduced)
