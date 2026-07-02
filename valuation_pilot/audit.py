"""The citation-invariant auditor and the evidence-selection (completeness) audit.

Citation invariant (paper eq. CI): for every value-affecting claim there is a
cited evidence item c such that (i) c is anchored on the ledger, and (ii) c
entails the claim. Clause (i) is a hash lookup. Clause (ii) is checked here by a
STRUCTURED entailment test: the cited evidence must be of the kind the claim
needs, and the figure the claim asserts must be recomputable from the cited
evidence within tolerance. This catches a mislabeled citation (a lease attached
to a sales sentence) or an altered number, which a citation-presence check would
pass. A neural NLI gate is the general free-text case and is left to future work.

The selection audit addresses threat T6 (an authorised valuer who cites only
favourable comparables): it recomputes the sales-comparison figure on the full
anchored candidate pool and flags a material divergence from the used subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ledger import Ledger
from .narrative import Claim


@dataclass
class AuditResult:
    ok: bool
    anchoring_ok: bool
    entailment_ok: bool
    failures: list[str] = field(default_factory=list)
    checked: int = 0


def _payload(ledger: Ledger, eid: str):
    rec = ledger.resolve(eid)
    return None if rec is None else rec.get("payload")


def _expected_value(claim: Claim, ledger: Ledger, subject_attrs: dict):
    """Recompute the figure the claim asserts from its cited evidence.

    Returns (expected_value or None, reason). None means the cited evidence does
    not support a claim of this kind (kind mismatch / missing evidence).
    """
    def _kind(e):
        r = ledger.resolve(e)
        return r.get("kind") if r else None
    def _pl(e):
        return _payload(ledger, e) or {}
    comp = [c for c in claim.cites if _kind(c) == "Comp"]
    idx = [c for c in claim.cites if _kind(c) == "Index"]
    lease = [c for c in claim.cites if _kind(c) == "Lease" and "annual_rent" in _pl(c)]
    yld = [c for c in claim.cites if _kind(c) == "Lease" and "gross_yield" in _pl(c)]
    cost = [c for c in claim.cites if "replacement_cost" in _pl(c)]

    if claim.kind == "SC":
        if not comp:
            return None, "sales claim cites no comparable-sale evidence"
        growth = 0.0
        if idx:
            growth = _payload(ledger, idx[0])["trailing_quarter_growth"]
        units = [_payload(ledger, c)["unit_price"] * (1.0 + growth) for c in comp]
        return float(np.median(units) * subject_attrs["built_up_area"]), ""
    if claim.kind == "Inc":
        if not lease or not yld:
            return None, "income claim missing lease or yield evidence"
        rents = [_payload(ledger, l)["annual_rent"] for l in lease]
        gy = _payload(ledger, yld[0])["gross_yield"]
        return float(np.median(rents) / gy), ""
    if claim.kind == "Cost":
        if not cost:
            return None, "cost claim cites no cost evidence"
        p = _payload(ledger, cost[0])
        return float(p["land_value"] + p["replacement_cost"] * (1.0 - p["depreciation"])), ""
    if claim.kind == "Recon":
        return claim.value, ""  # reconciliation value is checked by NE, not re-derived here
    return None, f"unknown claim kind {claim.kind}"


def audit_citation_invariant(claims: list[Claim], ledger: Ledger,
                             subject_attrs: dict, rtol: float = 1e-6) -> AuditResult:
    failures: list[str] = []
    anchoring_ok = True
    entailment_ok = True
    for i, claim in enumerate(claims):
        if not claim.cites:
            failures.append(f"claim {i}: no citation"); anchoring_ok = False; continue
        for eid in claim.cites:                      # clause (i): anchoring
            if not ledger.is_anchored(eid):
                failures.append(f"claim {i}: eid {eid} not anchored"); anchoring_ok = False
            elif not ledger.recompute_matches(eid):
                failures.append(f"claim {i}: eid {eid} failed hash re-check"); anchoring_ok = False
        try:                                                       # clause (ii)
            expected, reason = _expected_value(claim, ledger, subject_attrs)
        except (TypeError, KeyError):
            expected, reason = None, "cited evidence unreadable"
        if expected is None:
            failures.append(f"claim {i}: {reason}"); entailment_ok = False
        elif claim.value != 0 and abs(claim.value - expected) / abs(claim.value) > rtol:
            failures.append(f"claim {i}: asserted {claim.value:.0f} not entailed by "
                            f"cited evidence (recomputed {expected:.0f})")
            entailment_ok = False
    return AuditResult(ok=not failures, anchoring_ok=anchoring_ok,
                       entailment_ok=entailment_ok, failures=failures, checked=len(claims))


@dataclass
class SelectionAudit:
    ok: bool
    n_used: int
    n_pool: int
    omitted: list
    value_impact: float   # relative change in the median unit price from omissions


def audit_selection(used_comp_eids: list[str], ledger: Ledger,
                    dwelling_type: str) -> SelectionAudit:
    """Completeness check for threat T6 (comparable cherry-picking).

    The candidate pool is anchored, so an auditor can require that a valuation
    use every eligible comparable in the segment, or record a cited reason for
    each exclusion. Any un-justified omission is flagged for review; we also
    report the value impact of the omissions. (This pilot records no exclusion
    reasons, so any omission is flagged.)
    """
    def _is_comp(e):
        r = ledger.resolve(e)
        return bool(r) and r.get("kind") == "Comp"
    pool = [e for e in ledger.evidence_eids() if _is_comp(e)
            and (_payload(ledger, e) or {}).get("dwelling_type") == dwelling_type]
    used = [e for e in used_comp_eids if _is_comp(e)]
    omitted = sorted(set(pool) - set(used))
    def _units(eids):
        return [(_payload(ledger, e) or {}).get("unit_price") for e in eids
                if (_payload(ledger, e) or {}).get("unit_price") is not None]
    pu, uu = _units(pool), _units(used)
    pm = float(np.median(pu)) if pu else 0.0
    um = float(np.median(uu)) if uu else 0.0
    impact = abs(um - pm) / pm if pm else 0.0
    return SelectionAudit(ok=(len(omitted) == 0), n_used=len(used), n_pool=len(pool),
                          omitted=omitted, value_impact=impact)
