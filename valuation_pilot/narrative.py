"""Deterministic, citation-bearing template narrative.

Each claim carries structured fields (kind, value) in addition to text and
citations, so the auditor can check not only that a citation is present and
anchored (the anchoring clause) but that the cited evidence actually supports
the stated figure (a structured entailment check; see audit.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agents import Estimate
from .reconcile import Reconciled


@dataclass
class Claim:
    text: str
    cites: list[str]
    kind: str            # "SC" | "Inc" | "Cost" | "Recon"
    value: float         # the CU value the sentence asserts


def build_narrative(estimates: list[Estimate], rec: Reconciled) -> list[Claim]:
    by = {e.appr: e for e in estimates}
    claims: list[Claim] = []

    if "SC" in by:
        e = by["SC"]
        claims.append(Claim(
            text=(f"Comparable sales imply a sales-comparison value of "
                  f"CU {e.value/1e6:.2f}M, adjusted to the valuation date using the "
                  f"price index."),
            cites=list(e.evidence_ids), kind="SC", value=e.value,
        ))
    if "Inc" in by:
        e = by["Inc"]
        claims.append(Claim(
            text=(f"Comparable leases and the segment yield capitalise to an income "
                  f"value of CU {e.value/1e6:.2f}M."),
            cites=list(e.evidence_ids), kind="Inc", value=e.value,
        ))
    if "Cost" in by:
        e = by["Cost"]
        claims.append(Claim(
            text=(f"The depreciated replacement cost is CU {e.value/1e6:.2f}M, "
                  f"retained at low weight as an indicative check."),
            cites=list(e.evidence_ids), kind="Cost", value=e.value,
        ))

    union: list[str] = []
    for e in estimates:
        union.extend(e.evidence_ids)
    lo, hi = rec.interval
    claims.append(Claim(
        text=(f"Weighting the approaches by reliability gives a reconciled value of "
              f"CU {rec.value/1e6:.2f}M with a 90% range of "
              f"CU {lo/1e6:.2f}-{hi/1e6:.2f}M ({rec.reliability_label} reliability)."),
        cites=sorted(set(union)), kind="Recon", value=rec.value,
    ))
    return claims
