"""Tests for the architecture's invariants, provenance properties, and defences."""
from valuation_pilot import value_property, reproduce_from_ledger, verify
from valuation_pilot.audit import audit_citation_invariant
from valuation_pilot.narrative import Claim


def test_no_extrapolation_invariant():
    r = value_property(seed=7)
    lo, hi = r.reconciled.approach_range
    assert lo - 1e-6 <= r.reconciled.value <= hi + 1e-6


def test_weights_convex():
    r = value_property(seed=7)
    w = r.reconciled.weights
    assert abs(sum(w.values()) - 1.0) < 1e-9 and all(v >= 0 for v in w.values())


def test_citation_invariant_passes_on_template():
    r = value_property(seed=7)
    assert r.audit.ok and r.audit.anchoring_ok and r.audit.entailment_ok


def test_entailment_rejects_altered_number():
    """A citation-present but numerically wrong claim fails the entailment clause,
    even though the anchoring clause passes."""
    r = value_property(seed=7)
    alt = [Claim(c.text, list(c.cites), c.kind, c.value) for c in r.claims]
    for c in alt:
        if c.kind == "SC":
            c.value *= 1.08          # inflate the cited figure
    res = audit_citation_invariant(alt, r.ledger, r.subject_attrs)
    assert res.anchoring_ok and not res.entailment_ok and not res.ok


def test_entailment_rejects_mislabeled_citation():
    """A sales sentence citing a lease (kind mismatch) fails entailment."""
    r = value_property(seed=7)
    mis = [Claim(c.text, list(c.cites), c.kind, c.value) for c in r.claims]
    for c in mis:
        if c.kind == "SC":
            c.cites = ["lease:l1"]
    res = audit_citation_invariant(mis, r.ledger, r.subject_attrs)
    assert not res.entailment_ok and not res.ok


def test_selection_attack_flagged():
    """Comparable cherry-picking passes citation+NE+reproducibility but the
    selection audit (T6) flags it; the honest run passes."""
    honest = value_property(seed=7, attack=False)
    assert honest.selection.ok
    attack = value_property(seed=7, attack=True)
    assert attack.audit.ok                       # citation invariant still passes
    lo, hi = attack.reconciled.approach_range    # NE still holds
    assert lo - 1e-6 <= attack.reconciled.value <= hi + 1e-6
    assert reproduce_from_ledger(attack.certificate, attack.ledger)  # still reproducible
    assert not attack.selection.ok               # but selection audit catches it
    assert not verify(attack.certificate, attack.ledger).ok


def test_reproducibility_from_ledger():
    """Refetch anchored evidence and recompute -- no reseeding (P2)."""
    r = value_property(seed=7)
    assert reproduce_from_ledger(r.certificate, r.ledger)


def test_determinism_same_seed():
    a, b = value_property(seed=7), value_property(seed=7)
    assert a.certificate["value"] == b.certificate["value"]
    assert a.certificate["evidence_hash"] == b.certificate["evidence_hash"]


def test_verification_clean():
    r = value_property(seed=7)
    assert verify(r.certificate, r.ledger).ok


def test_tamper_is_detected():
    r = value_property(seed=7)
    victim = r.certificate["evidence_ids"][0]
    r.ledger._tamper_evidence(victim, {"kind": "Comp", "tampered": True})
    rep = verify(r.certificate, r.ledger)
    assert victim in rep.tampered_eids and not rep.ok


def test_non_contamination_no_evidence_write_api():
    from valuation_pilot.ledger import Ledger
    public = [m for m in dir(Ledger) if not m.startswith("_")]
    assert not any(m.startswith(("update", "delete", "set", "mutate")) for m in public)
    assert "commit_certificate" in public and "anchor_evidence" in public
