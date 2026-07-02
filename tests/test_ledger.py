"""Tests for the hash-chained ledger."""
from valuation_pilot.ledger import Ledger, canonical_hash


def test_chain_links_and_verifies():
    L = Ledger()
    L.anchor_evidence("e1", {"a": 1})
    L.anchor_evidence("e2", {"b": 2})
    L.commit_certificate({"subject": "s", "value": 1.0})
    assert L.verify_chain()
    blocks = L.blocks
    assert blocks[0].prev_hash == Ledger.GENESIS
    assert blocks[1].prev_hash == blocks[0].block_hash
    assert blocks[2].prev_hash == blocks[1].block_hash


def test_resolve_and_recompute():
    L = Ledger()
    L.anchor_evidence("e1", {"a": 1})
    assert L.is_anchored("e1")
    assert L.resolve("e1") == {"a": 1}
    assert L.recompute_matches("e1")


def test_tamper_breaks_recompute():
    L = Ledger()
    L.anchor_evidence("e1", {"a": 1})
    L._tamper_evidence("e1", {"a": 999})
    assert not L.recompute_matches("e1")   # content no longer matches the anchor


def test_canonical_hash_stable():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
