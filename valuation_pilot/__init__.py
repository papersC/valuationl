"""valuation_pilot: reference implementation of the multi-agent generative-AI
property-valuation architecture with a hash-chained provenance ledger, a
structured citation/entailment gate, and an evidence-selection audit.

Data is synthetic; the narrative uses the deterministic citation-bearing
template. Values are illustrative currency units (CU), not measurements.
"""
from .pipeline import value_property, reproduce_from_ledger, verify, PilotResult

__all__ = ["value_property", "reproduce_from_ledger", "verify", "PilotResult"]
__version__ = "0.2.0"
