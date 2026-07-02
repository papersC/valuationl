"""Evidence items and the four abstract evidence stores (interface I1).

Each store is a typed view over evidence that has been anchored on the ledger.
Agents read through these stores; they never write to them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ledger import Ledger


@dataclass(frozen=True)
class Evidence:
    eid: str
    kind: str          # "Comp" | "Lease" | "Index" | "Attr"
    subject: str
    ts: str
    payload: dict


class EvidenceStore:
    """Base store: holds evidence and anchors each item on the ledger."""

    kind = "Base"

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._items: list[Evidence] = []

    def add(self, ev: Evidence) -> None:
        # Anchor the payload on the ledger, then keep a local typed view.
        self.ledger.anchor_evidence(ev.eid, {"kind": ev.kind, "subject": ev.subject,
                                             "ts": ev.ts, "payload": ev.payload})
        self._items.append(ev)

    def all(self) -> list[Evidence]:
        return list(self._items)


class ComparableSalesStore(EvidenceStore):
    kind = "Comp"

    def comparables(self, dwelling_type: str) -> list[Evidence]:
        return [e for e in self._items if e.payload.get("dwelling_type") == dwelling_type]


class LeaseStore(EvidenceStore):
    kind = "Lease"

    def leases(self, dwelling_type: str) -> list[Evidence]:
        return [e for e in self._items if e.payload.get("dwelling_type") == dwelling_type]


class PriceIndexStore(EvidenceStore):
    kind = "Index"

    def latest_growth(self) -> Optional[Evidence]:
        return self._items[-1] if self._items else None


class AttributeStore(EvidenceStore):
    kind = "Attr"

    def for_subject(self, subject: str) -> Optional[Evidence]:
        for e in self._items:
            if e.subject == subject:
                return e
        return None


@dataclass
class EvidenceStores:
    comps: ComparableSalesStore
    leases: LeaseStore
    index: PriceIndexStore
    attrs: AttributeStore

    @classmethod
    def empty(cls, ledger: Ledger) -> "EvidenceStores":
        return cls(
            comps=ComparableSalesStore(ledger),
            leases=LeaseStore(ledger),
            index=PriceIndexStore(ledger),
            attrs=AttributeStore(ledger),
        )
