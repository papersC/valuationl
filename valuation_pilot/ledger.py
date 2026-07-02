"""Light permissioned-ledger stand-in: an append-only, hash-chained log.

Anchors evidence items (by content hash) and valuation certificates. The only
write capability is appending; there is no API to mutate an anchored evidence
payload, which is the structural basis of non-contamination (P1).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Block:
    index: int
    kind: str
    ref: str
    payload_hash: str
    prev_hash: str
    ts: float
    block_hash: str = ""

    def compute_hash(self) -> str:
        return canonical_hash({"index": self.index, "kind": self.kind, "ref": self.ref,
                               "payload_hash": self.payload_hash,
                               "prev_hash": self.prev_hash, "ts": self.ts})


class Ledger:
    """Append-only hash-chained ledger with a content-addressed store."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._blocks: list[Block] = []
        self._cas: dict[str, Any] = {}
        self._eid_to_hash: dict[str, str] = {}

    def _append(self, kind: str, ref: str, payload: Any) -> Block:
        payload_hash = canonical_hash(payload)
        prev = self._blocks[-1].block_hash if self._blocks else self.GENESIS
        blk = Block(len(self._blocks), kind, ref, payload_hash, prev, time.time())
        blk.block_hash = blk.compute_hash()
        self._blocks.append(blk)
        self._cas[payload_hash] = payload
        return blk

    def anchor_evidence(self, eid: str, payload: dict) -> str:
        blk = self._append("evidence", eid, payload)
        self._eid_to_hash[eid] = blk.payload_hash
        return blk.block_hash

    def commit_certificate(self, cert: dict) -> str:
        return self._append("certificate", cert.get("subject", "?"), cert).block_hash

    def is_anchored(self, eid: str) -> bool:
        return eid in self._eid_to_hash

    def resolve(self, eid: str) -> Optional[dict]:
        h = self._eid_to_hash.get(eid)
        return None if h is None else self._cas.get(h)

    def content_hash_of(self, eid: str) -> Optional[str]:
        return self._eid_to_hash.get(eid)

    def recompute_matches(self, eid: str) -> bool:
        h = self._eid_to_hash.get(eid)
        return False if h is None else canonical_hash(self._cas.get(h)) == h

    def evidence_eids(self) -> list[str]:
        return list(self._eid_to_hash.keys())

    def all_evidence(self) -> dict:
        return {eid: self.resolve(eid) for eid in self._eid_to_hash}

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for blk in self._blocks:
            if blk.prev_hash != prev or blk.block_hash != blk.compute_hash():
                return False
            prev = blk.block_hash
        return True

    # test/demo hook only; NOT part of the system interface
    def _tamper_evidence(self, eid: str, mutated_payload: dict) -> None:
        h = self._eid_to_hash.get(eid)
        if h is not None:
            self._cas[h] = mutated_payload

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)
