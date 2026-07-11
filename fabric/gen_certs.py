"""Generate real Dubai valuation certificates for on-chain anchoring.

For each sampled real subject, the deterministic audit pipeline (value_real)
builds a full certificate over real DLD comparable evidence. We then extract the
two things the ledger stores:

  cert_digest      SHA-256 over the canonical certificate  -> the paper's
                   constant-size on-chain commitment (32 bytes / 64 hex).
  evidence_digests {eid: content_hash} for every evidence item used.

To demonstrate the O(1) on-chain footprint claim, we vary the comparable-pool
size k across {4, 8, 16, 32, 64}: the evidence set grows but the certificate
digest stays exactly 64 hex chars, so the committed record size is constant.

Output: certs.json, consumed by the Go gateway client (measure_costs).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # pilot/ on path

from valuation_pilot.ledger import canonical_hash
from valuation_pilot.realdata.index import QuarterlyIndex
from valuation_pilot.realdata.pipeline import value_real

FRAME = os.path.join(os.path.dirname(HERE), "realdata_src", "dubai", "dubai_clean.pkl")
TEST_YEAR = 2025
N_LATENCY = 500          # certs for the latency / throughput run (k=15 each)
K_SWEEP = [4, 8, 16, 32, 64]   # evidence-volume sweep for the O(1) footprint test
SEED = 0


def _comps_for(df_hist, seg, qp, k):
    hist = df_hist[(df_hist["segment"] == seg)
                   & (pd.PeriodIndex(df_hist["quarter"], freq="Q") < qp)]
    return hist.sort_values("sale_date").tail(k)


def _digests_for(row, df, idx_model, k):
    seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
    comps = _comps_for(df, seg, qp, k)
    comps = comps[comps["sid"] != row["sid"]]
    if len(comps) < 3:
        return None
    growth = idx_model.trailing_growth(seg, row["quarter"])
    res = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth)
    cert = res.certificate
    cert_digest = canonical_hash(cert)
    ev = {eid: res.ledger.content_hash_of(eid) for eid in cert["evidence_ids"]}
    return {
        "certId": f"CERT-{row['sid']}",
        "certDigest": cert_digest,
        "nEvidence": len(ev),
        "evidenceDigests": ev,
        "value": cert["value"],
        "k": int(k),
    }


def main():
    df = pd.read_pickle(FRAME)
    idx_model = QuarterlyIndex(df)
    test = df[df["year"] == TEST_YEAR]
    rng = np.random.default_rng(SEED)
    cand = test.index.to_numpy().copy()
    rng.shuffle(cand)

    latency, sweep = [], []
    i = 0
    while len(latency) < N_LATENCY and i < len(cand):
        rec = _digests_for(df.loc[cand[i]], df, idx_model, k=15); i += 1
        if rec:
            latency.append(rec)
    # evidence-volume sweep: same subjects, varying k
    for k in K_SWEEP:
        j = 0; got = 0
        while got < 20 and j < len(cand):
            rec = _digests_for(df.loc[cand[j]], df, idx_model, k=k); j += 1
            if rec:
                sweep.append(rec); got += 1

    out = {"test_year": TEST_YEAR, "seed": SEED, "n_latency": len(latency),
           "latency_certs": latency, "sweep_certs": sweep}
    dst = os.path.join(HERE, "certs.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    # sanity: every cert digest is a 64-hex string regardless of k
    lens = sorted({len(c["certDigest"]) for c in latency + sweep})
    print(f"wrote {len(latency)} latency certs + {len(sweep)} sweep certs -> {dst}")
    print(f"cert-digest lengths present: {lens}  (constant 64 => O(1) footprint)")
    print(f"evidence-count range in sweep: "
          f"{min(c['nEvidence'] for c in sweep)}..{max(c['nEvidence'] for c in sweep)}")


if __name__ == "__main__":
    main()
