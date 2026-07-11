"""Make the containment rule non-trivial on the real feed (referee comment M2).

On this open feed the income and cost approaches are n/a, so the base pipeline
reconciles the sales-comparison estimate alone and the containment interval
collapses to a point -- containment holds, but vacuously. Here a second,
genuinely different estimator (``IndexAgent``: the segment price index applied to
the subject, independent of the comparable selection) is activated, so
reconciliation blends two approaches whose values differ and the containment rule
becomes a real constraint. The reliability-weighted blend must fall strictly
between the two approach values, and every claim -- including the index claim --
is still anchored, recomputed by the auditor, and replayed under a pinned recipe.

Under a cherry-pick the divergence is the point: the sales-comparison leg inflates
with the favourable subset while the index anchor does not, so the containment
band widens and the selection audit still fires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents import IndexAgent, SalesComparisonAgent
from ..audit import audit_citation_invariant, audit_selection
from ..evidence import Evidence, EvidenceStores
from ..ledger import Ledger, canonical_hash
from ..narrative import Claim, build_narrative
from ..pipeline import (IDX_VER, MODEL_VER, STACK_VER, PilotResult,
                        _stores_from_ledger)
from ..reconcile import reconcile
from .index import QuarterlyIndex
from .pipeline import _residuals_sc


def _reproduce_two(certificate, ledger, alpha=0.10):
    """Replay the two-approach valuation from the ledger under its pinned recipe
    (sales comparison + index), refetching evidence -- no reseeding (P2)."""
    stores, subject_attrs = _stores_from_ledger(ledger)
    used = [ev for ev in stores.comps.all() if ev.eid in certificate["used_comp_eids"]]
    sc = SalesComparisonAgent().estimate(stores, subject_attrs, comps=used)
    idx = IndexAgent().estimate(stores, subject_attrs, reliability=0.5)
    growth = stores.index.latest_growth().payload["trailing_quarter_growth"]
    size = subject_attrs["built_up_area"]
    rec = reconcile([sc, idx], _residuals_sc(used, growth, size, sc.value), alpha)
    return abs(round(rec.value, 2) - certificate["value"]) < 1e-6


def _verify_two(certificate, ledger):
    """Independent verification of a two-approach certificate using only the
    certificate and the ledger (chain, citation invariant, selection, tamper,
    replay)."""
    _, subject_attrs = _stores_from_ledger(ledger)
    claims = [Claim(c["text"], c["cites"], c["kind"], c["value"]) for c in certificate["claims"]]
    aud = audit_citation_invariant(claims, ledger, subject_attrs)
    sel = audit_selection(certificate["used_comp_eids"], ledger, subject_attrs["dwelling_type"])
    tampered = [eid for eid in certificate["evidence_ids"] if not ledger.recompute_matches(eid)]
    try:
        reproduced = _reproduce_two(certificate, ledger)
    except Exception:
        reproduced = False
    return bool(ledger.verify_chain() and aud.ok and sel.ok and not tampered and reproduced)


def _build_stores(ledger, subject, segment, size, comps_rows, growth, level):
    stores = EvidenceStores.empty(ledger)
    subject_attrs = {"dwelling_type": segment, "built_up_area": float(size),
                     "floor_level": "na", "parking": 0, "tenure": "freehold"}
    stores.attrs.add(Evidence("attr:subject", "Attr", subject, "real", subject_attrs))
    pool_eids = []
    for _, c in comps_rows.iterrows():
        eid = f"comp:{c['sid']}"
        stores.comps.add(Evidence(eid, "Comp", subject, str(c["quarter"]), {
            "dwelling_type": segment, "unit_price": round(float(c["unit_price"]), 2),
            "size": round(float(c["gross_sqft"]), 1), "floor": 0, "parking": 0}))
        pool_eids.append(eid)
    stores.index.add(Evidence("idx:seg", "Index", segment, "real", {
        "segment": segment, "trailing_quarter_growth": round(float(growth), 6),
        "level_unit_price": round(float(level), 4)}))
    return stores, subject_attrs, pool_eids


def value_real_two(subject, segment, size, comps_rows, growth, level, attack=False, alpha=0.10):
    """Two-approach real valuation: sales comparison + index, reconciled."""
    ledger = Ledger()
    stores, subject_attrs, pool_eids = _build_stores(
        ledger, subject, segment, size, comps_rows, growth, level)
    all_comps = stores.comps.comparables(segment)
    if attack:
        presented = sorted(all_comps, key=lambda c: c.payload["unit_price"],
                           reverse=True)[: max(1, len(all_comps) // 2)]
    else:
        presented = all_comps

    sc = SalesComparisonAgent().estimate(stores, subject_attrs, comps=presented)
    idx = IndexAgent().estimate(stores, subject_attrs, reliability=0.5)
    estimates = [sc, idx]
    growth_v = stores.index.latest_growth().payload["trailing_quarter_growth"]
    rec = reconcile(estimates, _residuals_sc(presented, growth_v, size, sc.value), alpha)

    claims = build_narrative(estimates, rec)
    used = [c.eid for c in presented]
    audit = audit_citation_invariant(claims, ledger, subject_attrs)
    selection = audit_selection(used, ledger, segment)
    E = sorted({eid for e in estimates for eid in e.evidence_ids})
    certificate = {
        "subject": subject, "attr_hash": canonical_hash(subject_attrs),
        "evidence_hash": canonical_hash({eid: ledger.content_hash_of(eid) for eid in E}),
        "evidence_ids": E, "used_comp_eids": used, "candidate_pool_eids": list(pool_eids),
        "model_ver": MODEL_VER, "idx_ver": IDX_VER,
        "value": round(rec.value, 2),
        "interval": [round(rec.interval[0], 2), round(rec.interval[1], 2)],
        "reliability": rec.reliability_label,
        "claims": [{"text": c.text, "cites": c.cites, "kind": c.kind, "value": c.value} for c in claims],
        "narrative_hash": canonical_hash([c.text for c in claims]),
        "stack_ver": STACK_VER, "derived": True,
    }
    ledger.commit_certificate(certificate)
    return PilotResult(subject, subject_attrs, estimates, rec, claims, audit,
                       selection, certificate, None, ledger)


def run_containment_demo(df, test_year, n_sample=1000, seed=0):
    """Activate the second approach on real subjects; report that containment is a
    binding, non-vacuous constraint that nonetheless never fails, that the two
    approaches genuinely diverge, and that every two-approach certificate still
    replays and verifies."""
    rng = np.random.default_rng(seed)
    idx_model = QuarterlyIndex(df)
    qp_all = pd.PeriodIndex(df["quarter"], freq="Q")
    dfx = df.assign(_qp=qp_all)
    seg_groups = {s: g.sort_values("sale_date") for s, g in dfx.groupby("segment")}

    def comps_fast(seg, qp, k=15):
        g = seg_groups.get(seg)
        return dfx.iloc[0:0] if g is None else g[g["_qp"] < qp].tail(k)

    cand = df[df["year"] == test_year].index.to_numpy().copy()
    rng.shuffle(cand)

    ran = 0; contain_ok = 0; interior = 0; audit_ok = 0; replay_ok = 0; verify_ok = 0
    two_approach = 0; skipped = 0
    gaps = []
    pos = 0
    while ran < n_sample and pos < len(cand):
        i = cand[pos]; pos += 1
        row = df.loc[i]; seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = comps_fast(seg, qp, k=15)
        comps = comps[comps["sid"] != row["sid"]]
        if len(comps) < 3:
            skipped += 1
            continue
        level = idx_model.level(seg, row["quarter"])
        growth = idx_model.trailing_growth(seg, row["quarter"])
        if not (level == level) or level <= 0:
            skipped += 1
            continue
        res = value_real_two(row["sid"], seg, float(row["gross_sqft"]), comps, growth, level)
        ran += 1
        lo, hi = res.reconciled.approach_range
        v = res.reconciled.value
        contain_ok += int(lo - 1e-6 <= v <= hi + 1e-6)
        vals = {e.appr: e.value for e in res.estimates if e.reliability > 0}
        if len(vals) == 2:
            two_approach += 1
            gaps.append(abs(vals["SC"] - vals["Idx"]) / v if v else 0.0)
            interior += int(lo + 1e-6 < v < hi - 1e-6)
        audit_ok += int(res.audit.ok)
        replay_ok += int(_reproduce_two(res.certificate, res.ledger))
        verify_ok += int(_verify_two(res.certificate, res.ledger))

    # cherry-pick vs honest containment band on a comparable subsample
    n_attack = 0
    honest_band = []; cherry_band = []
    for i in cand:
        row = df.loc[i]; seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = comps_fast(seg, qp, k=15); comps = comps[comps["sid"] != row["sid"]]
        level = idx_model.level(seg, row["quarter"]); growth = idx_model.trailing_growth(seg, row["quarter"])
        if len(comps) < 6 or not (level == level) or level <= 0:
            continue
        h = value_real_two(row["sid"], seg, float(row["gross_sqft"]), comps, growth, level, attack=False)
        c = value_real_two(row["sid"], seg, float(row["gross_sqft"]), comps, growth, level, attack=True)
        def span(r):
            lo, hi = r.reconciled.approach_range
            return (hi - lo) / r.reconciled.value if r.reconciled.value else 0.0
        honest_band.append(span(h)); cherry_band.append(span(c))
        n_attack += 1
        if n_attack >= 200:
            break

    gaps = np.array(gaps) if gaps else np.array([0.0])
    return {
        "n_run": int(ran), "skipped": int(skipped),
        "two_approach_active": int(two_approach),
        "containment_ok": int(contain_ok),
        "blend_strictly_interior": int(interior),
        "approach_gap_median_pct": round(float(np.median(gaps) * 100), 3),
        "approach_gap_p90_pct": round(float(np.quantile(gaps, 0.90) * 100), 3),
        "citation_audit_ok": int(audit_ok),
        "replay_ok": int(replay_ok), "verify_ok": int(verify_ok),
        "containment_nontrivial": bool(two_approach == ran and contain_ok == ran
                                       and replay_ok == ran and float(np.median(gaps)) > 0.01),
        "attack": {
            "n": int(n_attack),
            "containment_band_honest_median_pct": round(float(np.median(honest_band) * 100), 3) if honest_band else None,
            "containment_band_cherry_median_pct": round(float(np.median(cherry_band) * 100), 3) if cherry_band else None,
        },
    }
