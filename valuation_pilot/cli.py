"""Command-line demo.

    python -m valuation_pilot                  # value the worked example + verify
    python -m valuation_pilot --tamper         # tamper detection (T2)
    python -m valuation_pilot --attack          # comparable cherry-picking (T6)
    python -m valuation_pilot --entailment-attack  # mislabeled / altered citation
"""
from __future__ import annotations

import argparse
import copy
import json

from .audit import audit_citation_invariant
from .narrative import Claim
from .pipeline import value_property, verify


def cu(x): return f"CU {x/1e6:.3f}M"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Multi-agent valuation pilot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tamper", action="store_true")
    ap.add_argument("--attack", action="store_true", help="comparable cherry-picking (T6)")
    ap.add_argument("--entailment-attack", action="store_true",
                    help="mislabeled/altered citation")
    args = ap.parse_args(argv)

    r = value_property(seed=args.seed, alpha=args.alpha, attack=args.attack)
    print("=" * 66)
    print(f"SUBJECT {r.subject}  (illustrative currency units CU; attack={args.attack})")
    print("=" * 66)
    for e in r.estimates:
        print(f"  {e.appr:>4}  value={cu(e.value):>12}  reliability={e.reliability}")
    lo, hi = r.reconciled.approach_range
    print(f"  reconciled = {cu(r.reconciled.value)}  in approach range "
          f"[{lo/1e6:.3f},{hi/1e6:.3f}]M -> {lo-1e-6 <= r.reconciled.value <= hi+1e-6}")
    ci = r.reconciled.interval
    print(f"  90% conformal = [{ci[0]/1e6:.3f},{ci[1]/1e6:.3f}]M  ({r.reconciled.reliability_label})")
    print(f"\n  citation invariant: anchoring={r.audit.anchoring_ok} "
          f"entailment={r.audit.entailment_ok}  (overall {'PASS' if r.audit.ok else 'FAIL'})")
    print(f"  selection audit (T6): {'PASS' if r.selection.ok else 'FLAGGED'} "
          f"(used {r.selection.n_used}/{r.selection.n_pool} eligible comps; "
          f"omitted {len(r.selection.omitted)}; value impact {r.selection.value_impact*100:.1f}%)")

    rep = verify(r.certificate, r.ledger)
    print(f"\n  VERIFY (ledger-only): chain={rep.chain_ok} citation={rep.audit.ok} "
          f"selection={rep.selection.ok} tamper={rep.tampered_eids or 'none'} "
          f"reproduced={rep.reproduced} -> {'OK' if rep.ok else 'REJECTED'}")

    if args.attack:
        print("\n  NOTE: the cherry-picked valuation passes citation+NE+reproducibility "
              "but is REJECTED by the selection audit (T6).")

    if args.tamper:
        victim = r.certificate["evidence_ids"][0]
        r.ledger._tamper_evidence(victim, {"kind": "Comp", "tampered": True})
        rep2 = verify(r.certificate, r.ledger)
        print(f"\n  TAMPER {victim}: tampered={rep2.tampered_eids} -> "
              f"{'OK' if rep2.ok else 'REJECTED (tamper detected)'}")

    if args.entailment_attack:
        # (a) altered number: inflate the sales-comparison figure by 8%
        claims = [Claim(c.text, list(c.cites), c.kind, c.value) for c in r.claims]
        alt = [Claim(c.text, list(c.cites), c.kind, c.value) for c in claims]
        for c in alt:
            if c.kind == "SC":
                c.value *= 1.08
        res_num = audit_citation_invariant(alt, r.ledger, r.subject_attrs)
        # (b) mislabeled citation: cite a lease on the sales sentence
        mis = [Claim(c.text, list(c.cites), c.kind, c.value) for c in claims]
        for c in mis:
            if c.kind == "SC":
                c.cites = ["lease:l1"]
        res_mis = audit_citation_invariant(mis, r.ledger, r.subject_attrs)
        print("\n  ENTAILMENT ATTACK (anchoring alone would pass both):")
        print(f"    altered number (+8%): entailment_ok={res_num.entailment_ok} "
              f"-> {'REJECTED' if not res_num.ok else 'accepted'}")
        print(f"    mislabeled citation : entailment_ok={res_mis.entailment_ok} "
              f"-> {'REJECTED' if not res_mis.ok else 'accepted'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
