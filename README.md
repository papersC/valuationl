# Valuation Pilot

A runnable reference implementation (proof-of-concept) of the multi-agent
generative-AI property-valuation architecture with a hash-chained provenance
ledger and citation invariant.

It is a **pilot**, not a production system: data is synthetic, the narrative
uses the paper's deterministic citation-bearing template (no external LLM), and
the "blockchain" is a local append-only hash-chained log rather than a
distributed ledger. All figures are illustrative currency units (CU).

## Run

```bash
pip install -r requirements.txt

# value the worked example, print certificate + narrative + verification
python -m valuation_pilot

# demonstrate tamper detection on an anchored evidence item
python -m valuation_pilot --tamper

# run the test suite
pytest
```

Python 3.10+ and NumPy are the only requirements.

## What it implements (and where)

| Paper element | Module |
|---|---|
| Ledger layer: append-only, hash-chained anchors + content-addressed store | `ledger.py` |
| Evidence stores (I1): comparable sales, leases, price index, attributes | `evidence.py`, `data.py` |
| Three specialist valuer agents (I2): sales-comparison, income, cost | `agents.py` |
| Reliability-weighted reconciliation + **no-extrapolation invariant** | `reconcile.py` |
| Conformal prediction interval | `reconcile.py` |
| Deterministic, citation-bearing template narrative | `narrative.py` |
| **Citation-invariant** auditor | `audit.py` |
| Valuation certificate, reproduction (P2), verification (P3), tamper detection (T2) | `pipeline.py` |
| End-to-end CLI demo | `cli.py` |

## Properties demonstrated

- **P1 non-contamination** — the ledger exposes no method to mutate anchored
  evidence; the only write is appending a *derived* certificate.
- **P2 reproducibility** — `reproduce()` re-runs the pinned, deterministic stack
  from the anchored evidence and reproduces the certificate value.
- **P3 verifiability** — `verify()` checks the hash chain and that every cited
  figure resolves to anchored, untampered evidence, using only the ledger.
- **P4 bounded reconciliation** — the reconciled value is a convex combination,
  so it always lies within the approach range (asserted at run time).
- **T2 tamper detection** — rewriting an anchored payload breaks its content
  hash and is rejected by `verify()` (`--tamper` demo).

## From pilot to deployment

- Replace synthetic `data.py` with adapters onto real transaction, lease,
  index, and attribute feeds (interface I1 is unchanged).
- Swap the sales-comparison adjustments for a calibrated hedonic/GBM model with
  SHAP attributions; keep the per-adjustment citation.
- Replace `default_calibration()` with real held-out residuals.
- Optionally add an LLM narrator in front of the template; the citation-invariant
  auditor already gates its output, and the template remains the fallback.
- Swap the local `Ledger` for a permissioned ledger (e.g. Hyperledger Fabric)
  with the same anchor/commit/verify surface.

## Running tests without pytest

If `pytest` is not installed, run `python run_tests.py` (uses only NumPy).

## What the pilot demonstrates (v0.2)

Beyond the by-construction invariants, the pilot now tests the parts that can
actually fail:

- **Structured entailment gate** (`audit.py`): the auditor recomputes each figure
  from its cited evidence and rejects a claim whose number is altered or whose
  citation is the wrong kind (e.g. a lease cited for a sales figure) — cases a
  citation-*presence* check would accept. A neural NLI gate over free-text
  narration is future work.
- **Evidence-selection audit / threat T6** (`audit.py`): the full candidate
  comparable pool is anchored; a completeness check flags a valuation that omits
  eligible comparables (cherry-picking), which still passes citation + NE +
  reproducibility.
- **Refetch-based reproducibility** (`pipeline.py`): `reproduce_from_ledger()`
  rebuilds the evidence from the ledger and recomputes — it does not reseed.
- **Conformal coverage experiment** (`coverage_experiment.py`): measures empirical
  interval coverage under exchangeability (~0.93 at nominal 0.90) and under a 5%
  injected price drift (~0.49), showing the guarantee and its failure mode.

```bash
python -m valuation_pilot --attack             # comparable cherry-picking (T6) is flagged
python -m valuation_pilot --entailment-attack  # altered/mislabeled citation is rejected
python -m valuation_pilot.coverage_experiment  # conformal coverage, exchangeable vs drift
python -m valuation_pilot.experiments          # full measured sweep (JSON; feeds the paper figure)
```

## Measured experiments (`experiments.py`)

`python -m valuation_pilot.experiments` prints a JSON document with the three
measured result sets behind the paper's measured-results figure
(`fig_experiments.pdf`), cached in `results/experiment_results.json`:

- **coverage vs nominal level** — empirical split-conformal coverage across
  nominal levels 0.50–0.95, exchangeable vs 5% drifted test set;
- **coverage vs drift magnitude** — coverage at nominal 0.90 as injected test
  drift grows 0–8% (0.93 → 0.07);
- **audit latency vs narrative length** — measured wall-clock of the structural
  citation audit (real anchor resolution + hash and figure recomputation),
  linear at ~0.4 ms per claim.
