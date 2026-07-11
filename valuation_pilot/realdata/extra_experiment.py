"""Referee-revision experiments (M2, M3, M4, and two minor comments).

Writes results/realdata_extra_results.json alongside the immutable
realdata_results.json. Every number the revised paper adds traces here. Runs one
stage per invocation (the sandbox caps each call), persisting to /tmp/extra;
`assemble` then merges the staged pieces. Run repeatedly until FINAL:

    python -m valuation_pilot.realdata.extra_experiment          # next stage
    python -m valuation_pilot.realdata.extra_experiment status
"""
from __future__ import annotations

import json
import os
import time

from .loader_dubai import load_sales_dubai

STATE = "/tmp/extra/state.json"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "realdata_extra_results.json"))
COMMITTED = os.path.abspath(os.path.join(HERE, "..", "..", "results", "realdata_results.json"))
TEST_YEAR = 2025
SEED = 0

STAGES = ["baselines_cheap", "baselines_knn", "baselines_hedonic",
          "recalibration", "decision_boundary", "containment", "entailment"]


def _load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"done": [], "res": {}}


def _save(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=2)


def _run_stage(name, df):
    if name == "baselines_cheap":
        from .baselines import index_baseline, segment_median_wf
        return {"segment_median": segment_median_wf(df, TEST_YEAR),
                "index": index_baseline(df, TEST_YEAR)}
    if name == "baselines_knn":
        from .baselines import knn_baseline
        return {"knn": knn_baseline(df, TEST_YEAR, k=10, seed=SEED)}
    if name == "baselines_hedonic":
        from .baselines import hedonic_wf
        return {"hedonic": hedonic_wf(df, TEST_YEAR, seed=SEED)}
    if name == "recalibration":
        from .recalibration import recalibration_ablation
        return recalibration_ablation(df, TEST_YEAR, seed=SEED)
    if name == "decision_boundary":
        from .recalibration import decision_boundary
        return decision_boundary(df, TEST_YEAR, seed=SEED)
    if name == "containment":
        from .containment import run_containment_demo
        return run_containment_demo(df, TEST_YEAR, n_sample=1000, seed=SEED)
    if name == "entailment":
        # structured-entailment rate on the real narratives, read from the
        # committed structural run (audit.ok requires anchoring AND entailment).
        C = json.load(open(COMMITTED))
        ps = C["pipeline_structural"]
        return {"n_narratives": ps["n_run"], "value_bearing_claims_per_narrative": 2,
                "structured_entailment_ok": ps["citation_audit_ok"],
                "structured_entailment_rate": round(ps["citation_audit_ok"] / ps["n_run"], 4),
                "neural_nli_gate": "future work (free-text clause; not wired on this feed)"}
    raise ValueError(name)


def assemble(st):
    res = st["res"]
    base = {"segment_median": res.get("baselines_cheap", {}).get("segment_median"),
            "index": res.get("baselines_cheap", {}).get("index"),
            "knn": res.get("baselines_knn", {}).get("knn"),
            "hedonic": res.get("baselines_hedonic", {}).get("hedonic")}
    out = {
        "note": "Referee-revision experiments (M2/M3/M4 + minors). Sibling to the "
                "immutable realdata_results.json; same Dubai DLD clean frame.",
        "test_year": TEST_YEAR, "seed": SEED,
        "baselines": base,
        "recalibration": res.get("recalibration"),
        "decision_boundary": res.get("decision_boundary"),
        "containment_nontrivial": res.get("containment"),
        "entailment": res.get("entailment"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    return out


def main():
    import sys
    st = _load()
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps({"done": st["done"], "remaining": [s for s in STAGES if s not in st["done"]]}))
        return
    df = load_sales_dubai().df
    t0 = time.time()
    ran = []
    for name in STAGES:
        if name in st["done"]:
            continue
        r = _run_stage(name, df)
        st["res"][name] = r
        st["done"].append(name)
        _save(st)
        ran.append(name)
        print(json.dumps({"stage": name, "ok": True}))
        if time.time() - t0 > 33:      # stay under the per-call cap
            break
    if all(s in st["done"] for s in STAGES):
        out = assemble(st)
        print(json.dumps({"stage": "FINAL", "out": OUT,
                          "hedonic_mdape": (out["baselines"]["hedonic"] or {}).get("mdape"),
                          "segmed_mdape": (out["baselines"]["segment_median"] or {}).get("mdape"),
                          "index_mdape": (out["baselines"]["index"] or {}).get("mdape"),
                          "containment_nontrivial": (out["containment_nontrivial"] or {}).get("containment_nontrivial")}))
    else:
        print(json.dumps({"ran_this_call": ran, "remaining": [s for s in STAGES if s not in st["done"]]}))


if __name__ == "__main__":
    main()
