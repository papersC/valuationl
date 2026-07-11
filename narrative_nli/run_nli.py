"""Part 3: a real LLM narrative + a neural NLI entailment gate on real Dubai
appraisals. Fills the dashed 'entailment (neural)' row of the paper's targets.

For each sampled real subject:
  1. Build the certificate with the deterministic pipeline (value_real): real
     comparable evidence, a reconciled value, and the value-bearing claims.
  2. Render the cited evidence as a factual PREMISE (the comparable unit prices,
     the time adjustment, the subject area, the reconciled figure).
  3. Generate a free-text valuer NARRATIVE with a pinned, self-hosted instruct
     LLM under greedy decoding (deterministic -> replayable, as the paper
     requires), conditioned only on that premise.
  4. Split the narrative into sentences and run a neural NLI model (DeBERTa
     MNLI/ANLI, per nie2020) with premise = evidence context, hypothesis =
     sentence. A sentence is admitted iff P(entail) >= tau.

We report the entailment rate on the generated narratives (target >= 0.95) and,
as an adversarial check that the gate has teeth, the rate on narratives whose
headline figure has been corrupted (should collapse).

CPU-only; models are small. Run with the numpy-1.x main env (torch + transformers).
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          AutoModelForCausalLM)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # pilot/ on path

from valuation_pilot.realdata.index import QuarterlyIndex
from valuation_pilot.realdata.pipeline import build_real_stores, value_real
from valuation_pilot.agents import SalesComparisonAgent
from valuation_pilot.realdata.experiment import _comps_for

FRAME = os.path.join(os.path.dirname(HERE), "realdata_src", "dubai", "dubai_clean.pkl")
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # premise/hypothesis NLI (MNLI+FEVER+ANLI)
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"                     # pinned, greedy
N_APPRAISALS = 40
TAU = 0.5              # entailment-probability admission threshold
TEST_YEAR = 2025
SEED = 0
torch.manual_seed(SEED)


def evidence_premise(seg, comps_rows, growth, size, value, interval):
    units = sorted(round(float(u), 0) for u in comps_rows["unit_price"])
    lo, hi = interval
    return (
        f"Market segment: {seg}. Subject built-up area: {int(size)} square feet. "
        f"Comparable sale unit prices (AED per sqft): {units}. "
        f"Time adjustment to the valuation date: {growth*100:+.1f} percent. "
        f"Reconciled valuation: AED {value/1e6:.2f} million. "
        f"Ninety percent valuation range: AED {lo/1e6:.2f} to {hi/1e6:.2f} million. "
        f"No rental or replacement-cost evidence is available for this subject."
    )


def gen_narrative(llm_tok, llm, premise):
    msg = [
        {"role": "system", "content":
         "You are a property valuer writing a short factual valuation note. "
         "Use ONLY the facts provided. State the reconciled value and the range. "
         "Do not invent numbers, rents, yields, or property features not given. "
         "Write 3 to 4 sentences."},
        {"role": "user", "content": f"Facts:\n{premise}\n\nWrite the valuation note."},
    ]
    text = llm_tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = llm_tok(text, return_tensors="pt")
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=160, do_sample=False,
                           temperature=None, top_p=None, top_k=None,
                           pad_token_id=llm_tok.eos_token_id)
    gen = llm_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


VALUE_BEARING = re.compile(r"\d")   # sentences asserting a figure


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


def nli_entail_prob(nli_tok, nli, premise, hypothesis, entail_idx):
    inp = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = nli(**inp).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[entail_idx])


def main():
    df = pd.read_pickle(FRAME)
    idx_model = QuarterlyIndex(df)
    test = df[df["year"] == TEST_YEAR]
    rng = np.random.default_rng(SEED)
    cand = test.index.to_numpy().copy(); rng.shuffle(cand)

    print("loading NLI:", NLI_MODEL)
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    entail_idx = [i for i, l in nli.config.id2label.items()
                  if l.lower().startswith("entail")][0]
    print("loading LLM:", LLM_MODEL)
    llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32).eval()

    records = []
    n = 0
    for i in cand:
        if n >= N_APPRAISALS:
            break
        row = df.loc[i]
        seg = row["segment"]; qp = pd.Period(row["quarter"], freq="Q")
        comps = _comps_for(df, seg, qp, k=8)
        comps = comps[comps["sid"] != row["sid"]]
        if len(comps) < 3:
            continue
        growth = idx_model.trailing_growth(seg, row["quarter"])
        res = value_real(row["sid"], seg, float(row["gross_sqft"]), comps, growth)
        size = float(row["gross_sqft"]); val = res.reconciled.value
        interval = res.reconciled.interval
        premise = evidence_premise(seg, comps, growth, size, val, interval)
        narr = gen_narrative(llm_tok, llm, premise)

        sents = [s for s in split_sentences(narr) if VALUE_BEARING.search(s)]
        claim_probs = [nli_entail_prob(nli_tok, nli, premise, s, entail_idx) for s in sents]
        # adversarial: corrupt the headline figure by +35% in the premise-free hypothesis
        bad_val = val * 1.35
        bad_sent = f"The reconciled valuation is AED {bad_val/1e6:.2f} million."
        bad_prob = nli_entail_prob(nli_tok, nli, premise, bad_sent, entail_idx)

        records.append({
            "sid": str(row["sid"]), "segment": seg, "value": val,
            "narrative": narr, "n_claims": len(sents),
            "claim_entail_probs": claim_probs,
            "adversarial_entail_prob": bad_prob,
        })
        n += 1
        print(f"[{n}/{N_APPRAISALS}] {seg[:22]:22s} claims={len(sents)} "
              f"min_p={min(claim_probs) if claim_probs else float('nan'):.3f} "
              f"adv_p={bad_prob:.3f}")

    all_probs = [p for r in records for p in r["claim_entail_probs"]]
    admitted = [p for p in all_probs if p >= TAU]
    adv = [r["adversarial_entail_prob"] for r in records]
    summary = {
        "nli_model": NLI_MODEL, "llm_model": LLM_MODEL, "decoding": "greedy",
        "n_appraisals": len(records), "n_claims": len(all_probs), "tau": TAU,
        "entailment_rate": len(admitted) / len(all_probs) if all_probs else None,
        "mean_entail_prob": float(np.mean(all_probs)) if all_probs else None,
        "adversarial_mean_entail_prob": float(np.mean(adv)) if adv else None,
        "adversarial_admitted_rate": float(np.mean([a >= TAU for a in adv])) if adv else None,
    }
    out = {"summary": summary, "appraisals": records}
    dst = os.path.join(os.path.dirname(HERE), "results", "narrative_nli_results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
