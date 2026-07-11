#!/usr/bin/env python3
"""Dedup-first cascade: stream sales_raw.csv keeping the first occurrence of
each transaction_id (the export is an append-log with 7 snapshot copies per
transaction), then apply the residential-resale cascade. Checkpointed."""
import json, os, pickle, sys, time

import pandas as pd

SRC = "/tmp/dld/sales_raw.csv"
OUTDIR = "/tmp/dld"
STATE = os.path.join(OUTDIR, "f2_state.json")
SEEN = os.path.join(OUTDIR, "f2_seen.pkl")
PARTS = os.path.join(OUTDIR, "clean2_parts")
CHUNK = 1_500_000
SQM_TO_SQFT = 10.7639
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

STAGES = ["sales_rows_raw", "unique_transactions", "procedure_sell",
          "residential_usage", "dwelling_flat_villa", "has_core_fields",
          "window_2019_2025", "price_ge_100k_aed", "gross_sqft_sane",
          "unit_price_sane"]


def cascade(df, c):
    df = df[df["procedure_name_en"] == "Sell"]          # existing-market sale
    c["procedure_sell"] += len(df)
    df = df[df["property_usage_en"] == "Residential"]
    c["residential_usage"] += len(df)
    df = df[df["property_sub_type_en"].isin(["Flat", "Villa"])]
    c["dwelling_flat_villa"] += len(df)
    df = df.copy()
    df["price"] = pd.to_numeric(df["actual_worth"], errors="coerce")
    df["area_sqm"] = pd.to_numeric(df["procedure_area"], errors="coerce")
    df["sale_date"] = pd.to_datetime(df["instance_date"], errors="coerce")
    df = df.dropna(subset=["price", "area_sqm", "sale_date"])
    df = df[df["area_name_en"].astype(str).str.strip() != ""]
    c["has_core_fields"] += len(df)
    df["year"] = df["sale_date"].dt.year
    df = df[(df["year"] >= 2019) & (df["year"] <= 2025)]
    c["window_2019_2025"] += len(df)
    df = df[df["price"] >= 100_000]
    c["price_ge_100k_aed"] += len(df)
    df["gross_sqft"] = df["area_sqm"] * SQM_TO_SQFT
    df = df[(df["gross_sqft"] >= 200) & (df["gross_sqft"] <= 20_000)]
    c["gross_sqft_sane"] += len(df)
    df["unit_price"] = df["price"] / df["gross_sqft"]
    df = df[(df["unit_price"] >= 100) & (df["unit_price"] <= 10_000)]
    c["unit_price_sane"] += len(df)
    return df


def main():
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()
    st = {"next_chunk": 0, "counts": {s: 0 for s in STAGES}, "done": False}
    seen = set()
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        if st.get("done"):
            print(json.dumps(st["counts"])); return
        seen = pickle.load(open(SEEN, "rb"))

    ci = -1
    for chunk in pd.read_csv(SRC, chunksize=CHUNK, dtype=str):
        ci += 1
        if ci < st["next_chunk"]:
            continue
        if time.time() - t0 > BUDGET:
            break
        st["counts"]["sales_rows_raw"] += len(chunk)
        fresh = ~chunk["transaction_id"].isin(seen)
        chunk = chunk[fresh]
        # keep first occurrence within the chunk as well
        chunk = chunk.drop_duplicates(subset=["transaction_id"], keep="first")
        seen.update(chunk["transaction_id"].tolist())
        st["counts"]["unique_transactions"] += len(chunk)
        part = cascade(chunk, st["counts"])
        part.to_pickle(os.path.join(PARTS, f"part_{ci:03d}.pkl"))
        st["next_chunk"] = ci + 1
    else:
        st["done"] = True
    pickle.dump(seen, open(SEEN, "wb"))
    json.dump(st, open(STATE, "w"))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
