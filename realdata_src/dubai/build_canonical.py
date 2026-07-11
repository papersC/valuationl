#!/usr/bin/env python3
"""Concat filtered parts -> canonical valuation frame (loader contract), dedup,
derived columns. Saves /tmp/dld/dubai_clean.pkl + /tmp/dld/build_meta.json."""
import glob, json, re

import numpy as np
import pandas as pd

parts = sorted(glob.glob("/tmp/dld/clean2_parts/part_*.pkl"))
df = pd.concat([pd.read_pickle(p) for p in parts], ignore_index=True)
counts = json.load(open("/tmp/dld/f2_state.json"))["counts"]

# dedup: transaction_id is the registry's own identity; belt-and-braces on
# (area, building, date, price, size) exact duplicates
before = len(df)
df = df.drop_duplicates(subset=["transaction_id"])
df = df.drop_duplicates(subset=["area_name_en", "building_name_en", "sale_date",
                                "price", "gross_sqft"])
counts["deduped"] = int(len(df))

def rooms_num(s):
    s = str(s).strip()
    if s in ("", "nan", "NA"):
        return np.nan
    if "studio" in s.lower():
        return 0.0
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else np.nan

d = pd.DataFrame({
    "borough": "DXB",                                     # single market tier
    "neighborhood": df["area_name_en"].astype(str).str.strip().str.upper(),
    "category": df["property_sub_type_en"].astype(str).str.strip(),   # Flat / Villa
    "building_class": df["rooms_en"].astype(str).str.strip(),
    "price": df["price"].astype(float),                    # AED
    "sale_date": df["sale_date"],
    "gross_sqft": df["gross_sqft"].astype(float),
    "land_sqft": 0.0,                                      # not in this feed
    "year_built": 0.0,                                     # not in this feed
    "res_units": df["rooms_en"].map(rooms_num),            # bedroom count
    "zip": df["project_name_en"].astype(str).str.strip().replace({"": "NONE", "nan": "NONE"}),
    "block": df["transaction_id"].astype(str),
    "lot": df["building_name_en"].astype(str).str.strip(),
    "address": (df["area_name_en"].astype(str) + " / " + df["building_name_en"].astype(str)),
    "unit_price": df["unit_price"].astype(float),          # AED per sqft
})
d = d.sort_values("sale_date").reset_index(drop=True)
d["year"] = d["sale_date"].dt.year
d["quarter"] = d["sale_date"].dt.to_period("Q").astype(str)
d["age"] = 0.0                                             # feed carries no build year
d["res_units"] = d["res_units"].fillna(d["res_units"].median())
d["segment"] = d["neighborhood"] + "|" + d["category"]
d["sid"] = [f"sale:{i}" for i in range(len(d))]

d.to_pickle("/tmp/dld/dubai_clean.pkl")
meta = {
    "counts": counts,
    "n_clean": int(len(d)),
    "years": sorted(int(y) for y in d["year"].unique()),
    "segments": int(d["segment"].nunique()),
    "areas": int(d["neighborhood"].nunique()),
    "projects": int((d["zip"] != "NONE").sum()),
    "flat_share": float((d["category"] == "Flat").mean()),
    "price_median_aed": float(d["price"].median()),
    "unit_price_median_aed_sqft": float(d["unit_price"].median()),
    "rows_by_year": {int(k): int(v) for k, v in d["year"].value_counts().sort_index().items()},
}
json.dump(meta, open("/tmp/dld/build_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
