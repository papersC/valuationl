"""Load and clean real residential sales into one canonical DataFrame.

Source: NYC Department of Finance annualized property sales (per-borough XLSX,
open data, no registration). The output schema is source-agnostic, so a Dubai
DLD loader (or any other market) can be substituted behind the same contract:

    columns: sale_date, year, quarter, borough, neighborhood, category,
             building_class, gross_sqft, land_sqft, year_built, age,
             res_units, price, unit_price, zip, block, lot, address, sid

Every filter stage records its surviving count in `counts` for the paper.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# NYC building-class categories kept as residential dwellings that carry a
# usable gross floor area (coops report share-based areas and are excluded by
# the gross-sqft filter rather than by category).
RESIDENTIAL_CATEGORY_CODES = {"01", "02", "03", "04", "12", "13", "15"}

HEADER_MARKER = "BOROUGH"


@dataclass
class LoadResult:
    df: pd.DataFrame
    counts: list = field(default_factory=list)   # [(stage, rows), ...]
    files: list = field(default_factory=list)    # [(name, sha256, rows)]

    def counts_dict(self) -> dict:
        return {stage: n for stage, n in self.counts}


def _read_one(path: str) -> pd.DataFrame:
    # read the sheet exactly once (openpyxl re-parses on every read_excel call,
    # so a header-detection pass + a data pass would double the cost)
    cache = path + ".parquet"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_parquet(cache)
    raw = pd.read_excel(path, header=None, engine="openpyxl", dtype=object)
    hdr = None
    for i in range(min(12, len(raw))):
        rowvals = [str(x).upper() for x in raw.iloc[i].tolist()]
        if any(HEADER_MARKER in v for v in rowvals):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"no header row found in {path}")
    cols = [re.sub(r"\s+", " ", str(c)).strip().upper() for c in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    df = df.reset_index(drop=True)
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def _col(df: pd.DataFrame, *cands: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c for k in cands):
            return c
    raise KeyError(f"none of {cands} in {list(df.columns)}")


def load_sales(src_dir: str) -> LoadResult:
    paths = sorted(glob.glob(os.path.join(src_dir, "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"no .xlsx files in {src_dir}")

    frames, files = [], []
    for p in paths:
        df = _read_one(p)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        files.append((os.path.basename(p), sha, int(len(df))))
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    counts = [("raw_rows", int(len(raw)))]

    cat_c = _col(raw, "BUILDING CLASS CATEGORY")
    price_c = _col(raw, "SALE PRICE")
    date_c = _col(raw, "SALE DATE")
    gsf_c = _col(raw, "GROSS SQUARE FEET", "GROSS")
    lsf_c = _col(raw, "LAND SQUARE FEET", "LAND")
    yb_c = _col(raw, "YEAR BUILT")
    boro_c = _col(raw, "BOROUGH")
    nbhd_c = _col(raw, "NEIGHBORHOOD")
    resu_c = _col(raw, "RESIDENTIAL UNITS")
    bc_c = _col(raw, "BUILDING CLASS AT TIME OF SALE", "BUILDING CLASS AT PRESENT")
    zip_c = _col(raw, "ZIP CODE")
    blk_c = _col(raw, "BLOCK")
    lot_c = _col(raw, "LOT")
    addr_c = _col(raw, "ADDRESS")

    d = pd.DataFrame({
        "borough": raw[boro_c].astype(str).str.strip(),
        "neighborhood": raw[nbhd_c].astype(str).str.strip().str.upper(),
        "category_raw": raw[cat_c].astype(str).str.strip(),
        "building_class": raw[bc_c].astype(str).str.strip(),
        "price": pd.to_numeric(raw[price_c], errors="coerce"),
        "sale_date": pd.to_datetime(raw[date_c], errors="coerce"),
        "gross_sqft": pd.to_numeric(raw[gsf_c], errors="coerce"),
        "land_sqft": pd.to_numeric(raw[lsf_c], errors="coerce"),
        "year_built": pd.to_numeric(raw[yb_c], errors="coerce"),
        "res_units": pd.to_numeric(raw[resu_c], errors="coerce"),
        "zip": raw[zip_c].astype(str).str.strip(),
        "block": raw[blk_c].astype(str).str.strip(),
        "lot": raw[lot_c].astype(str).str.strip(),
        "address": raw[addr_c].astype(str).str.strip(),
    })
    d["category"] = d["category_raw"].str.extract(r"^(\d{2})")[0]

    # -- filter cascade (each stage recorded) --------------------------------
    d = d[d["category"].isin(RESIDENTIAL_CATEGORY_CODES)]
    counts.append(("residential_category", int(len(d))))

    d = d.dropna(subset=["price", "sale_date", "gross_sqft", "year_built"])
    d = d[d["sale_date"].notna()]
    counts.append(("has_core_fields", int(len(d))))

    d = d[d["price"] >= 100_000]
    counts.append(("price_ge_100k", int(len(d))))

    d = d[(d["gross_sqft"] >= 200) & (d["gross_sqft"] <= 20_000)]
    counts.append(("gross_sqft_sane", int(len(d))))

    d["unit_price"] = d["price"] / d["gross_sqft"]
    d = d[(d["unit_price"] >= 50) & (d["unit_price"] <= 5_000)]
    counts.append(("unit_price_sane", int(len(d))))

    d["year"] = d["sale_date"].dt.year
    d = d[(d["year_built"] >= 1800) & (d["year_built"] <= d["year"])]
    counts.append(("year_built_sane", int(len(d))))

    # resale proxy: NYC data does not flag resale vs. first sale, so exclude
    # sales in (or one year after) the building's construction year, which are
    # almost all developer first sales. Documented as an approximation.
    d = d[d["year_built"] <= d["year"] - 2]
    counts.append(("resale_proxy", int(len(d))))

    # dedup exact duplicate rows (same parcel, date, price)
    d = d.drop_duplicates(subset=["borough", "block", "lot", "address",
                                  "sale_date", "price"])
    counts.append(("deduped", int(len(d))))

    d = d.sort_values("sale_date").reset_index(drop=True)
    d["quarter"] = d["sale_date"].dt.to_period("Q").astype(str)
    d["age"] = (d["year"] - d["year_built"]).clip(lower=0)
    d["res_units"] = d["res_units"].fillna(1).clip(lower=0)
    d["land_sqft"] = d["land_sqft"].fillna(0).clip(lower=0)
    # segment used for comparables and the price index
    d["segment"] = d["borough"].astype(str) + "|" + d["category"]
    d["sid"] = [f"sale:{i}" for i in range(len(d))]

    return LoadResult(df=d, counts=counts, files=files)
