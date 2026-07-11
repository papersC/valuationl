"""Dubai DLD loader: registered open-data sale transactions -> the same
canonical DataFrame contract as `loader.load_sales` (NYC).

Source: Dubai Land Department transactions bulk export from the Dubai open-data
platform (snapshot 2026-05-29). The export is an append-log: each transaction
appears once per snapshot load (7 copies here), so the cascade dedups on the
registry's own `transaction_id` FIRST, then filters:

  procedure `Sell` (existing-market sales; off-plan is `Delayed Sell`),
  Residential usage, Flat/Villa dwellings, parseable core fields,
  2019--2025 window, price >= AED 100k, sane size (200--20,000 sqft after
  sqm->sqft), sane unit price (AED 100--10,000 / sqft), and a final
  belt-and-braces exact-duplicate drop on (area, building, date, price, size).

Column semantics on the shared contract, for this market:
  borough        constant "DXB" (single market tier; geography is carried by
                 the target-encoded community and project columns)
  neighborhood   community (`area_name_en`, 68 with existing-market resales)
  category       dwelling class: Flat / Villa
  zip            project name (fine-grained location proxy; "NONE" if absent)
  res_units      bedroom count parsed from `rooms_en` (Studio -> 0)
  age, land_sqft, year_built  not in this feed; constant 0 (the paper reports
                 this as a limitation of the public feed)
  price          AED; unit_price AED per sqft

The heavy streaming (12.0M-row raw file -> staged clean frame) runs in
`extract_dld.py` / `filter_dld2.py` / `build_canonical.py`; this loader reads
their staged artifacts and re-checks the invariants.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .loader import LoadResult

STAGED_FRAME = "/tmp/dld/dubai_clean.pkl"
STAGED_META = "/tmp/dld/build_meta.json"
SRC_NAME = "transactions_2026-05-29_02-08-58_2.csv.gz"


def _default_src_dir():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo = os.path.join(here, "realdata_src", "dubai")
    return repo if os.path.exists(os.path.join(repo, "dubai_clean.pkl")) else "/tmp/dld"


def load_sales_dubai(src_dir: str = None) -> LoadResult:
    src_dir = src_dir or _default_src_dir()
    frame = os.path.join(src_dir, "dubai_clean.pkl")
    meta_p = os.path.join(src_dir, "build_meta.json")
    d = pd.read_pickle(frame)
    meta = json.load(open(meta_p))
    counts = [(k, int(v)) for k, v in meta["counts"].items()]

    # invariants the downstream pipeline relies on
    assert d["sid"].is_unique
    assert (d["price"] > 0).all() and (d["gross_sqft"] > 0).all()
    assert set(["segment", "quarter", "year", "unit_price"]) <= set(d.columns)

    sha = meta.get("src_sha256", "see MANIFEST; hash staged separately")
    files = [(SRC_NAME, sha, int(meta["counts"]["sales_rows_raw"]))]
    return LoadResult(df=d, counts=counts, files=files)
