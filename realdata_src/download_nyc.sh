#!/usr/bin/env bash
# Reproducibly fetch the NYC DOF annualized residential sales used by the
# real-data accuracy experiment (open data, no registration). Run from this dir.
# The .xlsx files are intentionally NOT committed; this script re-creates them.
set -e
BASE="https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales"
for y in 2019 2020 2021 2022 2023 2024; do
  for b in manhattan brooklyn queens; do
    for u in "$BASE/$y/${y}_${b}.xlsx" "$BASE/${y}_${b}.xlsx"; do
      curl -sfL -A "Mozilla/5.0" -m 120 "$u" -o "nyc_${y}_${b}.xlsx" && break
    done
    echo "${y}_${b}: $(wc -c < nyc_${y}_${b}.xlsx) bytes"
  done
done
echo "done. Run:  python -m valuation_pilot.realdata.experiment   (from ../)"
