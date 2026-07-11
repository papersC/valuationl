#!/usr/bin/env python3
"""Resumable single-pass filter over the DLD transactions CSV.

Streams the (mis-named .gz, actually plain) CSV from a byte offset saved in a
state file, keeps Sales-group rows with the columns the valuation loader needs,
appends them to an output CSV, and stops when the time budget is spent.
Run repeatedly until state.done == true. Line-based: verified upfront that the
export contains no embedded newlines inside quoted fields (all rows parse to
the header's column count; parse failures are counted and must stay 0).
"""
import csv, io, json, os, sys, time

SRC = "/sessions/zen-cool-hopper/mnt/paper 1-1/dubai_data/transactions_2026-05-29_02-08-58_2.csv.gz"
OUTDIR = "/tmp/dld"
OUT = os.path.join(OUTDIR, "sales_raw.csv")
STATE = os.path.join(OUTDIR, "state.json")

KEEP = ["transaction_id", "procedure_name_en", "instance_date", "property_type_en",
        "property_sub_type_en", "property_usage_en", "reg_type_en", "area_name_en",
        "building_name_en", "project_name_en", "master_project_en", "rooms_en",
        "procedure_area", "actual_worth", "meter_sale_price"]

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 33.0
CHUNK = 8 * 1024 * 1024


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    st = {"offset": 0, "raw_rows": 0, "sales_rows": 0, "bad_rows": 0, "done": False,
          "src_size": os.path.getsize(SRC)}
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        if st.get("done"):
            print(json.dumps(st)); return

    f = open(SRC, "rb")
    # header (always re-read from 0 for column mapping)
    hdr_line = f.readline()
    hdr = next(csv.reader(io.StringIO(hdr_line.decode("utf-8"))))
    idx = {c: hdr.index(c) for c in KEEP}
    gi = hdr.index("trans_group_en")
    ncol = len(hdr)

    if st["offset"] == 0:
        st["offset"] = f.tell()
        with open(OUT, "w", newline="") as o:
            csv.writer(o).writerow(KEEP)
    else:
        f.seek(st["offset"])

    out = open(OUT, "a", newline="")
    w = csv.writer(out)
    buf = b""
    while time.time() - t0 < BUDGET:
        blk = f.read(CHUNK)
        if not blk:
            st["done"] = True
            break
        buf += blk
        nl = buf.rfind(b"\n")
        if nl < 0:
            continue
        chunk, buf = buf[:nl + 1], buf[nl + 1:]
        st["offset"] += len(chunk)
        for row in csv.reader(io.StringIO(chunk.decode("utf-8", "replace"))):
            if not row:
                continue
            st["raw_rows"] += 1
            if len(row) != ncol:
                st["bad_rows"] += 1
                continue
            if row[gi] == "Sales":
                w.writerow([row[idx[c]] for c in KEEP])
                st["sales_rows"] += 1
    # if we exited on EOF with a leftover buffer, it has no trailing newline
    if st.get("done") and buf.strip():
        for row in csv.reader(io.StringIO(buf.decode("utf-8", "replace"))):
            if row:
                st["raw_rows"] += 1
                if len(row) == ncol and row[gi] == "Sales":
                    w.writerow([row[idx[c]] for c in KEEP]); st["sales_rows"] += 1
        st["offset"] += len(buf)
    out.close()
    json.dump(st, open(STATE, "w"))
    st["pct"] = round(100.0 * st["offset"] / st["src_size"], 1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
