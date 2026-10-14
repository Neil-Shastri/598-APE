import argparse
import csv
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from profiling.labels import is_region
from profiling.trace_analysis import busy_time, write_csv


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    # nsys stats headers carry the unit, e.g. "Start (ns)"
    return next(v for k, v in row.items() if k == prefix or k.startswith(prefix + " ("))


def step_rows(kern_csv, nvtx_csv):
    # both reports exported with --timeunit nsec
    kernels = sorted((float(col(r, "Start")), float(col(r, "Duration"))) for r in read(kern_csv))
    starts = [s for s, _ in kernels]
    ranges = []
    for r in read(nvtx_csv):
        # nsys prints "<domain>:<range>"; our ranges are in the default (empty) domain
        r["Name"] = r["Name"].removeprefix(":")
        if is_region(r["Name"]) and ":" not in r["Name"]:
            ranges.append(r)
    ranges.sort(key=lambda r: float(col(r, "Projected Start")))
    rows, idx = [], defaultdict(int)
    for r in ranges:
        p0 = float(col(r, "Projected Start"))
        p1 = p0 + float(col(r, "Projected Duration"))
        ks = kernels[bisect_left(starts, p0):bisect_left(starts, p1)]
        if not ks:
            continue
        busy = busy_time([(s, s + d) for s, d in ks])
        gaps = [b[0] - (a[0] + a[1]) for a, b in zip(ks, ks[1:])]
        wall = float(col(r, "Orig Duration"))
        rows.append({
            "region": r["Name"], "index": idx[r["Name"]], "host_wall_us": round(wall / 1e3, 1),
            "gpu_span_us": round((p1 - p0) / 1e3, 1), "gpu_busy_us": round(busy / 1e3, 1),
            "gpu_idle_us": round((p1 - p0 - busy) / 1e3, 1), "busy_pct_of_wall": round(100 * busy / wall, 1),
            "kernels": len(ks), "median_gap_us": round(statistics.median(gaps) / 1e3, 2) if gaps else 0,
            "max_gap_us": round(max(gaps, default=0) / 1e3, 1),
        })
        idx[r["Name"]] += 1
    return rows


def summary_rows(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["region"]].append(r)
    keys = ["host_wall_us", "gpu_span_us", "gpu_busy_us", "gpu_idle_us", "busy_pct_of_wall", "kernels",
            "median_gap_us"]
    return [{"region": name, "n": len(rs),
             **{f"median_{k}": round(statistics.median(r[k] for r in rs), 2) for k in keys}}
            for name, rs in by.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kern_csv", help="nsys stats -r cuda_gpu_trace --timeunit nsec -f csv")
    p.add_argument("nvtx_csv", help="nsys stats -r nvtx_gpu_proj_trace --timeunit nsec -f csv")
    p.add_argument("--out", default=".")
    args = p.parse_args()
    rows = step_rows(args.kern_csv, args.nvtx_csv)
    if not rows:
        raise SystemExit(f"no prefill_b*/decode_b* ranges with kernels in {args.nvtx_csv}")
    summary = summary_rows(rows)
    write_csv(Path(args.out) / "nsys_steps.csv", rows)
    write_csv(Path(args.out) / "nsys_steps_summary.csv", summary)
    for r in summary:
        print(", ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
