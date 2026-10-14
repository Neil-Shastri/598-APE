import argparse
import csv
from collections import defaultdict
from pathlib import Path

from profiling.labels import GROUP, category
from profiling.trace_analysis import write_csv

CEILING_GBS = 578  # measured useful bandwidth on the A40, docs/platform.md

METRICS = {
    "duration": "gpu__time_duration.sum",
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "regs": "launch__registers_per_thread",
}
# to microseconds / bytes; --print-units base should give nsecond and byte
SCALE = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1, "us": 1, "msecond": 1e3, "ms": 1e3,
         "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}


def read_raw(path):
    # `ncu --csv --page raw`: header row, units row, then one row per kernel
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    start = next(i for i, r in enumerate(rows) if "ID" in r)
    header, rest = rows[start], rows[start + 1:]
    units = dict(zip(header, rest[0])) if rest and rest[0][header.index("ID")] == "" else {}
    data = [dict(zip(header, r)) for r in rest if r and r[header.index("ID")] != ""]
    return data, units


def number(row, units, metric, scale=1.0):
    v = row.get(metric, "").replace(",", "")
    if v in ("", "n/a"):
        return None
    return float(v) * SCALE.get(units.get(metric, ""), 1) / scale


def short_name(name):
    return name.removeprefix("void ").replace("at::native::", "").replace("(anonymous namespace)::", "")[:110]


def kernel_rows(raw_csv, nvtx_csv):
    data, units = read_raw(raw_csv)
    renamed, _ = read_raw(nvtx_csv)
    # --print-nvtx-rename kernel gives "<innermost range>/<kernel name>"
    label = {r["ID"]: r["Kernel Name"].split("/", 1)[0] for r in renamed}
    rows = []
    for r in data:
        lab = label.get(r["ID"])
        cat = category(lab)
        us = number(r, units, METRICS["duration"])  # microseconds
        rd = number(r, units, METRICS["dram_read"], 1e6)  # MB
        wr = number(r, units, METRICS["dram_write"], 1e6)
        mb = rd + wr if rd is not None and wr is not None else None
        bw = mb / us * 1e3 if mb is not None and us else None  # GB/s
        rows.append({
            "id": r["ID"], "region": lab.split(":")[0] if lab else "", "category": cat,
            "group": GROUP.get(cat, ""), "kernel": short_name(r["Kernel Name"]),
            "duration_us": us, "dram_read_mb": rd, "dram_write_mb": wr, "dram_mb": mb,
            "bw_gbs": bw, "pct_of_578": 100 * bw / CEILING_GBS if bw is not None else None,
            **{k: number(r, units, METRICS[k]) for k in
               ("sm_pct", "mem_pct", "dram_pct", "l2_hit_pct", "occupancy_pct", "grid", "block", "regs")},
            "kernel_full": r["Kernel Name"],
        })
    return [{k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()} for row in rows]


def rollup(rows, key):
    # sums per (region, key); l2 hit rate and occupancy are duration-weighted means
    acc = defaultdict(lambda: {"kernels": 0, "us": 0.0, "mb": 0.0, "l2": 0.0, "occ": 0.0})
    for r in rows:
        a = acc[(r["region"], r[key])]
        us = r["duration_us"] or 0
        a["kernels"] += 1
        a["us"] += us
        a["mb"] += r["dram_mb"] or 0
        a["l2"] += (r["l2_hit_pct"] or 0) * us
        a["occ"] += (r["occupancy_pct"] or 0) * us
    totals = defaultdict(float)
    for (reg, _), a in acc.items():
        totals[reg] += a["us"]
    out = []
    for (reg, name), a in sorted(acc.items(), key=lambda x: (x[0][0], -x[1]["us"])):
        us = a["us"]
        bw = a["mb"] / us * 1e3 if us else 0
        row = {"region": reg, key: name}
        if key == "category":
            row["group"] = GROUP.get(name, "")
        out.append(row | {
            "kernels": a["kernels"], "duration_us": round(us, 2),
            "share_pct": round(100 * us / totals[reg], 2) if totals[reg] else 0,
            "avg_kernel_us": round(us / a["kernels"], 2), "dram_mb": round(a["mb"], 3),
            "bw_gbs": round(bw, 1), "pct_of_578": round(100 * bw / CEILING_GBS, 1),
            "l2_hit_pct": round(a["l2"] / us, 1) if us else None,
            "occupancy_pct": round(a["occ"] / us, 1) if us else None,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_csv", help="ncu --import X --csv --page raw --print-units base")
    p.add_argument("nvtx_csv", help="the same with --print-nvtx-rename kernel")
    p.add_argument("--out", default=".")
    args = p.parse_args()
    rows = kernel_rows(args.raw_csv, args.nvtx_csv)
    summary = rollup(rows, "category")
    write_csv(Path(args.out) / "ncu_kernels.csv", rows)
    write_csv(Path(args.out) / "ncu_categories.csv", summary)
    write_csv(Path(args.out) / "ncu_groups.csv", rollup(rows, "group"))
    for r in summary:
        print(f"{r['region']:10s} {r['category']:13s} {r['kernels']:3d} kernels {r['duration_us']:9.1f} us "
              f"{r['dram_mb']:9.2f} MB {r['bw_gbs']:7.1f} GB/s ({r['pct_of_578']:.0f}% of 578)")


if __name__ == "__main__":
    main()
