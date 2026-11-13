import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from profiling.trace_analysis import load_events


# optimized traces have no per-op labels (graph replay skips them), so group kernels by name
def group(name):
    n = name.lower()
    if re.search(r"flash|fmha|attention", n):  # check first, flash kernel names contain "cutlass"
        return "attention"
    if "catarraybatchedcopy" in n or "index_copy" in n:  # DynamicCache cat, static cache write
        return "kv_cache"
    if re.search(r"gemm|gemv|cutlass|cublas", n):
        return "gemm"
    return "nongemm"


def split(trace, steps):
    time, count = Counter(), Counter()
    for e in load_events(trace):
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            g = group(e["name"])
            time[g] += e["dur"]
            count[g] += 1
    total = sum(time.values())
    row = {"gpu_ms_per_step": round(total / steps / 1000, 2), "kernels_per_step": round(sum(count.values()) / steps)}
    for g in ("gemm", "nongemm", "attention", "kv_cache"):
        row[f"{g}_pct"] = round(100 * time[g] / total, 1)
    row["nongemm_ms_per_step"] = round(time["nongemm"] / steps / 1000, 2)
    return row


def main():
    p = argparse.ArgumentParser(description="GPU time split into gemm / non-gemm / attention / kv cache by kernel name")
    p.add_argument("--run", nargs=3, action="append", metavar=("VERSION", "TRACE_DIR", "PREFIX"), required=True,
                   help="e.g. baseline profiles/baseline_20261008/torch ''")
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for version, trace_dir, prefix in args.run:
        for region in ("prefill_b1", "prefill_b8", "decode_b1", "decode_b8"):
            steps = args.decode_steps if region.startswith("decode") else 1
            trace = Path(trace_dir) / f"{prefix}{region}.json.gz"
            rows.append({"region": region, "version": version, **split(trace, steps)})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
