import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from profiling.labels import GROUP, category, is_region

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
HOST_CATS = {"cpu_op", "user_annotation"}


def load_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def regions(events):
    return sorted((e for e in events if e.get("cat") == "user_annotation" and is_region(e["name"])
                   and ":" not in e["name"]), key=lambda e: e["ts"])


def gpu_kernels(events):
    """Each GPU kernel/memcpy/memset with the host ranges open when it was launched.

    Kernels are matched to their launch call (cudaLaunchKernel etc.) by the CUPTI correlation id,
    and the launch call to the enclosing cpu_op / record_function ranges on its thread by time.
    """
    host = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and (e.get("cat") in HOST_CATS or e.get("cat") in LAUNCH_CATS):
            host[(e["pid"], e["tid"])].append(e)

    stacks = {}
    for evs in host.values():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        stack = []
        for e in evs:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
                stack.pop()
            if e["cat"] in LAUNCH_CATS:
                stacks[e["args"]["correlation"]] = list(stack)
            else:
                stack.append(e)

    out = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in KERNEL_CATS:
            stack = stacks.get(e["args"].get("correlation"), [])
            labels = [s for s in stack if s["cat"] == "user_annotation" and is_region(s["name"])]
            out.append({
                "name": e["name"], "ts": e["ts"], "dur": e["dur"], "stack": stack,
                "region": next((s for s in labels if ":" not in s["name"]), None),
                "category": category(labels[-1]["name"] if labels else None),
            })
    return out


def category_rows(kernels, region_events):
    # one table per region name (prefill_b1, decode_b1, ...)
    steps = defaultdict(int)
    for r in region_events:
        steps[r["name"]] += 1
    acc = defaultdict(lambda: [0.0, 0])
    for k in kernels:
        name = k["region"]["name"] if k["region"] else "none"
        acc[(name, k["category"])][0] += k["dur"]
        acc[(name, k["category"])][1] += 1

    rows = []
    for name in sorted({n for n, _ in acc}):
        n_steps = steps.get(name, 1)
        total_us = sum(v[0] for (n, _), v in acc.items() if n == name)
        total_k = sum(v[1] for (n, _), v in acc.items() if n == name)
        cats = sorted(((c, v) for (n, c), v in acc.items() if n == name), key=lambda x: -x[1][0])
        for cat, (us, cnt) in cats + [("total", (total_us, total_k))]:
            rows.append({
                "region": name, "steps": n_steps, "category": cat, "group": GROUP.get(cat, ""),
                "gpu_us": round(us, 2), "share_pct": round(100 * us / total_us, 2) if total_us else 0,
                "kernels": cnt, "kernels_per_step": round(cnt / n_steps, 2),
                "gpu_us_per_step": round(us / n_steps, 2),
            })
    return rows


def busy_time(intervals):
    busy, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e > end:
            busy += e - max(s, end)
            end = e
    return busy


def step_rows(kernels, region_events):
    # wall time on the host vs time the GPU was busy, per region instance (each decode step)
    by_region = defaultdict(list)
    for k in kernels:
        if k["region"] is not None:
            by_region[id(k["region"])].append(k)
    rows = []
    idx = defaultdict(int)
    for r in region_events:
        ks = sorted(by_region.get(id(r), []), key=lambda k: k["ts"])
        i = idx[r["name"]]
        idx[r["name"]] += 1
        if not ks:
            continue
        span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
        busy = busy_time([(k["ts"], k["ts"] + k["dur"]) for k in ks])
        gaps = [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(ks, ks[1:])]
        rows.append({
            "region": r["name"], "index": i, "host_wall_us": round(r["dur"], 1),
            "gpu_span_us": round(span, 1), "gpu_busy_us": round(busy, 1),
            "gpu_idle_us": round(span - busy, 1), "busy_pct_of_wall": round(100 * busy / r["dur"], 1),
            "kernels": len(ks), "max_gap_us": round(max(gaps, default=0), 1),
        })
    return rows


def folded_stacks(kernels):
    # flamegraph.pl / speedscope "collapsed" format, weights in ns of GPU time
    acc = defaultdict(int)
    for k in kernels:
        frames = [s["name"].split(":", 1)[1] if is_region(s["name"]) and ":" in s["name"] else s["name"]
                  for s in k["stack"]]
        frames.append(k["name"][:160])
        acc[";".join(f.replace(";", ",") for f in frames)] += round(k["dur"] * 1000)
    return [f"{stack} {w}" for stack, w in sorted(acc.items()) if w > 0]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(trace, out_dir, name):
    events = load_events(trace)
    regs = regions(events)
    kernels = gpu_kernels(events)
    cats, steps = category_rows(kernels, regs), step_rows(kernels, regs)
    out_dir = Path(out_dir)
    write_csv(out_dir / f"{name}_categories.csv", cats)
    write_csv(out_dir / f"{name}_steps.csv", steps)
    (out_dir / f"{name}_kernels.folded").write_text("\n".join(folded_stacks(kernels)) + "\n")
    return cats, steps


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="re-run the category/step/flamegraph analysis on a saved trace")
    p.add_argument("trace")
    p.add_argument("--out", default=".")
    args = p.parse_args()
    name = Path(args.trace).name.split(".")[0]
    cats, _ = analyze(args.trace, args.out, name)
    for r in cats:
        print(f"{r['region']:12s} {r['category']:13s} {r['gpu_us_per_step']:10.1f} us/step "
              f"{r['share_pct']:6.2f}%  {r['kernels_per_step']:7.1f} kernels/step")
