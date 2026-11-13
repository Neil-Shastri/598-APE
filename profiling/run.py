import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from olmo_bench.benchmark import apply_opts, make_runner, parse_opts, setup_runner
from olmo_bench.model import MODEL_ID, REVISION, load_model
from olmo_bench.workload import make_input_ids
from profiling.labels import Labels, region
from profiling.trace_analysis import analyze, write_csv


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_prefill(runner, ids, labels, nvtx):
    name = f"prefill_b{ids.shape[0]}"
    labels.phase = name
    with region(name, nvtx):
        token = runner.prefill(ids)
        sync(ids.device)
    return token


def run_decode(runner, token, steps, labels, nvtx, suffix=""):
    # one range per step; the sync is inside it so the range covers the step's GPU work
    name = f"decode_b{token.shape[0]}{suffix}"
    labels.phase = name
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        with region(name, nvtx):
            token = runner.step(token)
            sync(token.device)
        times.append(time.perf_counter() - t0)
    return token, times


def warmup(runners, inputs, args, labels, nvtx):
    for runner, ids in zip(runners, inputs):
        for _ in range(args.warmup):
            token = run_prefill(runner, ids, labels, nvtx)
            run_decode(runner, token, args.decode_steps, labels, nvtx)


def label_overhead(runners, inputs, args, labels):
    # median decode step time with and without labels (no profiler attached)
    rows = []
    for runner, ids in zip(runners, inputs):
        res = {}
        for on in (False, True):
            if on:
                labels.enable()
            token = run_prefill(runner, ids, labels, False)
            res[on] = run_decode(runner, token, args.decode_steps, labels, False)[1]
            if on:
                labels.disable()
        off_ms, on_ms = statistics.median(res[False]) * 1e3, statistics.median(res[True]) * 1e3
        rows.append({"batch": ids.shape[0], "steps": args.decode_steps, "median_step_ms_no_labels": round(off_ms, 3),
                     "median_step_ms_labels": round(on_ms, 3), "overhead_pct": round(100 * (on_ms / off_ms - 1), 2)})
    return rows


def save_profile(prof, name, device, out, raw):
    dev = "cuda" if device.type == "cuda" else "cpu"
    table = prof.key_averages()
    (out / f"{name}_ops_self.txt").write_text(table.table(sort_by=f"self_{dev}_time_total", row_limit=60))
    (out / f"{name}_ops_total.txt").write_text(table.table(sort_by=f"{dev}_time_total", row_limit=60))
    trace = raw / f"{name}.json.gz"
    prof.export_chrome_trace(str(trace))
    return analyze(trace, out, name)


def torch_tool(runners, inputs, args, labels, device, use_labels):
    out, raw = Path(args.out), Path(args.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])

    if use_labels:
        labels.disable()
        write_csv(out / "label_overhead.csv", label_overhead(runners, inputs, args, labels))
        labels.enable()

    all_cats, all_steps = [], []
    for runner, ids in zip(runners, inputs):
        b = ids.shape[0]
        with profile(activities=acts) as prof:
            token = run_prefill(runner, ids, labels, False)
        cats, _ = save_profile(prof, f"prefill_b{b}", device, out, raw)
        all_cats += cats
        with profile(activities=acts) as prof:
            run_decode(runner, token, args.decode_steps, labels, False)
        cats, steps = save_profile(prof, f"decode_b{b}", device, out, raw)
        all_cats += cats
        all_steps += steps
    write_csv(out / "categories.csv", all_cats)
    write_csv(out / "decode_steps.csv", all_steps)
    for r in all_cats:
        print(f"{r['region']:12s} {r['category']:13s} {r['gpu_us_per_step']:10.1f} us/step "
              f"{r['share_pct']:6.2f}%  {r['kernels_per_step']:7.1f} kernels/step")


def nsys_tool(runners, inputs, args, labels):
    # nsys runs with --capture-range=cudaProfilerApi, so only this part is recorded
    torch.cuda.profiler.start()
    for runner, ids in zip(runners, inputs):
        token = run_prefill(runner, ids, labels, True)
        run_decode(runner, token, args.decode_steps, labels, True)
    # same steps without the per-op labels, for launch gaps without label overhead
    labels.disable()
    for runner, ids in zip(runners, inputs):
        token = runner.prefill(ids)
        run_decode(runner, token, args.decode_steps, labels, True, suffix="_nolabels")
    torch.cuda.profiler.stop()


def ncu_tool(model, runner, inputs, args, labels):
    # ncu runs with --profile-from-start off: profile only one decoder layer of one prefill
    # and of one steady-state decode step
    layer = model.model.layers[args.ncu_layer]
    armed = [False]

    def start(*_):
        if armed[0]:
            torch.cuda.profiler.start()

    def stop(*_):
        if armed[0]:
            torch.cuda.profiler.stop()
            armed[0] = False

    layer.register_forward_pre_hook(start)
    layer.register_forward_hook(stop)
    ids = inputs[0]
    armed[0] = True
    token = run_prefill(runner, ids, labels, True)
    for _ in range(args.ncu_step):
        token, _ = run_decode(runner, token, 1, labels, True)
    armed[0] = True
    run_decode(runner, token, 1, labels, True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["torch", "nsys", "ncu"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batches", default="1,8")
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--ncu-layer", type=int, default=3, help="layer 3 is a full-attention layer")
    p.add_argument("--ncu-step", type=int, default=4, help="decode steps to run before the profiled one")
    p.add_argument("--out", help="small outputs (torch tool)")
    p.add_argument("--raw", help="traces (torch tool)")
    p.add_argument("--opt", default="", help="same as benchmark.py; per-op labels are off when set")
    args = p.parse_args()
    opts = parse_opts(args.opt)
    if "graph" in opts and args.tool == "ncu":
        raise SystemExit("the ncu tool hooks a layer's forward, which doesn't run during graph replay")

    device = torch.device(args.device)
    model, tokenizer = load_model(args.device, getattr(torch, args.dtype), args.model_id, args.revision)
    print(f"attn_implementation={model.config._attn_implementation}", flush=True)
    inputs = [make_input_ids(tokenizer, int(b), args.seq_len, device) for b in args.batches.split(",")]
    # the fused kernels skip the norm/RoPE modules and graph replay skips Python entirely,
    # so the per-op labels would be wrong or empty; optimized runs only keep the step ranges
    apply_opts(model, opts)
    max_len = args.seq_len + max(args.decode_steps, args.ncu_step + 1) + 1
    runners = [make_runner(model, opts, ids.shape[0], max_len) for ids in inputs]
    nvtx = args.tool != "torch"
    labels = Labels(model, nvtx=nvtx)
    use_labels = not opts
    if use_labels:
        labels.enable()
    with torch.no_grad():
        for runner, ids in zip(runners, inputs):
            setup_runner(runner, ids)
        warmup(runners, inputs, args, labels, nvtx)
        if args.tool == "torch":
            torch_tool(runners, inputs, args, labels, device, use_labels)
        elif args.tool == "nsys":
            nsys_tool(runners, inputs, args, labels)
        else:
            ncu_tool(model, runners[0], inputs, args, labels)
    labels.disable()
    print("done", flush=True)


if __name__ == "__main__":
    main()
