import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from olmo_bench import correctness, env_info, numerics
from olmo_bench.model import MODEL_ID, REVISION, load_model
from olmo_bench.stats import summarize
from olmo_bench.workload import LONG_TEXT_FILE, TEXT_FILE, decode_step, make_input_ids, prefill

PREFILL = [(1, 128), (1, 2048), (1, 8192)]  # (batch, seq_len)
DECODE = [(1, 512, 128), (8, 512, 128), (1, 8192, 128)]  # (batch, prompt_len, new_tokens)
LONG_PROMPT = 8192  # past the 4096 sliding window


def text_for(seq_len):
    return LONG_TEXT_FILE if seq_len >= LONG_PROMPT else TEXT_FILE


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Eager:
    # the baseline path: stock HF forward with a DynamicCache
    def __init__(self, model):
        self.model = model
        self.cache = None

    def prefill(self, input_ids):
        token, self.cache = prefill(self.model, input_ids)
        return token

    def step(self, token):
        token, self.cache = decode_step(self.model, token, self.cache)
        return token


def make_runner(model, opts, batch, max_len):
    if "static" in opts:
        from olmo_bench.fast_decode import GraphDecoder
        return GraphDecoder(model, batch, max_len, **decoder_args(opts))
    return Eager(model)


def decoder_args(opts):
    return {"use_graph": "graph" in opts, "additive_mask": "mask" in opts, "cached_slots": "slots" in opts}


def timed_prefill(runner, input_ids):
    sync(input_ids.device)
    t0 = time.perf_counter()
    token = runner.prefill(input_ids)
    sync(input_ids.device)
    return time.perf_counter() - t0, token


def decode_run(runner, input_ids, new_tokens):
    # the first new token comes from prefill, the other new_tokens - 1 from decode steps
    ttft, token = timed_prefill(runner, input_ids)
    steps = []
    for _ in range(new_tokens - 1):
        t0 = time.perf_counter()
        token = runner.step(token)
        sync(input_ids.device)
        steps.append(time.perf_counter() - t0)
    return ttft, steps


def setup_runner(runner, input_ids):
    # graph capture happens once per config, outside the timed runs
    if not getattr(runner, "use_graph", False):
        return 0.0
    runner.prefill(input_ids)
    sync(input_ids.device)
    t0 = time.perf_counter()
    runner.capture()
    sync(input_ids.device)
    return time.perf_counter() - t0


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mib(device):
    if device.type != "cuda":
        return None
    return {"allocated": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved": torch.cuda.max_memory_reserved(device) / 2**20}


def bench_prefill(model, tokenizer, device, batch, seq_len, warmup, repeats, opts=()):
    ids = make_input_ids(tokenizer, batch, seq_len, device, text_for(seq_len))
    runner = make_runner(model, opts, batch, seq_len + 1)
    reset_peak(device)
    ttft = []
    for i in range(warmup + repeats):
        t, _ = timed_prefill(runner, ids)
        if i >= warmup:
            ttft.append(t)
    return {
        "mode": "prefill", "batch": batch, "seq_len": seq_len, "new_tokens": 1,
        "raw": {"ttft_s": ttft},
        "metrics": {
            "ttft_ms": summarize([t * 1e3 for t in ttft]),
            "prefill_tok_s": summarize([batch * seq_len / t for t in ttft]),
        },
        "peak_mem_mib": peak_mib(device),
    }


def bench_decode(model, tokenizer, device, batch, seq_len, new_tokens, warmup, repeats, opts=()):
    ids = make_input_ids(tokenizer, batch, seq_len, device, text_for(seq_len))
    runner = make_runner(model, opts, batch, seq_len + new_tokens)
    setup_s = setup_runner(runner, ids)
    reset_peak(device)
    runs = []
    for i in range(warmup + repeats):
        ttft, steps = decode_run(runner, ids, new_tokens)
        if i >= warmup:
            runs.append((ttft, steps))
    n = new_tokens - 1
    decode_s = [sum(s) for _, s in runs]
    return {
        "mode": "decode", "batch": batch, "seq_len": seq_len, "new_tokens": new_tokens,
        "decode_steps": n, "setup_s": setup_s,
        "raw": {"ttft_s": [t for t, _ in runs], "step_s": [s for _, s in runs]},
        "metrics": {
            "ttft_ms": summarize([t * 1e3 for t, _ in runs]),
            "tpot_ms": summarize([d / n * 1e3 for d in decode_s]),
            "decode_tok_s": summarize([batch * n / d for d in decode_s]),
            "e2e_tok_s": summarize([batch * new_tokens / (t + d) for (t, _), d in zip(runs, decode_s)]),
            "step_ms": summarize([x * 1e3 for _, s in runs for x in s]),
        },
        "peak_mem_mib": peak_mib(device),
    }


def write_results(out_dir, meta, check, configs):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(
        {"meta": meta, "correctness": check, "configs": configs}, indent=2))
    fields = ["mode", "batch", "seq_len", "new_tokens", "metric",
              "n", "median", "min", "max", "iqr", "cv", "mean", "peak_alloc_mib"]
    with open(out_dir / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in configs:
            mem = c["peak_mem_mib"]["allocated"] if c["peak_mem_mib"] else None
            for name, s in c["metrics"].items():
                w.writerow({"mode": c["mode"], "batch": c["batch"], "seq_len": c["seq_len"],
                            "new_tokens": c["new_tokens"], "metric": name, **s, "peak_alloc_mib": mem})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default=None)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out-root", default=str(env_info.REPO / "results" / "baseline"))
    p.add_argument("--run-tag", default=None)
    p.add_argument("--golden-dir", default=str(correctness.GOLDEN_DIR))
    p.add_argument("--skip-check", action="store_true")
    p.add_argument("--opt", default="", help="comma list of: fused, static, graph")
    args = p.parse_args()
    opts = parse_opts(args.opt)
    if opts and args.out_root == p.get_default("out_root"):
        args.out_root = str(env_info.REPO / "results" / "experiments")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    model, tokenizer = load_model(args.device, dtype, args.model_id, args.revision, args.attn)
    meta = env_info.collect(model, device, dtype, args.model_id, args.revision)
    meta["load_s"] = time.perf_counter() - t0
    meta["workload"] = {"prefill": PREFILL, "decode": DECODE, "warmup": args.warmup,
                        "repeats": args.repeats,
                        "texts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in (TEXT_FILE, LONG_TEXT_FILE)},
                        "long_text_from": LONG_PROMPT}
    meta["opts"] = sorted(opts)
    print(json.dumps(meta, indent=2))

    check = None
    if not opts:
        if not args.skip_check:
            check = correctness.run_checks(model, tokenizer, device, args.golden_dir, meta)
            correctness.report(check)
    else:
        ref = None if args.skip_check else numerics.snapshot(model, tokenizer, device,
                                                            numerics.eager_generate(model))
        t1 = time.perf_counter()
        apply_opts(model, opts)
        meta["apply_s"] = time.perf_counter() - t1
        if ref is not None:
            new = numerics.snapshot(model, tokenizer, device, opt_generate(model, opts))
            check = numerics.compare(ref, new, numerics.golden_generations(meta))
            numerics.report(check)

    tag = args.run_tag or "_".join([datetime.now().strftime("%Y%m%d_%H%M%S"), meta["slurm_job_id"] or "local"]
                                   + ["-".join(sorted(opts))] * bool(opts))
    out_dir = Path(args.out_root) / tag
    configs = []
    for batch, seq_len in PREFILL:
        configs.append(bench_prefill(model, tokenizer, device, batch, seq_len, args.warmup, args.repeats, opts))
        print_config(configs[-1])
        write_results(out_dir, meta, check, configs)  # keep partial results if the job is cut off
    for batch, seq_len, new_tokens in DECODE:
        configs.append(bench_decode(model, tokenizer, device, batch, seq_len, new_tokens,
                                    args.warmup, args.repeats, opts))
        print_config(configs[-1])
        write_results(out_dir, meta, check, configs)
    meta["nvidia_smi_end"] = env_info.gpu_query()
    meta["total_s"] = time.perf_counter() - t0

    write_results(out_dir, meta, check, configs)
    print(f"wrote {out_dir}/bench.json and bench.csv ({meta['total_s']:.0f} s total)")
    if check is not None and not check["passed"]:
        print("WARNING: correctness checks failed, see bench.json")


def parse_opts(text):
    opts = {o for o in text.split(",") if o}
    unknown = opts - {"fused", "static", "graph", "mask", "slots", "qkv", "gateup", "band", "tunable"}
    if unknown:
        raise SystemExit(f"unknown --opt {sorted(unknown)}")
    if opts & {"graph", "mask", "slots"}:
        opts.add("static")
    if opts & {"qkv", "gateup"} and "fused" not in opts:
        raise SystemExit("--opt qkv/gateup needs fused (the merged weights are read by the fused kernels)")
    return opts


def apply_opts(model, opts):
    if "qkv" in opts:
        from olmo_bench import merge
        merge.merge_qkv(model)
    if "gateup" in opts:
        from olmo_bench import merge
        merge.merge_gateup(model)
    if "fused" in opts:
        from olmo_bench import kernels
        kernels.enable(model)
        kernels.warmup(model)
    if "band" in opts:
        from olmo_bench import band_attention
        band_attention.enable(model)
    if "tunable" in opts:
        enable_tunable()


TUNABLE_CSV = env_info.REPO / "results" / "tunable" / "tunableop_a40.csv"


def enable_tunable():
    # GEMM shapes are tuned the first time they show up (warmups, graph capture warmup) and
    # saved to the csv when the process exits; later runs just read the csv
    from torch.cuda import tunable
    TUNABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.cublaslt_workspace_size(8 << 20)
    tunable.set_filename(str(TUNABLE_CSV))
    tunable.enable(True)
    tune = not TUNABLE_CSV.exists()
    if not tune:
        tunable.read_file()
    tunable.tuning_enable(tune)
    print(f"tunableop: {'tuning, writing' if tune else 'reading'} {TUNABLE_CSV}", flush=True)


def opt_generate(model, opts):
    if "static" not in opts:
        return numerics.eager_generate(model)
    from olmo_bench.fast_decode import GraphDecoder
    args = decoder_args(opts)
    return lambda ids, n: GraphDecoder(model, ids.shape[0], ids.shape[1] + n, **args).greedy_generate(ids, n)


def print_config(c):
    m = c["metrics"]
    line = f"{c['mode']:7s} b={c['batch']} s={c['seq_len']}"
    for name, s in m.items():
        line += f"  {name}={s['median']:.2f} (cv {s['cv']:.3f})"
    if c["peak_mem_mib"]:
        line += f"  peak={c['peak_mem_mib']['allocated']:.0f} MiB"
    print(line, flush=True)


if __name__ == "__main__":
    main()
