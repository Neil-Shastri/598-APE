import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from olmo_bench import correctness, env_info
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


def timed_prefill(model, input_ids):
    sync(input_ids.device)
    t0 = time.perf_counter()
    token, cache = prefill(model, input_ids)
    sync(input_ids.device)
    return time.perf_counter() - t0, token, cache


def decode_run(model, input_ids, new_tokens):
    # the first new token comes from prefill, the other new_tokens - 1 from decode steps
    ttft, token, cache = timed_prefill(model, input_ids)
    steps = []
    for _ in range(new_tokens - 1):
        t0 = time.perf_counter()
        token, cache = decode_step(model, token, cache)
        sync(input_ids.device)
        steps.append(time.perf_counter() - t0)
    return ttft, steps


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mib(device):
    if device.type != "cuda":
        return None
    return {"allocated": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved": torch.cuda.max_memory_reserved(device) / 2**20}


def bench_prefill(model, tokenizer, device, batch, seq_len, warmup, repeats):
    ids = make_input_ids(tokenizer, batch, seq_len, device, text_for(seq_len))
    reset_peak(device)
    ttft = []
    for i in range(warmup + repeats):
        t, _, _ = timed_prefill(model, ids)
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


def bench_decode(model, tokenizer, device, batch, seq_len, new_tokens, warmup, repeats):
    ids = make_input_ids(tokenizer, batch, seq_len, device, text_for(seq_len))
    reset_peak(device)
    runs = []
    for i in range(warmup + repeats):
        ttft, steps = decode_run(model, ids, new_tokens)
        if i >= warmup:
            runs.append((ttft, steps))
    n = new_tokens - 1
    decode_s = [sum(s) for _, s in runs]
    return {
        "mode": "decode", "batch": batch, "seq_len": seq_len, "new_tokens": new_tokens,
        "decode_steps": n,
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
    args = p.parse_args()

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
    print(json.dumps(meta, indent=2))

    check = None
    if not args.skip_check:
        check = correctness.run_checks(model, tokenizer, device, args.golden_dir, meta)
        correctness.report(check)

    configs = []
    for batch, seq_len in PREFILL:
        configs.append(bench_prefill(model, tokenizer, device, batch, seq_len, args.warmup, args.repeats))
        print_config(configs[-1])
    for batch, seq_len, new_tokens in DECODE:
        configs.append(bench_decode(model, tokenizer, device, batch, seq_len, new_tokens,
                                    args.warmup, args.repeats))
        print_config(configs[-1])
    meta["nvidia_smi_end"] = env_info.gpu_query()
    meta["total_s"] = time.perf_counter() - t0

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + (meta["slurm_job_id"] or "local")
    out_dir = Path(args.out_root) / tag
    write_results(out_dir, meta, check, configs)
    print(f"wrote {out_dir}/bench.json and bench.csv ({meta['total_s']:.0f} s total)")
    if check is not None and not check["passed"]:
        print("WARNING: correctness checks failed, see bench.json")


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
