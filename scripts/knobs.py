"""GPU microbenchmarks of configuration knobs on synthetic tensors with the real OLMo 3 shapes (no model load).

Run inside a job from the repo root:  python scripts/knobs.py [--quick] [--dry-run]
Every timing captures `inner` calls in one CUDA graph (as production replays them) and reports the median
per-call time over the replays. Writes results/knobs/knobs_<SLURM_JOB_ID or local>.json.
"""
import argparse
import json
import os
import random
import statistics
import sys
import traceback
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from olmo_bench import kernels  # noqa: E402

HIDDEN, INTER, HEADS, HEAD_DIM = 4096, 11008, 32, 128
GEMM_SHAPES = [("o_proj", 4096, 4096), ("gate/up", 11008, 4096), ("qkv merged", 12288, 4096),
               ("gate_up merged", 22016, 4096), ("down_proj", 4096, 11008)]  # (name, N, K)
GEMM_M = [1, 8, 128, 512, 2048]
GEMM_KNOBS = ["default", "cublaslt", "no_bf16_rpr"]
WS_M = [128, 512]
WS_SIZES = [1 << 20, 8 << 20]
SDPA_SHAPES = [(1, 640), (8, 640), (1, 4096), (8, 4096), (1, 8320), (8, 8320)]  # (batch, kv slots)
SDPA_MASKS = ["bool", "additive", "none"]
SDPA_BACKENDS = {
    "default": None,
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "math": [SDPBackend.MATH],
    "math_bf16_reduction": [SDPBackend.MATH],
    "flash": [SDPBackend.FLASH_ATTENTION],
}
TRITON_ROWS = [1, 8, 512, 2048]
TRITON_WARPS = [2, 4, 8, 16]
TRITON_DEFAULT_WARPS = {"rmsnorm": 8, "qk_norm_rope": 8, "silu_mul": 4}

WARMUP, ITERS = 5, 50
DEV = "cuda"
BF = torch.bfloat16


def timed(fn, inner):
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(WARMUP):
        graph.replay()
    # all replays are queued before the first sync so the GPU never waits on the host between them
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    us = [start.elapsed_time(end) * 1000 / inner for start, end in events]
    del graph
    return {"us": statistics.median(us), "min_us": min(us)}


def attempt(row, body):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            row.update(body())
            row["status"] = "ok"
        except Exception as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
            row["status"] = "unsupported" if "No available kernel" in msg else "error"
            row["error"] = f"{type(e).__name__}: {msg}"
            row["warnings"] = [str(w.message)[:300] for w in caught][:6]
    return row


def add_ratio(rows, group, is_base):
    base = {tuple(r[g] for g in group): r["us"] for r in rows if is_base(r) and "us" in r}
    for r in rows:
        b = base.get(tuple(r[g] for g in group))
        if b and "us" in r:
            r["x_default"] = r["us"] / b


@contextmanager
def blas_knob(knob, workspace=None):
    cuda = torch.backends.cuda
    old = cuda.preferred_blas_library(), cuda.matmul.allow_bf16_reduced_precision_reduction, cuda.cublaslt_workspace_size()
    try:
        if knob == "cublaslt":
            cuda.preferred_blas_library("cublaslt")
        if knob == "no_bf16_rpr":
            cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if workspace is not None:
            # the Lt buffer is shared with cuBLAS (8 MiB + 128 KiB on sm_86); this only changes what Lt may use
            cuda.cublaslt_workspace_size(workspace)
        yield
    finally:
        cuda.preferred_blas_library(old[0])
        cuda.matmul.allow_bf16_reduced_precision_reduction = old[1]
        cuda.cublaslt_workspace_size(old[2])


def gemm_rows(rows, ms, knobs):
    for name, n, k in GEMM_SHAPES:
        w = torch.randn(n, k, device=DEV, dtype=BF) / k ** 0.5
        for m in ms:
            x = torch.randn(m, k, device=DEV, dtype=BF)
            ref = F.linear(x, w).float()
            for knob, ws in knobs:
                row = {"op": name, "N": n, "K": k, "M": m, "knob": knob}
                if ws is not None:
                    row["workspace_MiB"] = ws >> 20

                def body():
                    with blas_knob(knob, ws):
                        out = F.linear(x, w)
                        t = timed(lambda: F.linear(x, w), 20 if m <= 128 else 5)
                    t["TFLOP/s"] = 2 * m * n * k / t["us"] / 1e6
                    t["max_abs_diff_vs_default"] = (out.float() - ref).abs().max().item()
                    return t

                rows.append(attempt(row, body))
        del w
        torch.cuda.empty_cache()


def gemm_section(rows):
    gemm_rows(rows, GEMM_M, [(knob, None) for knob in GEMM_KNOBS])
    add_ratio(rows, ["N", "K", "M"], lambda r: r["knob"] == "default")


def workspace_section(rows):
    gemm_rows(rows, WS_M, [("cublaslt", ws) for ws in WS_SIZES])
    add_ratio(rows, ["N", "K", "M"], lambda r: r["workspace_MiB"] == 1)


def sdpa_section(rows):
    cuda = torch.backends.cuda
    for b, n in SDPA_SHAPES:
        q = torch.randn(b, HEADS, 1, HEAD_DIM, device=DEV, dtype=BF)
        k = torch.randn(b, HEADS, n, HEAD_DIM, device=DEV, dtype=BF)
        v = torch.randn(b, HEADS, n, HEAD_DIM, device=DEV, dtype=BF)
        # like StaticKV.step_masks: slots up to the current position are valid, the last 128 are not yet written
        valid = (torch.arange(n, device=DEV) < n - 128).view(1, 1, 1, n)
        masks = {
            "bool": valid,
            "additive": torch.zeros(valid.shape, device=DEV, dtype=BF).masked_fill(~valid, float("-inf")),
            "none": None,
        }
        for mask_name in SDPA_MASKS:
            mask = masks[mask_name]
            ref_mask = mask.float() if mask_name == "additive" else mask
            with sdpa_kernel([SDPBackend.MATH]):
                ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=ref_mask)
            for backend, chosen in SDPA_BACKENDS.items():
                row = {"B": b, "n": n, "mask": mask_name, "backend": backend}

                def body():
                    old = cuda.fp16_bf16_reduction_math_sdp_allowed()
                    cuda.allow_fp16_bf16_reduction_math_sdp(backend == "math_bf16_reduction")
                    try:
                        with nullcontext() if chosen is None else sdpa_kernel(chosen):
                            fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            diff = (fn().float() - ref).abs().max().item()
                            t = timed(fn, 8)
                    finally:
                        cuda.allow_fp16_bf16_reduction_math_sdp(old)
                    t["KV GB/s"] = 2 * k.numel() * k.element_size() / t["us"] / 1e3
                    t["max_abs_diff_vs_math_fp32"] = diff
                    return t

                rows.append(attempt(row, body))
        del q, k, v, masks, ref
        torch.cuda.empty_cache()
    add_ratio(rows, ["B", "n", "mask"], lambda r: r["backend"] == "default")


def triton_section(rows):
    if not kernels.available():
        raise RuntimeError("triton or CUDA is missing")
    eps = 1e-6
    for r in TRITON_ROWS:
        # decode: r sequences of one token; prefill: one sequence of r tokens
        s = 1 if r <= 8 else r
        x, res, out = (torch.randn(r, HIDDEN, device=DEV, dtype=BF) for _ in range(3))
        w, qw, kw = (torch.randn(HIDDEN, device=DEV, dtype=BF) for _ in range(3))
        qkv = torch.randn(r, 3 * HIDDEN, device=DEV, dtype=BF)  # merged projection output, as in production
        q, k = qkv[:, :HIDDEN], qkv[:, HIDDEN:2 * HIDDEN]
        qo, ko = torch.empty(r, HIDDEN, device=DEV, dtype=BF), torch.empty(r, HIDDEN, device=DEV, dtype=BF)
        cos, sin = torch.randn(1, s, HEAD_DIM, device=DEV), torch.randn(1, s, HEAD_DIM, device=DEV)
        gate_up = torch.randn(r, 2 * INTER, device=DEV, dtype=BF)
        g, u = gate_up[:, :INTER], gate_up[:, INTER:]
        act = torch.empty(r, INTER, device=DEV, dtype=BF)
        launches = {
            "rmsnorm": (3 * r * HIDDEN * 2, lambda nw: kernels._rmsnorm_kernel[(r,)](
                x, w, res, out, eps, N=HIDDEN, HAS_RES=True, num_warps=nw)),
            "qk_norm_rope": (4 * r * HIDDEN * 2, lambda nw: kernels._qk_norm_rope_kernel[(r, 2)](
                q, k, q.stride(0), k.stride(0), qw, kw, cos, sin, qo, ko, s, 0, eps,
                H=HEADS, D=HEAD_DIM, num_warps=nw, enable_fp_fusion=False)),
            "silu_mul": (3 * r * INTER * 2, lambda nw: kernels._silu_mul_kernel[(r, kernels.triton.cdiv(INTER, 1024))](
                g, u, act, g.stride(0), u.stride(0), INTER, BLOCK=1024, num_warps=nw)),
        }
        for name, (nbytes, launch) in launches.items():
            for nw in TRITON_WARPS:
                row = {"kernel": name, "rows": r, "num_warps": nw}

                def body():
                    t = timed(lambda: launch(nw), 100)
                    t["GB/s"] = nbytes / t["us"] / 1e3
                    return t

                rows.append(attempt(row, body))
    add_ratio(rows, ["kernel", "rows"], lambda r: r["num_warps"] == TRITON_DEFAULT_WARPS[r["kernel"]])


def tf32_section(rows):
    # the model's only fp32 matmul is RoPE's inv_freq [64, 1] @ positions [1, S]; K = 1
    inv_freq = 1.0 / 500000.0 ** (torch.arange(0, HEAD_DIM, 2, device=DEV).float() / HEAD_DIM)
    a = inv_freq.view(1, -1, 1)
    p = torch.arange(8192, device=DEV).float().view(1, 1, -1)
    exact = a.double() @ p.double()
    old = torch.backends.cuda.matmul.allow_tf32
    for tf32 in (False, True):
        row = {"op": "rope inv_freq @ pos", "S": 8192, "allow_tf32": tf32}

        def body():
            torch.backends.cuda.matmul.allow_tf32 = tf32
            try:
                t = timed(lambda: a @ p, 20)
                t["max_abs_diff_vs_fp64"] = ((a @ p).double() - exact).abs().max().item()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old
            return t

        rows.append(attempt(row, body))
    add_ratio(rows, ["op"], lambda r: not r["allow_tf32"])


SECTIONS = [("gemm", gemm_section), ("workspace", workspace_section), ("sdpa", sdpa_section),
            ("triton", triton_section), ("tf32", tf32_section)]

COLUMNS = {
    "gemm": ["op", "N", "K", "M", "knob", "status", "us", "x_default", "TFLOP/s", "max_abs_diff_vs_default"],
    "workspace": ["op", "N", "K", "M", "workspace_MiB", "status", "us", "x_default", "TFLOP/s"],
    "sdpa": ["B", "n", "mask", "backend", "status", "us", "x_default", "KV GB/s", "max_abs_diff_vs_math_fp32"],
    "triton": ["kernel", "rows", "num_warps", "status", "us", "x_default", "GB/s"],
    "tf32": ["op", "S", "allow_tf32", "status", "us", "x_default", "max_abs_diff_vs_fp64"],
}


def fake_results(results):
    # same rows as a real run with random numbers, to exercise the table and JSON code without a GPU
    def ok(row, **extra):
        row.update(status="ok", us=random.uniform(2, 500), min_us=1.0, **extra)
        return row

    for name, n, k in GEMM_SHAPES:
        for m in GEMM_M:
            results["gemm"] += [ok({"op": name, "N": n, "K": k, "M": m, "knob": kn}, **{"TFLOP/s": 50.0,
                                "max_abs_diff_vs_default": 0.0}) for kn in GEMM_KNOBS]
        for m in WS_M:
            results["workspace"] += [ok({"op": name, "N": n, "K": k, "M": m, "knob": "cublaslt",
                                         "workspace_MiB": ws >> 20}, **{"TFLOP/s": 50.0}) for ws in WS_SIZES]
    for b, n in SDPA_SHAPES:
        for mask in SDPA_MASKS:
            for backend in SDPA_BACKENDS:
                row = {"B": b, "n": n, "mask": mask, "backend": backend}
                if backend == "flash" and mask != "none":
                    row.update(status="unsupported", error="RuntimeError: No available kernel. Aborting execution.")
                    results["sdpa"].append(row)
                else:
                    results["sdpa"].append(ok(row, **{"KV GB/s": 400.0, "max_abs_diff_vs_math_fp32": 1e-3}))
    for kernel in TRITON_DEFAULT_WARPS:
        for r in TRITON_ROWS:
            results["triton"] += [ok({"kernel": kernel, "rows": r, "num_warps": nw}, **{"GB/s": 300.0})
                                  for nw in TRITON_WARPS]
    results["tf32"] += [ok({"op": "rope inv_freq @ pos", "S": 8192, "allow_tf32": t}, max_abs_diff_vs_fp64=0.0)
                        for t in (False, True)]
    add_ratio(results["gemm"], ["N", "K", "M"], lambda r: r["knob"] == "default")
    add_ratio(results["workspace"], ["N", "K", "M"], lambda r: r["workspace_MiB"] == 1)
    add_ratio(results["sdpa"], ["B", "n", "mask"], lambda r: r["backend"] == "default")
    add_ratio(results["triton"], ["kernel", "rows"], lambda r: r["num_warps"] == TRITON_DEFAULT_WARPS[r["kernel"]])
    add_ratio(results["tf32"], ["op"], lambda r: not r["allow_tf32"])


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1e}" if v != 0 and abs(v) < 0.01 else f"{v:.2f}"
    return str(v)


def print_table(title, rows, cols):
    cells = [cols] + [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(cols))]
    print(f"\n== {title}")
    for line in cells:
        print("  ".join(s.rjust(w) for s, w in zip(line, widths)))


def meta(args):
    info = {"job": os.environ.get("SLURM_JOB_ID", "local"), "quick": args.quick, "dry_run": args.dry_run,
            "warmup": WARMUP, "iters": ITERS, "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        cuda = torch.backends.cuda
        info.update(gpu=torch.cuda.get_device_name(), cudnn=torch.backends.cudnn.version(),
                    defaults={"preferred_blas_library": str(cuda.preferred_blas_library()),
                              "allow_bf16_reduced_precision_reduction":
                                  cuda.matmul.allow_bf16_reduced_precision_reduction,
                              "cublaslt_workspace_bytes": cuda.cublaslt_workspace_size(),
                              "allow_tf32": cuda.matmul.allow_tf32,
                              "triton_num_warps": TRITON_DEFAULT_WARPS})
    return info


def main():
    global WARMUP, ITERS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="2 warmup + 10 timed replays per row instead of 5 + 50")
    p.add_argument("--dry-run", action="store_true", help="fake timings, no GPU; checks the table and JSON output")
    args = p.parse_args()
    if args.quick:
        WARMUP, ITERS = 2, 10
    if not args.dry_run and not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU (use --dry-run to check the output code)")

    path = ROOT / "results" / "knobs" / f"knobs_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {"meta": meta(args), "errors": [], **{name: [] for name, _ in SECTIONS}}

    def save():
        path.write_text(json.dumps(results, indent=1))

    if args.dry_run:
        fake_results(results)
    else:
        print(json.dumps(results["meta"]))
        for name, section in SECTIONS:
            try:
                with torch.no_grad():
                    section(results[name])
            except Exception:
                results["errors"].append({"section": name, "error": traceback.format_exc(limit=4)})
                print(f"section {name} failed:\n{results['errors'][-1]['error']}")
            torch.cuda.empty_cache()
            save()
            print(f"done: {name} ({len(results[name])} rows)", flush=True)

    for name, _ in SECTIONS:
        print_table(name, results[name], COLUMNS[name])
    save()
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
