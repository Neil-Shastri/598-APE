# 598-project

Profiling non-GEMM ops (RMSNorm, QK-Norm, RoPE, residual adds, activations) in OLMo 3 7B inference on NCSA Delta.

## File structure

```
598-project/
├── README.md
├── requirements.txt
├── olmo_bench/
│   ├── model.py               loads OLMo 3 7B
│   ├── workload.py            prompts, prefill, decode step
│   ├── benchmark.py           TTFT / TPOT / tokens per second
│   ├── correctness.py         output and perplexity checks
│   ├── kernels.py             fused Triton kernels (--opt fused)
│   ├── fast_decode.py         static KV cache + CUDA graph decode (--opt static,graph)
│   ├── numerics.py            accuracy check for optimized runs
│   ├── env_info.py
│   ├── stats.py
│   └── data/
│       ├── scandal_in_bohemia.txt
│       └── long_prompt.txt
├── profiling/
│   ├── labels.py
│   ├── run.py
│   ├── trace_analysis.py
│   ├── nsys_gaps.py
│   ├── ncu_table.py
│   └── test_profiling.py
├── tests/
├── slurm/
│   ├── baseline.sbatch
│   ├── experiment.sbatch      baseline and optimized variants in one job
│   ├── profile_torch.sbatch
│   ├── profile_nsys.sbatch
│   └── profile_ncu.sbatch
├── scripts/
│   ├── env.sh
│   ├── usage.sh
│   └── gpucheck/
│       ├── vecadd.cu
│       ├── gpucheck.sh
│       └── ncu_check.sh
└── results/
    ├── platform/              GPU check output
    ├── baseline/              benchmark runs and golden outputs
    ├── baseline_profile/      profiler summaries and SUMMARY.md
    └── experiments/           optimization runs
```

## Setup so far

1. Set up folders on Delta. Code and the venv are in `~/classes/598`, model weights in `/work/nvme/bikd/$USER/598/hf`, and big profiler files in `/work/hdd/bikd/$USER/598/runs`.
2. Wrote `scripts/env.sh` to load Python 3.13.5 and the CUDA 13.2 toolkit, set `HF_HOME`, and add Nsight Compute to the PATH. Run `source ~/classes/598/env.sh` at the start of every session.
3. Made a venv and installed torch 2.14.1 (cu130), transformers 5.19.0 and the rest of `requirements.txt`.
4. Logged in to Hugging Face with a read token and downloaded `allenai/Olmo-3-1025-7B` (revision `a81bae4`) on the login node.
5. Ran a quick GPU check on one A40 with a vector add kernel to make sure nvcc, compute-sanitizer, nsys and ncu all work. The vector add hit 578 GB/s (83% of the A40's 696 GB/s), which we use as the bandwidth ceiling (`results/platform/2026-10-08_a40/`).
6. Wrote the benchmark and correctness checks, tested them on a tiny model on my laptop, then ran the first baseline on the A40.
7. Added an 8192 token prompt so the sliding window layers actually get used.
8. Profiled the 512 token workload with torch.profiler, Nsight Systems and Nsight Compute (`slurm/profile_*.sbatch`). Summary is in `results/baseline_profile/baseline_20261008/SUMMARY.md`.

## Baseline results

One A40, bf16, stock transformers (SDPA, no torch.compile). Median of 5 runs after 3 warmups, CV under 0.5% everywhere.
Run: `results/baseline/20261008_193718_22764605/`

| config | TTFT (ms) | TPOT (ms) | tokens/s |
|---|---|---|---|
| prefill, batch 1, 128 tokens | 46.1 | - | 2777 |
| prefill, batch 1, 2048 tokens | 387.2 | - | 5290 |
| prefill, batch 1, 8192 tokens | 2066.7 | - | 3964 |
| decode, batch 1, 512 prompt + 128 new | 109.2 | 38.44 | 26.0 |
| decode, batch 8, 512 prompt + 128 new | 740.4 | 45.58 | 175.5 |
| decode, batch 1, 8192 prompt + 128 new | 2109.7 | 51.14 | 19.6 |

Prefill tokens/s is prompt tokens / TTFT. Decode tokens/s is batch * 127 decode steps / decode time.
Correctness: greedy outputs match `model.generate` and the golden file, perplexity 2.9 (1536 tokens) and 3.7 (8192 tokens).

## Optimization log

Speedup is against the baseline above, same config.

| commit | optimization | speedup |
|---|---|---|
| 8c03ee4 | baseline | 1.00x |
| 7b9baf8 | fused Triton kernels: q/k norm + RoPE, post-norm + residual add, SiLU*mul (`--opt fused`) | TTFT 128: 1.25x, 2048: 1.36x, 8192: 1.24x; TPOT b1: 1.32x, b8: 1.13x, b1 8192: 1.10x |
| | static ring-buffer KV cache + CUDA graph decode (`--opt static,graph`) | TPOT b1: 1.26x, b8: 1.30x, b1 8192: 1.44x; TTFT unchanged |
| | both (`--opt fused,static,graph`) | TTFT 128: 1.41x, 2048: 1.35x, 8192: 1.23x; TPOT b1: 1.38x, b8: 1.44x, b1 8192: 1.56x |

### Fused Triton kernels (`--opt fused`)

The baseline profile showed RMSNorm, QK-Norm and RoPE at about 10% of prefill GPU time each, with every op split into 6-8 small fp32 kernels that move 10-11x the bytes they need. In decode, 79% of the ~1979 kernels per token were these small ops. `olmo_bench/kernels.py` replaces them with three Triton kernels, swapped into the HF model classes when `--opt fused` is set:
- q_norm + k_norm + RoPE in one launch (28 kernels per layer become 1)
- post-attention / post-MLP RMSNorm fused with the residual add (18 kernels become 2), also used for the final norm
- SiLU(gate) * up

Math is done in fp32 and rounded to bf16 at the same points as the HF code.

Run: `TAG=x1 VARIANTS="baseline fused" sbatch slurm/experiment.sbatch` (job 22913289, gpub076, both variants on the same GPU). Results are in `results/experiments/x1_22913289/`.

| config | baseline | fused | speedup |
|---|---|---|---|
| prefill b1 128, TTFT (ms) | 46.30 | 37.16 | 1.25x |
| prefill b1 2048, TTFT (ms) | 390.15 | 287.55 | 1.36x |
| prefill b1 8192, TTFT (ms) | 2063.11 | 1668.87 | 1.24x |
| decode b1 512+128, TPOT (ms) | 38.41 | 29.20 | 1.32x |
| decode b8 512+128, TPOT (ms) | 45.40 | 40.29 | 1.13x |
| decode b1 8192+128, TPOT (ms) | 51.15 | 46.61 | 1.10x |

Decode tokens/s went from 26.0 to 34.2 (b1), 176.2 to 198.6 (b8) and 19.6 to 21.5 (b1 8192). The same-job baseline matches the original baseline within 1%.

Correctness (`olmo_bench/numerics.py`, compared with the stock model in the same run): perplexity changed by -0.001% (1536 tokens) and +0.015% (8192 tokens), and top-1 agreement over those texts was 99.28% and 99.58%, which passes our 99% / 0.5% gate. All four golden prompts still match. The kernel unit tests (`tests/test_kernels.py`) passed on the A40.
Each Triton launch costs about 30 us of host time, versus about 6-10 us for a PyTorch op (`python -m olmo_bench.kernels --bench`, job 22913240). Fusing the norm and RoPE chains is a big win, but the SiLU*mul kernel on its own is slower than the two PyTorch ops it replaces in decode.

### Static KV cache + CUDA graphs (`--opt static,graph`)

After fusing, decode was still slow for two reasons. HF's DynamicCache uses `torch.cat` to add each new token, which copies the whole KV cache every step (18.5% of decode GPU time at batch 8). And at batch 1 the GPU was idle about 7 ms per token waiting on the CPU to launch kernels.

`olmo_bench/fast_decode.py` allocates the cache once and writes one slot per token (sliding layers use a 4096 slot ring buffer), then records a decode step as a CUDA graph and replays it. We wrote our own cache because HF's StaticCache keeps the position as a Python int, which doesn't work with graph replay.

Same job as the fused run (22913289):

| config | baseline | static,graph | fused,static,graph |
|---|---|---|---|
| prefill b1 128, TTFT (ms) | 46.30 | 41.11 (1.13x) | 32.93 (1.41x) |
| prefill b1 2048, TTFT (ms) | 390.15 | 393.89 (0.99x) | 288.52 (1.35x) |
| prefill b1 8192, TTFT (ms) | 2063.11 | 2096.54 (0.98x) | 1678.83 (1.23x) |
| decode b1 512+128, TPOT (ms) | 38.41 | 30.54 (1.26x) | 27.76 (1.38x) |
| decode b8 512+128, TPOT (ms) | 45.40 | 34.83 (1.30x) | 31.45 (1.44x) |
| decode b1 8192+128, TPOT (ms) | 51.15 | 35.57 (1.44x) | 32.74 (1.56x) |

Prefill doesn't change since this only touches decode. The output is bit-exact with the baseline. With both optimizations batch 1 decode is 27.8 ms/token, close to the ~24 ms it takes just to read the weights.

All GPU jobs use the `bikd-delta-gpu` account on the `gpuA40x4` partition. Run them from the repo root on Delta, for example `sbatch slurm/baseline.sbatch`. `scripts/usage.sh` shows how many GPU hours have been used.
