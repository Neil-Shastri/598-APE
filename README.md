# 598-APE

Profiling non-GEMM ops (RMSNorm, QK-Norm, RoPE, residual adds, activations) in OLMo 3 7B inference on NCSA Delta.

## File structure

```
598-APE/
├── README.md
├── requirements.txt
├── olmo_bench/
│   ├── model.py               loads OLMo 3 7B
│   ├── workload.py            prompts, prefill, decode step
│   ├── benchmark.py           TTFT / TPOT / tokens per second
│   ├── correctness.py         output and perplexity checks
│   ├── kernels.py             fused Triton kernels (--opt fused)
│   ├── fast_decode.py         static KV cache + CUDA graph decode (--opt static,graph,mask,slots,cudnn)
│   ├── merge.py               merged q/k/v and gate/up weights (--opt qkv,gateup)
│   ├── band_attention.py      banded sliding-window attention (--opt band)
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
│   ├── kernel_groups.py
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
│   ├── knobs.py               knob sweep (cuBLAS, SDPA backends, Triton warps)
│   ├── usage.sh
│   └── gpucheck/
│       ├── vecadd.cu
│       ├── gpucheck.sh
│       └── ncu_check.sh
└── results/
    ├── platform/              GPU check output
    ├── baseline/              benchmark runs and golden outputs
    ├── baseline_profile/      profiler summaries and SUMMARY.md
    ├── optimized_profile/     same profile with --opt fused,static,graph
    ├── experiments/           optimization runs
    ├── knobs/                 knob sweep results
    └── tunable/               TunableOp csv for the A40
```

## Hardware and software

- One NVIDIA GPU with at least 24 GB of memory (we used an A40, 48 GB). The baseline peaks at about 23 GB on the 8192 token prompt.
- CUDA 13 driver and toolkit, Python 3.13, torch 2.14.1, transformers 5.19.0, Triton 3.8.0 (`requirements.txt`).
- About 15 GB of disk for the model weights.
- The tests and the benchmark logic also run on a CPU-only laptop with a tiny model; the GPU tests skip there.

## Running

```
python -m pytest -q profiling tests                         # unit tests
python -m olmo_bench.benchmark                               # baseline
python -m olmo_bench.benchmark --opt fused,static,graph,mask,slots,qkv,gateup,band,tunable,cudnn
VARIANTS="baseline fused,static,graph" sbatch slurm/experiment.sbatch   # same-node comparison on Delta
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
| c984939 | static ring-buffer KV cache + CUDA graph decode (`--opt static,graph`) | TPOT b1: 1.26x, b8: 1.30x, b1 8192: 1.44x; TTFT unchanged |
| c984939 | both (`--opt fused,static,graph`) | TTFT 128: 1.41x, 2048: 1.35x, 8192: 1.23x; TPOT b1: 1.38x, b8: 1.44x, b1 8192: 1.56x |
| 54b0a51 | build the decode mask once per step (`--opt mask`) | TPOT b1 -0.6%, b8 -0.5% vs previous row |
| 95c762c | compute prefill cache slots once (`--opt slots`) | TTFT 128 -0.5% vs previous row |
| 367ed7b | merge q/k/v into one GEMM (`--opt qkv`) | TTFT 2048 -2.8%; TPOT b1 -0.5%, b8 -1.4% vs previous row |
| a27dee8 | merge gate/up into one GEMM (`--opt gateup`) | TPOT b1 -2.3%, TTFT 2048 -1.7%; TTFT 128 +2.8%, TPOT b8 +0.7% vs previous row |
| ec614ab | banded attention for sliding layers in long prefill (`--opt band`) | TTFT 8192 -21.7% vs previous row |
| d047493 | TunableOp GEMM selection (`--opt tunable`) | TTFT 128 -5.2%; TPOT b1 -1.9%, b8 -1.2% vs previous row |
| e5f533d | cuDNN attention for the decode step (`--opt cudnn`) | TPOT b1 -1.9%, b8 -2.1%, b1 8192 -5.4% vs previous row |

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

### Profile after both optimizations

We profiled `--opt fused,static,graph` the same way as the baseline (512 token prompt, 16 decode steps): `TAG=opt_20261020 OPT=fused,static,graph sbatch slurm/profile_torch.sbatch` (job 23073963). The per-op labels don't work inside a CUDA graph, so `profiling/kernel_groups.py` splits GPU time by kernel name instead (`results/optimized_profile/opt_20261020/kernel_groups.csv`).

| | non-GEMM share of GPU time | kernels per step |
|---|---|---|
| prefill b1 | 32.0% -> 5.7% | 1883 -> 534 |
| prefill b8 | 35.2% -> 6.8% | 1883 -> 534 |
| decode b1 | 12.1% -> 2.3% | 1915 -> 633 |
| decode b8 | 10.1% -> 2.0% | 1883 -> 633 |

Most of the remaining time is now matrix multiplies (81-93%), and at decode batch 8 attention is next (16.5%) since the KV cache copy is gone.

### Building the decode mask once (`--opt mask`)

With the static cache the decode step passes a True/False mask over the whole buffer, and SDPA turns it into a 0/-inf mask again in every one of the 32 layers (3 small kernels each, so 96 per token). Now `fast_decode.py` builds the 0/-inf mask once per step and every layer reuses it. Output is bit-exact.

Measured in job 23916330 (all the new optimizations were stacked one at a time in the same job, `results/experiments/ape2_23916330/`): TPOT b1 27.75 -> 27.57 ms, b8 31.45 -> 31.30 ms, prefill unchanged.

### Computing the prefill slots once (`--opt slots`)

During prefill every layer worked out which cache slot each prompt token goes into, but the answer only depends on the buffer size (there are two, sliding and full). Now it's computed once per size. Bit-exact. TTFT at 128 tokens went 32.76 -> 32.61 ms; the other configs didn't change, as expected.

### Merging the q/k/v projections (`--opt qkv`)

Each layer did three separate 4096x4096 GEMMs for q, k and v. At batch 1 each one only launches 64 blocks for 84 SMs, so `merge.py` stacks the three weights into one 12288x4096 weight (the old weights become views into it, no extra memory) and the fused attention does one GEMM and slices the result. The Triton q/k-norm+RoPE kernel now takes a row stride so the slices aren't copied.

TTFT 2048 291.5 -> 283.3 ms, TPOT b1 27.56 -> 27.42 ms, b8 31.27 -> 30.82 ms. Numerics pass (same top-1 and perplexity as before). Note: in this job the stack also had a RoPE-table option turned on that we later dropped because it made no measurable difference, so the row before this one is `...,rope`.

### Merging the gate/up projections (`--opt gateup`)

Same idea for the MLP: gate and up (11008x4096 each) become one 22016x4096 GEMM, and the fused SiLU*mul kernel reads the two halves with a row stride. TPOT b1 27.42 -> 26.79 ms, b1 at 8192 context 32.53 -> 31.92 ms, TTFT 2048 283.3 -> 278.4 ms. It isn't a win everywhere though: TTFT 128 got slower (32.55 -> 33.45 ms) and TPOT b8 slightly slower (30.82 -> 31.02 ms), since cuBLAS picks a different kernel for the merged shape. TunableOp later fixes most of that.

### Banded sliding-window attention in long prefill (`--opt band`)

24 of the 32 layers only look back 4096 tokens, but in an 8192 token prefill SDPA still scores all 8192x8192 pairs and masks most of them out. `band_attention.py` does the first 4096 queries with plain causal attention and the rest in chunks of 1024 against only their own window of keys, so most of the thrown-away work is skipped. Only prompts longer than the window are affected.

TTFT 8192 1687 -> 1320 ms (-21.7%), everything else unchanged. Long-context perplexity changed by +0.022%, top-1 agreement 99.56%.

### TunableOp (`--opt tunable`)

PyTorch's TunableOp times the cuBLAS kernel against a handful of cuBLASLt algorithms for every GEMM shape the first time that shape shows up, and saves the winner to `results/tunable/tunableop_a40.csv`. Later runs just read the csv, so no tuning happens in the timed loop. This fixed most of what the gate/up merge made worse: TTFT 128 33.55 -> 31.80 ms, and decode got a bit faster too (TPOT b1 26.78 -> 26.26 ms, b8 31.01 -> 30.64 ms). Numerics still pass (top-1 99.09% / 99.56%).

### Knobs we tried and didn't keep

`scripts/knobs.py` measures each knob on its own with the real shapes (job 23927854, `results/knobs/`):

| knob | result | kept? |
|---|---|---|
| `preferred_blas_library("cublaslt")` | 0.95-1.08x depending on shape, TunableOp already picks per shape | no |
| `allow_bf16_reduced_precision_reduction = False` | 0.94-1.23x, mixed, and changes results | no |
| cuBLASLt workspace 1 vs 8 MiB | 1.00x | no |
| Triton `num_warps` (2/4/8/16) | best setting saves < 1 us per kernel, about 0.03 ms per token | no |
| TF32 | no effect, our math is bf16 | no |
| SDPA backend for the masked decode step | cuDNN is 1.1-1.9x faster than the default memory-efficient kernel | yes, `--opt cudnn` below |

### Ablation

`--opt static` alone (no CUDA graph, no fused kernels): TPOT b1 37.78 -> 37.63 ms, b8 45.37 -> 38.04 ms, b1 8192 51.13 -> 38.61 ms. So the static cache is what fixes batch 8 and long context (no more copying the cache), and the CUDA graph is what fixes batch 1.

All the new optimizations above were measured stacked one at a time in job 23916330 (`slurm/experiment.sbatch`, log in `results/experiments/`). The GPU unit tests ran first (126 passed), and `compute-sanitizer --tool memcheck` on the kernel tests reported 0 errors (job 23916331).

### cuDNN attention in the decode step (`--opt cudnn`)

The knobs sweep showed that for our masked decode attention, PyTorch's default (memory-efficient) kernel is slow at batch 1: it only launches 32 blocks. The cuDNN backend does the same masked attention 1.6x faster at 640 keys and 1.9x faster at 8320. Since the backend gets baked into the CUDA graph when it's captured, `fast_decode.py` just captures the decode step inside `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`.

Job 23928600, same node, with and without it: TPOT b1 26.27 -> 25.77 ms, b8 30.64 -> 30.00 ms, b1 8192 31.40 -> 29.69 ms. Prefill doesn't change. Numerics pass and the golden prompts match; the 128-token decode run first differs at token 92 (a near tie).

### Where it ended up

Final stack: `--opt fused,static,graph,mask,slots,qkv,gateup,band,tunable,cudnn` (job 23928600) vs the baseline measured in job 23916330 (different node, so allow 1-3%):

| config | baseline | final | speedup |
|---|---|---|---|
| prefill b1 128, TTFT (ms) | 45.78 | 31.78 | 1.44x |
| prefill b1 2048, TTFT (ms) | 392.32 | 275.60 | 1.42x |
| prefill b1 8192, TTFT (ms) | 2090.82 | 1309.19 | 1.60x |
| decode b1 512+128, TPOT (ms) | 37.78 | 25.77 | 1.47x |
| decode b8 512+128, TPOT (ms) | 45.37 | 30.00 | 1.51x |
| decode b1 8192+128, TPOT (ms) | 51.13 | 29.69 | 1.72x |

All GPU jobs use the `bikd-delta-gpu` account on the `gpuA40x4` partition. Run them from the repo root on Delta, for example `sbatch slurm/baseline.sbatch`. `scripts/usage.sh` shows how many GPU hours have been used.
