# 598-project

Profiling non-GEMM ops (RMSNorm, QK-Norm, RoPE, residual adds, activations) in OLMo 3 7B inference on NCSA Delta.

## File structure

```
598-project/
├── README.md
├── requirements.txt
├── docs/
│   └── platform.md            GPU specs, software versions, vecadd results
├── olmo_bench/
│   ├── model.py               loads OLMo 3 7B
│   ├── workload.py            prompts, prefill, decode step
│   ├── benchmark.py           TTFT / TPOT / tokens per second
│   ├── correctness.py         output and perplexity checks
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
    └── baseline_profile/      profiler summaries and SUMMARY.md
```

## Setup so far

1. Set up folders on Delta. Code and the venv are in `~/classes/598`, model weights in `/work/nvme/bikd/$USER/598/hf`, and big profiler files in `/work/hdd/bikd/$USER/598/runs`.
2. Wrote `scripts/env.sh` to load Python 3.13.5 and the CUDA 13.2 toolkit, set `HF_HOME`, and add Nsight Compute to the PATH. Run `source ~/classes/598/env.sh` at the start of every session.
3. Made a venv and installed torch 2.14.1 (cu130), transformers 5.19.0 and the rest of `requirements.txt`.
4. Logged in to Hugging Face with a read token and downloaded `allenai/Olmo-3-1025-7B` (revision `a81bae4`) on the login node.
5. Ran a quick GPU check on one A40 with a vector add kernel to make sure nvcc, compute-sanitizer, nsys and ncu all work (see `docs/platform.md`).
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
| | baseline | 1.00x |

All GPU jobs use the `bikd-delta-gpu` account on the `gpuA40x4` partition. Run them from the repo root on Delta, for example `sbatch slurm/baseline.sbatch`. `scripts/usage.sh` shows how many GPU hours have been used.
