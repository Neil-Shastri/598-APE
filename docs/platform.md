# Platform

All experiments run on NCSA Delta, one NVIDIA A40 (`gpuA40x4` partition, account `bikd-delta-gpu`).

## Hardware

| | |
|---|---|
| GPU | NVIDIA A40 (GA102, sm_86), 48 GB GDDR6 with ECC (46,068 MiB usable) |
| Peak DRAM bandwidth | 696 GB/s (spec) |
| Max clocks | SM 1740 MHz, memory 7251 MHz |
| Power limit | 300 W |
| Host link | PCIe gen4 x16 |
| Host CPU | AMD EPYC Milan, 64 cores, 256 GB per node (4 GPUs per node, shared) |

## Software

| | |
|---|---|
| Driver | 595.71.05 (CUDA 13.2) |
| CUDA toolkit | 13.2 (`cudatoolkit/26.5_13.2`, nvcc V13.2.78) |
| Nsight Compute | 2026.1.1 |
| Nsight Systems | 2026.1.3 |
| Python | 3.13.5 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.19.0 |
| Model | `allenai/Olmo-3-1025-7B`, revision `a81bae42db3975be1671e27b9c9a56da1a9f980f`, bf16 |

## Storage

| Path | Use |
|---|---|
| `~/classes/598` | code, venv, `env.sh` |
| `/work/nvme/bikd/$USER/598/hf` | `HF_HOME` (model weights) |
| `/work/hdd/bikd/$USER/598/runs` | raw profiler output |

## Toolchain check (2026-10-08)

`scripts/gpucheck/` runs a vector add (3 x 64 MiB fp32 arrays) under compute-sanitizer, Nsight Systems and Nsight Compute.
Raw output is in `results/platform/2026-10-08_a40/`.

| Metric | Value |
|---|---|
| compute-sanitizer | 0 errors |
| Kernel time (nsys / ncu) | 347.2 us / 348.5 us |
| Useful bandwidth (201.3 MB / time) | 578 GB/s (83% of peak) |
| DRAM read / write (ncu) | 146.8 MB / 76.1 MB (expected 134.2 / 67.1) |
| DRAM throughput (ncu) | 639.6 GB/s (92% of peak) |
| Compute (SM) throughput | 8.3% |
| L2 hit rate | 33% |

We use 578 GB/s as the practical bandwidth ceiling for memory-bound kernels on this GPU.

Notes:
- First `cudaMalloc` takes ~127 ms (context creation) and the first kernel launch ~189 us (lazy module loading), so all benchmarks discard warmup iterations.
- ncu reports ~10% more DRAM traffic than the kernel needs. Not yet explained (possibly ECC); to verify.
- Nsight Compute is not on the `cudatoolkit` module's path; `env.sh` adds `/opt/nvidia/nsight-compute/2026.1.1`.
