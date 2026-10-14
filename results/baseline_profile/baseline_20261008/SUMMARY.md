# Baseline profile: OLMo 3 7B on A40 (2026-10-08)

Stock HF transformers 5.19.0, bf16, SDPA, eager, DynamicCache. Prompt length 512. Decode numbers are per step (one token per sequence).
Sources: `torch/` (torch.profiler, 16 decode steps), `nsys/`, `ncu/` (decoder layer 3 only, batch 1), and `results/baseline/20261008_193718_22764605/bench.json`.
"Measured" = read directly from a tool. "Est." = computed from sizes and the 578 GB/s ceiling.

## 1. Where GPU time goes (torch profiler)

| | prefill b1 | prefill b8 | decode b1 | decode b8 |
|---|---|---|---|---|
| GPU time per pass/step | 105.9 ms | 730.4 ms | 31.5 ms | 42.3 ms |
| GEMM | 63.3% | 59.7% | 81.7% | 60.3% |
| non-GEMM | **33.2%** | **36.7%** | 12.8% | 10.6% |
| attention (SDPA + KV cache) | 3.4% | 3.6% | 5.5% | **29.0%** |

| category | prefill b1 | prefill b8 | decode b1 | decode b8 |
|---|---|---|---|---|
| rmsnorm (post-attn + post-MLP) | 9.8% | 10.7% | 4.1% | 3.6% |
| qk_norm | 9.8% | 10.6% | 4.0% | 3.5% |
| rope | 9.4% | 10.2% | 3.7% | 2.7% |
| mlp_mul (act x up) | 1.7% | 2.1% | 0.2% | 0.2% |
| residual adds | 1.2% | 1.6% | 0.3% | 0.3% |
| act (SiLU) | 1.1% | 1.4% | 0.3% | 0.2% |
| final_norm + rope_table | 0.2% | 0.2% | 0.1% | 0.1% |
| kv_cache (torch.cat) | 1.7% | 1.7% | 2.8% | **18.5%** |
| attn_core (SDPA) | 1.7% | 1.8% | 2.7% | 10.5% |
| gate_up / qkv / down / o_proj | 29.7 / 14.7 / 12.7 / 4.9% | 25.3 / 15.6 / 13.5 / 5.2% | 34.4 / 19.5 / 16.5 / 6.6% | 24.5 / 14.9 / 12.6 / 5.0% |
| lm_head | 1.4% | 0.2% | 4.6% | 3.4% |

Full tables: `torch/categories.csv`. Flamegraphs: `torch/*_kernels.folded`.

## 2. Kernel counts

| | prefill | decode b1 | decode b8 |
|---|---|---|---|
| kernels per pass/step | 1907 | 1979 | 1883 |
| non-GEMM kernels | 1558 (81.7%) | 1558 (78.7%) | 1558 (82.7%) |
| per layer: rmsnorm / qk_norm / rope / residual / act / mul | 16 / 16 / 12 / 2 / 1 / 1 | same | same |

Each RMSNorm is 8 kernels (fp32 cast, pow, mean, +eps, rsqrt, mul, mul weight, bf16 cast). RoPE is 6 kernels per tensor.
Decode b1 has 96 more kernels than b8 because of split-KV flash plus a combine kernel, and gate/up GEMMs running as 2 kernels each.

## 3. Decode: GPU busy vs wall

| decode step | b1 | b8 | source |
|---|---|---|---|
| GPU busy (union of kernel time) | 31.5 ms | 42.3 ms | measured, nsys = torch |
| TPOT, benchmark (no profiler, no labels) | 38.4 ms | 45.6 ms | measured |
| GPU idle = TPOT - busy | **~6.9 ms (18%)** | ~3.3 ms (7%) | est. |
| host time per kernel = TPOT / kernels | ~19 us | ~24 us | est. |
| step under nsys, unlabeled | 57.1 ms (55% busy) | 56.2 ms (75%) | measured |
| step under nsys, labeled | 66.4 ms | 65.4 ms | measured |
| step under torch.profiler + labels | ~85 ms | | measured |

At b1, decode is host-bound. The tools add host time: nsys adds about 10 us per launch, and the labels add +49% at b1 and +24% at b8 (`torch/label_overhead.csv`), so only the TPOT-based estimate reflects a plain run. GPU busy time is the same with labels on or off.
Prefill is GPU-bound: busy 105.9 vs TTFT 109.2 ms (b1), 730.4 vs 740.4 ms (b8).

## 4. Memory traffic per category (ncu, layer 3, b1)

ncu flushes caches and locks clocks, so tiny kernels run about 35% slower than in torch; large kernels are within about 5%.
Values over 100% of 578 are possible because 578 is the useful-bytes ceiling while ncu counts raw DRAM traffic (about 10% overhead, see `docs/platform.md`).

Prefill (512 tokens):

| category | kernels | time (us) | DRAM MB | GB/s | % of 578 | ideal MB (est.) | L2 hit | occupancy |
|---|---|---|---|---|---|---|---|---|
| gate_up/qkv/down/o_proj | 7 | 2071.5 | 677.0 | 327 | 57% | | 90% | 17% |
| qk_norm | 16 | 342.0 | 188.9 | 552 | 96% | 16.8 | 47% | 71% |
| rmsnorm | 16 | 341.2 | 188.3 | 552 | 96% | 16.8 | 47% | 71% |
| rope | 12 | 314.9 | 167.3 | 531 | 92% | 16.8 | 52% | 83% |
| mlp_mul | 1 | 59.7 | 37.7 | 632 | 109% | 33.8 | 34% | 84% |
| residual | 2 | 44.7 | 27.0 | 605 | 105% | 25.2 | 34% | 71% |
| act | 1 | 40.1 | 24.5 | 611 | 106% | 22.5 | 51% | 81% |
| attn_core | 1 | 56.3 | 26.6 | 473 | 82% | | 62% | 15% |
| kv_cache | 2 | 56.2 | 19.2 | 341 | 59% | | 52% | 63% |

In prefill every non-GEMM kernel already runs at bandwidth. The norms and RoPE move 10-11x the ideal single-pass bf16 bytes, while residual, act and mul are within about 10% of ideal. GEMMs reach about 92-110 TFLOPS (est.), so they are compute-bound.

Decode (b1, context ~516):

| category | kernels | time (us) | DRAM MB | GB/s | % of 578 | L2 hit | occupancy |
|---|---|---|---|---|---|---|---|
| gate_up/qkv/down/o_proj | 7 | 789.5 | 497.4 | 630 | 109% | 2% | 8% |
| qk_norm | 16 | 55.4 | 0.59 | 11 | 2% | ~100% | 12% |
| rmsnorm | 16 | 55.2 | 0.60 | 11 | 2% | ~100% | 12% |
| rope | 12 | 53.8 | 0.55 | 10 | 2% | ~100% | 21% |
| residual/act/mlp_mul | 4 | 10.4 | 0.15 | 14 | 2-4% | 70-110% | 8% |
| attn_core (split-KV + combine) | 2 | 29.1 | 11.1 | 383 | 66% | 23% | 8% |
| kv_cache (cat) | 2 | 28.3 | 16.1 | 568 | 98% | 52% | 71% |

GEMVs stream the weights at the DRAM limit (13.8 GB per step, so at least 23.8 ms est.; measured GEMM time is 25.7 ms). Non-GEMM kernels take 2-6 us each and launch 1-16 blocks on 84 SMs, so they are latency- and launch-bound rather than bandwidth-bound. L2 hit rates over 100% are an ncu artifact on tiny kernels.

## 5. KV cache at b8 and at long context

All of `kv_cache` is `CatArrayBatchedCopy_vectorized`, the `torch.cat` in `DynamicLayer.update`, which copies the whole K/V cache every step.
- b1, context 512: 0.88 ms/step (est. 0.93).
- b8: **7.8 ms/step, 18.5%** (est. 7.4).
- The cat copy plus SDPA explain the b8 - b1 GPU difference (+10.8 ms: kv_cache +6.9, attn_core +3.6). GEMM time is unchanged.

Context 8192, b1 (not profiled): TPOT 51.1 vs 38.4 ms (+12.7 ms). The cat traffic is about 5.4 GB/step, which is 9.3 ms (est.), with the sliding layers holding 4095 tokens and the full layers 8192. SDPA reads add about 4.6 ms (est.), for a total of about +12.5 ms (est.) against +12.7 ms measured. A profile at context 8192 is needed to confirm.

## 6. Cross-tool checks

| quantity | torch | nsys | ncu | note |
|---|---|---|---|---|
| decode b1 GPU busy | 31.49 ms | 31.49 ms | 32.7 ms (layer 3 x 32) | ncu slightly slow |
| decode b1 kernels | 1979 | 1979 | 59/layer | 59x32 + 91 = 1979 |
| prefill b1 GPU | 105.9 ms | 105.9 ms | | |
| prefill rmsnorm/layer | 326 us | | 341 us | +5% |
| decode rmsnorm/layer | 41 us | | 55 us | +35% (tiny kernels) |

## 7. Candidate bottlenecks, ranked

1. **Prefill RMSNorm + QK-Norm + RoPE traffic**: 29-31% of prefill GPU time (30.8 of 105.9 ms). These kernels run at the bandwidth ceiling but move 10-11x the ideal bytes, so ideal traffic would save about 27 ms (est.).
2. **Decode host/launch overhead at small batch**: 1979 kernels per step, 79% of them non-GEMM (44 of the 59 per layer come from the norm and RoPE chains). The GPU is idle about 6.9 of 38.4 ms at b1 (est.).
3. **KV cache torch.cat copy**: 18.5% of decode GPU time at b8 (measured). It is the likely main cause of the +12.7 ms at context 8192 (est.) and scales with batch x context.
4. **Decode non-GEMM GPU time**: 4.0 ms at b1 (12.8%) and 4.5 ms at b8 (10.6%). These kernels are latency-bound, at 2% of bandwidth and 8-12% occupancy.
5. **SDPA at b8**: 10.5% of decode GPU time at 66-82% of bandwidth, and it grows with context.

Out of scope: the decode GEMVs are at the weight-streaming limit, and the prefill GEMMs reach about 60-70% of tensor peak.
