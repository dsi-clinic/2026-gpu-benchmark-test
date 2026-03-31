# B300 vs B200 Benchmark Analysis Report

**Date:** 2026-03-31
**Source Data:** DSAI Arch Cluster Benchmarks (presentation dated 2026-03-16)

**Relevant Repositories:**
- **Synthetic benchmark:** [`rcc-uchicago/nodes-testing-tools`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) -- `gpu_benchmark.py` (slides 5-7)
- **Pangu ML training:** [`ndtrung81/3dViT-training`](https://github.com/ndtrung81/3dViT-training) -- 3D Vision Transformer / Pangu-Weather training scripts (slide 4)
- **CV & LLM workloads:** [`MohsenZand/GPUBenchmarking`](https://github.com/MohsenZand/GPUBenchmarking) -- ResNet-50, ViT-Large, LLM inference benchmarks (slides 9-10, 12-13)

---

## Executive Summary

The DSAI Arch Cluster benchmarks show the B300 GPU underperforming the B200 in several workloads. Since the B300 is a higher-clocked variant of the B200 with ~7% more compute, identical memory bandwidth, and identical NVLink 5.0 interconnect, the B300 should match or exceed the B200 everywhere.

However, **close examination of the data reveals a nuanced picture**. The B300 does not uniformly underperform. For compute-bound operations, there is a pattern where **small problem sizes penalize the B300 while large problem sizes show the B300 matching or exceeding the B200** (clearly visible in GEMM and D2D benchmarks). For communication benchmarks, results are inconsistent across runs -- two separate runs of the same benchmark code produced contradictory All-Reduce and P2P results, suggesting system-level variability. Workloads that use well-optimized software stacks (GROMACS, ViT-Large, LLM inference in BF16) generally show normal B300 performance. This pattern points to a combination of **CUDA/driver kernel maturity issues on B300** and **benchmark code weaknesses that disproportionately expose those issues**.

**Important note on data sources:** The presentation contains results from multiple runs of `gpu_benchmark.py`. Slide 5 shows one set of multi-GPU results, while slides 6-7 show a separate run with different communication numbers. Where these conflict, we note both.

## Observed Performance: Full Data from Presentation

### Where B300 Underperforms B200

| Workload | B200 | B300 | Delta | Source |
|---|---|---|---|---|
| Pangu ML (FP16, 4 GPU, batch=8) | 4.22 samples/s | 2.18 samples/s | **-48%** | Slide 4 |
| Pangu ML (FP16, 1 GPU) | 1.67 samples/s | 1.13 samples/s | **-32%** | Slide 4 |
| Attention B32 H12 S512 (FP16) | 87.32 TFLOPS | 62.53 TFLOPS | **-28%** | Slide 5 |
| DDP 1024x1024 (4 GPU) | 15.99 TFLOPS | 1.24 TFLOPS | **-92%** | Slide 5 |
| DDP 4096x4096 (4 GPU) | 1033.57 TFLOPS | 82.09 TFLOPS | **-92%** | Slide 5 |
| All-Reduce 100 MB (4 GPU) | 472.06 GB/s | 251.73 GB/s | **-47%** | Slide 5 (Run A) |
| D2D Copy 512 MB | 2937 GB/s | 1547 GB/s | **-47%** | Slides 6-7 (Run B) |
| P2P Transfer 100 MB | 452.20 GB/s | 357.76 GB/s | **-21%** | Slide 5 (Run A) |
| GEMM BF16 small 4096 | 1504 TFLOPS | 782 TFLOPS | **-48%** | Slides 6-7 (Run B) |
| ResNet-50 (1 GPU) | 15,081 FPS | 4,444 FPS | **-70%** | Slide 9 |
| ViT-Large (4 GPU) | 13,365 FPS | 11,394 FPS | **-15%** | Slide 9 |
| LAMMPS (all particle counts) | -- | -- | **-8% to -19%** | Slide 8 |

### Where B300 Matches or Exceeds B200

| Workload | B200 | B300 | Delta | Source |
|---|---|---|---|---|
| Conv2D 128x128x112x112 (FP16) | 349.28 TFLOPS | 345.95 TFLOPS | **-1%** | Slide 5 |
| All-Reduce 500 MB (4 GPU) | 510.73 GB/s | 491.80 GB/s | **-4%** | Slide 5 (Run A); Run B shows -21% |
| P2P Transfer 500 MB | 600.05 GB/s | 617.57 GB/s | **+3%** | Slide 5 (Run A); Run B shows -22% |
| All-Reduce 100 MB (4 GPU) | 519.60 GB/s | 523.71 GB/s | **+1%** | Slides 6-7 (Run B); Run A shows -47% |
| D2D Copy 1024 MB | 3002 GB/s | 3031 GB/s | **+1%** | Slides 6-7 (Run B) |
| GEMM BF16 large 8192 | 1405 TFLOPS | 1632 TFLOPS | **+16%** | Slides 6-7 (Run B) |
| GEMM BF16 xlarge 16384 | 1511 TFLOPS | 1576 TFLOPS | **+4%** | Slides 6-7 (Run B) |
| ResNet-50 (4 GPU) | 43,370 FPS | 43,329 FPS | **0%** | Slide 9 |
| ResNet-50 (8 GPU) | 59,251 FPS | 112,769 FPS | **+90%** | Slide 9 |
| ViT-Large (1 GPU) | 5,658 FPS | 5,980 FPS | **+6%** | Slide 9 |
| ViT-Large (8 GPU) | 43,413 FPS | 45,416 FPS | **+5%** | Slide 9 |
| LLM 8B inference BF16 (1 GPU) | 7,632 TPS | 8,158 TPS | **+7%** | Slide 10 |
| LLM 70B inference BF16 (4 GPU) | 2,874 TPS | 2,860 TPS | **0%** | Slide 10 |
| GROMACS STMV 1M atoms (4 GPU) | N/A (no B200 data) | ~110 ns/day | N/A | Slide 8 |

**Note on conflicting runs:** The two runs of `gpu_benchmark.py` produced wildly inconsistent communication results:

| Test | Run A (slide 5) B300 vs B200 | Run B (slides 6-7) B300 vs B200 |
|---|---|---|
| All-Reduce 10 MB | not tested | -51% (131.61 vs 270.05) |
| All-Reduce 100 MB | **-47%** (251.73 vs 472.06) | **+1%** (523.71 vs 519.60) |
| All-Reduce 500 MB | -4% (491.80 vs 510.73) | -21% (457.19 vs 577.88) |
| P2P 10 MB | not tested | -57% (88.87 vs 206.74) |
| P2P 100 MB | -21% (357.76 vs 452.20) | +1% (454.38 vs 451.29) |
| P2P 500 MB | +3% (617.57 vs 600.05) | -22% (479.43 vs 617.69) |

The same benchmark code produces opposite outcomes across runs. This level of variability strongly suggests **system-level issues** (NCCL initialization, topology detection, other cluster traffic, thermal state) rather than a deterministic hardware or code problem.

**Note on LLM inference at 8 GPU:** While BF16 inference at 1-4 GPUs shows B300 matching or exceeding B200, the 8 GPU results show B300 trailing: 8B model B300=8,276 vs B200=8,765 (-6%); 70B model B300=3,029 vs B200=3,372 (-10%). The presentation notes an "Over-Sharding Penalty" where communication overhead chokes smaller models at high GPU counts (slide 10).

**Note on GROMACS:** Only B300 results are shown in the presentation (compared against A40, A100, H200). No B200 GROMACS data is available for direct comparison.

### Key Pattern: Size-Dependent Regressions

The data reveals a consistent pattern -- B300 underperforms on small problem sizes but matches or exceeds B200 on large ones:

| Operation | Small Size (B300 vs B200) | Large Size (B300 vs B200) | Source |
|---|---|---|---|
| GEMM BF16 | 4096: **-48%** | 8192: **+16%** | Slides 6-7 (Run B, same run) |
| D2D Copy | 512 MB: **-47%** | 1024 MB: **+1%** | Slides 6-7 (Run B, same run) |

**Note: Communication tests removed from this table.** Run A (slide 5) showed a size-dependent pattern for All-Reduce and P2P, but Run B (slides 6-7) contradicts it -- Run B shows the opposite pattern at some message sizes (see the conflicting runs table above). The communication behavior appears driven by system-level variability rather than a deterministic size-dependent effect.

**Note on ResNet-50:** The 1 GPU → 8 GPU progression (B300 goes from -70% to +90% vs B200) is a GPU-scaling comparison rather than a problem-size comparison. It is listed in the underperforms/exceeds tables above rather than here, since it involves a different axis of variation. The single-GPU anomaly is discussed separately in Issue 11 and the investigation steps.

For GEMM and D2D (both from the same Run B), the size-dependent pattern is consistent within a single run. This is consistent with **higher kernel launch overhead or suboptimal kernel selection on B300 for small workloads**, likely due to CUDA toolkit/driver immaturity for the B300's compute capability. Small problems are dominated by launch overhead and kernel selection; large problems amortize this overhead and become compute/bandwidth-bound, where B300's higher clocks provide a genuine advantage.

---

## Issues in Benchmark Code

The issues below fall into two categories: **code quality bugs** that affect reported numbers for both GPUs, and **missing best practices** that may disproportionately expose B300 kernel immaturity. We note the distinction explicitly.

### Issue 1: Manual Attention Implementation (No Flash Attention / SDPA)

**Location:** `benchmark_attention()` method, lines 172-222 of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** The benchmark manually computes attention as three separate operations:
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, V)
```

This bypasses `torch.nn.functional.scaled_dot_product_attention()` (SDPA), which auto-selects FlashAttention v2 or memory-efficient attention backends.

**Does this explain the B300 vs B200 delta?** Partially. The manual path materializes the full S x S attention matrix, making the operation memory-bandwidth-bound rather than compute-bound. Both GPUs have the same HBM bandwidth (8 TB/s spec), but B300's higher compute means it becomes relatively more starved for memory bandwidth. However, both GPUs have the same 5th-gen Tensor Cores and Blackwell architecture, so the claim that "B300's Tensor Cores are specifically optimized for the fused path" compared to B200 is **not well-supported** -- they share the same Tensor Core design. The -28% regression more likely reflects B300 kernel selection differences (driver maturity) rather than an architectural sensitivity.

**Regardless of the B300 delta, this is a genuine code quality issue.** SDPA should be used for any attention benchmark on modern GPUs. The benchmark as written does not exercise the hardware's fused attention capabilities on either GPU.

**Likely impact:** Contributes to the -28% attention regression; also means both GPUs' attention results understate their actual capability.

**Recommended fix:** Replace the three manual operations with `F.scaled_dot_product_attention(Q, K, V)`.

**Sources:**
- [PyTorch SDPA API reference](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) -- documents backend auto-selection (FlashAttention, memory-efficient, cuDNN attention)
- [Implementing High-Performance Transformers with SDPA (PyTorch tutorial)](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) -- benchmarks comparing fused vs manual attention backends
- [Accelerated PyTorch 2 Transformers (PyTorch blog)](https://pytorch.org/blog/accelerated-pytorch-2/) -- performance comparisons of SDPA vs manual implementation
- [PyTorch 2.2: FlashAttention-v2 integration (PyTorch blog)](https://pytorch.org/blog/pytorch2-2/) -- reports ~2x speedup and 50-73% theoretical max FLOPS on A100

---

### Issue 2: No `torch.compile()` or JIT Compilation

**Location:** Entire file [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools` -- no calls to `torch.compile()` anywhere (985 lines total).

**Problem:** None of the benchmarks use `torch.compile()`. `torch.compile(mode="max-autotune")` enables Triton-generated kernels optimized for the specific GPU architecture, operator fusion, and automatic CUDA graph capture.

**Does this explain the B300 vs B200 delta?** Possibly. Without `torch.compile()`, both GPUs rely on eager-mode CUDA kernels from cuBLAS/cuDNN. If these libraries have more mature B200-optimized kernels than B300-optimized ones, B300 would disproportionately benefit from `torch.compile()` generating architecture-specific Triton kernels. This is plausible but unverified.

**Regardless of the B300 delta, this is a genuine code quality issue.** Modern GPU benchmarks should offer a compiled mode to measure peak achievable performance.

**Recommended fix:** Add a `--compile` CLI flag; wrap benchmark functions with `torch.compile(mode="max-autotune")`.

**Sources:**
- [torch.compile API reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) -- documents `mode="max-autotune"` which "leverages Triton-based matrix multiplications and convolutions on GPU" and enables CUDA graphs
- [Introduction to torch.compile (PyTorch tutorial)](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) -- end-to-end examples showing performance gains
- [CUDA Blackwell Tuning Guide (NVIDIA)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- Blackwell-specific optimization guidance
- [Accelerating PyTorch with CUDA Graphs (PyTorch blog)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) -- explains CUDA graph capture benefits for reducing kernel launch overhead

---

### Issue 3: No cuDNN Benchmark Mode

**Location:** `__init__()` method and `benchmark_conv2d()` method (lines 124-170) of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** `torch.backends.cudnn.benchmark` is never set to `True` anywhere in the file. Without this, PyTorch uses a default convolution algorithm rather than auto-tuning to find the optimal algorithm for the specific GPU architecture and tensor shapes.

**Does this explain the B300 vs B200 delta?** **No, not for the simulated benchmark.** The Conv2D result from the simulated benchmark (slide 5) shows B200=349.28 TFLOPS vs B300=345.95 TFLOPS -- essentially identical (-1%). So the lack of cuDNN benchmark mode does not cause a B300-specific regression in this benchmark.

However, the ResNet-50 single-GPU result (slide 9) does show a severe B300 regression (B200=15,081 vs B300=4,444, -70%). This is from Mohsen's separate "Representative AI Workloads" benchmark, not from `gpu_benchmark.py`. Whether cuDNN benchmark mode is enabled in that benchmark is unknown from the data available. Notably, ResNet-50 at 4 and 8 GPUs shows B300 matching or greatly exceeding B200 (43,329 vs 43,370 at 4 GPU; 112,769 vs 59,251 at 8 GPU), so the issue is specific to single-GPU execution.

**Regardless of the B300 delta, this is a genuine code quality issue.** Setting `cudnn.benchmark = True` is standard practice for any convolution benchmark with fixed input sizes.

**Recommended fix:** Add `torch.backends.cudnn.benchmark = True` in `__init__()`.

**Sources:**
- [torch.backends (PyTorch docs)](https://docs.pytorch.org/docs/stable/backends.html) -- defines `torch.backends.cudnn.benchmark`: "if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest"
- [Performance Tuning Guide (PyTorch tutorial)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- states: "autotuning is an efficient method to ensure the selection of the ideal algorithm for each convolution in the network"
- [Optimizing Convolutional Layers (NVIDIA Deep Learning Performance Guide)](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) -- explains cuDNN algorithm selection and per-architecture optimization

---

### Issue 4: DDP Data Parallel -- Incorrect FLOPS Calculation

**Location:** `benchmark_data_parallel()` method, lines 572-640 of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

**Problem (FLOPS calculation -- affects both GPUs equally):** Each GPU processes `local_batch = M // self.world_size` samples (line 585), but the FLOPS calculation on line 619 uses the full `M`:
```python
flops_per_iter = 2 * M * N * K * 3  # Approximate
```
This inflates reported TFLOPS by `world_size`x for both B200 and B300 equally. It is a reporting bug, not a B300-specific issue. The absolute TFLOPS numbers (e.g., "1033 TFLOPS" for B200 DDP 4096x4096) are inflated and should not be compared to single-GPU GEMM results.

**Problem (DDP regression -- B300-specific):** The -92% DDP regression (B200=1033 vs B300=82 for 4096x4096) is the largest anomaly in the data and cannot be explained by the FLOPS bug alone since it affects both GPUs equally. This magnitude suggests B300 is either falling back to single-GPU execution, hitting a severe NCCL communication bottleneck, or encountering a topology misconfiguration. The All-Reduce data supports this: at small message sizes (100 MB), B300 shows -47%, but at 500 MB it recovers to -4%. The DDP benchmark uses small gradient synchronization messages that would be disproportionately affected.

**Recommended fix:** Fix FLOPS to use `local_batch` instead of `M`. Separately, investigate NCCL topology on B300 nodes (`nvidia-smi topo -m`, `NCCL_DEBUG=INFO`) and whether B300 needs explicit `NCCL_TOPO_FILE` or `NCCL_GRAPH_FILE`.

**Sources:**
- [DistributedDataParallel API reference (PyTorch docs)](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) -- documents `device_ids` parameter behavior
- [DDP Design Notes (PyTorch docs)](https://docs.pytorch.org/docs/stable/notes/ddp.html) -- explains device placement and gradient synchronization
- [NCCL Environment Variables (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) -- documents `NCCL_DEBUG`, `NCCL_TOPO_FILE`, `NCCL_TOPO_DUMP_FILE`, and other topology-related variables
- [NCCL Tuning for GB200 NVL Multi-Node (NVIDIA docs)](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html) -- Blackwell-specific NCCL configuration guidance
- [Understanding GB200 System Topology (NVIDIA docs)](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html) -- NVLink topology for Blackwell systems

---

### Issue 5: Device-to-Device Copy Expected Values Are Miscalibrated

**Location:** `benchmark_memory_bandwidth()` method of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py); `memory.device_to_device` section of both `expected_performance_B200.yaml` and `expected_performance_B300.yaml` in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** The D2D copy test copies between `src` and `dst` tensors on the **same device** (lines 228-229). This measures HBM copy bandwidth. The expected performance YAML files set the expected range to 7000-8000 GB/s:
```yaml
device_to_device:
  min_bandwidth_gbs: 7000.0
  max_bandwidth_gbs: 8000.0
  typical_bandwidth_gbs: 7500.0
```

These values correspond to the HBM3e **theoretical peak specification** (8 TB/s). A single `dst.copy_(src)` performs 1 read + 1 write and achieves roughly 35-45% of this theoretical peak in practice. Both GPUs are flagged "BELOW" when in fact their results are normal for this test methodology.

**Does this explain the B300 vs B200 delta?** Partially. At 512 MB, B300 (1547 GB/s) is 47% lower than B200 (2937 GB/s). But at 1024 MB, they are nearly identical (B300=3031 vs B200=3002). This follows the same size-dependent pattern seen in GEMM -- B300 has higher overhead for small transfers but achieves parity at larger sizes.

**Recommended fix:** Correct expected D2D values to ~2500-3500 GB/s (realistic single-copy HBM bandwidth).

**Sources:**
- [NVIDIA Blackwell Architecture Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) -- B200/B300 HBM3e spec: 8 TB/s memory bandwidth (theoretical bidirectional peak)
- [CUDA Demo Suite -- bandwidthTest (NVIDIA docs)](https://docs.nvidia.com/cuda/demo-suite/index.html) -- documents device-to-device copy bandwidth measurement methodology
- [CUDA C++ Best Practices Guide (NVIDIA docs)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) -- discusses memory bandwidth measurement and optimization

---

## Secondary Issues

### Issue 6: P2P Benchmark Uses Blocking Send/Recv

**Location:** `benchmark_peer_to_peer()` method, lines 695-776 of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

Rank 0 uses `dist.send()` / `dist.recv()` and rank 1 uses the mirror pattern. These are blocking operations that serialize the bidirectional transfer. Using `dist.isend()` / `dist.irecv()` (async) would allow simultaneous bidirectional transfer, potentially doubling measured bandwidth. This affects both GPUs equally.

**Sources:**
- [Distributed Communication Package (PyTorch docs)](https://docs.pytorch.org/docs/stable/distributed.html) -- documents `isend()`, `irecv()`, and `batch_isend_irecv()` async alternatives
- [Writing Distributed Applications with PyTorch (tutorial)](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) -- examples of blocking vs non-blocking P2P communication

---

### Issue 7: No Warmup Between Architecture-Sensitive Tests

**Location:** `run_full_suite()` method of [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

The benchmark calls `self.warmup()` once, then runs all test categories (GEMM, Conv2D, Attention, Memory, Multi-GPU) without intermediate warmup. Individual tests have their own small warmup loops (e.g., 10 iterations for matmul, 5 for attention). Given the observed pattern that B300 has higher small-problem overhead (consistent with JIT compilation or kernel caching differences), insufficient warmup could disproportionately penalize B300.

**Sources:**
- [Performance Tuning Guide (PyTorch tutorial)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- discusses warmup runs before measurement, particularly for JIT-compiled models and cuDNN autotuning

---

### Issue 8: Expected Performance YAML Thresholds May Be Miscalibrated

**Location:** [`expected_performance_B300.yaml`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/expected_performance_B300.yaml) in repo `rcc-uchicago/nodes-testing-tools`

The B300 expected ranges appear to be derived from B200 values with a ~7-8% uplift applied uniformly, rather than from actual B300 silicon characterization:

| Metric | B200 typical | B300 typical | Ratio |
|---|---|---|---|
| GEMM FP16 small | 1800 | 1950 | 1.083x |
| GEMM FP16 large | 2000 | 2150 | 1.075x |
| Conv2D FP16 | 1850 | 2000 | 1.081x |
| Attention FP16 large | 1750 | 1900 | 1.086x |

The uniform ~8% scaling suggests copy-paste adjustment rather than empirical characterization. Additionally, both B200 and B300 actual GEMM results fall BELOW their respective expected ranges (B200 at 1504 vs expected min 1600; B300 at 782 vs expected min 1750), suggesting the expected values may be too optimistic for both GPUs.

**Sources:**
- [NVIDIA Blackwell Architecture Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) -- official B200 specs for comparison
- [NVIDIA GB300 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) -- B300 specs
- [Inside NVIDIA Blackwell Ultra (NVIDIA developer blog)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) -- B300 architecture details and performance claims

---

### Issue 9: Additional Code Bugs in gpu_benchmark.py

**Location:** [`gpu_benchmark.py`](https://github.com/rcc-uchicago/nodes-testing-tools/blob/main/ml-training/gpu_benchmark.py) in repo `rcc-uchicago/nodes-testing-tools`

Several additional code quality issues affect both GPUs equally:

- **Unbound variable bug (line 776):** For ranks >= 2 in the P2P benchmark, `bandwidth` is never assigned before `return bandwidth if self.rank == 0 else 0`, causing a `NameError` on multi-node runs with more than 2 GPUs.
- **No `torch.no_grad()` or `torch.inference_mode()`** on inference-only benchmarks (matmul, conv2d, attention, memory), adding unnecessary autograd overhead to measurements.
- **No FP8 benchmarks** despite targeting B200/B300 which have dedicated FP8 Tensor Core support. FP8 is the key datatype for Blackwell's Transformer Engine.

These do not explain the B300 vs B200 delta but affect the quality and accuracy of all reported numbers.

---

### Issue 10: vLLM Not Optimized for Blackwell

**Source:** Presentation slide 12 (DSAI-bench-16Mar2026.pdf)

The presentation explicitly notes: "the employed open-source inference software (like vLLM) has not yet been optimized for the Blackwell's new hardware-level Transformer Engines." This affects LLM inference results at FP8/FP4 precision. Slide 12 shows that for the 8B model at 1 GPU, FP8 and FP4 performance drops sharply on B300 (FP8≈3,383 TPS, FP4≈3,362 TPS) compared to BF16 (~8,158 TPS), while B200 maintains higher FP8/FP4 throughput. The slide also notes: "There is a drop in FP8/FP4 performance for the 1 GPU runs." At 4 GPUs, both B200 and B300 show similar FP8/FP4 levels (~9,400-9,700 TPS). BF16 inference performance is normal (B300 matches or slightly exceeds B200), confirming this is a software optimization issue for new precision formats, not a hardware defect.

**Note:** Slide 12's BF16 bar labels appear to swap B200/B300 values compared to slide 10 (B200 shows 8,158 on slide 12 but 7,632 on slide 10). This may be a labeling error in the presentation; we use slide 10 BF16 values as authoritative since that slide is dedicated to BF16 results.

---

### Issue 11: CUDA/Driver Maturity -- Likely Primary Root Cause

B300 is newer silicon that may have a different CUDA compute capability than B200. If the installed CUDA toolkit does not have fully optimized kernels for B300, all operations would fall back to generic code paths. This is the **most parsimonious explanation** for the observed pattern:

- **Small problems**: Launch overhead and kernel selection dominate execution time. Generic/fallback kernels have higher setup costs, causing large percentage regressions (-48% for small GEMM, -47% for small All-Reduce).
- **Large problems**: Compute and bandwidth dominate. B300's higher clocks provide a genuine advantage, and the fixed overhead of suboptimal kernel selection is amortized (+16% for large GEMM, +1% for large D2D).
- **Well-optimized workloads**: GROMACS, ViT-Large, and LLM inference likely use their own optimized kernels or `torch.compile()`, bypassing the immature eager-mode paths.

This explanation is consistent with the GEMM and D2D size-dependent pattern, the ResNet-50 single-GPU anomaly with multi-GPU recovery, and the normal performance on optimized workloads. Communication results (All-Reduce, P2P) are inconsistent across runs and likely reflect additional system-level configuration issues (see conflicting runs table above).

**Sources:**
- [CUDA Blackwell Tuning Guide (NVIDIA docs)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- Blackwell-specific CUDA optimization guidance
- [NCCL Environment Variables (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) -- NCCL configuration for newer architectures

---

## Recommended Investigation Steps

### Step 1: Verify CUDA Compute Capability and Toolkit Support
- [ ] Run `nvidia-smi` on B300 nodes to confirm actual compute capability (10.0? 10.3? other?)
- [ ] Verify installed CUDA toolkit version has native support for B300's compute capability
- [ ] Check if cuBLAS/cuDNN have B300-specific kernel libraries installed
- [ ] Compare `CUDA_VISIBLE_DEVICES` and GPU enumeration between B200 and B300 nodes

### Step 2: Investigate NCCL/NVLink Configuration on B300
- [ ] Run `nvidia-smi topo -m` on B300 nodes to verify NVLink connectivity
- [ ] Run with `NCCL_DEBUG=INFO` to check topology detection
- [ ] Check if B300 needs `NCCL_TOPO_FILE` or `NCCL_GRAPH_FILE`
- [ ] Compare NCCL topology output between B200 and B300 nodes

### Step 3: Run Size-Sweep Diagnostic
- [ ] Run GEMM benchmark at sizes 1024, 2048, 4096, 8192, 16384 on both GPUs to confirm size-dependent pattern
- [ ] Run D2D copy at 128, 256, 512, 1024, 2048, 4096 MB to find the crossover point
- [ ] This will confirm whether B300's penalty is purely an overhead/launch-latency issue

### Step 4: Fix Benchmark Code Quality Issues
Modify `gpu_benchmark.py` in `rcc-uchicago/nodes-testing-tools`:
- [ ] Replace manual attention with `F.scaled_dot_product_attention()`
- [ ] Add `torch.backends.cudnn.benchmark = True`
- [ ] Add `torch.compile(mode="max-autotune")` option
- [ ] Fix DDP FLOPS calculation to use `local_batch`
- [ ] Fix P2P benchmark to use async ops
- [ ] Correct expected D2D bandwidth values in YAML files

### Step 5: Investigate ResNet-50 Single-GPU Anomaly
- [ ] Review [`MohsenZand/GPUBenchmarking`](https://github.com/MohsenZand/GPUBenchmarking) (`cv.py`) which produced the ResNet-50 and ViT-Large results
- [ ] Check if that code enables `cudnn.benchmark` and `torch.compile()`
- [ ] The fact that ResNet-50 recovers at 4 GPU (identical to B200) and exceeds B200 at 8 GPU suggests the single-GPU issue is a specific kernel/configuration problem, not a fundamental compute limitation

### Step 6: Re-run Benchmarks
Run the fixed benchmark code on both B200 and B300, then compare. Expected outcome: with optimized code paths and correct CUDA support, B300 should match or exceed B200 by ~7% across all workloads.

---

## Issue-to-Anomaly Mapping

| Anomaly | Likely Root Cause(s) | Confidence | Notes |
|---|---|---|---|
| GEMM small -48%, large +16% | Issue 11 (CUDA maturity / kernel launch overhead) | **High** | Size-dependent pattern is strong evidence |
| Attention -28% | Issue 1 (no SDPA) + Issue 11 (CUDA maturity) | **Medium** | SDPA is a real issue but both GPUs have same Tensor Cores |
| DDP -92% | Issue 4 (NCCL topology) + Issue 11 | **High** | Magnitude suggests config/topology issue, not just code |
| All-Reduce 100MB: -47% (Run A) / +1% (Run B) | Issue 11 (NCCL config) + system variability | **Medium** | Contradictory results across runs; pattern unreliable |
| D2D 512MB -47%, 1024MB +1% | Issue 5 (miscalibrated expected) + Issue 11 | **High** | Both GPUs flagged "BELOW" due to wrong expected values |
| ResNet-50 1-GPU -70% | Unknown (separate benchmark: [`MohsenZand/GPUBenchmarking`](https://github.com/MohsenZand/GPUBenchmarking)) | **Low** | Recovers at 4 GPU, exceeds at 8 GPU; not from gpu_benchmark.py |
| Pangu ML -48% (all configs) | Issue 11 + possibly Issues 1-2; see [`ndtrung81/3dViT-training`](https://github.com/ndtrung81/3dViT-training) | **Medium** | Consistent across precisions and GPU counts |
| LAMMPS -8% to -19% (all sizes) | Issue 11 (CUDA maturity for HPC kernels) | **Medium** | B300 trails B200 at all particle counts; presentation note "performs surprisingly well with FP64" refers to B300 vs H200, not vs B200 |
| Conv2D -1% | No B300-specific issue | **High** | Essentially identical; cuDNN benchmark would help both GPUs |
| P2P 500MB: +3% (Run A) / -22% (Run B) | System variability + possibly Issue 11 | **Low** | Contradictory results; NVLink hardware likely fine but NCCL config unstable |
| ViT-Large: +6% (1 GPU), -15% (4 GPU), +5% (8 GPU) | Issue 11 at 4 GPU; fine at 1 and 8 GPU | **Medium** | 4 GPU regression not explained by any identified code issue; may be multi-GPU scaling/NCCL related |
| LLM BF16 +7% | No issue -- B300 performs correctly | **High** | Well-optimized inference stack works as expected |

---

## Conclusions

1. **The B300 hardware is not defective.** It matches or exceeds B200 on large GEMM/D2D problems, optimized workloads (ViT-Large, LLM inference BF16), and at scale (ResNet-50 at 8 GPUs: 112,769 FPS vs B200's 59,251 -- +90%). Communication results are inconsistent across runs, suggesting system-level configuration issues rather than hardware limitations.

2. **The primary issue is CUDA/driver kernel maturity for B300**, manifesting as higher kernel launch overhead and suboptimal kernel selection for small problem sizes. This is the most parsimonious explanation for the size-dependent regression pattern.

3. **The benchmark code has genuine quality issues** (manual attention, no cuDNN benchmark mode, no `torch.compile()`, FLOPS bug, miscalibrated expected values) that affect both GPUs. These should be fixed regardless of the B300 investigation, but they largely do **not** explain why B300 specifically underperforms B200.

4. **The DDP -92% regression is the most severe outlier** and likely involves NCCL topology misconfiguration specific to B300 nodes. This requires targeted investigation with `NCCL_DEBUG=INFO` and `nvidia-smi topo -m`.

5. **The ResNet-50 single-GPU anomaly** (B300=4,444 vs B200=15,081) is from a separate benchmark (not `gpu_benchmark.py`) and resolves itself at 4+ GPUs. It warrants separate investigation of that benchmark's configuration.
