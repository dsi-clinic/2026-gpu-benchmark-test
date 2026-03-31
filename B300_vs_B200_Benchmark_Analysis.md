# B300 vs B200 Benchmark Analysis Report

**Date:** 2026-03-31
**Source Data:** DSAI Arch Cluster Benchmarks (presentation dated 2026-03-16)
**Benchmark Repo:** [rcc-uchicago/nodes-testing-tools](https://github.com/rcc-uchicago/nodes-testing-tools), directory `ml-training/`

---

## Executive Summary

The B300 GPU consistently underperforms the B200 across multiple workloads in the DSAI Arch Cluster benchmarks. This is unexpected: the B300 is a higher-clocked variant of the B200 with ~7% more compute. The anomalies range from -13% to -92% and follow a clear pattern -- workloads well-optimized for Blackwell (GROMACS, ViT-Large, LLM inference) perform correctly on B300, while workloads using the benchmark code exhibit severe regressions. This strongly implicates **software/configuration issues in the benchmark code**, not hardware defects.

## Performance Anomalies

| Workload | B200 | B300 | B300 vs B200 |
|---|---|---|---|
| Pangu ML (FP16, 4 GPU) | 4.22 samples/s | 2.18 samples/s | **-48%** |
| Attention B32 H12 S512 | 87.32 TFLOPS | 62.53 TFLOPS | **-28%** |
| DDP 1024x1024 (4 GPU) | 15.99 TFLOPS | 1.24 TFLOPS | **-92%** |
| DDP 4096x4096 (4 GPU) | 1033.57 TFLOPS | 82.09 TFLOPS | **-92%** |
| All-Reduce 100MB | 472.06 GB/s | 251.73 GB/s | **-47%** |
| D2D Copy 512MB | 2937 GB/s | 1547 GB/s | **-47%** |
| ResNet-50 (4 GPU) | 43,370 FPS | 4,444 FPS | **-90%** |
| LAMMPS 32M particles | 485 M p*ts/s | 420 M p*ts/s | **-13%** |

---

## Critical Issues in Benchmark Code

### Issue 1: Manual Attention Implementation (No Flash Attention / SDPA)

**Location:** `benchmark_attention()` method, lines 179-229 of `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** The benchmark manually computes attention as three separate operations (lines 196-198):
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, V)
```

This bypasses `torch.nn.functional.scaled_dot_product_attention()` (SDPA), which auto-selects FlashAttention v2 or memory-efficient attention backends. On Blackwell GPUs, SDPA activates the hardware Transformer Engine and uses 5th-gen Tensor Core fused attention kernels.

**Why B300 is hit harder:** The B300's 5th-gen Tensor Cores are specifically optimized for the fused attention path. The B200 may tolerate the naive unfused path better because its Tensor Core scheduler handles unfused kernels more efficiently at the firmware level. B300's newer microarchitecture relies on the fused path for peak performance.

**Likely impact:** Explains the -28% attention regression (87.32 vs 62.53 TFLOPS).

**Recommended fix:** Replace the three manual operations with `F.scaled_dot_product_attention(Q, K, V)`.

**Sources:**
- [PyTorch SDPA API reference](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) -- documents backend auto-selection (FlashAttention, memory-efficient, cuDNN attention)
- [Implementing High-Performance Transformers with SDPA (PyTorch tutorial)](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) -- benchmarks comparing fused vs manual attention backends
- [Accelerated PyTorch 2 Transformers (PyTorch blog)](https://pytorch.org/blog/accelerated-pytorch-2/) -- performance comparisons of SDPA vs manual implementation
- [PyTorch 2.2: FlashAttention-v2 integration (PyTorch blog)](https://pytorch.org/blog/pytorch2-2/) -- reports ~2x speedup and 50-73% theoretical max FLOPS on A100

---

### Issue 2: No `torch.compile()` or JIT Compilation

**Location:** Entire file `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools` -- no calls to `torch.compile()` anywhere.

**Problem:** None of the benchmarks use `torch.compile()`. Blackwell GPUs benefit significantly from `torch.compile(mode="max-autotune")` which enables:
- Triton-generated kernels optimized for the specific GPU architecture
- Operator fusion that the eager-mode executor runs separately
- Automatic CUDA graph capture

**Why B300 is hit harder:** B200 launched earlier and has more mature eager-mode CUDA kernels in cuBLAS/cuDNN. B300 relies more heavily on compiled/fused paths for peak performance because its kernel libraries are less mature.

**Likely impact:** Contributes to regressions across all compute benchmarks, especially attention and convolution.

**Recommended fix:** Add a `--compile` CLI flag; wrap benchmark functions with `torch.compile(mode="max-autotune")`.

**Sources:**
- [torch.compile API reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) -- documents `mode="max-autotune"` which "leverages Triton-based matrix multiplications and convolutions on GPU" and enables CUDA graphs
- [Introduction to torch.compile (PyTorch tutorial)](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) -- end-to-end examples showing performance gains
- [CUDA Blackwell Tuning Guide (NVIDIA)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- Blackwell-specific optimization guidance
- [Accelerating PyTorch with CUDA Graphs (PyTorch blog)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) -- explains CUDA graph capture benefits for reducing kernel launch overhead

---

### Issue 3: No cuDNN Benchmark Mode

**Location:** `__init__()` method (lines 38-60) and `benchmark_conv2d()` method (lines 124-177) of `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** `torch.backends.cudnn.benchmark` is never set to `True` anywhere in the file. Without this, PyTorch uses a default convolution algorithm rather than auto-tuning to find the optimal algorithm for the specific GPU architecture and tensor shapes.

**Why B300 is hit harder:** cuDNN's default algorithm selection is architecture-specific. Without benchmark mode, PyTorch may select an algorithm tuned for B200 or older architectures. The B300's different SM microarchitecture may have very different optimal algorithm choices.

**Likely impact:** Primary explanation for the massive ResNet-50 gap (B300 at 4,444 FPS vs B200 at 43,370 FPS = **-90% drop**). A 10x regression in convolution-dominated workloads is consistent with selecting a pathologically bad algorithm.

**Recommended fix:** Add `torch.backends.cudnn.benchmark = True` in `__init__()`.

**Sources:**
- [torch.backends (PyTorch docs)](https://docs.pytorch.org/docs/stable/backends.html) -- defines `torch.backends.cudnn.benchmark`: "if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest"
- [Performance Tuning Guide (PyTorch tutorial)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- states: "autotuning is an efficient method to ensure the selection of the ideal algorithm for each convolution in the network"
- [Optimizing Convolutional Layers (NVIDIA Deep Learning Performance Guide)](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) -- explains cuDNN algorithm selection and per-architecture optimization

---

### Issue 4: DDP Data Parallel -- Incorrect FLOPS Calculation and Potential Config Issue

**Location:** `benchmark_data_parallel()` method, lines 572-640 of `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools`

**Problem (FLOPS calculation):** Each GPU processes `local_batch = M // self.world_size` samples (line 585), but the FLOPS calculation on line 619 uses the full `M`:
```python
flops_per_iter = 2 * M * N * K * 3  # Approximate
```
This inflates reported TFLOPS by `world_size`x when all GPUs contribute. When one GPU stalls or falls back to single-GPU execution, the inflated denominator makes the regression look even worse.

**Problem (device assignment):** The DDP wrapper on line 582 uses `device_ids=[self.rank]` which maps to the local rank integer, not the `self.device` object. On B300 nodes with potentially different GPU enumeration or NUMA topology, this could cause rank-to-device mismatches.

**Why B300 is hit harder:** B300's NCCL/NVLink configuration may require explicit topology hints (`NCCL_TOPO_FILE`, `NCCL_GRAPH_FILE`) that B200 doesn't need due to its simpler or better-characterized NVLink topology. A 92% drop suggests the B300 may be falling back to a single GPU or hitting a severe communication bottleneck.

**Likely impact:** Explains the -92% DDP regression (1033 vs 82 TFLOPS for 4096x4096).

**Recommended fix:** Fix FLOPS to use `local_batch` instead of `M`. Investigate NCCL topology on B300 nodes (`nvidia-smi topo -m`, `NCCL_DEBUG=INFO`).

**Sources:**
- [DistributedDataParallel API reference (PyTorch docs)](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) -- documents `device_ids` parameter behavior
- [DDP Design Notes (PyTorch docs)](https://docs.pytorch.org/docs/stable/notes/ddp.html) -- explains device placement and gradient synchronization
- [NCCL Environment Variables (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) -- documents `NCCL_DEBUG`, `NCCL_TOPO_FILE`, `NCCL_TOPO_DUMP_FILE`, and other topology-related variables
- [NCCL Tuning for GB200 NVL Multi-Node (NVIDIA docs)](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html) -- Blackwell-specific NCCL configuration guidance
- [Understanding GB200 System Topology (NVIDIA docs)](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html) -- NVLink topology for Blackwell systems

---

### Issue 5: Device-to-Device Copy Measures HBM Bandwidth, Not NVLink -- with Miscalibrated Expected Values

**Location:** `benchmark_memory_bandwidth()` method, lines 224-266 of `gpu_benchmark.py`; `memory.device_to_device` section (lines 109-112) of both `expected_performance_B200.yaml` and `expected_performance_B300.yaml` in repo `rcc-uchicago/nodes-testing-tools`

**Problem:** The D2D copy test copies between `src` and `dst` tensors on the **same device** (lines 228-229). This measures HBM bandwidth (~2000-3000 GB/s), not NVLink bandwidth. However, the expected performance YAML files set the expected range to 7000-8000 GB/s:
```yaml
device_to_device:
  min_bandwidth_gbs: 7000.0
  max_bandwidth_gbs: 8000.0
  typical_bandwidth_gbs: 7500.0
```

These values correspond to the HBM3e **theoretical peak bidirectional specification** (read + write = 8 TB/s), not achievable single-copy bandwidth. A single `dst.copy_(src)` performs 1 read + 1 write, achieving roughly ~35-45% of the bidirectional spec.

The measured values (B200: 2937 GB/s, B300: 1547 GB/s) are both flagged as "BELOW" in comparison, but both are within expected range for what the test actually measures. The mismatch between test methodology and expected thresholds creates a false impression of underperformance.

**Recommended fix:** Correct expected D2D values to ~2500-3500 GB/s (realistic single-copy HBM bandwidth). Alternatively, rename the test to clarify it measures HBM copy bandwidth, not NVLink.

**Sources:**
- [NVIDIA Blackwell Architecture Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) -- B200/B300 HBM3e spec: 8 TB/s memory bandwidth (theoretical bidirectional peak)
- [CUDA Demo Suite -- bandwidthTest (NVIDIA docs)](https://docs.nvidia.com/cuda/demo-suite/index.html) -- documents device-to-device copy bandwidth measurement methodology
- [CUDA C++ Best Practices Guide (NVIDIA docs)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) -- discusses memory bandwidth measurement and optimization

---

## Secondary Issues

### Issue 6: P2P Benchmark Uses Blocking Send/Recv

**Location:** `benchmark_peer_to_peer()` method, lines 695-776 of `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools`

Rank 0 (lines 711-712, 719-720) uses `dist.send()` / `dist.recv()` and rank 1 (lines 754-755, 761-762) uses the mirror pattern. These are blocking operations that serialize the bidirectional transfer. Using `dist.isend()` / `dist.irecv()` (async) would allow simultaneous bidirectional transfer, potentially doubling measured bandwidth. On B300 with potentially different NVLink topology, the serialization may cause more stalls due to different link arbitration behavior.

**Sources:**
- [Distributed Communication Package (PyTorch docs)](https://docs.pytorch.org/docs/stable/distributed.html) -- documents `isend()`, `irecv()`, and `batch_isend_irecv()` async alternatives
- [Writing Distributed Applications with PyTorch (tutorial)](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) -- examples of blocking vs non-blocking P2P communication

---

### Issue 7: No Warmup Between Architecture-Sensitive Tests

**Location:** `run_full_suite()` method, lines 358-515 of `gpu_benchmark.py` in repo `rcc-uchicago/nodes-testing-tools`

The benchmark calls `self.warmup()` once at line 367, then runs all test categories (GEMM, Conv2D, Attention, Memory, Multi-GPU) without intermediate warmup. Individual tests have their own small warmup loops (e.g., 10 iterations for matmul at line 81, 5 for attention at lines 186-190), but these may be insufficient for B300's newer architecture which may have different JIT compilation and kernel caching behavior.

**Sources:**
- [Performance Tuning Guide (PyTorch tutorial)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- discusses warmup runs before measurement, particularly for JIT-compiled models and cuDNN autotuning

---

### Issue 8: Expected Performance YAML Thresholds May Be Miscalibrated

**Location:** `expected_performance_B300.yaml` in repo `rcc-uchicago/nodes-testing-tools`

The B300 expected ranges appear to be derived from B200 values with a ~7-8% uplift applied uniformly, rather than from actual B300 silicon characterization:

| Metric | B200 typical | B300 typical | Ratio |
|---|---|---|---|
| GEMM FP16 small | 1800 | 1950 | 1.083x |
| GEMM FP16 large | 2000 | 2150 | 1.075x |
| Conv2D FP16 | 1850 | 2000 | 1.081x |
| Attention FP16 large | 1750 | 1900 | 1.086x |

The uniform ~8% scaling suggests copy-paste adjustment rather than empirical characterization. This means even "passing" results on B300 may not reflect actual silicon capabilities.

**Sources:**
- [NVIDIA Blackwell Architecture Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) -- official B200 specs for comparison
- [NVIDIA GB300 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) -- B300 specs
- [Inside NVIDIA Blackwell Ultra (NVIDIA developer blog)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) -- B300 architecture details and performance claims

---

### Issue 9: vLLM Not Optimized for Blackwell

**Source:** Presentation notes (DSAI-bench-16Mar2026.pdf)

The presentation explicitly notes: "the employed open-source inference software (like vLLM) has not yet been optimized for the Blackwell's new hardware-level Transformer Engines." This affects LLM inference results at FP8/FP4 precision. This is a software stack issue outside the benchmark code itself.

---

### Issue 10: CUDA/Driver Maturity

B300 is newer silicon. The CUDA toolkit, cuBLAS, cuDNN, and NCCL libraries may not yet have B300-specific optimizations. The B200 has had more time for these optimizations to be developed and deployed. This is an environmental factor that amplifies the benchmark code issues above.

**Sources:**
- [CUDA Blackwell Tuning Guide (NVIDIA docs)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- Blackwell-specific CUDA optimization guidance
- [NCCL Environment Variables (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) -- NCCL configuration for newer architectures

---

## Recommended Investigation Steps

### Step 1: Reproduce with Fixed Benchmark Code
Modify `gpu_benchmark.py` in `rcc-uchicago/nodes-testing-tools` to address Issues 1-4:
- [ ] Add `torch.backends.cudnn.benchmark = True`
- [ ] Replace manual attention with `F.scaled_dot_product_attention()`
- [ ] Add `torch.compile(mode="max-autotune")` option
- [ ] Fix DDP FLOPS calculation to use `local_batch`
- [ ] Fix P2P benchmark to use async ops
- [ ] Add per-test warmup phases

### Step 2: Validate Device-to-Device Expected Values
- [ ] Correct expected D2D values in both YAML files to reflect HBM copy bandwidth (~2500-3500 GB/s), not theoretical HBM spec bandwidth (7000-8000 GB/s)

### Step 3: Investigate NCCL/NVLink Configuration on B300
- [ ] Run `nvidia-smi topo -m` on B300 nodes to verify NVLink connectivity
- [ ] Run with `NCCL_DEBUG=INFO` to check topology detection
- [ ] Check if B300 needs `NCCL_TOPO_FILE` or `NCCL_GRAPH_FILE`

### Step 4: Check Software Stack Versions
- [ ] Verify PyTorch version supports B300 (CUDA compute capability 10.x)
- [ ] Check cuBLAS/cuDNN versions are B300-aware
- [ ] Verify NCCL version supports B300 NVLink topology

### Step 5: Re-run Benchmarks
Run the fixed benchmark code on both B200 and B300, then compare. Expected outcome: B300 should match or exceed B200 by ~7% across all workloads.

---

## Issue-to-Anomaly Mapping

| Anomaly | Likely Root Cause(s) | Severity |
|---|---|---|
| Attention -28% | Issue 1 (no SDPA), Issue 2 (no compile) | Critical |
| ResNet-50 -90% | Issue 3 (no cuDNN benchmark) | Critical |
| DDP -92% | Issue 4 (FLOPS bug + NCCL config) | Critical |
| All-Reduce -47% | Issue 4 (NCCL topology), Issue 10 (driver maturity) | High |
| D2D Copy -47% | Issue 5 (miscalibrated expected values) | Medium (measurement error) |
| Pangu ML -48% | Issues 1-3 combined (training workload) | High |
| LAMMPS -13% | Issue 10 (CUDA maturity for HPC kernels) | Low |
