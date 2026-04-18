# LOB-TCN Project [TPU Version]

### GCP Cloud TPU VM Configuration

| Configuration | Details |
| :--- | :--- |
| **Node Status** | ✅ Running (`tpu-v6e-node`) |
| **Zone** | `europe-west4-a` |
| **TPU Type** | `v6e-4` (4 TPU v6e cores) |
| **Software Version**| `v2-alpha-tpuv6e` |

### TPU Core Utilization Analysis

| Scenario | Execution Logic | TPU Hardware Status |
| :--- | :--- | :--- |
| **Standard `@jax.jit`** | Executes the compiled graph on a single default device (typically `Devices[0]`). | **1 Core Active**, 3 Cores Idle. |
| **`jax.pmap` (Parallel Map)** | Replicates the graph and distributes data chunks across all available devices. | **All 4 Cores Active** (Parallel Processing). |
| **Implicit Parallelism** | JAX does NOT automatically split your batch across TPU cores unless explicitly told to. | **Single Device** restriction. |

### Performance Benchmark: CPU vs. Single-Core TPU v6e

| Hardware Configuration | Processing Time per Epoch | Acceleration Factor |
| :--- | :--- | :--- |
| **CPU (Baseline)** | ~90.0 seconds | 1x |
| **Cloud TPU v6e (Single Core)** | ~1.0 second | **~90x** |

**Observations:**
- **Execution Efficiency**: The TCN model architecture (utilizing Dilated Convolutions) is highly optimized for XLA (Accelerated Linear Algebra) compilation, leading to massive speedups on TPU hardware.
- **Backend Success**: The environment successfully utilized the `libtpu.so` driver with the `PJRT_DEVICE=TPU` configuration.
- **Optimization Potential**: This result was achieved using only **1 out of 4** available TPU cores. Implementing `jax.pmap` for multi-core parallelism will further enhance throughput for larger datasets.

### Performance Anomaly Record: Single vs. Multi-TPU Sharding

| Configuration | Epoch 1 (Compile) | Epoch 2 | Epoch 3 | Stable Time/Epoch |
| :--- | :--- | :--- | :--- | :--- |
| **Single TPU (No Sharding)** | ~3.25s | ~1.88s | ~1.85s | **~1.86s** |
| **4-Core TPU (Auto-Sharding)**| ~4.07s | ~3.15s | ~3.13s | **~3.14s** |

**Diagnostic Analysis:**
Paradoxically, the 4-core sharded training is significantly slower than the single-core baseline. This indicates that the computation graph is highly optimized, but the system is severely bound by **I/O and Communication Overheads**.

**Hypotheses & Root Causes:**

1. **The Host-to-Device (H2D) Bottleneck (Confirmed):** The current loop uses `batch_x.numpy()` followed by `jax.device_put()`. This forces the data to travel from the TensorFlow C++ backend -> Python CPU RAM (NumPy) -> JAX Sharding Logic -> PCIe/Network -> 4 separate TPU HBMs. For a relatively small dataset, the time taken by the CPU to slice and dispatch the data to 4 cores is longer than the time the TPUs take to perform the actual math.

2. **Graph Recompilation (Partially Ruled Out):**
   If the batch sizes were misaligned, JAX would recompile the graph frequently, causing every epoch to be as slow as Epoch 1. However, since Epochs 2 and 3 stabilize (~3.14s), the XLA compilation only happens once. The issue is purely the per-step data transfer latency.

3. **Missing Asynchronous Pipeline:**
   The TPUs are currently "starving". They compute the batch in milliseconds, but then sit idle waiting for the CPU to prepare and send the next batch.