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