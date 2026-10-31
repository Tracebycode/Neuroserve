# NeuroServe

**NeuroServe** is a high-performance C++20 AI inference serving system designed to demonstrate robust systems engineering, concurrent networking, bounded queueing, adaptive dynamic batching, resource pooling, and load shedding.

---

## 1. System Architecture

```text
                  [ Concurrent Clients ]
                            │ HTTP POST /v1/infer
                            ▼
               ┌──────────────────────────┐
               │    HttpServer / Network  │ (Lightweight, non-blocking HTTP)
               └────────────┬─────────────┘
                            │ Deserializes Request & Input Tensors
                            ▼
               ┌──────────────────────────┐
               │     InferenceService     │ (Decoupled core engine)
               └────────────┬─────────────┘
                            │
               ┌────────────▼─────────────┐
               │       LoadShedder        │ ◄── Rejects with 429 if queue full
               └────────────┬─────────────┘
                            │ Enqueue (Bounded RAII promise)
                            ▼
               ┌──────────────────────────┐
               │   BoundedRequestQueue    │ (Thread-safe, mutex & CV contention minimized)
               └────────────┬─────────────┘
                            │ Dequeue batch (N items or max_wait_ms timeout)
                            ▼
               ┌──────────────────────────┐
               │      DynamicBatcher      │ (Coalesces multiple requests along dim 0)
               └────────────┬─────────────┘
                            │ Dispatches batch task
                            ▼
               ┌──────────────────────────┐
               │       WorkerPool         │ (Fixed std::jthread worker pool)
               └────────────┬─────────────┘
                            │ Acquire available model session (ScopedLease)
                            ▼
               ┌──────────────────────────┐
               │    ModelInstancePool     │ (Reusable Ort::Session instances)
               └────────────┬─────────────┘
                            │ Single batched inference run
                            ▼
               ┌──────────────────────────┐
               │     ONNX Runtime C++     │ (CPU / TensorRT execution providers)
               └────────────┬─────────────┘
                            │ Slice batched outputs & fulfill promises
                            ▼
               ┌──────────────────────────┐
               │   Metrics Registry       │ (Monotonic clock: p50, p95, p99, queue vs infer)
               └────────────┬─────────────┘
                            │
                            ▼
                  [ Client Receives 200 ]
```

---

## 2. Key Systems Engineering Highlights

- **C++20 Concurrency**: Utilizes `std::jthread` with cooperative `std::stop_token`, `std::unique_lock`, `std::condition_variable`, and atomics with explicit memory ordering.
- **Bounded Queue & Load Shedding**: Prevents catastrophic out-of-memory (OOM) failures under burst traffic by rejecting excessive requests in \(\mathcal{O}(1)\) time with `HTTP 429 Too Many Requests`.
- **Dynamic Request Batching**: Dynamically coalesces concurrent requests into a single tensor along dimension 0, amortizing GEMM weight loading and dramatically boosting memory bandwidth efficiency.
- **Model Instance Pooling**: Pre-allocates and reuses `Ort::Session` instances via an RAII lease (`ScopedLease`), avoiding session creation latency and isolating thread execution.
- **Decoupled Architecture**: The core inference pipeline (`InferenceService`, `DynamicBatcher`, `WorkerPool`, `ModelManager`) is completely independent of the transport/HTTP layer.
- **Production Observability**: Live Prometheus-compatible `/metrics` endpoint calculating running percentiles (p50, p95, p99) for end-to-end, queue wait, and model execution latencies.
- **Deterministic Graceful Shutdown**: Intercepts `SIGINT`/`SIGTERM`, stops network ingress, drains queued requests, joins all worker threads, and releases ONNX memory cleanly.

---

## 3. Technology Stack & Dependencies

| Dependency | Purpose | Integration Method |
| :--- | :--- | :--- |
| **C++20 (GCC 13.3+ / Clang 16+)** | Language standard: concepts, `std::jthread`, `std::span` | Native compiler |
| **CMake 3.22+** | Build configuration and dependency orchestration | Build system |
| **ONNX Runtime C++ (v1.18.0)** | High-performance graph execution engine | Prebuilt C++ SDK via FetchContent |
| **spdlog (v1.14.1)** | Fast, structured, leveled logging | CMake FetchContent |
| **yaml-cpp (v0.8.0)** | Strongly typed configuration file parsing | CMake FetchContent |
| **nlohmann/json (v3.11.3)** | JSON serialization and deserialization | CMake FetchContent |
| **cpp-httplib (v0.15.3)** | Lightweight HTTP networking | CMake FetchContent |
| **GoogleTest (v1.14.0)** | Unit and integration test suite | CMake FetchContent |
| **Google Benchmark (v1.8.3)** | Microbenchmarks and throughput analysis | CMake FetchContent |

---

## 4. Configuration (`config/config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8080

workers:
  count: 4

queue:
  capacity: 1024

batching:
  enabled: true
  max_batch_size: 8
  max_wait_ms: 5

model:
  name: "demo_mlp"
  path: "models/model.onnx"
  instances: 2

load_shedding:
  enabled: true
  max_queue_wait_ms: 100

logging:
  level: "info"
```

---

## 5. Build & Installation

### Prerequisites (Linux / WSL2 Ubuntu)
```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build python3 python3-pip
python3 -m pip install onnx numpy --break-system-packages
```

### Build Steps
```bash
# 1. Clone repository
git clone https://github.com/example/neuroserve.git
cd neuroserve

# 2. Generate test model
python3 models/generate_test_model.py

# 3. Configure and compile
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## 6. Running the Server

From the project root:
```bash
./build/neuroserve_server config/config.yaml
```

Output:
```text
[2026-10-31 20:23:52.052] [info] [thread 2778] ==================================================
[2026-10-31 20:23:52.052] [info] [thread 2778]   Starting NeuroServe C++20 High-Performance Server
[2026-10-31 20:23:52.052] [info] [thread 2778] ==================================================
[2026-10-31 20:23:52.052] [info] [thread 2778] Configuration loaded from: config/config.yaml
[2026-10-31 20:23:52.074] [info] [thread 2778] Loading model 'demo_mlp' from 'models/model.onnx' with 2 instances
[2026-10-31 20:23:52.104] [info] [thread 2778] WorkerPool initialized with 4 workers
[2026-10-31 20:23:52.104] [info] [thread 2778] Starting InferenceService...
[2026-10-31 20:23:52.104] [info] [thread 2778] Starting DynamicBatcher (max_batch_size=8, max_wait_ms=5)
[2026-10-31 20:23:52.104] [info] [thread 2778] InferenceService ready to accept traffic
[2026-10-31 20:23:52.104] [info] [thread 2778] Starting HTTP server on 0.0.0.0:8080
[2026-10-31 20:23:52.105] [info] [thread 2778] NeuroServe is running at http://0.0.0.0:8080
```

---

## 7. API Endpoints & Usage Examples

### 1. Health (Liveness)
```bash
curl -s http://127.0.0.1:8080/health
```
Response:
```json
{"status":"UP"}
```

### 2. Readiness
```bash
curl -s http://127.0.0.1:8080/ready
```
Response:
```json
{"status":"READY"}
```

### 3. Inference Request
```bash
curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "demo_mlp",
    "inputs": [
      {
        "name": "input",
        "shape": [1, 8],
        "data": [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]
      }
    ]
  }'
```
Response:
```json
{
  "request_id": 1,
  "success": true,
  "outputs": [
    {
      "name": "output",
      "shape": [1, 2],
      "data": [0.421, -0.187]
    }
  ],
  "latency_ms": {
    "queue": 0.059,
    "inference": 0.041,
    "total": 0.152
  }
}
```

### 4. Prometheus Metrics
```bash
curl -s http://127.0.0.1:8080/metrics
```
Output:
```text
# HELP neuroserve_requests_total Total number of inference requests received
# TYPE neuroserve_requests_total counter
neuroserve_requests_total 500

# HELP neuroserve_requests_success_total Total successful requests
# TYPE neuroserve_requests_success_total counter
neuroserve_requests_success_total 500

# HELP neuroserve_requests_rejected_total Total rejected requests due to load shedding
# TYPE neuroserve_requests_rejected_total counter
neuroserve_requests_rejected_total 0

# HELP neuroserve_queue_depth Current number of pending requests in queue
# TYPE neuroserve_queue_depth gauge
neuroserve_queue_depth 0

# HELP neuroserve_active_workers Current number of workers actively running tasks
# TYPE neuroserve_active_workers gauge
neuroserve_active_workers 0

# HELP neuroserve_inference_latency_ms Model execution latency in milliseconds
# TYPE neuroserve_inference_latency_ms summary
neuroserve_inference_latency_ms{quantile="0.5"} 0.041
neuroserve_inference_latency_ms{quantile="0.95"} 0.134
neuroserve_inference_latency_ms{quantile="0.99"} 1.200
```

---

## 8. Testing

Run the full GoogleTest suite:
```bash
./build/neuroserve_tests
```

Output:
```text
[==========] Running 18 tests from 7 test suites ran. (322 ms total)
[  PASSED  ] 18 tests.
```

Test suites include:
- `BoundedQueueTest`: Single/multi-producer/consumer, bounded capacity, timeout dequeue, shutdown wakeups.
- `WorkerPoolTest`: Clean startup/shutdown, concurrent execution, exception safety.
- `ModelManagerTest`: Model loading, invalid path handling, metadata inspection, instance lease acquisition.
- `DynamicBatcherTest`: Full batch immediate dispatch, partial batch timeout dispatch.
- `LoadSheddingTest`: Queue capacity enforcement, 429 rejection, automatic recovery after drainage.
- `InferenceServiceTest`: End-to-end multi-threaded inference, missing model validation.
- `ConfigTest`: YAML parsing and fail-fast validation.

---

## 9. Benchmarks

Run the Google Benchmark binary:
```bash
./build/neuroserve_bench
```

Empirical Benchmark Results:
```text
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
BM_Queue_SingleThread_Throughput       35.1 ns         34.3 ns     20315473
BM_Queue_Concurrent_Contention/1        205 ns          195 ns      3485166
BM_Queue_Concurrent_Contention/2       1078 ns          616 ns      1165803
BM_Queue_Concurrent_Contention/4      17126 ns        11009 ns       113178
BM_Queue_Concurrent_Contention/8      12279 ns         7584 ns        86423
BM_Inference_Batch_Scaling/1           2739 ns         2676 ns       235321 items_per_second=373.665k/s
BM_Inference_Batch_Scaling/2           2846 ns         2789 ns       244409 items_per_second=717.081k/s
BM_Inference_Batch_Scaling/4           3407 ns         3338 ns       219200 items_per_second=1.19824M/s
BM_Inference_Batch_Scaling/8           3864 ns         3786 ns       192799 items_per_second=2.11277M/s
BM_Inference_Batch_Scaling/16          5464 ns         5354 ns       134528 items_per_second=2.98841M/s
BM_Inference_Batch_Scaling/32          7784 ns         7627 ns        89543 items_per_second=4.19547M/s
BM_ModelInstance_Acquisition           39.1 ns         38.3 ns     22372886
```

### Load Testing Tool (`tools/load_tester.py`)
Run the load tester across concurrency levels 1 to 64:
```bash
python3 tools/load_tester.py --sweep --requests 200
```

Results Summary:
```text
========================= CONCURRENCY SWEEP SUMMARY =========================
Concurrency  | Throughput (req/s)   | p50 (ms)   | p95 (ms)   | p99 (ms)   | Shed (429)
--------------------------------------------------------------------------------
1            | 594.0                | 1.41       | 2.14       | 2.75       | 0         
2            | 784.0                | 1.96       | 4.15       | 4.96       | 0         
4            | 1201.2               | 2.70       | 4.55       | 5.74       | 0         
8            | 1010.1               | 6.42       | 12.32      | 14.98      | 0         
16           | 853.0                | 13.63      | 29.17      | 39.96      | 0         
32           | 1000.8               | 16.88      | 35.61      | 41.19      | 0         
64           | 915.9                | 11.84      | 21.74      | 25.57      | 0         
```

---

## 10. Design Trade-offs & Deep Systems Analysis

1. **Why Bounded Queueing & Load Shedding?**
   - Unbounded queues cause memory exhaustion (OOM) under sustained traffic spikes.
   - When request arrival rate exceeds processing rate (\(\lambda > \mu\)), queue latency increases without bound. Clients time out before requests are dequeued, wasting computation.
   - NeuroServe sheds requests immediately with **HTTP 429**, preserving throughput and tail latency for already-accepted requests.

2. **Why Batching Improves Throughput but Costs Latency**
   - Inference on small batch sizes is DRAM bandwidth-bound (loading weights once per single input vector). Stacking \(B\) inputs into a batched GEMM reuses weights in CPU cache, achieving 11.2x higher throughput.
   - Under low traffic, batching introduces queue wait time (`max_wait_ms`). NeuroServe balances this with dynamic early dispatch as soon as `max_batch_size` is reached.

3. **Why Worker Count != Model Instance Count**
   - Model execution on CPU is compute and memory intensive. Running more sessions than physical cores causes severe cache thrashing and context switching.
   - In NeuroServe, `workers.count = 4` can handle pre-processing, input validation, and tensor collation while `model.instances = 2` ensures CPU cores are never over-subscribed.

---

## 11. Known Limitations & Future Improvements

- **GPU Acceleration**: Add TensorRT and CUDA execution provider support via `Ort::SessionOptions::AppendExecutionProvider_TensorRT`.
- **Zero-Copy Tensor Buffers**: Support shared-memory IPC or io_uring for zero-copy tensor ingestion from local clients.
- **Dynamic Priority Queuing**: Allow latency-critical inference requests to bypass background batch queues using multi-level feedback queues.

