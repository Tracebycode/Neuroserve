# NeuroServe Architecture Specification

NeuroServe is a high-performance C++20 AI inference serving system designed for predictable latency, high concurrency, and bounded resource utilization.

---

## 1. System Topology & Pipeline

The pipeline decouples network ingestion from model execution:

```
                  Concurrent Clients
                          │ HTTP POST /v1/infer
                          ▼
            ┌───────────────────────────┐
            │       HttpServer          │
            └─────────────┬─────────────┘
                          │ Deserializes Request & Validates JSON
                          ▼
            ┌───────────────────────────┐
            │     InferenceService      │ (Core inference engine decoupled from HTTP)
            └─────────────┬─────────────┘
                          │ Validates against ModelMetadata
                          ▼
            ┌───────────────────────────┐
            │       LoadShedder         │ ◄── Rejects with HTTP 429 if queue full
            └─────────────┬─────────────┘
                          │ Enqueues with RAII promise
                          ▼
            ┌───────────────────────────┐
            │    BoundedRequestQueue    │ (Thread-safe, bounded capacity)
            └─────────────┬─────────────┘
                          │ Dequeue batch (N items or timeout T)
                          ▼
            ┌───────────────────────────┐
            │      DynamicBatcher       │ (Batches inputs along dimension 0)
            └─────────────┬─────────────┘
                          │ Dispatches batch task
                          ▼
            ┌───────────────────────────┐
            │        WorkerPool         │ (Fixed pool of std::jthread workers)
            └─────────────┬─────────────┘
                          │ RAII lease acquisition
                          ▼
            ┌───────────────────────────┐
            │     ModelInstancePool     │ (Reusable Ort::Session instances)
            └─────────────┬─────────────┘
                          │ ONNX Runtime inference
                          ▼
            ┌───────────────────────────┐
            │     Output Slicing        │ (Splits batched outputs to individual items)
            └─────────────┬─────────────┘
                          │ Sets promise values
                          ▼
            ┌───────────────────────────┐
            │     Metrics Registry      │ (Tracks queue latency, infer latency, throughput)
            └─────────────┬─────────────┘
                          │
                          ▼
                 Client Receives 200 OK
```

---

## 2. Request Lifecycle

1. **Ingestion & Validation**:
   - `HttpServer` receives a JSON payload specifying `model_name` and input tensors.
   - Assigns a unique monotonic 64-bit `request_id` and records `arrival_time` via `std::chrono::steady_clock`.
   - `InferenceService` checks whether the model is registered and verifies input dimensions against `ModelMetadata`.

2. **Admission Control & Load Shedding**:
   - The request is submitted to `DynamicBatcher::submit(req, shed)`.
   - If `BoundedQueue` is at full capacity, the request is shed immediately in \(\mathcal{O}(1)\) time, returning `HTTP 429 Too Many Requests` without allocating memory or blocking workers.

3. **Dynamic Batching**:
   - The dedicated batcher thread monitors the bounded queue.
   - It awaits up to `max_wait_ms` or until `max_batch_size` requests arrive.
   - If \(B\) requests arrive before timeout, they are drained immediately as a batch of size \(B\).
   - If timeout expires with \(k < B\) requests, the partial batch of size \(k\) is drained to avoid latency starvation.

4. **Worker Dispatch & Model Execution**:
   - The batch is handed to `WorkerPool::submit(...)`.
   - A worker thread acquires a `ModelInstance::ScopedLease` from the `ModelInstancePool`.
   - The worker concatenates the inputs along dimension 0 into a contiguous memory buffer.
   - The worker executes `Ort::Session::Run(...)` once for the entire batch.
   - The batched output tensor is sliced back into individual response payloads.

5. **Response & Metric Recording**:
   - The worker calculates:
     $$\text{queue\_latency} = t_{\text{dequeue}} - t_{\text{arrival}}$$
     $$\text{inference\_latency} = t_{\text{end\_infer}} - t_{\text{start\_infer}}$$
     $$\text{total\_latency} = t_{\text{end\_infer}} - t_{\text{arrival}}$$
   - Metrics are recorded into `MetricsRegistry` (updating running histograms for p50, p95, and p99).
   - The promise `prom.set_value(std::move(response))` is satisfied, resuming the HTTP handler to serialize and send the response.

---

## 3. Worker Lifecycle & Exception Safety

- Workers are managed as modern C++20 `std::jthread` instances that accept `std::stop_token`.
- Thread termination is non-detached: all worker threads join deterministically on `WorkerPool::shutdown()`.
- **Exception Boundary**: Every task executed by a worker is wrapped in `try { task(); } catch (...)`. An individual failing inference request (e.g., malformed tensor or kernel crash) records an error and fulfills its promise with HTTP 500, but **never crashes the worker thread**.

---

## 4. Reusable Model Instances & Memory Management

- **Zero-Session Recreation**: ONNX Runtime sessions are expensive to initialize (involving model file parsing, memory planning, graph optimization, and operator kernel allocation).
- NeuroServe pre-allocates a configured number of `ModelInstance` objects at startup into `ModelInstancePool`.
- Workers check out instances using an RAII lease (`ScopedLease`). Even if an unexpected exception occurs during tensor execution, the destructor of `ScopedLease` automatically returns the instance to the pool.
- CPU memory allocations utilize `OrtArenaAllocator`, avoiding repeated OS heap calls during tensor creation.

---

## 5. Graceful Shutdown Protocol

Upon receiving `SIGINT` or `SIGTERM`:
1. **Network Ingestion Closes**: `HttpServer::stop()` stops listening on the TCP port.
2. **Readiness Probe Flips**: `HealthChecker::set_ready(false)` signals load balancers that the server is not ready for new traffic.
3. **Queue Draining**: In-flight requests already queued are allowed to complete execution.
4. **Worker Pool Teardown**: `WorkerPool::shutdown()` wakes all sleeping workers and joins their threads cleanly.
5. **Model Resource Reclamation**: `ModelManager` frees ONNX Runtime sessions and environment contexts cleanly.

