# NeuroServe Concurrency & Synchronization Design

This document details the concurrency model, synchronization primitives, memory visibility, and lock contention avoidance strategies implemented in NeuroServe.

---

## 1. Concurrency Primitives & Memory Model

NeuroServe is written entirely in modern C++20 using standard concurrency primitives:
- `std::mutex` and `std::unique_lock` for critical queue and pool operations
- `std::condition_variable` for non-busy waiting and thread signaling
- `std::jthread` and `std::stop_token` for deterministic cooperative thread cancellation
- `std::atomic<T>` with explicit memory orderings (`relaxed`, `acquire`, `release`, and `seq_cst`)

---

## 2. Bounded Queue Synchronization & Contention Strategy

### Synchronization Strategy
The `BoundedQueue<T>` coordinates multiple producers (HTTP request worker threads) and multiple consumers / batcher threads.

- **Lock Scope Minimization**: Locks are held strictly for the duration of the pointer/move manipulation:
  ```cpp
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.size() >= capacity_) return EnqueueResult::Full;
  queue_.push(std::move(item));
  lock.unlock(); // Unlocked BEFORE notify to prevent priority inversion / thundering herd
  not_empty_cv_.notify_one();
  ```
- **Signaling Optimization**: Releasing the lock *before* calling `notify_one()` prevents a woken consumer thread from immediately blocking on the mutex held by the notifying producer.
- **Batch Drain Contention Amortization**: Rather than having workers dequeue items one by one under lock contention, `DynamicBatcher` calls `dequeue_batch(...)`, acquiring the mutex **once** to drain up to `max_batch_size` items into a local vector, amortizing synchronization overhead by a factor equal to the batch size.

---

## 3. Producer / Consumer Interactions & Shutdown Visibility

- **Spurious Wakeup Protection**: All condition variable wait calls are predicate-guarded:
  ```cpp
  not_empty_cv_.wait_for(lock, timeout, [this] {
      return shutdown_ || !queue_.empty();
  });
  ```
- **Memory Visibility**: Mutex acquisition and release establish an *acquire-release* synchronization sequence between the producer enqueuing an item and the consumer dequeuing it. All writes to the request payload made by the producer thread are visible to the worker thread without data races.
- **Shutdown Signaling**: When `shutdown()` is called:
  - `shutdown_` is flipped under the mutex.
  - `notify_all()` is issued on both `not_empty_cv_` and `not_full_cv_`.
  - Any thread blocked in `enqueue_for` or `dequeue_for` awakens, detects `shutdown_ == true`, and exits cleanly.

---

## 4. Model Instance Pooling & Worker Decoupling

### Why Worker Count != Model Instance Count
A common architectural flaw is 1:1 coupling between worker threads and model instances.

In NeuroServe:
- Workers perform tasks beyond ONNX graph execution: request deserialization, input validation, tensor concatenation, slicing, and latency metric calculations.
- ONNX Runtime sessions consume significant working memory (workspace buffers, intermediate activation caches).
- If `worker_count = 16` but hardware has 4 physical CPU cores, running 16 concurrent ONNX sessions causes massive cache eviction, L3 cache pollution, and CPU context-switching overhead.
- By separating `workers.count = 4` and `model.instances = 2`, workers wait on a finite pool of sessions (`ModelInstancePool::ScopedLease`). When all sessions are busy, workers wait on the pool condition variable rather than thrashing the hardware.

---

## 5. False Sharing Mitigations

In high-frequency multi-threaded environments, distinct atomic counters located on the same CPU cache line (64 bytes) experience cache invalidation traffic even when threads access different variables (*false sharing*).

In NeuroServe:
- Request counters, active worker counters, and metric histograms are structured to avoid co-locating heavily mutated atomic counters on the same cache line.
- Metrics accumulation for latency percentiles batches samples before sorting, preventing per-request mutex contention on the metrics registry.

