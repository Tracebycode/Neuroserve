#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace neuroserve {

class WorkerPool {
public:
    using Task = std::function<void()>;

    explicit WorkerPool(size_t worker_count);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    bool submit(Task task);
    void shutdown();

    size_t worker_count() const noexcept { return workers_.size(); }
    size_t active_workers() const noexcept { return active_workers_count_.load(std::memory_order_relaxed); }
    size_t completed_tasks() const noexcept { return completed_tasks_.load(std::memory_order_relaxed); }
    bool is_shutdown() const noexcept { return shutting_down_.load(std::memory_order_relaxed); }

private:
    void worker_loop(std::stop_token st);

    std::vector<std::jthread> workers_;
    std::queue<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutting_down_{false};
    std::atomic<size_t> active_workers_count_{0};
    std::atomic<size_t> completed_tasks_{0};
};

} // namespace neuroserve

