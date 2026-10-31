#include "neuroserve/worker/worker_pool.hpp"
#include "neuroserve/common/logger.hpp"

namespace neuroserve {

WorkerPool::WorkerPool(size_t worker_count) {
    if (worker_count == 0) {
        worker_count = 1;
    }
    workers_.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this](std::stop_token st) {
            worker_loop(st);
        });
    }
    NS_LOG_INFO("WorkerPool initialized with {} workers", worker_count);
}

WorkerPool::~WorkerPool() {
    shutdown();
}

bool WorkerPool::submit(Task task) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutting_down_.load(std::memory_order_relaxed)) {
            return false;
        }
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
    return true;
}

void WorkerPool::shutdown() {
    bool expected = false;
    if (!shutting_down_.compare_exchange_strong(expected, true)) {
        return;
    }

    NS_LOG_INFO("Shutting down WorkerPool...");
    cv_.notify_all();

    for (auto& worker : workers_) {
        worker.request_stop();
    }
    // std::jthread automatically joins on destruction/clear
    workers_.clear();
    NS_LOG_INFO("WorkerPool shutdown complete");
}

void WorkerPool::worker_loop(std::stop_token st) {
    while (!st.stop_requested()) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this, &st] {
                return shutting_down_.load(std::memory_order_relaxed) ||
                       !tasks_.empty() ||
                       st.stop_requested();
            });

            if (st.stop_requested()) {
                return;
            }

            if (tasks_.empty()) {
                if (shutting_down_.load(std::memory_order_relaxed)) {
                    return;
                }
                continue;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        active_workers_count_.fetch_add(1, std::memory_order_relaxed);
        try {
            task();
        } catch (const std::exception& ex) {
            NS_LOG_ERROR("Worker task threw exception: {}", ex.what());
        } catch (...) {
            NS_LOG_ERROR("Worker task threw unknown exception");
        }
        active_workers_count_.fetch_sub(1, std::memory_order_relaxed);
        completed_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace neuroserve

