#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <optional>
#include <chrono>
#include <cstddef>

namespace neuroserve {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity)
        : capacity_(capacity), shutdown_(false) {}

    ~BoundedQueue() {
        shutdown();
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;

    enum class EnqueueResult {
        Success,
        Full,
        Shutdown
    };

    // Non-blocking enqueue for load shedding
    EnqueueResult try_enqueue(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) {
            return EnqueueResult::Shutdown;
        }
        if (queue_.size() >= capacity_) {
            return EnqueueResult::Full;
        }
        queue_.push(std::move(item));
        lock.unlock();
        not_empty_cv_.notify_one();
        return EnqueueResult::Success;
    }

    // Timed enqueue
    template <typename Rep, typename Period>
    EnqueueResult enqueue_for(T item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_full_cv_.wait_for(lock, timeout, [this] {
            return shutdown_ || queue_.size() < capacity_;
        })) {
            return EnqueueResult::Full;
        }
        if (shutdown_) {
            return EnqueueResult::Shutdown;
        }
        queue_.push(std::move(item));
        lock.unlock();
        not_empty_cv_.notify_one();
        return EnqueueResult::Success;
    }

    // Blocking dequeue with timeout
    template <typename Rep, typename Period>
    std::optional<T> dequeue_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_cv_.wait_for(lock, timeout, [this] {
            return shutdown_ || !queue_.empty();
        })) {
            return std::nullopt;
        }
        if (queue_.empty() && shutdown_) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_cv_.notify_one();
        return item;
    }

    // Batch dequeue: waits up to max_wait for at least 1 item, then drains up to max_items
    template <typename Rep, typename Period>
    size_t dequeue_batch(std::vector<T>& batch_out, size_t max_items,
                         const std::chrono::duration<Rep, Period>& max_wait) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto deadline = std::chrono::steady_clock::now() + max_wait;

        if (!not_empty_cv_.wait_until(lock, deadline, [this] {
            return shutdown_ || !queue_.empty();
        })) {
            return 0;
        }

        if (queue_.empty() && shutdown_) {
            return 0;
        }

        while (!queue_.empty() && batch_out.size() < max_items) {
            batch_out.push_back(std::move(queue_.front()));
            queue_.pop();
        }

        size_t count = batch_out.size();
        lock.unlock();
        not_full_cv_.notify_all();
        return count;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const noexcept {
        return capacity_;
    }

    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    const size_t capacity_;
    bool shutdown_;
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
};

} // namespace neuroserve

