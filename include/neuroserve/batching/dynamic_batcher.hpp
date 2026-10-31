#pragma once

#include "neuroserve/queue/bounded_queue.hpp"
#include "neuroserve/common/types.hpp"
#include <future>
#include <functional>
#include <thread>
#include <vector>

namespace neuroserve {

struct QueueItem {
    InferenceRequest request;
    std::promise<InferenceResponse> promise;
};

class DynamicBatcher {
public:
    using BatchDispatchHandler = std::function<void(std::vector<QueueItem>)>;

    DynamicBatcher(size_t max_batch_size,
                   std::chrono::milliseconds max_wait,
                   size_t queue_capacity,
                   BatchDispatchHandler handler);

    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    void start();
    void stop();

    // Returns a future to the response. If shed is set to true, request was dropped due to overload.
    std::future<InferenceResponse> submit(InferenceRequest req, bool& shed);

    size_t queue_depth() const noexcept { return queue_.size(); }
    size_t queue_capacity() const noexcept { return queue_.capacity(); }
    bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }

private:
    void batcher_loop(std::stop_token st);

    const size_t max_batch_size_;
    const std::chrono::milliseconds max_wait_;
    BoundedQueue<QueueItem> queue_;
    BatchDispatchHandler handler_;
    std::atomic<bool> running_{false};
    std::jthread batcher_thread_;
};

} // namespace neuroserve

