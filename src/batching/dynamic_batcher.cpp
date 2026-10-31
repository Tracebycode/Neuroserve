#include "neuroserve/batching/dynamic_batcher.hpp"
#include "neuroserve/common/logger.hpp"

namespace neuroserve {

DynamicBatcher::DynamicBatcher(size_t max_batch_size,
                               std::chrono::milliseconds max_wait,
                               size_t queue_capacity,
                               BatchDispatchHandler handler)
    : max_batch_size_(max_batch_size > 0 ? max_batch_size : 1),
      max_wait_(max_wait),
      queue_(queue_capacity),
      handler_(std::move(handler)) {}

DynamicBatcher::~DynamicBatcher() {
    stop();
}

void DynamicBatcher::start() {
    if (running_.exchange(true)) {
        return;
    }
    NS_LOG_INFO("Starting DynamicBatcher (max_batch_size={}, max_wait_ms={})",
                max_batch_size_, max_wait_.count());
    batcher_thread_ = std::jthread([this](std::stop_token st) {
        batcher_loop(st);
    });
}

void DynamicBatcher::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    NS_LOG_INFO("Stopping DynamicBatcher...");
    queue_.shutdown();
    if (batcher_thread_.joinable()) {
        batcher_thread_.request_stop();
        batcher_thread_.join();
    }
    NS_LOG_INFO("DynamicBatcher stopped");
}

std::future<InferenceResponse> DynamicBatcher::submit(InferenceRequest req, bool& shed) {
    std::promise<InferenceResponse> prom;
    auto fut = prom.get_future();
    const uint64_t req_id = req.request_id;

    if (!running_.load(std::memory_order_relaxed)) {
        shed = true;
        InferenceResponse resp;
        resp.request_id = req_id;
        resp.success = false;
        resp.status_code = 503;
        resp.error_message = "Server is shutting down";
        prom.set_value(resp);
        return fut;
    }

    QueueItem item{std::move(req), std::move(prom)};
    auto res = queue_.try_enqueue(std::move(item));

    if (res == BoundedQueue<QueueItem>::EnqueueResult::Full) {
        shed = true;
        InferenceResponse resp;
        resp.request_id = req_id;
        resp.success = false;
        resp.status_code = 429;
        resp.error_message = "Server Overloaded / Request Queue Saturated (Load Shedding)";
        // Return a future with the 429 response already set
        std::promise<InferenceResponse> shed_prom;
        auto shed_fut = shed_prom.get_future();
        shed_prom.set_value(resp);
        return shed_fut;
    } else if (res == BoundedQueue<QueueItem>::EnqueueResult::Shutdown) {
        shed = true;
        InferenceResponse resp;
        resp.request_id = req_id;
        resp.success = false;
        resp.status_code = 503;
        resp.error_message = "Server queue is shutdown";
        std::promise<InferenceResponse> shed_prom;
        auto shed_fut = shed_prom.get_future();
        shed_prom.set_value(resp);
        return shed_fut;
    }

    shed = false;
    return fut;
}

void DynamicBatcher::batcher_loop(std::stop_token st) {
    while (!st.stop_requested() && !queue_.is_shutdown()) {
        std::vector<QueueItem> batch;
        batch.reserve(max_batch_size_);

        size_t count = queue_.dequeue_batch(batch, max_batch_size_, max_wait_);
        if (count > 0) {
            auto now = Clock::now();
            for (auto& item : batch) {
                item.request.dequeue_time = now;
            }
            handler_(std::move(batch));
        }
    }
}

} // namespace neuroserve
