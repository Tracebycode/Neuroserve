#include "neuroserve/inference/inference_service.hpp"
#include "neuroserve/metrics/metrics_registry.hpp"
#include "neuroserve/health/health_checker.hpp"
#include "neuroserve/common/logger.hpp"

namespace neuroserve {

InferenceService::InferenceService(const Config& config, std::shared_ptr<ModelManager> model_manager)
    : config_(config),
      model_manager_(std::move(model_manager)) {
    worker_pool_ = std::make_unique<WorkerPool>(config_.workers.count);

    size_t batch_size = config_.batching.enabled ? config_.batching.max_batch_size : 1;
    std::chrono::milliseconds max_wait(config_.batching.enabled ? config_.batching.max_wait_ms : 0);

    batcher_ = std::make_unique<DynamicBatcher>(
        batch_size,
        max_wait,
        config_.queue.capacity,
        [this](std::vector<QueueItem> batch) {
            dispatch_batch(std::move(batch));
        }
    );
}

InferenceService::~InferenceService() {
    stop();
}

void InferenceService::start() {
    if (running_.exchange(true)) {
        return;
    }
    NS_LOG_INFO("Starting InferenceService...");
    batcher_->start();
    HealthChecker::instance().set_ready(true);
    NS_LOG_INFO("InferenceService ready to accept traffic");
}

void InferenceService::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    NS_LOG_INFO("Stopping InferenceService...");
    HealthChecker::instance().set_ready(false);
    if (batcher_) {
        batcher_->stop();
    }
    if (worker_pool_) {
        worker_pool_->shutdown();
    }
    NS_LOG_INFO("InferenceService stopped");
}

InferenceResponse InferenceService::infer(InferenceRequest req) {
    MetricsRegistry::instance().record_request();

    if (!running_.load(std::memory_order_relaxed)) {
        MetricsRegistry::instance().record_failure();
        InferenceResponse resp;
        resp.request_id = req.request_id;
        resp.success = false;
        resp.status_code = 503;
        resp.error_message = "InferenceService is not running";
        return resp;
    }

    if (!model_manager_->has_model(req.model_name)) {
        MetricsRegistry::instance().record_failure();
        InferenceResponse resp;
        resp.request_id = req.request_id;
        resp.success = false;
        resp.status_code = 404;
        resp.error_message = "Model '" + req.model_name + "' not found in registry";
        return resp;
    }

    // Input validation
    const auto& meta = model_manager_->get_metadata(req.model_name);
    for (const auto& in_tensor : req.inputs) {
        if (!meta.validate_input(in_tensor)) {
            MetricsRegistry::instance().record_failure();
            InferenceResponse resp;
            resp.request_id = req.request_id;
            resp.success = false;
            resp.status_code = 400;
            resp.error_message = "Invalid input tensor: " + in_tensor.name;
            return resp;
        }
    }

    bool shed = false;
    auto fut = batcher_->submit(std::move(req), shed);

    if (shed) {
        MetricsRegistry::instance().record_rejection();
        return fut.get();
    }

    auto resp = fut.get();
    if (resp.success) {
        MetricsRegistry::instance().record_success();
    } else {
        MetricsRegistry::instance().record_failure();
    }
    return resp;
}

void InferenceService::dispatch_batch(std::vector<QueueItem> batch) {
    if (batch.empty()) return;

    auto shared_items = std::make_shared<std::vector<QueueItem>>(std::move(batch));

    // Submit batch task to WorkerPool
    worker_pool_->submit([this, shared_items]() {
        auto& items = *shared_items;
        const size_t batch_size = items.size();
        const std::string& model_name = items[0].request.model_name;

        std::shared_ptr<ModelInstancePool> pool;
        try {
            pool = model_manager_->get_pool(model_name);
        } catch (const std::exception& ex) {
            for (auto& it : items) {
                InferenceResponse resp;
                resp.request_id = it.request.request_id;
                resp.success = false;
                resp.status_code = 500;
                resp.error_message = ex.what();
                it.promise.set_value(std::move(resp));
            }
            return;
        }

        // Acquire model instance from pool (RAII lease)
        auto lease = pool->acquire();
        auto start_infer = Clock::now();

        std::vector<InferenceRequest> requests;
        requests.reserve(batch_size);
        for (const auto& it : items) {
            requests.push_back(it.request);
        }

        std::vector<InferenceResponse> responses;
        if (batch_size == 1) {
            responses.push_back(lease->run(requests[0]));
        } else {
            responses = lease->run_batch(requests);
        }

        auto end_infer = Clock::now();
        auto infer_latency = std::chrono::duration_cast<Microseconds>(end_infer - start_infer);

        MetricsRegistry::instance().record_model_inference(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            auto& resp = responses[i];
            auto arrival = items[i].request.arrival_time;
            auto dequeue = items[i].request.dequeue_time;

            resp.queue_latency = std::chrono::duration_cast<Microseconds>(dequeue - arrival);
            resp.inference_latency = infer_latency;
            resp.total_latency = std::chrono::duration_cast<Microseconds>(end_infer - arrival);

            MetricsRegistry::instance().record_queue_latency_us(resp.queue_latency.count());
            MetricsRegistry::instance().record_inference_latency_us(resp.inference_latency.count());
            MetricsRegistry::instance().record_total_latency_us(resp.total_latency.count());

            items[i].promise.set_value(std::move(resp));
        }
    });
}

size_t InferenceService::queue_depth() const noexcept {
    return batcher_ ? batcher_->queue_depth() : 0;
}

size_t InferenceService::active_workers() const noexcept {
    return worker_pool_ ? worker_pool_->active_workers() : 0;
}

bool InferenceService::is_ready() const noexcept {
    return running_.load(std::memory_order_relaxed) && HealthChecker::instance().is_ready();
}

} // namespace neuroserve
