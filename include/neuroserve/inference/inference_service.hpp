#pragma once

#include "neuroserve/model/model_manager.hpp"
#include "neuroserve/worker/worker_pool.hpp"
#include "neuroserve/batching/dynamic_batcher.hpp"
#include "neuroserve/config/config.hpp"
#include <memory>
#include <string>

namespace neuroserve {

class InferenceService {
public:
    InferenceService(const Config& config, std::shared_ptr<ModelManager> model_manager);
    ~InferenceService();

    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    void start();
    void stop();

    InferenceResponse infer(InferenceRequest req);

    size_t queue_depth() const noexcept;
    size_t active_workers() const noexcept;
    bool is_ready() const noexcept;

    std::shared_ptr<ModelManager> model_manager() const noexcept { return model_manager_; }

private:
    void dispatch_batch(std::vector<QueueItem> batch);

    Config config_;
    std::shared_ptr<ModelManager> model_manager_;
    std::unique_ptr<WorkerPool> worker_pool_;
    std::unique_ptr<DynamicBatcher> batcher_;
    std::atomic<bool> running_{false};
};

} // namespace neuroserve

