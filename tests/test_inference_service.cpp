#include <gtest/gtest.h>
#include "neuroserve/inference/inference_service.hpp"
#include "neuroserve/metrics/metrics_registry.hpp"

using namespace neuroserve;

TEST(InferenceServiceTest, EndToEndBatchInference) {
    Config cfg;
    cfg.workers.count = 2;
    cfg.batching.enabled = true;
    cfg.batching.max_batch_size = 4;
    cfg.batching.max_wait_ms = 10;
    cfg.queue.capacity = 128;

    auto model_manager = std::make_shared<ModelManager>();
    model_manager->load_model("test_mlp", "models/model.onnx", 2);

    InferenceService service(cfg, model_manager);
    service.start();
    EXPECT_TRUE(service.is_ready());

    std::vector<InferenceResponse> responses;
    std::vector<std::thread> threads;
    std::mutex resp_mutex;

    for (size_t i = 0; i < 4; ++i) {
        threads.emplace_back([&service, &responses, &resp_mutex, i] {
            InferenceRequest req;
            req.request_id = i;
            req.model_name = "test_mlp";
            TensorData in;
            in.name = "input";
            in.shape = {1, 8};
            in.float_data = {1.0f, 0.5f, -0.5f, 0.2f, 0.1f, -0.2f, 0.3f, 0.4f};
            req.inputs.push_back(in);

            auto resp = service.infer(std::move(req));
            std::lock_guard<std::mutex> lock(resp_mutex);
            responses.push_back(std::move(resp));
        });
    }

    for (auto& t : threads) t.join();

    EXPECT_EQ(responses.size(), 4);
    for (const auto& r : responses) {
        EXPECT_TRUE(r.success);
        EXPECT_EQ(r.status_code, 200);
        EXPECT_EQ(r.outputs.size(), 1);
        EXPECT_EQ(r.outputs[0].float_data.size(), 2);
    }

    service.stop();
}

TEST(InferenceServiceTest, UnknownModelRejected) {
    Config cfg;
    cfg.workers.count = 1;
    cfg.queue.capacity = 16;
    auto model_manager = std::make_shared<ModelManager>();

    InferenceService service(cfg, model_manager);
    service.start();

    InferenceRequest req;
    req.request_id = 1;
    req.model_name = "nonexistent_model";

    auto resp = service.infer(std::move(req));
    EXPECT_FALSE(resp.success);
    EXPECT_EQ(resp.status_code, 404);

    service.stop();
}

