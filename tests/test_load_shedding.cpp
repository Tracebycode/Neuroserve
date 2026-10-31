#include <gtest/gtest.h>
#include "neuroserve/batching/dynamic_batcher.hpp"
#include <atomic>

using namespace neuroserve;

TEST(LoadSheddingTest, DropsRequestsWhenQueueSaturated) {
    constexpr size_t queue_cap = 4;
    std::atomic<bool> hold_worker{true};

    // Handler that holds items to cause queue backlog
    DynamicBatcher batcher(
        4,
        std::chrono::milliseconds(100),
        queue_cap,
        [&](std::vector<QueueItem> items) {
            while (hold_worker.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            for (auto& it : items) {
                InferenceResponse resp;
                resp.success = true;
                it.promise.set_value(resp);
            }
        }
    );
    batcher.start();

    // Fill queue to capacity
    std::vector<std::future<InferenceResponse>> futures;
    size_t shed_count = 0;

    for (size_t i = 0; i < 20; ++i) {
        InferenceRequest req;
        req.request_id = i;
        bool shed = false;
        auto fut = batcher.submit(std::move(req), shed);
        if (shed) {
            shed_count++;
            auto resp = fut.get();
            EXPECT_EQ(resp.status_code, 429);
            EXPECT_FALSE(resp.success);
        } else {
            futures.push_back(std::move(fut));
        }
    }

    // Must have shed requests when capacity was exceeded
    EXPECT_GT(shed_count, 0);

    // Release worker and let accepted requests finish
    hold_worker.store(false);
    for (auto& fut : futures) {
        auto resp = fut.get();
        EXPECT_TRUE(resp.success);
    }

    // After draining, server recovers and accepts new requests without shedding
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    bool recovered_shed = false;
    InferenceRequest new_req;
    new_req.request_id = 999;
    auto recovered_fut = batcher.submit(std::move(new_req), recovered_shed);
    EXPECT_FALSE(recovered_shed);

    batcher.stop();
}

